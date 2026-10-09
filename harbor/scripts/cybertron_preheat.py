#!/usr/bin/env python3
"""One-time cache preheat for the Cybertron (devspace) Harbor backend.

The Cybertron backend shares one read-only model/dataset cache across every
trial devspace at ``/user/<cluster_user>/ptdbench-cache/{models,datasets}``
(see ``cybertron-harbor-impl-plan.md`` §6.1). That cache is populated *once*
by a BATCH job that clones the ptdbench repo, points ``PTDBENCH_MODELS`` /
``PTDBENCH_DATASETS`` at the cache, and runs every task's
``environment/prepare.sh``. Trials then only read it — no PVC / RBAC needed.

This script builds (and optionally submits) that ``cctl job create`` command.
It defaults to ``--dry-run``: it prints the command for review rather than
launching a large, long-running, quota-consuming job. Note the cu130 image set
(``slime`` v0.3.2) will not run on wuxi_train/speech-wx yet — see
``cybertron-cuda13-driver-mismatch``; preheat that set on a newer-driver pool.

Example:
    python scripts/cybertron_preheat.py \\
        --project xinfer-chenyinghao --cluster wuxi_train \\
        --resource-pool speech-wx --billing-account-id N00069 \\
        --image 25607122 --gpu 1 --gpu-model h100 \\
        --git-path git@github.com:your-org/ptdbench.git --git-ref main \\
        --cluster-user chenyinghao
    # review the printed command, then re-run with --run to submit.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys


def proxy_export(http_proxy: str | None) -> str:
    """Shell `export http(s)_proxy=… no_proxy=…;` for the whitelist egress proxy.

    Cybertron nodes only reach the HF mirror through this proxy, and a BATCH
    ``--entry`` / non-interactive ssh does not inherit it from Teleport. Exported
    first so even DNS/connect for the mirror goes through it. Empty when no proxy.
    Shared with scripts/cybertron_stage.py so both paths set it identically.
    """
    if not http_proxy:
        return ""
    q = shlex.quote(http_proxy)
    no_proxy = "localhost,127.0.0.1,.modelbest.co,.svc.cluster.local"
    return (
        f"export http_proxy={q} https_proxy={q} HTTP_PROXY={q} HTTPS_PROXY={q} "
        f"no_proxy={no_proxy} NO_PROXY={no_proxy}; "
    )


def hf_export(hf_endpoint: str | None) -> str:
    """Shell `export HF_ENDPOINT=… HF_HUB_DISABLE_XET=1;` for the HF mirror.

    huggingface_hub reads HF_ENDPOINT at runtime; xet is disabled because the
    mirror does not serve it (same as the tasks' workspace/.env). Empty when no
    endpoint. Shared with scripts/cybertron_stage.py.
    """
    if not hf_endpoint:
        return ""
    return f"export HF_ENDPOINT={shlex.quote(hf_endpoint)} HF_HUB_DISABLE_XET=1; "


def build_entry(
    *,
    cache_dir: str,
    prepare_glob: str,
    repo_subdir: str | None = None,
    hf_endpoint: str | None = None,
    http_proxy: str | None = None,
) -> str:
    """Build the BATCH ``--entry`` shell command that preheats the cache.

    It exports the ptdbench cache env vars, creates the cache subtree on the
    persistent ``/user`` fs (never container rootfs — ephemeral is tiny), then
    runs every ``prepare.sh`` matched by *prepare_glob* with those env vars set.

    Cybertron nodes have no direct egress to ``huggingface.co``; *hf_endpoint*
    (a mirror such as ``https://hf-mirror.com``) is exported as ``HF_ENDPOINT``
    so ``ptdbench_fetch_hf`` / ``huggingface_hub`` route through it, matching the
    task ``workspace/.env`` files. Pass an empty string to skip it.

    The mirror itself is only reachable through the cluster's whitelist egress
    proxy (*http_proxy*, e.g. ``http://whitelist-proxy.cybertron.svc.cluster.local:7891``).
    Teleport injects that proxy into interactive ssh sessions but a BATCH
    ``--entry`` never gets it, so without this export the fetch times out. It is
    exported as http(s)_proxy (both cases); ``no_proxy`` keeps loopback and the
    internal ``.modelbest.co`` / ``.svc.cluster.local`` traffic off the proxy.
    Pass an empty string to skip it.
    """
    models = f"{cache_dir}/models"
    datasets = f"{cache_dir}/datasets"
    # Egress proxy + HF mirror for the on-node fetches (see proxy_export/hf_export).
    # proxy first so even DNS/connect for the mirror goes through it.
    proxy_exports = proxy_export(http_proxy)
    hf_exports = hf_export(hf_endpoint)
    # Throwaway scratch base on the persistent /user fs (container ephemeral is
    # tiny). prepare.sh writes weights into $PTDBENCH_WORKSPACE and KEY=VALUE
    # lines into $PTDBENCH_PREPARED/env; neither is what preheat is here for (the
    # durable cache is $PTDBENCH_MODELS / $PTDBENCH_DATASETS), but both dirs must
    # exist and be set or a script under `set -u` aborts on the unbound var.
    # See docker/warm-cache.sh, which sets the same env for the docker path.
    scratch = f"{cache_dir}/.warm-scratch"
    root = repo_subdir or "."
    return (
        "set -euo pipefail; "
        f"export PTDBENCH_STAGE=prepare WANDB_MODE=offline WANDB_SILENT=true; "
        f"{proxy_exports}"
        f"{hf_exports}"
        f"export PTDBENCH_MODELS={shlex.quote(models)} "
        f"PTDBENCH_DATASETS={shlex.quote(datasets)}; "
        f"mkdir -p {shlex.quote(models)} {shlex.quote(datasets)} "
        f"{shlex.quote(scratch)}; "
        f"cd {shlex.quote(root)}; "
        # prepare.sh invokes ptdbench_fetch_hf / ptdbench_dataset, shims that exec
        # /usr/local/libexec/*.py. The Dockerfiles no longer bake those (only the
        # activate hook) -- docker/install-ptdbench-tooling.sh installs the clone's
        # CURRENT fetch/dataset tooling into /usr/local at prepare time, so a fix
        # (e.g. materialize()'s symlink-dereference) reaches a running job without
        # an image rebuild. Run it after the cd into the repo, before the loop.
        # Guarded: a layout without the installer (older checkout) is a no-op.
        f"if [ -f docker/install-ptdbench-tooling.sh ]; then "
        f"bash docker/install-ptdbench-tooling.sh && "
        f'echo "[preheat] installed latest ptdbench tooling from clone"; fi; '
        f"found=0; failed=''; "
        f"for p in {prepare_glob}; do "
        f'  [ -f "$p" ] || continue; '
        f"  found=$((found+1)); "
        f'  echo ">>> preparing $p"; '
        f'  ws="$(mktemp -d {shlex.quote(scratch)}/XXXXXX)"; '
        f'  export PTDBENCH_WORKSPACE="$ws/workspace" PTDBENCH_PREPARED="$ws/prepared"; '
        f'  mkdir -p "$PTDBENCH_WORKSPACE" "$PTDBENCH_PREPARED"; '
        # Seed the scratch workspace with the task's own workspace/ tree (sibling
        # of environment/). prepare.sh for slime/rlve tasks does its torch_dist
        # conversion against $PTDBENCH_WORKSPACE/{scripts/models,tools} shipped by
        # the task -- harbor materializes that tree before prepare runs, so we do
        # too, or the GPU conversion step aborts on the missing files.
        f'  tw="$(dirname "$(dirname "$p")")/workspace"; '
        f'  [ -d "$tw" ] && cp -a "$tw/." "$PTDBENCH_WORKSPACE/"; '
        # Per-task non-fatal: the durable deliverable is the download cache, which
        # ptdbench_fetch_hf writes before any post-download step. A conversion /
        # unpack failure (it only touches throwaway $ws) must not abort warming
        # the remaining tasks; tally and report instead.
        f'  if (cd "$(dirname "$p")" && bash "$(basename "$p")"); then '
        f'    echo "<<< done $p"; '
        f"  else "
        f'    echo "!!! prepare failed (rc=$?): $p (downloads still cached)" >&2; '
        f'    failed="$failed $p"; '
        f"  fi; "
        f'  rm -rf "$ws"; '
        f"done; "
        f'echo "preheated $found prepare.sh script(s) into {cache_dir}"; '
        # Honest exit status: the loop above runs every task (one failure never
        # aborts the batch -- the download cache is already warmed by then), but
        # if any prepare.sh failed we exit non-zero so the job is marked Failed
        # (and its logs archived) instead of a misleading Succeeded. The stderr
        # line names which scripts to look at.
        f'[ -z "$failed" ] || {{ '
        f'echo "prepare.sh failed (downloads cached):$failed" >&2; exit 1; }}'
    )


def build_cctl_command(args: argparse.Namespace, entry: str) -> list[str]:
    cmd = ["cctl"]
    if args.profile:
        cmd += ["--profile", args.profile]
    cmd += [
        "job",
        "create",
        "--project",
        args.project,
        "--cluster",
        args.cluster,
        "--resource-pool",
        args.resource_pool,
        "--billing-account-id",
        args.billing_account_id,
        "--image",
        args.image,
        "--gpu",
        str(args.gpu),
        "--gpu-model",
        args.gpu_model,
        "--cpu",
        str(args.cpu),
        "--memory",
        str(args.memory),
        "--code-type",
        "git",
        "--git-path",
        args.git_path,
    ]
    if args.git_ref:
        cmd += ["--git-ref", args.git_ref]
    cmd += ["--entry", entry]
    return cmd


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--cluster", default="wuxi_train")
    parser.add_argument("--resource-pool", default="speech-wx")
    parser.add_argument("--billing-account-id", default="N00069")
    parser.add_argument(
        "--image", required=True, help="Cybertron platform image id / canonicalName"
    )
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--gpu-model", default="h100")
    parser.add_argument("--cpu", type=int, default=16)
    parser.add_argument("--memory", type=int, default=128, help="memory in GiB")
    parser.add_argument(
        "--git-path", required=True, help="ptdbench repo SSH URL (--code-type git)"
    )
    parser.add_argument("--git-ref", default=None)
    parser.add_argument("--cluster-user", default="chenyinghao")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="cache root (default /user/<cluster-user>/ptdbench-cache)",
    )
    parser.add_argument(
        "--prepare-glob",
        default="tasks/*/environment/prepare.sh environment/prepare.sh",
        help="shell glob(s) for prepare.sh scripts, relative to the repo",
    )
    parser.add_argument(
        "--repo-subdir",
        default=None,
        help="cd into this subdir of the cloned repo before globbing",
    )
    parser.add_argument(
        "--hf-endpoint",
        default="https://hf-mirror.com",
        help="HF mirror exported as HF_ENDPOINT (nodes can't reach "
        "huggingface.co); pass empty string to disable",
    )
    parser.add_argument(
        "--http-proxy",
        default="http://whitelist-proxy.cybertron.svc.cluster.local:7891",
        help="cluster egress proxy exported as http(s)_proxy so the HF mirror "
        "is reachable from the BATCH job (Teleport only injects it into "
        "interactive ssh); pass empty string to disable",
    )
    parser.add_argument("--profile", default=None, help="cctl login profile")
    parser.add_argument(
        "--run",
        action="store_true",
        help="actually submit the job (default: dry-run, just print)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cache_dir = args.cache_dir or f"/user/{args.cluster_user}/ptdbench-cache"
    entry = build_entry(
        cache_dir=cache_dir,
        prepare_glob=args.prepare_glob,
        repo_subdir=args.repo_subdir,
        hf_endpoint=args.hf_endpoint,
        http_proxy=args.http_proxy,
    )
    cmd = build_cctl_command(args, entry)
    printable = " ".join(shlex.quote(part) for part in cmd)
    if not args.run:
        print("# dry-run; re-run with --run to submit:\n")
        print(printable)
        return 0
    print(f"submitting preheat job:\n{printable}\n", file=sys.stderr)
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
