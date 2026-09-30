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


def build_entry(
    *,
    cache_dir: str,
    prepare_glob: str,
    repo_subdir: str | None = None,
) -> str:
    """Build the BATCH ``--entry`` shell command that preheats the cache.

    It exports the ptdbench cache env vars, creates the cache subtree on the
    persistent ``/user`` fs (never container rootfs — ephemeral is tiny), then
    runs every ``prepare.sh`` matched by *prepare_glob* with those env vars set.
    """
    models = f"{cache_dir}/models"
    datasets = f"{cache_dir}/datasets"
    root = repo_subdir or "."
    return (
        "set -euo pipefail; "
        f"export PTDBENCH_MODELS={shlex.quote(models)} "
        f"PTDBENCH_DATASETS={shlex.quote(datasets)}; "
        f"mkdir -p {shlex.quote(models)} {shlex.quote(datasets)}; "
        f"cd {shlex.quote(root)}; "
        f"found=0; "
        f"for p in {prepare_glob}; do "
        f'  [ -f "$p" ] || continue; '
        f"  found=$((found+1)); "
        f'  echo ">>> preparing $p"; '
        f'  (cd "$(dirname "$p")" && bash "$(basename "$p")"); '
        f"done; "
        f'echo "preheated $found prepare.sh script(s) into {cache_dir}"'
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
