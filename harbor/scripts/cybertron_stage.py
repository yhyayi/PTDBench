#!/usr/bin/env python3
"""Stage a PTDBench task to the pre-agent state on a Cybertron devspace.

This is the Cybertron counterpart of ``ptdbench harbor-shell`` (which is
docker-only): it reaches exactly the state a Harbor trial is in *just before the
agent starts* -- prepared workspace, downloaded weights, tooling installed --
and then stops, leaving the devspace running for manual experiments (launch the
training script by hand, watch logs, reproduce a bug).

It reuses the real backend ``CybertronEnvironment`` for the whole devspace
lifecycle (create *or* attach, resolve host, ssh transport, tar-over-ssh upload,
Landlock ``llrun`` + rename-shim setup) and the ``cybertron_preheat`` proxy / HF
mirror recipe, so the environment matches a real eval run. Unlike the eval flow
(which runs ``prepare.sh`` in LOCAL docker and uploads the result), this runs
``prepare.sh`` ON the devspace against the shared cluster cache + GPUs -- the
manual §2b runbook flow, automated.

It defaults to ``--dry-run``: prints the plan for review rather than touching the
cluster. Pass ``--run`` to actually create/attach + stage.

``ptdbench harbor-stage --backend cybertron`` shells out to this script (the same
ptdbench→harbor boundary ``harbor-evaluate`` uses for ``harbor run``).

Example:
    python scripts/cybertron_stage.py tasks/reward/retool_length_hack \\
        --gpu 8 --image slime-v0.3.2-sglang0.5.15-torch2.11.0-flashattn2.8.3-cu129-py3.12 \\
        --environment-kwarg project=xinfer-chenyinghao \\
        --environment-kwarg cluster=wuxi_train \\
        --environment-kwarg resource_pool=speech-wx \\
        --environment-kwarg billing_account_id=N00069 \\
        --run
    # or attach to an existing devspace: --environment-kwarg devspace_id=tasks/123
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import shlex
import sys
from pathlib import Path
from typing import Any

# Import siblings whether invoked as `python scripts/cybertron_stage.py` (script
# dir on sys.path) or `python -m scripts.cybertron_stage`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# The recipe and the remote layout are shared with CybertronEnvironment, which
# runs the same prepare step as part of start() for the eval path; they live in
# the package so the backend does not have to import from scripts/.
from harbor.environments.ptdbench_prepare import (
    DEFAULT_HF_ENDPOINT as _DEFAULT_HF_ENDPOINT,
)
from harbor.environments.ptdbench_prepare import (
    DEFAULT_HTTP_PROXY as _DEFAULT_HTTP_PROXY,
)
from harbor.environments.ptdbench_prepare import (
    REMOTE_DATASETS as _REMOTE_DATASETS,
)
from harbor.environments.ptdbench_prepare import (
    REMOTE_ENVIRONMENT as _REMOTE_ENVIRONMENT,
)
from harbor.environments.ptdbench_prepare import (
    REMOTE_MODELS as _REMOTE_MODELS,
)
from harbor.environments.ptdbench_prepare import (
    REMOTE_PREPARED as _REMOTE_PREPARED,
)
from harbor.environments.ptdbench_prepare import (
    REMOTE_TOOLING as _REMOTE_TOOLING,
)
from harbor.environments.ptdbench_prepare import (
    REMOTE_WORKSPACE as _REMOTE_WORKSPACE,
)
from harbor.environments.ptdbench_prepare import (
    build_prepare_recipe,
)

# Constructor kwargs that are ints (everything else from --environment-kwarg is a
# str). Coerced so `cpu=16` from the CLI reaches CybertronEnvironment as an int.
_INT_KWARGS = frozenset(
    {
        "cpu",
        "memory_gb",
        "status_timeout_sec",
        "ssh_connect_timeout_sec",
        "heartbeat_interval_sec",
    }
)

# Constructor kwargs that are bools. Without this, `requeue=false` from the CLI
# would reach the constructor as the non-empty (hence truthy) string "false".
_BOOL_KWARGS = frozenset({"requeue"})

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off", ""})


def _coerce_bool(key: str, value: str) -> bool:
    low = value.strip().lower()
    if low in _TRUE:
        return True
    if low in _FALSE:
        return False
    raise ValueError(f"--environment-kwarg {key} expects a boolean, got {value!r}")


def _log(msg: str) -> None:
    """Progress line to stderr (the devspace create/start phase is slow)."""
    print(f"[stage] {msg}", file=sys.stderr, flush=True)


async def _stream_exec(env, recipe: str) -> int:
    """Run ``recipe`` on the devspace, streaming its output to our terminal.

    ``env.exec()`` collects stdout/stderr with ``process.communicate()`` and
    only returns (prints) when the remote command *exits*, so a long
    ``prepare.sh`` -- which, even on a full cache hit, sha256-verifies multi-GB
    weights, gunzips the dataset archive and re-hashes the unpacked parquet --
    looks hung for minutes with zero output. We reuse the backend's own
    transport (``tsh``/``ssh``) but let the child inherit our stdout/stderr, so
    the script's ``[fetch] cache hit`` / ``[prepare] unpacking …`` lines appear
    live. Returns the remote exit code.
    """
    argv = [*env._transport_cmd(), f"bash -lc {shlex.quote(recipe)}"]
    proc = await asyncio.create_subprocess_exec(*argv)  # inherits stdout/stderr
    return await proc.wait()


def parse_kwargs(pairs: list[str]) -> dict[str, Any]:
    """Parse repeatable ``key=value`` into a dict, coercing known int fields.

    Mirrors Harbor's ``[environment].kwargs`` (passed verbatim to the
    CybertronEnvironment constructor), so project / cluster / resource_pool /
    billing_account_id / cluster_user / devspace_id / cache_dir / image_map_path
    all flow through unchanged.
    """
    out: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--environment-kwarg must be key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        key = key.strip()
        if key in _INT_KWARGS:
            out[key] = int(value)
        elif key in _BOOL_KWARGS:
            out[key] = _coerce_bool(key, value)
        else:
            out[key] = value
    return out


def build_stage_recipe(
    *,
    workspace: str = _REMOTE_WORKSPACE,
    models: str = _REMOTE_MODELS,
    datasets: str = _REMOTE_DATASETS,
    prepared: str = _REMOTE_PREPARED,
    environment: str = _REMOTE_ENVIRONMENT,
    tooling: str = _REMOTE_TOOLING,
    http_proxy: str | None = None,
    hf_endpoint: str | None = None,
) -> str:
    """Shell that installs the tooling then runs the uploaded task's prepare.sh.

    Thin wrapper kept for this script's CLI; the implementation is shared with
    CybertronEnvironment via harbor.environments.ptdbench_prepare so the manual
    staging path and the eval path run byte-identical prepare shells.
    """
    return build_prepare_recipe(
        workspace=workspace,
        models=models,
        datasets=datasets,
        prepared=prepared,
        environment=environment,
        tooling=tooling,
        http_proxy=http_proxy,
        hf_endpoint=hf_endpoint,
    )


def constructor_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve the CybertronEnvironment keyword args from parsed CLI args.

    --environment-kwarg pairs are the base; --image / --cache-dir are
    conveniences folded in as image_name / cache_dir (without clobbering an
    explicit pair). Passing devspace_id (via a pair) flips the backend into
    attach mode (_owns_devspace=False); omitting it creates a fresh devspace.
    """
    kwargs = parse_kwargs(args.environment_kwarg)
    if args.image is not None:
        kwargs.setdefault("image_name", args.image)
    if args.cache_dir is not None:
        kwargs.setdefault("cache_dir", args.cache_dir)
    return kwargs


def build_env_config(docker_image: str | None, gpu: int):
    """Build the EnvironmentConfig the backend validates + reads gpus from.

    docker_image (the task metadata image name the image_map resolves) OR an
    explicit image_name kwarg is required by _validate_definition; gpu becomes
    _effective_gpus.
    """
    from harbor.models.task.config import EnvironmentConfig

    return EnvironmentConfig(docker_image=docker_image, gpus=gpu)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task_root", help="PTDBench task directory (contains environment/)")
    p.add_argument(
        "--evaluation-level",
        default="solution",
        choices=["issue", "solution"],
        help="evaluation level (informational; staging is level-agnostic)",
    )
    p.add_argument("--gpu", type=int, default=8, help="devspace GPU count (default: 8)")
    p.add_argument(
        "--image",
        default=None,
        help="image name = the task metadata.yaml `image` field (e.g. "
        "slime-v0.3.2-sglang0.5.15-torch2.11.0-flashattn2.8.3-cu129-py3.12); "
        "mapped to a Cybertron canonicalName (highest precedence). Omit to use "
        "the task's own image.",
    )
    p.add_argument(
        "--docker-image",
        default=None,
        help="task image name used as the EnvironmentConfig.docker_image when "
        "staging a task that has none (alternative to --image)",
    )
    p.add_argument(
        "--cache-dir",
        default=None,
        help="shared cache root on the persistent fs "
        "(default: /user/<user>/ptdbench-cache); passed as cache_dir",
    )
    p.add_argument(
        "--http-proxy",
        default=_DEFAULT_HTTP_PROXY,
        help="whitelist egress proxy for HF fetches ('' to disable)",
    )
    p.add_argument(
        "--hf-endpoint",
        default=_DEFAULT_HF_ENDPOINT,
        help="HF mirror base URL ('' to disable)",
    )
    p.add_argument(
        "--environment-kwarg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra CybertronEnvironment kwarg (repeatable): project, cluster, "
        "resource_pool, billing_account_id, cluster_user, devspace_id, "
        "cache_dir, image_map_path, priority (HIGH/NORMAL/PREEMPTABLE), "
        "requeue (bool), …",
    )
    stop = p.add_mutually_exclusive_group()
    stop.add_argument(
        "--keep",
        dest="keep",
        action="store_true",
        default=True,
        help="leave the devspace running after staging (default)",
    )
    stop.add_argument(
        "--stop",
        dest="keep",
        action="store_false",
        help="stop an owned devspace after staging (ignored in attach mode)",
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="repo root holding docker/ (default: this script's repo parent)",
    )
    p.add_argument(
        "--run",
        action="store_true",
        help="actually create/attach + stage (default: dry-run, print the plan)",
    )
    return p.parse_args(argv)


def _repo_docker_dir(args: argparse.Namespace) -> Path:
    if args.repo_root is not None:
        return args.repo_root / "docker"
    # scripts/ -> harbor/ -> repo root
    return Path(__file__).resolve().parents[2] / "docker"


async def run_stage(args: argparse.Namespace) -> int:
    from harbor.environments.cybertron import CybertronEnvironment
    from harbor.models.trial.paths import TrialPaths

    task_root = Path(args.task_root).resolve()
    environment_dir = task_root / "environment"
    if not environment_dir.is_dir():
        raise SystemExit(f"{task_root} has no environment/ directory")

    kwargs = constructor_kwargs(args)

    scratch = Path("outputs") / "harbor-stage" / task_root.name
    scratch.mkdir(parents=True, exist_ok=True)

    env = CybertronEnvironment(
        environment_dir=environment_dir,
        environment_name=task_root.name,
        session_id=f"{task_root.name}__stage__env",
        trial_paths=TrialPaths(trial_dir=scratch),
        # GPU count rides on EnvironmentConfig.gpus (the backend reads
        # task_env_config.gpus, there is no `gpu` constructor kwarg).
        task_env_config=build_env_config(args.docker_image, args.gpu),
        **kwargs,
    )

    attach = "devspace_id" in kwargs
    if attach:
        _log(f"attaching to existing devspace {kwargs['devspace_id']} …")
    else:
        _log("creating devspace and waiting for it to become ready …")
    _log(
        "  (waiting for Running status + Teleport host registration + ssh; "
        "this can take a few minutes)"
    )
    await env.start(force_build=False)
    _log(f"devspace {env._devspace_id} ready; ssh host {env._host}")
    try:
        docker_dir = _repo_docker_dir(args)
        if docker_dir.is_dir():
            _log(f"uploading tooling {docker_dir} -> {_REMOTE_TOOLING} …")
            await env.upload_dir(docker_dir, _REMOTE_TOOLING)
        _log(f"uploading task environment/ -> {_REMOTE_ENVIRONMENT} …")
        await env.upload_dir(environment_dir, _REMOTE_ENVIRONMENT)
        workspace_src = task_root / "workspace"
        if workspace_src.is_dir():
            _log(f"uploading task workspace/ -> {_REMOTE_WORKSPACE} …")
            await env.upload_dir(workspace_src, _REMOTE_WORKSPACE)

        recipe = build_stage_recipe(
            http_proxy=args.http_proxy or None,
            hf_endpoint=args.hf_endpoint or None,
        )
        _log("running prepare.sh on the devspace (streaming output; even a warm")
        _log("  cache re-verifies multi-GB weights + unpacks the dataset) …")
        return_code = await _stream_exec(env, recipe)
        if return_code != 0:
            raise SystemExit(f"prepare.sh failed with exit {return_code}")

        _print_handoff(env, args)
    finally:
        # Attach mode never stops; owned devspace stopped only with --stop.
        await env.stop(delete=not args.keep)
    return 0


def _print_handoff(env, args: argparse.Namespace) -> None:
    """Tell the user how to ssh in and launch training by hand."""
    host = getattr(env, "_host", None)
    teleport = getattr(env, "_teleport_cluster", "teleport.cybertron.modelbest.co")
    devspace = getattr(env, "_devspace_id", None)
    lines = [
        "",
        "=" * 72,
        "Devspace staged to the pre-agent state.",
        f"  devspace: {devspace}",
        f"  ssh:      tsh ssh --cluster={teleport} root@{host}",
        "  then, inside:",
        f"    cd {_REMOTE_WORKSPACE}   # prepared workspace (weights downloaded)",
        "    # launch the task's training script by hand (see its README/metadata)",
    ]
    if args.keep:
        lines.append(
            f"  when done: cctl devspace stop {devspace}   # release the resources"
        )
    lines.append("=" * 72)
    print("\n".join(lines), file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.run:
        kwargs = constructor_kwargs(args)
        attach = "devspace_id" in kwargs
        print("dry-run: would stage to the pre-agent state on Cybertron")
        print(f"  task:    {args.task_root}")
        print(f"  gpu:     {args.gpu}")
        print(
            f"  image:   {args.image or args.docker_image or '(required: --image <name>)'}"
        )
        print(f"  mode:    {'attach' if attach else 'create'}")
        print(f"  kwargs:  {kwargs}")
        print(f"  keep:    {args.keep}")
        print("  re-run with --run to create/attach + stage.")
        return 0
    # Surface the backend's own INFO progress (devspace created, host resolved,
    # ssh ready) during the otherwise-silent, multi-minute env.start(); its
    # logger is DEBUG but unconfigured in this standalone script, so without a
    # handler those lines go nowhere.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    return asyncio.run(run_stage(args))


if __name__ == "__main__":
    raise SystemExit(main())
