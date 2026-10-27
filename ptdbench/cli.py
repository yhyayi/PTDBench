from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
from pathlib import Path

from .core.harbor_types import HarborEvaluationConfig
from .core.types import AgentSpec, EnvironmentImage, EvaluationConfig, known_images
from .evaluation.evaluator import TaskEvaluator
from .evaluation.harbor_evaluator import HarborEvaluator, HarborShell
from .runtimes.providers.docker import DockerRuntime
from .tasks import SchemaError


_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _parse_env_assignments(assignments: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for assignment in assignments:
        key, separator, value = assignment.partition("=")
        if not separator:
            raise SystemExit(f"environment assignment must be KEY=VALUE: {assignment}")
        if not _ENV_KEY_RE.fullmatch(key):
            raise SystemExit(f"invalid environment variable name: {key}")
        env[key] = value
    return env


def _is_task_root(path: Path) -> bool:
    """A PTDBench task is identified by its metadata file, as the loader does."""
    return any((path / name).is_file() for name in ("metadata.yaml", "metadata.yml"))


def _metadata_image_and_gpus(task_root: Path) -> tuple[str | None, int | None]:
    """Read the environment image name and GPU count from a task's metadata.yaml.

    The GPU count is the number of entries in
    ``environment.env.CUDA_VISIBLE_DEVICES`` — the same signal harbor-evaluate
    uses to size the devspace. Either value is ``None`` when the task does not
    declare it, leaving the caller's own default in place.
    """
    from .tasks import load_task_package

    package = load_task_package(task_root, strict=False)
    image = package.environment.image
    visible = (package.environment.env or {}).get("CUDA_VISIBLE_DEVICES")
    gpus = None
    if visible:
        gpus = len([part for part in str(visible).split(",") if part.strip()])
    return image, gpus


def expand_task_roots(paths: list[str]) -> list[Path]:
    """Expand each argument into task roots, keeping the caller's order.

    Accepts either a task directory or a directory whose children are task
    directories, so a whole family of tasks can be submitted as one argument.
    """
    roots: list[Path] = []
    seen: set[Path] = set()
    for raw in paths:
        path = Path(raw).expanduser()
        if not path.is_dir():
            raise SystemExit(f"task path does not exist or is not a directory: {raw}")
        if _is_task_root(path):
            found = [path]
        else:
            found = sorted(
                child
                for child in path.iterdir()
                if child.is_dir() and _is_task_root(child)
            )
        if not found:
            raise SystemExit(
                f"{raw} is neither a task directory (no metadata.yaml) nor a "
                "directory containing task directories"
            )
        for task_root in found:
            resolved = task_root.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            roots.append(task_root)
    return roots


def build_parser() -> argparse.ArgumentParser:
    # 两个 evaluator 使用两个明确的入口，避免把 Harbor 混进 runtime 选择。
    ap = argparse.ArgumentParser(prog="ptdbench", allow_abbrev=False)
    sub = ap.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate", help="validate a task package", allow_abbrev=False)
    v.add_argument("task_root")
    v.add_argument("--lenient", action="store_true", help="skip non-critical directory warnings")

    e = sub.add_parser(
        "evaluate",
        help="run an agent + verifier for a task package",
        allow_abbrev=False,
    )
    e.add_argument("task_root")
    e.add_argument(
        "--evaluation-level",
        choices=["issue", "solution"],
        default="solution",
        help="task evaluation to run",
    )
    e.add_argument(
        "--log-level",
        choices=["quiet", "debug"],
        default="quiet",
        help="terminal logging level; run artifacts are always written",
    )
    e.add_argument(
        "--agent",
        choices=["codex", "opencode", "shell"],
        default=None,
        help="agent adapter to use",
    )
    e.add_argument("--agent-command", default=None, help="agent executable or shell command")
    e.add_argument("--model", default=None, help="agent model name")
    e.add_argument("--judge-model", default=None, help="model name used by API-backed verifiers")
    e.add_argument(
        "--reasoning-effort",
        choices=["minimal", "low", "medium", "high", "xhigh"],
        default=None,
        help="reasoning effort for agents that support it",
    )
    e.add_argument("--base-url", default=None, help="API base URL for hosted agents")
    e.add_argument("--api-key", default=None, help="API key for hosted agents")
    e.add_argument(
        "--api-key-env",
        default=None,
        help="read the API key from this environment variable (keeps it out of argv)",
    )
    e.add_argument("--agent-arg", action="append", default=[], help="extra agent argument, repeatable")
    e.add_argument(
        "--agent-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="environment variable for the agent stage; repeatable",
    )
    e.add_argument(
        "--verifier-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="environment variable for the verifier stage; repeatable",
    )
    e.add_argument("--agent-cmd", default=None, help=argparse.SUPPRESS)
    e.add_argument("--verifier-cmd", default=None, help="override verifier entrypoint")
    e.add_argument("--backend", choices=["docker", "local"], default="docker")
    e.add_argument("--agent-network", choices=["none", "bridge", "host"], default="bridge")
    e.add_argument("--verifier-network", choices=["none", "bridge", "host"], default="bridge")
    e.add_argument(
        "--debian-mirror",
        default=None,
        help="Debian mirror base URL applied before agent setup package installs",
    )
    e.add_argument(
        "--huggingface-mirror",
        default=None,
        help="Hugging Face mirror base URL, passed to the prepare stage as HF_ENDPOINT",
    )
    e.add_argument(
        "--models-dir",
        default=os.environ.get("PTDBENCH_MODELS_DIR"),
        help=(
            "host directory shared by every run as the model download cache; "
            "mounted writable into the prepare stage and read-only into the "
            "agent and verifier stages, so a model fetched by one task is not "
            "fetched again by the next (default: <repo>/models) "
            "[env: PTDBENCH_MODELS_DIR]"
        ),
    )
    e.add_argument(
        "--datasets-dir",
        default=os.environ.get("PTDBENCH_DATASETS_DIR"),
        help=(
            "same as --models-dir, for datasets (default: <repo>/datasets) "
            "[env: PTDBENCH_DATASETS_DIR]"
        ),
    )
    e.add_argument("--timeout-s", type=int, default=None)
    e.add_argument("--keep-artifacts", action="store_true")
    e.add_argument("--work-dir", default=None, help="parent directory for PTDBench run artifacts")
    e.add_argument("--json", action="store_true", help="print JSON only")

    b = sub.add_parser(
        "build-images",
        help="pull the environment images under docker/images/ ahead of time",
        description=(
            "Pull each environment image referenced by docker/images/<name>.Dockerfile's "
            "FROM line. Nothing is baked anymore, so there is no build step -- "
            "`evaluate` pulls a missing image on demand; this command lets a "
            "machine pay that cost up front. Images that already exist are "
            "skipped unless --rebuild is given."
        ),
        allow_abbrev=False,
    )
    b.add_argument(
        "names",
        nargs="*",
        metavar="NAME",
        help="environment names (default: every docker/images/*.Dockerfile)",
    )
    b.add_argument(
        "--rebuild",
        action="store_true",
        help=(
            "rebuild even if the image exists; needed after editing a Dockerfile "
            "or the tooling under docker/, since the tag does not change"
        ),
    )

    h = sub.add_parser(
        "harbor-evaluate",
        help=(
            "run each PTDBench task's prepare.sh, then evaluate it with Harbor "
            "on the prebuilt environment image"
        ),
        allow_abbrev=False,
        epilog="All other arguments are forwarded unchanged to `harbor run`.",
    )
    h.add_argument(
        "task_root",
        nargs="+",
        help=(
            "task directory or a directory whose direct children are tasks; "
            "all task paths must precede forwarded Harbor arguments"
        ),
    )
    h.add_argument(
        "--evaluation-level",
        choices=["issue", "solution"],
        default="solution",
        help="task evaluation to run",
    )
    h.add_argument("--judge-model", default=None, help="model used by PTDBench verifiers")
    h.add_argument(
        "--huggingface-mirror",
        default=None,
        help="Hugging Face mirror base URL, passed to the prepare stage as HF_ENDPOINT",
    )
    h.add_argument(
        "--gpus",
        type=int,
        default=None,
        help=(
            "remote backends (--env cybertron): GPU count for the devspace, "
            "written to the generated task.toml as [environment].gpus "
            "(default: the number of entries in the task's CUDA_VISIBLE_DEVICES)"
        ),
    )
    h.add_argument(
        "--models-dir",
        default=os.environ.get("PTDBENCH_MODELS_DIR"),
        help=(
            "host model download cache; writable in prepare, mounted read-only "
            "into the Harbor container (default: <repo>/models) "
            "[env: PTDBENCH_MODELS_DIR]"
        ),
    )
    h.add_argument(
        "--datasets-dir",
        default=os.environ.get("PTDBENCH_DATASETS_DIR"),
        help=(
            "same as --models-dir, for datasets (default: <repo>/datasets) "
            "[env: PTDBENCH_DATASETS_DIR]"
        ),
    )
    h.add_argument(
        "--agent-network",
        choices=["no-network", "public", "allowlist"],
        default="public",
        help="network_mode written to the generated Harbor task agent phase",
    )
    h.add_argument(
        "--verifier-network",
        choices=["no-network", "public", "allowlist"],
        default="public",
        help="network_mode written to the generated Harbor task verifier phase",
    )
    h.add_argument(
        "--harbor-command",
        default="harbor",
        help="Harbor executable command, e.g. 'uv run --project harbor harbor'",
    )
    h.add_argument(
        "--harbor-temp-dir",
        type=Path,
        default=None,
        help="parent directory for temporary materialized Harbor tasks",
    )

    s = sub.add_parser(
        "harbor-shell",
        help=(
            "run a task's prepare.sh, then drop into an interactive shell inside "
            "the prebuilt environment image — same container as harbor-evaluate's "
            "agent phase, but no agent is started"
        ),
        allow_abbrev=False,
    )
    s.add_argument(
        "task_root",
        help="task directory (must contain metadata.yaml)",
    )
    s.add_argument(
        "--evaluation-level",
        choices=["issue", "solution"],
        default="solution",
        help="evaluation level whose environment variables and workspace are used",
    )
    s.add_argument(
        "--huggingface-mirror",
        default=None,
        help="Hugging Face mirror base URL, passed to the prepare stage as HF_ENDPOINT",
    )
    s.add_argument(
        "--models-dir",
        default=os.environ.get("PTDBENCH_MODELS_DIR"),
        help=(
            "host model download cache; mounted read-only into the container "
            "(default: <repo>/models) [env: PTDBENCH_MODELS_DIR]"
        ),
    )
    s.add_argument(
        "--datasets-dir",
        default=os.environ.get("PTDBENCH_DATASETS_DIR"),
        help=(
            "same as --models-dir, for datasets (default: <repo>/datasets) "
            "[env: PTDBENCH_DATASETS_DIR]"
        ),
    )
    s.add_argument(
        "--workspace-dir",
        type=Path,
        default=None,
        help=(
            "directory where the staged workspace and state file are stored "
            "across runs; defaults to outputs/harbor-shell/<task-id>"
        ),
    )
    s.add_argument(
        "--force-prepare",
        action="store_true",
        help="re-run prepare.sh even if a staged workspace already exists",
    )
    s.add_argument(
        "--shell",
        default="bash",
        dest="shell_cmd",
        help="shell to start inside the container (default: bash)",
    )
    s.add_argument(
        "--gpus",
        default=None,
        help=(
            "value passed to docker run --gpus; auto-detected from "
            "environment/docker-compose.yaml when absent"
        ),
    )
    s.add_argument(
        "--shm-size",
        default=None,
        help=(
            "value passed to docker run --shm-size; auto-detected from "
            "environment/docker-compose.yaml when absent"
        ),
    )

    g = sub.add_parser(
        "harbor-stage",
        help=(
            "stage a task to the pre-agent state (prepared workspace, downloaded "
            "weights) then stop — like harbor-shell, but also runs on a multi-GPU "
            "Cybertron devspace via --backend cybertron"
        ),
        allow_abbrev=False,
    )
    g.add_argument("task_root", help="task directory (must contain metadata.yaml)")
    g.add_argument(
        "--backend",
        choices=["docker", "cybertron"],
        default="docker",
        help=(
            "where to stage: local docker (interactive shell, like harbor-shell) "
            "or a Cybertron devspace (left running for manual experiments)"
        ),
    )
    g.add_argument(
        "--evaluation-level",
        choices=["issue", "solution"],
        default="solution",
        help="evaluation level whose environment variables and workspace are used",
    )
    g.add_argument(
        "--huggingface-mirror",
        default=None,
        help="Hugging Face mirror base URL, passed to the prepare stage as HF_ENDPOINT",
    )
    g.add_argument(
        "--models-dir",
        default=os.environ.get("PTDBENCH_MODELS_DIR"),
        help="docker backend only: host model cache [env: PTDBENCH_MODELS_DIR]",
    )
    g.add_argument(
        "--datasets-dir",
        default=os.environ.get("PTDBENCH_DATASETS_DIR"),
        help="docker backend only: host dataset cache [env: PTDBENCH_DATASETS_DIR]",
    )
    g.add_argument(
        "--workspace-dir",
        type=Path,
        default=None,
        help="docker backend only: staged workspace/state dir (default: "
        "outputs/harbor-stage/<task-id>)",
    )
    g.add_argument(
        "--force-prepare",
        action="store_true",
        help="docker backend only: re-run prepare.sh even if already staged",
    )
    g.add_argument(
        "--shell",
        default="bash",
        dest="shell_cmd",
        help="docker backend only: shell to start in the container (default: bash)",
    )
    g.add_argument(
        "--gpus",
        default=None,
        help="docker backend: docker run --gpus value; cybertron backend: "
        "devspace GPU count (default: the task metadata CUDA_VISIBLE_DEVICES "
        "count, else 8)",
    )
    g.add_argument(
        "--shm-size",
        default=None,
        help="docker backend only: docker run --shm-size value",
    )
    g.add_argument(
        "--environment-kwarg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="cybertron backend: CybertronEnvironment kwarg (repeatable): "
        "project, cluster, resource_pool, billing_account_id, cluster_user, …",
    )
    g.add_argument(
        "--devspace-id",
        default=None,
        help="cybertron backend: attach to this existing devspace instead of "
        "creating one (it is never stopped)",
    )
    g.add_argument(
        "--image",
        default=None,
        help="cybertron backend: image name = the task metadata.yaml `image` "
        "field (e.g. slime-v0.3.2-sglang0.5.15-torch2.11.0-flashattn2.8.3-cu129-"
        "py3.12); mapped to a Cybertron canonicalName. Defaults to the task's own "
        "metadata image, so you rarely need to pass it",
    )
    g.add_argument(
        "--keep",
        dest="keep",
        action="store_true",
        default=True,
        help="cybertron backend: leave the devspace running after staging (default)",
    )
    g.add_argument(
        "--stop",
        dest="keep",
        action="store_false",
        help="cybertron backend: stop an owned devspace after staging",
    )
    g.add_argument(
        "--run",
        action="store_true",
        help="cybertron backend: actually create/attach + stage (default: dry-run)",
    )
    g.add_argument(
        "--stage-command",
        default="uv run --project harbor python harbor/scripts/cybertron_stage.py",
        help="cybertron backend: executable for the harbor-side stager",
    )
    return ap


def main(argv=None) -> int:
    parser = build_parser()
    args, harbor_args = parser.parse_known_args(argv)
    if args.command != "harbor-evaluate" and harbor_args:
        parser.error("unrecognized arguments: " + " ".join(harbor_args))

    if args.command == "validate":
        evaluator = TaskEvaluator()
        try:
            report = evaluator.validate(args.task_root, strict=not args.lenient)
        except SchemaError as exc:
            # 结构性错误（YAML 解析失败、environment 字段非法）本来会直接抛栈，但
            # validate 的存在意义就是把问题讲清楚，所以也按 report 的形状输出。
            report = {
                "ok": False,
                "issues": [{"severity": "error", "path": args.task_root, "message": str(exc)}],
            }
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report.get("ok", False) else 1

    if args.command == "build-images":
        known = known_images()
        if not known:
            raise SystemExit("no docker/images/*.Dockerfile found")
        names = args.names or sorted(known)
        unknown = [name for name in names if name not in known]
        if unknown:
            raise SystemExit(
                f"unknown environment(s): {', '.join(unknown)}; "
                f"expected one of {', '.join(sorted(known))}"
            )
        runtime = DockerRuntime()
        for name in dict.fromkeys(names):
            image = EnvironmentImage(name)
            try:
                runtime.ensure_environment(image, rebuild=args.rebuild)
            except subprocess.CalledProcessError as exc:
                raise SystemExit(f"pulling {image.tag} failed (exit {exc.returncode})")
            print(image.tag)
        return 0

    if args.command == "evaluate":
        evaluator = TaskEvaluator(backend=args.backend)
        agent_command = args.agent_command or args.agent_cmd
        if args.api_key is not None and args.api_key_env is not None:
            raise SystemExit("--api-key and --api-key-env are mutually exclusive")
        api_key = args.api_key
        if args.api_key_env:
            api_key = os.environ.get(args.api_key_env)
            if not api_key:
                raise SystemExit(
                    f"API key environment variable is unset or empty: {args.api_key_env}"
                )
        agent_env = _parse_env_assignments(args.agent_env)
        verifier_env = _parse_env_assignments(args.verifier_env)
        agent_kind = args.agent
        if agent_kind is None:
            if (
                args.model
                or args.judge_model
                or args.reasoning_effort
                or args.base_url
                or api_key
                or args.agent_arg
                or agent_env
            ):
                agent_kind = "codex"
            elif agent_command:
                agent_kind = "shell"
        agent_spec = None
        if (
            agent_kind is not None
            or agent_command
            or args.model
            or args.judge_model
            or args.reasoning_effort
            or args.base_url
            or api_key
            or args.agent_arg
            or agent_env
        ):
            agent_spec = AgentSpec(
                kind=agent_kind or "codex",
                command=agent_command,
                model=args.model,
                judge_model=args.judge_model,
                reasoning_effort=args.reasoning_effort,
                base_url=args.base_url,
                api_key=api_key,
                extra_args=tuple(args.agent_arg),
                network=args.agent_network,
                env=agent_env,
            )
        result = evaluator.evaluate(
            args.task_root,
            EvaluationConfig(
                agent=agent_spec,
                agent_cmd=agent_command,
                verifier_cmd=args.verifier_cmd,
                backend=args.backend,
                verifier_network=args.verifier_network,
                timeout_s=args.timeout_s,
                keep_artifacts=args.keep_artifacts,
                work_dir=None if args.work_dir is None else Path(args.work_dir),
                evaluation_level=args.evaluation_level,
                log_level=args.log_level,
                debian_mirror=args.debian_mirror,
                huggingface_mirror=args.huggingface_mirror,
                verifier_env=verifier_env,
                models_dir=None if args.models_dir is None else Path(args.models_dir),
                datasets_dir=(
                    None if args.datasets_dir is None else Path(args.datasets_dir)
                ),
            ),
        )
        payload = result.to_dict()
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0 if result.passed else 1

    if args.command == "harbor-evaluate":
        harbor_command = tuple(shlex.split(args.harbor_command))
        report = HarborEvaluator().evaluate(
            expand_task_roots(args.task_root),
            HarborEvaluationConfig(
                evaluation_level=args.evaluation_level,
                judge_model=args.judge_model,
                agent_network=args.agent_network,
                verifier_network=args.verifier_network,
                temp_dir=args.harbor_temp_dir,
                huggingface_mirror=args.huggingface_mirror,
                models_dir=None if args.models_dir is None else Path(args.models_dir),
                datasets_dir=(
                    None if args.datasets_dir is None else Path(args.datasets_dir)
                ),
                harbor_command=harbor_command,
                harbor_args=tuple(harbor_args),
                gpus=args.gpus,
            ),
        )
        result = (
            report.trials[0]
            if len(report.task_ids) == 1 and len(report.trials) == 1
            else report
        )
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return 0 if result.passed else 1

    if args.command == "harbor-shell":
        from .evaluation.harbor_evaluator import HarborShell

        task_roots = expand_task_roots([args.task_root])
        if len(task_roots) != 1:
            raise SystemExit("harbor-shell accepts exactly one task directory")
        task_root = task_roots[0]
        workspace_dir = (
            Path(args.workspace_dir)
            if args.workspace_dir is not None
            else Path("outputs") / "harbor-shell" / task_root.name
        )
        HarborShell().shell(
            task_root,
            HarborEvaluationConfig(
                evaluation_level=args.evaluation_level,
                huggingface_mirror=args.huggingface_mirror,
                models_dir=None if args.models_dir is None else Path(args.models_dir),
                datasets_dir=(
                    None if args.datasets_dir is None else Path(args.datasets_dir)
                ),
            ),
            workspace_dir,
            shell_cmd=args.shell_cmd,
            force_prepare=args.force_prepare,
            gpus=args.gpus,
            shm_size=args.shm_size,
        )
        return 0  # unreachable: shell() calls os.execvp

    if args.command == "harbor-stage":
        task_roots = expand_task_roots([args.task_root])
        if len(task_roots) != 1:
            raise SystemExit("harbor-stage accepts exactly one task directory")
        task_root = task_roots[0]

        if args.backend == "docker":
            # Identical to harbor-shell: prepare then drop into the pre-agent
            # container. Reused verbatim so there is one staging path for docker.
            from .evaluation.harbor_evaluator import HarborShell

            workspace_dir = (
                Path(args.workspace_dir)
                if args.workspace_dir is not None
                else Path("outputs") / "harbor-stage" / task_root.name
            )
            HarborShell().shell(
                task_root,
                HarborEvaluationConfig(
                    evaluation_level=args.evaluation_level,
                    huggingface_mirror=args.huggingface_mirror,
                    models_dir=(
                        None if args.models_dir is None else Path(args.models_dir)
                    ),
                    datasets_dir=(
                        None if args.datasets_dir is None else Path(args.datasets_dir)
                    ),
                ),
                workspace_dir,
                shell_cmd=args.shell_cmd,
                force_prepare=args.force_prepare,
                gpus=args.gpus,
                shm_size=args.shm_size,
            )
            return 0  # unreachable: shell() calls os.execvp

        # cybertron: shell out to the harbor-side stager (ptdbench never imports
        # harbor as a library — same boundary harbor-evaluate uses for harbor run).
        # --image / --gpus default to the task's metadata.yaml (image name and the
        # CUDA_VISIBLE_DEVICES count) so neither has to be passed by hand; an
        # explicit flag still wins.
        meta_image, meta_gpus = _metadata_image_and_gpus(task_root)
        gpus = args.gpus if args.gpus is not None else meta_gpus
        image = args.image if args.image is not None else meta_image
        cmd = [*shlex.split(args.stage_command), str(task_root)]
        cmd += ["--evaluation-level", args.evaluation_level]
        cmd += ["--gpu", str(gpus) if gpus is not None else "8"]
        if image is not None:
            cmd += ["--image", image]
        if args.huggingface_mirror is not None:
            cmd += ["--hf-endpoint", args.huggingface_mirror]
        if args.devspace_id is not None:
            cmd += ["--environment-kwarg", f"devspace_id={args.devspace_id}"]
        for pair in args.environment_kwarg:
            cmd += ["--environment-kwarg", pair]
        cmd.append("--keep" if args.keep else "--stop")
        if args.run:
            cmd.append("--run")
        return subprocess.run(cmd).returncode

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
