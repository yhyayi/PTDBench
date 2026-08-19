from __future__ import annotations

import argparse
import json
import os
import re
import shlex
from pathlib import Path

from .core.harbor_types import HarborEvaluationConfig
from .core.types import AgentSpec, EvaluationConfig
from .evaluation.evaluator import TaskEvaluator
from .evaluation.harbor_evaluator import HarborEvaluator


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
    e.add_argument("--image-tag-prefix", default="ptdbench")
    e.add_argument("--timeout-s", type=int, default=None)
    e.add_argument("--keep-artifacts", action="store_true")
    e.add_argument("--work-dir", default=None, help="parent directory for PTDBench run artifacts")
    e.add_argument("--json", action="store_true", help="print JSON only")

    h = sub.add_parser(
        "harbor-evaluate",
        help="materialize a PTDBench task and run native Harbor arguments",
        allow_abbrev=False,
        epilog="All other arguments are forwarded unchanged to `harbor run`.",
    )
    h.add_argument("task_root")
    h.add_argument(
        "--evaluation-level",
        choices=["issue", "solution"],
        default="solution",
        help="task evaluation to run",
    )
    h.add_argument(
        "--verifier-mode",
        choices=["shared", "separate"],
        default="separate",
        help="environment_mode written to the generated Harbor task",
    )
    h.add_argument("--judge-model", default=None, help="model used by PTDBench verifiers")
    h.add_argument(
        "--debian-mirror",
        default=None,
        help="Debian mirror base URL written to the materialized environment",
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
    return ap


def main(argv=None) -> int:
    parser = build_parser()
    args, harbor_args = parser.parse_known_args(argv)
    if args.command != "harbor-evaluate" and harbor_args:
        parser.error("unrecognized arguments: " + " ".join(harbor_args))

    if args.command == "validate":
        evaluator = TaskEvaluator()
        report = evaluator.validate(args.task_root, strict=not args.lenient)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report.get("ok", False) else 1

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
                image_tag_prefix=args.image_tag_prefix,
                timeout_s=args.timeout_s,
                keep_artifacts=args.keep_artifacts,
                work_dir=None if args.work_dir is None else Path(args.work_dir),
                evaluation_level=args.evaluation_level,
                log_level=args.log_level,
                debian_mirror=args.debian_mirror,
                verifier_env=verifier_env,
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
            args.task_root,
            HarborEvaluationConfig(
                evaluation_level=args.evaluation_level,
                judge_model=args.judge_model,
                verifier_mode=args.verifier_mode,
                agent_network=args.agent_network,
                verifier_network=args.verifier_network,
                debian_mirror=args.debian_mirror,
                temp_dir=args.harbor_temp_dir,
                harbor_command=harbor_command,
                harbor_args=tuple(harbor_args),
            ),
        )
        result = report.trials[0] if len(report.trials) == 1 else report
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return 0 if result.passed else 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
