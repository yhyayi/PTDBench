from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .core.types import AgentSpec, EvaluationConfig
from .evaluation.evaluator import CaseEvaluator


def build_parser() -> argparse.ArgumentParser:
    # CLI 先保持很小：只做 validate 和 evaluate。
    ap = argparse.ArgumentParser(prog="ptdbench")
    sub = ap.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate", help="validate a case package")
    v.add_argument("case_root")
    v.add_argument("--lenient", action="store_true", help="skip non-critical directory warnings")

    e = sub.add_parser("evaluate", help="run an agent + verifier for a case package")
    e.add_argument("case_root")
    e.add_argument(
        "--agent",
        choices=["codex", "opencode", "shell"],
        default=None,
        help="agent adapter to use",
    )
    e.add_argument("--agent-command", default=None, help="agent executable or shell command")
    e.add_argument("--model", default=None, help="agent model name")
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
    e.add_argument("--agent-cmd", default=None, help=argparse.SUPPRESS)
    e.add_argument("--verifier-cmd", default=None, help="override verifier entrypoint")
    e.add_argument("--backend", choices=["docker", "local"], default="docker")
    e.add_argument("--agent-backend", choices=["docker", "local"], default=None)
    e.add_argument("--agent-network", choices=["none", "bridge", "host"], default="bridge")
    e.add_argument("--verifier-network", choices=["none", "bridge", "host"], default="none")
    e.add_argument("--image-tag-prefix", default="ptdbench")
    e.add_argument("--timeout-s", type=int, default=None)
    e.add_argument("--keep-artifacts", action="store_true")
    e.add_argument("--work-dir", default=None, help="parent directory for PTDBench run artifacts")
    e.add_argument("--json", action="store_true", help="print JSON only")

    return ap


def main(argv=None) -> int:
    # CLI 只是 CaseEvaluator 的薄封装。
    args = build_parser().parse_args(argv)
    evaluator = CaseEvaluator(backend=getattr(args, "backend", "docker"))

    if args.command == "validate":
        report = evaluator.validate(args.case_root, strict=not args.lenient)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report.get("ok", False) else 1

    if args.command == "evaluate":
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
        agent_kind = args.agent
        if agent_kind is None:
            if args.model or args.reasoning_effort or args.base_url or api_key or args.agent_arg:
                agent_kind = "codex"
            elif agent_command:
                agent_kind = "shell"
        agent_spec = None
        if (
            agent_kind is not None
            or agent_command
            or args.model
            or args.reasoning_effort
            or args.base_url
            or api_key
            or args.agent_arg
        ):
            agent_spec = AgentSpec(
                kind=agent_kind or "codex",
                command=agent_command,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                base_url=args.base_url,
                api_key=api_key,
                extra_args=tuple(args.agent_arg),
                backend=args.agent_backend,
                network=args.agent_network,
            )
        result = evaluator.evaluate(
            args.case_root,
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
            ),
        )
        payload = result.to_dict()
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0 if result.passed else 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
