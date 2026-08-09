#!/usr/bin/env python3
"""Self-contained per-task diagnosis verifier (standard library only)."""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import fcntl
import json
import os
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Sequence, Tuple


MAX_DIAGNOSIS_BYTES = 64 * 1024
MAX_AGENT_TRACE_BYTES = 8 * 1024 * 1024
DEFAULT_AGENT_IMAGE = "ptdbench-agent-harness:latest"
ALLOWED_TASK_KEYS = {
    "task_id",
    "environment",
    "hollow_file",
    "hollow_files",
    "hollow_symbols",
    "hollow_hparams",
    "baseline_path",
    "oracle",
}
ALLOWED_ORACLE_KEYS = {"buggy_keys", "buggy_args", "golden_args"}
FORBIDDEN_KEY_PARTS = ("hash", "digest", "sha", "checksum")
JUDGE_KEYS = {
    "score",
    "passed",
    "verdict_match",
    "attribution_match",
    "specificity_match",
    "matched_requirements",
    "missing_requirements",
    "rationale",
}
ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


class EvaluationError(RuntimeError):
    """A fail-closed evaluation error."""


class SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise EvaluationError("CLI error: " + message)


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value.strip())
    except ValueError as exc:
        raise EvaluationError(f"{name} must be an integer") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = SafeArgumentParser(
        description="Run isolated Agent A and direct-HTTP Judge B attribution evaluation."
    )
    parser.add_argument(
        "--evaluation-mode",
        choices=("auto", "standalone", "judge-only"),
        default=os.environ.get("PTDBENCH_EVALUATION_MODE", "auto"),
        help=(
            "auto uses judge-only only when central PTDBench supplied both an "
            "Agent trace and diagnosis; otherwise it uses standalone; standalone "
            "runs Agent A and Judge B; judge-only grades an existing staged "
            "diagnosis."
        ),
    )
    parser.add_argument(
        "--agent-harness",
        choices=("codex", "claude", "gemini", "opencode"),
        default=os.environ.get("PTDBENCH_AGENT_HARNESS", "codex"),
    )
    parser.add_argument(
        "--agent-model", default=os.environ.get("PTDBENCH_AGENT_MODEL", "")
    )
    parser.add_argument(
        "--agent-image",
        default=os.environ.get("PTDBENCH_AGENT_IMAGE", DEFAULT_AGENT_IMAGE),
        help=(
            "Agent container image. The default image is built automatically "
            "from verifier/agent.Dockerfile when it is not already local; a "
            "missing custom image is pulled from its registry."
        ),
    )
    parser.add_argument(
        "--agent-api-key-env",
        default=os.environ.get("PTDBENCH_AGENT_API_KEY_ENV", ""),
        help="Name of the parent environment variable containing Agent A's key.",
    )
    parser.add_argument(
        "--agent-base-url", default=os.environ.get("PTDBENCH_AGENT_BASE_URL", "")
    )
    parser.add_argument(
        "--agent-network",
        default=os.environ.get("PTDBENCH_AGENT_NETWORK", "bridge"),
    )
    parser.add_argument(
        "--agent-timeout",
        type=int,
        default=env_int("PTDBENCH_AGENT_TIMEOUT", 0),
        help="Seconds before Agent cancellation; unset or 0 means unlimited.",
    )
    parser.add_argument(
        "--judge-provider",
        choices=("openai-compatible", "anthropic", "gemini"),
        default=os.environ.get(
            "PTDBENCH_JUDGE_PROVIDER", "openai-compatible"
        ),
    )
    parser.add_argument(
        "--judge-model", default=os.environ.get("PTDBENCH_JUDGE_MODEL", "")
    )
    parser.add_argument(
        "--judge-api-key-env",
        default=os.environ.get("PTDBENCH_JUDGE_API_KEY_ENV", ""),
        help="Name of the parent environment variable containing Judge B's key.",
    )
    parser.add_argument(
        "--judge-base-url", default=os.environ.get("PTDBENCH_JUDGE_BASE_URL", "")
    )
    parser.add_argument(
        "--judge-timeout",
        type=int,
        default=env_int("PTDBENCH_JUDGE_TIMEOUT", 0),
        help="Seconds before Judge cancellation; unset or 0 means unlimited.",
    )
    parser.add_argument("--run-id", default="")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--agent-command", default="")
    parser.add_argument("--judge-command", default="")
    return parser


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + secrets.token_hex(4))
    temporary.write_text(text, encoding="utf-8")
    os.replace(str(temporary), str(path))


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def redact(text: str, secret_values: Sequence[str]) -> str:
    result = text
    variants = set()
    for item in secret_values:
        if not item:
            continue
        variants.update(
            {
                item,
                urllib.parse.quote(item, safe=""),
                urllib.parse.quote_plus(item, safe=""),
                json.dumps(item, ensure_ascii=False)[1:-1],
            }
        )
    for value in sorted(variants, key=len, reverse=True):
        result = result.replace(value, "[REDACTED]")
    return result


def validate_env_name(value: str, label: str) -> None:
    if not ENV_NAME_RE.fullmatch(value):
        raise EvaluationError(f"{label} must be an environment-variable name")


def validate_url(value: str, label: str) -> None:
    if not value:
        return
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise EvaluationError(
            f"{label} must be an http(s) base URL without credentials, query, or fragment"
        )


def scalar_value(raw: str) -> Any:
    value = raw.strip()
    if value in ("", "null", "Null", "NULL", "~"):
        return None
    if value.startswith(("'", '"')) and value.endswith(value[0]):
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, str) else value
        except (SyntaxError, ValueError):
            return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else value
        except json.JSONDecodeError:
            return value
    return value


def forbidden_context_key(key: str) -> bool:
    lowered = key.lower()
    return lowered == "baselines" or any(part in lowered for part in FORBIDDEN_KEY_PARTS)


def reject_forbidden_context_keys(value: Any) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if forbidden_context_key(str(key)):
                raise EvaluationError(f"task context contains forbidden key: {key}")
            reject_forbidden_context_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            reject_forbidden_context_keys(nested)


def parse_task_yaml_allowlist(text: str) -> Dict[str, Any]:
    """Extract only the benchmark-approved task fields from simple YAML."""
    context: Dict[str, Any] = {}
    active_top: Optional[str] = None
    active_oracle: Optional[str] = None
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        top = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):(?:[ \t]*(.*))?$", raw_line)
        if top:
            key, raw_value = top.group(1), top.group(2) or ""
            active_oracle = None
            active_top = key if key in ALLOWED_TASK_KEYS else None
            if active_top == "oracle":
                context["oracle"] = {}
            elif active_top:
                value = scalar_value(raw_value)
                context[active_top] = [] if value is None and not raw_value else value
            continue
        if active_top == "oracle":
            oracle_key = re.match(
                r"^  ([A-Za-z_][A-Za-z0-9_]*):(?:[ \t]*(.*))?$", raw_line
            )
            if oracle_key:
                key, raw_value = oracle_key.group(1), oracle_key.group(2) or ""
                active_oracle = key if key in ALLOWED_ORACLE_KEYS else None
                if active_oracle:
                    value = scalar_value(raw_value)
                    context["oracle"][active_oracle] = (
                        [] if value is None and not raw_value else value
                    )
                continue
            if active_oracle and re.match(r"^  -[ \t]+", raw_line):
                target = context["oracle"].get(active_oracle)
                if not isinstance(target, list):
                    target = []
                    context["oracle"][active_oracle] = target
                target.append(
                    scalar_value(re.sub(r"^  -[ \t]+", "", raw_line, count=1))
                )
                continue
            nested = re.match(
                r"^    ([A-Za-z_][A-Za-z0-9_.-]*):(?:[ \t]*(.*))?$", raw_line
            )
            if active_oracle and nested:
                target = context["oracle"].get(active_oracle)
                if not isinstance(target, dict):
                    target = {}
                    context["oracle"][active_oracle] = target
                target[nested.group(1)] = scalar_value(nested.group(2) or "")
                continue
            continue
        if active_top and re.match(r"^[ \t]*-[ \t]+", raw_line):
            target = context.get(active_top)
            if not isinstance(target, list):
                target = []
                context[active_top] = target
            target.append(
                scalar_value(re.sub(r"^[ \t]*-[ \t]+", "", raw_line, count=1))
            )
            continue
        active_top = None

    if not context.get("oracle"):
        context.pop("oracle", None)
    baseline = context.get("baseline_path")
    if isinstance(baseline, str):
        context["baseline_path"] = sanitize_baseline_path(
            baseline, context.get("hollow_file")
        )
    reject_forbidden_context_keys(context)
    return {key: context[key] for key in ALLOWED_TASK_KEYS if key in context}


def sanitize_baseline_path(baseline: str, hollow_file: Any) -> str:
    normalized = baseline.replace("\\", "/").lstrip("/")
    if isinstance(hollow_file, str) and normalized.endswith(hollow_file):
        return hollow_file
    for prefix in (
        "hidden/baseline/",
        "oracle/code_target/baseline/",
        "oracle/baseline/",
        "baseline/",
    ):
        if normalized.startswith(prefix):
            return normalized[len(prefix) :]
    if baseline.startswith("/"):
        return PurePosixPath(normalized).name
    parts = [part for part in PurePosixPath(normalized).parts if part not in (".", "..")]
    return "/".join(parts)


def assemble_agent_prompt(template: str, instruction: str) -> str:
    marker = "{{INSTRUCTION}}"
    if marker not in template:
        raise EvaluationError("agent_prompt.txt is missing the instruction marker")
    return template.replace(marker, instruction.rstrip()) + "\n"


def validate_judge_result(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict) or set(value) != JUDGE_KEYS:
        raise EvaluationError("Judge JSON must contain exactly the required fields")
    if type(value["score"]) is not int or value["score"] not in (0, 1):
        raise EvaluationError("Judge score must be the integer 0 or 1")
    for key in (
        "passed",
        "verdict_match",
        "attribution_match",
        "specificity_match",
    ):
        if type(value[key]) is not bool:
            raise EvaluationError(f"Judge field {key} must be boolean")
    if value["passed"] is not (value["score"] == 1):
        raise EvaluationError("Judge passed must agree with score")
    for key in ("matched_requirements", "missing_requirements"):
        items = value[key]
        if not isinstance(items, list) or any(
            not isinstance(item, str) for item in items
        ):
            raise EvaluationError(f"Judge field {key} must be a string list")
    if not isinstance(value["rationale"], str):
        raise EvaluationError("Judge rationale must be a string")
    return value


def parse_strict_judge_json(text: str) -> Dict[str, Any]:
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise EvaluationError("Judge response is not a JSON object") from exc
    return validate_judge_result(value)


def default_endpoint(provider: str, base_url: str, model: str) -> str:
    if provider == "openai-compatible":
        base = base_url or "https://api.openai.com/v1"
        if base.rstrip("/").endswith("/chat/completions"):
            return base.rstrip("/")
        return base.rstrip("/") + "/chat/completions"
    if provider == "anthropic":
        base = base_url or "https://api.anthropic.com"
        if base.rstrip("/").endswith("/v1/messages"):
            return base.rstrip("/")
        return base.rstrip("/") + "/v1/messages"
    base = base_url or "https://generativelanguage.googleapis.com/v1beta"
    if "{model}" in base:
        return base.replace("{model}", urllib.parse.quote(model, safe=""))
    if ":generateContent" in base:
        return base
    return (
        base.rstrip("/")
        + "/models/"
        + urllib.parse.quote(model, safe="")
        + ":generateContent"
    )


def judge_payload(
    provider: str, model: str, system_prompt: str, user_prompt: str
) -> Dict[str, Any]:
    if provider == "openai-compatible":
        return {
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
    if provider == "anthropic":
        return {
            "model": model,
            "max_tokens": 1200,
            "temperature": 0,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
    return {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
        },
    }


def extract_provider_text(provider: str, raw: str) -> str:
    try:
        envelope = json.loads(raw)
        if provider == "openai-compatible":
            content = envelope["choices"][0]["message"]["content"]
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict)
                )
        elif provider == "anthropic":
            return "".join(
                part.get("text", "")
                for part in envelope["content"]
                if isinstance(part, dict) and part.get("type") == "text"
            )
        else:
            return "".join(
                part.get("text", "")
                for part in envelope["candidates"][0]["content"]["parts"]
                if isinstance(part, dict)
            )
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise EvaluationError("Could not extract Judge text from provider response") from exc
    raise EvaluationError("Judge provider returned no textual response")


def direct_http_judge(
    provider: str,
    endpoint: str,
    api_key: str,
    payload: Dict[str, Any],
    timeout: Optional[int],
) -> str:
    headers = {"Content-Type": "application/json"}
    if provider == "openai-compatible":
        headers["Authorization"] = "Bearer " + api_key
    elif provider == "anthropic":
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
    else:
        headers["x-goog-api-key"] = api_key
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        if timeout is None:
            response_context = urllib.request.urlopen(request)
        else:
            response_context = urllib.request.urlopen(request, timeout=timeout)
        with response_context as response:
            return response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read(8192).decode("utf-8", errors="replace")
        raise EvaluationError(f"Judge HTTP {exc.code}: {body}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise EvaluationError(f"Judge HTTP request failed: {exc}") from exc


def timeout_kwargs(timeout: Optional[int]) -> Dict[str, int]:
    """Return a subprocess timeout only when a deadline was requested."""
    return {} if timeout is None else {"timeout": timeout}


def agent_shell_script(harness: str) -> str:
    setup = r"""
set -uo pipefail
export PATH="/opt/harness-npm/bin:/usr/local/bin:${PATH}"
export HOME=/home/agent
export XDG_CONFIG_HOME="$HOME/.config"
export XDG_CACHE_HOME="$HOME/.cache"
export CLAUDE_CONFIG_DIR="$HOME/.claude"
mkdir -p "$HOME" "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME" "$CLAUDE_CONFIG_DIR" /workspace/outputs
secret="$(printenv "$PTDBENCH_SECRET_ENV")"
if [ -z "$secret" ]; then echo "Agent API key is empty" >&2; exit 2; fi
"""
    if harness == "codex":
        command = r"""
export OPENAI_API_KEY="$secret" CODEX_API_KEY="$secret"
exec codex --search exec --json -c model_reasoning_summary=detailed \
  --skip-git-repo-check --yolo --model "$PTDBENCH_AGENT_MODEL" \
  < /task/agent_prompt.txt
"""
    elif harness == "claude":
        command = r"""
export ANTHROPIC_API_KEY="$secret" ANTHROPIC_AUTH_TOKEN="$secret"
exec claude --print --verbose --model "$PTDBENCH_AGENT_MODEL" \
  --output-format stream-json --dangerously-skip-permissions \
  < /task/agent_prompt.txt
"""
    elif harness == "gemini":
        command = r"""
export GEMINI_API_KEY="$secret" GEMINI_SANDBOX=false
exec gemini --yolo --model "$PTDBENCH_AGENT_MODEL" \
  --output-format stream-json -p "$(cat /task/agent_prompt.txt)"
"""
    else:
        command = r"""
export OPENAI_API_KEY="$secret" ANTHROPIC_API_KEY="$secret" OPENCODE_API_KEY="$secret"
exec opencode run --model "$PTDBENCH_AGENT_MODEL" --format json \
  < /task/agent_prompt.txt
"""
    return setup + command


def agent_base_url_env(harness: str) -> str:
    return {
        "codex": "OPENAI_BASE_URL",
        "claude": "ANTHROPIC_BASE_URL",
        "gemini": "GOOGLE_GEMINI_BASE_URL",
        "opencode": "OPENAI_BASE_URL",
    }[harness]


def docker_image_is_local(image: str) -> bool:
    try:
        process = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise EvaluationError(
            "Docker CLI was not found; install Docker before running this verifier"
        ) from exc
    return process.returncode == 0


def agent_image_acquisition_command(
    image: str, verifier_dir: Path
) -> Tuple[str, List[str]]:
    if image == DEFAULT_AGENT_IMAGE:
        dockerfile = verifier_dir / "agent.Dockerfile"
        return (
            "build",
            [
                "docker",
                "build",
                "--file",
                str(dockerfile),
                "--tag",
                image,
                str(verifier_dir),
            ],
        )
    return "pull", ["docker", "pull", image]


def ensure_agent_image(
    args: argparse.Namespace,
    verifier_dir: Path,
    run_dir: Path,
    secret_values: Sequence[str],
) -> None:
    image = args.agent_image
    status_path = run_dir / "agent_image_status.json"
    stdout_path = run_dir / "agent_image_stdout.txt"
    stderr_path = run_dir / "agent_image_stderr.txt"

    if docker_image_is_local(image):
        write_json(
            status_path,
            {"image": image, "source": "existing_local", "returncode": 0},
        )
        return

    action, command = agent_image_acquisition_command(image, verifier_dir)
    if action == "build":
        dockerfile = verifier_dir / "agent.Dockerfile"
        if (
            not dockerfile.is_file()
            or dockerfile.is_symlink()
            or dockerfile.resolve().parent != verifier_dir.resolve()
        ):
            raise EvaluationError(
                "default Agent image is missing and verifier/agent.Dockerfile "
                "is not a regular local file"
            )

    lock_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", image)
    lock_path = Path(tempfile.gettempdir()) / (
        "ptdbench-agent-image-" + lock_name + ".lock"
    )
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if docker_image_is_local(image):
            write_json(
                status_path,
                {
                    "image": image,
                    "source": "existing_local_after_wait",
                    "returncode": 0,
                },
            )
            return
        try:
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise EvaluationError(
                "Docker CLI was not found; install Docker before running this verifier"
            ) from exc

    safe_stdout = redact(process.stdout or "", secret_values)
    safe_stderr = redact(process.stderr or "", secret_values)
    atomic_write_text(stdout_path, safe_stdout)
    atomic_write_text(stderr_path, safe_stderr)
    source = "built_local" if action == "build" else "pulled_registry"
    write_json(
        status_path,
        {
            "image": image,
            "source": source,
            "returncode": process.returncode,
            "command": command,
        },
    )
    if process.returncode != 0:
        detail = safe_stderr.strip() or safe_stdout.strip() or "no Docker output"
        raise EvaluationError(
            f"could not {action} Agent image {image!r}: {detail[-4000:]}"
        )
    if not docker_image_is_local(image):
        raise EvaluationError(
            f"Docker reported success but Agent image {image!r} is unavailable"
        )


def build_agent_docker_command(
    args: argparse.Namespace,
    workspace: Path,
    instruction_path: Path,
    prompt_path: Path,
    container_name: str,
) -> List[str]:
    uid_gid = f"{os.getuid()}:{os.getgid()}"
    command: List[str] = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--log-driver",
        "none",
        "--network",
        args.agent_network,
        "--user",
        uid_gid,
        "--workdir",
        "/workspace",
        "--security-opt",
        "no-new-privileges",
        "--cap-drop",
        "ALL",
        "--pids-limit",
        "1024",
        "--tmpfs",
        "/home/agent:rw,nosuid,nodev",
        "-e",
        args.agent_api_key_env,
        "-e",
        "PTDBENCH_SECRET_ENV=" + args.agent_api_key_env,
        "-e",
        "PTDBENCH_AGENT_MODEL=" + args.agent_model,
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
    ]
    if args.agent_base_url:
        command.extend(
            [
                "-e",
                agent_base_url_env(args.agent_harness) + "=" + args.agent_base_url,
            ]
        )
    command.extend(
        [
            "--mount",
            f"type=bind,source={workspace},target=/workspace",
            "--mount",
            f"type=bind,source={instruction_path},target=/task/instruction.md,readonly",
            "--mount",
            f"type=bind,source={prompt_path},target=/task/agent_prompt.txt,readonly",
            "--entrypoint",
            "bash",
            args.agent_image,
            "-c",
            agent_shell_script(args.agent_harness),
        ]
    )
    return command


def run_agent_docker(
    args: argparse.Namespace,
    run_dir: Path,
    workspace: Path,
    instruction_path: Path,
    prompt_path: Path,
    secret_values: Sequence[str],
) -> int:
    container_name = "ptdb-agent-" + secrets.token_hex(8)
    command = build_agent_docker_command(
        args, workspace, instruction_path, prompt_path, container_name
    )
    stdout = ""
    stderr = ""
    return_code = 1
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            **timeout_kwargs(args.agent_timeout),
        )
        stdout, stderr, return_code = (
            process.stdout or "",
            process.stderr or "",
            process.returncode,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = (
            exc.stdout.decode("utf-8", errors="replace")
            if isinstance(exc.stdout, bytes)
            else exc.stdout or ""
        )
        stderr = (
            exc.stderr.decode("utf-8", errors="replace")
            if isinstance(exc.stderr, bytes)
            else exc.stderr or ""
        ) + "\nAgent timed out."
        return_code = 124
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    safe_stdout = redact(stdout, secret_values)
    safe_stderr = redact(stderr, secret_values)
    atomic_write_text(run_dir / "agent_stdout.txt", safe_stdout)
    atomic_write_text(run_dir / "agent_stderr.txt", safe_stderr)
    atomic_write_text(
        run_dir / "agent_trace.txt",
        "=== STDOUT ===\n"
        + safe_stdout
        + "\n=== STDERR ===\n"
        + safe_stderr,
    )
    write_json(
        run_dir / "agent_status.json",
        {"returncode": return_code, "timed_out": return_code == 124},
    )
    return return_code


def run_agent_fixture(
    args: argparse.Namespace,
    run_dir: Path,
    workspace: Path,
    instruction_path: Path,
    prompt_path: Path,
    secret_values: Sequence[str],
) -> int:
    environment = os.environ.copy()
    environment.update(
        {
            "PTDBENCH_WORKSPACE": str(workspace),
            "PTDBENCH_INSTRUCTION": str(instruction_path),
            "PTDBENCH_AGENT_PROMPT": str(prompt_path),
        }
    )
    try:
        process = subprocess.run(
            args.agent_command,
            shell=True,
            cwd=str(workspace),
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            **timeout_kwargs(args.agent_timeout),
        )
        stdout, stderr, return_code = (
            process.stdout or "",
            process.stderr or "",
            process.returncode,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + "\nAgent fixture timed out."
        return_code = 124
    safe_stdout = redact(stdout, secret_values)
    safe_stderr = redact(stderr, secret_values)
    atomic_write_text(run_dir / "agent_stdout.txt", safe_stdout)
    atomic_write_text(run_dir / "agent_stderr.txt", safe_stderr)
    atomic_write_text(
        run_dir / "agent_trace.txt",
        "=== STDOUT ===\n"
        + safe_stdout
        + "\n=== STDERR ===\n"
        + safe_stderr,
    )
    write_json(
        run_dir / "agent_status.json",
        {"returncode": return_code, "test_mode": True},
    )
    return return_code


def judge_user_prompt(
    standard_answer: str, task_context: Dict[str, Any], diagnosis: str
) -> str:
    return (
        "<STANDARD_ANSWER>\n"
        + standard_answer.rstrip()
        + "\n</STANDARD_ANSWER>\n\n<TASK_LOCATION_CONTEXT>\n"
        + json.dumps(task_context, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n</TASK_LOCATION_CONTEXT>\n\n<UNTRUSTED_AGENT_DIAGNOSIS>\n"
        + diagnosis.rstrip()
        + "\n</UNTRUSTED_AGENT_DIAGNOSIS>\n"
    )


def run_judge_attempt(
    args: argparse.Namespace,
    endpoint: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
) -> Tuple[str, str, Dict[str, Any]]:
    payload = judge_payload(
        args.judge_provider, args.judge_model, system_prompt, user_prompt
    )
    public_request = {
        "provider": args.judge_provider,
        "endpoint": endpoint,
        "payload": payload,
    }
    if args.test_mode and args.judge_command:
        process = subprocess.run(
            args.judge_command,
            shell=True,
            input=json.dumps(public_request, ensure_ascii=False),
            capture_output=True,
            text=True,
            check=False,
            **timeout_kwargs(args.judge_timeout),
        )
        if process.returncode != 0:
            raise EvaluationError(
                "Judge fixture failed: " + (process.stderr or "").strip()
            )
        raw = process.stdout or ""
        return raw, raw, public_request
    raw = direct_http_judge(
        args.judge_provider, endpoint, api_key, payload, args.judge_timeout
    )
    return raw, extract_provider_text(args.judge_provider, raw), public_request


def initialize_run_files(run_dir: Path) -> None:
    for name, content in (
        ("agent_prompt.txt", ""),
        ("agent_stdout.txt", ""),
        ("agent_stderr.txt", ""),
        ("agent_trace.txt", ""),
        ("external_agent_trace.json", "{}\n"),
        ("agent_status.json", "{}\n"),
        ("agent_image_status.json", "{}\n"),
        ("agent_image_stdout.txt", ""),
        ("agent_image_stderr.txt", ""),
        ("judge_request.json", "{}\n"),
        ("judge_raw_response.txt", ""),
        ("judge_result.json", "{}\n"),
        ("metrics.json", "{}\n"),
        ("result.json", "{}\n"),
        ("error.txt", ""),
        ("run_config.json", "{}\n"),
    ):
        atomic_write_text(run_dir / name, content)


def write_run_config(
    run_dir: Path, args: argparse.Namespace, start_time: str
) -> None:
    """Persist public run settings without ever serializing key values."""
    write_json(
        run_dir / "run_config.json",
        {
            "evaluation_mode": {
                "requested": args.evaluation_mode,
                "effective": effective_evaluation_mode(args),
            },
            "agent": {
                "model": args.agent_model,
                "harness": args.agent_harness,
                "api_key_env": args.agent_api_key_env,
                "base_url": args.agent_base_url,
                "image": args.agent_image,
            },
            "judge": {
                "model": args.judge_model,
                "provider": args.judge_provider,
                "api_key_env": args.judge_api_key_env,
                "base_url": args.judge_base_url,
            },
            "start_time": start_time,
            "ptdbench": {
                "case_id": os.environ.get("PTDBENCH_CASE_ID", ""),
                "workspace": os.environ.get("PTDBENCH_WORKSPACE", ""),
                "oracle": os.environ.get("PTDBENCH_ORACLE", ""),
                "verifier": os.environ.get("PTDBENCH_VERIFIER", ""),
                "results": os.environ.get("PTDBENCH_RESULTS", ""),
            },
            "timeouts": {
                "agent_seconds": args.agent_timeout,
                "judge_seconds": args.judge_timeout,
            },
        },
    )


def guarded_diagnosis_path(workspace: Path, require_file: bool = False) -> Path:
    root = workspace.resolve(strict=True)
    outputs = workspace / "outputs"
    diagnosis = outputs / "diagnosis.txt"
    for candidate in (outputs, diagnosis):
        if candidate.is_symlink():
            raise EvaluationError(f"staged {candidate.name} path must not be a symlink")
    resolved_outputs = outputs.resolve(strict=False)
    resolved_diagnosis = diagnosis.resolve(strict=False)
    for resolved in (resolved_outputs, resolved_diagnosis):
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise EvaluationError("diagnosis path escapes staged workspace") from exc
    if require_file:
        if not diagnosis.is_file() or diagnosis.is_symlink():
            raise EvaluationError("diagnosis.txt is missing or not a regular file")
        if diagnosis.resolve(strict=True).parent != resolved_outputs:
            raise EvaluationError("diagnosis.txt resolves outside staged outputs")
    return diagnosis


def score_documents(
    score: int, agent_returncode: int
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    metrics = {
        "diagnosis_attribution_score": score,
        "agent_returncode": agent_returncode,
    }
    result = {
        "passed": score == 1,
        "score": score,
        "metrics": {"diagnosis_attribution_score": score},
    }
    return metrics, result


def write_score(
    run_dir: Path, latest_root: Path, score: int, agent_returncode: int
) -> None:
    if type(score) is not int or score not in (0, 1):
        raise EvaluationError("Internal score must be integer 0 or 1")
    if type(agent_returncode) is not int:
        raise EvaluationError("Internal agent return code must be an integer")
    metrics, result = score_documents(score, agent_returncode)
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "result.json", result)
    if latest_root != run_dir:
        write_json(latest_root / "metrics.json", metrics)
        write_json(latest_root / "result.json", result)


def sanitize_model_name(model: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", model.strip())
    return sanitized or "model"


def make_run_id(requested: str, model: str, harness: str) -> str:
    if requested:
        if not RUN_ID_RE.fullmatch(requested) or requested in (".", ".."):
            raise EvaluationError("run-id contains unsupported characters")
        return requested
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{sanitize_model_name(model)}-{harness}-{timestamp}"


def prepare_run_paths(
    verifier_dir: Path, run_id: str
) -> Tuple[Path, Path]:
    external = os.environ.get("PTDBENCH_RESULTS", "").strip()
    if external:
        latest_root = Path(external).expanduser().resolve()
        latest_root.mkdir(parents=True, exist_ok=True)
        run_dir = latest_root / run_id
    else:
        run_dir = verifier_dir / run_id
        latest_root = run_dir
        probe = verifier_dir / (".ptd-write-probe-" + secrets.token_hex(4))
        try:
            probe.write_text("", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            raise EvaluationError(
                "verifier is not writable; set PTDBENCH_RESULTS to a writable external path"
            ) from exc
    run_dir.mkdir(parents=False, exist_ok=False)
    initialize_run_files(run_dir)
    return run_dir, latest_root


def central_agent_handoff_available() -> bool:
    if os.environ.get("PTDBENCH_STAGE", "").strip() != "verifier":
        return False
    workspace_value = os.environ.get("PTDBENCH_WORKSPACE", "").strip()
    results_value = os.environ.get("PTDBENCH_RESULTS", "").strip()
    if not workspace_value or not results_value:
        return False

    workspace = Path(workspace_value).expanduser()
    diagnosis_path = workspace / "outputs" / "diagnosis.txt"
    trace_path = Path(results_value).expanduser() / "agent_trace.json"
    try:
        diagnosis_size = diagnosis_path.stat().st_size
        trace_size = trace_path.stat().st_size
    except OSError:
        return False
    return (
        diagnosis_path.is_file()
        and not diagnosis_path.is_symlink()
        and not diagnosis_path.parent.is_symlink()
        and 0 < diagnosis_size <= MAX_DIAGNOSIS_BYTES
        and trace_path.is_file()
        and not trace_path.is_symlink()
        and 0 < trace_size <= MAX_AGENT_TRACE_BYTES
    )


def effective_evaluation_mode(args: argparse.Namespace) -> str:
    requested = getattr(args, "evaluation_mode", "auto")
    if requested != "auto":
        return requested
    if central_agent_handoff_available():
        return "judge-only"
    return "standalone"


def load_central_agent_trace() -> Dict[str, Any]:
    results_value = os.environ.get("PTDBENCH_RESULTS", "").strip()
    if not results_value:
        return {}
    trace_path = Path(results_value).expanduser().resolve() / "agent_trace.json"
    if not trace_path.is_file() or trace_path.is_symlink():
        return {}
    if trace_path.stat().st_size > MAX_AGENT_TRACE_BYTES:
        raise EvaluationError("central agent_trace.json exceeds 8 MiB")
    try:
        value = json.loads(read_text(trace_path))
    except json.JSONDecodeError as exc:
        raise EvaluationError("central agent_trace.json is invalid JSON") from exc
    if not isinstance(value, dict):
        raise EvaluationError("central agent_trace.json must contain an object")
    return value


def model_from_agent_trace(trace: Dict[str, Any]) -> str:
    result = trace.get("result")
    argv = result.get("argv") if isinstance(result, dict) else None
    if not isinstance(argv, list):
        return ""
    for index, item in enumerate(argv):
        if item == "--model" and index + 1 < len(argv):
            candidate = argv[index + 1]
            return candidate if isinstance(candidate, str) else ""
        if isinstance(item, str) and item.startswith("--model="):
            return item.split("=", 1)[1]
    return ""


def infer_central_judge_configuration(args: argparse.Namespace) -> None:
    if effective_evaluation_mode(args) != "judge-only":
        return
    trace = load_central_agent_trace()
    inferred_model = model_from_agent_trace(trace)
    if not args.judge_model and inferred_model:
        args.judge_model = inferred_model
    if not args.agent_model and inferred_model:
        args.agent_model = inferred_model

    trace_env = trace.get("env")
    if not isinstance(trace_env, dict):
        trace_env = {}
    if not args.judge_base_url:
        base_key = {
            "openai-compatible": "OPENAI_BASE_URL",
            "anthropic": "ANTHROPIC_BASE_URL",
            "gemini": "GOOGLE_GEMINI_BASE_URL",
        }[args.judge_provider]
        candidate = trace_env.get(base_key) or os.environ.get(base_key, "")
        if isinstance(candidate, str) and candidate and candidate != "<redacted>":
            args.judge_base_url = candidate
    if not args.judge_api_key_env:
        args.judge_api_key_env = {
            "openai-compatible": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GOOGLE_API_KEY",
        }[args.judge_provider]


def normalize_timeouts(args: argparse.Namespace) -> None:
    if args.agent_timeout == 0:
        args.agent_timeout = None
    if args.judge_timeout == 0:
        args.judge_timeout = None


def normalized_env_secret(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        os.environ[name] = value
    return value


def validate_configuration(args: argparse.Namespace) -> List[str]:
    normalize_timeouts(args)
    mode = effective_evaluation_mode(args)
    for value, label in (
        (args.agent_timeout, "agent timeout"),
        (args.judge_timeout, "judge timeout"),
    ):
        if value is not None and value < 0:
            raise EvaluationError(f"{label} must not be negative")
    if mode == "standalone" and (
        not args.agent_network or "\x00" in args.agent_network
    ):
        raise EvaluationError("agent network must be non-empty")
    if args.agent_command and not args.test_mode:
        raise EvaluationError("--agent-command requires --test-mode")
    if args.judge_command and not args.test_mode:
        raise EvaluationError("--judge-command requires --test-mode")
    if mode == "judge-only" and not os.environ.get(
        "PTDBENCH_WORKSPACE", ""
    ).strip():
        raise EvaluationError(
            "judge-only mode requires PTDBENCH_WORKSPACE from the central evaluator"
        )
    if args.test_mode:
        if not args.judge_command or (
            mode == "standalone" and not args.agent_command
        ):
            raise EvaluationError(
                "--test-mode requires --judge-command and standalone mode also "
                "requires --agent-command"
            )
        values = []
        names = [(args.judge_api_key_env, "judge API key env-name")]
        if mode == "standalone":
            names.insert(0, (args.agent_api_key_env, "agent API key env-name"))
        for name, label in names:
            if name:
                validate_env_name(name, label)
                value = normalized_env_secret(name)
                if value:
                    values.append(value)
        return values
    required = [
        (args.judge_model, "judge model"),
        (args.judge_api_key_env, "judge API key env-name"),
    ]
    if mode == "standalone":
        required[0:0] = [
            (args.agent_model, "agent model"),
            (args.agent_image, "agent image"),
            (args.agent_api_key_env, "agent API key env-name"),
        ]
    for value, label in required:
        if not value:
            raise EvaluationError(f"{label} is required")
    validate_env_name(args.judge_api_key_env, "judge API key env-name")
    validate_url(args.judge_base_url, "judge base URL")
    names = [args.judge_api_key_env]
    if mode == "standalone":
        validate_env_name(args.agent_api_key_env, "agent API key env-name")
        validate_url(args.agent_base_url, "agent base URL")
        names.insert(0, args.agent_api_key_env)
    normalized_secrets = {name: normalized_env_secret(name) for name in names}
    missing = [name for name, value in normalized_secrets.items() if not value]
    if missing:
        raise EvaluationError(
            "required API key environment variable is unset: "
            + ", ".join(missing)
            + "; export the key in the parent shell (do not pass only "
            "--api-key to the central PTDBench CLI)"
        )
    return [normalized_secrets[name] for name in names]


def run_evaluation(
    args: argparse.Namespace,
    verifier_dir: Path,
    run_dir: Path,
    latest_root: Path,
    secret_values: Sequence[str],
) -> int:
    mode = effective_evaluation_mode(args)
    case_root = verifier_dir.parent
    verifier_contract = os.environ.get("PTDBENCH_VERIFIER", "").strip()
    if verifier_contract:
        contract_path = Path(verifier_contract).expanduser().resolve()
        if contract_path != verifier_dir.resolve():
            raise EvaluationError(
                "PTDBENCH_VERIFIER does not identify this verifier directory"
            )

    if mode == "judge-only":
        workspace_source = Path(
            os.environ["PTDBENCH_WORKSPACE"]
        ).expanduser().resolve()
        oracle_value = os.environ.get("PTDBENCH_ORACLE", "").strip()
        oracle_dir = (
            Path(oracle_value).expanduser().resolve()
            if oracle_value
            else case_root / "oracle"
        )
    else:
        workspace_source = case_root / "workspace"
        oracle_dir = case_root / "oracle"
    instruction_path = case_root / "instruction.md"
    standard_answer_path = oracle_dir / "standard_answer.txt"
    task_yaml_path = oracle_dir / "task.yaml"
    for path, label in (
        (workspace_source, "workspace"),
        (instruction_path, "instruction.md"),
        (standard_answer_path, "standard_answer.txt"),
        (task_yaml_path, "task.yaml"),
    ):
        if not path.exists():
            raise EvaluationError(f"missing case {label}")

    if mode == "standalone" and not args.test_mode:
        ensure_agent_image(args, verifier_dir, run_dir, secret_values)

    workspace = run_dir / "workspace"
    shutil.copytree(workspace_source, workspace, symlinks=True)
    for directory, directory_names, file_names in os.walk(workspace):
        directory_path = Path(directory)
        if not directory_path.is_symlink():
            directory_path.chmod(directory_path.stat().st_mode | 0o700)
        for name in directory_names + file_names:
            path = directory_path / name
            if not path.is_symlink():
                path.chmod(path.stat().st_mode | (0o700 if path.is_dir() else 0o600))
    diagnosis_path = guarded_diagnosis_path(workspace)
    if mode == "standalone":
        if diagnosis_path.is_file():
            diagnosis_path.unlink()
        elif diagnosis_path.exists():
            raise EvaluationError("staged diagnosis path is not a regular file")

    instruction = read_text(instruction_path)
    prompt_path = run_dir / "agent_prompt.txt"
    if mode == "judge-only":
        central_trace = load_central_agent_trace()
        trace_text = json.dumps(
            central_trace, ensure_ascii=False, indent=2, sort_keys=True
        ) + "\n"
        safe_trace = redact(trace_text, secret_values)
        atomic_write_text(run_dir / "external_agent_trace.json", safe_trace)
        atomic_write_text(run_dir / "agent_trace.txt", safe_trace)
        central_prompt = central_trace.get("prompt")
        if isinstance(central_prompt, str) and central_prompt.strip():
            atomic_write_text(prompt_path, redact(central_prompt, secret_values))
        else:
            atomic_write_text(
                prompt_path,
                "Agent A was executed by the central PTDBench evaluator. "
                "See external_agent_trace.json when available.\n",
            )
        trace_result = central_trace.get("result")
        if not isinstance(trace_result, dict):
            trace_result = {}
        trace_returncode = trace_result.get("returncode")
        agent_rc = trace_returncode if type(trace_returncode) is int else 0
        atomic_write_text(
            run_dir / "agent_stdout.txt",
            redact(str(trace_result.get("stdout", "")), secret_values),
        )
        atomic_write_text(
            run_dir / "agent_stderr.txt",
            redact(str(trace_result.get("stderr", "")), secret_values),
        )
        write_json(
            run_dir / "agent_status.json",
            {
                "returncode": agent_rc,
                "source": "central_ptdbench",
                "executed_by_verifier": False,
            },
        )
        write_json(
            run_dir / "agent_image_status.json",
            {
                "source": "central_ptdbench",
                "skipped": True,
                "reason": "judge-only mode does not start a nested Agent container",
            },
        )
    else:
        agent_template = read_text(verifier_dir / "agent_prompt.txt")
        assembled_prompt = assemble_agent_prompt(agent_template, instruction)
        atomic_write_text(prompt_path, assembled_prompt)
        if args.test_mode:
            agent_rc = run_agent_fixture(
                args,
                run_dir,
                workspace,
                instruction_path,
                prompt_path,
                secret_values,
            )
        else:
            agent_rc = run_agent_docker(
                args,
                run_dir,
                workspace,
                instruction_path,
                prompt_path,
                secret_values,
            )

    try:
        diagnosis_path = guarded_diagnosis_path(workspace, require_file=True)
    except EvaluationError as exc:
        raise EvaluationError(f"Agent returned {agent_rc}; {exc}") from exc
    diagnosis_size = diagnosis_path.stat().st_size
    if diagnosis_size <= 0:
        raise EvaluationError(f"Agent returned {agent_rc}; diagnosis.txt is empty")
    if diagnosis_size > MAX_DIAGNOSIS_BYTES:
        raise EvaluationError(
            f"Agent returned {agent_rc}; diagnosis.txt exceeds 64 KiB"
        )
    diagnosis = redact(read_text(diagnosis_path), secret_values)

    standard_answer = read_text(standard_answer_path)
    task_context = parse_task_yaml_allowlist(read_text(task_yaml_path))
    system_prompt = read_text(verifier_dir / "judge_prompt.txt")
    base_user_prompt = judge_user_prompt(
        standard_answer, task_context, diagnosis
    )
    endpoint = default_endpoint(
        args.judge_provider, args.judge_base_url, args.judge_model
    )
    api_key = (
        ""
        if args.test_mode
        else os.environ.get(args.judge_api_key_env, "")
    )

    public_requests: List[Dict[str, Any]] = []
    raw_responses: List[str] = []
    last_format_error = ""
    for attempt in range(2):
        user_prompt = base_user_prompt
        if attempt:
            user_prompt += (
                "\n<FORMAT_RETRY>\nYour previous response was invalid: "
                + last_format_error
                + ". Return only the exact required JSON object.\n</FORMAT_RETRY>\n"
            )
        raw, judge_text, public_request = run_judge_attempt(
            args, endpoint, system_prompt, user_prompt, api_key
        )
        public_requests.append(public_request)
        raw_responses.append(redact(raw, secret_values))
        write_json(run_dir / "judge_request.json", {"attempts": public_requests})
        atomic_write_text(
            run_dir / "judge_raw_response.txt",
            "\n\n===== ATTEMPT =====\n".join(raw_responses),
        )
        try:
            judged = parse_strict_judge_json(judge_text)
        except EvaluationError as exc:
            last_format_error = str(exc)
            if attempt == 0:
                continue
            raise EvaluationError(
                "Judge returned invalid format twice: " + last_format_error
            ) from exc
        score = judged["score"]
        write_json(run_dir / "judge_result.json", judged)
        write_score(run_dir, latest_root, score, agent_rc)
        return 0 if score == 1 else 1
    raise EvaluationError("Judge did not produce a result")


def run_self_tests(verifier_dir: Path) -> int:
    failures: List[str] = []

    def check(condition: bool, label: str) -> None:
        if not condition:
            failures.append(label)

    parser = build_parser()
    parsed = parser.parse_args(
        [
            "--self-test",
            "--agent-harness",
            "gemini",
            "--agent-model",
            "cli-wins",
        ]
    )
    check(
        parsed.self_test
        and parsed.agent_model == "cli-wins"
        and parsed.evaluation_mode == "auto",
        "CLI parser",
    )
    default_timeouts = parser.parse_args([])
    normalize_timeouts(default_timeouts)
    check(
        default_timeouts.agent_timeout is None
        and default_timeouts.judge_timeout is None
        and timeout_kwargs(None) == {},
        "unset and zero timeouts are unlimited",
    )
    check(
        default_timeouts.agent_image == DEFAULT_AGENT_IMAGE,
        "portable default Agent image",
    )
    acquisition_action, acquisition_command = agent_image_acquisition_command(
        DEFAULT_AGENT_IMAGE, verifier_dir
    )
    check(
        acquisition_action == "build"
        and acquisition_command[:2] == ["docker", "build"]
        and str(verifier_dir / "agent.Dockerfile") in acquisition_command,
        "default image auto-build command",
    )
    custom_action, custom_command = agent_image_acquisition_command(
        "ghcr.io/example/agent:v1", verifier_dir
    )
    check(
        custom_action == "pull"
        and custom_command == ["docker", "pull", "ghcr.io/example/agent:v1"],
        "custom image auto-pull command",
    )
    central_env_names = (
        "PTDBENCH_STAGE",
        "PTDBENCH_WORKSPACE",
        "PTDBENCH_RESULTS",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
    )
    previous_central_env = {
        name: os.environ.get(name) for name in central_env_names
    }
    with tempfile.TemporaryDirectory() as temporary:
        central_root = Path(temporary)
        central_workspace = central_root / "workspace"
        central_results = central_root / "results"
        central_workspace.mkdir()
        central_results.mkdir()
        write_json(
            central_results / "agent_trace.json",
            {
                "kind": "codex",
                "env": {"OPENAI_BASE_URL": "https://example.invalid/v1"},
                "result": {
                    "argv": ["codex", "exec", "--model", "central-model"],
                    "returncode": 0,
                },
            },
        )
        os.environ.update(
            {
                "PTDBENCH_STAGE": "verifier",
                "PTDBENCH_WORKSPACE": str(central_workspace),
                "PTDBENCH_RESULTS": str(central_results),
                "OPENAI_API_KEY": " central-secret ",
            }
        )
        try:
            central_args = parser.parse_args(
                ["--test-mode", "--judge-command", "true"]
            )
            check(
                effective_evaluation_mode(central_args) == "standalone",
                "central PTDBench without diagnosis defaults to standalone",
            )
            diagnosis_dir = central_workspace / "outputs"
            diagnosis_dir.mkdir()
            atomic_write_text(
                diagnosis_dir / "diagnosis.txt",
                "VERDICT: healthy\nPRIMARY_ATTRIBUTION: none\n",
            )
            central_args = parser.parse_args(
                ["--test-mode", "--judge-command", "true"]
            )
            infer_central_judge_configuration(central_args)
            central_secrets = validate_configuration(central_args)
            check(
                effective_evaluation_mode(central_args) == "judge-only"
                and central_args.agent_model == "central-model"
                and central_args.judge_model == "central-model"
                and central_args.judge_api_key_env == "OPENAI_API_KEY"
                and central_args.judge_base_url
                == "https://example.invalid/v1"
                and central_secrets == ["central-secret"],
                "central PTDBench judge-only inference",
            )
        finally:
            for name, value in previous_central_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
    positive_timeouts = parser.parse_args(
        ["--agent-timeout", "9", "--judge-timeout", "11"]
    )
    normalize_timeouts(positive_timeouts)
    check(
        positive_timeouts.agent_timeout == 9
        and positive_timeouts.judge_timeout == 11
        and timeout_kwargs(9) == {"timeout": 9},
        "positive timeout compatibility",
    )
    negative_timeouts = parser.parse_args(
        [
            "--test-mode",
            "--evaluation-mode",
            "standalone",
            "--agent-command",
            "true",
            "--judge-command",
            "true",
            "--agent-timeout",
            "-1",
        ]
    )
    normalize_timeouts(negative_timeouts)
    try:
        validate_configuration(negative_timeouts)
        rejected = False
    except EvaluationError:
        rejected = True
    check(rejected, "negative timeout rejected")

    generated_run_id = make_run_id("", "deepseek/v4 pro", "claude")
    check(
        re.fullmatch(
            r"deepseek-v4-pro-claude-\d{8}T\d{12}Z", generated_run_id
        )
        is not None,
        "sanitized model and high-precision UTC run id",
    )
    check(
        make_run_id("manual.valid-id", "ignored", "codex")
        == "manual.valid-id",
        "validated run-id override",
    )

    sample_yaml = """
task_id: sample
environment: Demo
hollow_file: pkg/target.py
hollow_files:
- pkg/target.py
hollow_symbols:
- target_fn
hollow_hparams:
- LR
baseline_path: hidden/baseline/pkg/target.py
primary_cause: forbidden
effective_hparams:
  hidden: true
source_path: /secret/case
scoring:
  diagnosis_scored: true
baselines:
  pkg/target.py: deadbeefhash
metadata_hash: forbidden
oracle:
  primary_cause: forbidden nested answer
  buggy_keys:
  - attention_dropout
  buggy_args:
    attention_dropout: 0.5
  golden_args:
    attention_dropout: 0.0
  effective_hparams:
    attention_dropout: 0.5
"""
    context = parse_task_yaml_allowlist(sample_yaml)
    serialized_context = json.dumps(context, sort_keys=True).lower()
    check(
        set(context)
        == {
            "task_id",
            "environment",
            "hollow_file",
            "hollow_files",
            "hollow_symbols",
            "hollow_hparams",
            "baseline_path",
            "oracle",
        },
        "task top-level allowlist keys",
    )
    check(
        "hash" not in serialized_context
        and "deadbeef" not in serialized_context
        and context.get("baseline_path") == "pkg/target.py",
        "task allowlist excludes hashes and sanitizes baseline",
    )
    check(
        context.get("oracle")
        == {
            "buggy_keys": ["attention_dropout"],
            "buggy_args": {"attention_dropout": "0.5"},
            "golden_args": {"attention_dropout": "0.0"},
        },
        "RLVE nested oracle allowlist",
    )
    check(
        all(
            forbidden not in serialized_context
            for forbidden in (
                "primary_cause",
                "effective_hparams",
                "source_path",
                "scoring",
                "baselines",
            )
        ),
        "forbidden task fields excluded",
    )
    rejected = False
    try:
        reject_forbidden_context_keys(
            {"safe": [{"nested": {"artifact_checksum": "secret"}}]}
        )
    except EvaluationError:
        rejected = True
    check(rejected, "recursive forbidden hash key rejection")

    valid = {
        "score": 1,
        "passed": True,
        "verdict_match": True,
        "attribution_match": True,
        "specificity_match": True,
        "matched_requirements": ["one"],
        "missing_requirements": [],
        "rationale": "ok",
    }
    check(validate_judge_result(dict(valid))["score"] == 1, "valid Judge schema")
    for bad_score in (0.5, True, "1"):
        invalid = dict(valid)
        invalid["score"] = bad_score
        rejected = False
        try:
            validate_judge_result(invalid)
        except EvaluationError:
            rejected = True
        check(rejected, f"reject Judge score {bad_score!r}")
    extra = dict(valid)
    extra["unexpected"] = True
    try:
        validate_judge_result(extra)
        rejected = False
    except EvaluationError:
        rejected = True
    check(rejected, "strict Judge schema rejects extra keys")

    template = read_text(verifier_dir / "agent_prompt.txt")
    prompt = assemble_agent_prompt(template, "Inspect /workspace only.")
    check("Inspect /workspace only." in prompt, "prompt embeds instruction")
    check("oracle" not in prompt.lower(), "Agent prompt excludes oracle")
    check("/cases/" not in prompt.lower(), "Agent prompt excludes case path")
    check(
        "benchmark contract overrides" in prompt
        and "abnormal" in prompt
        and "incomplete" in prompt
        and "only scored deliverable" in prompt,
        "prompt precedence and diagnosis-only contract",
    )

    expected_cli = {
        "codex": "exec codex ",
        "claude": "exec claude ",
        "gemini": "exec gemini ",
        "opencode": "exec opencode ",
    }
    for harness, token in expected_cli.items():
        script = agent_shell_script(harness)
        check(token in script, f"{harness} harness CLI")
        check(
            'export PATH="/opt/harness-npm/bin:/usr/local/bin:${PATH}"' in script,
            f"{harness} harness PATH",
        )
        check(
            "export HOME=/home/agent" in script
            and "/workspace/.ptd-home" not in script,
            f"{harness} ephemeral HOME",
        )

    openai_payload = judge_payload("openai-compatible", "m", "s", "u")
    anthropic_payload = judge_payload("anthropic", "m", "s", "u")
    gemini_payload = judge_payload("gemini", "m", "s", "u")
    check(openai_payload["messages"][0]["role"] == "system", "OpenAI payload")
    check(anthropic_payload["system"] == "s", "Anthropic payload")
    check(
        gemini_payload["systemInstruction"]["parts"][0]["text"] == "s",
        "Gemini payload",
    )
    provider_responses = {
        "openai-compatible": json.dumps(
            {"choices": [{"message": {"content": "openai-text"}}]}
        ),
        "anthropic": json.dumps(
            {"content": [{"type": "text", "text": "anthropic-text"}]}
        ),
        "gemini": json.dumps(
            {"candidates": [{"content": {"parts": [{"text": "gemini-text"}]}}]}
        ),
    }
    for provider, raw in provider_responses.items():
        check(
            extract_provider_text(provider, raw).endswith("-text"),
            f"{provider} response extraction",
        )

    secret_name = "PTDBENCH_SELF_TEST_AGENT_KEY"
    judge_secret_name = "PTDBENCH_SELF_TEST_JUDGE_KEY"
    previous_agent_secret = os.environ.get(secret_name)
    previous_judge_secret = os.environ.get(judge_secret_name)
    os.environ[secret_name] = "  agent-secret \r\n"
    os.environ[judge_secret_name] = "\tjudge-secret  "
    try:
        docker_args = parser.parse_args([])
        docker_args.agent_harness = "claude"
        docker_args.agent_model = "model"
        docker_args.agent_image = "final-rl-agent:latest"
        docker_args.agent_api_key_env = secret_name
        docker_args.agent_base_url = ""
        docker_args.agent_network = "none"
        command = build_agent_docker_command(
            docker_args,
            Path("/tmp/staged-workspace"),
            Path("/tmp/instruction.md"),
            Path("/tmp/agent_prompt.txt"),
            "self-test-container",
        )
        mounts = [
            command[index + 1]
            for index, item in enumerate(command[:-1])
            if item == "--mount"
        ]
        check(
            command[command.index("--tmpfs") + 1]
            == "/home/agent:rw,nosuid,nodev",
            "Docker ephemeral home tmpfs",
        )
        check(
            len(mounts) == 3
            and {mount.split(",target=", 1)[1].split(",", 1)[0] for mount in mounts}
            == {"/workspace", "/task/instruction.md", "/task/agent_prompt.txt"},
            "Docker mount allowlist",
        )
        check(
            os.environ[secret_name] not in "\n".join(command),
            "secret value absent from Docker argv",
        )

        test_args = parser.parse_args(
            [
                "--test-mode",
                "--evaluation-mode",
                "standalone",
                "--agent-command",
                "true",
                "--judge-command",
                "true",
                "--agent-api-key-env",
                secret_name,
                "--judge-api-key-env",
                judge_secret_name,
            ]
        )
        collected = validate_configuration(test_args)
        check(
            os.environ[secret_name] == "agent-secret"
            and os.environ[judge_secret_name] == "judge-secret"
            and collected == ["agent-secret", "judge-secret"],
            "API keys are stripped and normalized in parent environment",
        )
        raw_secret = 'sec ret/+"line\nnext'
        encoded = " ".join(
            (
                raw_secret,
                urllib.parse.quote(raw_secret, safe=""),
                urllib.parse.quote_plus(raw_secret, safe=""),
                json.dumps(raw_secret, ensure_ascii=False)[1:-1],
            )
        )
        redacted = redact(encoded, [raw_secret])
        check(raw_secret not in redacted and "sec%20ret" not in redacted, "secret variants")
    finally:
        if previous_agent_secret is None:
            os.environ.pop(secret_name, None)
        else:
            os.environ[secret_name] = previous_agent_secret
        if previous_judge_secret is None:
            os.environ.pop(judge_secret_name, None)
        else:
            os.environ[judge_secret_name] = previous_judge_secret

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        workspace = root / "workspace"
        workspace.mkdir()
        outputs = workspace / "outputs"
        outputs.mkdir()
        diagnosis = guarded_diagnosis_path(workspace)
        diagnosis.write_text("VERDICT: healthy\n", encoding="utf-8")
        check(
            guarded_diagnosis_path(workspace, require_file=True) == diagnosis,
            "regular diagnosis path guard",
        )
        diagnosis.unlink()
        outside = root / "outside"
        outside.mkdir()
        outputs.rmdir()
        outputs.symlink_to(outside, target_is_directory=True)
        try:
            guarded_diagnosis_path(workspace)
            rejected = False
        except EvaluationError:
            rejected = True
        check(rejected, "symlink parent path guard")
        outputs.unlink()
        outputs.mkdir()
        diagnosis.symlink_to(outside / "diagnosis.txt")
        try:
            guarded_diagnosis_path(workspace)
            rejected = False
        except EvaluationError:
            rejected = True
        check(rejected, "diagnosis symlink path guard")

    previous_results = os.environ.get("PTDBENCH_RESULTS")
    with tempfile.TemporaryDirectory() as temporary:
        external = Path(temporary) / "external-results"
        os.environ["PTDBENCH_RESULTS"] = str(external)
        try:
            run_dir, latest_root = prepare_run_paths(verifier_dir, "self-test-run")
            config_args = parser.parse_args(
                [
                    "--agent-harness",
                    "claude",
                    "--agent-model",
                    "deepseek/v4",
                    "--agent-api-key-env",
                    secret_name,
                    "--judge-model",
                    "judge-model",
                    "--judge-api-key-env",
                    judge_secret_name,
                ]
            )
            normalize_timeouts(config_args)
            write_run_config(
                run_dir, config_args, "2026-08-08T15:30:00.123456+00:00"
            )
            config_text = read_text(run_dir / "run_config.json")
            check(
                latest_root == external.resolve()
                and run_dir.parent == external.resolve()
                and verifier_dir not in run_dir.parents,
                "external results path",
            )
            check(
                '"agent_seconds": null' in config_text
                and '"judge_seconds": null' in config_text
                and secret_name in config_text
                and judge_secret_name in config_text
                and "agent-secret" not in config_text
                and "judge-secret" not in config_text,
                "run_config records unlimited timeouts and no secrets",
            )
            write_score(run_dir, latest_root, 1, 23)
            check(
                (external / "metrics.json").is_file()
                and (external / "result.json").is_file()
                and json.loads(read_text(external / "metrics.json"))[
                    "agent_returncode"
                ]
                == 23,
                "external PTDBench result contract",
            )
        finally:
            if previous_results is None:
                os.environ.pop("PTDBENCH_RESULTS", None)
            else:
                os.environ["PTDBENCH_RESULTS"] = previous_results

    with tempfile.TemporaryDirectory() as temporary:
        os.environ.pop("PTDBENCH_RESULTS", None)
        try:
            standalone_verifier = Path(temporary) / "verifier"
            standalone_verifier.mkdir()
            run_id = make_run_id("", "deepseek/v4 pro", "claude")
            run_dir, latest_root = prepare_run_paths(
                standalone_verifier, run_id
            )
            write_score(run_dir, latest_root, 0, -1)
            check(
                run_dir.parent == standalone_verifier
                and latest_root == run_dir
                and not (standalone_verifier / "runs").exists()
                and not (standalone_verifier / "metrics.json").exists()
                and not (standalone_verifier / "result.json").exists(),
                "standalone run is direct under verifier with no top-level score",
            )
        finally:
            if previous_results is not None:
                os.environ["PTDBENCH_RESULTS"] = previous_results

    metrics, result = score_documents(1, 17)
    check(
        metrics
        == {"diagnosis_attribution_score": 1, "agent_returncode": 17}
        and result
        == {
            "passed": True,
            "score": 1,
            "metrics": {"diagnosis_attribution_score": 1},
        },
        "metrics include Agent rc and result keeps PTDBench schema",
    )

    with tempfile.TemporaryDirectory() as temporary:
        synthetic_case = Path(temporary) / "case"
        synthetic_verifier = synthetic_case / "verifier"
        (synthetic_case / "workspace").mkdir(parents=True)
        (synthetic_case / "oracle").mkdir()
        synthetic_verifier.mkdir()
        (synthetic_case / "instruction.md").write_text(
            "Diagnose the staged run.\n", encoding="utf-8"
        )
        (synthetic_case / "oracle" / "standard_answer.txt").write_text(
            "VERDICT: healthy\n", encoding="utf-8"
        )
        (synthetic_case / "oracle" / "task.yaml").write_text(
            "task_id: synthetic\n", encoding="utf-8"
        )
        shutil.copy2(verifier_dir / "agent_prompt.txt", synthetic_verifier)
        shutil.copy2(verifier_dir / "judge_prompt.txt", synthetic_verifier)
        agent_code = (
            "import os; from pathlib import Path; "
            "p=Path(os.environ['PTDBENCH_WORKSPACE'])/'outputs'; "
            "p.mkdir(parents=True, exist_ok=True); "
            "(p/'diagnosis.txt').write_text("
            "'VERDICT: healthy\\nPRIMARY_ATTRIBUTION: none\\nEVIDENCE:\\nfixture\\n"
            "CAUSAL_EXPLANATION:\\nfixture\\n', encoding='utf-8'); "
            "raise SystemExit(7)"
        )
        judge_code = "print(" + repr(json.dumps(valid)) + ")"
        fixture_args = parser.parse_args(
            [
                "--test-mode",
                "--agent-command",
                shlex.quote(sys.executable) + " -c " + shlex.quote(agent_code),
                "--judge-command",
                shlex.quote(sys.executable) + " -c " + shlex.quote(judge_code),
                "--judge-model",
                "fixture",
            ]
        )
        normalize_timeouts(fixture_args)
        fixture_run = Path(temporary) / "results" / "fixture"
        fixture_run.mkdir(parents=True)
        initialize_run_files(fixture_run)
        write_run_config(
            fixture_run,
            fixture_args,
            "2026-08-08T15:30:00.123456+00:00",
        )
        fixture_rc = run_evaluation(
            fixture_args,
            synthetic_verifier,
            fixture_run,
            fixture_run,
            [],
        )
        fixture_metrics = json.loads(read_text(fixture_run / "metrics.json"))
        expected_artifacts = {
            "workspace",
            "agent_prompt.txt",
            "agent_stdout.txt",
            "agent_stderr.txt",
            "agent_trace.txt",
            "agent_status.json",
            "judge_request.json",
            "judge_raw_response.txt",
            "judge_result.json",
            "metrics.json",
            "result.json",
            "error.txt",
            "run_config.json",
        }
        check(
            fixture_rc == 0
            and fixture_metrics.get("agent_returncode") == 7
            and fixture_metrics.get("diagnosis_attribution_score") == 1,
            "nonzero Agent rc with legal diagnosis still reaches Judge",
        )
        check(
            expected_artifacts.issubset(
                {path.name for path in fixture_run.iterdir()}
            ),
            "complete process artifacts initialized in run folder",
        )

        central_workspace = Path(temporary) / "central-workspace"
        central_outputs = central_workspace / "outputs"
        central_outputs.mkdir(parents=True)
        central_diagnosis = (
            "VERDICT: healthy\nPRIMARY_ATTRIBUTION: none\nEVIDENCE:\n"
            "central fixture\nCAUSAL_EXPLANATION:\ncentral fixture\n"
        )
        (central_outputs / "diagnosis.txt").write_text(
            central_diagnosis, encoding="utf-8"
        )
        central_results = Path(temporary) / "central-results"
        central_results.mkdir()
        write_json(
            central_results / "agent_trace.json",
            {
                "kind": "shell",
                "prompt": "central Agent prompt",
                "result": {
                    "argv": ["sh", "-lc", "fixture"],
                    "returncode": 9,
                    "stdout": "central stdout",
                    "stderr": "",
                },
            },
        )
        contract_env = {
            "PTDBENCH_STAGE": "verifier",
            "PTDBENCH_WORKSPACE": str(central_workspace),
            "PTDBENCH_ORACLE": str(synthetic_case / "oracle"),
            "PTDBENCH_VERIFIER": str(synthetic_verifier),
            "PTDBENCH_RESULTS": str(central_results),
            "PTDBENCH_CASE_ID": "synthetic",
        }
        previous_contract_env = {
            name: os.environ.get(name) for name in contract_env
        }
        os.environ.update(contract_env)
        try:
            central_fixture_args = parser.parse_args(
                [
                    "--test-mode",
                    "--evaluation-mode",
                    "judge-only",
                    "--judge-command",
                    shlex.quote(sys.executable)
                    + " -c "
                    + shlex.quote(judge_code),
                    "--judge-model",
                    "fixture",
                ]
            )
            central_fixture_secrets = validate_configuration(
                central_fixture_args
            )
            central_run = central_results / "central-fixture"
            central_run.mkdir()
            initialize_run_files(central_run)
            write_run_config(
                central_run,
                central_fixture_args,
                "2026-08-08T15:30:00.123456+00:00",
            )
            central_rc = run_evaluation(
                central_fixture_args,
                synthetic_verifier,
                central_run,
                central_results,
                central_fixture_secrets,
            )
            central_metrics = json.loads(
                read_text(central_run / "metrics.json")
            )
            central_image_status = json.loads(
                read_text(central_run / "agent_image_status.json")
            )
            check(
                central_rc == 0
                and central_metrics.get("agent_returncode") == 9
                and central_metrics.get("diagnosis_attribution_score") == 1
                and read_text(
                    central_run / "workspace" / "outputs" / "diagnosis.txt"
                )
                == central_diagnosis
                and central_image_status.get("skipped") is True
                and (central_run / "external_agent_trace.json").is_file()
                and (central_results / "result.json").is_file(),
                "central staged workspace judge-only contract",
            )
        finally:
            for name, value in previous_contract_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    if failures:
        for failure in failures:
            print("FAIL:", failure, file=sys.stderr)
        return 1
    print(
        "SELF-TEST PASS: allowlist, harnesses, Docker, providers, paths, prompts, and schemas"
    )
    return 0


def emergency_failure_artifacts(
    verifier_dir: Path,
    args: Optional[argparse.Namespace],
    run_dir: Optional[Path],
    latest_root: Optional[Path],
    message: str,
    secret_values: Sequence[str],
) -> None:
    safe_message = redact(message, secret_values)
    try:
        if run_dir is None or latest_root is None:
            model = getattr(args, "agent_model", "") if args is not None else ""
            harness = getattr(args, "agent_harness", "codex") if args is not None else "codex"
            run_id = make_run_id("", model, harness)
            run_dir, latest_root = prepare_run_paths(verifier_dir, run_id)
            if args is not None:
                normalize_timeouts(args)
                write_run_config(
                    run_dir,
                    args,
                    dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds"),
                )
        agent_returncode = -1
        status_path = run_dir / "agent_status.json"
        if status_path.is_file() and not status_path.is_symlink():
            try:
                status = json.loads(read_text(status_path))
                if type(status.get("returncode")) is int:
                    agent_returncode = status["returncode"]
            except (json.JSONDecodeError, OSError):
                pass
        atomic_write_text(run_dir / "error.txt", safe_message + "\n")
        write_score(run_dir, latest_root, 0, agent_returncode)
    except Exception as artifact_error:
        print(
            "Could not write failure artifacts: "
            + redact(str(artifact_error), secret_values),
            file=sys.stderr,
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    verifier_dir = Path(__file__).resolve().parent
    args: Optional[argparse.Namespace] = None
    run_dir: Optional[Path] = None
    latest_root: Optional[Path] = None
    secret_values: List[str] = []
    try:
        args = build_parser().parse_args(argv)
        if args.self_test:
            return run_self_tests(verifier_dir)
        infer_central_judge_configuration(args)
        start_time = dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds")
        run_id = make_run_id(args.run_id, args.agent_model, args.agent_harness)
        run_dir, latest_root = prepare_run_paths(verifier_dir, run_id)
        normalize_timeouts(args)
        write_run_config(run_dir, args, start_time)
        secret_values = validate_configuration(args)
        return run_evaluation(
            args,
            verifier_dir,
            run_dir,
            latest_root,
            secret_values,
        )
    except (EvaluationError, OSError, subprocess.SubprocessError) as exc:
        emergency_failure_artifacts(
            verifier_dir,
            args,
            run_dir,
            latest_root,
            str(exc),
            secret_values,
        )
        print("Evaluation failed: " + redact(str(exc), secret_values), file=sys.stderr)
        return 1
    except Exception as exc:
        emergency_failure_artifacts(
            verifier_dir,
            args,
            run_dir,
            latest_root,
            "unexpected error: " + str(exc),
            secret_values,
        )
        print(
            "Evaluation failed: unexpected error: "
            + redact(str(exc), secret_values),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
