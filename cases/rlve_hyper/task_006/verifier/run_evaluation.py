#!/usr/bin/env python3
"""Self-contained Judge-B-only PTDBench verifier (standard library only)."""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import os
import re
import secrets
import shlex
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Sequence, Tuple


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
    if value is None or not value.strip():
        return default
    try:
        return int(value.strip())
    except ValueError as exc:
        raise EvaluationError(f"{name} must be an integer") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = SafeArgumentParser(
        description=(
            "Grade the diagnosis produced by central PTDBench Agent A with an "
            "independent LLM Judge B."
        )
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
        help="Name of the environment variable containing Judge B's API key.",
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
    parser.add_argument("--judge-command", default="")
    return parser


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + secrets.token_hex(4))
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(str(temporary), str(path))
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


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
    """Extract only benchmark-approved fields from the simple task YAML."""
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
            raw = response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read(8192).decode("utf-8", errors="replace")
        raise EvaluationError(f"Judge HTTP {exc.code}: {body}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise EvaluationError(f"Judge HTTP request failed: {exc}") from exc
    return raw.decode("utf-8", errors="replace")


def timeout_kwargs(timeout: Optional[int]) -> Dict[str, int]:
    return {} if timeout is None else {"timeout": timeout}


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
    if args.test_mode:
        if not args.judge_command:
            raise EvaluationError("--test-mode requires --judge-command")
        try:
            process = subprocess.run(
                args.judge_command,
                shell=True,
                input=json.dumps(public_request, ensure_ascii=False),
                capture_output=True,
                text=True,
                check=False,
                **timeout_kwargs(args.judge_timeout),
            )
        except subprocess.TimeoutExpired as exc:
            raise EvaluationError("Judge fixture timed out") from exc
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


def required_directory_from_env(name: str) -> Path:
    value = os.environ.get(name, "").strip()
    if not value:
        raise EvaluationError(f"{name} is required")
    path = Path(value).expanduser()
    if not path.is_dir():
        raise EvaluationError(f"{name} must identify an existing directory")
    if path.is_symlink():
        raise EvaluationError(f"{name} must not identify a symlink")
    return path.resolve(strict=True)


def central_contract_paths(verifier_dir: Path) -> Tuple[Path, Path, Path]:
    if os.environ.get("PTDBENCH_STAGE", "").strip() != "verifier":
        raise EvaluationError("PTDBENCH_STAGE must be verifier")
    workspace = required_directory_from_env("PTDBENCH_WORKSPACE")
    oracle = required_directory_from_env("PTDBENCH_ORACLE")
    contract_verifier = required_directory_from_env("PTDBENCH_VERIFIER")
    if contract_verifier != verifier_dir.resolve(strict=True):
        raise EvaluationError(
            "PTDBENCH_VERIFIER does not identify this verifier directory"
        )
    results_value = os.environ.get("PTDBENCH_RESULTS", "").strip()
    if not results_value:
        raise EvaluationError("PTDBENCH_RESULTS is required")
    results = Path(results_value).expanduser()
    results.mkdir(parents=True, exist_ok=True)
    if results.is_symlink():
        raise EvaluationError("PTDBENCH_RESULTS must not identify a symlink")
    return workspace, oracle, results.resolve(strict=True)


def regular_file_below(root: Path, relative: str, label: str) -> Path:
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise EvaluationError(f"{label} is missing or not a regular file")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise EvaluationError(f"{label} escapes its approved directory") from exc
    return resolved


def read_required_text(path: Path, label: str) -> str:
    size = path.stat().st_size
    if size <= 0:
        raise EvaluationError(f"{label} is empty")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise EvaluationError(f"{label} is not UTF-8 text") from exc


def load_diagnosis(workspace: Path) -> str:
    outputs = workspace / "outputs"
    if outputs.is_symlink():
        raise EvaluationError("staged outputs path must not be a symlink")
    path = regular_file_below(
        workspace, "outputs/diagnosis.txt", "diagnosis.txt"
    )
    return read_required_text(path, "diagnosis.txt")


def load_agent_trace(results: Path) -> Tuple[Dict[str, Any], str]:
    path = regular_file_below(results, "agent_trace.json", "agent_trace.json")
    text = read_required_text(path, "agent_trace.json")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EvaluationError("agent_trace.json is invalid JSON") from exc
    if not isinstance(value, dict):
        raise EvaluationError("agent_trace.json must contain an object")
    return value, text


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


def infer_judge_configuration(
    args: argparse.Namespace, trace: Dict[str, Any]
) -> None:
    if not args.judge_model:
        args.judge_model = model_from_agent_trace(trace)
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


def sanitize_model_name(model: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", model.strip())
    return sanitized or "judge"


def make_run_id(requested: str, model: str) -> str:
    if requested:
        if not RUN_ID_RE.fullmatch(requested) or requested in (".", ".."):
            raise EvaluationError("run-id contains unsupported characters")
        return requested
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"judge-{sanitize_model_name(model)}-{timestamp}"


def initialize_run_files(run_dir: Path) -> None:
    for name, content in (
        ("agent_trace.json", "{}\n"),
        ("judge_request.json", "{}\n"),
        ("judge_raw_response.txt", ""),
        ("judge_result.json", "{}\n"),
        ("run_config.json", "{}\n"),
        ("metrics.json", "{}\n"),
        ("result.json", "{}\n"),
        ("error.txt", ""),
    ):
        atomic_write_text(run_dir / name, content)


def prepare_run(results: Path, args: argparse.Namespace) -> Path:
    run_dir = results / make_run_id(args.run_id, args.judge_model)
    try:
        run_dir.mkdir(parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise EvaluationError("run-id already exists") from exc
    initialize_run_files(run_dir)
    return run_dir


def score_documents(score: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if type(score) is not int or score not in (0, 1):
        raise EvaluationError("Internal score must be integer 0 or 1")
    metrics = {"diagnosis_attribution_score": score}
    result = {
        "passed": score == 1,
        "score": score,
        "metrics": {"diagnosis_attribution_score": score},
    }
    return metrics, result


def write_score(run_dir: Path, results: Path, score: int) -> None:
    metrics, result = score_documents(score)
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "result.json", result)
    write_json(results / "metrics.json", metrics)
    write_json(results / "result.json", result)


def write_run_config(
    run_dir: Path,
    args: argparse.Namespace,
    workspace: Path,
    oracle: Path,
    results: Path,
) -> None:
    write_json(
        run_dir / "run_config.json",
        {
            "mode": "judge-only",
            "judge": {
                "provider": args.judge_provider,
                "model": args.judge_model,
                "api_key_env": args.judge_api_key_env,
                "base_url": args.judge_base_url,
                "timeout_seconds": args.judge_timeout,
            },
            "ptdbench": {
                "case_id": os.environ.get("PTDBENCH_CASE_ID", ""),
                "workspace": str(workspace),
                "oracle": str(oracle),
                "verifier": os.environ.get("PTDBENCH_VERIFIER", ""),
                "results": str(results),
            },
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "test_mode": args.test_mode,
        },
    )


def normalized_secret(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        os.environ[name] = value
    return value


def run_evaluation(
    args: argparse.Namespace,
    verifier_dir: Path,
    workspace: Path,
    oracle: Path,
    results: Path,
    run_dir: Path,
) -> int:
    trace, trace_text = load_agent_trace(results)
    infer_judge_configuration(args, trace)
    if not args.judge_model:
        raise EvaluationError(
            "Judge model is missing; set PTDBENCH_JUDGE_MODEL or central --model"
        )
    if args.judge_timeout < 0:
        raise EvaluationError("judge-timeout must be non-negative")
    if args.judge_timeout == 0:
        args.judge_timeout = None
    validate_url(args.judge_base_url, "Judge base URL")
    validate_env_name(args.judge_api_key_env, "Judge API key variable")
    api_key = ""
    if not args.test_mode:
        api_key = normalized_secret(args.judge_api_key_env)
        if not api_key:
            raise EvaluationError(
                f"Judge API key variable {args.judge_api_key_env} is empty"
            )
    secrets_to_redact = [api_key]
    write_run_config(run_dir, args, workspace, oracle, results)
    atomic_write_text(
        run_dir / "agent_trace.json",
        redact(trace_text, secrets_to_redact),
    )

    diagnosis = load_diagnosis(workspace)
    standard_path = regular_file_below(
        oracle, "standard_answer.txt", "standard_answer.txt"
    )
    task_yaml_path = regular_file_below(oracle, "task.yaml", "task.yaml")
    judge_prompt_path = regular_file_below(
        verifier_dir, "judge_prompt.txt", "judge_prompt.txt"
    )
    standard_answer = read_required_text(standard_path, "standard_answer.txt")
    task_context = parse_task_yaml_allowlist(
        read_required_text(task_yaml_path, "task.yaml")
    )
    system_prompt = read_required_text(judge_prompt_path, "judge_prompt.txt")
    user_prompt = judge_user_prompt(standard_answer, task_context, diagnosis)
    endpoint = default_endpoint(
        args.judge_provider, args.judge_base_url, args.judge_model
    )

    requests: List[Dict[str, Any]] = []
    raw_responses: List[str] = []
    judge_result: Optional[Dict[str, Any]] = None
    parse_error = ""
    for attempt in range(2):
        attempt_system = system_prompt
        if attempt:
            attempt_system += (
                "\nYour previous response violated the required JSON schema. "
                "Return only one valid JSON object with exactly the specified fields."
            )
        raw, judge_text, public_request = run_judge_attempt(
            args, endpoint, attempt_system, user_prompt, api_key
        )
        requests.append(public_request)
        raw_responses.append(raw)
        try:
            judge_result = parse_strict_judge_json(judge_text)
            break
        except EvaluationError as exc:
            parse_error = str(exc)
    write_json(
        run_dir / "judge_request.json",
        json.loads(
            redact(
                json.dumps({"attempts": requests}, ensure_ascii=False),
                secrets_to_redact,
            )
        ),
    )
    atomic_write_text(
        run_dir / "judge_raw_response.txt",
        redact(
            "\n\n===== RETRY =====\n\n".join(raw_responses),
            secrets_to_redact,
        ),
    )
    if judge_result is None:
        raise EvaluationError(
            "Judge returned invalid JSON after one retry: " + parse_error
        )
    write_json(run_dir / "judge_result.json", judge_result)
    atomic_write_text(run_dir / "error.txt", "")
    write_score(run_dir, results, judge_result["score"])
    return 0 if judge_result["score"] == 1 else 1


def fixed_judge_result(score: int) -> Dict[str, Any]:
    return {
        "score": score,
        "passed": score == 1,
        "verdict_match": score == 1,
        "attribution_match": score == 1,
        "specificity_match": score == 1,
        "matched_requirements": ["fixture"] if score == 1 else [],
        "missing_requirements": [] if score == 1 else ["fixture"],
        "rationale": "fixture result",
    }


def run_self_tests(verifier_dir: Path) -> None:
    assert parse_strict_judge_json(json.dumps(fixed_judge_result(1)))["score"] == 1
    for invalid in (
        dict(fixed_judge_result(1), score=True),
        dict(fixed_judge_result(1), score=0.5),
        dict(fixed_judge_result(1), extra="no"),
    ):
        try:
            validate_judge_result(invalid)
        except EvaluationError:
            pass
        else:
            raise AssertionError("invalid Judge JSON was accepted")

    context = parse_task_yaml_allowlist(
        "task_id: demo\n"
        "hollow_file: src/a.py\n"
        "baseline_path: hidden/baseline/src/a.py\n"
        "baselines:\n"
        "  src/a.py: deadbeef\n"
    )
    assert context == {
        "task_id": "demo",
        "hollow_file": "src/a.py",
        "baseline_path": "src/a.py",
    }
    assert "secret-value" not in redact(
        "x secret-value y", ["secret-value"]
    )
    parser_dests = {action.dest for action in build_parser()._actions}
    assert not any(dest.startswith("agent_") for dest in parser_dests)
    assert "evaluation_mode" not in parser_dests

    previous = os.environ.copy()
    try:
        with tempfile.TemporaryDirectory(prefix="ptdbench-judge-selftest-") as temp:
            root = Path(temp)
            workspace = root / "workspace"
            oracle = root / "oracle"
            results = root / "results"
            (workspace / "outputs").mkdir(parents=True)
            oracle.mkdir()
            results.mkdir()
            (workspace / "outputs" / "diagnosis.txt").write_text(
                "VERDICT: unhealthy\n"
                "PRIMARY_ATTRIBUTION: attention_dropout=0.5\n",
                encoding="utf-8",
            )
            (oracle / "standard_answer.txt").write_text(
                "VERDICT: unhealthy\nREQUIRED_ALL:\n- attention_dropout=0.5\n",
                encoding="utf-8",
            )
            (oracle / "task.yaml").write_text(
                "task_id: fixture\noracle:\n"
                "  buggy_keys:\n  - attention_dropout\n"
                "  buggy_args:\n    attention_dropout: 0.5\n",
                encoding="utf-8",
            )
            write_json(
                results / "agent_trace.json",
                {
                    "result": {
                        "argv": ["codex", "--model", "fixture-model"]
                    },
                    "env": {"OPENAI_BASE_URL": "https://example.invalid/v1"},
                },
            )
            fixture = root / "judge_fixture.py"
            fixture.write_text(
                "import json\nprint(json.dumps("
                + repr(fixed_judge_result(1))
                + "))\n",
                encoding="utf-8",
            )
            os.environ.update(
                {
                    "PTDBENCH_STAGE": "verifier",
                    "PTDBENCH_WORKSPACE": str(workspace),
                    "PTDBENCH_ORACLE": str(oracle),
                    "PTDBENCH_VERIFIER": str(verifier_dir),
                    "PTDBENCH_RESULTS": str(results),
                    "PTDBENCH_CASE_ID": "fixture",
                }
            )
            args = build_parser().parse_args(
                [
                    "--test-mode",
                    "--judge-command",
                    shlex.quote(sys.executable) + " " + shlex.quote(str(fixture)),
                    "--run-id",
                    "judge-selftest",
                ]
            )
            contract = central_contract_paths(verifier_dir)
            run_dir = prepare_run(results, args)
            assert (
                run_evaluation(
                    args,
                    verifier_dir,
                    contract[0],
                    contract[1],
                    contract[2],
                    run_dir,
                )
                == 0
            )
            assert json.loads(
                (results / "metrics.json").read_text(encoding="utf-8")
            )["diagnosis_attribution_score"] == 1
            assert not list(run_dir.glob("agent_stdout*"))

            diagnosis_path = workspace / "outputs" / "diagnosis.txt"
            diagnosis_path.unlink()
            try:
                load_diagnosis(workspace)
            except EvaluationError:
                pass
            else:
                raise AssertionError("missing diagnosis was accepted")

            outside = root / "outside-diagnosis.txt"
            outside.write_text("VERDICT: healthy\n", encoding="utf-8")
            diagnosis_path.symlink_to(outside)
            try:
                load_diagnosis(workspace)
            except EvaluationError:
                pass
            else:
                raise AssertionError("symlinked diagnosis was accepted")
    finally:
        os.environ.clear()
        os.environ.update(previous)
    print("self-test: ok")


def persist_failure(
    run_dir: Optional[Path],
    results: Optional[Path],
    message: str,
    secret_values: Sequence[str],
) -> None:
    if run_dir is None or results is None:
        return
    safe_message = redact(message, secret_values)
    atomic_write_text(run_dir / "error.txt", safe_message + "\n")
    write_score(run_dir, results, 0)


def main(argv: Optional[Sequence[str]] = None) -> int:
    verifier_dir = Path(__file__).resolve().parent
    run_dir: Optional[Path] = None
    results: Optional[Path] = None
    secret_values: List[str] = []
    try:
        args = build_parser().parse_args(argv)
        if args.self_test:
            run_self_tests(verifier_dir)
            return 0
        workspace, oracle, results = central_contract_paths(verifier_dir)
        run_dir = prepare_run(results, args)
        configured_env = args.judge_api_key_env
        if configured_env and ENV_NAME_RE.fullmatch(configured_env):
            secret_values.append(os.environ.get(configured_env, ""))
        try:
            return run_evaluation(
                args, verifier_dir, workspace, oracle, results, run_dir
            )
        except EvaluationError as exc:
            configured_env = args.judge_api_key_env
            if configured_env and ENV_NAME_RE.fullmatch(configured_env):
                secret_values.append(os.environ.get(configured_env, ""))
            persist_failure(run_dir, results, str(exc), secret_values)
            print("verification failed: " + redact(str(exc), secret_values), file=sys.stderr)
            return 1
    except EvaluationError as exc:
        persist_failure(run_dir, results, str(exc), secret_values)
        print("verification failed: " + redact(str(exc), secret_values), file=sys.stderr)
        return 1
    except Exception as exc:  # fail closed without exposing a traceback or secret
        persist_failure(
            run_dir, results, "unexpected verifier error: " + str(exc), secret_values
        )
        print("verification failed: unexpected verifier error", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
