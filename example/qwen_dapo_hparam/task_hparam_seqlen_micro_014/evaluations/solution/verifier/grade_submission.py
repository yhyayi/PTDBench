#!/usr/bin/env python3
"""Grade the repaired post-training run artifacts for this task."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import secrets
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


REQUIRED_SUBMISSION_KEYS = {
    "run_dir",
    "metrics_path",
    "receipt_path",
    "log_path",
    "command_path",
    "changed_files",
}
EXPECTED_HPARAMS = {
    "MAX_PROMPT_LENGTH": "1024",
    "MAX_RESPONSE_LENGTH": "1024",
    "PPO_MICRO_BATCH_SIZE_PER_GPU": "1",
    "LOG_PROB_MICRO_BATCH_SIZE_PER_GPU": "1",
    "TRAIN_BATCH_SIZE": "8",
}
DIRTY_HPARAMS = {
    "MAX_PROMPT_LENGTH": "512",
    "MAX_RESPONSE_LENGTH": "512",
}
LAUNCH_SCRIPT = Path("examples/grpo_trainer/run_qwen2_5_0_5b_instruct_dapo_math_fsdp.sh")
METRIC_KEY = "val-core/math_dapo/acc/mean@1"
MIN_FINAL_SCORE = 0.045
MIN_BEST_SCORE = 0.05
MAX_RESPONSE_CLIP_RATIO = 0.45
MIN_RESPONSE_MAX = 900.0
REQUIRED_FINAL_STEP = 200


class GradingError(RuntimeError):
    """A malformed or unsafe verifier input."""


def atomic_write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + secrets.token_hex(4))
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(str(temporary), str(path))
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def read_json(path: Path, label: str) -> Any:
    if path.is_symlink():
        raise GradingError(f"{label} must not be a symlink: {path}")
    if not path.is_file():
        raise GradingError(f"missing {label}: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise GradingError(f"{label} must be UTF-8: {path}") from exc
    except json.JSONDecodeError as exc:
        raise GradingError(f"{label} is invalid JSON: {path}: {exc}") from exc


def read_text(path: Path, label: str, limit_bytes: int = 8_000_000) -> str:
    if path.is_symlink():
        raise GradingError(f"{label} must not be a symlink: {path}")
    if not path.is_file():
        raise GradingError(f"missing {label}: {path}")
    if path.stat().st_size > limit_bytes:
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - limit_bytes))
            payload = handle.read()
        return payload.decode("utf-8", errors="replace")
    return path.read_text(encoding="utf-8", errors="replace")


def find_workspace(verifier_dir: Path) -> Path:
    for parent in verifier_dir.parents:
        candidate = parent / "workspace"
        if candidate.is_dir() and (candidate / LAUNCH_SCRIPT).is_file():
            return candidate.resolve()
    raise GradingError("could not locate task workspace from verifier path")


def resolve_workspace_path(raw: str, workspace: Path, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise GradingError(f"{label} must be a non-empty string")
    path = Path(raw)
    if not path.is_absolute():
        path = workspace / path
    resolved = path.resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise GradingError(f"{label} must stay inside workspace: {raw}") from exc
    return resolved


def load_submission(path: Path) -> Dict[str, Any]:
    value = read_json(path, "submission.json")
    if not isinstance(value, dict):
        raise GradingError("submission must be a JSON object")
    if set(value) != REQUIRED_SUBMISSION_KEYS:
        raise GradingError(
            "submission keys must be exactly run_dir, metrics_path, receipt_path, "
            "log_path, command_path, changed_files"
        )
    for key in REQUIRED_SUBMISSION_KEYS - {"changed_files"}:
        if not isinstance(value[key], str) or not value[key].strip():
            raise GradingError(f"{key} must be a non-empty string")
    if not isinstance(value["changed_files"], list) or any(
        not isinstance(item, str) or not item.strip()
        for item in value["changed_files"]
    ):
        raise GradingError("changed_files must be a non-empty array of strings")
    if not value["changed_files"]:
        raise GradingError("changed_files must name the repair target")
    return value


def extract_shell_default(script: str, name: str) -> Optional[str]:
    default_pattern = re.compile(
        rf"^\s*{re.escape(name)}=\$\{{{re.escape(name)}:-([^}}]+)\}}\s*$",
        re.MULTILINE,
    )
    match = default_pattern.search(script)
    if match:
        return match.group(1).strip().strip("\"'")
    assignment_pattern = re.compile(
        rf"^\s*{re.escape(name)}=([^\s#]+)\s*(?:#.*)?$",
        re.MULTILINE,
    )
    match = assignment_pattern.search(script)
    if match:
        return match.group(1).strip().strip("\"'")
    return None


def check_launch_script(workspace: Path, changed_files: list[str]) -> Dict[str, Any]:
    script_path = workspace / LAUNCH_SCRIPT
    script = read_text(script_path, "launch script")
    observed = {
        name: extract_shell_default(script, name)
        for name in EXPECTED_HPARAMS
    }
    checks = {
        name: observed.get(name) == expected
        for name, expected in EXPECTED_HPARAMS.items()
    }
    dirty_left = {
        name: observed.get(name) == dirty
        for name, dirty in DIRTY_HPARAMS.items()
    }
    changed_norm = set()
    for item in changed_files:
        path = Path(item)
        if path.is_absolute():
            try:
                changed_norm.add(path.resolve().relative_to(workspace).as_posix())
                continue
            except ValueError:
                changed_norm.add(path.as_posix())
                continue
        changed_norm.add(path.as_posix().lstrip("./"))
    launch_named = LAUNCH_SCRIPT.as_posix() in changed_norm
    allowed_changes = {LAUNCH_SCRIPT.as_posix()}
    unexpected_changes = sorted(changed_norm - allowed_changes)
    passed = (
        all(checks.values())
        and not any(dirty_left.values())
        and launch_named
        and not unexpected_changes
    )
    return {
        "passed": passed,
        "launch_script": LAUNCH_SCRIPT.as_posix(),
        "changed_files_mentions_launch": launch_named,
        "unexpected_changed_files": unexpected_changes,
        "observed": observed,
        "expected": EXPECTED_HPARAMS,
        "checks": checks,
        "dirty_values_left": dirty_left,
    }


def numeric(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def iter_dicts(value: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from iter_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_dicts(child)


def metric_from_raw(item: Dict[str, Any], key: str) -> Optional[float]:
    direct = numeric(item.get(key))
    if direct is not None:
        return direct
    raw = item.get("raw")
    if isinstance(raw, dict):
        return numeric(raw.get(key))
    return None


def collect_eval_points(metrics: Any) -> list[Dict[str, float]]:
    points: Dict[int, float] = {}
    for item in iter_dicts(metrics):
        step_value = numeric(item.get("step"))
        if step_value is None:
            step_value = numeric(item.get("global_step"))
        score = metric_from_raw(item, METRIC_KEY)
        if score is None and item.get("eval_key") == METRIC_KEY:
            score = numeric(item.get("eval_score"))
        if score is None and "score" in item and len(item) <= 3:
            score = numeric(item.get("score"))
        if step_value is None or score is None:
            continue
        step = int(step_value)
        points[step] = score
    return [
        {"step": float(step), "score": score}
        for step, score in sorted(points.items())
    ]


def collect_latest_key_values(log_text: str) -> Dict[str, float]:
    wanted = {
        METRIC_KEY,
        "response_length/clip_ratio",
        "response_length/max",
        "response_length_non_aborted/clip_ratio",
        "response_length_non_aborted/max",
        "prompt_length/max",
        "training/global_step",
    }
    values: Dict[str, float] = {}
    token = re.compile(r"([A-Za-z0-9_./@-]+):\s*(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")
    for key, raw in token.findall(log_text):
        if key in wanted:
            values[key] = float(raw)
    return values


def collect_final_metrics_from_log(log_text: str) -> Dict[str, float]:
    values = collect_latest_key_values(log_text)
    final_match = re.search(r"Final validation metrics:\s*(\{.*?\})", log_text, flags=re.DOTALL)
    if final_match:
        try:
            parsed = ast.literal_eval(final_match.group(1))
            if isinstance(parsed, dict):
                score = numeric(parsed.get(METRIC_KEY))
                if score is not None:
                    values[METRIC_KEY] = score
        except (SyntaxError, ValueError):
            pass
    return values


def check_receipt(receipt: Any) -> Dict[str, Any]:
    if not isinstance(receipt, dict):
        return {"passed": False, "why": "receipt is not an object"}
    latest_step = numeric(receipt.get("latest_step"))
    max_steps = numeric(receipt.get("max_train_steps"))
    passed = (
        receipt.get("ok") is True
        and int(latest_step or -1) >= 1
        and int(max_steps or -1) == REQUIRED_FINAL_STEP
        and receipt.get("startup_failed") is not True
        and receipt.get("early_stopped") is not True
        and receipt.get("killed_on_first_step") is not True
    )
    return {
        "passed": passed,
        "ok": receipt.get("ok"),
        "returncode": receipt.get("returncode"),
        "latest_step": latest_step,
        "max_train_steps": max_steps,
        "startup_failed": receipt.get("startup_failed"),
        "early_stopped": receipt.get("early_stopped"),
        "killed_on_first_step": receipt.get("killed_on_first_step"),
    }


def check_command(command_text: str) -> Dict[str, Any]:
    launch = LAUNCH_SCRIPT.as_posix()
    mentions_launch = launch in command_text
    forbidden = [
        "historical_evidence",
        "evaluations/solution/verifier",
        "evaluations/solution/oracle",
        "gold.yaml",
    ]
    forbidden_hits = [item for item in forbidden if item in command_text]
    return {
        "passed": mentions_launch and not forbidden_hits,
        "mentions_launch_script": mentions_launch,
        "forbidden_mentions": forbidden_hits,
    }


def check_log(log_text: str) -> Dict[str, Any]:
    values = collect_final_metrics_from_log(log_text)
    final_step = values.get("training/global_step")
    final_score = values.get(METRIC_KEY)
    response_clip = values.get("response_length/clip_ratio")
    response_max = values.get("response_length/max")
    if response_clip is None:
        response_clip = values.get("response_length_non_aborted/clip_ratio")
    if response_max is None:
        response_max = values.get("response_length_non_aborted/max")
    no_failure = not re.search(
        r"\b(traceback|outofmemory|oom|startup_failed|killed_on_first_step)\b",
        log_text,
        flags=re.IGNORECASE,
    )
    has_training_signal = any(
        value is not None
        for value in (final_step, final_score, response_clip, response_max)
    )
    step_ok = final_step is not None and int(final_step) >= REQUIRED_FINAL_STEP
    score_ok = final_score is not None and final_score >= MIN_FINAL_SCORE
    response_ok = response_clip is not None and (
        response_clip <= MAX_RESPONSE_CLIP_RATIO
        or (response_max is not None and response_max >= MIN_RESPONSE_MAX)
    )
    return {
        "passed": no_failure and has_training_signal and step_ok and score_ok and response_ok,
        "has_training_signal": has_training_signal,
        "no_failure_tokens": no_failure,
        "latest_values": values,
        "step_ok": step_ok,
        "score_ok": score_ok,
        "response_length_signal_ok": response_ok,
    }


def check_metrics(metrics: Any, log_values: Dict[str, float]) -> Dict[str, Any]:
    points = collect_eval_points(metrics)
    final_point = points[-1] if points else None
    final_score = final_point["score"] if final_point else log_values.get(METRIC_KEY)
    final_step = int(final_point["step"]) if final_point else int(log_values.get("training/global_step", -1))
    best_score = max((point["score"] for point in points), default=final_score)
    if best_score is None:
        best_score = -1.0
    score_ok = final_score is not None and final_score >= MIN_FINAL_SCORE
    best_ok = best_score >= MIN_BEST_SCORE
    step_ok = final_step >= REQUIRED_FINAL_STEP
    return {
        "passed": score_ok and best_ok and step_ok,
        "metric_key": METRIC_KEY,
        "final_step": final_step,
        "final_score": final_score,
        "best_score": best_score,
        "num_eval_points": len(points),
        "thresholds": {
            "min_final_score": MIN_FINAL_SCORE,
            "min_best_score": MIN_BEST_SCORE,
            "required_final_step": REQUIRED_FINAL_STEP,
        },
        "score_ok": score_ok,
        "best_ok": best_ok,
        "step_ok": step_ok,
    }


def grade(submission: Dict[str, Any], workspace: Path) -> Dict[str, Any]:
    paths = {
        key: resolve_workspace_path(submission[key], workspace, key)
        for key in ("metrics_path", "receipt_path", "log_path", "command_path")
    }
    run_dir = resolve_workspace_path(submission["run_dir"], workspace, "run_dir")
    if not run_dir.is_dir():
        raise GradingError(f"run_dir is not a directory: {run_dir}")
    for key, path in paths.items():
        try:
            path.relative_to(run_dir)
        except ValueError:
            raise GradingError(f"{key} must be inside run_dir: {path}")

    metrics = read_json(paths["metrics_path"], "metrics_path")
    receipt = read_json(paths["receipt_path"], "receipt_path")
    log_text = read_text(paths["log_path"], "log_path")
    command_text = read_text(paths["command_path"], "command_path", limit_bytes=200_000)
    changed_files = list(submission["changed_files"])

    launch_check = check_launch_script(workspace, changed_files)
    receipt_check = check_receipt(receipt)
    command_check = check_command(command_text)
    log_check = check_log(log_text)
    metrics_check = check_metrics(metrics, log_check["latest_values"])
    artifact_checks = {
        "run_dir": str(run_dir.relative_to(workspace)),
        "metrics_path": str(paths["metrics_path"].relative_to(workspace)),
        "receipt_path": str(paths["receipt_path"].relative_to(workspace)),
        "log_path": str(paths["log_path"].relative_to(workspace)),
        "command_path": str(paths["command_path"].relative_to(workspace)),
    }
    checks = {
        "artifacts": {"passed": True, **artifact_checks},
        "launch_config": launch_check,
        "receipt": receipt_check,
        "command": command_check,
        "metrics": metrics_check,
        "log": log_check,
    }
    passed = all(item.get("passed") is True for item in checks.values())
    weights = {
        "artifacts": 0.10,
        "launch_config": 0.30,
        "receipt": 0.15,
        "command": 0.10,
        "metrics": 0.30,
        "log": 0.05,
    }
    score = sum(weights[name] for name, item in checks.items() if item.get("passed") is True)
    failures = [name for name, item in checks.items() if item.get("passed") is not True]
    return {
        "task_key": "hyperparameter/verl/task_hparam_seqlen_micro_014",
        "evaluation_level": "solution",
        "verdict": "healthy_run" if passed else "unhealthy_or_unproven_run",
        "passed": passed,
        "case_score": 1.0 if passed else round(score, 4),
        "failures": failures,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", default=os.environ.get("PTDBENCH_EVALUATION_LEVEL", "solution"))
    parser.add_argument("--submission")
    args, _unknown = parser.parse_known_args()
    if args.level != "solution":
        raise GradingError("this verifier only implements solution-level run grading")

    verifier_dir = Path(os.environ.get("PTDBENCH_VERIFIER", Path(__file__).resolve().parent)).resolve()
    results_dir = Path(os.environ.get("PTDBENCH_RESULTS", "/results")).resolve()
    submission_path = Path(args.submission).resolve() if args.submission else results_dir / "submission.json"

    try:
        workspace = find_workspace(verifier_dir)
        submission = load_submission(submission_path)
        grading = grade(submission, workspace)
        payload = {
            "passed": grading["passed"],
            "score": grading["case_score"],
            "evaluation_level": "solution",
            "metrics": grading,
        }
        atomic_write_json(results_dir / "metrics.json", grading)
        atomic_write_json(results_dir / "result.json", payload)
        return 0 if grading["passed"] else 1
    except (GradingError, OSError, ValueError) as exc:
        grading = {
            "evaluation_level": "solution",
            "verdict": "input_error",
            "passed": False,
            "case_score": 0.0,
            "input_error": str(exc),
        }
        payload = {
            "passed": False,
            "score": 0.0,
            "evaluation_level": "solution",
            "metrics": grading,
        }
        try:
            atomic_write_json(results_dir / "metrics.json", grading)
            atomic_write_json(results_dir / "result.json", payload)
        except OSError:
            pass
        print(f"verifier input error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
