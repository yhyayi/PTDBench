#!/usr/bin/env python3
"""Extract the target Verl metric from fresh logs and assign a binary score."""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|[@-_])")
NUMBER = (
    r"[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    r"|nan|inf(?:inity)?)"
)
VAL_RE = re.compile(
    r"(?:^|\s)step:(\d+)\s+-.*?"
    r"\b(val-core/([^\s:]+)/(acc|reward)/mean@(\d+)):"
    rf"({NUMBER})",
    re.IGNORECASE,
)
TEXT_SUFFIXES = {".json", ".jsonl", ".log", ".out", ".txt", ""}
MAX_FILE_BYTES = 128 * 1024 * 1024


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise OSError(f"refusing to replace symlink: {path}")
    temporary = path.with_name(f"{path.name}.tmp-{secrets.token_hex(4)}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def load_spec(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "task_id",
        "metric_key",
        "final_step",
        "historical_final_baseline",
        "required_final_score",
        "log_roots",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("grader_spec.json has unexpected fields")
    if value["schema_version"] != 2:
        raise ValueError("unsupported grader spec")
    if value["log_roots"] != ["experiment_logs"]:
        raise ValueError("log roots must be experiment_logs")
    return value


def log_files(results: Path, roots: list[str]) -> list[Path]:
    found = []
    for root_name in roots:
        root = results / root_name
        if root.is_symlink() or not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_symlink() or not path.is_file():
                continue
            if path.suffix.lower() == ".wandb":
                continue
            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            found.append(path)
    return sorted(found, key=lambda path: (path.stat().st_mtime_ns, str(path)))


def extract(spec: dict[str, Any], results: Path):
    expected_key = spec["metric_key"]
    values = {}
    keys_seen: set[str] = set()
    source_logs = []
    for path in log_files(results, spec["log_roots"]):
        try:
            data = path.read_bytes()
            if b"\x00" in data[:4096]:
                continue
            text = ANSI_RE.sub("", data.decode("utf-8", errors="replace"))
            mtime = path.stat().st_mtime_ns
        except OSError:
            continue
        relative = path.relative_to(results).as_posix()
        matched = False
        for order, match in enumerate(VAL_RE.finditer(text)):
            matched = True
            step = int(match.group(1))
            key = match.group(2)
            token = match.group(6)
            keys_seen.add(key)
            if key != expected_key:
                continue
            try:
                value = Decimal(token)
            except InvalidOperation:
                continue
            if not value.is_finite():
                continue
            rank = (mtime, relative, order)
            previous = values.get(step)
            if (
                previous is None
                or value > previous[1]
                or (value == previous[1] and rank > previous[0])
            ):
                values[step] = (rank, value, relative)
        if matched:
            source_logs.append(relative)

    points = []
    scores = []
    for step in sorted(values):
        score = float(values[step][1])
        points.append(
            {
                "step": step,
                "eval_score": score,
                "eval_key": expected_key,
                "truncate_ratio": None,
                "raw": {expected_key: score},
                "source_log": values[step][2],
            }
        )
        scores.append({"step": step, "score": score})
    experiment = {
        "task_id": spec["task_id"],
        "logs": {"source_logs": sorted(set(source_logs))},
        "metrics": {
            "metric_key": expected_key,
            "metric_status": "available" if points else "missing",
            "eval_points": points,
            "eval_scores": scores,
        },
    }
    return experiment, values, keys_seen


def select_peak(values: dict[int, tuple[Any, Decimal, str]]):
    """Highest point over the whole curve. Informational only, never graded.

    The peak is a max over ~11 noisy eval points, so it carries an upward
    selection bias and is inflatable by simply evaluating more often. See
    `resources/README.md` for the calibration that moved grading to
    `final_step`.
    """
    peak = None
    for step, (rank, value, relative) in values.items():
        candidate = (value, step, rank, relative)
        if peak is None or candidate > peak:
            peak = candidate
    return peak


def grade(spec: dict[str, Any], results: Path):
    experiment, values, keys_seen = extract(spec, results)
    metric_key = spec["metric_key"]
    final_step = spec["final_step"]
    baseline = Decimal(str(spec["historical_final_baseline"]))
    required_score = Decimal(str(spec["required_final_score"]))
    errors = []
    peak = select_peak(values)
    final = values.get(final_step)
    final_value = final[1] if final is not None else None
    final_source = final[2] if final is not None else None
    improvement = final_value - baseline if final_value is not None else None
    if final_value is None:
        if values:
            steps_seen = ", ".join(str(step) for step in sorted(values))
            errors.append(
                f"missing {metric_key} at step {final_step}; "
                f"steps present: {steps_seen}"
            )
        elif metric_key in keys_seen:
            errors.append(f"missing finite {metric_key}")
        elif keys_seen:
            errors.append(
                f"wrong metric in experiment_logs: {', '.join(sorted(keys_seen))}"
            )
        else:
            errors.append(f"missing {metric_key} in experiment_logs")
    elif final_value < required_score:
        errors.append(
            f"final score {final_value} at step {final_step} is below "
            f"required score {required_score}"
        )
    passed = not errors
    score = 1.0 if passed else 0.0
    judged = float(final_value) if final_value is not None else None
    peak_step = peak[1] if peak is not None else None
    peak_score = float(peak[0]) if peak is not None else None
    metrics = {
        "task_id": spec["task_id"],
        "metric_key": metric_key,
        "final_step": final_step,
        "historical_final_baseline": float(baseline),
        "required_final_score": float(required_score),
        "final_score": judged,
        "final_source_log": final_source,
        "peak_step": peak_step,
        "peak_score": peak_score,
        "absolute_improvement": (
            float(improvement) if improvement is not None else None
        ),
        "source_logs": experiment["logs"]["source_logs"],
        "errors": errors,
        "passed": passed,
        "score": score,
        "case_score": score,
        "deterministic": True,
        "judge_used": False,
    }
    experiment["metrics"]["final_step"] = final_step
    experiment["metrics"]["final_score"] = judged
    experiment["metrics"]["final_source_log"] = final_source
    experiment["metrics"]["peak_step"] = peak_step
    experiment["metrics"]["peak_score"] = peak_score
    return experiment, metrics, {"passed": passed, "score": score, "metrics": metrics}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(os.environ.get("PTDBENCH_RESULTS", "/results")),
    )
    args = parser.parse_args()
    results = args.results.expanduser().resolve()
    try:
        spec = load_spec(Path(__file__).resolve().parent.parent / "resources" / "grader_spec.json")
        experiment, metrics, result = grade(spec, results)
        write_json(results / "experiment_metrics.json", experiment)
        write_json(results / "metrics.json", metrics)
        write_json(results / "result.json", result)
        if metrics["errors"]:
            print("; ".join(metrics["errors"]), file=sys.stderr)
        return 0 if result["passed"] else 1
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        metrics = {
            "score": 0.0,
            "case_score": 0.0,
            "passed": False,
            "errors": [f"verifier error: {exc}"],
            "deterministic": True,
            "judge_used": False,
        }
        result = {"passed": False, "score": 0.0, "metrics": metrics}
        try:
            write_json(results / "metrics.json", metrics)
            write_json(results / "result.json", result)
        except OSError:
            pass
        print(metrics["errors"][0], file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
