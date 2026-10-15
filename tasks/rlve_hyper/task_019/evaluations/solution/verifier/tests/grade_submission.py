#!/usr/bin/env python3
"""Extract held-out eval and learning rate from fresh logs and assign a weighted score.

The graded validation number is the metric at ``final_step`` -- the last step of
the locked training budget -- not the peak over all steps. Grading the peak let
a run that stopped at 40% of the budget pass on an early eval point, and made
the score grow with the budget instead of with the repair. ``final_step`` is a
fixed index, so lengthening the run cannot inflate it either.

Thresholds are calibrated against two reruns of this exact staged workspace on
4xH100 (see resources/README.md): defect arm -0.5321 at step 99, repaired arm
-0.1021 at step 99.
"""

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
        "min_eval_points",
        "historical_final_baseline",
        "defect_rerun_final_score",
        "repaired_rerun_final_score",
        "required_final_score",
        "required_final_score_max",
        "learning_rate_key",
        "required_learning_rate",
        "min_learning_rate_points",
        "validation_weight",
        "learning_rate_weight",
        "log_roots",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("grader_spec.json has unexpected fields")
    if value["schema_version"] != 3:
        raise ValueError("unsupported grader spec")
    if value["log_roots"] != ["experiment_logs"]:
        raise ValueError("log roots must be experiment_logs")
    return value


def metric_pattern(metric_key: str) -> re.Pattern[str]:
    """Match slime held-out lines: `eval 19: {'eval/HELD-OUT_ENVIRONMENTS_128': -0.64, ...}`.

    The `(?!-)` after the quoted key skips sibling keys such as
    `eval/HELD-OUT_ENVIRONMENTS_128-truncated_ratio`.
    """
    escaped = re.escape(metric_key)
    return re.compile(
        rf"eval\s+(\d+)\s*:\s*\{{[^\n]*?['\"]({escaped})['\"](?!-)\s*:\s*({NUMBER})",
        re.IGNORECASE,
    )


def learning_rate_pattern(key: str) -> re.Pattern[str]:
    """Match slime train lines: `step 0: {'train/lr-pg_0': 1e-06, ...}`."""
    escaped = re.escape(key)
    return re.compile(
        rf"(?:^|\s)step\s+(\d+)\s*:\s*\{{[^\n]*?['\"]({escaped})['\"]\s*:\s*({NUMBER})",
        re.IGNORECASE,
    )


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


def record_point(
    values: dict[int, tuple[Any, Decimal, str]],
    step: int,
    value: Decimal,
    rank: tuple[Any, ...],
    relative: str,
) -> None:
    previous = values.get(step)
    if (
        previous is None
        or value > previous[1]
        or (value == previous[1] and rank > previous[0])
    ):
        values[step] = (rank, value, relative)


def extract(spec: dict[str, Any], results: Path):
    expected_key = spec["metric_key"]
    lr_key = spec["learning_rate_key"]
    pattern = metric_pattern(expected_key)
    lr_pattern = learning_rate_pattern(lr_key)
    values: dict[int, tuple[Any, Decimal, str]] = {}
    lr_values: dict[int, tuple[Any, Decimal, str]] = {}
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
        for order, match in enumerate(pattern.finditer(text)):
            matched = True
            step = int(match.group(1))
            key = match.group(2)
            token = match.group(3)
            keys_seen.add(key)
            if key != expected_key:
                continue
            try:
                value = Decimal(token)
            except InvalidOperation:
                continue
            if not value.is_finite():
                continue
            record_point(values, step, value, (mtime, relative, order), relative)
        for order, match in enumerate(lr_pattern.finditer(text)):
            matched = True
            step = int(match.group(1))
            key = match.group(2)
            token = match.group(3)
            keys_seen.add(key)
            if key != lr_key:
                continue
            try:
                value = Decimal(token)
            except InvalidOperation:
                continue
            if not value.is_finite():
                continue
            record_point(lr_values, step, value, (mtime, relative, order), relative)
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
    lr_points = []
    for step in sorted(lr_values):
        rate = float(lr_values[step][1])
        lr_points.append(
            {
                "step": step,
                "learning_rate": rate,
                "learning_rate_key": lr_key,
                "source_log": lr_values[step][2],
            }
        )
    experiment = {
        "task_id": spec["task_id"],
        "logs": {"source_logs": sorted(set(source_logs))},
        "metrics": {
            "metric_key": expected_key,
            "metric_status": "available" if points else "missing",
            "eval_points": points,
            "eval_scores": scores,
            "learning_rate_key": lr_key,
            "learning_rate_status": "available" if lr_points else "missing",
            "learning_rate_points": lr_points,
        },
    }
    return experiment, values, lr_values, keys_seen


def select_best(values: dict[int, tuple[Any, Decimal, str]]):
    best = None
    for step, (rank, value, relative) in values.items():
        candidate = (value, step, rank, relative)
        if best is None or candidate > best:
            best = candidate
    return best


def mean_value(values: dict[int, tuple[Any, Decimal, str]]):
    if not values:
        return None
    total = sum(item[1] for item in values.values())
    return total / Decimal(len(values))


def binary_score(passed: bool) -> Decimal:
    return Decimal("1") if passed else Decimal("0")


def grade(spec: dict[str, Any], results: Path):
    experiment, values, lr_values, keys_seen = extract(spec, results)
    metric_key = spec["metric_key"]
    lr_key = spec["learning_rate_key"]
    final_step = spec["final_step"]
    min_eval_points = spec["min_eval_points"]
    min_lr_points = spec["min_learning_rate_points"]
    baseline = Decimal(str(spec["historical_final_baseline"]))
    required_score = Decimal(str(spec["required_final_score"]))
    required_score_max = Decimal(str(spec["required_final_score_max"]))
    required_lr = Decimal(str(spec["required_learning_rate"]))
    validation_weight = Decimal(str(spec["validation_weight"]))
    learning_rate_weight = Decimal(str(spec["learning_rate_weight"]))
    errors = []
    best = select_best(values)
    best_value = best[0] if best is not None else None
    best_step = best[1] if best is not None else None
    best_source = best[3] if best is not None else None
    final_point = values.get(final_step)
    final_value = final_point[1] if final_point is not None else None
    final_source = final_point[2] if final_point is not None else None
    improvement = final_value - baseline if final_value is not None else None
    lr_mean = mean_value(lr_values)
    lr_source = None
    if lr_values:
        latest_step = max(lr_values)
        lr_source = lr_values[latest_step][2]

    # Validation grades the locked budget's last step. A run that never logged
    # it did not finish the budget, whatever its earlier points look like.
    if final_value is None:
        if values:
            errors.append(
                f"no {metric_key} at step {final_step}; steps present: "
                f"{', '.join(str(step) for step in sorted(values))}"
            )
        elif metric_key in keys_seen:
            errors.append(f"missing finite {metric_key}")
        elif keys_seen:
            errors.append(
                f"wrong metric in experiment_logs: {', '.join(sorted(keys_seen))}"
            )
        else:
            errors.append(f"missing {metric_key} in experiment_logs")
        validation_passed = False
    elif len(values) < min_eval_points:
        errors.append(
            f"only {len(values)} {metric_key} point(s); at least "
            f"{min_eval_points} are required"
        )
        validation_passed = False
    elif final_value < required_score:
        errors.append(
            f"step {final_step} score {final_value} is below "
            f"required score {required_score}"
        )
        validation_passed = False
    elif final_value > required_score_max:
        # Reward is bounded to [-1, 1] and the repaired rerun finished at
        # -0.102, so anything past this bound is not a 100-step GRPO result on
        # the locked held-out set.
        errors.append(
            f"step {final_step} score {final_value} exceeds the plausible "
            f"maximum {required_score_max}; check for fabricated or "
            f"off-protocol logs"
        )
        validation_passed = False
    else:
        validation_passed = True

    if lr_mean is None:
        errors.append(f"missing finite {lr_key} in experiment_logs")
        learning_rate_passed = False
    elif final_step not in lr_values:
        errors.append(
            f"no {lr_key} at step {final_step}; the training budget did not run "
            f"to completion"
        )
        learning_rate_passed = False
    elif len(lr_values) < min_lr_points:
        errors.append(
            f"only {len(lr_values)} {lr_key} step(s); at least {min_lr_points} "
            f"are required"
        )
        learning_rate_passed = False
    elif lr_mean < required_lr:
        errors.append(
            f"mean learning rate {lr_mean} is below required rate {required_lr}"
        )
        learning_rate_passed = False
    else:
        learning_rate_passed = True

    validation_score = binary_score(validation_passed)
    learning_rate_score = binary_score(learning_rate_passed)
    score = float(
        validation_weight * validation_score + learning_rate_weight * learning_rate_score
    )
    passed = score == 1.0
    judged = float(final_value) if final_value is not None else None
    peak = float(best_value) if best_value is not None else None
    metrics = {
        "task_id": spec["task_id"],
        "metric_key": metric_key,
        "final_step": final_step,
        "historical_final_baseline": float(baseline),
        "defect_rerun_final_score": float(spec["defect_rerun_final_score"]),
        "repaired_rerun_final_score": float(spec["repaired_rerun_final_score"]),
        "required_final_score": float(required_score),
        "required_final_score_max": float(required_score_max),
        # Peak is diagnostic only; `final_score` is what validation grades.
        "best_step": best_step,
        "best_score": peak,
        "best_source_log": best_source,
        "final_score": judged,
        "final_source_log": final_source,
        "n_eval_points": len(values),
        "absolute_improvement": (
            float(improvement) if improvement is not None else None
        ),
        "validation_score": float(validation_score),
        "learning_rate_key": lr_key,
        "learning_rate_mean": float(lr_mean) if lr_mean is not None else None,
        "learning_rate_n": len(lr_values),
        "learning_rate_source_log": lr_source,
        "required_learning_rate": float(required_lr),
        "learning_rate_score": float(learning_rate_score),
        "source_logs": experiment["logs"]["source_logs"],
        "errors": errors,
        "passed": passed,
        "score": score,
        "case_score": score,
        "deterministic": True,
        "judge_used": False,
    }
    experiment["metrics"]["best_step"] = best_step
    experiment["metrics"]["best_score"] = peak
    experiment["metrics"]["best_source_log"] = best_source
    experiment["metrics"]["final_score"] = judged
    experiment["metrics"]["final_source_log"] = final_source
    experiment["metrics"]["n_eval_points"] = len(values)
    experiment["metrics"]["learning_rate_mean"] = metrics["learning_rate_mean"]
    experiment["metrics"]["learning_rate_n"] = metrics["learning_rate_n"]
    experiment["metrics"]["validation_score"] = metrics["validation_score"]
    experiment["metrics"]["learning_rate_score"] = metrics["learning_rate_score"]
    experiment["metrics"]["score"] = score
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
