#!/usr/bin/env python3
"""Grade one repaired RLVE run: held-out eval plus RLVE training accuracy.

Scoring is deliberately bound to a *single* launch. The log root accumulates one
subtree per `rlve.sh` invocation, so an agent that debugs with a throwaway launch
before the real one leaves several runs behind. Pooling them mixes a repaired run
with an unrepaired one and sinks both halves of the score, so the grader first
elects one launch log -- the newest text log that both reports held-out eval and
announces its offline W&B directories -- and reads everything from that launch.

Text logs are trivial to hand-write, so the elected launch log is not trusted on
its own: every eval step it reports that also appears in the launch's own
`.wandb` stream must match. W&B writes a length-prefixed binary record stream,
which is far harder to fabricate consistently than a line of text. The tail of
the curve is exempt because the final eval is flushed to the text log but not to
the binary stream -- observed on every reference run -- so corroboration is
required on the overlap, not on the graded point itself.
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

# `wandb: Run data is saved locally in <dir>/wandb/offline-run-<ts>-<id>`.
# Only the basename is used: the path is relative when WANDB_DIR is relative and
# absolute otherwise, but the leaf is stable either way.
WANDB_DIR_RE = re.compile(r"Run data is saved locally in\s+(\S+)")
OFFLINE_DIR_RE = re.compile(r"^offline-run-[0-9]{8}_[0-9]{6}-[0-9a-z]+$")

# W&B history/summary records encode a value as `\x82\x01<len><ascii digits>`.
WANDB_ACC_RE = re.compile(rb"RLVE/([A-Za-z0-9_]+)/accuracy\x82\x01(.)", re.DOTALL)
WANDB_STEP_RE = re.compile(rb"\x12\x09eval/step\x82\x01(.)", re.DOTALL)
# Distance from an eval value to the `eval/step` field of the same record.
WANDB_STEP_WINDOW = 512

# Tolerance when comparing a text-log eval value against the binary stream. Both
# sides serialise the same float with repr(), so equality is exact in practice;
# the epsilon only absorbs a differing trailing digit.
EVAL_AGREEMENT_EPSILON = Decimal("1e-9")


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
        "required_final_step",
        "historical_final_baseline",
        "required_final_score",
        "required_final_score_max",
        "min_corroborated_eval_points",
        "training_reward_key",
        "training_reward_tail_fraction",
        "required_training_reward",
        "min_training_reward_points",
        "validation_weight",
        "training_weight",
        "log_roots",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("grader_spec.json has unexpected fields")
    if value["schema_version"] != 3:
        raise ValueError("unsupported grader spec")
    if value["log_roots"] != ["experiment_logs"]:
        raise ValueError("log roots must be experiment_logs")
    if value["training_reward_key"] != "RLVE/*/accuracy":
        raise ValueError("training reward key must be RLVE/*/accuracy")
    tail = Decimal(str(value["training_reward_tail_fraction"]))
    if not Decimal(0) < tail <= Decimal(1):
        raise ValueError("training_reward_tail_fraction must be in (0, 1]")
    return value


def metric_pattern(metric_key: str) -> re.Pattern[str]:
    """Match slime held-out lines: `eval 19: {'eval/HELD-OUT_ENVIRONMENTS_128': -0.64, ...}`.

    The `(?!-)` after the quoted key skips sibling keys such as
    `eval/HELD-OUT_ENVIRONMENTS_128-truncated_ratio`.
    """
    escaped = re.escape(metric_key)
    return re.compile(
        rf"eval\s+(\d+)\s*:\s*\{{[^\n]*?['\"]({escaped})['\"](?!-)\s*:\s*({NUMBER})"
    )


def wandb_eval_pattern(metric_key: str) -> re.Pattern[bytes]:
    """Match the metric inside a W&B record: `\\x12<len><key>\\x82\\x01<len><value>`.

    The explicit key-length byte is what keeps `...-truncated_ratio` out: it is a
    different, longer key, so its length prefix never equals this one.
    """
    key = metric_key.encode("ascii")
    return re.compile(
        rb"\x12" + bytes([len(key)]) + re.escape(key) + rb"\x82\x01(.)",
        re.DOTALL,
    )


def iter_log_files(root: Path, include_wandb: bool):
    found = []
    if root.is_symlink() or not root.is_dir():
        return found
    for path in root.rglob("*"):
        try:
            if path.is_symlink() or not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix == ".wandb":
                if not include_wandb:
                    continue
            elif suffix not in TEXT_SUFFIXES:
                continue
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        found.append(path)
    return sorted(found, key=lambda path: (path.stat().st_mtime_ns, str(path)))


def read_text(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data[:4096]:
        return None
    return ANSI_RE.sub("", data.decode("utf-8", errors="replace"))


def parse_decimal(token: str) -> Decimal | None:
    try:
        value = Decimal(token)
    except InvalidOperation:
        return None
    return value if value.is_finite() else None


def parse_accuracy_token(token: str) -> Decimal | None:
    value = parse_decimal(token)
    if value is None or value < 0 or value > 1:
        return None
    return value


def basename(raw: str) -> str:
    """Basename of a logged path, tolerating both separators and trailing slashes."""
    return raw.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def announced_wandb_dirs(text: str) -> set[str]:
    """Offline W&B directory names this launch log says it created."""
    names = set()
    for match in WANDB_DIR_RE.finditer(text):
        name = basename(match.group(1))
        if OFFLINE_DIR_RE.match(name):
            names.add(name)
    return names


def extract_text_eval(text: str, pattern: re.Pattern[str]) -> dict[int, Decimal]:
    """Step -> eval value. A later line for a step wins (resumed/duplicated logs)."""
    points: dict[int, Decimal] = {}
    for match in pattern.finditer(text):
        value = parse_decimal(match.group(3))
        if value is not None:
            points[int(match.group(1))] = value
    return points


def extract_wandb_eval(data: bytes, pattern: re.Pattern[bytes]) -> dict[int, Decimal]:
    """Step -> eval value from the binary stream.

    Each point lands twice (history record, then summary record) and only the
    history record carries `eval/step` nearby; summary copies without a step in
    range are dropped rather than guessed at.
    """
    points: dict[int, Decimal] = {}
    for match in pattern.finditer(data):
        length = match.group(1)[0]
        try:
            token = data[match.end() : match.end() + length].decode("ascii")
        except UnicodeDecodeError:
            continue
        value = parse_decimal(token)
        if value is None:
            continue
        step_match = WANDB_STEP_RE.search(
            data, match.end(), match.end() + WANDB_STEP_WINDOW
        )
        if step_match is None:
            continue
        step_length = step_match.group(1)[0]
        try:
            step_token = data[
                step_match.end() : step_match.end() + step_length
            ].decode("ascii")
        except UnicodeDecodeError:
            continue
        if not step_token.isdigit():
            continue
        points[int(step_token)] = value
    return points


def extract_wandb_accuracy(data: bytes, relative: str) -> list[dict[str, Any]]:
    """Chronologically ordered `RLVE/<env>/accuracy` points (file order is time order)."""
    points = []
    for match in WANDB_ACC_RE.finditer(data):
        length = match.group(2)[0]
        try:
            token = data[match.end() : match.end() + length].decode("ascii")
        except UnicodeDecodeError:
            continue
        value = parse_accuracy_token(token)
        if value is None:
            continue
        points.append(
            {
                "env": match.group(1).decode("ascii"),
                "accuracy": float(value),
                "source_log": relative,
            }
        )
    return points


def elect_launch(results: Path, root_name: str, pattern: re.Pattern[str]):
    """Pick the launch to grade: newest text log that has eval lines and names its W&B dirs.

    Requiring both markers is what makes the election meaningful -- a bare file of
    eval lines is not a launch, and a launch with no W&B directories cannot have
    its training reward read.
    """
    root = results / root_name
    elected = None
    candidates = []
    for path in iter_log_files(root, include_wandb=False):
        text = read_text(path)
        if text is None:
            continue
        eval_points = extract_text_eval(text, pattern)
        if not eval_points:
            continue
        wandb_dirs = announced_wandb_dirs(text)
        relative = path.relative_to(results).as_posix()
        candidates.append(relative)
        if not wandb_dirs:
            continue
        # iter_log_files sorts oldest-first, so the last match is the newest.
        elected = (relative, eval_points, wandb_dirs)
    return elected, candidates


def collect_wandb(results: Path, root_name: str, wandb_dirs: set[str], pattern):
    """Read only the `.wandb` streams belonging to the elected launch."""
    eval_points: dict[int, Decimal] = {}
    accuracy: list[dict[str, Any]] = []
    sources: list[str] = []
    root = results / root_name
    for path in iter_log_files(root, include_wandb=True):
        if path.suffix.lower() != ".wandb" or path.parent.name not in wandb_dirs:
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        relative = path.relative_to(results).as_posix()
        found_eval = extract_wandb_eval(data, pattern)
        found_acc = extract_wandb_accuracy(data, relative)
        if not found_eval and not found_acc:
            continue
        eval_points.update(found_eval)
        accuracy.extend(found_acc)
        sources.append(relative)
    return eval_points, accuracy, sources


def tail_mean(points: list[dict[str, Any]], fraction: Decimal):
    """Mean accuracy over the last `fraction` of the run.

    A whole-run mean is the wrong summary here: `RLVE/<env>/accuracy` is a
    curriculum-relative rate that resets on every difficulty promotion, so the
    warm-up phase -- where a healthy run and a stuck run look alike -- drags the
    average toward the stuck run and compresses the gap the gate depends on.
    """
    if not points:
        return None, 0
    count = len(points)
    take = int((Decimal(count) * fraction).to_integral_value(rounding="ROUND_CEILING"))
    take = max(1, min(count, take))
    tail = points[count - take :]
    total = sum(Decimal(str(item["accuracy"])) for item in tail)
    return total / Decimal(len(tail)), len(tail)


def binary_score(passed: bool) -> Decimal:
    return Decimal("1") if passed else Decimal("0")


def grade(spec: dict[str, Any], results: Path):
    metric_key = spec["metric_key"]
    train_key = spec["training_reward_key"]
    root_name = spec["log_roots"][0]
    required_step = int(spec["required_final_step"])
    baseline = Decimal(str(spec["historical_final_baseline"]))
    required_score = Decimal(str(spec["required_final_score"]))
    required_score_max = Decimal(str(spec["required_final_score_max"]))
    min_corroborated = int(spec["min_corroborated_eval_points"])
    tail_fraction = Decimal(str(spec["training_reward_tail_fraction"]))
    required_train = Decimal(str(spec["required_training_reward"]))
    min_train_points = int(spec["min_training_reward_points"])
    validation_weight = Decimal(str(spec["validation_weight"]))
    training_weight = Decimal(str(spec["training_weight"]))

    text_pattern = metric_pattern(metric_key)
    binary_pattern = wandb_eval_pattern(metric_key)
    errors: list[str] = []

    elected, candidates = elect_launch(results, root_name, text_pattern)
    if elected is None:
        if candidates:
            errors.append(
                f"no gradable launch in {root_name}: {len(candidates)} log(s) report "
                f"{metric_key} but none announce their offline W&B directories"
            )
        else:
            errors.append(f"missing {metric_key} in {root_name}")
        launch_log, text_eval, wandb_dirs = None, {}, set()
    else:
        launch_log, text_eval, wandb_dirs = elected

    wandb_eval, train_points, wandb_sources = (
        collect_wandb(results, root_name, wandb_dirs, binary_pattern)
        if wandb_dirs
        else ({}, [], [])
    )

    # Cross-check the elected text log against its own binary stream.
    overlap = sorted(set(text_eval) & set(wandb_eval))
    conflicts = [
        {
            "step": step,
            "text": float(text_eval[step]),
            "wandb": float(wandb_eval[step]),
        }
        for step in overlap
        if abs(text_eval[step] - wandb_eval[step]) > EVAL_AGREEMENT_EPSILON
    ]
    conflicting_steps = {item["step"] for item in conflicts}
    corroborated = [step for step in overlap if step not in conflicting_steps]

    # Every reason the eval half can fail. Collected rather than short-circuited so
    # the report names all of them at once.
    validation_errors: list[str] = []
    if elected is not None:
        if conflicts:
            validation_errors.append(
                f"{len(conflicts)} eval step(s) disagree between {launch_log} and its "
                f"W&B stream (first: step {conflicts[0]['step']}, "
                f"{conflicts[0]['text']} vs {conflicts[0]['wandb']})"
            )
        elif len(corroborated) < min_corroborated:
            validation_errors.append(
                f"only {len(corroborated)} eval point(s) in {launch_log} are "
                f"corroborated by its W&B stream, need {min_corroborated}"
            )

    final_step = max(text_eval) if text_eval else None
    final_value = text_eval[final_step] if final_step is not None else None
    improvement = final_value - baseline if final_value is not None else None

    if final_value is None:
        pass  # already reported by elect_launch
    elif final_step < required_step:
        validation_errors.append(
            f"last eval is step {final_step}, which is short of the required "
            f"step {required_step}; the run did not finish"
        )
    elif final_value < required_score:
        validation_errors.append(
            f"final score {final_value} at step {final_step} is below "
            f"required score {required_score}"
        )
    elif final_value > required_score_max:
        validation_errors.append(
            f"final score {final_value} at step {final_step} exceeds the plausible "
            f"ceiling {required_score_max}"
        )

    errors.extend(validation_errors)
    validation_passed = final_value is not None and not validation_errors

    train_mean, tail_n = tail_mean(train_points, tail_fraction)
    if train_mean is None:
        errors.append(f"missing finite {train_key} in the elected launch's W&B stream")
        training_passed = False
    elif len(train_points) < min_train_points:
        errors.append(
            f"only {len(train_points)} {train_key} point(s) in the elected launch, "
            f"need {min_train_points}; the run is too short to grade"
        )
        training_passed = False
    elif train_mean < required_train:
        errors.append(
            f"tail-{tail_fraction} mean training reward {train_mean} is below "
            f"required reward {required_train}"
        )
        training_passed = False
    else:
        training_passed = True

    validation_score = binary_score(validation_passed)
    training_score = binary_score(training_passed)
    score = float(validation_weight * validation_score + training_weight * training_score)
    passed = score == 1.0
    judged = float(final_value) if final_value is not None else None

    eval_points = [
        {
            "step": step,
            "eval_score": float(text_eval[step]),
            "eval_key": metric_key,
            "wandb_corroborated": step in corroborated,
            "source_log": launch_log,
        }
        for step in sorted(text_eval)
    ]
    source_logs = sorted({*(x for x in [launch_log] if x), *wandb_sources})

    metrics = {
        "task_id": spec["task_id"],
        "metric_key": metric_key,
        "launch_log": launch_log,
        "wandb_dirs": sorted(wandb_dirs),
        "required_final_step": required_step,
        "historical_final_baseline": float(baseline),
        "required_final_score": float(required_score),
        "required_final_score_max": float(required_score_max),
        "final_step": final_step,
        "final_score": judged,
        "absolute_improvement": float(improvement) if improvement is not None else None,
        "corroborated_eval_steps": corroborated,
        "eval_conflicts": conflicts,
        "validation_score": float(validation_score),
        "training_reward_key": train_key,
        "training_reward_tail_fraction": float(tail_fraction),
        "training_reward_mean": float(train_mean) if train_mean is not None else None,
        "training_reward_n": len(train_points),
        "training_reward_tail_n": tail_n,
        "required_training_reward": float(required_train),
        "training_reward_score": float(training_score),
        "source_logs": source_logs,
        "errors": errors,
        "passed": passed,
        "score": score,
        "case_score": score,
        "deterministic": True,
        "judge_used": False,
    }
    experiment = {
        "task_id": spec["task_id"],
        "logs": {"launch_log": launch_log, "source_logs": source_logs},
        "metrics": {
            "metric_key": metric_key,
            "metric_status": "available" if eval_points else "missing",
            "eval_points": eval_points,
            "eval_scores": [
                {"step": item["step"], "score": item["eval_score"]} for item in eval_points
            ],
            "final_step": final_step,
            "final_score": judged,
            "train_reward_key": train_key,
            "train_reward_status": "available" if train_points else "missing",
            "train_reward_n": len(train_points),
            "train_reward_tail_n": tail_n,
            "training_reward_mean": metrics["training_reward_mean"],
            "train_reward_points": train_points,
            "validation_score": metrics["validation_score"],
            "training_reward_score": metrics["training_reward_score"],
            "score": score,
        },
    }
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
        spec = load_spec(
            Path(__file__).resolve().parent.parent / "resources" / "grader_spec.json"
        )
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
