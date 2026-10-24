#!/usr/bin/env python3
"""Grade a fresh fully-async GRPO rerun for the abort-eviction task.

Two signals, both read only from $PTDBENCH_RESULTS/experiment_logs:

  * train-step throughput (primary): how many training steps the fixed
    wall-clock budget completed, counted from the driver log's ``rollout N:``
    lines. The defect keeps interrupted in-flight requests alive across each
    weight sync, so the engine wastes full-length decode on trajectories that
    are later dropped as stale, and far fewer steps finish in the same wall
    clock (historical 92 defective vs 148 control).
  * last-window solve rate (secondary): the mean fraction of samples with a
    correct answer (``reward.acc``) over the last ``solve_window_frac`` of the
    dumped per-step rollout data. It is throughput-driven -- more completed
    steps means more learning by the end of the fixed window -- so it corroborates
    the throughput signal rather than standing alone.

Pass requires BOTH thresholds. Score is 1.0 or 0.0.
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
# matches e.g. "log_utils.py:66 - rollout 17: {'rollout/response_lengths': ...}"
STEP_RE = re.compile(r"\brollout\s+(\d+):\s*\{", re.IGNORECASE)
MAX_FILE_BYTES = 256 * 1024 * 1024
REQUIRED_SPEC_KEYS = {
    "schema_version",
    "task_id",
    "log_roots",
    "step_metric_key",
    "required_train_steps",
    "solve_rate_metric_key",
    "solve_window_frac",
    "required_solve_rate",
}


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
    if not isinstance(value, dict):
        raise ValueError("grader_spec.json must be an object")
    if not REQUIRED_SPEC_KEYS <= set(value):
        raise ValueError("grader_spec.json is missing required fields")
    if value["schema_version"] != 1:
        raise ValueError("unsupported grader spec")
    if value["log_roots"] != ["experiment_logs"]:
        raise ValueError("log roots must be experiment_logs")
    return value


def _dec(value: Any) -> Decimal:
    try:
        return Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"not a number: {value!r}") from exc


def count_train_steps(results: Path, roots: list[str]) -> tuple[int, list[str]]:
    """Largest ``rollout N`` index + 1 across all text logs under the roots."""
    seen: set[int] = set()
    sources: list[str] = []
    for root_name in roots:
        root = results / root_name
        if root.is_symlink() or not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_symlink() or not path.is_file():
                continue
            if path.suffix.lower() not in {".log", ".txt", ".out", ""}:
                continue
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
                data = path.read_bytes()
            except OSError:
                continue
            if b"\x00" in data[:4096]:
                continue
            text = ANSI_RE.sub("", data.decode("utf-8", errors="replace"))
            hits = [int(m.group(1)) for m in STEP_RE.finditer(text)]
            if hits:
                seen.update(hits)
                sources.append(path.relative_to(results).as_posix())
    steps = (max(seen) + 1) if seen else 0
    return steps, sorted(set(sources))


def _iter_samples(node: Any):
    if isinstance(node, list):
        for child in node:
            yield from _iter_samples(child)
    else:
        yield node


def solve_rate_window(
    results: Path, roots: list[str], window_frac: Decimal
) -> tuple[Decimal | None, dict[str, Any]]:
    """Mean solve rate over the last ``window_frac`` of dumped rollout steps."""
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:
        raise ValueError(f"torch unavailable for solve-rate grading: {exc}") from exc

    pt_files: list[Path] = []
    for root_name in roots:
        root = results / root_name
        if root.is_symlink() or not root.is_dir():
            continue
        for path in root.rglob("dump/rollout_data/*.pt"):
            if path.is_file() and not path.is_symlink():
                pt_files.append(path)

    def step_of(path: Path) -> int:
        try:
            return int(path.stem)
        except ValueError:
            return -1

    pt_files = sorted((p for p in pt_files if step_of(p) >= 0), key=step_of)
    per_step: list[dict[str, Any]] = []
    for path in pt_files:
        try:
            dumped = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:  # noqa: BLE001 - a corrupt dump should not crash grading
            continue
        samples = dumped.get("samples") if isinstance(dumped, dict) else None
        if samples is None:
            continue
        accs = []
        for sample in _iter_samples(samples):
            if not isinstance(sample, dict):
                continue
            reward = sample.get("reward")
            if isinstance(reward, dict):
                accs.append(1.0 if reward.get("acc") else 0.0)
        if accs:
            per_step.append({"step": step_of(path), "n": len(accs),
                             "solve_rate": sum(accs) / len(accs)})

    info: dict[str, Any] = {
        "dump_files": len(pt_files),
        "scored_steps": len(per_step),
        "window_frac": float(window_frac),
    }
    if not per_step:
        info["window_steps"] = []
        return None, info
    k = max(1, int(len(per_step) * float(window_frac)))
    window = per_step[-k:]
    info["window_steps"] = [s["step"] for s in window]
    mean = Decimal(sum(s["solve_rate"] for s in window)) / Decimal(len(window))
    return mean, info


def grade(spec: dict[str, Any], results: Path):
    roots = spec["log_roots"]
    required_steps = int(_dec(spec["required_train_steps"]))
    required_solve = _dec(spec["required_solve_rate"])
    window_frac = _dec(spec["solve_window_frac"])

    train_steps, step_sources = count_train_steps(results, roots)
    solve_rate, solve_info = solve_rate_window(results, roots, window_frac)

    errors: list[str] = []
    if train_steps <= 0:
        errors.append(
            f"no {spec['step_metric_key']} found in experiment_logs "
            "(expected driver log 'rollout N:' lines)"
        )
    elif train_steps < required_steps:
        errors.append(
            f"train_steps {train_steps} is below required {required_steps}"
        )
    if solve_rate is None:
        errors.append(
            f"no {spec['solve_rate_metric_key']} found "
            "(expected dump/rollout_data/*.pt with reward.acc)"
        )
    elif solve_rate < required_solve:
        errors.append(
            f"solve_rate {solve_rate} is below required {required_solve}"
        )

    passed = not errors
    score = 1.0 if passed else 0.0
    metrics = {
        "task_id": spec["task_id"],
        "step_metric_key": spec["step_metric_key"],
        "train_steps": train_steps,
        "required_train_steps": required_steps,
        "train_steps_passed": train_steps >= required_steps and train_steps > 0,
        "solve_rate_metric_key": spec["solve_rate_metric_key"],
        "solve_rate": float(solve_rate) if solve_rate is not None else None,
        "required_solve_rate": float(required_solve),
        "solve_rate_passed": solve_rate is not None and solve_rate >= required_solve,
        "solve_rate_info": solve_info,
        "step_source_logs": step_sources,
        "errors": errors,
        "passed": passed,
        "score": score,
        "case_score": score,
        "deterministic": True,
        "judge_used": False,
    }
    experiment = {
        "task_id": spec["task_id"],
        "logs": {"source_logs": step_sources},
        "metrics": {
            "train_steps": train_steps,
            "solve_rate_last_window": float(solve_rate) if solve_rate is not None else None,
            "solve_rate_info": solve_info,
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
