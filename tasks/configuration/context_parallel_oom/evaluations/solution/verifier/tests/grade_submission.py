#!/usr/bin/env python3
"""Grade a context-parallel OOM repair from the resolved training log.

Pass requires max_response_length left at 30720, no CUDA OOM, and one logged
actor update. Any other response length scores 0. TACO accuracy is not graded.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import sys
from pathlib import Path
from typing import Any


ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|[@-_])")
TEXT_SUFFIXES = {".json", ".jsonl", ".log", ".out", ".txt", ""}
LENGTH_RE = re.compile(r"max_response_length['\"]?\s*[:=]\s*(\d+)")
OOM_RE = re.compile(r"OutOfMemoryError|CUDA out of memory|out of memory", re.IGNORECASE)
STEP_RE = re.compile(r"(?:^|\s)step:(\d+)\b")
METRIC_RE = re.compile(r"actor/|pg_loss|critic/rewards")


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
        "required_response_length",
        "log_roots",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("grader_spec.json has unexpected fields")
    if value["schema_version"] != 2:
        raise ValueError("unsupported grader spec")
    if value["log_roots"] != ["experiment_logs"]:
        raise ValueError("log roots must be experiment_logs")
    return value


def log_text(results: Path, roots: list[str]) -> tuple[str, list[str]]:
    chunks = []
    names = []
    for root_name in roots:
        root = results / root_name
        if root.is_symlink() or not root.is_dir():
            continue
        for path in sorted(root.rglob("*"), key=lambda item: str(item)):
            if path.is_symlink() or not path.is_file():
                continue
            if path.suffix.lower() == ".wandb":
                continue
            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            try:
                data = path.read_bytes()
            except OSError:
                continue
            if b"\x00" in data[:4096]:
                continue
            chunks.append(ANSI_RE.sub("", data.decode("utf-8", errors="replace")))
            names.append(path.relative_to(results).as_posix())
    return "\n".join(chunks), names


def grade(spec: dict[str, Any], results: Path) -> dict[str, Any]:
    required_length = int(spec["required_response_length"])
    text, names = log_text(results, spec["log_roots"])
    errors: list[str] = []
    if not text.strip():
        errors.append("missing experiment_logs")
        return _result(spec, errors, names, {})

    lengths = [int(item) for item in LENGTH_RE.findall(text)]
    if not lengths:
        errors.append("missing max_response_length in experiment_logs")
    elif any(item != required_length for item in lengths):
        errors.append(
            f"max_response_length must stay {required_length}; found {sorted(set(lengths))}"
        )

    if OOM_RE.search(text):
        errors.append("CUDA OOM in experiment_logs")
    if not (STEP_RE.search(text) and METRIC_RE.search(text)):
        errors.append("actor update was not logged")

    observed = {"max_response_length": sorted(set(lengths))}
    return _result(spec, errors, names, observed)


def _result(
    spec: dict[str, Any],
    errors: list[str],
    names: list[str],
    observed: dict[str, Any],
) -> dict[str, Any]:
    passed = not errors
    score = 1.0 if passed else 0.0
    metrics = {
        "task_id": spec["task_id"],
        "required_response_length": spec["required_response_length"],
        "observed": observed,
        "source_logs": names,
        "errors": errors,
        "passed": passed,
        "score": score,
        "case_score": score,
        "deterministic": True,
        "judge_used": False,
    }
    return {"passed": passed, "score": score, "metrics": metrics}


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
        result = grade(spec, results)
        write_json(results / "metrics.json", result["metrics"])
        write_json(results / "result.json", result)
        if result["metrics"]["errors"]:
            print("; ".join(result["metrics"]["errors"]), file=sys.stderr)
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
