#!/usr/bin/env python3
"""Remove legacy package/hash validation artifacts from materialized tasks."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import yaml

try:
    from .write_standard_answers import write_one
except ImportError:
    from write_standard_answers import write_one


VALIDATION_FILES = {
    "checkpoint_exclusion_manifest.json",
    "scoring_rubric.yaml",
    "target_manifest.json",
}
VALIDATION_DIRECTORIES = {
    "source_snapshot",
}
MINIMAL_TEST_SH = """#!/usr/bin/env bash
set -euo pipefail
echo "No built-in verifier is configured; use the external attribution judge." >&2
exit 2
"""


def is_hash_key(key: Any) -> bool:
    normalized = str(key).lower()
    return (
        "sha256" in normalized
        or normalized in {"hash", "hashes"}
        or normalized.endswith("_hash")
        or normalized.endswith("_hashes")
    )


def strip_hash_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: strip_hash_fields(item)
            for key, item in value.items()
            if not is_hash_key(key)
        }
    if isinstance(value, list):
        return [strip_hash_fields(item) for item in value]
    return value


def clean_structured_file(path: Path) -> None:
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(
            json.dumps(strip_hash_fields(value), indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
    elif path.suffix in {".yaml", ".yml"}:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        path.write_text(
            yaml.safe_dump(
                strip_hash_fields(value),
                sort_keys=False,
                allow_unicode=True,
                width=100000,
            ),
            encoding="utf-8",
        )


def remove_empty_directories(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass


def finalize_task(task_root: Path) -> None:
    oracle = task_root / "oracle"
    verifier = task_root / "verifier"
    visible_evidence = task_root / "workspace/historical_evidence"

    for name in VALIDATION_DIRECTORIES:
        path = oracle / name
        if path.exists():
            shutil.rmtree(path)

    for path in sorted(oracle.rglob("*")):
        if not path.is_file():
            continue
        if "hash" in path.name.lower() or path.name in VALIDATION_FILES:
            path.unlink()

    for path in sorted(oracle.rglob("*")):
        if path.is_file() and path.suffix in {".json", ".yaml", ".yml"}:
            clean_structured_file(path)
    remove_empty_directories(oracle)

    if visible_evidence.exists():
        for path in sorted(visible_evidence.rglob("*")):
            if path.is_file() and path.suffix in {".json", ".yaml", ".yml"}:
                clean_structured_file(path)

    if verifier.exists():
        shutil.rmtree(verifier)
    verifier.mkdir(parents=True)
    test_sh = verifier / "test.sh"
    test_sh.write_text(MINIMAL_TEST_SH, encoding="utf-8")
    os.chmod(test_sh, 0o755)

    write_one(task_root)


def finalize_all(tasks_root: Path) -> int:
    tasks = sorted(path.parent for path in tasks_root.glob("*/*/metadata.yaml"))
    for task_root in tasks:
        finalize_task(task_root)
    return len(tasks)


if __name__ == "__main__":
    here = Path(__file__).resolve()
    count = finalize_all(here.parents[1] / "tasks")
    print(f"finalized {count} tasks")
