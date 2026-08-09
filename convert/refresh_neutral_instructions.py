#!/usr/bin/env python3
"""Refresh only Agent-visible instructions without rebuilding case workspaces."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

import convert_qwen_dapo_hparam as qwen_hparam
import convert_rlve_hyper as rlve
import convert_verl_implementation as verl_impl


DATA_CALL = Path(__file__).resolve().parents[1]
BE_ROOT = DATA_CALL.parent / "Benchmark_Evaluation"
CASES = DATA_CALL / "cases"


def read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_instruction(path: Path, text: str) -> None:
    if not path.parent.is_dir():
        raise FileNotFoundError(path.parent)
    path.write_text(text, encoding="utf-8")


def refresh_rlve() -> int:
    count = 0
    for task_id in rlve.TASK_IDS:
        source = BE_ROOT / "tasks/task_RLVE_hyper" / task_id
        case = CASES / "rlve_hyper" / task_id
        public = read_yaml(source / "task.yaml")
        manifest = read_json(
            case / "workspace/historical_evidence/manifest.json"
        )
        write_instruction(
            case / "instruction.md",
            rlve.render_instruction(task_id, public, manifest),
        )
        count += 1
    return count


def refresh_qwen_hparam() -> int:
    count = 0
    for task_id in qwen_hparam.TASK_IDS:
        source = BE_ROOT / "tasks/task_qwen_dapo_hparam" / task_id
        case = CASES / "qwen_dapo_hparam" / task_id
        public = read_yaml(source / "task.yaml")
        metrics = read_json(
            case / "workspace/historical_evidence/manifest.json"
        )
        write_instruction(
            case / "instruction.md",
            qwen_hparam.render_instruction(task_id, public, metrics),
        )
        count += 1
    return count


def refresh_verl_implementation() -> int:
    count = 0
    for family, config in verl_impl.FAMILIES.items():
        for task_id in config["tasks"]:
            source = (
                BE_ROOT
                / "tasks"
                / config["source_category"]
                / task_id
            )
            case = CASES / family / task_id
            public = read_yaml(source / "task.yaml")
            hidden = read_yaml(source / "hidden/task.yaml")
            symbols = verl_impl.target_symbols(public, hidden)
            manifest = read_json(
                case / "workspace/historical_evidence/manifest.json"
            )
            write_instruction(
                case / "instruction.md",
                verl_impl.render_instruction(
                    task_id,
                    family,
                    config,
                    public,
                    symbols,
                    manifest["metrics"],
                ),
            )
            count += 1
    return count


def main() -> int:
    counts = {
        "rlve_hyper": refresh_rlve(),
        "qwen_dapo_hparam": refresh_qwen_hparam(),
        "verl_implementation": refresh_verl_implementation(),
    }
    print(json.dumps({"counts": counts, "total": sum(counts.values())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
