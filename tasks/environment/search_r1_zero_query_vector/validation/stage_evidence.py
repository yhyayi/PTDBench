#!/usr/bin/env python3
"""Stage the frozen defective-run evidence into the task workspace.

Slims the 37MB results/baseline32/ tree to the files a diagnosis actually needs
and rewrites the strings that would name the answer (the original run lived in a
directory called `broken`, next to one called `fixed`). Writes a report of every
substitution so environment/source_manifest.json can record it.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

RAW = Path("/Users/chenyinghao/workspace/PTDBench/raw_tasks/hsz_search-r1-delivery")
SRC = RAW / "workspace_broken/results/baseline32"
DST = Path(
    "/Users/chenyinghao/workspace/PTDBench/tasks/search_r1_zero_query_vector"
    "/workspace/results/baseline32"
)

# Longest first: the run directory must be rewritten before the generic prefix.
SUBSTITUTIONS = [
    (
        "/mnt/data/hsz/RSI_case/runs/hotpotqa-comparison-lr1e6-b32-s42-32steps/broken",
        "/workspace/runs/baseline32",
    ),
    (
        "/mnt/data/hsz/RSI_case/agent/Search-R1/runtime/train/.venv",
        "/opt/search-r1-runtime/.venv",
    ),
    ("/mnt/data/hsz/RSI_case/agent/Search-R1", "/workspace/Search-R1"),
    ("/mnt/data/hsz/RSI_case/agent/models", "/workspace/shared/models"),
    ("/mnt/data/hsz/RSI_case/data/hotpotqa", "/workspace/shared/data"),
    ("/mnt/data/hsz/RSI_case", "/workspace"),
    ("/home/hsz/", "/root/"),
    ("hotpotqa-broken-lr1e6-b32-s42-32steps", "hotpotqa-lr1e6-b32-s42-32steps"),
    # retrieval_server.log's banner line labelled which half of the pair it was.
    ("Retriever mode=broken, source=", "Retriever source="),
    ("127.0.0.1:8004", "127.0.0.1:8000"),
    ("219.223.251.156", "127.0.0.1"),
    ('"username": "hsz"', '"username": "root"'),
]

TEXT_FILES = [
    "train.log",
    "launch.log",
    "retrieval_server.log",
]
ROLLOUT_STEPS = [1, 2, 16, 32]
WANDB_RUN = "wandb/wandb/offline-run-20261014_111535-tojxuay5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize(text: str, counts: dict[str, int]) -> str:
    for needle, replacement in SUBSTITUTIONS:
        hits = text.count(needle)
        if hits:
            counts[needle] = counts.get(needle, 0) + hits
            text = text.replace(needle, replacement)
    return text


def copy_text(relative: str, report: list[dict], counts: dict[str, int]) -> None:
    source = SRC / relative
    target = DST / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    original = source.read_text(encoding="utf-8", errors="surrogateescape")
    cleaned = sanitize(original, counts)
    target.write_text(cleaned, encoding="utf-8", errors="surrogateescape")
    report.append(
        {
            "path": f"results/baseline32/{relative}",
            "action": "copied" if cleaned == original else "path-sanitized",
            "sha256_before": sha256(source),
            "sha256_after": sha256(target),
        }
    )


def stage_retrieval_trace(report: list[dict], counts: dict[str, int]) -> None:
    """Keep eight of the 132 batched retrieval records (~94KB each)."""
    lines = (SRC / "retrieval.jsonl").read_text(encoding="utf-8").splitlines()
    flags = [bool(json.loads(line).get("validation")) for line in lines]
    validation = [i for i, flag in enumerate(flags) if flag]
    training = [i for i, flag in enumerate(flags) if not flag]
    keep = sorted(
        set(validation[:2] + validation[-2:] + training[:2] + training[-2:])
    )
    target = DST / "retrieval.jsonl"
    target.write_text(
        sanitize("\n".join(lines[i] for i in keep) + "\n", counts), encoding="utf-8"
    )
    report.append(
        {
            "path": "results/baseline32/retrieval.jsonl",
            "action": f"slimmed to records {keep} of {len(lines)}; paths sanitized",
            "sha256_before": sha256(SRC / "retrieval.jsonl"),
            "sha256_after": sha256(target),
        }
    )


def stage_experiment_json(report: list[dict]) -> None:
    """Drop the two fields and the maintainer hashes that name the answer."""
    source = SRC / "experiment.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    removed = {
        key: data.pop(key) for key in ("mode", "validation_retriever") if key in data
    }
    data["input_sha256"] = {
        re.sub(r"^data/hotpotqa/", "shared/data/", key).replace(
            "agent/models/", "shared/models/"
        ): value
        for key, value in data["input_sha256"].items()
    }
    kept_sources = {}
    for key, value in data["source_sha256"].items():
        if key.startswith("maintainer/"):
            removed[f"source_sha256[{key}]"] = value
            continue
        kept_sources[key.replace("agent/Search-R1/", "Search-R1/")] = value
    data["source_sha256"] = kept_sources
    # The index is rebuilt from the corpus at prepare time rather than carried in
    # the package (environment/build_index.py), so the hash recorded above for
    # e5_Flat.index belongs to the file this run used and will not match a
    # rebuild. Say so here: otherwise the first person to hash the staged tree
    # concludes the data is corrupt.
    data["input_sha256_note"] = (
        "Hashes as recorded by this run. shared/data/{train,test}.parquet and "
        "corpus.jsonl still match. shared/data/e5_Flat.index and the model "
        "weights are fetched or rebuilt by environment/prepare.sh; the index is "
        "rebuilt from corpus.jsonl and is semantically but not bitwise the file "
        "hashed here."
    )
    target = DST / "experiment.json"
    target.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report.append(
        {
            "path": "results/baseline32/experiment.json",
            "action": "rewritten: dropped "
            + ", ".join(sorted(removed))
            + "; asset keys re-rooted to workspace-relative paths",
            "sha256_before": sha256(source),
            "sha256_after": sha256(target),
        }
    )


def stage_rollouts(report: list[dict], counts: dict[str, int]) -> None:
    (DST / "rollouts").mkdir(parents=True, exist_ok=True)
    kept, dropped = [], []
    for path in sorted(SRC.glob("rollouts/metrics_*.json")):
        shutil.copy2(path, DST / "rollouts" / path.name)
        kept.append(path.name)
    for path in sorted(SRC.glob("rollouts/step_*.jsonl")):
        step = int(path.stem.split("_")[1])
        if step in ROLLOUT_STEPS:
            text = sanitize(path.read_text(encoding="utf-8"), counts)
            (DST / "rollouts" / path.name).write_text(text, encoding="utf-8")
            kept.append(path.name)
        else:
            dropped.append(path.name)
    report.append(
        {
            "path": "results/baseline32/rollouts/",
            "action": (
                f"kept all {len([n for n in kept if n.startswith('metrics_')])} "
                f"metrics_*.json and step_*.jsonl for steps {ROLLOUT_STEPS}; "
                f"dropped {len(dropped)} other step_*.jsonl (18MB of trajectories)"
            ),
        }
    )


def stage_wandb(report: list[dict], counts: dict[str, int]) -> None:
    for relative in (
        f"{WANDB_RUN}/files/requirements.txt",
        f"{WANDB_RUN}/files/wandb-metadata.json",
        f"{WANDB_RUN}/files/output.log",
        f"{WANDB_RUN}/logs/debug.log",
        f"{WANDB_RUN}/logs/debug-core.log",
        f"{WANDB_RUN}/logs/debug-internal.log",
        "wandb/wandb/debug.log",
        "wandb/wandb/debug-internal.log",
    ):
        copy_text(relative, report, counts)
    report.append(
        {
            "path": f"results/baseline32/{WANDB_RUN}/run-tojxuay5.wandb",
            "action": (
                "dropped: 3.4MB binary metric history cannot be path-sanitized, and "
                "the same curve is in train.log, rollouts/metrics_*.json and "
                "metrics_summary.json"
            ),
        }
    )
    report.append(
        {
            "path": "results/baseline32/wandb/wandb/latest-run/",
            "action": "dropped: byte-identical duplicate of the offline-run directory",
        }
    )


def stage_metrics_summary(report: list[dict]) -> None:
    """Machine-readable copy of the staged run's held-out curve."""
    comparison = json.loads(
        (RAW / "shared/baseline_evidence/comparison.json").read_text(encoding="utf-8")
    )
    points = []
    for entry in comparison["broken"]["curve"]:
        metrics = entry["metrics"]
        points.append(
            {
                "step": entry["step"],
                "val/em": metrics["val/em"],
                "val/f1": metrics["val/f1"],
                "val/format": metrics["val/format"],
                "val/searched": metrics["val/searched"],
                "val/count": metrics["val/count"],
                "decoding": entry["decoding"],
            }
        )
    payload = {
        "schema_version": 1,
        "run": "baseline32",
        "metric_key": "val/em",
        "total_training_steps": 32,
        "eval_every": 10,
        "eval_steps": [point["step"] for point in points],
        "note": (
            "Held-out curve of this workspace's completed run: the same 128 held-out "
            "HotpotQA questions at every eval, greedy decoding, no gradients. "
            "Reconstructed from the trainer's own step lines in train.log."
        ),
        "points": points,
    }
    target = DST / "metrics_summary.json"
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report.append(
        {
            "path": "results/baseline32/metrics_summary.json",
            "action": "added: machine-readable held-out curve of the staged run",
            "sha256_after": sha256(target),
        }
    )


def main() -> int:
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)
    report: list[dict] = []
    counts: dict[str, int] = {}
    for name in TEXT_FILES:
        copy_text(name, report, counts)
    stage_experiment_json(report)
    stage_retrieval_trace(report, counts)
    stage_rollouts(report, counts)
    stage_wandb(report, counts)
    stage_metrics_summary(report)
    out = Path("/tmp/ptdbench_stage/evidence_report.json")
    out.write_text(
        json.dumps(
            {
                "substitution_counts": counts,
                "substitutions": [
                    {"from": needle, "to": replacement}
                    for needle, replacement in SUBSTITUTIONS
                ],
                "files": report,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(counts, indent=2))
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
