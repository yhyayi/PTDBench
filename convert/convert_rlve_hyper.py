#!/usr/bin/env python3
"""Materialize the selected RLVE hyperparameter tasks as PTDBench B0 cases."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import stat
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

try:
    from .strip_validation_artifacts import finalize_case
except ImportError:
    from strip_validation_artifacts import finalize_case

try:
    from .sanitize_packaging import (
        align_outputs_metrics,
        clean_agent_outputs,
        redact_runtime_secrets,
        strip_attribution_annotations,
    )
except ImportError:
    from sanitize_packaging import (
        align_outputs_metrics,
        clean_agent_outputs,
        redact_runtime_secrets,
        strip_attribution_annotations,
    )


TASK_IDS = (
    "task_001",
    "task_002",
    "task_003",
    "task_005",
    "task_006",
    "task_007",
    "task_008",
    "task_009",
    "task_010",
    "task_012",
    "task_015",
    "task_016",
    "task_018",
    "task_019",
    "task_021",
)
METRIC_KEY = "eval/HELD-OUT_ENVIRONMENTS_128"
EVAL_RE = re.compile(r"eval\s+(\d+)\s*:\s*(\{.*\})", re.IGNORECASE)
SHORT_CURVE_TASKS = {"task_002", "task_003", "task_018", "task_021"}
LABEL_REVIEW_TASKS = {"task_007", "task_010", "task_015"}
TRUNCATE_PRIMARY_TASKS = {"task_005", "task_010", "task_021"}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve()
    default_data_call = here.parents[1]
    default_root = default_data_call.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--be-root",
        type=Path,
        default=default_root / "Benchmark_Evaluation",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=default_data_call / "cases" / "rlve_hyper",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_yaml(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            data,
            sort_keys=False,
            allow_unicode=True,
            width=100000,
        ),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def copy_ignore(_directory: str, names: List[str]) -> set:
    ignored = set()
    for name in names:
        if name in {".git", "__pycache__"}:
            ignored.add(name)
        elif name.endswith((".pyc", ".pyo", ".lock")):
            ignored.add(name)
        elif name.startswith("debug-core"):
            ignored.add(name)
    return ignored


def copy_workspace(source: Path, target: Path) -> None:
    shutil.copytree(
        source,
        target,
        symlinks=False,
        ignore=copy_ignore,
    )
    (target / "outputs").mkdir(parents=True, exist_ok=True)


def log_candidates(workspace: Path) -> Iterable[Path]:
    for root_name in ("wandb", "logs"):
        root = workspace / root_name
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.log")):
            if root_name == "wandb" and not (
                path.name == "output.log" or path.name.startswith("output_")
            ):
                continue
            yield path


def parse_eval_payload(raw: str) -> Dict[str, Any]:
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def collect_historical_metrics(workspace: Path) -> Dict[str, Any]:
    by_step: Dict[int, Dict[str, Any]] = {}
    source_logs: Dict[str, str] = {}
    conflicts: List[Dict[str, Any]] = []
    for path in log_candidates(workspace):
        text = path.read_text(encoding="utf-8", errors="ignore")
        matched = False
        for match in EVAL_RE.finditer(text):
            payload = parse_eval_payload(match.group(2))
            if METRIC_KEY not in payload:
                continue
            matched = True
            step = int(match.group(1))
            score = float(payload[METRIC_KEY])
            truncate = payload.get(
                METRIC_KEY + "-truncated_ratio",
                payload.get(METRIC_KEY + "-truncate_ratio"),
            )
            point = {
                "step": step,
                "eval_score": score,
                "eval_key": METRIC_KEY,
                "truncate_ratio": None if truncate is None else float(truncate),
            }
            previous = by_step.get(step)
            if previous is not None and previous != point:
                conflicts.append(
                    {
                        "step": step,
                        "previous": previous,
                        "replacement": point,
                        "source": str(path.relative_to(workspace)),
                    }
                )
            by_step[step] = point
        if matched:
            relative = str(path.relative_to(workspace))
            source_logs[relative] = sha256_file(path)
    if conflicts:
        raise RuntimeError(f"historical metric conflicts: {conflicts[:3]}")
    points = [by_step[step] for step in sorted(by_step)]
    filtered = [
        point
        for point in points
        if point["step"] % 20 in {0, 19}
    ] or points
    if not filtered:
        raise RuntimeError(f"no {METRIC_KEY} points found under {workspace}")
    scores = [point["eval_score"] for point in filtered]
    truncates = [
        point["truncate_ratio"]
        for point in filtered
        if point["truncate_ratio"] is not None
    ]
    return {
        "metric_key": METRIC_KEY,
        "eval_interval": 20,
        "n_eval_points": len(filtered),
        "eval_points": filtered,
        "eval_first": scores[0],
        "eval_last": scores[-1],
        "eval_min": min(scores),
        "eval_max": max(scores),
        "truncate_ratio_max": max(truncates) if truncates else None,
        "source_logs": [
            {"path": path, "sha256": source_logs[path]}
            for path in sorted(source_logs)
        ],
        "conflicts": [],
    }


def failure_signature(task_id: str, metrics: Dict[str, Any]) -> str:
    if task_id in {"task_005", "task_010", "task_021"}:
        return "eval_truncation"
    if task_id == "task_006":
        return "rollout_truncation"
    if task_id in SHORT_CURVE_TASKS:
        return "early_run"
    if task_id in LABEL_REVIEW_TASKS:
        return "boundary_metric_failure"
    if metrics["eval_min"] <= -0.95:
        return "collapse"
    return "metric_failure"


def load_materialize_parser(be_root: Path):
    sys.path.insert(0, str(be_root))
    from packaging.materialize import parse_rlve_sh_hparams

    return parse_rlve_sh_hparams


def historical_evidence(
    task_id: str,
    workspace: Path,
    hidden: Dict[str, Any],
    parse_rlve_sh_hparams,
) -> Dict[str, Any]:
    held_out = collect_historical_metrics(workspace)
    script = workspace / "scripts/training/Qwen2.5-0.5B-Instruct/rlve.sh"
    materialized_hparams = parse_rlve_sh_hparams(str(script))
    raw_metrics_path = workspace.parent / "oracle" / "failed_metrics.json"
    raw_metrics = (
        json.loads(raw_metrics_path.read_text(encoding="utf-8"))
        if raw_metrics_path.exists()
        else {}
    )
    return {
        "schema_version": 1,
        "evidence_kind": "historical_failure",
        "task_id": task_id,
        "source_run": hidden.get("source_run"),
        "failure_signature": failure_signature(task_id, held_out),
        "held_out_eval": held_out,
        "raw_reward": raw_metrics,
        "effective_hparams_visible": hidden.get("oracle", {}).get(
            "effective_hparams", {}
        ),
        "materialized_hparams": materialized_hparams,
    }


def render_instruction(
    task_id: str,
    public: Dict[str, Any],
    metrics: Dict[str, Any],
) -> str:
    train_envs = "\n".join(
        f"- {name}" for name in public.get("locked_environment_list", [])
    )
    heldout_envs = "\n".join(
        f"- {name}" for name in public.get("locked_heldout_environment_list", [])
    )
    return f"""# Historical RLVE run health assessment

Assess the historical RLVE/slime GRPO run using Qwen2.5-0.5B-Instruct.
Determine from the workspace evidence whether the run is healthy, abnormal,
incomplete, or indeterminate. Do not assume in advance that any particular
setting, file, subsystem, or prior change is responsible.

Inspect:

- `logs/` and `wandb/` for the original run evidence;
- `historical_evidence/` for the frozen evaluation record;
- launch scripts, effective configuration, source code, and environment
  implementations where relevant.

Use `{METRIC_KEY}` together with rollout, truncation, runtime, and training
signals. Training reward alone does not establish held-out health.

## Required deliverables

1. Write `outputs/diagnosis.txt` in English. State a verdict
   (`healthy`, `abnormal`, `incomplete`, or `indeterminate`), cite evidence,
   explain the causal chain, and separate root causes from symptoms.
2. If and only if the evidence supports a repair, make the minimal necessary
   change. Do not change files merely because they look unusual.
3. Write an executable `outputs/run_command.sh` that reproduces or validates
   your conclusion using a fresh run name and `--eval-interval 20`.
4. Do not launch the full training job yourself; the verifier owns execution.
5. Do not modify historical `logs/**`, `wandb/**`, or held-out data/code.

## Locked training environments

{train_envs}

## Locked held-out environments

{heldout_envs}

You may edit training scripts, slime Python code, training Gym implementations,
`outputs/**`, and `experiement/**`. Do not add, remove, or rename locked
environment types, and do not retarget held-out evaluation data.
"""


def metadata_for(
    task_id: str,
    public: Dict[str, Any],
    hidden: Dict[str, Any],
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    oracle = hidden.get("oracle", {})
    held = manifest["held_out_eval"]
    label_review = task_id in LABEL_REVIEW_TASKS
    return {
        "case_id": task_id,
        "title": f"RLVE historical diagnosis: {task_id}",
        "submitter": {"name": "benchmark_evaluation"},
        "license": "MIT",
        "created_date": "2026-08-08",
        "domain": "post_train",
        "model": {"name": public.get("model", "Qwen2.5-0.5B-Instruct")},
        "dataset": {"name": "RLVE held-out environments"},
        "training": {
            "framework": public.get("framework", "RLVE / slime + GRPO"),
            "algo": "grpo",
            "run_command": "bash outputs/run_command.sh",
        },
        "environment": {"dockerfile": "environment/Dockerfile"},
        "outcome": {
            "failure_type": "training_collapse",
            "failure_reason": hidden.get("oracle", {}).get(
                "primary_cause", "Historical held-out metric failure"
            ),
        },
        "bench_category": "rlve_hyper",
        "bench_subcategory": public.get("subcategory", "collapse"),
        "source_task": f"Benchmark_Evaluation/tasks/task_RLVE_hyper/{task_id}",
        "source_run": hidden.get("source_run"),
        "evidence_kind": "historical_failure",
        "packaging_tier": "B0",
        "packaging_ready": True,
        "b1_scoring_ready": False,
        "label_review": label_review,
        "bug_type": hidden.get("bug_type"),
        "buggy_keys": ",".join(oracle.get("buggy_keys", [])),
        "verify": {
            "num_gpus": int(public.get("verify", {}).get("num_gpus", 4)),
            "timeout_minutes": int(
                public.get("verify", {}).get("timeout_minutes", 90)
            ),
            "eval_interval": 20,
            "eval_metric_key": METRIC_KEY,
        },
        "historical_baseline": {
            "eval_last": held["eval_last"],
            "eval_min": held["eval_min"],
            "truncate_ratio_max": held["truncate_ratio_max"],
            "n_eval_points": held["n_eval_points"],
        },
        "failure_signature": manifest["failure_signature"],
    }


def scoring_rubric(task_id: str, manifest: Dict[str, Any]) -> Dict[str, Any]:
    min_points = manifest["held_out_eval"]["n_eval_points"]
    if task_id in SHORT_CURVE_TASKS:
        min_points = min(min_points, 2)
    return {
        "schema_version": 1,
        "eval_metric": METRIC_KEY,
        "eval_interval": 20,
        "b0_structural_only": True,
        "b1_requires_gpu": True,
        "b1_requires_healthy_baseline": task_id in {"task_005", "task_006"},
        "pass_requires": {
            "deliverables_ok": True,
            "protected_evidence_unchanged": True,
            "hparams_match_golden": True,
            "first_rollout_ok": True,
            "metric_repaired": True,
        },
        "min_eval_points": min_points,
        "truncate_primary": task_id in TRUNCATE_PRIMARY_TASKS,
        "raw_reward_primary": False,
        "label_review": task_id in LABEL_REVIEW_TASKS,
    }


def protected_hashes(workspace: Path, public: Dict[str, Any]) -> Dict[str, str]:
    paths: set = set()
    for root_name in ("logs", "wandb", "data/HELD-OUT_ENVIRONMENTS"):
        root = workspace / root_name
        paths.update(iter_files(root) or [])
    gym_root = workspace / "Gym/environments"
    for name in public.get("locked_heldout_environment_list", []):
        paths.update(iter_files(gym_root / name) or [])
    return {
        str(path.relative_to(workspace)): sha256_file(path)
        for path in sorted(paths)
    }


def render_readme(
    task_id: str,
    hidden: Dict[str, Any],
    manifest: Dict[str, Any],
) -> str:
    oracle = hidden.get("oracle", {})
    return f"""# {task_id}

PTDBench B0 package generated from a Benchmark_Evaluation RLVE historical run.

- Source run: `{hidden.get("source_run")}`
- Failure signature: `{manifest["failure_signature"]}`
- Buggy keys: `{", ".join(oracle.get("buggy_keys", []))}`
- Packaging tier: B0 (structure and frozen evidence validated)
- B1: requires local 4-GPU Phase2 execution and metric gate calibration

The Agent sees `workspace/` and `instruction.md`; `oracle/` and `verifier/` are
verifier-only.
"""


def verifier_script() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail
echo "No built-in verifier is configured; use the external attribution judge." >&2
exit 2
"""


def verifier_python() -> str:
    return ""




def materialize_one(
    task_id: str,
    be_root: Path,
    destination_root: Path,
    force: bool,
    parse_rlve_sh_hparams,
) -> Dict[str, Any]:
    source = be_root / "tasks/task_RLVE_hyper" / task_id
    target = destination_root / task_id
    if target.exists():
        if not force:
            raise FileExistsError(f"{target} exists; pass --force to replace")
        shutil.rmtree(target)
    target.mkdir(parents=True)

    public = read_yaml(source / "task.yaml")
    hidden = read_yaml(source / "hidden/task.yaml")
    oracle_cfg = hidden.get("oracle", {})
    if not oracle_cfg.get("buggy_keys") or not oracle_cfg.get("golden_args"):
        raise ValueError(f"{task_id}: oracle is incomplete")

    copy_workspace(source / "workspace", target / "workspace")
    shutil.copytree(source / "hidden", target / "oracle", dirs_exist_ok=True)
    secret_redactions = {
        "workspace": redact_runtime_secrets(target / "workspace"),
        "oracle": redact_runtime_secrets(target / "oracle"),
    }
    removed_agent_outputs = clean_agent_outputs(target / "workspace")
    attribution_cleanup = strip_attribution_annotations(target / "workspace")

    raw_failed = target / "oracle/failed_metrics.json"
    if not raw_failed.exists():
        raise FileNotFoundError(raw_failed)
    manifest = historical_evidence(
        task_id,
        target / "workspace",
        hidden,
        parse_rlve_sh_hparams,
    )
    manifest["metrics_alignment"] = align_outputs_metrics(
        target / "workspace",
        manifest,
        task_id=task_id,
    )
    write_json(
        target / "workspace/historical_evidence/manifest.json",
        manifest,
    )
    write_json(
        target / "workspace/historical_evidence/eval_scores.json",
        manifest["held_out_eval"]["eval_points"],
    )
    write_json(
        target / "workspace/historical_evidence/raw_reward_summary.json",
        manifest["raw_reward"],
    )
    write_json(
        target / "workspace/historical_evidence/hparams_effective.json",
        manifest["effective_hparams_visible"],
    )
    excerpt = "\n".join(
        f"step={point['step']} score={point['eval_score']:.9f} "
        f"truncated_ratio={point['truncate_ratio']}"
        for point in manifest["held_out_eval"]["eval_points"]
    )
    (target / "workspace/historical_evidence/README.txt").write_text(
        "Frozen historical evidence. These are not verifier results.\n",
        encoding="utf-8",
    )
    (target / "workspace/historical_evidence/log_excerpt.txt").write_text(
        excerpt + "\n",
        encoding="utf-8",
    )

    write_json(target / "oracle/historical_evidence_manifest.json", manifest)
    write_json(target / "oracle/buggy_args.json", oracle_cfg["buggy_args"])
    write_json(target / "oracle/golden_args.json", oracle_cfg["golden_args"])
    write_json(
        target / "oracle/baseline_hparams.json",
        oracle_cfg.get("baseline_hparams", {}),
    )
    write_yaml(
        target / "oracle/scoring_rubric.yaml",
        scoring_rubric(task_id, manifest),
    )
    write_json(
        target / "oracle/protected_hashes.json",
        protected_hashes(target / "workspace", public),
    )

    source_ref: Dict[str, Any] = {
        "source_task": f"Benchmark_Evaluation/tasks/task_RLVE_hyper/{task_id}",
        "source_run": hidden.get("source_run"),
        "source_run_ref": (
            f"Benchmark_Data/{hidden.get('source_run')}"
            if hidden.get("source_run")
            else None
        ),
    }
    association = (
        be_root.parent
        / "Benchmark_Data"
        / str(hidden.get("source_run"))
        / "association.json"
    )
    if association.exists():
        source_ref["association"] = json.loads(
            association.read_text(encoding="utf-8")
        )
    write_json(
        target / "oracle/source_snapshot/source_run_ref.json",
        {
            **source_ref,
            "secret_redactions": secret_redactions,
            "removed_agent_outputs": removed_agent_outputs,
            "attribution_annotations": attribution_cleanup,
        },
    )
    launch_rel = Path(public["launch_script"])
    workspace_launch = target / "workspace" / launch_rel
    seed_launch = source / "workspace_seed" / launch_rel
    write_json(
        target / "oracle/source_snapshot/hashes.json",
        {
            "source_task_yaml": sha256_file(source / "task.yaml"),
            "source_hidden_task_yaml": sha256_file(source / "hidden/task.yaml"),
            "workspace_launch": sha256_file(workspace_launch),
            "workspace_seed_launch": sha256_file(seed_launch),
            "workspace_seed_marker_present": (
                source / "workspace_seed/.bench_workspace_seed"
            ).exists(),
        },
    )
    (target / "oracle/source_snapshot/workspace_diff.txt").write_text(
        "Historical-failure case. Critical launch script hashes are recorded in "
        "hashes.json; workspace and seed use the same reconstructed start.\n",
        encoding="utf-8",
    )

    instruction = render_instruction(task_id, public, manifest)
    (target / "instruction.md").write_text(instruction, encoding="utf-8")
    (target / "readme.md").write_text(
        render_readme(task_id, hidden, manifest),
        encoding="utf-8",
    )
    write_yaml(
        target / "metadata.yaml",
        metadata_for(task_id, public, hidden, manifest),
    )

    environment = target / "environment"
    environment.mkdir()
    (environment / "Dockerfile").write_text(
        "FROM slimerl/slime:v0.5.0rc0-cu126\n"
        "WORKDIR /workspace\n"
        "RUN mkdir -p /opt/bench /models /hidden\n"
        'CMD ["tail", "-f", "/dev/null"]\n',
        encoding="utf-8",
    )

    verifier = target / "verifier"
    (verifier / "tests").mkdir(parents=True)
    (verifier / "resources").mkdir()
    (verifier / "test.sh").write_text(verifier_script(), encoding="utf-8")
    (verifier / "verify.py").write_text(verifier_python(), encoding="utf-8")
    os.chmod(verifier / "test.sh", 0o755)
    os.chmod(verifier / "verify.py", 0o755)
    write_json(
        verifier / "resources/verify_config.json",
        {
            "case_id": task_id,
            "launch_script": public["launch_script"],
            "locked_environment_list": public.get(
                "locked_environment_list", []
            ),
            "locked_heldout_environment_list": public.get(
                "locked_heldout_environment_list", []
            ),
            "verify": public.get("verify", {}),
            "packaging_tier": "B0",
        },
    )
    (verifier / "tests/README.md").write_text(
        "B0 checks are implemented by ../verify.py in package mode.\n",
        encoding="utf-8",
    )
    finalize_case(target)
    return {
        "case_id": task_id,
        "target": str(target),
        "n_eval_points": manifest["held_out_eval"]["n_eval_points"],
        "eval_last": manifest["held_out_eval"]["eval_last"],
        "failure_signature": manifest["failure_signature"],
    }


def main() -> int:
    args = parse_args()
    be_root = args.be_root.resolve()
    destination_root = args.dst.resolve()
    if not be_root.is_dir():
        raise FileNotFoundError(be_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    parse_rlve_sh_hparams = load_materialize_parser(be_root)
    results = []
    for task_id in TASK_IDS:
        print(f"materializing {task_id}...", flush=True)
        results.append(
            materialize_one(
                task_id,
                be_root,
                destination_root,
                args.force,
                parse_rlve_sh_hparams,
            )
        )
    write_json(destination_root / "manifest.json", {"cases": results})
    print(json.dumps({"cases": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
