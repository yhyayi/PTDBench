#!/usr/bin/env python3
"""Materialize the selected reward-design dirty workspaces as PTDBench B0 tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

try:
    from .strip_validation_artifacts import finalize_task
except ImportError:
    from strip_validation_artifacts import finalize_task

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
    "reward_difference_constraint_system_034",
    "reward_integer_programming_029",
    "reward_land_acquisition_008",
    "reward_max_different_group_pair_division_023",
    "reward_min_cost_reducing_lnds_020",
    "reward_min_pair_sum_multiplication_permutation_007",
    "reward_new_nim_game_021",
    "reward_polynomial_factorization_035",
    "reward_prefix_product_mod_distinct_permutation_011",
    "reward_prefix_sum_mod_distinct_permutation_010",
    "reward_quadratic_function_segmentation_019",
    "reward_sat_031",
    "reward_set_splitting_012",
    "reward_topological_sort_minimal_lexicographical_order_001",
    "reward_visible_line_038",
    "reward_bez_minimalist_security_002",
    "reward_campfire_party_016",
    "reward_difference_constraint_system_dag_003",
    "reward_dyn_dynamite_017",
    "reward_maf_mafia_004",
    "reward_min_cost_tree_coverage_014",
    "reward_protecting_flowers_006",
    "reward_two_sat_030",
    "reward_tetris_attack_025",
    "reward_hungry_rabbit_036",
)
METRIC_KEY = "eval/HELD-OUT_ENVIRONMENTS_128"
# The launch script every reward-design workspace ships; recorded on the task
# card as training.script so that the task card and the agent's instruction
# name the same entry point.
LAUNCH_REL = "examples/grpo_trainer/run_qwen2_5_1.5b_instruct_taco_fsdp.sh"
BEZ_TASK = "reward_bez_minimalist_security_002"
TOPO_TASK = "reward_topological_sort_minimal_lexicographical_order_001"
TOPO_ARCHIVE = "20260724T155405Z"
MANUAL_BASELINES = {
    BEZ_TASK: {
        "initial": -0.574,
        "maximum": -0.140,
        "provenance": "user_supplied_reference",
        "reason": "The source baseline run failed during SGLang startup and emitted no eval curve.",
    }
}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve()
    data_call = here.parents[1]
    workspace_root = data_call.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--be-root",
        type=Path,
        default=workspace_root / "Benchmark_Evaluation",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=data_call / "tasks" / "reward_design",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def read_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain an object")
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
        if name in {
            ".git",
            "__pycache__",
            ".gpu_locks",
            "experiement",
            "checkpoints",
        }:
            ignored.add(name)
        elif name.endswith((".pyc", ".pyo", ".lock", ".bak", ".agent.bak")):
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


def strip_packaging_annotations(path: Path) -> int:
    """Remove packaging-only comments without altering executable statements."""
    text = path.read_text(encoding="utf-8")
    kept: List[str] = []
    removed = 0
    for line in text.splitlines(keepends=True):
        normalized = line.strip().lower()
        if normalized.startswith("#") and (
            "hollowed symbol" in normalized
            or "hollowed file" in normalized
            or "hollowing" in normalized
            or "挖空" in normalized
            or normalized.startswith("# stub for category=")
            or "__hollow__" in normalized
        ):
            removed += 1
            continue
        kept.append(line)
    if removed:
        path.write_text("".join(kept), encoding="utf-8")
    return removed


def strip_workspace_annotations(root: Path) -> Dict[str, int]:
    removed: Dict[str, int] = {}
    for path in iter_files(root) or []:
        if path.suffix not in {".py", ".sh"}:
            continue
        count = strip_packaging_annotations(path)
        if count:
            removed[str(path.relative_to(root))] = count
    return removed


def resolve_target(workspace: Path, environment: str) -> str:
    root = workspace / "Gym/environments"
    normalized = re.sub(r"[^a-z0-9]", "", environment.lower())
    candidates = []
    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        compact = re.sub(r"[^a-z0-9]", "", directory.name.lower())
        if compact == normalized and (directory / "environment.py").is_file():
            candidates.append(directory)
    if len(candidates) != 1:
        raise ValueError(
            f"cannot uniquely resolve {environment!r} under {root}: {candidates}"
        )
    return str(candidates[0].relative_to(workspace) / "environment.py")


def metric_points(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    points = payload.get("metrics", {}).get("eval_points") or []
    return [point for point in points if point.get("eval_key") == METRIC_KEY]


def metric_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    points = metric_points(payload)
    scores = [float(point["eval_score"]) for point in points]
    return {
        "metric_key": METRIC_KEY,
        "eval_interval": 20,
        "eval_points": points,
        "n_eval_points": len(points),
        "eval_first": scores[0] if scores else None,
        "eval_last": scores[-1] if scores else None,
        "eval_min": min(scores) if scores else None,
        "eval_max": max(scores) if scores else None,
        "first_rollout_ok": bool(payload.get("first_rollout_ok")),
        "attempt": payload.get("attempt"),
        "run_once": payload.get("run_once", {}),
    }


def baseline_reference(
    task_id: str,
    baseline_metrics: Dict[str, Any],
    authoritative_archive: str,
) -> Dict[str, Any]:
    if task_id in MANUAL_BASELINES:
        manual = MANUAL_BASELINES[task_id]
        return {
            "schema_version": 1,
            "status": "manual_reference",
            "metric_key": METRIC_KEY,
            "initial": manual["initial"],
            "maximum": manual["maximum"],
            "provenance": manual["provenance"],
            "reason": manual["reason"],
            "source_baseline_run_ok": bool(
                baseline_metrics.get("run", {}).get("ok")
            ),
            "source_baseline_eval_points": len(metric_points(baseline_metrics)),
            "authoritative_archive": authoritative_archive,
        }
    points = metric_points(baseline_metrics)
    if not baseline_metrics.get("run", {}).get("ok") or not points:
        raise ValueError(f"{task_id}: baseline run is not usable")
    scores = [float(point["eval_score"]) for point in points]
    return {
        "schema_version": 1,
        "status": "measured",
        "metric_key": METRIC_KEY,
        "initial": scores[0],
        "maximum": max(scores),
        "final": scores[-1],
        "n_eval_points": len(points),
        "provenance": "hidden/reward_design_metrics.json",
        "authoritative_archive": authoritative_archive,
    }


def authoritative_archive(task_id: str, metrics: Dict[str, Any]) -> str:
    if task_id == TOPO_TASK:
        return TOPO_ARCHIVE
    archive_rel = metrics.get("run", {}).get("archive_rel")
    if not archive_rel:
        raise ValueError(f"{task_id}: missing baseline archive_rel")
    return Path(str(archive_rel)).name


def checkpoint_manifest(source_workspace: Path) -> Dict[str, Any]:
    found = []
    for name in ("experiement", "checkpoints"):
        root = source_workspace / name
        if root.exists():
            found.append(
                {
                    "path": name,
                    "top_level_entries": sorted(
                        path.name for path in root.iterdir()
                    ),
                }
            )
    return {
        "excluded_from_task": True,
        "reason": "Runtime checkpoints are not needed for diagnosis or B0 validation.",
        "source_locations": found,
    }


def copy_oracle(
    source: Path,
    target: Path,
    task_id: str,
    archive: str,
) -> None:
    oracle = target / "oracle"
    oracle.mkdir()
    shutil.copy2(source / "hidden/task.yaml", oracle / "task.yaml")
    shutil.copytree(source / "hidden/baseline", oracle / "baseline")
    shutil.copytree(source / "hidden/gym_eval", oracle / "gym_eval")
    shutil.copy2(
        source / "hidden/reward_design_metrics.json",
        oracle / "reward_design_metrics.json",
    )
    archive_source = source / "hidden/baseline_runs" / archive
    if not archive_source.is_dir():
        raise FileNotFoundError(archive_source)
    archive_target = oracle / "baseline_runs" / archive
    archive_target.parent.mkdir()
    archive_target.mkdir()
    for name in ("manifest.json", "reward_design_metrics.json"):
        source_file = archive_source / name
        if source_file.is_file():
            shutil.copy2(source_file, archive_target / name)
    if not (archive_target / "reward_design_metrics.json").is_file():
        raise FileNotFoundError(archive_target / "reward_design_metrics.json")
    (archive_target.parent / "LATEST").symlink_to(archive)


def historical_evidence(
    source_workspace: Path,
    packaged_workspace: Path,
) -> Dict[str, Any]:
    metrics_path = packaged_workspace / "outputs/metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    payload = read_json(metrics_path)
    summary = metric_summary(payload)
    source_logs = []
    for relative in payload.get("logs", {}).get("source_logs") or []:
        path = packaged_workspace / relative
        if path.is_file():
            source_logs.append(
                {"path": relative, "sha256": sha256_file(path)}
            )
    return {
        "schema_version": 1,
        "evidence_kind": "historical_run",
        "held_out_eval": summary,
        "source_metrics": {
            "path": "outputs/metrics.json",
            "sha256": sha256_file(metrics_path),
        },
        "source_logs": source_logs,
        "checkpoint_exclusion": checkpoint_manifest(source_workspace),
    }


def protected_hashes(workspace: Path, target_rel: str) -> Dict[str, str]:
    records = {}
    for path in iter_files(workspace) or []:
        relative = str(path.relative_to(workspace))
        if relative == target_rel or relative.startswith("outputs/"):
            continue
        records[relative] = sha256_file(path)
    return records


def failure_class(task_id: str, historical: Dict[str, Any]) -> str:
    if task_id == BEZ_TASK:
        return "historical_run_unavailable"
    if task_id == "reward_difference_constraint_system_dag_003":
        return "incomplete_historical_run"
    if task_id == "reward_difference_constraint_system_034":
        return "semantic_implementation_error"
    if historical["held_out_eval"]["n_eval_points"] < 6:
        return "incomplete_historical_run"
    return "held_out_metric_requires_assessment"


def render_instruction() -> str:
    return f"""# Reward-design run health assessment

Assess the historical RLVE/slime GRPO run using the evidence in the workspace.
Determine whether the run is healthy, abnormal, incomplete, or indeterminate.
Do not assume in advance that any particular setting, file, subsystem, or prior
change is responsible.

Inspect `outputs/`, `logs/`, `wandb/`, `historical_evidence/`, the launch
configuration, source code, and Gym environments where relevant. Use
`{METRIC_KEY}` together with rollout, truncation, runtime, and training signals.
Training reward alone does not establish held-out health.

## Required deliverables

1. Write `outputs/diagnosis.txt` in English. State a verdict (`healthy`,
   `abnormal`, `incomplete`, or `indeterminate`), cite evidence, explain the
   causal chain, and distinguish root causes from symptoms.
2. If and only if the evidence supports a repair, make the minimal necessary
   change. Do not change files merely because they look unusual.
3. Write an executable `outputs/run_command.sh` that reproduces or validates
   your conclusion using a fresh run name and `--eval-interval 20`.
4. Do not launch the full GPU job yourself; the verifier owns execution.
5. Do not modify historical `logs/**`, `wandb/**`,
   `historical_evidence/**`, or held-out data.
"""


def metadata_for(
    task_id: str,
    public: Dict[str, Any],
    historical: Dict[str, Any],
    baseline: Dict[str, Any],
) -> Dict[str, Any]:
    held = historical["held_out_eval"]
    return {
        "title": f"Reward-design run assessment: {task_id}",
        "submitter": {"name": "benchmark_evaluation"},
        "license": "MIT",
        "created_date": "2026-08-08",
        "domain": "post_train",
        "model": {"name": "Qwen2.5-0.5B-Instruct"},
        "dataset": {"name": "RLVE held-out environments"},
        "training": {
            "framework": "RLVE / slime",
            "algo": "grpo",
            "script": LAUNCH_REL,
        },
        "environment": {
            "image": "slime-rc0",
            "assets": "environment/assets.yaml",
        },
        "outcome": {
            "failure_type": "run_health_unknown",
            "failure_reason": "Historical run requires independent evidence-based assessment",
        },
        "bench_category": "reward_design",
        "source_task": (
            f"Benchmark_Evaluation/tasks/task_reward_design/{task_id}"
        ),
        "evidence_kind": "dirty_agent_workspace",
        "packaging_tier": "B0",
        "packaging_ready": True,
        "b1_scoring_ready": False,
        "verify": {
            "num_gpus": int(public.get("verify", {}).get("num_gpus", 4)),
            "timeout_minutes": int(
                public.get("verify", {}).get("timeout_minutes", 90)
            ),
            "eval_interval": 20,
            "eval_metric_key": METRIC_KEY,
        },
        "historical_run": {
            "eval_first": held["eval_first"],
            "eval_last": held["eval_last"],
            "eval_max": held["eval_max"],
            "n_eval_points": held["n_eval_points"],
        },
        "baseline_reference": {
            "status": baseline["status"],
            "initial": baseline["initial"],
            "maximum": baseline["maximum"],
        },
    }


def scoring_rubric(
    task_id: str,
    public: Dict[str, Any],
    baseline: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "eval_metric": METRIC_KEY,
        "eval_interval": 20,
        "b0_structural_only": True,
        "b1_requires_gpu": True,
        "b1_requires_healthy_baseline": baseline["status"] != "measured",
        "baseline_reference": baseline,
        "heldout_environment_count": len(
            public.get("locked_heldout_environment_list", [])
        ),
        "pass_requires": {
            "deliverables_ok": True,
            "target_matches_oracle": True,
            "protected_workspace_unchanged": True,
            "first_rollout_ok": True,
            "metric_repaired": True,
        },
        "task_note": (
            "historical_run_incomplete"
            if task_id == "reward_difference_constraint_system_dag_003"
            else None
        ),
    }


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
) -> Dict[str, Any]:
    source = be_root / "tasks/task_reward_design" / task_id
    target = destination_root / task_id
    if target.exists():
        if not force:
            raise FileExistsError(f"{target} exists; pass --force")
        shutil.rmtree(target)
    target.mkdir(parents=True)

    public = read_yaml(source / "task.yaml")
    hidden = read_yaml(source / "hidden/task.yaml")
    environment_name = str(hidden["environment"])
    source_workspace = source / "workspace"
    target_rel = resolve_target(source_workspace, environment_name)
    source_dirty = source_workspace / target_rel
    source_golden = source / "hidden/baseline/environment.py"
    if sha256_file(source_golden) != hidden["baseline_sha256"]:
        raise ValueError(f"{task_id}: source baseline hash mismatch")

    source_metrics = read_json(source / "hidden/reward_design_metrics.json")
    archive = authoritative_archive(task_id, source_metrics)
    copy_workspace(source_workspace, target / "workspace")
    copy_oracle(source, target, task_id, archive)
    secret_redactions = {
        "workspace": redact_runtime_secrets(target / "workspace"),
        "oracle": redact_runtime_secrets(target / "oracle"),
    }
    removed_agent_outputs = clean_agent_outputs(target / "workspace")
    reward_spec = target / "workspace/REWARD_SPEC.md"
    if reward_spec.is_file():
        reward_spec.unlink()
        removed_agent_outputs.append("REWARD_SPEC.md")
    workspace_annotation_cleanup = strip_workspace_annotations(
        target / "workspace"
    )
    attribution_cleanup = strip_attribution_annotations(target / "workspace")

    annotation_cleanup = {
        "workspace_files": workspace_annotation_cleanup,
        "attribution_annotations": attribution_cleanup,
        "oracle_baseline": strip_packaging_annotations(
            target / "oracle/baseline/environment.py"
        ),
        "removed_agent_outputs": removed_agent_outputs,
    }
    packaged_dirty = target / "workspace" / target_rel
    packaged_golden = target / "oracle/baseline/environment.py"
    dirty_hash = sha256_file(packaged_dirty)
    golden_hash = sha256_file(packaged_golden)

    packaged_hidden = read_yaml(target / "oracle/task.yaml")
    packaged_hidden["baseline_path"] = "oracle/baseline/environment.py"
    packaged_hidden["gym_eval_path"] = "oracle/gym_eval"
    packaged_hidden["baseline_sha256"] = golden_hash
    write_yaml(target / "oracle/task.yaml", packaged_hidden)

    historical = historical_evidence(
        source_workspace,
        target / "workspace",
    )
    historical["metrics_alignment"] = align_outputs_metrics(
        target / "workspace",
        historical,
        task_id=task_id,
    )
    public_historical = {
        "schema_version": historical["schema_version"],
        "evidence_kind": historical["evidence_kind"],
        "held_out_eval": historical["held_out_eval"],
        "checkpoint_exclusion": historical["checkpoint_exclusion"],
    }
    evidence = target / "workspace/historical_evidence"
    write_json(evidence / "manifest.json", public_historical)
    write_json(
        evidence / "eval_points.json",
        historical["held_out_eval"]["eval_points"],
    )
    (evidence / "README.txt").write_text(
        "Frozen summary from a historical run. Runtime checkpoints were excluded.\n",
        encoding="utf-8",
    )
    (evidence / "log_excerpt.txt").write_text(
        "\n".join(
            f"step={point['step']} key={point['eval_key']} "
            f"score={float(point['eval_score']):.9f}"
            for point in historical["held_out_eval"]["eval_points"]
        )
        + "\n",
        encoding="utf-8",
    )

    baseline = baseline_reference(task_id, source_metrics, archive)
    write_json(target / "oracle/baseline_reference.json", baseline)
    write_json(
        target / "oracle/historical_evidence_manifest.json",
        historical,
    )
    target_manifest = {
        "schema_version": 1,
        "targets": [
            {
                "path": target_rel,
                "source_dirty_sha256": sha256_file(source_dirty),
                "source_golden_sha256": sha256_file(source_golden),
                "dirty_sha256": dirty_hash,
                "golden_sha256": golden_hash,
                "already_at_baseline": dirty_hash == golden_hash,
            }
        ],
        "annotation_cleanup": annotation_cleanup,
        "runtime_secret_redactions": secret_redactions,
    }
    write_json(target / "oracle/target_manifest.json", target_manifest)
    write_json(
        target / "oracle/protected_hashes.json",
        protected_hashes(target / "workspace", target_rel),
    )
    write_json(
        target / "oracle/checkpoint_exclusion_manifest.json",
        historical["checkpoint_exclusion"],
    )
    write_yaml(
        target / "oracle/scoring_rubric.yaml",
        scoring_rubric(task_id, public, baseline),
    )
    write_json(
        target / "oracle/source_snapshot/hashes.json",
        {
            "source_task_yaml": sha256_file(source / "task.yaml"),
            "source_hidden_task_yaml": sha256_file(source / "hidden/task.yaml"),
            "workspace_dirty_target": sha256_file(source_dirty),
            "workspace_seed_target": sha256_file(
                source / "workspace_seed" / target_rel
            ),
            "hidden_baseline_target": sha256_file(source_golden),
            "packaged_dirty_target": dirty_hash,
            "packaged_golden_target": golden_hash,
        },
    )
    write_json(
        target / "oracle/classification.json",
        {
            "failure_class": failure_class(task_id, historical),
            "historical_eval_points": historical["held_out_eval"]["n_eval_points"],
            "baseline_status": baseline["status"],
            "heldout_environment_count": len(
                hidden.get("locked_heldout_environment_list", [])
            ),
            "authoritative_baseline_archive": archive,
        },
    )

    (target / "instruction.md").write_text(
        render_instruction(),
        encoding="utf-8",
    )
    (target / "readme.md").write_text(
        f"# {task_id}\n\n"
        "PTDBench B0 reward-design dirty-workspace package.\n\n"
        f"- Historical eval points: `{historical['held_out_eval']['n_eval_points']}`\n"
        f"- Baseline status: `{baseline['status']}`\n"
        f"- Held-out environments: `{len(hidden.get('locked_heldout_environment_list', []))}`\n"
        "- Runtime checkpoints are excluded.\n"
        "- B1 requires a local 4-GPU Phase2 runner.\n",
        encoding="utf-8",
    )
    write_yaml(
        target / "metadata.yaml",
        metadata_for(task_id, public, historical, baseline),
    )

    environment = target / "environment"
    environment.mkdir()
    # 环境不再逐任务建镜像：metadata.environment.image 指向 docker/images/ 下的
    # 某个 Dockerfile。这里只铺一个 assets.yaml 骨架，给 environment/
    # 留个存在的理由（git 不跟空目录），具体内容靠打包时人工填。
    (environment / "assets.yaml").write_text(
        "# 这个任务要用的模型 / 数据集。迁移时必须填：\n"
        "#   - id / kind / destination（destination 是 /workspace 下的相对路径）\n"
        "#   - hf_repo + hf_revision（下载用）\n"
        "#   - files: 里至少给权重 / parquet 的 sha256 + size（只校大文件，见 PLAN §7）\n"
        "#\n"
        "# 现在是空的——意味着容器里不会挂任何模型，训练会在\n"
        "# “模型路径不存在”上失败。raw 任务里没有 repo/revision/sha256，\n"
        "# 所以这一段只能人工补。\n"
        "assets:\n",
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
            "task_id": task_id,
            "target": target_rel,
            "locked_heldout_environment_list": public.get(
                "locked_heldout_environment_list", []
            ),
            "verify": public.get("verify", {}),
            "packaging_tier": "B0",
        },
    )
    (verifier / "tests/README.md").write_text(
        "B0 package checks are implemented by ../verify.py.\n",
        encoding="utf-8",
    )
    finalize_task(target)
    return {
        "task_id": task_id,
        "target": target_rel,
        "historical_eval_points": historical["held_out_eval"]["n_eval_points"],
        "historical_eval_max": historical["held_out_eval"]["eval_max"],
        "baseline_status": baseline["status"],
        "baseline_initial": baseline["initial"],
        "baseline_maximum": baseline["maximum"],
        "heldout_environment_count": len(
            hidden.get("locked_heldout_environment_list", [])
        ),
        "failure_class": failure_class(task_id, historical),
        "already_at_baseline": dirty_hash == golden_hash,
    }


def main() -> int:
    args = parse_args()
    be_root = args.be_root.resolve()
    destination_root = args.dst.resolve()
    if not be_root.is_dir():
        raise FileNotFoundError(be_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    results = []
    for task_id in TASK_IDS:
        print(f"materializing reward_design/{task_id}...", flush=True)
        results.append(
            materialize_one(
                task_id,
                be_root,
                destination_root,
                args.force,
            )
        )
    write_json(
        destination_root / "manifest.json",
        {
            "family": "reward_design",
            "selected_count": len(TASK_IDS),
            "tasks": results,
        },
    )
    print(json.dumps({"tasks": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
