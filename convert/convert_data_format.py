#!/usr/bin/env python3
"""Materialize selected data-format variants as PTDBench B0 cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml

try:
    from .strip_validation_artifacts import finalize_case
except ImportError:
    from strip_validation_artifacts import finalize_case

try:
    from .sanitize_packaging import (
        align_outputs_metrics,
        clean_agent_outputs,
        strip_attribution_annotations,
    )
except ImportError:
    from sanitize_packaging import (
        align_outputs_metrics,
        clean_agent_outputs,
        strip_attribution_annotations,
    )


CASES: Dict[str, Dict[str, Any]] = {
    "task_agent_loop_022-qwen-math-dapo-math": {
        "canonical": "task_qwen_dapo_math_17k/task_agent_loop_022",
        "mother": "qwen_dapo_math_17k",
        "model": "Qwen2.5-Math-1.5B-Instruct",
        "metric": "val-core/math_dapo/reward/mean@1",
        "classification": "format_metric_collapse",
    },
    "task_alt_adv_rloo_reinforce_039-qwen-math-dapo-math": {
        "canonical": "task_qwen_dapo_math_17k/task_alt_adv_rloo_reinforce_039",
        "mother": "qwen_dapo_math_17k",
        "model": "Qwen2.5-Math-1.5B-Instruct",
        "metric": "val-core/math_dapo/acc/mean@1",
        "classification": "noncritical_code_target_and_format_collapse",
    },
    "task_exp_dapo_rm_007-llama-dapo-math": {
        "canonical": "task_llama_dapo_math_17k/task_exp_dapo_rm_007",
        "mother": "llama_dapo_math_17k",
        "model": "Llama-3.2-1B-Instruct",
        "metric": "val-core/math_dapo/acc/mean@1",
        "classification": "code_repair_and_format_contract_review",
    },
    "task_core_algos_013-llama-dapo-math": {
        "canonical": "task_llama_dapo_math_17k/task_core_algos_013",
        "mother": "llama_dapo_math_17k",
        "model": "Llama-3.2-1B-Instruct",
        "metric": "val-core/math_dapo/acc/mean@1",
        "classification": "code_repair_and_format_contract_review",
    },
    "task_agent_loop_022-llama-dapo-math": {
        "canonical": "task_llama_dapo_math_17k/task_agent_loop_022",
        "mother": "llama_dapo_math_17k",
        "model": "Llama-3.2-1B-Instruct",
        "metric": "val-core/math_dapo/reward/mean@1",
        "classification": "format_metric_collapse",
    },
}
FORMAT_ASSETS = (
    "DAPO-Math-17k_verl/train.parquet",
    "DAPO-Math-17k_verl/test.parquet",
    "verl/utils/reward_score/math_dapo.py",
    "examples/grpo_trainer/reward_math_dapo_01.py",
)
MUTABLE_FORMAT_PATHS = {
    "verl/utils/reward_score/math_dapo.py",
    "examples/grpo_trainer/reward_math_dapo_01.py",
}
VAL_RE = re.compile(
    r"step:(\d+)\s+-.*?"
    r"(val-core/math_dapo/(?:acc|reward)/mean@\d+):"
    r"([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)",
    flags=re.IGNORECASE,
)
SECRET_RE = re.compile(
    r'^\s*export\s+WANDB_API_KEY\s*=\s*["\'][^"\']+["\']\s*$',
    flags=re.MULTILINE,
)


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
        default=data_call / "cases" / "data_format",
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
        if name in {".git", "__pycache__", "checkpoints", "experiement"}:
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
            or "user-adjustable" in normalized
        ):
            removed += 1
            continue
        kept.append(line)
    if removed:
        path.write_text("".join(kept), encoding="utf-8")
    return removed


def sanitize_tree(root: Path) -> Dict[str, int]:
    changed: Dict[str, int] = {}
    for path in iter_files(root) or []:
        if path.suffix not in {".py", ".sh"}:
            continue
        count = strip_packaging_annotations(path)
        if count:
            changed[str(path.relative_to(root))] = count
    return changed


def redact_runtime_secrets(workspace: Path) -> Dict[str, int]:
    changed: Dict[str, int] = {}
    for path in workspace.rglob("*.sh"):
        text = path.read_text(encoding="utf-8")
        updated, count = SECRET_RE.subn(
            ': "${WANDB_API_KEY:?Set WANDB_API_KEY in the runtime environment}"',
            text,
        )
        if count:
            path.write_text(updated, encoding="utf-8")
            changed[str(path.relative_to(workspace))] = count
    return changed


def target_files(public: Dict[str, Any]) -> List[str]:
    return list(public.get("hollow_files") or [public["hollow_file"]])


def baseline_map(hidden: Dict[str, Any]) -> Dict[str, str]:
    mapping = dict(hidden.get("baselines") or {})
    if mapping:
        return mapping
    baseline_path = Path(hidden["baseline_path"])
    relative = str(baseline_path.relative_to("hidden/baseline"))
    return {relative: hidden["baseline_sha256"]}


def copy_code_oracle(source: Path, target: Path) -> None:
    code = target / "oracle/code_target"
    code.mkdir(parents=True)
    shutil.copy2(source / "hidden/task.yaml", code / "task.yaml")
    shutil.copytree(source / "hidden/baseline", code / "baseline")
    shutil.copy2(source / "hidden/task.yaml", target / "oracle/task.yaml")


def copy_format_gold(
    be_root: Path,
    target: Path,
    mother: str,
) -> Dict[str, str]:
    source = be_root / "packaging" / mother / "mother_workspace"
    gold = target / "oracle/format_gold"
    hashes = {}
    for relative in FORMAT_ASSETS:
        src = source / relative
        if not src.exists():
            raise FileNotFoundError(src)
        dst = gold / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst, follow_symlinks=True)
        hashes[relative] = sha256_file(dst)
    return hashes


def log_candidates(workspace: Path) -> Iterable[Path]:
    logs = workspace / "logs"
    if logs.exists():
        yield from sorted(logs.rglob("*.log"))
    wandb = workspace / "wandb"
    if wandb.exists():
        for path in sorted(wandb.rglob("*.log")):
            if path.name == "output.log" or path.name.startswith("output_"):
                yield path


def reconstruct_metrics(
    workspace: Path,
    expected_key: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    candidates = []
    for path in log_candidates(workspace):
        points: Dict[int, Dict[str, Any]] = {}
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in VAL_RE.finditer(text):
            key = match.group(2)
            if key != expected_key:
                continue
            step = int(match.group(1))
            points[step] = {
                "step": step,
                "eval_score": float(match.group(3)),
                "eval_key": key,
                "truncate_ratio": None,
            }
        if points:
            candidates.append(
                (
                    len(points),
                    path.stat().st_mtime_ns,
                    path,
                    [points[step] for step in sorted(points)],
                )
            )
    if not candidates:
        raise ValueError(f"no {expected_key} under {workspace}")
    _, _, selected_path, selected_points = max(candidates)
    relative = str(selected_path.relative_to(workspace))
    return selected_points, {relative: sha256_file(selected_path)}


def historical_evidence(
    workspace: Path,
    expected_key: str,
) -> Dict[str, Any]:
    points, source_logs = reconstruct_metrics(workspace, expected_key)
    scores = [float(point["eval_score"]) for point in points]
    metrics_path = workspace / "outputs/metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    payload = read_json(metrics_path)
    return {
        "schema_version": 1,
        "evidence_kind": "historical_run",
        "metric_status": "log_reconstructed",
        "expected_metric_key": expected_key,
        "eval_points": points,
        "n_eval_points": len(points),
        "eval_first": scores[0],
        "eval_last": scores[-1],
        "eval_min": min(scores),
        "eval_max": max(scores),
        "attempt": payload.get("attempt"),
        "first_rollout_ok": bool(payload.get("first_rollout_ok")),
        "run_once": payload.get("run_once", {}),
        "source_metrics": {
            "path": "outputs/metrics.json",
            "sha256": sha256_file(metrics_path),
            "source_eval_points": len(
                payload.get("metrics", {}).get("eval_points") or []
            ),
        },
        "source_logs": [
            {"path": path, "sha256": digest}
            for path, digest in sorted(source_logs.items())
        ],
    }


def protected_hashes(
    workspace: Path,
    targets: List[str],
) -> Dict[str, str]:
    excluded = set(targets) | MUTABLE_FORMAT_PATHS
    records = {}
    for path in iter_files(workspace) or []:
        relative = str(path.relative_to(workspace))
        if relative in excluded or relative.startswith("outputs/"):
            continue
        records[relative] = sha256_file(path)
    return records


def checkpoint_manifest(source_workspace: Path) -> Dict[str, Any]:
    present = [
        name
        for name in ("checkpoints", "experiement")
        if (source_workspace / name).exists()
    ]
    return {
        "excluded_from_case": True,
        "source_locations": present,
        "reason": "Runtime checkpoints are not required for B0 diagnosis.",
    }


def render_instruction(config: Dict[str, Any], launch: str) -> str:
    return f"""# Verl run health assessment

Assess the historical Verl GRPO run. Determine from the workspace evidence
whether it is healthy, abnormal, incomplete, or indeterminate. Do not assume in
advance that a particular configuration value, source component, data issue,
runtime condition, or prior change is responsible.

Model: `{config['model']}`

Launch: `{launch}`

Expected validation signal: `{config['metric']}`

## Constraints

- Inspect `outputs/`, `logs/`, `wandb/`, effective configuration, source code,
  data flow, response processing, and `historical_evidence/`.
- If and only if the evidence supports a repair, make the minimal necessary
  change. Do not change files merely because they look unusual.
- Keep the model, training budget, evaluation objective, and held-out labels
  fixed.
- Do not modify historical logs, wandb, or frozen evidence.

## Deliverables

1. Write `outputs/diagnosis.txt` in English. State a verdict (`healthy`,
   `abnormal`, `incomplete`, or `indeterminate`), cite evidence, explain the
   causal chain, and distinguish root causes from symptoms.
2. Write executable `outputs/run_command.sh` containing:

```bash
#!/usr/bin/env bash
set -euo pipefail
bash {launch}
```

Do not run the full GPU job yourself; the verifier owns execution.
"""


def metadata_for(
    case_id: str,
    config: Dict[str, Any],
    public: Dict[str, Any],
    historical: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "case_id": case_id,
        "title": f"Data-format run assessment: {case_id}",
        "submitter": {"name": "benchmark_evaluation"},
        "license": "MIT",
        "created_date": "2026-08-08",
        "domain": "post_train",
        "model": {"name": config["model"]},
        "dataset": {"name": "DAPO-Math-17k"},
        "training": {
            "framework": "verl",
            "algo": "grpo",
            "run_command": "bash outputs/run_command.sh",
        },
        "environment": {"dockerfile": "environment/Dockerfile"},
        "outcome": {
            "failure_type": "run_health_unknown",
            "failure_reason": "Historical run requires independent evidence-based assessment",
        },
        "bench_category": "data_format",
        "source_task": (
            f"Benchmark_Evaluation/tasks/task_data_format/{case_id}"
        ),
        "canonical_task": config["canonical"],
        "data_format_variant": True,
        "evidence_kind": "dirty_agent_workspace",
        "packaging_tier": "B0",
        "packaging_ready": True,
        "b1_scoring_ready": False,
        "verify": {
            "num_gpus": int(public["verify"]["num_gpus"]),
            "timeout_minutes": int(public["verify"]["timeout_minutes"]),
            "max_train_steps": 100,
            "eval_metric_key": config["metric"],
        },
        "historical_run": {
            "metric_status": historical["metric_status"],
            "eval_first": historical["eval_first"],
            "eval_last": historical["eval_last"],
            "eval_max": historical["eval_max"],
            "n_eval_points": historical["n_eval_points"],
            "attempt": historical["attempt"],
        },
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
    case_id: str,
    config: Dict[str, Any],
    be_root: Path,
    destination_root: Path,
    force: bool,
) -> Dict[str, Any]:
    source = be_root / "tasks/task_data_format" / case_id
    target = destination_root / case_id
    if target.exists():
        if not force:
            raise FileExistsError(f"{target} exists; pass --force")
        shutil.rmtree(target)
    target.mkdir(parents=True)

    public = read_yaml(source / "task.yaml")
    hidden = read_yaml(source / "hidden/task.yaml")
    targets = target_files(public)
    baselines = baseline_map(hidden)
    if set(targets) != set(baselines):
        raise ValueError(f"{case_id}: target/baseline mismatch")

    source_workspace = source / "workspace"
    source_records = []
    for relative in targets:
        dirty = source_workspace / relative
        golden = source / "hidden/baseline" / relative
        if not dirty.is_file() or not golden.is_file():
            raise FileNotFoundError(relative)
        if sha256_file(golden) != baselines[relative]:
            raise ValueError(f"{case_id}: source golden hash mismatch")
        source_records.append(
            {
                "path": relative,
                "source_dirty_sha256": sha256_file(dirty),
                "source_golden_sha256": sha256_file(golden),
            }
        )

    copy_workspace(source_workspace, target / "workspace")
    (target / "oracle").mkdir()
    copy_code_oracle(source, target)
    format_hashes = copy_format_gold(
        be_root,
        target,
        config["mother"],
    )
    cleanup = {
        "workspace": sanitize_tree(target / "workspace"),
        "code_oracle": sanitize_tree(target / "oracle/code_target/baseline"),
        "format_oracle": sanitize_tree(target / "oracle/format_gold"),
    }
    secret_redactions = redact_runtime_secrets(target / "workspace")
    cleanup["attribution_annotations"] = strip_attribution_annotations(
        target / "workspace"
    )
    cleanup["removed_agent_outputs"] = clean_agent_outputs(
        target / "workspace"
    )

    packaged_hidden = read_yaml(target / "oracle/code_target/task.yaml")
    target_records = []
    packaged_baselines = {}
    for record in source_records:
        relative = record["path"]
        dirty_hash = sha256_file(target / "workspace" / relative)
        golden_hash = sha256_file(
            target / "oracle/code_target/baseline" / relative
        )
        record.update(
            {
                "dirty_sha256": dirty_hash,
                "golden_sha256": golden_hash,
                "already_at_baseline": dirty_hash == golden_hash,
            }
        )
        target_records.append(record)
        packaged_baselines[relative] = golden_hash
    packaged_hidden["baselines"] = packaged_baselines
    packaged_hidden["baseline_sha256"] = packaged_baselines[targets[0]]
    packaged_hidden["baseline_path"] = (
        "oracle/code_target/baseline/" + targets[0]
    )
    packaged_hidden.pop("note", None)
    packaged_hidden.pop("hollow_symbols", None)
    packaged_hidden.pop("hollowed_symbols", None)
    write_yaml(target / "oracle/code_target/task.yaml", packaged_hidden)
    write_yaml(target / "oracle/task.yaml", packaged_hidden)

    historical = historical_evidence(
        target / "workspace",
        config["metric"],
    )
    historical["metrics_alignment"] = align_outputs_metrics(
        target / "workspace",
        historical,
        task_id=case_id,
    )
    public_historical = {
        key: value
        for key, value in historical.items()
        if key not in {"source_logs", "source_metrics"}
    }
    evidence = target / "workspace/historical_evidence"
    write_json(evidence / "manifest.json", public_historical)
    write_json(evidence / "eval_scores.json", historical["eval_points"])
    (evidence / "README.txt").write_text(
        "Frozen validation summary reconstructed from the historical run log.\n",
        encoding="utf-8",
    )
    (evidence / "log_excerpt.txt").write_text(
        "\n".join(
            f"step={point['step']} key={point['eval_key']} "
            f"score={point['eval_score']:.9f}"
            for point in historical["eval_points"]
        )
        + "\n",
        encoding="utf-8",
    )

    format_hashes = {
        relative: sha256_file(target / "oracle/format_gold" / relative)
        for relative in FORMAT_ASSETS
    }
    write_json(
        target / "oracle/format_gold/manifest.json",
        {
            "schema_version": 1,
            "source": (
                f"Benchmark_Evaluation/packaging/{config['mother']}/"
                "mother_workspace"
            ),
            "status": "healthy_format_stack",
            "hashes": format_hashes,
            "model_and_launch_inherited": False,
        },
    )
    dirty_format_hashes = {
        relative: sha256_file(target / "workspace" / relative)
        for relative in MUTABLE_FORMAT_PATHS
    }
    write_json(
        target / "oracle/dirty_reference/format_hashes.json",
        dirty_format_hashes,
    )
    write_json(
        target / "oracle/dirty_reference/parquet_runtime_note.json",
        {
            "workspace_bundled": False,
            "historical_runtime_path": "/workspace/DAPO-Math-17k_verl",
            "historical_runtime_sha256": None,
            "previously_documented_raw_train_sha256": (
                "9656de825e924e851abedb8d069a66d894d71abc78d0cd3df04b69796cb3e1c3"
            ),
            "previously_documented_artifact_status": "missing",
            "format_gold_is_preprocessed": True,
        },
    )
    write_json(
        target / "oracle/dirty_reference/historical_metrics.json",
        historical,
    )
    write_json(
        target / "oracle/historical_evidence_manifest.json",
        historical,
    )
    write_json(
        target / "oracle/target_manifest.json",
        {
            "schema_version": 1,
            "targets": target_records,
            "annotation_cleanup": cleanup,
            "runtime_secret_redactions": secret_redactions,
        },
    )
    write_json(
        target / "oracle/code_stack_hashes.json",
        {
            str(path.relative_to(target / "oracle/code_target/baseline")): (
                sha256_file(path)
            )
            for path in iter_files(target / "oracle/code_target/baseline") or []
        },
    )
    write_json(
        target / "oracle/protected_hashes.json",
        protected_hashes(target / "workspace", targets),
    )
    checkpoint_info = checkpoint_manifest(source_workspace)
    write_json(
        target / "oracle/checkpoint_exclusion_manifest.json",
        checkpoint_info,
    )
    write_json(
        target / "oracle/source_snapshot/hashes.json",
        {
            "source_task_yaml": sha256_file(source / "task.yaml"),
            "source_hidden_task_yaml": sha256_file(source / "hidden/task.yaml"),
            "workspace_seed_targets": {
                relative: sha256_file(source / "workspace_seed" / relative)
                for relative in targets
            },
            "packaged_dirty_targets": {
                item["path"]: item["dirty_sha256"] for item in target_records
            },
        },
    )
    write_json(
        target / "oracle/classification.json",
        {
            "data_format_variant": True,
            "canonical_task": config["canonical"],
            "classification": config["classification"],
            "code_oracle_source": "self_hidden",
            "format_oracle_source": (
                f"packaging/{config['mother']}/mother_workspace"
            ),
            "historical_metric_key": config["metric"],
            "historical_eval_points": historical["n_eval_points"],
            "healthy_metric_threshold_status": "not_calibrated",
        },
    )
    write_yaml(
        target / "oracle/scoring_rubric.yaml",
        {
            "schema_version": 1,
            "eval_metric": config["metric"],
            "max_train_steps": 100,
            "b0_structural_only": True,
            "b1_requires_gpu": True,
            "b1_requires_format_gold_injection": True,
            "healthy_metric_threshold_status": "not_calibrated",
            "pass_requires": {
                "code_target_matches_golden": True,
                "format_contract_healthy": True,
                "protected_workspace_unchanged": True,
                "first_rollout_ok": True,
                "metric_repaired": True,
            },
        },
    )

    launch = str(public["launch_script"])
    (target / "instruction.md").write_text(
        render_instruction(config, launch),
        encoding="utf-8",
    )
    (target / "readme.md").write_text(
        f"# {case_id}\n\n"
        "PTDBench B0 data-format dirty-workspace package.\n\n"
        f"- Historical metric: `{config['metric']}`\n"
        f"- Historical eval points: `{historical['n_eval_points']}`\n"
        "- Code and format oracles are stored separately.\n"
        "- B1 requires a local 2-GPU runner and healthy-metric calibration.\n",
        encoding="utf-8",
    )
    write_yaml(
        target / "metadata.yaml",
        metadata_for(case_id, config, public, historical),
    )
    environment = target / "environment"
    environment.mkdir()
    (environment / "Dockerfile").write_text(
        "FROM verlai/verl:vllm017.latest\n"
        "WORKDIR /workspace\n"
        "RUN mkdir -p /models /hidden /datasets\n"
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
            "case_id": case_id,
            "canonical_task": config["canonical"],
            "model": config["model"],
            "launch_script": launch,
            "metric": config["metric"],
            "targets": targets,
            "verify": public["verify"],
            "packaging_tier": "B0",
        },
    )
    (verifier / "tests/README.md").write_text(
        "B0 package checks are implemented by ../verify.py.\n",
        encoding="utf-8",
    )
    finalize_case(target)
    return {
        "case_id": case_id,
        "canonical_task": config["canonical"],
        "model": config["model"],
        "metric": config["metric"],
        "targets": targets,
        "historical_eval_points": historical["n_eval_points"],
        "historical_eval_first": historical["eval_first"],
        "historical_eval_last": historical["eval_last"],
        "historical_eval_max": historical["eval_max"],
        "source_metrics_eval_points": historical["source_metrics"][
            "source_eval_points"
        ],
        "secret_redactions": sum(secret_redactions.values()),
        "already_at_baseline": all(
            item["already_at_baseline"] for item in target_records
        ),
    }


def main() -> int:
    args = parse_args()
    be_root = args.be_root.resolve()
    destination_root = args.dst.resolve()
    if not be_root.is_dir():
        raise FileNotFoundError(be_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    results = []
    for case_id, config in CASES.items():
        print(f"materializing data_format/{case_id}...", flush=True)
        results.append(
            materialize_one(
                case_id,
                config,
                be_root,
                destination_root,
                args.force,
            )
        )
    write_json(
        destination_root / "manifest.json",
        {
            "family": "data_format",
            "selected_count": len(CASES),
            "cases": results,
        },
    )
    print(json.dumps({"cases": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
