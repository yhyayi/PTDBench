#!/usr/bin/env python3
"""Materialize eight Verl implementation tasks into three PTDBench families."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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


FAMILIES: Dict[str, Dict[str, Any]] = {
    "qwen_dapo_implementation": {
        "source_category": "task_qwen_dapo_math_17k",
        "tasks": (
            "task_ray_trainer_015",
            "task_reward_adv_grpo_core_031",
            "task_rl_dataset_026",
        ),
        "model": "Qwen2.5-0.5B-Instruct",
        "launch": "examples/grpo_trainer/run_qwen2_5_0_5b_instruct_dapo_math_fsdp.sh",
        "metric": "val-core/math_dapo/acc/mean@1",
        "validation_stack": "DAPO_V4_QWEN",
        "max_train_steps": 200,
    },
    "llama_dapo_implementation": {
        "source_category": "task_llama_dapo_math_17k",
        "tasks": (
            "task_monkey_patch_011",
            "task_core_algos_013",
            "task_kl_reward_adv_chain_032",
        ),
        "model": "Llama-3.2-1B-Instruct",
        "launch": "examples/grpo_trainer/run_llama3_2_1b_instruct_dapo_math_fsdp.sh",
        "metric": "val-core/math_dapo/acc/mean@1",
        "validation_stack": "DAPO_V4_LLAMA",
        "max_train_steps": 200,
    },
    "verl_reward_implementation": {
        "source_category": "task_verl_reward_implementation",
        "tasks": ("task_engine_base_025",),
        "model": "Qwen2.5-0.5B-Instruct",
        "launch": "examples/grpo_trainer/run_qwen2_5_0.5b_instruct_fsdp.sh",
        "metric": "val-core/openai/gsm8k/acc/mean@1",
        "validation_stack": "GSM8K_G3",
        "max_train_steps": 100,
    },
}
REMOVED_TASKS = {
    "llama_dapo_implementation": ("task_agent_loop_022",),
}
VAL_RE = re.compile(
    r"(?:^|\s)step:(\d+)\s+-.*?"
    r"\b(val-core/([^\s:]+)/(acc|reward)/mean@(\d+)):"
    r"([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)",
    flags=re.IGNORECASE,
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
        default=data_call / "tasks",
    )
    parser.add_argument(
        "--family",
        choices=("all", *FAMILIES),
        default="all",
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
        if name in {".git", "__pycache__", "checkpoints"}:
            ignored.add(name)
        elif name.endswith((".pyc", ".pyo", ".lock", ".bak")):
            ignored.add(name)
        elif name.startswith("debug-core"):
            ignored.add(name)
    return ignored


def strip_hollow_annotations(path: Path) -> int:
    """Remove packaging-only comments without changing executable code."""
    text = path.read_text(encoding="utf-8")
    replacements = {
        "the workspace (Agent hollow files), then restore after validation.": (
            "the workspace, then restore after validation."
        ),
        "Bench hooks that must survive hollowing of ray_trainer / reward_loop.": (
            "Validation hooks for ray_trainer / reward_loop."
        ),
        "Installed from main_ppo_v0 (never hollowed):": (
            "Installed from main_ppo_v0:"
        ),
    }
    removed = 0
    for old, new in replacements.items():
        count = text.count(old)
        if count:
            text = text.replace(old, new)
            removed += count
    lines = text.splitlines(keepends=True)
    kept = []
    for line in lines:
        normalized = line.strip().lower()
        is_comment = normalized.startswith("#")
        leaks_packaging = (
            "hollow" in normalized
            or "挖空" in normalized
            or "user-adjustable" in normalized
            or normalized.startswith("# stub for category=")
            or "non-target code" in normalized
            or "do not delete or rename interfaces" in normalized
        )
        if is_comment and leaks_packaging:
            removed += 1
            continue
        kept.append(line)
    if removed:
        path.write_text("".join(kept), encoding="utf-8")
    return removed


def strip_workspace_annotations(root: Path) -> Dict[str, int]:
    removed = {}
    for path in iter_files(root) or []:
        if path.suffix not in {".py", ".sh"}:
            continue
        count = strip_hollow_annotations(path)
        if count:
            removed[str(path.relative_to(root))] = count
    return removed


def copy_workspace(source: Path, target: Path) -> None:
    shutil.copytree(
        source,
        target,
        symlinks=False,
        ignore=copy_ignore,
    )
    (target / "outputs").mkdir(parents=True, exist_ok=True)


def source_metrics_payload(outputs: Path) -> Dict[str, Any]:
    path = outputs / "metrics.json"
    return (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else {}
    )


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
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]], str]:
    expected: Dict[int, Dict[str, Any]] = {}
    matched_files: Dict[str, str] = {}
    for path in log_candidates(workspace):
        text = path.read_text(encoding="utf-8", errors="ignore")
        file_matched = False
        for match in VAL_RE.finditer(text):
            step = int(match.group(1))
            key = match.group(2)
            score = float(match.group(6))
            point = {
                "step": step,
                "eval_score": score,
                "eval_key": key,
                "truncate_ratio": None,
            }
            if key == expected_key:
                expected[step] = point
                file_matched = True
        if file_matched:
            relative = str(path.relative_to(workspace))
            matched_files[relative] = sha256_file(path)
    if not expected:
        raise ValueError(f"no {expected_key} metrics found under {workspace}")
    points = [expected[step] for step in sorted(expected)]
    source_logs = [
        {"path": relative, "sha256": matched_files[relative]}
        for relative in sorted(matched_files)
    ]
    return points, source_logs, "log_reconstructed"


def collect_metrics(
    workspace: Path,
    outputs: Path,
    expected_key: str,
) -> Dict[str, Any]:
    payload = source_metrics_payload(outputs)
    points = payload.get("metrics", {}).get("eval_points") or []
    if points:
        source_logs = []
        for relative in payload.get("logs", {}).get("source_logs") or []:
            path = workspace / relative
            if path.is_file():
                source_logs.append(
                    {"path": relative, "sha256": sha256_file(path)}
                )
        status = "metrics_json"
    else:
        points, source_logs, status = reconstruct_metrics(
            workspace,
            expected_key,
        )
    scores = [float(point["eval_score"]) for point in points]
    return {
        "metric_status": status,
        "expected_metric_key": expected_key,
        "actual_metric_key": points[-1]["eval_key"] if points else None,
        "eval_points": points,
        "n_eval_points": len(points),
        "eval_first": scores[0] if scores else None,
        "eval_last": scores[-1] if scores else None,
        "eval_min": min(scores) if scores else None,
        "eval_max": max(scores) if scores else None,
        "source_logs": source_logs,
        "first_rollout_ok": bool(payload.get("first_rollout_ok")),
        "attempt": payload.get("attempt"),
        "run_once": payload.get("run_once", {}),
    }


def attempts_summary(outputs: Path) -> Dict[str, Any]:
    path = outputs / "run_attempts.json"
    attempts = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else []
    )
    return {
        "n_attempts": len(attempts),
        "n_ok": sum(bool(item.get("ok")) for item in attempts),
        "n_startup_failed": sum(
            bool(item.get("startup_failed")) for item in attempts
        ),
        "final_attempt": attempts[-1].get("attempt") if attempts else None,
        "final_ok": bool(attempts and attempts[-1].get("ok")),
    }


def baseline_map(hidden: Dict[str, Any]) -> Dict[str, str]:
    mapping = dict(hidden.get("baselines") or {})
    if not mapping:
        baseline_path = Path(hidden["baseline_path"])
        prefix = Path("hidden/baseline")
        mapping[str(baseline_path.relative_to(prefix))] = hidden[
            "baseline_sha256"
        ]
    return mapping


def target_files(public: Dict[str, Any]) -> List[str]:
    return list(public.get("hollow_files") or [public["hollow_file"]])


def target_symbols(
    public: Dict[str, Any],
    hidden: Dict[str, Any],
) -> Dict[str, List[str]]:
    symbol_map = public.get("hollow_symbol_map")
    if symbol_map:
        return {
            str(path): list(symbols or [])
            for path, symbols in symbol_map.items()
        }
    symbols = list(
        public.get("hollow_symbols")
        or hidden.get("hollowed_symbols")
        or []
    )
    return {public["hollow_file"]: symbols}


def protected_hashes(
    workspace: Path,
    targets: List[str],
) -> Dict[str, str]:
    records = {}
    target_set = set(targets)
    for path in iter_files(workspace) or []:
        relative = str(path.relative_to(workspace))
        if relative in target_set:
            continue
        if relative.startswith(("outputs/", "historical_evidence/")):
            continue
        records[relative] = sha256_file(path)
    return records


def checkpoint_manifest(source_workspace: Path) -> Dict[str, Any]:
    root = source_workspace / "checkpoints"
    return {
        "excluded_from_task": True,
        "reason": "Implementation diagnosis does not require 12-148GB checkpoints.",
        "source_present": root.exists(),
        "top_level_entries": (
            sorted(path.name for path in root.iterdir()) if root.exists() else []
        ),
    }


def render_instruction(
    task_id: str,
    family: str,
    config: Dict[str, Any],
    public: Dict[str, Any],
    symbols: Dict[str, List[str]],
    metrics: Dict[str, Any],
) -> str:
    return f"""# Verl run health assessment

Assess the historical Verl GRPO run. Determine from the workspace evidence
whether it is healthy, abnormal, incomplete, or indeterminate. Do not assume in
advance that a particular configuration value, source component, data issue,
runtime condition, or prior change is responsible.

Model: `{config['model']}`

Launch: `{config['launch']}`

Expected validation signal: `{config['metric']}`

## Constraints

- Inspect the archived outputs, logs, wandb, effective configuration, source
  code, and `historical_evidence/` before reaching a conclusion.
- If and only if the evidence supports a repair, make the minimal necessary
  change. Do not change files merely because they look unusual.
- Keep the model, training data, launch budget, and evaluation objective fixed.
- Do not modify historical logs, wandb, or frozen evidence.
- Checkpoints were intentionally excluded because they are not needed for
  run assessment and are tens to hundreds of GB.

## Deliverables

1. Write `outputs/diagnosis.txt` in English. State a verdict
   (`healthy`, `abnormal`, `incomplete`, or `indeterminate`), cite evidence,
   explain the causal chain, and separate root causes from symptoms.
2. Write executable `outputs/run_command.sh` containing:

```bash
#!/usr/bin/env bash
set -euo pipefail
bash {config['launch']}
```

Do not run the full GPU job yourself; the verifier owns execution.
"""


def metadata_for(
    task_id: str,
    family: str,
    config: Dict[str, Any],
    public: Dict[str, Any],
    metrics: Dict[str, Any],
    targets: List[str],
    already_at_baseline: bool,
) -> Dict[str, Any]:
    return {
        "title": f"{family} implementation diagnosis: {task_id}",
        "submitter": {"name": "benchmark_evaluation"},
        "license": "MIT",
        "created_date": "2026-08-08",
        "domain": "post_train",
        "model": {"name": config["model"]},
        "dataset": {
            "name": (
                "GSM8K"
                if family == "verl_reward_implementation"
                else "DAPO-Math-17k"
            )
        },
        "training": {
            "framework": "verl",
            "algo": "grpo",
        },
        "environment": {"dockerfile": "environment/Dockerfile"},
        "outcome": {
            "failure_type": (
                "historical_recovered"
                if already_at_baseline
                else "implementation_bug"
            ),
            "failure_reason": (
                "Previous Agent implementation requires oracle and metric verification"
            ),
        },
        "bench_category": family,
        "source_task": (
            f"Benchmark_Evaluation/tasks/{config['source_category']}/{task_id}"
        ),
        "evidence_kind": "dirty_agent_failure",
        "packaging_tier": "B0",
        "packaging_ready": True,
        "b1_scoring_ready": False,
        "target_files": ",".join(targets),
        "validation_stack": config["validation_stack"],
        "already_at_baseline": already_at_baseline,
        "metric_status": metrics["metric_status"],
        "verify": {
            "num_gpus": int(public["verify"]["num_gpus"]),
            "timeout_minutes": int(public["verify"]["timeout_minutes"]),
            "max_train_steps": int(config["max_train_steps"]),
            "eval_metric_key": config["metric"],
        },
        "historical_baseline": {
            "actual_metric_key": metrics["actual_metric_key"],
            "eval_last": metrics["eval_last"],
            "eval_max": metrics["eval_max"],
            "n_eval_points": metrics["n_eval_points"],
            "attempt": metrics["attempt"],
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
    task_id: str,
    family: str,
    config: Dict[str, Any],
    be_root: Path,
    destination_root: Path,
    force: bool,
) -> Dict[str, Any]:
    source = be_root / "tasks" / config["source_category"] / task_id
    target = destination_root / family / task_id
    if target.exists():
        if not force:
            raise FileExistsError(f"{target} exists; pass --force")
        shutil.rmtree(target)
    target.mkdir(parents=True)

    public = read_yaml(source / "task.yaml")
    hidden = read_yaml(source / "hidden/task.yaml")
    targets = target_files(public)
    symbols = target_symbols(public, hidden)
    baselines = baseline_map(hidden)
    if set(targets) != set(baselines):
        raise ValueError(
            f"{task_id}: targets {targets} != baseline keys {list(baselines)}"
        )
    source_workspace = source / "workspace"
    target_records = []
    for relative in targets:
        dirty_path = source_workspace / relative
        golden_path = source / "hidden/baseline" / relative
        if not dirty_path.is_file() or not golden_path.is_file():
            raise FileNotFoundError(relative)
        golden_hash = sha256_file(golden_path)
        if golden_hash != baselines[relative]:
            raise ValueError(f"{task_id}: baseline hash mismatch for {relative}")
        dirty_hash = sha256_file(dirty_path)
        target_records.append(
            {
                "path": relative,
                "symbols": symbols.get(relative, []),
                "source_dirty_sha256": dirty_hash,
                "source_golden_sha256": golden_hash,
                "golden_source": f"source_task/hidden/baseline/{relative}",
            }
        )

    copy_workspace(source_workspace, target / "workspace")
    shutil.copytree(source / "hidden", target / "oracle")
    secret_redactions = {
        "workspace": redact_runtime_secrets(target / "workspace"),
        "oracle": redact_runtime_secrets(target / "oracle"),
    }
    removed_agent_outputs = clean_agent_outputs(target / "workspace")
    workspace_annotation_cleanup = strip_workspace_annotations(
        target / "workspace"
    )
    attribution_cleanup = strip_attribution_annotations(target / "workspace")
    annotation_cleanup = {
        "workspace_files": workspace_annotation_cleanup,
        "attribution_annotations": attribution_cleanup,
        "oracle_baseline_targets": {},
        "secret_redactions": secret_redactions,
        "removed_agent_outputs": removed_agent_outputs,
    }
    packaged_baselines = {}
    for record in target_records:
        relative = record["path"]
        dirty_path = target / "workspace" / relative
        golden_path = target / "oracle/baseline" / relative
        dirty_removed = workspace_annotation_cleanup.get(relative, 0)
        golden_removed = strip_hollow_annotations(golden_path)
        dirty_hash = sha256_file(dirty_path)
        golden_hash = sha256_file(golden_path)
        record.update(
            {
                "dirty_sha256": dirty_hash,
                "golden_sha256": golden_hash,
                "already_at_baseline": dirty_hash == golden_hash,
                "annotations_removed": {
                    "workspace": dirty_removed,
                    "oracle_baseline": golden_removed,
                },
            }
        )
        packaged_baselines[relative] = golden_hash
        if golden_removed:
            annotation_cleanup["oracle_baseline_targets"][
                relative
            ] = golden_removed
    packaged_hidden = read_yaml(target / "oracle/task.yaml")
    packaged_hidden["baselines"] = packaged_baselines
    first_target = targets[0]
    packaged_hidden["baseline_sha256"] = packaged_baselines[first_target]
    write_yaml(target / "oracle/task.yaml", packaged_hidden)

    metrics = collect_metrics(
        target / "workspace",
        source_workspace / "outputs",
        config["metric"],
    )
    attempts = attempts_summary(source_workspace / "outputs")
    checkpoint_info = checkpoint_manifest(source_workspace)
    manifest = {
        "schema_version": 1,
        "evidence_kind": "dirty_agent_failure",
        "task_id": task_id,
        "family": family,
        "validation_stack": config["validation_stack"],
        "metrics": metrics,
        "attempts": attempts,
        "targets": target_records,
        "annotation_cleanup": annotation_cleanup,
        "checkpoints": checkpoint_info,
    }
    manifest["metrics_alignment"] = align_outputs_metrics(
        target / "workspace",
        manifest,
        task_id=task_id,
    )
    public_manifest = {
        key: value
        for key, value in manifest.items()
        if key not in {"task_id", "targets"}
    }
    evidence = target / "workspace/historical_evidence"
    write_json(evidence / "manifest.json", public_manifest)
    write_json(evidence / "eval_points.json", metrics["eval_points"])
    write_json(evidence / "run_attempts_summary.json", attempts)
    (evidence / "README.txt").write_text(
        "Frozen evidence from the previous Agent run. Checkpoints were excluded.\n",
        encoding="utf-8",
    )
    (evidence / "log_excerpt.txt").write_text(
        "\n".join(
            f"step={point['step']} key={point['eval_key']} "
            f"score={point['eval_score']:.9f}"
            for point in metrics["eval_points"]
        )
        + "\n",
        encoding="utf-8",
    )

    write_json(target / "oracle/historical_evidence_manifest.json", manifest)
    write_json(
        target / "oracle/target_manifest.json",
        {
            "family": family,
            "validation_stack": config["validation_stack"],
            "targets": target_records,
        },
    )
    stack_hashes = {
        str(path.relative_to(target / "oracle/baseline")): sha256_file(path)
        for path in iter_files(target / "oracle/baseline") or []
    }
    write_json(target / "oracle/validation_stack_hashes.json", stack_hashes)
    write_json(target / "oracle/checkpoint_exclusion_manifest.json", checkpoint_info)
    write_json(
        target / "oracle/protected_hashes.json",
        protected_hashes(target / "workspace", targets),
    )
    write_yaml(
        target / "oracle/scoring_rubric.yaml",
        {
            "schema_version": 1,
            "eval_metric": config["metric"],
            "historical_metric_status": metrics["metric_status"],
            "validation_stack": config["validation_stack"],
            "max_train_steps": config["max_train_steps"],
            "b0_structural_only": True,
            "b1_requires_gpu": True,
            "b1_requires_healthy_baseline": (
                family == "verl_reward_implementation"
            ),
            "pass_requires": {
                "targets_match_golden": True,
                "protected_workspace_unchanged": True,
                "first_rollout_ok": True,
                "metric_repaired": True,
            },
        },
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
        },
    )

    (target / "instruction.md").write_text(
        render_instruction(
            task_id,
            family,
            config,
            public,
            symbols,
            metrics,
        ),
        encoding="utf-8",
    )
    (target / "readme.md").write_text(
        f"# {task_id}\n\n"
        f"PTDBench B0 `{family}` implementation package.\n\n"
        f"- Validation stack: `{config['validation_stack']}`\n"
        f"- Metric status: `{metrics['metric_status']}`\n"
        f"- Checkpoints excluded: `{checkpoint_info['source_present']}`\n"
        "- B1 requires a local 2-GPU runner.\n",
        encoding="utf-8",
    )
    write_yaml(
        target / "metadata.yaml",
        metadata_for(
            task_id,
            family,
            config,
            public,
            metrics,
            targets,
            all(item["already_at_baseline"] for item in target_records),
        ),
    )

    environment = target / "environment"
    environment.mkdir()
    (environment / "Dockerfile").write_text(
        "FROM verlai/verl:vllm017.latest\n"
        "WORKDIR /workspace\n"
        "RUN mkdir -p /models /hidden\n"
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
            "task_id": task_id,
            "family": family,
            "launch_script": config["launch"],
            "targets": targets,
            "validation_stack": config["validation_stack"],
            "verify": public["verify"],
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
        "family": family,
        "targets": len(targets),
        "metric_status": metrics["metric_status"],
        "eval_last": metrics["eval_last"],
        "already_at_baseline": all(
            item["already_at_baseline"] for item in target_records
        ),
    }


def main() -> int:
    args = parse_args()
    be_root = args.be_root.resolve()
    destination_root = args.dst.resolve()
    selected = (
        FAMILIES
        if args.family == "all"
        else {args.family: FAMILIES[args.family]}
    )
    results = []
    for family, config in selected.items():
        if args.force:
            for task_id in REMOVED_TASKS.get(family, ()):
                stale = destination_root / family / task_id
                if stale.exists():
                    shutil.rmtree(stale)
        for task_id in config["tasks"]:
            print(f"materializing {family}/{task_id}...", flush=True)
            results.append(
                materialize_one(
                    task_id,
                    family,
                    config,
                    be_root,
                    destination_root,
                    args.force,
                )
            )
        write_json(
            destination_root / family / "manifest.json",
            {"family": family, "tasks": [r for r in results if r["family"] == family]},
        )
    write_json(
        destination_root / "verl_implementation_manifest.json",
        {"tasks": results},
    )
    print(json.dumps({"tasks": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
