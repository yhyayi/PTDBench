#!/usr/bin/env python3
"""Materialize six selected Qwen DAPO hparam tasks as PTDBench B0 tasks."""

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
    "task_hparam_overlong_023",
    "task_hparam_seqlen_micro_014",
    "task_hparam_clip_grpo_022",
    "task_hparam_rollout_temp_n_019",
    "task_hparam_rollout_len_sample_020",
    "task_hparam_batch_clip_015",
)
METRIC_KEY = "val-core/math_dapo/acc/mean@1"
LAUNCH_REL = Path(
    "examples/grpo_trainer/"
    "run_qwen2_5_0_5b_instruct_dapo_math_fsdp.sh"
)
HPARAM_RE_TEMPLATE = r"^(\s*{name}=\$\{{\s*{name}\s*:-)(.*)(\}})\s*$"


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
        default=data_call / "tasks/qwen_dapo_hparam",
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
    """Remove comments that reveal benchmark hollowing or candidate regions."""
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
    in_hollow_guard = False
    saw_guard_exit = False
    for line in lines:
        normalized = line.strip().lower()
        if normalized.startswith("_hollow_sentinel="):
            in_hollow_guard = True
        if in_hollow_guard:
            removed += 1
            if normalized == "exit 2":
                saw_guard_exit = True
            elif saw_guard_exit and normalized == "fi":
                in_hollow_guard = False
            continue
        is_comment = normalized.startswith("#")
        leaks_hollowing = (
            "hollow" in normalized
            or "挖空" in normalized
            or "user-adjustable" in normalized
        )
        if is_comment and leaks_hollowing:
            removed += 1
            continue
        kept.append(line)
    if in_hollow_guard:
        raise ValueError(f"unterminated hollow guard in {path}")
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


def normalize_shell_default(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return text


def extract_hparams(text: str, names: List[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for name in names:
        pattern = re.compile(
            HPARAM_RE_TEMPLATE.format(name=re.escape(name)),
            flags=re.MULTILINE,
        )
        match = pattern.search(text)
        if not match:
            raise ValueError(f"missing shell default for {name}")
        result[name] = normalize_shell_default(match.group(2))
    return result


def copy_workspace(source: Path, target: Path) -> None:
    shutil.copytree(
        source,
        target,
        symlinks=False,
        ignore=copy_ignore,
    )
    (target / "outputs").mkdir(parents=True, exist_ok=True)


def load_metrics(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    points = payload.get("metrics", {}).get("eval_points") or []
    if not points:
        raise ValueError(f"{path}: no eval points")
    for point in points:
        if point.get("eval_key") != METRIC_KEY:
            raise ValueError(f"{path}: unexpected metric key {point.get('eval_key')}")
    return payload


def source_log_hashes(workspace: Path, metrics: Dict[str, Any]) -> List[Dict[str, str]]:
    records = []
    for relative in metrics.get("logs", {}).get("source_logs") or []:
        path = workspace / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append({"path": relative, "sha256": sha256_file(path)})
    return records


def attempts_summary(outputs: Path) -> Dict[str, Any]:
    attempts_path = outputs / "run_attempts.json"
    attempts = (
        json.loads(attempts_path.read_text(encoding="utf-8"))
        if attempts_path.exists()
        else []
    )
    failures = [
        {
            "attempt": item.get("attempt"),
            "returncode": item.get("returncode"),
            "failure_reason": item.get("failure_reason"),
            "startup_failed": item.get("startup_failed"),
        }
        for item in attempts
        if not item.get("ok")
    ]
    return {
        "n_attempts": len(attempts),
        "n_failed": len(failures),
        "failures": failures,
        "final_ok": bool(attempts and attempts[-1].get("ok")),
        "final_attempt": attempts[-1].get("attempt") if attempts else None,
    }


def checkpoint_manifest(source_workspace: Path) -> Dict[str, Any]:
    root = source_workspace / "checkpoints"
    return {
        "excluded_from_task": True,
        "reason": "Checkpoints are 13-61GB per task and are not needed for hparam diagnosis.",
        "source_present": root.exists(),
        "top_level_entries": (
            sorted(path.name for path in root.iterdir()) if root.exists() else []
        ),
    }


def protected_hashes(workspace: Path) -> Dict[str, str]:
    excluded_prefixes = (
        str(LAUNCH_REL),
        "outputs/",
        "historical_evidence/",
    )
    records = {}
    for path in iter_files(workspace) or []:
        relative = str(path.relative_to(workspace))
        if relative == str(LAUNCH_REL):
            continue
        if any(relative.startswith(prefix) for prefix in excluded_prefixes[1:]):
            continue
        records[relative] = sha256_file(path)
    return records


def render_instruction(
    task_id: str,
    public: Dict[str, Any],
    metrics: Dict[str, Any],
) -> str:
    return f"""# Qwen DAPO run health assessment

Assess the historical Verl GRPO run. Determine from the workspace evidence
whether it is healthy, abnormal, incomplete, or indeterminate. Do not assume in
advance that a particular configuration value, source component, runtime
condition, or prior change is responsible.

The expected validation signal is `{METRIC_KEY}`. Evidence is available under
`outputs/`, `historical_evidence/`, `logs/`, and `wandb/`. Correlate
the metric trajectory with runtime behavior and the effective configuration.

## Constraints

- If and only if the evidence supports a repair, make the minimal necessary
  change. Do not change files merely because they look unusual.
- Keep `TOTAL_TRAINING_STEPS=200`, `TEST_FREQ=20`, model/data/reward paths,
  GPU settings, and Hydra arrays unchanged.
- Do not edit training data, reward/evaluation code, historical logs, or wandb.
- Checkpoints were intentionally excluded because they are tens of GB and are
  not required to assess the run.

## Deliverables

1. Write `outputs/diagnosis.txt` in English. State a verdict
   (`healthy`, `abnormal`, `incomplete`, or `indeterminate`), cite evidence,
   explain the causal chain, and separate root causes from symptoms.
2. Write executable `outputs/run_command.sh` containing:

```bash
#!/usr/bin/env bash
set -euo pipefail
bash {LAUNCH_REL}
```

Do not launch the full 200-step GPU job yourself; the verifier owns execution.
"""


def metadata_for(
    task_id: str,
    public: Dict[str, Any],
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    points = metrics["metrics"]["eval_points"]
    return {
        "title": f"Qwen DAPO hparam diagnosis: {task_id}",
        "submitter": {"name": "benchmark_evaluation"},
        "license": "MIT",
        "created_date": "2026-08-08",
        "domain": "post_train",
        "model": {"name": public["settings"]["model"]},
        "dataset": {"name": "DAPO-Math-17k"},
        "training": {
            "framework": "verl",
            "algo": "grpo",
        },
        "environment": {"dockerfile": "environment/Dockerfile"},
        "outcome": {
            "failure_type": "metric_failure",
            "failure_reason": (
                "Previous Agent launch hyperparameters produced a weak or "
                "incomplete validation trajectory"
            ),
        },
        "bench_category": "qwen_dapo_hparam",
        "source_task": (
            "Benchmark_Evaluation/tasks/task_qwen_dapo_hparam/" + task_id
        ),
        "evidence_kind": "dirty_agent_failure",
        "packaging_tier": "B0",
        "packaging_ready": True,
        "b1_scoring_ready": False,
        "hollow_hparams": ",".join(public["hollow_hparams"]),
        "verify": {
            "num_gpus": int(public["verify"]["num_gpus"]),
            "timeout_minutes": int(public["verify"]["timeout_minutes"]),
            "max_train_steps": int(public["verify"]["max_train_steps"]),
            "eval_interval": int(public["settings"]["test_freq"]),
            "eval_metric_key": METRIC_KEY,
        },
        "historical_baseline": {
            "eval_first": points[0]["eval_score"],
            "eval_last": points[-1]["eval_score"],
            "n_eval_points": len(points),
            "attempt": metrics.get("attempt"),
            "first_rollout_ok": bool(metrics.get("first_rollout_ok")),
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
    be_root: Path,
    destination_root: Path,
    force: bool,
) -> Dict[str, Any]:
    source = be_root / "tasks/task_qwen_dapo_hparam" / task_id
    target = destination_root / task_id
    if target.exists():
        if not force:
            raise FileExistsError(f"{target} exists; pass --force to replace")
        shutil.rmtree(target)
    target.mkdir(parents=True)

    public = read_yaml(source / "task.yaml")
    hidden = read_yaml(source / "hidden/task.yaml")
    if Path(public["launch_script"]) != LAUNCH_REL:
        raise ValueError(f"{task_id}: unexpected launch script")

    source_workspace = source / "workspace"
    dirty_launch = source_workspace / LAUNCH_REL
    baseline_launch = source / hidden["baseline_path"]
    if sha256_file(baseline_launch) != hidden["baseline_sha256"]:
        raise ValueError(f"{task_id}: baseline sha256 mismatch")
    dirty_text = dirty_launch.read_text(encoding="utf-8")
    baseline_text = baseline_launch.read_text(encoding="utf-8")
    names = list(public["hollow_hparams"])
    dirty_hparams = extract_hparams(dirty_text, names)
    golden_hparams = extract_hparams(baseline_text, names)
    if any(value == "__HOLLOW__" for value in dirty_hparams.values()):
        raise ValueError(f"{task_id}: target hparam still contains __HOLLOW__")

    copy_workspace(source_workspace, target / "workspace")
    shutil.copytree(source / "hidden", target / "oracle")
    secret_redactions = {
        "workspace": redact_runtime_secrets(target / "workspace"),
        "oracle": redact_runtime_secrets(target / "oracle"),
    }
    removed_agent_outputs = clean_agent_outputs(target / "workspace")
    packaged_dirty_launch = target / "workspace" / LAUNCH_REL
    packaged_baseline_launch = (
        target
        / "oracle"
        / Path(hidden["baseline_path"]).relative_to("hidden")
    )
    workspace_annotation_cleanup = strip_workspace_annotations(
        target / "workspace"
    )
    attribution_cleanup = strip_attribution_annotations(target / "workspace")
    annotation_cleanup = {
        "workspace_files": workspace_annotation_cleanup,
        "attribution_annotations": attribution_cleanup,
        "oracle_baseline": strip_hollow_annotations(packaged_baseline_launch),
        "secret_redactions": secret_redactions,
        "removed_agent_outputs": removed_agent_outputs,
    }
    packaged_dirty_sha = sha256_file(packaged_dirty_launch)
    packaged_golden_sha = sha256_file(packaged_baseline_launch)
    packaged_hidden = read_yaml(target / "oracle/task.yaml")
    packaged_hidden["baseline_sha256"] = packaged_golden_sha
    write_yaml(target / "oracle/task.yaml", packaged_hidden)

    metrics = load_metrics(source_workspace / "outputs/metrics.json")
    attempts = attempts_summary(source_workspace / "outputs")
    points = metrics["metrics"]["eval_points"]
    manifest = {
        "schema_version": 1,
        "evidence_kind": "dirty_agent_failure",
        "task_id": task_id,
        "metric_key": METRIC_KEY,
        "eval_interval": 20,
        "eval_points": points,
        "eval_first": points[0]["eval_score"],
        "eval_last": points[-1]["eval_score"],
        "eval_min": min(point["eval_score"] for point in points),
        "eval_max": max(point["eval_score"] for point in points),
        "first_rollout_ok": bool(metrics.get("first_rollout_ok")),
        "attempt": metrics.get("attempt"),
        "run_once": metrics.get("run_once", {}),
        "attempts": attempts,
        "source_logs": source_log_hashes(target / "workspace", metrics),
        "dirty_hparams": dirty_hparams,
        "golden_hparams": golden_hparams,
        "annotation_cleanup": annotation_cleanup,
        "checkpoints": checkpoint_manifest(source_workspace),
    }
    manifest["metrics_alignment"] = align_outputs_metrics(
        target / "workspace",
        manifest,
        task_id=task_id,
    )
    public_manifest = {
        key: value
        for key, value in manifest.items()
        if key not in {"task_id", "dirty_hparams", "golden_hparams"}
    }
    evidence = target / "workspace/historical_evidence"
    write_json(evidence / "manifest.json", public_manifest)
    write_json(evidence / "eval_points.json", points)
    write_json(evidence / "run_attempts_summary.json", attempts)
    (evidence / "README.txt").write_text(
        "Frozen evidence from the historical run. Checkpoints were excluded.\n",
        encoding="utf-8",
    )
    (evidence / "log_excerpt.txt").write_text(
        "\n".join(
            f"step={point['step']} score={point['eval_score']:.9f}"
            for point in points
        )
        + "\n",
        encoding="utf-8",
    )

    write_json(target / "oracle/historical_evidence_manifest.json", manifest)
    write_json(target / "oracle/dirty_hparams.json", dirty_hparams)
    write_json(target / "oracle/golden_hparams.json", golden_hparams)
    write_json(
        target / "oracle/dirty_launch.json",
        {"path": str(LAUNCH_REL), "sha256": packaged_dirty_sha},
    )
    write_json(
        target / "oracle/golden_launch.json",
        {
            "path": str(LAUNCH_REL),
            "sha256": packaged_golden_sha,
        },
    )
    write_json(
        target / "oracle/checkpoint_exclusion_manifest.json",
        manifest["checkpoints"],
    )
    write_yaml(
        target / "oracle/scoring_rubric.yaml",
        {
            "schema_version": 1,
            "eval_metric": METRIC_KEY,
            "eval_interval": 20,
            "max_train_steps": 200,
            "b0_structural_only": True,
            "b1_requires_gpu": True,
            "pass_requires": {
                "launch_matches_golden_sha256": True,
                "protected_workspace_unchanged": True,
                "first_rollout_ok": True,
                "metric_repaired": True,
            },
        },
    )
    write_json(
        target / "oracle/protected_hashes.json",
        protected_hashes(target / "workspace"),
    )
    write_json(
        target / "oracle/source_snapshot/hashes.json",
        {
            "source_task_yaml": sha256_file(source / "task.yaml"),
            "source_hidden_task_yaml": sha256_file(source / "hidden/task.yaml"),
            "dirty_launch": sha256_file(dirty_launch),
            "packaged_dirty_launch": packaged_dirty_sha,
            "workspace_seed_launch": sha256_file(
                source / "workspace_seed" / LAUNCH_REL
            ),
            "source_golden_launch": hidden["baseline_sha256"],
            "packaged_golden_launch": packaged_golden_sha,
            "annotation_cleanup": annotation_cleanup,
        },
    )
    (target / "oracle/source_snapshot/workspace_diff.txt").write_text(
        "Dirty Agent launch and seed launch hashes are recorded in hashes.json.\n",
        encoding="utf-8",
    )

    (target / "instruction.md").write_text(
        render_instruction(task_id, public, metrics),
        encoding="utf-8",
    )
    (target / "readme.md").write_text(
        f"# {task_id}\n\n"
        "PTDBench B0 package of a dirty Qwen DAPO hparam workspace.\n\n"
        f"- Previous attempt: {metrics.get('attempt')}\n"
        f"- Eval points: {len(points)}\n"
        "- Checkpoints excluded; logs/wandb and old outputs retained.\n"
        "- B1 requires a local 2-GPU runner.\n",
        encoding="utf-8",
    )
    write_yaml(target / "metadata.yaml", metadata_for(task_id, public, metrics))

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
            "launch_script": str(LAUNCH_REL),
            "hollow_hparams": names,
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
        "n_eval_points": len(points),
        "eval_last": points[-1]["eval_score"],
        "attempt": metrics.get("attempt"),
        "checkpoint_excluded": True,
    }


def main() -> int:
    args = parse_args()
    be_root = args.be_root.resolve()
    destination_root = args.dst.resolve()
    if not be_root.is_dir():
        raise FileNotFoundError(be_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    tasks = []
    for task_id in TASK_IDS:
        print(f"materializing {task_id}...", flush=True)
        tasks.append(
            materialize_one(task_id, be_root, destination_root, args.force)
        )
    write_json(destination_root / "manifest.json", {"tasks": tasks})
    print(json.dumps({"tasks": tasks}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
