"""Shared secret-redaction helpers for PTDBench materializers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List


TEXT_SUFFIXES = {
    ".json",
    ".log",
    ".md",
    ".py",
    ".sh",
    ".txt",
    ".yaml",
    ".yml",
}
WANDB_ASSIGNMENT_RE = re.compile(
    r'^\s*export\s+WANDB_API_KEY\s*=\s*["\'][^"\']+["\']\s*$',
    flags=re.MULTILINE,
)
WANDB_TOKEN_RE = re.compile(r"wandb_v1_[A-Za-z0-9_]+")
ANNOTATION_REPLACEMENTS = {
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


def redact_runtime_secrets(root: Path) -> Dict[str, int]:
    """Remove embedded W&B credentials while keeping launch scripts runnable."""
    changed: Dict[str, int] = {}
    if not root.exists():
        return changed
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        updated, assignments = WANDB_ASSIGNMENT_RE.subn(
            ': "${WANDB_API_KEY:?Set WANDB_API_KEY in the runtime environment}"',
            text,
        )
        updated, tokens = WANDB_TOKEN_RE.subn("[REDACTED_WANDB_API_KEY]", updated)
        count = assignments + tokens
        if count:
            path.write_text(updated, encoding="utf-8")
            changed[str(path.relative_to(root))] = count
    return changed


def strip_attribution_annotations(root: Path) -> Dict[str, int]:
    """Remove benchmark-generation notes without removing normal code comments."""
    changed: Dict[str, int] = {}
    if not root.exists():
        return changed
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".py", ".sh"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        count = 0
        for old, new in ANNOTATION_REPLACEMENTS.items():
            hits = text.count(old)
            if hits:
                text = text.replace(old, new)
                count += hits
        kept: List[str] = []
        for line in text.splitlines(keepends=True):
            normalized = line.strip().lower()
            is_comment = normalized.startswith("#")
            packaging_note = (
                "hollow" in normalized
                or "挖空" in normalized
                or "category: qwen_dapo_hparam" in normalized
                or normalized.startswith("# stub for category=")
                or normalized.startswith("# stub for reward_design")
                or "benchmark tasks replace task_" in normalized
                or (
                    normalized.startswith("# bench:")
                    and (
                        "baseline" in normalized
                        or "sealed" in normalized
                        or "agent" in normalized
                    )
                )
            )
            if is_comment and packaging_note:
                count += 1
                continue
            kept.append(line)
        if count:
            path.write_text("".join(kept), encoding="utf-8")
            changed[str(path.relative_to(root))] = count
    return changed


def clean_agent_outputs(workspace: Path) -> List[str]:
    """Remove prior-Agent answers and attribution-only helper artifacts."""
    removed = []
    for relative in (
        "outputs/diagnosis.txt",
        "outputs/agent_loop_impl.py",
    ):
        path = workspace / relative
        if path.is_file():
            path.unlink()
            removed.append(relative)
    return removed


def _metric_view(manifest: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(manifest.get("held_out_eval"), dict):
        return manifest["held_out_eval"]
    if isinstance(manifest.get("metrics"), dict):
        return manifest["metrics"]
    return manifest


def align_outputs_metrics(
    workspace: Path,
    manifest: Dict[str, Any],
    *,
    task_id: str,
) -> Dict[str, Any]:
    """Make Agent-visible metrics faithfully mirror frozen historical evidence."""
    path = workspace / "outputs/metrics.json"
    payload: Dict[str, Any] = {}
    if path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            payload = loaded

    view = _metric_view(manifest)
    points = list(view.get("eval_points") or [])
    metric_key = (
        view.get("expected_metric_key")
        or view.get("actual_metric_key")
        or view.get("metric_key")
        or manifest.get("expected_metric_key")
        or manifest.get("metric_key")
        or (points[-1].get("eval_key") if points else None)
    )
    status = (
        view.get("metric_status")
        or manifest.get("metric_status")
        or ("available" if points else "missing")
    )
    eval_interval = int(
        view.get("eval_interval")
        or manifest.get("eval_interval")
        or 20
    )
    metrics = {
        "eval_interval": eval_interval,
        "metric_key": metric_key,
        "metric_status": status,
        "eval_points": points,
        "eval_scores": [
            {"step": point["step"], "score": point["eval_score"]}
            for point in points
        ],
    }

    source_logs = (
        manifest.get("source_logs")
        or view.get("source_logs")
        or []
    )
    logs = dict(payload.get("logs") or {})
    logs["source_logs"] = [
        item["path"] if isinstance(item, dict) else str(item)
        for item in source_logs
    ]
    payload.update(
        {
            "task_id": payload.get("task_id") or task_id,
            "scoring_note": (
                "Historical validation metrics frozen from the original run."
            ),
            "logs": logs,
            "metrics": metrics,
        }
    )
    for key in ("run_once", "first_rollout_ok", "attempt"):
        value = view.get(key, manifest.get(key))
        if value is not None:
            payload[key] = value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    source_metrics = manifest.get("source_metrics")
    if isinstance(source_metrics, dict):
        source_metrics["path"] = "outputs/metrics.json"
        source_metrics["source_eval_points"] = len(points)
    return {
        "path": "outputs/metrics.json",
        "metric_key": metric_key,
        "metric_status": status,
        "n_eval_points": len(points),
    }
