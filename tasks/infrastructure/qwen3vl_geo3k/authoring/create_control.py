"""Reconstruct the recorded control from the hidden, upstream reference file."""
import hashlib
import json
from pathlib import Path
import shutil


def create_control(workspace, destination, oracle):
    workspace, destination, oracle = map(Path, (workspace, destination, oracle))
    repo = workspace / "verl"
    recorded = json.loads((oracle / "source_manifest.json").read_text())
    for name, expected in recorded["files"].items():
        path = repo / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Recorded source changed before authoring qualification: {name}")
    if destination.exists():
        raise RuntimeError(f"Control already exists: {destination}; use a fresh attempt directory")
    destination.mkdir(parents=True)
    shutil.copytree(repo, destination / "verl", ignore=shutil.ignore_patterns("__pycache__", ".git", ".pytest_cache"))
    shutil.copy2(oracle / recorded["control_overlay"], destination / "verl" / recorded["target"])
    shutil.copy2(workspace / "run_qwen3vl_geo3k_grpo.sh", destination / "run_qwen3vl_geo3k_grpo.sh")
    for name in ("models", "data"):
        (destination / name).symlink_to((workspace / name).resolve(), target_is_directory=True)
    return recorded["upstream_commit"]
