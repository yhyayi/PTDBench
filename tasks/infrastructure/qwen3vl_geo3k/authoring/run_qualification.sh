#!/usr/bin/env bash
# Maintainer entrypoint: prepared inputs -> paired measurement -> both RL runs.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="${QWEN3VL_RUNTIME:-current}"
RUN_DIR="${QWEN3VL_RUN_DIR:-$ROOT/authoring/runs/qualification-$(date -u +%Y%m%dT%H%M%S%N)-$$}"
[[ ! -e "$RUN_DIR" ]] || { printf 'Use a fresh QWEN3VL_RUN_DIR; existing output is preserved: %s\n' "$RUN_DIR" >&2; exit 2; }
mkdir -p "$RUN_DIR"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
export PTDBENCH_WORKSPACE="${PTDBENCH_WORKSPACE:-$ROOT/workspace}"
exec > >(tee "$RUN_DIR/qualification.log") 2>&1
printf 'Qualification runtime=%s workspace=%s output=%s\n' "$RUNTIME" "$PTDBENCH_WORKSPACE" "$RUN_DIR"
case "$RUNTIME" in
    current) exec bash "$ROOT/authoring/qualify.sh" --workspace "$PTDBENCH_WORKSPACE" --run-dir "$RUN_DIR" "$@" ;;
    *) printf 'QWEN3VL_RUNTIME must be current\n' >&2; exit 2 ;;
esac
