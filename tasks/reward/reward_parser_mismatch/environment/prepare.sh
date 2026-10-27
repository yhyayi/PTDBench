#!/usr/bin/env bash
set -euo pipefail

python3 "$PTDBENCH_ENVIRONMENT/ptdbench_fetch.py" \
    --repo "Qwen/Qwen3-4B-Base" \
    --dest "$PTDBENCH_WORKSPACE/Qwen3-4B-Base"
