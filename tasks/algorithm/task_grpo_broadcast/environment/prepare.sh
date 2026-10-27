#!/usr/bin/env bash
set -euo pipefail

python3 "$PTDBENCH_ENVIRONMENT/ptdbench_fetch.py" \
    --repo Qwen/Qwen2.5-1.5B-Instruct \
    --dest "$PTDBENCH_WORKSPACE/Qwen2.5-1.5B-Instruct"

python3 "$PTDBENCH_ENVIRONMENT/ptdbench_fetch.py" \
    --type dataset \
    --repo cyh2004/ptdbench-taco-verl-easy \
    --dest "$PTDBENCH_WORKSPACE/taco_verl"
