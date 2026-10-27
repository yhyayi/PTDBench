#!/usr/bin/env bash
set -euo pipefail

python3 "$PTDBENCH_ENVIRONMENT/ptdbench_fetch.py" \
    --repo LIF1014/ptdbench-Llama-3.2-1B-Instruct \
    --dest "$PTDBENCH_WORKSPACE/verl/Llama-3.2-1B-instruct"

python3 "$PTDBENCH_ENVIRONMENT/ptdbench_dataset.py" \
    --recipe dapo_math_17k \
    --dest "$PTDBENCH_WORKSPACE/DAPO-Math-17k_verl"
