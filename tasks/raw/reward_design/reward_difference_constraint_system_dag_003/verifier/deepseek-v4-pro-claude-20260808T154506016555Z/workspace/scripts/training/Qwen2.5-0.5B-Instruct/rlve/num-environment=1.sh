#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 WANDB_PROJECT"
    exit 1
fi

WANDB_PROJECT=$1

bash scripts/training/Qwen2.5-0.5B-Instruct/rlve.sh "${WANDB_PROJECT}" \
    "[Qwen2.5-0.5B-Instruct]_[num-environment=1]" \
    "PolyaModel"
