#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 WANDB_PROJECT [RUN_NAME] [EXTRA_TRAIN_ARGS...]"
    exit 1
fi

WANDB_PROJECT=$1
if [ $# -ge 2 ]; then
    RUN_NAME=$2
    shift 2
else
    RUN_NAME="[Qwen2.5-0.5B-Instruct]_[num-environment=4]"
    shift 1
fi

bash scripts/training/Qwen2.5-0.5B-Instruct/rlve.sh "${WANDB_PROJECT}" \
    "${RUN_NAME}" \
    "Division EuclidGame Multiplication Sorting" \
    "$@"
