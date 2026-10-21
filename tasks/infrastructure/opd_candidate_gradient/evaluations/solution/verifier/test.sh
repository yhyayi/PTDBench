#!/usr/bin/env bash
set -euo pipefail
: "${PTDBENCH_WORKSPACE:?PTDBENCH_WORKSPACE is required}"
: "${PTDBENCH_RESULTS:?PTDBENCH_RESULTS is required}"
: "${PTDBENCH_VERIFIER:?PTDBENCH_VERIFIER is required}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PTDBENCH_WORKSPACE/verl" python3 \
  "$PTDBENCH_VERIFIER/tests/test_candidate_gradient.py"

if [[ "${PTDBENCH_RUN_FULL_OPD:-0}" == "1" ]]; then
    # shellcheck disable=SC1091
    set -a
    source "$PTDBENCH_VERIFIER/eval_params.env"
    set +a
    # prepare.sh fetches both checkpoints under $PTDBENCH_WORKSPACE/model/,
    # which is exactly where opd.sh's defaults look; no external mount needed.
    export ACTOR_MODEL_PATH="$PTDBENCH_WORKSPACE/model/DeepSeek-R1-Distill-Qwen-1.5B"
    export TEACHER_MODEL_PATH="$PTDBENCH_WORKSPACE/model/JustRL-DeepSeek-1.5B"
    export TRAIN_DATASET="$PTDBENCH_WORKSPACE/datasets/dapo-math-17k-processed.parquet"
    export TEST_DATASET="['$PTDBENCH_WORKSPACE/datasets/test_data/AIME26/test.parquet']"
    export WORKSPACE_ROOT="$PTDBENCH_WORKSPACE"
    export REPO_ROOT="$PTDBENCH_WORKSPACE/verl"
    export LOGGER=console
    export SWANLAB_MODE=offline
    cd "$PTDBENCH_WORKSPACE"
    ./opd.sh \
      trainer.total_training_steps="${TRAINER_TOTAL_STEPS:-1}" \
      trainer.val_before_train=False \
      trainer.log_val_generations=0
fi

printf 'solution regression passed\n'
