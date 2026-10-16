#!/usr/bin/env bash
# O3PD by default; TRAIN_MODE=opd_topk_jsd runs the matched JSD baseline.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export HF_HUB_OFFLINE=1
export PYTHONPATH="$ROOT/verl${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export VLLM_USE_FLASHINFER_SAMPLER=0
export TRAIN_MODE=${TRAIN_MODE:-o3pd}
case "$TRAIN_MODE" in o3pd|opd_topk_jsd) ;; *) echo 'Expected o3pd or opd_topk_jsd' >&2; exit 2 ;; esac
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-$ROOT/model/DeepSeek-R1-Distill-Qwen-1.5B}
export TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-$ROOT/model/DeepSeek-R1-Distill-Qwen-7B}
export TRAIN_DATASET=${TRAIN_DATASET:-$ROOT/datasets/dapo-math-17k-processed.parquet}
export OPD_TOP_K=${OPD_TOP_K:-8}
export OPD_STRATEGY=only_tch
export LOSS_AGG_MODE=seq-mean-token-mean
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-32}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-$MINI_BATCH_SIZE}
export N_RESPONSES=1
export ACTOR_LR=${ACTOR_LR:-1e-6}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-7168}
export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-31744}
export TEMPERATURE=${TEMPERATURE:-1.0}
export VAL_TEMPERATURE=1.0 VAL_TOP_P=1.0 VAL_N=${VAL_N:-16}
export TEACHER_MODEL_DTYPE=bf16
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}
export TEACHER_MAX_TOKEN_LEN_PER_GPU=${TEACHER_MAX_TOKEN_LEN_PER_GPU:-8192}
export OPD_CHUNK_SIZE=${OPD_CHUNK_SIZE:-128}
export USE_KL=False
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
IFS=',' read -ra O3PD_DEVICES <<< "$CUDA_VISIBLE_DEVICES"
export NGPUS_PER_NODE=${#O3PD_DEVICES[@]}
export RAY_START=0
export LOGGER=${LOGGER:-'[console]'}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-${TRAIN_MODE}_ds7b_ds1p5b_dapo17k_k${OPD_TOP_K}_seed42}
exec bash "$ROOT/opd.sh" \
    data.seed=42 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.data_loader_seed=42 \
    actor_rollout_ref.actor.fsdp_config.seed=42 \
    actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.teacher.fsdp_config.use_torch_compile=False \
    trainer.total_training_steps=200 \
    "ray_kwargs.ray_init.num_cpus=${O3PD_RAY_CPUS:-32}" \
    "+ray_kwargs.ray_init._temp_dir=/tmp/ray_o3pd_${USER}_$$" "$@"
