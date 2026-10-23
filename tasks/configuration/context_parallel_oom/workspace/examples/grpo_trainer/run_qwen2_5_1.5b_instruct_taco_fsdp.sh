#!/usr/bin/env bash
# GRPO | Qwen2.5-1.5B-Instruct | TACO | 8 GPUs
# Context parallel (FSDP Ulysses) defaults to 1. max_response_length is 30720
# and must stay 30720: 2048 + 30720 = 32768, the model context limit.
# EOS is honored. A degenerate repetition can still grow a response up to that cap.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${LOG_DIR:-${VERL_ROOT}/logs}"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/qwen2_5_1.5b_instruct_taco_fsdp_${TIMESTAMP}.log"

export WANDB_DIR="${WANDB_DIR:-${LOG_DIR}}"
export WANDB_PROJECT="${WANDB_PROJECT:-verl_grpo_taco}"
export WANDB_MODE=offline
export WANDB_API_KEY=offline

DEVICE=${DEVICE:-$(python3 -c 'import torch_npu' 2>/dev/null && echo npu || echo gpu)}
INFER_BACKEND=${INFER_BACKEND:-vllm}

MODEL_PATH="${MODEL_PATH:-${VERL_ROOT}/Qwen2.5-1.5B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-${VERL_ROOT}/taco_verl/train.parquet}"
TEST_FILE="${TEST_FILE:-${VERL_ROOT}/taco_verl/test.parquet}"
TACO_REWARD_PATH="${TACO_REWARD_PATH:-${VERL_ROOT}/verl/utils/reward_score/taco.py}"

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-30720}
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
# FSDP context parallel. This script has no separate `cp` flag; cp is
# ulysses_sequence_parallel_size on the actor and the ref.
ULYSSES_SEQUENCE_PARALLEL_SIZE=${ULYSSES_SEQUENCE_PARALLEL_SIZE:-1}
ENABLE_GRADIENT_CHECKPOINTING=${ENABLE_GRADIENT_CHECKPOINTING:-False}

ACTOR_LR=${ACTOR_LR:-1e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0}
USE_KL_LOSS=${USE_KL_LOSS:-False}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}

ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.4}
ROLLOUT_N=${ROLLOUT_N:-8}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.7}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-True}
REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-32}

PROJECT_NAME=${PROJECT_NAME:-verl_grpo_taco}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-taco_qwen2_5_1.5b}
# test_freq stays above the step budget so this run is the training updates,
# not a held-out eval.
TEST_FREQ=${TEST_FREQ:-1000}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
# save_freq must be >0 or verl skips the final save too; equal to total steps => only last ckpt.
SAVE_FREQ=${SAVE_FREQ:-${TOTAL_TRAINING_STEPS}}
USE_V1=${USE_V1:-False}

case "${DEVICE}" in
    gpu) ;;
    npu) NGPUS_PER_NODE=${NGPUS_PER_NODE:-8} ;;
    *)
        echo "Unsupported DEVICE=${DEVICE}. Expected 'gpu' or 'npu'." >&2
        exit 1
        ;;
esac

DATA=(
    algorithm.adv_estimator=grpo
    data.train_files=${TRAIN_FILE}
    data.val_files=${TEST_FILE}
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.dataloader_num_workers=0
    algorithm.use_kl_in_reward=False
)

MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=${ENABLE_GRADIENT_CHECKPOINTING}
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS}
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ULYSSES_SEQUENCE_PARALLEL_SIZE}
)

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_MODEL_LEN}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE}
    actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE}
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${ULYSSES_SEQUENCE_PARALLEL_SIZE}
)

REWARD=(
    reward.num_workers=${REWARD_NUM_WORKERS}
    reward.reward_manager.name=naive
    reward.custom_reward_function.path=${TACO_REWARD_PATH}
    reward.custom_reward_function.name=compute_score
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
    trainer.use_v1=${USE_V1}
)

echo "Logging to ${LOG_FILE}"
export PYTHONPATH="${VERL_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export RAY_DEDUP_LOGS=0

dump_failure_context() {
    local status=$1
    {
        echo
        echo "===== failure context (exit ${status}) $(date -Is) ====="
        echo "--- nvidia-smi ---"
        nvidia-smi || true
        echo "--- dmesg oom/kill (last 40) ---"
        dmesg -T 2>/dev/null | grep -Ei 'out of memory|killed process|oom-kill|Out of memory' | tail -n 40 || true
        echo "--- ray worker logs ---"
        local ray_root="${RAY_TMPDIR:-/tmp/ray}"
        local ray_logs="${ray_root}/session_latest/logs"
        if [[ -d "${ray_logs}" ]]; then
            echo "ray logs dir: ${ray_logs}"
            find "${ray_logs}" -type f \( -name 'worker-*.err' -o -name 'worker-*.out' -o -name 'python-core-worker*.log' \) -printf '%T@ %p\n' \
                | sort -nr | head -n 8 | while read -r _ path; do
                    echo "----- ${path} (tail) -----"
                    tail -n 120 "${path}" || true
                done
        else
            echo "no ray logs at ${ray_logs}"
        fi
    } >>"${LOG_FILE}"
}

on_exit() {
    local status=$?
    if [[ ${status} -ne 0 ]]; then
        dump_failure_context "${status}"
    fi
}
trap on_exit EXIT

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "$@" \
    >"${LOG_FILE}" 2>&1
