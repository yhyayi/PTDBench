#!/usr/bin/env bash
# GRPO | Qwen2.5-0.5B-Instruct | FSDP training | NVIDIA GPUs or Ascend NPUs
#
# INFER_BACKEND controls rollout backend: vllm
#
# Defaults tuned for a small smoke / debug run:
#   train_batch_size=4, rollout.n=8, total_training_steps=100

set -euo pipefail

# ---- logging / wandb ----
# Host path: /REDACTED/BENCH_HOME/verl/logs  <->  container: /workspace/verl/logs
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${LOG_DIR:-${VERL_ROOT}/logs}"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/qwen2_5_0.5b_instruct_fsdp_${TIMESTAMP}.log"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in the runtime environment}"
export WANDB_PROJECT="${WANDB_PROJECT:-verl_grpo_gsm8k}"
# ---- end logging / wandb ----

# DEVICE is auto-detected by probing torch_npu; override only for special cases.
DEVICE=${DEVICE:-$(python3 -c 'import torch_npu' 2>/dev/null && echo npu || echo gpu)}
INFER_BACKEND=${INFER_BACKEND:-vllm}

# Resolve model path for host / docker. In verl-lyh, mount e.g.:
#   -v /REDACTED/BENCH_HOME/Qwen2.5-0.5B-Instruct:/models/Qwen2.5-0.5B-Instruct
if [[ -z "${MODEL_PATH:-}" ]]; then
    if [[ -d /models/Qwen2.5-0.5B-Instruct ]]; then
        MODEL_PATH=/models/Qwen2.5-0.5B-Instruct
    elif [[ -d /workspace/verl/Qwen2.5-0.5B-Instruct ]]; then
        MODEL_PATH=/workspace/verl/Qwen2.5-0.5B-Instruct
    else
        MODEL_PATH=/REDACTED/BENCH_HOME/Qwen2.5-0.5B-Instruct
    fi
fi

if [[ -z "${TRAIN_FILE:-}" ]]; then
    # Prefer /workspace/gsm8k_verl (Benchmark mother layout); keep legacy paths.
    if [[ -f /workspace/gsm8k_verl/train.parquet ]]; then
        TRAIN_FILE=/workspace/gsm8k_verl/train.parquet
    elif [[ -f /workspace/verl/gsm8k_verl/train.parquet ]]; then
        TRAIN_FILE=/workspace/verl/gsm8k_verl/train.parquet
    elif [[ -f /datasets/gsm8k_verl/train.parquet ]]; then
        TRAIN_FILE=/datasets/gsm8k_verl/train.parquet
    else
        TRAIN_FILE=$HOME/data/gsm8k/train.parquet
    fi
fi
if [[ -z "${TEST_FILE:-}" ]]; then
    if [[ -f /workspace/gsm8k_verl/test.parquet ]]; then
        TEST_FILE=/workspace/gsm8k_verl/test.parquet
    elif [[ -f /workspace/verl/gsm8k_verl/test.parquet ]]; then
        TEST_FILE=/workspace/verl/gsm8k_verl/test.parquet
    elif [[ -f /datasets/gsm8k_verl/test.parquet ]]; then
        TEST_FILE=/datasets/gsm8k_verl/test.parquet
    else
        TEST_FILE=$HOME/data/gsm8k/test.parquet
    fi
fi

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-2}

# prompts per step; with ROLLOUT_N=8 -> 4*8=32 responses / step
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}

ACTOR_LR=${ACTOR_LR:-1e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}

# 0.5B fits on 1 GPU; keep vLLM TP=1
ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.4}
ROLLOUT_N=${ROLLOUT_N:-8}

PROJECT_NAME=${PROJECT_NAME:-verl_grpo_gsm8k}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-${BENCH_TASK_LEAF_ID:-task_engine_base_025}}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-20}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}

# V1 trainer requires TransferQueue (often missing in offline images). Use legacy runner.
USE_V1=${USE_V1:-False}

case "${DEVICE}" in
    gpu)
        ;;
    npu)
        NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
        ;;
    *)
        echo "Unsupported DEVICE=${DEVICE}. Expected 'gpu' or 'npu'." >&2
        exit 1
        ;;
esac

########################### parameter arrays ###########################

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
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

TRAINER=(
    trainer.critic_warmup=0
    # Metrics still go through verl console logger, but process stdout/stderr
    # are redirected to LOG_FILE only (no terminal spam).
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

EXTRA=(
)

########################### launch ###########################
# Only print log path on console; all training output goes to the log file.
echo "Logging to ${LOG_FILE} (no console training output)"
export PYTHONPATH="${VERL_ROOT}:${PYTHONPATH:-}"

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@" \
    >"${LOG_FILE}" 2>&1
