#!/usr/bin/env bash
# GRPO | Qwen2.5-0.5B-Instruct (dense) | FSDP + vLLM | DAPO-Math-17k
#
# Reward / advantage (defaults):
#   - algorithm.adv_estimator = grpo
#   - reward.reward_manager.name = dapo (overlong off by default)
#   - score: examples/grpo_trainer/reward_math_dapo_01.py::compute_score
#   - trainer.resume_mode = disable
#   - DAPO filter_groups / dynamic sampling is NOT enabled (default null)
#
# TEST_FREQ=20, reward fn path/name, experiment identity.

set -euo pipefail

# ---- logging / wandb ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${LOG_DIR:-${VERL_ROOT}/logs}"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/qwen2_5_0_5b_instruct_dapo_math_fsdp_${TIMESTAMP}.log"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in the runtime environment}"
export WANDB_PROJECT="${WANDB_PROJECT:-verl_grpo_dapo_math_qwen05b}"
# ---- end logging / wandb ----

# ---- infra (locked) ----
if [[ -z "${MODEL_PATH:-}" ]]; then
    if [[ -d /models/Qwen2.5-0.5B-Instruct ]]; then
        MODEL_PATH=/models/Qwen2.5-0.5B-Instruct
    elif [[ -d /workspace/verl/Qwen2.5-0.5B-Instruct ]]; then
        MODEL_PATH=/workspace/verl/Qwen2.5-0.5B-Instruct
    else
        MODEL_PATH=/home/test/test06/lyh/Qwen2.5-0.5B-Instruct
    fi
fi

if [[ -z "${TRAIN_FILE:-}" ]]; then
    if [[ -f /workspace/DAPO-Math-17k_verl/train.parquet ]]; then
        TRAIN_FILE=/workspace/DAPO-Math-17k_verl/train.parquet
    elif [[ -f /workspace/verl/datasets/DAPO-Math-17k_verl/train.parquet ]]; then
        TRAIN_FILE=/workspace/verl/datasets/DAPO-Math-17k_verl/train.parquet
    elif [[ -f /datasets/DAPO-Math-17k_verl/train.parquet ]]; then
        TRAIN_FILE=/datasets/DAPO-Math-17k_verl/train.parquet
    else
        TRAIN_FILE=/home/test/test06/lyh/datasets/DAPO-Math-17k_verl/train.parquet
    fi
fi
if [[ -z "${TEST_FILE:-}" ]]; then
    if [[ -f /workspace/DAPO-Math-17k_verl/test.parquet ]]; then
        TEST_FILE=/workspace/DAPO-Math-17k_verl/test.parquet
    elif [[ -f /workspace/verl/datasets/DAPO-Math-17k_verl/test.parquet ]]; then
        TEST_FILE=/workspace/verl/datasets/DAPO-Math-17k_verl/test.parquet
    elif [[ -f /datasets/DAPO-Math-17k_verl/test.parquet ]]; then
        TEST_FILE=/datasets/DAPO-Math-17k_verl/test.parquet
    else
        TEST_FILE=/home/test/test06/lyh/datasets/DAPO-Math-17k_verl/test.parquet
    fi
fi

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-2}
PROJECT_NAME=${PROJECT_NAME:-verl_grpo_dapo_math_qwen05b}
# Benchmark tasks replace task_hparam_rollout_len_sample_020 with leaf id.
EXPERIMENT_NAME=${EXPERIMENT_NAME:-${BENCH_TASK_LEAF_ID:-task_hparam_rollout_len_sample_020}}
# Fixed verify schedule (val-core every 20 steps through 200).
TEST_FREQ=${TEST_FREQ:-20}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-200}
USE_V1=${USE_V1:-False}
RESUME_MODE=${RESUME_MODE:-disable}
REWARD_FN_PATH=${REWARD_FN_PATH:-${SCRIPT_DIR}/reward_math_dapo_01.py}
REWARD_FN_NAME=${REWARD_FN_NAME:-compute_score}
# ---- end infra ----

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}

ACTOR_LR=${ACTOR_LR:-1e-6}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}
GRAD_CLIP=${GRAD_CLIP:-1.0}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-"-1"}
PPO_EPOCHS=${PPO_EPOCHS:-1}
LOSS_AGG_MODE=${LOSS_AGG_MODE:-token-mean}
CLIP_RATIO=${CLIP_RATIO:-0.2}
CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.2}
CLIP_RATIO_C=${CLIP_RATIO_C:-3.0}

USE_KL_LOSS=${USE_KL_LOSS:-False}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.0}
USE_KL_IN_REWARD=${USE_KL_IN_REWARD:-True}
KL_CTRL_COEF=${KL_CTRL_COEF:-0.01}
KL_PENALTY=${KL_PENALTY:-kl}
NORM_ADV_BY_STD_IN_GRPO=${NORM_ADV_BY_STD_IN_GRPO:-True}

ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.5}
ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-2048}
ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-False}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-1.0}
ROLLOUT_TOP_K=${ROLLOUT_TOP_K:-"-1"}

OVERLONG_BUFFER_ENABLE=${OVERLONG_BUFFER_ENABLE:-False}
OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN:-512}
OVERLONG_PENALTY_FACTOR=${OVERLONG_PENALTY_FACTOR:-1.0}

SAVE_FREQ=${SAVE_FREQ:-20}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}


########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=${USE_KL_IN_REWARD}
    algorithm.kl_ctrl.kl_coef=${KL_CTRL_COEF}
    algorithm.kl_penalty=${KL_PENALTY}
    algorithm.norm_adv_by_std_in_grpo=${NORM_ADV_BY_STD_IN_GRPO}
    data.train_files=${TRAIN_FILE}
    data.val_files=${TEST_FILE}
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.dataloader_num_workers=0
)

REWARD=(
    reward.reward_manager.name=dapo
    reward.custom_reward_function.path=${REWARD_FN_PATH}
    reward.custom_reward_function.name=${REWARD_FN_NAME}
    +reward.reward_kwargs.overlong_buffer_cfg.enable=${OVERLONG_BUFFER_ENABLE}
    +reward.reward_kwargs.overlong_buffer_cfg.len=${OVERLONG_BUFFER_LEN}
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${OVERLONG_PENALTY_FACTOR}
    +reward.reward_kwargs.overlong_buffer_cfg.log=False
    +reward.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH}
)

MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.optim.weight_decay=${WEIGHT_DECAY}
    actor_rollout_ref.actor.optim.lr_warmup_steps=${LR_WARMUP_STEPS}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.ppo_epochs=${PPO_EPOCHS}
    actor_rollout_ref.actor.clip_ratio=${CLIP_RATIO}
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW}
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}
    actor_rollout_ref.actor.clip_ratio_c=${CLIP_RATIO_C}
    actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE}
    actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS}
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.grad_clip=${GRAD_CLIP}
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN}
    actor_rollout_ref.rollout.enforce_eager=${ROLLOUT_ENFORCE_EAGER}
    actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE}
    actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P}
    actor_rollout_ref.rollout.top_k=${ROLLOUT_TOP_K}
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
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
    trainer.resume_mode=${RESUME_MODE}
)

EXTRA=(
)

########################### launch ###########################
echo "Logging to ${LOG_FILE} (no console training output)"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TRAIN_FILE=${TRAIN_FILE} TEST_FILE=${TEST_FILE}"
echo "adv=grpo reward_manager=dapo score=0/1(${REWARD_FN_PATH}) resume_mode=${RESUME_MODE} max_model_len=${ROLLOUT_MAX_MODEL_LEN}"
echo "use_kl_in_reward=${USE_KL_IN_REWARD} kl_ctrl.kl_coef=${KL_CTRL_COEF} kl_penalty=${KL_PENALTY} (actor use_kl_loss=${USE_KL_LOSS} kl_loss_coef=${KL_LOSS_COEF})"
echo "clip_ratio=${CLIP_RATIO}/${CLIP_RATIO_LOW}/${CLIP_RATIO_HIGH} c=${CLIP_RATIO_C} ppo_epochs=${PPO_EPOCHS} loss_agg=${LOSS_AGG_MODE} temp=${ROLLOUT_TEMPERATURE} top_p=${ROLLOUT_TOP_P}"
export PYTHONPATH="${VERL_ROOT}:${PYTHONPATH:-}"

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${REWARD[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@" \
    >"${LOG_FILE}" 2>&1
