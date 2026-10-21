#!/usr/bin/env bash
#SBATCH --job-name=opd
#SBATCH --output=opd_%j.out
#SBATCH --error=opd_%j.err
#SBATCH --account=test
#SBATCH --partition=TEST1
#SBATCH --exclude=g[81-82]
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

set -xeuo pipefail

# This file is intended to run from /workspace. It also works when invoked elsewhere.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=${WORKSPACE_ROOT:-$SCRIPT_DIR}
if [[ -f "$WORKSPACE_ROOT/verl/verl/trainer/main_ppo.py" ]]; then
    REPO_ROOT=${REPO_ROOT:-$WORKSPACE_ROOT/verl}
elif [[ -f "$SCRIPT_DIR/verl/trainer/main_ppo.py" ]]; then
    REPO_ROOT=${REPO_ROOT:-$SCRIPT_DIR}
else
    echo "Cannot find the verl checkout. Set REPO_ROOT=/path/to/verl." >&2
    exit 1
fi
cd "$REPO_ROOT"

# Mirror local runs to a timestamped log. Slurm uses opd_<jobid>.out above.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    LOG_DIR=${LOG_DIR:-logs}
    mkdir -p "$LOG_DIR"
    LOG_FILE="$LOG_DIR/opd_$(date +%Y%m%d_%H%M%S).log"
    exec > >(tee -a "$LOG_FILE") 2>&1
fi

export RAY_memory_usage_threshold=0.99
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR=1
export NCCL_TIMEOUT=7200
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
# SwanLab uses SWANLAB_API_KEY when provided, otherwise it reuses the local login.
export SWANLAB_MODE=${SWANLAB_MODE:-cloud}
export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-$WORKSPACE_ROOT/swanlog}

# ----- models and data -----
# Both models must use the same tokenizer/vocabulary.
ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-$WORKSPACE_ROOT/model/DeepSeek-R1-Distill-Qwen-1.5B}
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-${REWARD_MODEL_PATH:-$WORKSPACE_ROOT/model/JustRL-DeepSeek-1.5B}}
TRAIN_DATASET=${TRAIN_DATASET:-$WORKSPACE_ROOT/datasets/dapo-math-17k-processed.parquet}
TEST_DATASET=${TEST_DATASET:-"['$WORKSPACE_ROOT/datasets/test_data/AIME26/test.parquet','$WORKSPACE_ROOT/datasets/test_data/AIME25/test.parquet','$WORKSPACE_ROOT/datasets/test_data/AMC23/test.parquet']"}

# ----- OPD controls -----
# top_k=0 is naive sampled-token OPD. For top_k>0, choose only_stu,
# only_tch, intersection, or union.
OPD_TOP_K=${OPD_TOP_K:-16}
OPD_STRATEGY=${OPD_STRATEGY:-only_stu}
OPD_REWARD_WEIGHT_MODE=${OPD_REWARD_WEIGHT_MODE:-student_p}
# For top_k>0, normalize student/teacher probabilities over the candidate set
# before computing the KL advantage, and use the conditional student distribution
# in the candidate PPO ratio/score gradient. Ignored for top_k=0.
OPD_IS_NORM=${OPD_IS_NORM:-True}
# Aggregate all tokens outside the selected candidate set into one KL bucket.
# Requires OPD_IS_NORM=False and OPD_REWARD_WEIGHT_MODE=student_p.
OPD_OUTSIDE_BUCKET=${OPD_OUTSIDE_BUCKET:-False}
OPD_TEACHER_TEMPERATURE=${OPD_TEACHER_TEMPERATURE:-1.0}
# Smaller chunks lower the transient [chunk_size, vocab] buffer on teacher forwards.
OPD_CHUNK_SIZE=${OPD_CHUNK_SIZE:-1024}
ADV_ESTIMATOR=${ADV_ESTIMATOR:-opd_token_kl}

case "$OPD_TOP_K" in
    ''|*[!0-9]*) echo "OPD_TOP_K must be a non-negative integer" >&2; exit 2 ;;
esac
case "$OPD_STRATEGY" in
    only_stu|only_tch|intersection|union) ;;
    *) echo "OPD_STRATEGY must be only_stu, only_tch, intersection, or union" >&2; exit 2 ;;
esac
case "$OPD_REWARD_WEIGHT_MODE" in
    student_p|teacher_p|none) ;;
    *) echo "OPD_REWARD_WEIGHT_MODE must be student_p, teacher_p, or none" >&2; exit 2 ;;
esac
case "$OPD_IS_NORM" in
    True|False) ;;
    *) echo "OPD_IS_NORM must be True or False" >&2; exit 2 ;;
esac
case "$OPD_OUTSIDE_BUCKET" in
    True|False) ;;
    *) echo "OPD_OUTSIDE_BUCKET must be True or False" >&2; exit 2 ;;
esac
case "$ADV_ESTIMATOR" in
    opd_token_kl|opd_token_kl_plus_grpo) ;;
    *) echo "ADV_ESTIMATOR must be opd_token_kl or opd_token_kl_plus_grpo" >&2; exit 2 ;;
esac

# ----- sequence, batch, and optimization -----
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
PARALLEL_SIZE=${PARALLEL_SIZE:-1}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-5120}
MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-31744}
MAX_RESPONSE_LENGTH=$(( MAX_RESP_LENGTH > MAX_VAL_RESP_LENGTH ? MAX_RESP_LENGTH : MAX_VAL_RESP_LENGTH ))
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1))}
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-64}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-$((MINI_BATCH_SIZE * PARALLEL_SIZE))}
N_RESPONSES=${N_RESPONSES:-4}
TEMPERATURE=${TEMPERATURE:-1.0}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.6}
VAL_TOP_P=${VAL_TOP_P:-0.95}
VAL_TOP_K=${VAL_TOP_K:-20}
VAL_N=${VAL_N:-16}
MODEL_DTYPE=${MODEL_DTYPE:-fp32}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}
# Teacher forwards need less packed tokens because they materialize raw logits.
TEACHER_MAX_TOKEN_LEN_PER_GPU=${TEACHER_MAX_TOKEN_LEN_PER_GPU:-8192}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$MAX_MODEL_LEN}
ACTOR_LR=${ACTOR_LR:-1e-6}
USE_KL=${USE_KL:-False}
LR_SCHEDULER=${LR_SCHEDULER:-constant}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-20}
PROJECT_NAME=${PROJECT_NAME:-OnPolicyDistillation}
# Log scalars to both the terminal and SwanLab. Set LOGGER=console to disable upload.
LOGGER=${LOGGER:-'["console","swanlab"]'}
# Upload a small validation table containing input, student output, and task score.
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-8}
DATE_TAG=$(date +%Y-%m-%d_%H-%M-%S)
EXPERIMENT_NAME=${EXPERIMENT_NAME:-opd_${OPD_STRATEGY}_k${OPD_TOP_K}_${DATE_TAG}}
PROJECT_PATH=${PROJECT_PATH:-$WORKSPACE_ROOT/checkpoint}
CKPT_PATH="$PROJECT_PATH/$EXPERIMENT_NAME"
# Save every validation rollout locally. Each validation step produces one JSONL file.
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-$CKPT_PATH/validation_generations}

# Compare the last boxed answer with the ground truth without SymPy or signal-based timeouts.
CUSTOM_REWARD_PATH=${CUSTOM_REWARD_PATH:-$REPO_ROOT/verl/utils/reward_score/boxed_match.py}
CUSTOM_REWARD_NAME=${CUSTOM_REWARD_NAME:-reward_func}
CUSTOM_REWARD_ARGS=()
if [[ -n "$CUSTOM_REWARD_PATH" ]]; then
    CUSTOM_REWARD_ARGS=(
        "reward.custom_reward_function.path=$CUSTOM_REWARD_PATH"
        "reward.custom_reward_function.name=$CUSTOM_REWARD_NAME"
    )
fi

KL_ARGS=()
if [[ "$USE_KL" == "True" ]]; then
    KL_ARGS=(
        actor_rollout_ref.actor.use_kl_loss=True
        actor_rollout_ref.actor.kl_loss_coef=0.005
        actor_rollout_ref.actor.kl_loss_type=low_var_kl
    )
else
    KL_ARGS=(actor_rollout_ref.actor.use_kl_loss=False)
fi

LR_ARGS=()
if [[ "$LR_SCHEDULER" == "cosine" ]]; then
    LR_ARGS=(
        actor_rollout_ref.actor.optim.warmup_style=cosine
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03
    )
fi

DATA_ARGS=(
    "algorithm.adv_estimator=$ADV_ESTIMATOR"
    algorithm.use_kl_in_reward=False
    data.shuffle=False
    "data.train_files=$TRAIN_DATASET"
    "data.val_files=$TEST_DATASET"
    "data.train_batch_size=$TRAIN_BATCH_SIZE"
    "data.max_prompt_length=$MAX_PROMPT_LENGTH"
    "data.max_response_length=$MAX_RESP_LENGTH"
    data.filter_overlong_prompts=True
    data.truncation=error
    data.return_raw_chat=True
    # Leave the student's <think> block open so rollouts contain reasoning.
    +data.apply_chat_template_kwargs.enable_thinking=True
)

MODEL_ARGS=(
    "actor_rollout_ref.model.path=$ACTOR_MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.use_fused_kernels=False
    actor_rollout_ref.model.enable_activation_offload=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR_ARGS=(
    actor_rollout_ref.actor.strategy=fsdp
    actor_rollout_ref.actor.use_torch_compile=False
    "actor_rollout_ref.actor.optim.lr=$ACTOR_LR"
    "actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE"
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU"
    "actor_rollout_ref.actor.ulysses_sequence_parallel_size=$PARALLEL_SIZE"
    "actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE:-token-mean}"
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    "actor_rollout_ref.actor.fsdp_config.model_dtype=$MODEL_DTYPE"
)

ROLLOUT_ARGS=(
    actor_rollout_ref.rollout.name=vllm
    "actor_rollout_ref.rollout.temperature=$TEMPERATURE"
    "actor_rollout_ref.rollout.val_kwargs.temperature=$VAL_TEMPERATURE"
    "actor_rollout_ref.rollout.val_kwargs.top_p=$VAL_TOP_P"
    "actor_rollout_ref.rollout.val_kwargs.top_k=$VAL_TOP_K"
    "actor_rollout_ref.rollout.val_kwargs.n=$VAL_N"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=$PARALLEL_SIZE"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL:-0.8}"
    "actor_rollout_ref.rollout.n=$N_RESPONSES"
    "actor_rollout_ref.rollout.val_kwargs.response_length=$MAX_VAL_RESP_LENGTH"
    "actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN"
    "actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_MAX_NUM_BATCHED_TOKENS"
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU"
    actor_rollout_ref.rollout.calculate_log_probs=False
)

OPD_ARGS=(
    opd.enable=True
    "opd.teacher_model_path=$TEACHER_MODEL_PATH"
    "opd.top_k=$OPD_TOP_K"
    "opd.strategy=$OPD_STRATEGY"
    "opd.reward_weight_mode=$OPD_REWARD_WEIGHT_MODE"
    "opd.is_norm=$OPD_IS_NORM"
    "opd.outside_bucket=$OPD_OUTSIDE_BUCKET"
    "opd.teacher_temperature=$OPD_TEACHER_TEMPERATURE"
    "opd.chunk_size=$OPD_CHUNK_SIZE"
    actor_rollout_ref.teacher.fsdp_config.param_offload=True
    "actor_rollout_ref.teacher.fsdp_config.model_dtype=$MODEL_DTYPE"
    actor_rollout_ref.teacher.model.use_remove_padding=True
    actor_rollout_ref.teacher.model.use_fused_kernels=False
    actor_rollout_ref.teacher.log_prob_use_dynamic_bsz=True
    "actor_rollout_ref.teacher.log_prob_max_token_len_per_gpu=$TEACHER_MAX_TOKEN_LEN_PER_GPU"
    actor_rollout_ref.teacher.log_prob_micro_batch_size_per_gpu=1
)

TRAINER_ARGS=(
    trainer.use_v1=True
    trainer.v1.trainer_mode=sync
    trainer.balance_batch=True
    "trainer.logger=$LOGGER"
    "trainer.project_name=$PROJECT_NAME"
    "trainer.experiment_name=$EXPERIMENT_NAME"
    "trainer.n_gpus_per_node=$NGPUS_PER_NODE"
    "trainer.nnodes=$NNODES"
    trainer.val_before_train=True
    "trainer.log_val_generations=$LOG_VAL_GENERATIONS"
    "trainer.validation_data_dir=$VALIDATION_DATA_DIR"
    "trainer.save_freq=$SAVE_FREQ"
    "trainer.test_freq=$TEST_FREQ"
    "trainer.total_epochs=$TOTAL_EPOCHS"
    "trainer.default_local_dir=$CKPT_PATH"
)

# The OPD teacher is colocated; the separate teacher-pool implementation must stay disabled.
DISTILLATION_ARGS=(distillation.enabled=False)
REWARD_ARGS=(reward.reward_model.enable=False)

echo "Launching colocated OPD"
echo "  actor:   $ACTOR_MODEL_PATH"
echo "  teacher: $TEACHER_MODEL_PATH"
echo "  top_k:   $OPD_TOP_K ($OPD_STRATEGY)"
echo "  weights: $OPD_REWARD_WEIGHT_MODE"
echo "  is_norm: $OPD_IS_NORM"
echo "  outside bucket: $OPD_OUTSIDE_BUCKET"
echo "  logger:  $LOGGER (mode=$SWANLAB_MODE)"
echo "  data:    $TRAIN_DATASET"
echo "  val:     n=$VAL_N, temperature=$VAL_TEMPERATURE, top_p=$VAL_TOP_P, top_k=$VAL_TOP_K"
echo "  val dump: $VALIDATION_DATA_DIR"

COMMAND=(
    python3 -m verl.trainer.main_ppo
    "${DATA_ARGS[@]}"
    "${MODEL_ARGS[@]}"
    "${ACTOR_ARGS[@]}"
    "${ROLLOUT_ARGS[@]}"
    "${OPD_ARGS[@]}"
    "${DISTILLATION_ARGS[@]}"
    "${REWARD_ARGS[@]}"
    "${KL_ARGS[@]}"
    "${LR_ARGS[@]}"
    "${CUSTOM_REWARD_ARGS[@]}"
    "${TRAINER_ARGS[@]}"
    "$@"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf ' %q' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi

ray stop --force || true
ray start --head
"${COMMAND[@]}"

echo "Finished OPD: $(date)"
