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

# ----- training mode -----
# opd: colocated distillation, advantage log p_tch - log p_stu (sampled or top-k).
# direct_opd_sampled: Direct-OPD on the sampled token, advantage log p_tch - log p_tch_ref,
#   where tch_ref is the teacher before RL. Forces OPD_TOP_K=0.
# opd_full_vocab: full-vocabulary OPD. The teacher only exports its LM-head input hidden;
#   the student rebuilds the teacher distribution chunk by chunk in its fused forward and
#   minimizes the exact per-token KL(p_stu || p_tch). Forces OPD_TOP_K=0.
# diff_opd_full_vocab: full-vocabulary Direct-OPD. The target is
#   q ∝ p_S0 * p_tch / p_tch_ref with S0 the frozen initial student (the reference policy,
#   so USE_KL is forced on); the student minimizes the exact per-token KL(p_stu || q).
#   Needs OPD_TEACHER_REF_MODEL_PATH. Forces OPD_TOP_K=0.
# md_opd_sampled: Multi-Diff OPD. Several post-RL teachers (OPD_TEACHERS) share the reference
#   OPD_TEACHER_REF_MODEL_PATH; each sample is routed by OPD_ROUTE_FIELD to the teacher whose
#   tags hold its value and takes that teacher's Direct-OPD shift. TEACHER_MODEL_PATH is
#   ignored. Forces OPD_TOP_K=0.
# rl: verifiable-reward RL only; no teacher is built and every OPD_* knob is ignored.
TRAIN_MODE=${TRAIN_MODE:-opd}
OPD_OBJECTIVE=legacy
case "$TRAIN_MODE" in
    opd|rl) ;;
    o3pd|opd_topk_jsd)
        OPD_OBJECTIVE=adjacent_logistic
        [[ "$TRAIN_MODE" == "opd_topk_jsd" ]] && OPD_OBJECTIVE=topk_jsd
        OPD_TOP_K=${OPD_TOP_K:-8}
        OPD_STRATEGY=${OPD_STRATEGY:-only_tch}
        LOSS_AGG_MODE=${LOSS_AGG_MODE:-seq-mean-token-mean}
        N_RESPONSES=${N_RESPONSES:-1}
        MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-32}
        OPD_SAMPLED_ADV_CLIP_MIN=null
        OPD_SAMPLED_ADV_CLIP_MAX=null
        ;;
    opd_full_vocab|diff_opd_full_vocab)
        if [[ -n "${OPD_TOP_K:-}" && "$OPD_TOP_K" != "0" ]]; then
            echo "TRAIN_MODE=$TRAIN_MODE requires OPD_TOP_K=0, got $OPD_TOP_K" >&2; exit 2
        fi
        OPD_TOP_K=0
        # The sampled-token advantage clip does not apply to the full-vocabulary loss.
        OPD_SAMPLED_ADV_CLIP_MIN=null
        OPD_SAMPLED_ADV_CLIP_MAX=null
        if [[ "$TRAIN_MODE" == "diff_opd_full_vocab" ]]; then
            OPD_TEACHER_REF_MODEL_PATH=${OPD_TEACHER_REF_MODEL_PATH:-$WORKSPACE_ROOT/model/DeepSeek-R1-Distill-Qwen-1.5B}
            # The frozen initial student S0 is the reference policy, so the ref engine must exist.
            USE_KL=True
        fi
        ;;
    direct_opd_sampled)
        if [[ -n "${OPD_TOP_K:-}" && "$OPD_TOP_K" != "0" ]]; then
            echo "TRAIN_MODE=direct_opd_sampled requires OPD_TOP_K=0, got $OPD_TOP_K" >&2; exit 2
        fi
        OPD_TOP_K=0
        OPD_TEACHER_REF_MODEL_PATH=${OPD_TEACHER_REF_MODEL_PATH:-$WORKSPACE_ROOT/model/DeepSeek-R1-Distill-Qwen-1.5B}
        # Direct-OPD does not clip the teacher shift.
        OPD_SAMPLED_ADV_CLIP_MIN=${OPD_SAMPLED_ADV_CLIP_MIN:-null}
        OPD_SAMPLED_ADV_CLIP_MAX=${OPD_SAMPLED_ADV_CLIP_MAX:-null}
        ;;
    md_opd_sampled)
        if [[ -n "${OPD_TOP_K:-}" && "$OPD_TOP_K" != "0" ]]; then
            echo "TRAIN_MODE=md_opd_sampled requires OPD_TOP_K=0, got $OPD_TOP_K" >&2; exit 2
        fi
        OPD_TOP_K=0
        OPD_TEACHER_REF_MODEL_PATH=${OPD_TEACHER_REF_MODEL_PATH:-$WORKSPACE_ROOT/model/DeepSeek-R1-Distill-Qwen-1.5B}
        # name=path pairs separated by ';'. Tags default to the teacher name; OPD_TEACHER_TAGS
        # overrides them as name=tag1|tag2 pairs separated by ';'.
        OPD_TEACHERS=${OPD_TEACHERS:-"math=$WORKSPACE_ROOT/model/DeepScaleR-1.5B-Preview;code=$WORKSPACE_ROOT/model/DeepCoder-1.5B-Preview"}
        OPD_TEACHER_TAGS=${OPD_TEACHER_TAGS:-}
        OPD_ROUTE_FIELD=${OPD_ROUTE_FIELD:-ability}
        OPD_ROUTE_MODE=${OPD_ROUTE_MODE:-hard}
        # Per-teacher shift multipliers as name=scale pairs separated by ';'; unset names keep 1.0.
        OPD_DOMAIN_ADV_SCALE=${OPD_DOMAIN_ADV_SCALE:-}
        TEACHER_MODEL_PATH=null
        OPD_SAMPLED_ADV_CLIP_MIN=${OPD_SAMPLED_ADV_CLIP_MIN:-null}
        OPD_SAMPLED_ADV_CLIP_MAX=${OPD_SAMPLED_ADV_CLIP_MAX:-null}

        declare -A MD_TEACHER_TAGS=()
        IFS=';' read -ra MD_ITEMS <<< "$OPD_TEACHER_TAGS"
        for MD_ITEM in ${MD_ITEMS[@]+"${MD_ITEMS[@]}"}; do
            [[ -z "$MD_ITEM" ]] && continue
            MD_TEACHER_TAGS["${MD_ITEM%%=*}"]="${MD_ITEM#*=}"
        done
        MD_TEACHER_ENTRIES=()
        MD_TEACHER_SUMMARY=()
        IFS=';' read -ra MD_ITEMS <<< "$OPD_TEACHERS"
        for MD_ITEM in ${MD_ITEMS[@]+"${MD_ITEMS[@]}"}; do
            [[ -z "$MD_ITEM" ]] && continue
            MD_NAME="${MD_ITEM%%=*}"
            MD_PATH="${MD_ITEM#*=}"
            if [[ "$MD_NAME" == "$MD_ITEM" || -z "$MD_NAME" || -z "$MD_PATH" ]]; then
                echo "OPD_TEACHERS entries must be name=path, got '$MD_ITEM'" >&2; exit 2
            fi
            if [[ "$MD_PATH" == /* && ! -d "$MD_PATH" ]]; then
                echo "OPD_TEACHERS: teacher '$MD_NAME' path $MD_PATH does not exist" >&2; exit 2
            fi
            IFS='|' read -ra MD_TAG_LIST <<< "${MD_TEACHER_TAGS[$MD_NAME]:-$MD_NAME}"
            if [[ ${#MD_TAG_LIST[@]} -eq 0 ]]; then
                echo "OPD_TEACHER_TAGS: teacher '$MD_NAME' has an empty tag list" >&2; exit 2
            fi
            MD_TAG_ITEMS=()
            for MD_TAG in "${MD_TAG_LIST[@]}"; do MD_TAG_ITEMS+=("'$MD_TAG'"); done
            MD_TEACHER_ENTRIES+=("{name:'$MD_NAME',path:'$MD_PATH',tags:[$(IFS=,; echo "${MD_TAG_ITEMS[*]}")]}")
            MD_TEACHER_SUMMARY+=("$MD_NAME[${MD_TEACHER_TAGS[$MD_NAME]:-$MD_NAME}]=$MD_PATH")
        done
        if [[ ${#MD_TEACHER_ENTRIES[@]} -eq 0 ]]; then
            echo "TRAIN_MODE=md_opd_sampled needs at least one OPD_TEACHERS entry" >&2; exit 2
        fi
        OPD_TEACHERS_HYDRA="[$(IFS=,; echo "${MD_TEACHER_ENTRIES[*]}")]"
        MD_SCALE_ITEMS=()
        IFS=';' read -ra MD_ITEMS <<< "$OPD_DOMAIN_ADV_SCALE"
        for MD_ITEM in ${MD_ITEMS[@]+"${MD_ITEMS[@]}"}; do
            [[ -z "$MD_ITEM" ]] && continue
            # Hydra dict keys must be bare identifiers.
            MD_SCALE_ITEMS+=("${MD_ITEM%%=*}:${MD_ITEM#*=}")
        done
        OPD_DOMAIN_ADV_SCALE_HYDRA="{$(IFS=,; echo "${MD_SCALE_ITEMS[*]+${MD_SCALE_ITEMS[*]}}")}"
        ;;
    *) echo "Unknown TRAIN_MODE: $TRAIN_MODE (expected opd, o3pd, opd_topk_jsd, direct_opd_sampled, md_opd_sampled, opd_full_vocab, diff_opd_full_vocab, rl)" >&2; exit 2 ;;
esac
if [[ "$TRAIN_MODE" =~ ^(opd|o3pd|opd_topk_jsd|opd_full_vocab)$ && -n "${OPD_TEACHER_REF_MODEL_PATH:-}" ]]; then
    echo "OPD_TEACHER_REF_MODEL_PATH is only used by TRAIN_MODE=direct_opd_sampled, md_opd_sampled, or diff_opd_full_vocab" >&2; exit 2
fi

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
# For top_k=0, clamp the per-token advantage (log p_tch - log p_stu) to
# [MIN, MAX]. null disables that side. Ignored for top_k>0.
OPD_SAMPLED_ADV_CLIP_MIN=${OPD_SAMPLED_ADV_CLIP_MIN:--10}
OPD_SAMPLED_ADV_CLIP_MAX=${OPD_SAMPLED_ADV_CLIP_MAX:-10}
# Direct-OPD, MD-OPD and diff_opd_full_vocab: the teacher before RL (T0).
OPD_TEACHER_REF_MODEL_PATH=${OPD_TEACHER_REF_MODEL_PATH:-}
# Sampled-token modes only: comma-separated token ids whose advantage is zeroed, e.g.
# chat-template tokens the teachers never saw. Empty disables it.
OPD_MASK_TOKEN_IDS=${OPD_MASK_TOKEN_IDS:-}
# Smaller chunks lower the transient [chunk_size, vocab] buffer on teacher forwards.
OPD_CHUNK_SIZE=${OPD_CHUNK_SIZE:-1024}
# Full-vocabulary OPD only: rows per fp32 [chunk, vocab] buffer in the student's fused
# LM head (forward and recompute in backward), and the weight of the reverse KL loss.
OPD_FULL_VOCAB_CHUNK=${OPD_FULL_VOCAB_CHUNK:-256}
OPD_FULL_VOCAB_COEF=${OPD_FULL_VOCAB_COEF:-1.0}
# diff_opd_full_vocab only: q ∝ p_S0 * (p_tch / p_tch_ref)^(1/beta). Larger beta stays closer
# to S0; OPD_FULL_VOCAB_COEF=2*beta^2 recovers the squared-residual gradient scale.
OPD_FULL_VOCAB_BETA=${OPD_FULL_VOCAB_BETA:-1.0}
if [[ "$TRAIN_MODE" != "diff_opd_full_vocab" && "$OPD_FULL_VOCAB_BETA" != "1.0" && "$OPD_FULL_VOCAB_BETA" != "1" ]]; then
    echo "OPD_FULL_VOCAB_BETA only applies to TRAIN_MODE=diff_opd_full_vocab, got $OPD_FULL_VOCAB_BETA" >&2; exit 2
fi
# Full-vocabulary OPD runs student and teacher through the fused torch-backend forward:
# the student projects its own and the teacher's hidden chunk by chunk, and the teacher
# returns its LM-head input without projecting. Sparse OPD needs the raw logits instead.
if [[ "$TRAIN_MODE" =~ ^(opd_full_vocab|diff_opd_full_vocab)$ ]]; then
    USE_FUSED_KERNELS=True
else
    USE_FUSED_KERNELS=False
fi
if [[ "$TRAIN_MODE" == "rl" ]]; then
    ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}
else
    ADV_ESTIMATOR=${ADV_ESTIMATOR:-opd_token_kl}
fi

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
case "$TRAIN_MODE:$ADV_ESTIMATOR" in
    o3pd:opd_token_kl|opd_topk_jsd:opd_token_kl) ;;
    opd:opd_token_kl|opd:opd_token_kl_plus_grpo) ;;
    direct_opd_sampled:opd_token_kl|direct_opd_sampled:opd_token_kl_plus_grpo) ;;
    md_opd_sampled:opd_token_kl|md_opd_sampled:opd_token_kl_plus_grpo) ;;
    opd_full_vocab:opd_token_kl|opd_full_vocab:opd_token_kl_plus_grpo) ;;
    diff_opd_full_vocab:opd_token_kl|diff_opd_full_vocab:opd_token_kl_plus_grpo) ;;
    rl:opd_*) echo "TRAIN_MODE=rl needs a non-OPD ADV_ESTIMATOR such as grpo" >&2; exit 2 ;;
    rl:*) ;;
    *) echo "TRAIN_MODE=$TRAIN_MODE needs ADV_ESTIMATOR=opd_token_kl or opd_token_kl_plus_grpo" >&2; exit 2 ;;
esac

# ----- sequence, batch, and optimization -----
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
PARALLEL_SIZE=${PARALLEL_SIZE:-1}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-7168}
MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-31744}
MAX_RESPONSE_LENGTH=$(( MAX_RESP_LENGTH > MAX_VAL_RESP_LENGTH ? MAX_RESP_LENGTH : MAX_VAL_RESP_LENGTH ))
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1))}
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-64}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-$((MINI_BATCH_SIZE * PARALLEL_SIZE))}
N_RESPONSES=${N_RESPONSES:-4}
TEMPERATURE=${TEMPERATURE:-1.0}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-1.0}
VAL_TOP_P=${VAL_TOP_P:-0.95}
VAL_N=${VAL_N:-16}
MODEL_DTYPE=${MODEL_DTYPE:-fp32}
# The teacher is forward-only and computes in bf16 under FSDP mixed precision, so
# bf16 storage is numerically identical and halves its host memory.
TEACHER_MODEL_DTYPE=${TEACHER_MODEL_DTYPE:-$MODEL_DTYPE}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}
# Teacher forwards need less packed tokens because they materialize raw logits.
# The full-vocabulary teacher never projects onto the vocabulary, so it follows the actor.
if [[ "$TRAIN_MODE" =~ ^(opd_full_vocab|diff_opd_full_vocab)$ ]]; then
    TEACHER_MAX_TOKEN_LEN_PER_GPU=${TEACHER_MAX_TOKEN_LEN_PER_GPU:-$PPO_MAX_TOKEN_LEN_PER_GPU}
else
    TEACHER_MAX_TOKEN_LEN_PER_GPU=${TEACHER_MAX_TOKEN_LEN_PER_GPU:-8192}
fi
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$MAX_MODEL_LEN}
ACTOR_LR=${ACTOR_LR:-1e-6}
USE_KL=${USE_KL:-False}
# low_var_kl to the initial student (the reference policy); only used when USE_KL=True.
KL_LOSS_COEF=${KL_LOSS_COEF:-0.005}
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
if [[ "$TRAIN_MODE" == "rl" ]]; then
    EXPERIMENT_NAME=${EXPERIMENT_NAME:-rl_${ADV_ESTIMATOR}_${DATE_TAG}}
elif [[ "$TRAIN_MODE" == "direct_opd_sampled" ]]; then
    EXPERIMENT_NAME=${EXPERIMENT_NAME:-direct_opd_sampled_${DATE_TAG}}
elif [[ "$TRAIN_MODE" == "md_opd_sampled" ]]; then
    EXPERIMENT_NAME=${EXPERIMENT_NAME:-md_opd_sampled_${OPD_ROUTE_MODE}_${DATE_TAG}}
elif [[ "$TRAIN_MODE" == "opd_full_vocab" ]]; then
    EXPERIMENT_NAME=${EXPERIMENT_NAME:-opd_full_vocab_${DATE_TAG}}
elif [[ "$TRAIN_MODE" == "diff_opd_full_vocab" ]]; then
    EXPERIMENT_NAME=${EXPERIMENT_NAME:-diff_opd_full_vocab_${DATE_TAG}}
else
    EXPERIMENT_NAME=${EXPERIMENT_NAME:-opd_${OPD_STRATEGY}_k${OPD_TOP_K}_${DATE_TAG}}
fi
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
        "actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF"
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
)

MODEL_ARGS=(
    "actor_rollout_ref.model.path=$ACTOR_MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    "actor_rollout_ref.model.use_fused_kernels=$USE_FUSED_KERNELS"
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
    "actor_rollout_ref.rollout.val_kwargs.n=$VAL_N"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=$PARALLEL_SIZE"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL:-0.8}"
    "actor_rollout_ref.rollout.n=$N_RESPONSES"
    "actor_rollout_ref.rollout.val_kwargs.response_length=$MAX_VAL_RESP_LENGTH"
    "actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN"
    "actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_MAX_NUM_BATCHED_TOKENS"
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU"
    # vLLM's sampled-token log-probs feed the training/rollout_probs_diff_* and rollout_corr/*
    # mismatch diagnostics. rollout_is/rollout_rs stay null, so the update itself is unchanged.
    "actor_rollout_ref.rollout.calculate_log_probs=${CALCULATE_LOG_PROBS:-True}"
)

OPD_ARGS=(
    opd.enable=True
    "opd.objective=$OPD_OBJECTIVE"
    "opd.loss_coef=${OPD_LOSS_COEF:-1.0}"
    "opd.teacher_model_path=$TEACHER_MODEL_PATH"
    "opd.top_k=$OPD_TOP_K"
    "opd.strategy=$OPD_STRATEGY"
    "opd.reward_weight_mode=$OPD_REWARD_WEIGHT_MODE"
    "opd.is_norm=$OPD_IS_NORM"
    "opd.outside_bucket=$OPD_OUTSIDE_BUCKET"
    "opd.teacher_temperature=$OPD_TEACHER_TEMPERATURE"
    "opd.sampled_adv_clip_min=$OPD_SAMPLED_ADV_CLIP_MIN"
    "opd.sampled_adv_clip_max=$OPD_SAMPLED_ADV_CLIP_MAX"
    "opd.chunk_size=$OPD_CHUNK_SIZE"
    actor_rollout_ref.teacher.fsdp_config.param_offload=True
    "actor_rollout_ref.teacher.fsdp_config.model_dtype=$TEACHER_MODEL_DTYPE"
    actor_rollout_ref.teacher.model.use_remove_padding=True
    "actor_rollout_ref.teacher.model.use_fused_kernels=$USE_FUSED_KERNELS"
    actor_rollout_ref.teacher.log_prob_use_dynamic_bsz=True
    "actor_rollout_ref.teacher.log_prob_max_token_len_per_gpu=$TEACHER_MAX_TOKEN_LEN_PER_GPU"
    actor_rollout_ref.teacher.log_prob_micro_batch_size_per_gpu=1
)
if [[ "$TRAIN_MODE" =~ ^(direct_opd_sampled|md_opd_sampled|diff_opd_full_vocab)$ ]]; then
    OPD_ARGS+=("opd.teacher_ref_model_path=$OPD_TEACHER_REF_MODEL_PATH")
fi
if [[ "$TRAIN_MODE" =~ ^(direct_opd_sampled|md_opd_sampled)$ ]]; then
    # Both teachers also log their full-vocabulary entropy. Chunking bounds the fp32
    # [chunk, vocab] buffer; the chunk loop runs eager rather than through torch.compile.
    OPD_ARGS+=(
        actor_rollout_ref.teacher.fsdp_config.entropy_from_logits_with_chunking=True
        "actor_rollout_ref.teacher.fsdp_config.entropy_from_logits_chunk_size=$OPD_CHUNK_SIZE"
        actor_rollout_ref.teacher.fsdp_config.use_torch_compile=False
    )
fi
if [[ "$TRAIN_MODE" == "md_opd_sampled" ]]; then
    OPD_ARGS+=(
        "opd.teachers=$OPD_TEACHERS_HYDRA"
        "opd.route_field=$OPD_ROUTE_FIELD"
        "opd.route_mode=$OPD_ROUTE_MODE"
    )
    if [[ "$OPD_DOMAIN_ADV_SCALE_HYDRA" != "{}" ]]; then
        OPD_ARGS+=("opd.domain_adv_scale=$OPD_DOMAIN_ADV_SCALE_HYDRA")
    fi
fi
if [[ -n "$OPD_MASK_TOKEN_IDS" ]]; then
    OPD_ARGS+=("opd.mask_token_ids=[$OPD_MASK_TOKEN_IDS]")
fi
if [[ "$TRAIN_MODE" =~ ^(opd_full_vocab|diff_opd_full_vocab)$ ]]; then
    OPD_FULL_VOCAB_TARGET=rkl
    if [[ "$TRAIN_MODE" == "diff_opd_full_vocab" ]]; then
        OPD_FULL_VOCAB_TARGET=diff
    fi
    OPD_ARGS+=(
        opd.full_vocab.enable=True
        "opd.full_vocab.target=$OPD_FULL_VOCAB_TARGET"
        "opd.full_vocab.chunk_size=$OPD_FULL_VOCAB_CHUNK"
        "opd.full_vocab.loss_coef=$OPD_FULL_VOCAB_COEF"
        "opd.full_vocab.beta=$OPD_FULL_VOCAB_BETA"
    )
    MODEL_ARGS+=(actor_rollout_ref.model.fused_kernel_options.impl_backend=torch)
fi
if [[ "$TRAIN_MODE" == "rl" ]]; then
    OPD_ARGS=(opd.enable=False)
fi

TRAINER_ARGS=(
    trainer.use_v1=True
    trainer.v1.trainer_mode=sync
    trainer.balance_batch=True
    "trainer.logger=$LOGGER"
    "trainer.project_name=$PROJECT_NAME"
    "trainer.experiment_name=$EXPERIMENT_NAME"
    "trainer.n_gpus_per_node=$NGPUS_PER_NODE"
    "trainer.nnodes=$NNODES"
    "trainer.val_before_train=${VAL_BEFORE_TRAIN:-True}"
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

echo "Launching TRAIN_MODE=$TRAIN_MODE (adv_estimator=$ADV_ESTIMATOR)"
echo "  actor:   $ACTOR_MODEL_PATH"
echo "  teacher: $TEACHER_MODEL_PATH"
echo "  teacher_ref (Direct-OPD): ${OPD_TEACHER_REF_MODEL_PATH:-<disabled>}"
if [[ "$TRAIN_MODE" == "diff_opd_full_vocab" ]]; then
    echo "  student_ref S0 (frozen initial student): $ACTOR_MODEL_PATH"
fi
if [[ "$TRAIN_MODE" == "md_opd_sampled" ]]; then
    echo "  md teachers: ${MD_TEACHER_SUMMARY[*]}"
    echo "  md routing: field=$OPD_ROUTE_FIELD mode=$OPD_ROUTE_MODE scales=$OPD_DOMAIN_ADV_SCALE_HYDRA"
fi
echo "  mask token ids: ${OPD_MASK_TOKEN_IDS:-<none>}"
echo "  top_k:   $OPD_TOP_K ($OPD_STRATEGY)"
echo "  weights: $OPD_REWARD_WEIGHT_MODE"
echo "  is_norm: $OPD_IS_NORM"
echo "  outside bucket: $OPD_OUTSIDE_BUCKET"
echo "  sampled adv clip: [$OPD_SAMPLED_ADV_CLIP_MIN, $OPD_SAMPLED_ADV_CLIP_MAX]"
if [[ "$TRAIN_MODE" =~ ^(opd_full_vocab|diff_opd_full_vocab)$ ]]; then
    echo "  full vocab: target=$OPD_FULL_VOCAB_TARGET chunk=$OPD_FULL_VOCAB_CHUNK coef=$OPD_FULL_VOCAB_COEF beta=$OPD_FULL_VOCAB_BETA"
fi
echo "  kl loss: $USE_KL (coef=$KL_LOSS_COEF)"
echo "  logger:  $LOGGER (mode=$SWANLAB_MODE)"
echo "  data:    $TRAIN_DATASET"
echo "  val:     n=$VAL_N, temperature=$VAL_TEMPERATURE, top_p=$VAL_TOP_P"
echo "  val dump: $VALIDATION_DATA_DIR"

# Nodes with an old system glibc point this at a wrapper that starts the venv python.
PYTHON_BIN=${PYTHON_BIN:-python3}

COMMAND=(
    "$PYTHON_BIN" -m verl.trainer.main_ppo
    "${DATA_ARGS[@]}"
    "${MODEL_ARGS[@]}"
    "${ACTOR_ARGS[@]}"
    "${ROLLOUT_ARGS[@]}"
    "${OPD_ARGS[@]}"
    "${DISTILLATION_ARGS[@]}"
    "${REWARD_ARGS[@]}"
    "${KL_ARGS[@]}"
    # bash < 4.4 treats "${empty[@]}" as unbound under set -u.
    ${LR_ARGS[@]+"${LR_ARGS[@]}"}
    ${CUSTOM_REWARD_ARGS[@]+"${CUSTOM_REWARD_ARGS[@]}"}
    "${TRAINER_ARGS[@]}"
    "$@"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf ' %q' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi

# RAY_START=0 skips the shared head and lets ray.init() spawn a private local
# cluster, which avoids clashing with other users' Ray heads on shared nodes.
if [[ "${RAY_START:-1}" == "1" ]]; then
    ray stop --force || true
    ray start --head
fi
"${COMMAND[@]}"

echo "Finished OPD: $(date)"
