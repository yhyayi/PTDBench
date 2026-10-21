#!/bin/bash
# Sampled-token on-policy distillation (OPD) of DeepSeek-R1-Distill-Qwen-1.5B from JustRL-DeepSeek-1.5B on DAPO-Math-17k.
#
#   CUDA_VISIBLE_DEVICES=0,1 bash on_policy_distillation.sh [extra hydra overrides ...]
#
# Uses every GPU listed in CUDA_VISIBLE_DEVICES. Extra arguments are forwarded to verl.trainer.main_ppo
# after the defaults below, so they take precedence.

set -x
WORKSPACE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$WORKSPACE"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
N_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c .)
RUN_TS=${RUN_TS:-$(date +%Y-%m-%d_%H-%M-%S)}

LOG_DIR=${LOG_DIR:-logs}
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/run_opd_${RUN_TS}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=========================================="
echo "Log file: $LOG_FILE"
echo "Start time: $(date)  host: $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "=========================================="

export PYTHONPATH=${WORKSPACE}/verl${PYTHONPATH:+:$PYTHONPATH}
# vLLM's startup profiling would JIT-build FlashInfer's top-k/top-p sampler, which needs nvcc at runtime.
# Rollouts use temperature=1 without top-k/top-p, which take vLLM's native sampling path either way.
export VLLM_USE_FLASHINFER_SAMPLER=0
export RAY_memory_usage_threshold=0.99
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1
export PROJECT_NAME='OnPolicyDistillation'
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=7200
export TORCH_DISTRIBUTED_DEBUG=INFO
export ADV_ESTIMATOR=token_reward_direct
export GRPO_OUTCOME_WEIGHT=1.0

export MAX_PROMPT_LENGTH=1024
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-7168}
export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-7168}
export MAX_MODEL_LEN=$(( MAX_RESP_LENGTH + MAX_PROMPT_LENGTH > MAX_VAL_RESP_LENGTH + MAX_PROMPT_LENGTH ? MAX_RESP_LENGTH + MAX_PROMPT_LENGTH : MAX_VAL_RESP_LENGTH + MAX_PROMPT_LENGTH ))
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-64}
export TEMPERATURE=${TEMPERATURE:-1.0}
export TEACHER_TEMPERATURE=${TEACHER_TEMPERATURE:-1.0}
export REPETITION_PENALTY=${REPETITION_PENALTY:-1.0}
export N_RESPONSES=${N_RESPONSES:-4}
export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-0} # 0: sampled-token OPD
export TOP_K_STRATEGY=${TOP_K_STRATEGY:-"only_stu"}
export REWARD_WEIGHT_MODE=${REWARD_WEIGHT_MODE:-"student_p"}
export USE_KL=${USE_KL:-False}
export ENABLE_FORMAT_REWARD=${ENABLE_FORMAT_REWARD:-False}
export MODEL_DTYPE=${MODEL_DTYPE:-bf16} # actor/ref fsdp_config.model_dtype and reward model dtype: fp32 or bf16
export IS_PLOT=${IS_PLOT:-True}
export LOSS_AGG_MODE=${LOSS_AGG_MODE:-"token-mean"}
export LOGGER=${LOGGER:-"['console']"}

export TRAIN_DATASET=datasets/dapo-math-17k.parquet
export TRAIN_DATASET_NAME=DAPO-Math-17k
export TEST_DATA_DIR=datasets/test_data
TEST_DATASET=${TEST_FILE:-["$TEST_DATA_DIR/AIME25/test.parquet", "$TEST_DATA_DIR/AMC23/test.parquet", "$TEST_DATA_DIR/AIME24/test.parquet"]}

export ACTOR_MODEL_PATH=model/DeepSeek-R1-Distill-Qwen-1.5B
export ACTOR_MODEL_NAME=$(basename "$ACTOR_MODEL_PATH")
export REWARD_MODEL_PATH=model/JustRL-DeepSeek-1.5B
export REWARD_MODEL_NAME=$(basename "$REWARD_MODEL_PATH")

export PROJECT_PATH=checkpoint
export PARALLEL_SIZE=1
export EXPERIMENT_NAME=${ADV_ESTIMATOR}_${TRAIN_DATASET_NAME}_${ACTOR_MODEL_NAME}_${REWARD_MODEL_NAME}_${MAX_RESP_LENGTH}-T_${TEMPERATURE}-Tch_${TEACHER_TEMPERATURE}-n_${N_RESPONSES}-mbs_${MINI_BATCH_SIZE}-topk_${LOG_PROB_TOP_K}-topk_strategy_${TOP_K_STRATEGY}-rw_${REWARD_WEIGHT_MODE}-dtype_${MODEL_DTYPE}-${N_GPUS}gpu-${RUN_TS}
export CKPT_PATH=${PROJECT_PATH}/${EXPERIMENT_NAME}
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=true
export SWANLAB_LOG_DIR=${PROJECT_PATH}/swanlab_log
export SWANLAB_MODE=${SWANLAB_MODE:-offline}
export HYDRA_FULL_ERROR=1

KL_ARGS=""
if [ "$USE_KL" = "True" ]; then
    KL_ARGS="actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.005 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl"
else
    KL_ARGS="actor_rollout_ref.actor.use_kl_loss=False"
fi

PPO_MAX_TOKEN_LEN_PER_GPU=$(( ((1024 + MAX_RESP_LENGTH) > 32768) ? (1024 + MAX_RESP_LENGTH) : 32768))
echo "PPO_MAX_TOKEN_LEN_PER_GPU: $PPO_MAX_TOKEN_LEN_PER_GPU"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$ADV_ESTIMATOR \
    algorithm.grpo_outcome_weight=$GRPO_OUTCOME_WEIGHT \
    data.shuffle=False \
    data.train_files="$TRAIN_DATASET" \
    data.val_files="$TEST_DATASET" \
    data.train_batch_size=$((${MINI_BATCH_SIZE}*${PARALLEL_SIZE})) \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESP_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$PARALLEL_SIZE \
    $KL_ARGS \
    actor_rollout_ref.actor.loss_agg_mode=$LOSS_AGG_MODE \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=$MODEL_DTYPE \
    actor_rollout_ref.rollout.max_num_batched_tokens=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=$MODEL_DTYPE \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    +actor_rollout_ref.rollout.log_prob_top_k=$LOG_PROB_TOP_K \
    +actor_rollout_ref.rollout.top_k_strategy=$TOP_K_STRATEGY \
    +actor_rollout_ref.rollout.reward_weight_mode=$REWARD_WEIGHT_MODE \
    +actor_rollout_ref.rollout.teacher_temperature=$TEACHER_TEMPERATURE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$PARALLEL_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.n=$N_RESPONSES \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    +actor_rollout_ref.rollout.val_kwargs.max_tokens=$MAX_VAL_RESP_LENGTH \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.repetition_penalty=$REPETITION_PENALTY \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    reward_model.enable=True \
    +reward_model.reward_kwargs.enable_format_reward=$ENABLE_FORMAT_REWARD \
    reward_model.model.path=$REWARD_MODEL_PATH \
    reward_model.model.input_tokenizer=null \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=False \
    +reward_model.model.dtype=$MODEL_DTYPE \
    reward_model.micro_batch_size_per_gpu=24 \
    custom_reward_function.path="verl/verl/utils/reward_score/ttrl_math/__init__.py" \
    custom_reward_function.name=reward_func \
    trainer.val_before_train=False \
    trainer.log_val_generations=2 \
    trainer.logger="$LOGGER" \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.validation_data_dir=validation_log/$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="$CKPT_PATH" \
    trainer.is_plot=$IS_PLOT \
    ray_kwargs.ray_init.num_cpus=32 \
    +ray_kwargs.ray_init.address=local \
    +ray_kwargs.ray_init._temp_dir=/tmp/ray_opd \
    +ray_kwargs.ray_init.include_dashboard=False \
    +ray_kwargs.ray_init.object_store_memory=$((16 * 1024 * 1024 * 1024)) \
    "$@"

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
