#!/usr/bin/env bash
# Colocated on-policy distillation | KL-in-reward | vLLM rollout | FSDP training
#
# Unlike `distillation.*` (examples/on_policy_distillation_trainer), the teacher here is a
# forward-only engine worker inside the *actor's* resource pool. It needs no GPUs of its own,
# so a single-GPU box works: student and teacher take turns on the same device, and only
# [B, T, K] log-probs ever leave it.
#
# The KL becomes the advantage:
#   top_k=0                 sampled-token OPD; adv = -(log p_stu - log p_tch) at the token
#   top_k>0, strategy=...   a sparse KL policy gradient on a per-position token set
#     only_stu       the student's top-k
#     only_tch       the teacher's top-k
#     intersection   ids in both top-k sets
#     union          ids in either top-k set (needs one extra student forward)
#
# Forward passes per step: 2 for every strategy except `union`, which needs 3.

set -xeuo pipefail

# ---- user-adjustable ----
STUDENT_MODEL=${STUDENT_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}
TEACHER_MODEL=${TEACHER_MODEL:-Qwen/Qwen2.5-Math-7B-Instruct}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

opd_top_k=${OPD_TOP_K:-8}
opd_strategy=${OPD_STRATEGY:-only_stu}
opd_reward_weight_mode=${OPD_REWARD_WEIGHT_MODE:-student_p}
opd_is_norm=${OPD_IS_NORM:-False}
opd_outside_bucket=${OPD_OUTSIDE_BUCKET:-False}
opd_teacher_temperature=${OPD_TEACHER_TEMPERATURE:-1.0}
# Rows of the packed (total_nnz, vocab) logits gathered per step. Lower it if the teacher
# forward OOMs; it bounds the transient buffer without changing the result.
opd_chunk_size=${OPD_CHUNK_SIZE:-4096}
# opd_token_kl = pure distillation; opd_token_kl_plus_grpo adds the verifiable-reward term.
adv_estimator=${ADV_ESTIMATOR:-opd_token_kl}

train_batch_size=${TRAIN_BATCH_SIZE:-32}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-32}
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-2048}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-12288}

actor_lr=${ACTOR_LR:-1e-6}
rollout_n=${ROLLOUT_N:-4}
rollout_tp=${ROLLOUT_TP:-1}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.55}

total_epochs=${TOTAL_EPOCHS:-3}
save_freq=${SAVE_FREQ:-100}
test_freq=${TEST_FREQ:-10}

project_name=${PROJECT_NAME:-verl_opd_colocate}
experiment_name=${EXPERIMENT_NAME:-opd_${opd_strategy}_k${opd_top_k}}
# ---- end user-adjustable ----

train_files=${TRAIN_FILES:-$HOME/data/gsm8k/train.parquet}
val_files=${VAL_FILES:-$HOME/data/gsm8k/test.parquet}

max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=${adv_estimator}
    # The distillation KL *is* the reward here; a second in-reward KL would double-count it.
    algorithm.use_kl_in_reward=False
    data.train_files="$train_files"
    data.val_files="$val_files"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
)

MODEL=(
    actor_rollout_ref.model.path="$STUDENT_MODEL"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    # Fused kernels skip logits materialization, which the candidate gather needs.
    actor_rollout_ref.model.use_fused_kernels=False
)

ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=0.0
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
)

OPD=(
    opd.enable=True
    opd.teacher_model_path="$TEACHER_MODEL"
    opd.top_k=${opd_top_k}
    opd.strategy=${opd_strategy}
    opd.reward_weight_mode=${opd_reward_weight_mode}
    opd.is_norm=${opd_is_norm}
    opd.outside_bucket=${opd_outside_bucket}
    opd.teacher_temperature=${opd_teacher_temperature}
    opd.chunk_size=${opd_chunk_size}
    # The teacher is forward-only, so offloading its params costs nothing but a copy.
    actor_rollout_ref.teacher.fsdp_config.param_offload=True
    actor_rollout_ref.teacher.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.teacher.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
)

TRAINER=(
    trainer.logger=console
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=False
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
)

########################### launch ###########################
# Watch opd/kl_mean (should fall), opd/overlap_ratio (student/teacher top-k agreement) and
# opd/teacher_mass (how much teacher probability the candidate set covers).
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${OPD[@]}" \
    "${TRAINER[@]}" \
    "$@"
