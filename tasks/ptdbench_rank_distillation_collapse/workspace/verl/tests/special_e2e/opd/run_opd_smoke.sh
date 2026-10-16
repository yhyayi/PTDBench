#!/usr/bin/env bash
# Colocated-OPD smoke test: 2 steps per strategy, small models, short sequences.
#
# Checks that every pass ordering in `_compute_opd` actually runs end to end:
#   top_k=0 (sampled token) + only_stu / only_tch / intersection / union
# Student and teacher share a tokenizer (both Qwen2 arch, vocab 151936).
#
# Usage: bash tests/special_e2e/opd/run_opd_smoke.sh [strategy ...]
set -xeuo pipefail

STUDENT_MODEL=${STUDENT_MODEL:-/workspace/model/DeepSeek-R1-Distill-Qwen-1.5B}
TEACHER_MODEL=${TEACHER_MODEL:-/workspace/model/JustRL-DeepSeek-1.5B}
DATA_DIR=${DATA_DIR:-$HOME/data/gsm8k_smoke}
# One GPU is enough: the teacher is colocated, so a single device exercises every OPD path.
# It also keeps rank 0 the only rank, which avoids the meta-tensor init path in
# `get_init_weight_context_manager` (accelerate 1.12 / transformers 5.5 are incompatible there,
# unrelated to OPD).
NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}
LOG_DIR=${LOG_DIR:-/tmp/opd_smoke}

mkdir -p "${LOG_DIR}"
[[ -f "${DATA_DIR}/train.parquet" ]] || python3 tests/special_e2e/opd/make_smoke_data.py --local_save_dir "${DATA_DIR}"

# "0" means top_k=0 (sampled-token OPD); the rest are top_k>0 strategies.
CASES=("$@")
[[ ${#CASES[@]} -gt 0 ]] || CASES=(0 only_stu only_tch intersection union)

for case in "${CASES[@]}"; do
    if [[ "${case}" == "0" ]]; then
        top_k=0; strategy=only_stu
    else
        top_k=${OPD_TOP_K:-8}; strategy=${case}
    fi

    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=opd_token_kl \
        algorithm.use_kl_in_reward=False \
        data.train_files="${DATA_DIR}/train.parquet" \
        data.val_files="${DATA_DIR}/test.parquet" \
        data.train_batch_size=8 \
        data.max_prompt_length=256 \
        data.max_response_length=256 \
        data.filter_overlong_prompts=True \
        data.truncation=error \
        actor_rollout_ref.model.path="${STUDENT_MODEL}" \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.model.use_fused_kernels=False \
        actor_rollout_ref.actor.strategy=fsdp2 \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=8 \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.entropy_coeff=0.0 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
        actor_rollout_ref.rollout.n=2 \
        actor_rollout_ref.rollout.max_model_len=513 \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
        opd.enable=True \
        opd.teacher_model_path="${TEACHER_MODEL}" \
        opd.top_k=${top_k} \
        opd.strategy=${strategy} \
        opd.chunk_size=2048 \
        actor_rollout_ref.teacher.fsdp_config.param_offload=True \
        actor_rollout_ref.teacher.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.teacher.log_prob_max_token_len_per_gpu=4096 \
        trainer.logger=console \
        trainer.project_name=verl_opd_smoke \
        trainer.experiment_name="smoke_${case}" \
        trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
        trainer.nnodes=1 \
        trainer.val_before_train=False \
        trainer.save_freq=-1 \
        trainer.test_freq=-1 \
        trainer.total_training_steps=2 \
        trainer.total_epochs=1 \
        ${EXTRA_ARGS:-} 2>&1 | tee "${LOG_DIR}/${case}.log"
done
