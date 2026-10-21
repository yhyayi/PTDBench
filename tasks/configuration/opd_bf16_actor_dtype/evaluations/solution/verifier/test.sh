#!/usr/bin/env bash
# Launches a shortened OPD run through the repaired workspace's on_policy_distillation.sh and judges it
# with tests/check_run.py against the buggy and reference runs recorded in the oracle.
set -uo pipefail
: "${PTDBENCH_WORKSPACE:?PTDBENCH_WORKSPACE is required}"
: "${PTDBENCH_RESULTS:?PTDBENCH_RESULTS is required}"
: "${PTDBENCH_VERIFIER:?PTDBENCH_VERIFIER is required}"
: "${PTDBENCH_ORACLE:?PTDBENCH_ORACLE is required}"

STEPS=15
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
N_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c .)
mkdir -p "$PTDBENCH_RESULTS"
RUN_LOG="$PTDBENCH_RESULTS/verify_run.log"
CKPT_DIR="$PTDBENCH_RESULTS/verify_ckpt"

# Only the workspace's own launcher decides the settings the agent is allowed to change.
unset MODEL_DTYPE MINI_BATCH_SIZE N_RESPONSES MAX_RESP_LENGTH MAX_VAL_RESP_LENGTH LOG_PROB_TOP_K \
      TEMPERATURE TEACHER_TEMPERATURE USE_KL LOGGER RUN_TS TEST_FILE
export LOG_DIR="$PTDBENCH_RESULTS/launcher_logs"

# The recipe is pinned; batch, length and memory settings are scaled down to one shortened run.
OVERRIDES=(
    data.train_files=datasets/dapo-math-17k.parquet
    data.shuffle=False
    data.train_batch_size=16
    data.max_response_length=3072
    actor_rollout_ref.model.path=model/DeepSeek-R1-Distill-Qwen-1.5B
    reward_model.model.path=model/JustRL-DeepSeek-1.5B
    algorithm.adv_estimator=token_reward_direct
    ++actor_rollout_ref.rollout.log_prob_top_k=0
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.n=4
    actor_rollout_ref.rollout.max_model_len=4096
    ++actor_rollout_ref.rollout.val_kwargs.max_tokens=3072
    actor_rollout_ref.rollout.max_num_batched_tokens=16384
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5
    actor_rollout_ref.actor.ppo_mini_batch_size=16
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.weight_decay=0.01
    "actor_rollout_ref.actor.optim.betas=[0.9,0.999]"
    actor_rollout_ref.actor.grad_clip=1.0
    "actor_rollout_ref.actor.checkpoint.save_contents=['hf_model']"
    reward_model.micro_batch_size_per_gpu=4
    trainer.total_epochs=1
    trainer.total_training_steps=$STEPS
    trainer.save_freq=$STEPS
    trainer.test_freq=-1
    trainer.val_before_train=False
    trainer.is_plot=False
    "trainer.logger=['console']"
    trainer.nnodes=1
    trainer.n_gpus_per_node=$N_GPUS
    trainer.default_local_dir="$CKPT_DIR"
)

# A failure before the first training step (e.g. vLLM's startup memory profiling racing with other
# processes) is retried once; a failure after training started is judged as is.
cd "$PTDBENCH_WORKSPACE"
for attempt in 1 2; do
    rm -rf "$CKPT_DIR"
    timeout "${PTDBENCH_VERIFY_TRAIN_TIMEOUT:-10800}" bash on_policy_distillation.sh "${OVERRIDES[@]}" > "$RUN_LOG" 2>&1
    train_status=$?
    if [[ $train_status -eq 0 || $attempt -eq 2 ]] || grep -q " step:1 - " "$RUN_LOG"; then
        break
    fi
    mv "$RUN_LOG" "$PTDBENCH_RESULTS/verify_run.attempt1.log"
done

PYTHONDONTWRITEBYTECODE=1 python3 "$PTDBENCH_VERIFIER/tests/check_run.py" \
    --log "$RUN_LOG" \
    --ckpt-dir "$CKPT_DIR" \
    --steps "$STEPS" \
    --train-status "$train_status" \
    --init-model "$PTDBENCH_WORKSPACE/model/DeepSeek-R1-Distill-Qwen-1.5B" \
    --expected "$PTDBENCH_VERIFIER/resources/expected.json" \
    --oracle "$PTDBENCH_ORACLE/metrics.json" \
    --result "$PTDBENCH_RESULTS/result.json"
