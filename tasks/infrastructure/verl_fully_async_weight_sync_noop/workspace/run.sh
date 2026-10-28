#!/usr/bin/env bash
# Fully-async GRPO on GSM8K: Qwen3-1.7B, 2 GPUs, one trainer pool and one vLLM
# rollout pool in separate Ray resource pools (hybrid_engine=False). After every
# optimiser step the trainer pushes its new weights across to the rollout engine
# before the next generation round.
#
# Everything runs off the vendored tree in ./verl -- verl is NOT installed in the
# image, PYTHONPATH is what makes it importable, so editing ./verl is how you
# change this run's behaviour.
#
# The run is scored only from what lands under $PTDBENCH_RESULTS/experiment_logs:
#   train.log   the driver log, including the per-step metric lines and the
#               held-out validation lines (val-core/openai/gsm8k/acc/mean@4)
set -euo pipefail

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ./verl is the verl PACKAGE itself (./verl/trainer, ./verl/workers, ...), so what
# goes on PYTHONPATH -- and what hydra resolves its config searchpath against --
# is the directory that CONTAINS it, not the package directory.
VERL_ROOT="${WORKSPACE}"

# Both inputs are workspace-relative: environment/prepare.sh fetches them to
# exactly these paths, and nothing in the harness mounts a host path.
MODEL_PATH=${MODEL_PATH:-${WORKSPACE}/assets/Qwen3-1.7B}
DATA_DIR=${DATA_DIR:-${WORKSPACE}/assets/gsm8k}

RESULTS=${PTDBENCH_RESULTS:-${WORKSPACE}/results}
LOG_DIR=${LOG_DIR:-${RESULTS}/experiment_logs}
# Writable scratch for the checkpoint: large, rebuildable, and not scored.
CKPT_DIR=${CKPT_DIR:-${RESULTS}/ckpt}

# One rollout step is one prompt (gen_batch_size=1) and a trainer step consumes
# ppo_mini_batch_size=16 of them, so logged train steps = TOTAL_ROLLOUT_STEPS/16
# minus one (the last lands after the final metric line): 1632 -> 101.
# 1632 < 7473 rows, so this is one partial epoch and no prompt is seen twice.
TOTAL_ROLLOUT_STEPS=${TOTAL_ROLLOUT_STEPS:-1632}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
LR=${LR:-1e-6}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-384}
# Held-out validation every TEST_FREQ parameter versions, plus one before
# training. 1632/16 = 102 versions, so TEST_FREQ=20 gives 5 points plus step 0.
TEST_FREQ=${TEST_FREQ:-20}
# Validation sampling: 4 samples at temperature 1.0 is the distribution training
# samples from. The default (n=1, greedy) is one sample per prompt and therefore
# a noisier estimate of the same quantity.
VAL_N=${VAL_N:-4}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-1.0}

[ -s "${MODEL_PATH}/config.json" ] || { echo "ERROR: no model at ${MODEL_PATH}; run environment/prepare.sh" >&2; exit 2; }
[ -s "${DATA_DIR}/train.parquet" ] && [ -s "${DATA_DIR}/test.parquet" ] \
    || { echo "ERROR: no GSM8K parquet in ${DATA_DIR}; run environment/prepare.sh" >&2; exit 2; }

mkdir -p "${LOG_DIR}" "${CKPT_DIR}"
[ ! -e "${LOG_DIR}/train.log" ] || { echo "ERROR: ${LOG_DIR}/train.log exists; use a fresh results dir" >&2; exit 2; }

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${VERL_ROOT}:${PYTHONPATH:-}"
export VLLM_USE_V1=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0
export VERL_LOGGING_LEVEL=WARN
export NCCL_CUMEM_ENABLE=0

# The container is attached to exactly two GPUs, which appear here as 0 and 1.
# Do NOT set CUDA_VISIBLE_DEVICES to host indices -- NVML only sees the two
# attached devices and vLLM dies at import time on an out-of-range index.

# hydra's `searchpath: file://verl/trainer/config` is resolved against the cwd.
cd "${VERL_ROOT}"

# save_contents=[model,hf_model] makes the final actor a directory vLLM can load
# directly. save_freq=1 does NOT write a checkpoint per step: the per-step call
# into _fit_save_checkpoint passes force=False and never saves, so this yields
# exactly one checkpoint, of the final policy.
python3 -m verl.experimental.fully_async_policy.fully_async_main \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test.parquet" \
    data.prompt_key=prompt \
    data.truncation=left \
    data.max_prompt_length=512 \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    data.train_batch_size=0 \
    data.gen_batch_size=1 \
    data.return_raw_chat=True \
    data.shuffle=False \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.optim.lr="${LR}" \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.60 \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.rollout.val_kwargs.n="${VAL_N}" \
    actor_rollout_ref.rollout.val_kwargs.temperature="${VAL_TEMPERATURE}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    critic.strategy=fsdp2 \
    reward.custom_reward_function.path="${WORKSPACE}/tools/reward_gsm8k.py" \
    reward.custom_reward_function.name=compute_score \
    trainer.logger=['console'] \
    trainer.project_name=gsm8k_fully_async \
    trainer.experiment_name=fully_async_gsm8k \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=1 \
    "actor_rollout_ref.actor.checkpoint.save_contents=[model,hf_model]" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.val_before_train=True \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.save_freq=1 \
    trainer.resume_mode=disable \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.rollout_data_dir="${LOG_DIR}/rollouts" \
    rollout.nnodes=1 \
    rollout.n_gpus_per_node=1 \
    rollout.total_rollout_steps="${TOTAL_ROLLOUT_STEPS}" \
    async_training.staleness_threshold=0 \
    async_training.trigger_parameter_sync_step=1 \
    async_training.require_batches=1 \
    async_training.partial_rollout=False \
    2>&1 | tee "${LOG_DIR}/train.log"

echo "[run] done; logs in ${LOG_DIR}"
