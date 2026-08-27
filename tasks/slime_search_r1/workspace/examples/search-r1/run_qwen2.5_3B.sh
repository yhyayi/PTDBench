#!/bin/bash

# Kill leftover python from a previous run, but spare the retrieval server so it
# can be reused across reruns (reloading its ~61GB index takes minutes).
kill_python_except_retriever() {
   local keep pid
   keep=" $(pgrep -f 'local_dense_retriever/retrieval_server.py' 2>/dev/null | tr '\n' ' ')"
   for pid in $(pgrep -f 'python' 2>/dev/null); do
      case "$keep" in
         *" $pid "*) ;;                 # this is the retriever -> keep it
         *) kill -9 "$pid" 2>/dev/null ;;
      esac
   done
}

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
kill_python_except_retriever
sleep 3
pkill -9 ray
kill_python_except_retriever

set -ex

# The rollout runs up to search_concurrency (256) concurrent aiohttp requests to
# the local retrieval server, on top of sglang/ray file descriptors. The default
# soft limit (1024) is far too low and the rollout crashes with
# "OSError: [Errno 24] Too many open files". Raise it as high as the hard limit
# allows; ray workers launched below inherit this shell's limit.
ulimit -n "$(ulimit -Hn)"

# will prevent ray from buffering stdout/stderr
export PYTHONUNBUFFERED=1

# Online W&B initialization is unreliable from this devspace and a timeout
# kills the RolloutManager actor. Default to local logging; callers can opt in
# with WANDB_MODE=online once network access is stable.
export WANDB_MODE=${WANDB_MODE:-offline}

# The persistent faiss retriever (--faiss_gpu) holds ~9GB more on GPU0 than on the
# other GPUs, so SGLang's TP memory-balance check ("memory capacity is unbalanced")
# fatally aborts the GPU0-paired engine on restart. The imbalance is expected and
# within the mem-fraction-static headroom. NOTE: slime gives each rollout engine an
# isolated ray runtime_env, so exporting the var here does NOT reach those actors --
# the actual disable lives in slime/ray/rollout.py's engine env_vars. This export
# only covers the driver/train side and is kept as defense-in-depth.
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# slime repo root (this recipe lives at <slime>/examples/search-r1)
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/qwen2.5-3B.sh"

# Use the HuggingFace mirror for all downloads.
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

DATA_DIR=${DATA_DIR:-$PWD}
MODEL_DIR=${MODEL_DIR:-$PWD}
bash "$SCRIPT_DIR/prepare_data.sh"

DATASET_DIR=${DATASET_DIR:-$DATA_DIR/nq_hotpotqa_train}
TRAIN_DATA=${TRAIN_DATA:-$DATASET_DIR/train.parquet}
TEST_DATA=${TEST_DATA:-$DATASET_DIR/test.parquet}

# Model checkpoints under MODEL_DIR: the HF checkpoint and its mcore
# (torch_dist) conversion used as the reference weights.
HF_MODEL_DIR=${HF_MODEL_DIR:-$MODEL_DIR/Qwen2.5-3B}
DIST_MODEL_DIR=${DIST_MODEL_DIR:-$MODEL_DIR/Qwen2.5-3B_torch_dist}

# Download the HF checkpoint if it is not already present under MODEL_DIR.
if [ ! -d "$HF_MODEL_DIR" ] || [ -z "$(ls -A "$HF_MODEL_DIR" 2>/dev/null)" ]; then
   echo "[model] $HF_MODEL_DIR missing; downloading Qwen/Qwen2.5-3B from $HF_ENDPOINT"
   hf download Qwen/Qwen2.5-3B --local-dir "$HF_MODEL_DIR"
fi

# Convert to a torch_dist (mcore) checkpoint if not already done.
if [ ! -d "$DIST_MODEL_DIR" ] || [ -z "$(ls -A "$DIST_MODEL_DIR" 2>/dev/null)" ]; then
   echo "[model] converting $HF_MODEL_DIR -> $DIST_MODEL_DIR (torch_dist)"
   PYTHONPATH=${MEGATRON_LM_DIR:-/root/Megatron-LM} python "${SLIME_DIR}/tools/convert_hf_to_torch_dist.py" \
      "${MODEL_ARGS[@]}" \
      --hf-checkpoint "$HF_MODEL_DIR" \
      --save "$DIST_MODEL_DIR"
fi

CKPT_ARGS=(
   --hf-checkpoint "$HF_MODEL_DIR"
   --ref-load "$DIST_MODEL_DIR"
   # --load "$MODEL_DIR/Qwen2.5-3B_slime/"
   --save "$MODEL_DIR/Qwen2.5-3B_slime/"
   --save-interval 10
)

# --- require the local dense retrieval server -------------------------------
RETRIEVAL_PORT=${RETRIEVAL_PORT:-8000}
if ! (exec 3<>"/dev/tcp/127.0.0.1/${RETRIEVAL_PORT}") 2>/dev/null; then
   echo "[retrieval] no server on :${RETRIEVAL_PORT}. Start it first: examples/search-r1/start_retriever.sh" >&2
   exit 1
fi
echo "[retrieval] server up on :${RETRIEVAL_PORT}"

ROLLOUT_ARGS=(
   --prompt-data "$TRAIN_DATA"
   --input-key prompt
   --label-key reward_model
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-context-len 8129
   --rollout-max-response-len 1024
   --rollout-temperature 1

   # eval args
   --eval-interval 10
   --eval-prompt-data nq_test "$TEST_DATA@[0:3000]"
   --eval-input-key prompt
   --eval-label-key reward_model
   --n-samples-per-eval-prompt 1

   --global-batch-size 256
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28

   # whether enabling TIS
   # --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-search-r1
   --wandb-group search-r1_qwen2.5-3B
   --wandb-key ${WANDB_API_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.7
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_search.generate
   --custom-rm-path generate_with_search.reward_func

   # TIS-related args, recommended to enable when using TIS
   # --custom-config-path examples/train_infer_mismatch_helper/mis.yaml
   # --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_LM_DIR:-/root/Megatron-LM}/:${SCRIPT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK\": \"0\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --rollout-num-gpus 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}
