#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONUNBUFFERED=1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# slime repo root (this recipe lives at <slime>/examples/retool)
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
source "${SLIME_DIR}/scripts/models/qwen3-4B.sh"

# Use the HuggingFace mirror for all downloads.
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

# Root locations for data and models; default to $PWD, override via env.
DATA_DIR=${DATA_DIR:-$PWD}
MODEL_DIR=${MODEL_DIR:-$PWD}

# --- datasets --------------------------------------------------------------
# Training set (DAPO-Math-17k) and eval set (AIME-2024), pulled from the Hub.
TRAIN_DATA_DIR=${TRAIN_DATA_DIR:-$DATA_DIR/zhuzilin_dapo-math-17k}
EVAL_DATA_DIR=${EVAL_DATA_DIR:-$DATA_DIR/zhuzilin_aime-2024}
TRAIN_DATA=${TRAIN_DATA:-$TRAIN_DATA_DIR/dapo-math-17k.jsonl}
EVAL_DATA=${EVAL_DATA:-$EVAL_DATA_DIR/aime-2024.jsonl}

# Download the training set if it is not already present under DATA_DIR.
if [ ! -f "$TRAIN_DATA" ]; then
   echo "[data] $TRAIN_DATA missing; downloading zhuzilin/dapo-math-17k from $HF_ENDPOINT"
   hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir "$TRAIN_DATA_DIR"
fi

# Download the eval set if it is not already present under DATA_DIR.
if [ ! -f "$EVAL_DATA" ]; then
   echo "[data] $EVAL_DATA missing; downloading zhuzilin/aime-2024 from $HF_ENDPOINT"
   hf download --repo-type dataset zhuzilin/aime-2024 --local-dir "$EVAL_DATA_DIR"
fi

# --- model checkpoints -----------------------------------------------------
# The SFT'd HF checkpoint and its mcore (torch_dist) conversion used as ref weights.
HF_MODEL_DIR=${HF_MODEL_DIR:-$MODEL_DIR/font-info_qwen3-4b-sft}
DIST_MODEL_DIR=${DIST_MODEL_DIR:-$MODEL_DIR/font-info_qwen3-4b-sft_torch_dist}

# Download the HF checkpoint if it is not already present under MODEL_DIR.
if [ ! -d "$HF_MODEL_DIR" ] || [ -z "$(ls -A "$HF_MODEL_DIR" 2>/dev/null)" ]; then
   echo "[model] $HF_MODEL_DIR missing; downloading font-info/qwen3-4b-sft-SGLang-RL from $HF_ENDPOINT"
   hf download font-info/qwen3-4b-sft-SGLang-RL --local-dir "$HF_MODEL_DIR"
fi

# Convert to a torch_dist (mcore) checkpoint if not already done.
if [ ! -d "$DIST_MODEL_DIR" ] || [ -z "$(ls -A "$DIST_MODEL_DIR" 2>/dev/null)" ]; then
   echo "[model] converting $HF_MODEL_DIR -> $DIST_MODEL_DIR (torch_dist)"
   PYTHONPATH=${MEGATRON_LM_DIR:-/root/Megatron-LM} python "${SLIME_DIR}/tools/convert_hf_to_torch_dist.py" \
      "${MODEL_ARGS[@]}" \
      --hf-checkpoint "$HF_MODEL_DIR" \
      --rotary-base 5000000 \
      --save "$DIST_MODEL_DIR"
fi

CKPT_ARGS=(
   --hf-checkpoint "$HF_MODEL_DIR"
   --ref-load "$DIST_MODEL_DIR"
   # --load "$MODEL_DIR/Qwen3-4B_slime/"
   --save "$HF_MODEL_DIR/qwen3-4b-sft-multi-turn/"
   --save-interval 10
   --rotary-base 5000000
)

ROLLOUT_ARGS=(
   --prompt-data "$TRAIN_DATA"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --reward-key score
   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-context-len 16384
   --rollout-max-response-len 16384
   --rollout-temperature 1

   --global-batch-size 256
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 10
   --eval-prompt-data aime  "$EVAL_DATA"
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 1
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
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-retool
   --wandb-group qwen3-4B-retool-multi-turn
   --wandb-key ${WANDB_API_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.7
   --sglang-server-concurrency 128
   # 让 router 自己扛过 ~1s 抖动，客户端就不会看到 500
   --router-retry-max-retries 8
   --router-retry-initial-backoff-ms 200
   --router-retry-max-backoff-ms 3000
   --router-retry-backoff-multiplier 1.5
   # colocate 突发下熔断弊大于利
   --router-disable-circuit-breaker
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
   --custom-generate-function-path generate_with_retool.generate
   --custom-rm-path generate_with_retool.reward_func
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_LM_DIR:-/root/Megatron-LM}/:${SCRIPT_DIR}:${SLIME_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}