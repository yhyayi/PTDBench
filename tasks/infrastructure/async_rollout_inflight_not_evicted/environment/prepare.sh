#!/usr/bin/env bash
# miles_async_rollout_inflight_not_evicted: prepare assets before training.
#
# Runs in a separate prepare-stage container: root, the `miles` image,
# **the only networked stage**, and $PTDBENCH_WORKSPACE, $PTDBENCH_MODELS and
# $PTDBENCH_DATASETS are all writable.
#
# The DAPO-Math-17k dedup prompt set (data/dapo_math_17k_dedup.jsonl) ships in
# the workspace, so there is no dataset fetch here. One asset is downloaded (the
# base model) and one is converted on the spot (its torch_dist reference).
set -euo pipefail

# --- Base model (--hf-checkpoint) --------------------------------------------
# scripts/train_grpo.sh reads --hf-checkpoint "$WS/Qwen3-1.7B", so --dest must be
# exactly $PTDBENCH_WORKSPACE/Qwen3-1.7B. Qwen3-1.7B ships two safetensors shards
# on HF; the revision pins the small config/tokenizer files and the two shards
# are verified by sha256 + size.
ptdbench_fetch_hf \
    --repo Qwen/Qwen3-1.7B \
    --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
    --allow config.json \
    --allow generation_config.json \
    --allow merges.txt \
    --allow model-00001-of-00002.safetensors \
    --allow model-00002-of-00002.safetensors \
    --allow model.safetensors.index.json \
    --allow tokenizer.json \
    --allow tokenizer_config.json \
    --allow vocab.json \
    --expect model-00001-of-00002.safetensors=169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5 \
    --expect-size model-00001-of-00002.safetensors=3441185608 \
    --expect model-00002-of-00002.safetensors=912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9 \
    --expect-size model-00002-of-00002.safetensors=622329984 \
    --dest "$PTDBENCH_WORKSPACE/Qwen3-1.7B"

# --- Megatron torch_dist reference weights (--ref-load) ----------------------
# No upstream repo; converted from the HF checkpoint above. scripts/train_grpo.sh
# does this itself (its `if [ ! -d "$WS/Qwen3-1.7B_torch_dist" ]` guard) when it
# runs; doing it here keeps the GPU-hours for training. The miles source tree is
# in the task workspace, not the image.
#
# Conversion needs CUDA + NCCL (dist.init_process_group + torch.cuda.set_device).
# The harbor backend runs prepare as a GPU-less build layer, so **skip instead
# of erroring** there and let the training script's guard do it in the agent
# stage, where /workspace is still writable. The dist-checkpoint shard layout
# follows WORLD_SIZE; pinning it to 1 gives the pp=1 single-shard layout the
# training run (tp=pp=1) reloads regardless of machine.
WS="$PTDBENCH_WORKSPACE"
HF_MODEL_DIR="$WS/Qwen3-1.7B"
DIST_MODEL_DIR="$WS/Qwen3-1.7B_torch_dist"

if python3 -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    echo "[prepare] converting $HF_MODEL_DIR -> $DIST_MODEL_DIR (WORLD_SIZE=1)"
    # shellcheck source=/dev/null
    source "$WS/scripts/models/qwen3-1.7B.sh"
    mkdir -p "$DIST_MODEL_DIR"
    WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
    PYTHONPATH="/root/Megatron-LM:$WS" \
    python3 "$WS/tools/convert_hf_to_torch_dist.py" \
        "${MODEL_ARGS[@]}" \
        --hf-checkpoint "$HF_MODEL_DIR" \
        --save "$DIST_MODEL_DIR"
else
    echo "[prepare] no CUDA device visible; leaving the torch_dist conversion to"
    echo "[prepare] the training script's own guard (scripts/train_grpo.sh)."
fi
