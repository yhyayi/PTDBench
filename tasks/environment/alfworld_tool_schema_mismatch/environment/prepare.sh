#!/usr/bin/env bash
# ALFWorld tool-use GRPO run: fetch the base model before training.
#
# Runs in a separate prepare-stage container: root, the `agent-r1-alfworld`
# image, **the only networked stage**, and $PTDBENCH_WORKSPACE, $PTDBENCH_MODELS,
# and $PTDBENCH_DATASETS are all writable.
#
# The ALFWorld dataset (parquets + extracted game files) ships inside the
# workspace at tool_schema/datasets/alfworld, so nothing is fetched for it here.
set -euo pipefail

# --- Base model ---------------------------------------------------------------
# train.sh reads the policy weights from "${MODEL_PATH:-$CASE_ROOT/Qwen3-4B}",
# where CASE_ROOT is tool_schema/, so the model must land at exactly
# tool_schema/Qwen3-4B inside the workspace.
#
# Sizes are intentionally not pinned with --expect-size: this packaging host had
# no network to read the revision's Content-Length, and the sha256 digests below
# (from the delivery's MODEL_SHA256SUMS) already verify the files exactly.
ptdbench_fetch_hf \
    --repo Qwen/Qwen3-4B \
    --revision 82d62bb073771e7a1ea59435f548908540217d1f \
    --allow config.json \
    --allow generation_config.json \
    --allow merges.txt \
    --allow model.safetensors.index.json \
    --allow model-00001-of-00003.safetensors \
    --allow model-00002-of-00003.safetensors \
    --allow model-00003-of-00003.safetensors \
    --allow tokenizer.json \
    --allow tokenizer_config.json \
    --allow vocab.json \
    --expect config.json=8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a \
    --expect model.safetensors.index.json=6dc0981b8829fead746441f68f38f24c5ca4a3a66351f652c26c6df0efc43ab2 \
    --expect model-00001-of-00003.safetensors=328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223 \
    --expect model-00002-of-00003.safetensors=6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5 \
    --expect model-00003-of-00003.safetensors=e4bf436957184f4eeb86a80e9db394503f1f56446b2e6b7edeac5b81470f4ca1 \
    --dest "$PTDBENCH_WORKSPACE/tool_schema/Qwen3-4B"
