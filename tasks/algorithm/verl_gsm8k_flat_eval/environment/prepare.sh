#!/usr/bin/env bash
# Prepare the pinned Qwen3 model and VERL-formatted GSM8K data for this task.
set -euo pipefail

# The launcher reads all assets from these workspace-relative paths.
ptdbench_fetch_hf \
    --repo Qwen/Qwen3-1.7B \
    --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
    --allow LICENSE \
    --allow config.json \
    --allow generation_config.json \
    --allow merges.txt \
    --allow model.safetensors.index.json \
    --allow 'model-*.safetensors' \
    --allow tokenizer.json \
    --allow tokenizer_config.json \
    --allow vocab.json \
    --expect config.json=1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197 \
    --expect tokenizer.json=aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4 \
    --expect tokenizer_config.json=5a7303fcb1a27ede63134a2cbd61d5282c247ca6d769ce4746d4ffa124aedd63 \
    --expect model-00001-of-00002.safetensors=169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5 \
    --expect model-00002-of-00002.safetensors=912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9 \
    --expect-size config.json=726 \
    --expect-size tokenizer.json=11422654 \
    --expect-size tokenizer_config.json=9675 \
    --expect-size model-00001-of-00002.safetensors=3441185608 \
    --expect-size model-00002-of-00002.safetensors=622329984 \
    --dest "$PTDBENCH_WORKSPACE/assets/Qwen3-1.7B"

# This pinned PTDBench recipe recreates VERL's original GSM8K preprocessing;
# output hashes/byte sizes are checked against the files used by the recorded run.
ptdbench_dataset \
    --recipe gsm8k \
    --expect train.parquet=970809ed1f53c5c3ffe791d6d1b817cce74ee951927d795068297a50eca900fe \
    --expect-size train.parquet=3562932 \
    --expect test.parquet=1fcb15b596f6987ffaa7aa5f78d0b9584fd2959dce18bc2e04e5e02400144ed1 \
    --expect-size test.parquet=648121 \
    --dest "$PTDBENCH_WORKSPACE/assets/gsm8k"
