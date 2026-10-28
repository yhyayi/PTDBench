#!/usr/bin/env bash
set -euo pipefail
# Complete official GSM8K train/test splits ship with this task.
# environment/data_manifest.json records the pinned upstream files and row selection.
# Every model asset is pinned by revision, byte size, and SHA-256.

REPO=Qwen/Qwen2.5-0.5B-Instruct
REV=7ae557604adf67be50417f59c2c2f167def9a775
DEST="${PTDBENCH_WORKSPACE}/model"

fetch() {
  local file="$1" sha="$2" size="$3"
  python3 "${PTDBENCH_ENVIRONMENT}/ptdbench_fetch_url.py" \
    --type model \
    --url "https://huggingface.co/${REPO}/resolve/${REV}/${file}" \
    --sha256 "${sha}" \
    --size "${size}" \
    --dest "${DEST}/${file}"
}

fetch config.json 18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45 659
fetch generation_config.json e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6 242
fetch model.safetensors fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe 988097824
fetch tokenizer.json c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539 7031645
fetch tokenizer_config.json 5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583 7305
fetch vocab.json ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910 2776833
fetch merges.txt 599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3 1671839
fetch LICENSE 832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e 11343
