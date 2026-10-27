#!/usr/bin/env bash
# Recorded assets are immutable. Every downloaded file is checked before conversion.
set -euo pipefail
: "${PTDBENCH_WORKSPACE:?}"
export W="$PTDBENCH_WORKSPACE"
mkdir -p "$W/metadata" "$W/data/gsm8k" "$W/.tmp" "$W/.home"
export HOME="$W/.home" TMPDIR="$W/.tmp"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$W/src/$(cat "$W/variant.txt"):$W/scripts"
python3 "$PTDBENCH_ENVIRONMENT/ptdbench_fetch.py" \
    --repo Qwen/Qwen1.5-MoE-A2.7B-Chat \
    --type model \
    --dest "$W/models/Qwen1.5-MoE-A2.7B-Chat"

python3 "$PTDBENCH_ENVIRONMENT/ptdbench_fetch.py" \
    --repo openai/gsm8k \
    --type dataset \
    --dest "$W/data/raw/gsm8k"

python3 "$W/scripts/prepare_gsm8k.py"
