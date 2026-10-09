#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(
  cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd
)"

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# AIME25 is bundled with the task so runtime evaluation does not
# depend on a pre-existing host-side Inspect/Hugging Face cache.
BUNDLED_CACHE="$SCRIPT_DIR/assets/inspect_ai/hf_datasets"
TARGET_CACHE="${HOME}/.cache/inspect_ai/hf_datasets"

if [ ! -d "$BUNDLED_CACHE" ]; then
    echo "[ERROR] bundled AIME25 cache is missing:" >&2
    echo "        $BUNDLED_CACHE" >&2
    exit 3
fi

mkdir -p "$TARGET_CACHE"

cp -a \
  "$BUNDLED_CACHE"/yentinglin_aime_2025-* \
  "$TARGET_CACHE/"

exec python "$SCRIPT_DIR/run_eval.py" "$@"
