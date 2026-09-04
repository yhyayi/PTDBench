#!/usr/bin/env bash
set -euo pipefail

repo_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running}"

task_root="$repo_root/example/qwen_dapo_hparam/task_hparam_seqlen_micro_014"

cd "$repo_root"
exec env PYTHONDONTWRITEBYTECODE=1 python3 -m ptdbench evaluate "$task_root" \
  --evaluation-level issue \
  --log-level "${PTDBENCH_LOG_LEVEL:-debug}" \
  --backend "${PTDBENCH_BACKEND:-docker}" \
  --agent codex \
  --model "${PTDBENCH_MODEL:-gpt-5.5}" \
  --judge-model "${PTDBENCH_JUDGE_MODEL:-${PTDBENCH_MODEL:-gpt-5.5}}" \
  --reasoning-effort "${PTDBENCH_REASONING_EFFORT:-high}" \
  --base-url "${OPENAI_BASE_URL:-https://yeysai.com/v1}" \
  --api-key-env OPENAI_API_KEY \
  --timeout-s "${PTDBENCH_TIMEOUT_S:-3600}" \
  --work-dir "${PTDBENCH_WORK_DIR:-outputs/qwen-dapo-hparam-seqlen-micro-014-issue}" \
  --keep-artifacts \
  --json
