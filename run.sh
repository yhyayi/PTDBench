#!/usr/bin/env bash
set -euo pipefail

# The evaluator itself runs on the host and starts Docker containers for the
# agent/verifier stages. Keep credentials out of this script and out of git.
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running}"

PYTHONDONTWRITEBYTECODE=1 python3 -m ptdbench evaluate "${PTDBENCH_TASK:-example/swe_slugfix}" \
  --evaluation-level "${PTDBENCH_EVALUATION_LEVEL:-solution}" \
  --log-level "${PTDBENCH_LOG_LEVEL:-quiet}" \
  --backend "${PTDBENCH_BACKEND:-docker}" \
  --agent-backend "${PTDBENCH_AGENT_BACKEND:-docker}" \
  --agent codex \
  --model "${PTDBENCH_MODEL:-gpt-5.5}" \
  --reasoning-effort "${PTDBENCH_REASONING_EFFORT:-high}" \
  --base-url "${OPENAI_BASE_URL:-https://yeysai.com/v1}" \
  --api-key-env OPENAI_API_KEY \
  --timeout-s "${PTDBENCH_TIMEOUT_S:-3600}" \
  --work-dir "${PTDBENCH_WORK_DIR:-outputs/ptdbench-swe-debug}" \
  --keep-artifacts \
  --json
