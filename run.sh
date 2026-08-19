#!/usr/bin/env bash
set -euo pipefail

# The evaluator itself runs on the host and starts Docker containers for the
# agent/verifier stages. Keep credentials out of this script and out of git.
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running}"

args=(
  -m ptdbench evaluate "${PTDBENCH_TASK:-example/swe_slugfix}"
  --evaluation-level "${PTDBENCH_EVALUATION_LEVEL:-solution}"
  --log-level "${PTDBENCH_LOG_LEVEL:-debug}"
  --backend "${PTDBENCH_BACKEND:-docker}"
  --agent codex
  --model "${PTDBENCH_MODEL:-gpt-5.5}"
  --judge-model "${PTDBENCH_JUDGE_MODEL:-${PTDBENCH_MODEL:-gpt-5.5}}"
  --reasoning-effort "${PTDBENCH_REASONING_EFFORT:-high}"
  --base-url "${OPENAI_BASE_URL:-https://yeysai.com/v1}"
  --api-key-env OPENAI_API_KEY
  --debian-mirror "${PTDBENCH_DEBIAN_MIRROR:-https://mirrors.tuna.tsinghua.edu.cn}"
  --agent-env "NVM_NODEJS_ORG_MIRROR=${PTDBENCH_NODE_MIRROR:-https://npmmirror.com/mirrors/node}"
  --agent-env "npm_config_registry=${PTDBENCH_NPM_REGISTRY:-https://registry.npmmirror.com}"
  --timeout-s "${PTDBENCH_TIMEOUT_S:-3600}"
  --work-dir "${PTDBENCH_WORK_DIR:-outputs/ptdbench-swe-debug}"
  --keep-artifacts
  --json
)

PYTHONDONTWRITEBYTECODE=1 python3 "${args[@]}" "$@"
