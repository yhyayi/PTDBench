#!/usr/bin/env bash
set -euo pipefail

# Harbor 由独立 clone 提供；凭据通过环境变量传入，不进入命令参数或 git。
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://yeysai.com/v1}"

PYTHONDONTWRITEBYTECODE=1 python3 -m ptdbench harbor-evaluate \
  "${PTDBENCH_TASK:-example/swe_slugfix}" \
  --evaluation-level "${PTDBENCH_EVALUATION_LEVEL:-solution}" \
  --judge-model "${PTDBENCH_JUDGE_MODEL:-${PTDBENCH_MODEL:-gpt-5.5}}" \
  --debian-mirror "${PTDBENCH_DEBIAN_MIRROR:-https://mirrors.tuna.tsinghua.edu.cn}" \
  --verifier-mode "${PTDBENCH_VERIFIER_MODE:-separate}" \
  --agent-network "${PTDBENCH_AGENT_NETWORK:-public}" \
  --verifier-network "${PTDBENCH_VERIFIER_NETWORK:-public}" \
  --harbor-command "${PTDBENCH_HARBOR_COMMAND:-uv run --project harbor harbor}" \
  --harbor-temp-dir "${PTDBENCH_HARBOR_TEMP_DIR:-outputs/harbor-tmp}" \
  --agent "${PTDBENCH_HARBOR_AGENT:-codex}" \
  --model "${PTDBENCH_MODEL:-gpt-5.5}" \
  --agent-kwarg "reasoning_effort=${PTDBENCH_REASONING_EFFORT:-high}" \
  --env "${PTDBENCH_BACKEND:-docker}" \
  --timeout-multiplier "${PTDBENCH_TIMEOUT_MULTIPLIER:-4}" \
  --jobs-dir "${PTDBENCH_JOBS_DIR:-outputs/harbor-swe-debug}" \
  --yes \
  --debug \
  "$@"
