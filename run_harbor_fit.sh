#!/usr/bin/env bash
set -euo pipefail

# Harbor is provided by the local clone. PTDBENCH_TASK may be one task or a
# directory whose direct children are tasks. Credentials stay in environment vars.
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running}"

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://yeysai.com/v1}"

# Keep uv state inside the workspace and avoid downloading Python from GitHub.
export UV_CACHE_DIR="${UV_CACHE_DIR:-outputs/uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-outputs/uv-python}"
export UV_NO_MANAGED_PYTHON="${UV_NO_MANAGED_PYTHON:-1}"
export UV_NO_PYTHON_DOWNLOADS="${UV_NO_PYTHON_DOWNLOADS:-1}"
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

args=(
  -m ptdbench harbor-evaluate
  "${PTDBENCH_TASK:-tasks/slime_retool}"
  --evaluation-level "${PTDBENCH_EVALUATION_LEVEL:-solution}"
  --judge-model "${PTDBENCH_JUDGE_MODEL:-${PTDBENCH_MODEL:-gpt-5.5}}"
  --debian-mirror "${PTDBENCH_DEBIAN_MIRROR:-https://mirrors.tuna.tsinghua.edu.cn}"
  --verifier-mode "${PTDBENCH_VERIFIER_MODE:-separate}"
  --agent-network "${PTDBENCH_AGENT_NETWORK:-public}"
  --verifier-network "${PTDBENCH_VERIFIER_NETWORK:-public}"
  --harbor-command "${PTDBENCH_HARBOR_COMMAND:-uv run --no-project --python python3 --with-editable ./harbor harbor}"
  --harbor-temp-dir "${PTDBENCH_HARBOR_TEMP_DIR:-outputs/harbor-tmp}"
  --agent "${PTDBENCH_HARBOR_AGENT:-codex}"
  --model "${PTDBENCH_MODEL:-gpt-5.5}"
  --agent-kwarg "reasoning_effort=${PTDBENCH_REASONING_EFFORT:-high}"
  --agent-env "NVM_NODEJS_ORG_MIRROR=${PTDBENCH_NODE_MIRROR:-https://npmmirror.com/mirrors/node}"
  --agent-env "npm_config_registry=${PTDBENCH_NPM_REGISTRY:-https://registry.npmmirror.com}"
  --env "${PTDBENCH_BACKEND:-docker}"
  --n-concurrent "${PTDBENCH_N_CONCURRENT:-1}"
  --timeout-multiplier "${PTDBENCH_TIMEOUT_MULTIPLIER:-4}"
  --jobs-dir "${PTDBENCH_JOBS_DIR:-outputs/harbor-slime-retool}"
  --image "${PTDBENCH_IMAGE_TAG:-ptdbench/env:latest}"
  --environment-build-timeout-multiplier 36
  --verifier-timeout-multiplier 12
  --yes
  --debug
)

PYTHONDONTWRITEBYTECODE=1 python3 "${args[@]}" "$@"
