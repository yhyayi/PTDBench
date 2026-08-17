#!/usr/bin/env bash
set -euo pipefail

# 这个版本让 agent 和 verifier 共用 Docker backend；task 镜像需提供 opencode。
PYTHONDONTWRITEBYTECODE=1 python3 -m ptdbench evaluate example/swe_slugfix \
  --backend docker \
  --agent opencode \
  --agent-command "opencode" \
  --model /model \
  --base-url "http://127.0.0.1:30004/v1" \
  --api-key "EMPTY" \
  --work-dir "outputs/ptdbench-swe-opencode-docker-debug" \
  --keep-artifacts \
  --timeout-s 1800 \
  --json
