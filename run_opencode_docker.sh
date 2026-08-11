#!/usr/bin/env bash
set -euo pipefail

# 这个版本把 PTDBench 的 case / verifier 放进 Docker 隔离里，
# 但 OpenCode 仍然在宿主机上运行，方便直接访问 g22 上的 SGLang。
PYTHONDONTWRITEBYTECODE=1 python3 -m ptdbench evaluate example/swe_slugfix \
  --backend docker \
  --agent-backend docker \
  --agent opencode \
  --agent-command "/home/test/test1706/.nvm/versions/node/v24.15.0/bin/opencode" \
  --model /model \
  --base-url "http://127.0.0.1:30004/v1" \
  --api-key "EMPTY" \
  --work-dir "outputs/ptdbench-swe-opencode-docker-debug" \
  --keep-artifacts \
  --timeout-s 1800 \
  --json
