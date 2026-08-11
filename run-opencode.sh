#!/usr/bin/env bash
set -euo pipefail

# 使用 g22 上已经启动的 SGLang 服务作为 OpenCode 的模型后端。
# agent 使用 local，保证能够调用宿主机上的 opencode 命令；verifier
# 也使用 local，便于先快速调试 example。运行目录放在仓库外，避免
# OpenCode 把 PTDBench 源码仓库误识别为当前任务的项目根。
PYTHONDONTWRITEBYTECODE=1 python3 -m ptdbench evaluate example/swe_slugfix \
  --backend local \
  --agent-backend local \
  --agent opencode \
  --agent-command "/home/test/test1706/.nvm/versions/node/v24.15.0/bin/opencode" \
  --model /model \
  --base-url "http://127.0.0.1:30004/v1" \
  --api-key "EMPTY" \
  --work-dir "outputs/ptdbench-swe-opencode-debug" \
  --keep-artifacts \
  --timeout-s 1800 \
  --json
