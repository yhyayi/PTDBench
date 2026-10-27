#!/usr/bin/env bash
set -euo pipefail

python3 "$PTDBENCH_ENVIRONMENT/ptdbench_fetch.py" \
    --repo Qwen/Qwen3-4B \
    --dest "$PTDBENCH_WORKSPACE/tool_schema/Qwen3-4B"
