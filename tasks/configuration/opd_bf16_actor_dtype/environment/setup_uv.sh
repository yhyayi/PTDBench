#!/usr/bin/env bash
# Build this task's Python environment with uv:
#
#   bash environment/setup_uv.sh <venv-dir>
#
# requirements.txt is the frozen package closure of the environment the recorded run used
# (torch 2.8.0+cu128, vLLM 0.11.0, flash-attn 2.8.1, numpy 1.26.4), so it is installed as is with
# --no-deps. Needs uv >= 0.9, network access and an NVIDIA driver that supports CUDA 12.8.
# The verl source is not installed: on_policy_distillation.sh puts workspace/verl on PYTHONPATH.
# UV_DEFAULT_INDEX, UV_PYTHON_INSTALL_MIRROR, UV_PYTHON_INSTALL_DIR and UV_CACHE_DIR are honoured as usual.
set -euo pipefail

VENV=${1:?usage: setup_uv.sh <venv-dir>}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_VERSION=3.12.12

export UV_LINK_MODE=${UV_LINK_MODE:-copy}
uv venv "$VENV" --python "$PYTHON_VERSION"
uv pip install --python "$VENV/bin/python" --no-deps -r "$HERE/requirements.txt"

"$VENV/bin/python" - <<'PY'
import flash_attn, numpy, pyarrow, ray, torch, transformers, vllm

print(
    f"torch {torch.__version__} | vllm {vllm.__version__} | transformers {transformers.__version__} | "
    f"flash_attn {flash_attn.__version__} | ray {ray.__version__} | numpy {numpy.__version__} | "
    f"pyarrow {pyarrow.__version__}"
)
PY
