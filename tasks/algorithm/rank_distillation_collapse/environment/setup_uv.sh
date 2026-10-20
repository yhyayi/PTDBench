#!/usr/bin/env bash
set -euo pipefail
venv=${1:?usage: setup_uv.sh VENV [issue|training]}
profile=${2:-issue}
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
case "$profile" in issue|training) ;; *) echo 'profile must be issue or training' >&2; exit 2 ;; esac
export UV_LINK_MODE=${UV_LINK_MODE:-copy}
uv venv --python 3.12.12 --allow-existing "$venv"
if [[ "$profile" == issue ]]; then
  uv pip sync --python "$venv/bin/python" --require-hashes \
    --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
    "$HERE/requirements.txt"
  "$venv/bin/python" -c 'import torch, yaml; assert not torch.cuda.is_available(); print("CPU issue environment:", torch.__version__)'
else
  # The recorded checkout deliberately runs from PYTHONPATH: its generic setup.py
  # dependency ranges do not describe the already-tested recorded GPU closure.
  uv pip install --python "$venv/bin/python" --no-deps -r "$HERE/requirements-training.txt"
  : "${AUTORLBENCH_PREPARED:?training profile requires AUTORLBENCH_PREPARED}"
  uv pip install --python "$venv/bin/python" --no-deps --no-build-isolation \
    "$AUTORLBENCH_PREPARED/sources/flash_attn-2.8.3.post1.tar.gz"
  "$venv/bin/python" -c 'import torch, vllm, flash_attn, transformers, transfer_queue; print(torch.__version__, vllm.__version__, flash_attn.__version__, transformers.__version__)'
fi
