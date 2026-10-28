#!/usr/bin/env bash
# Recorded assets are immutable. Every downloaded file is size+SHA checked
# against assets.json before use; prepare_data.py re-verifies the selected rows
# and the produced parquet outputs.
set -euo pipefail
: "${PTDBENCH_WORKSPACE:?}"
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
export PYTHONDONTWRITEBYTECODE=1

# prepare_data.py reads/writes parquet and pins outputs by content digest, so
# any pyarrow that round-trips the recorded content passes; only install if the
# image is missing pyarrow entirely.
python3 -c 'import pyarrow' 2>/dev/null \
  || uv pip install --system --python "$(command -v python3)" \
       --index-url "${PTDBENCH_PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
       'pyarrow==21.0.0'

python3 "$HERE/fetch_assets.py"
