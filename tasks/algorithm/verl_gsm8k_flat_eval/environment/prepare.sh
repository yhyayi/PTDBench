#!/usr/bin/env bash
# Recorded assets are immutable. Every downloaded file is size+SHA checked
# against assets.json before use; prepare_data.py re-verifies the derived
# GSM8K rows against content pins (writer-independent).
set -euo pipefail
: "${PTDBENCH_WORKSPACE:?}"
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
export PYTHONDONTWRITEBYTECODE=1

# prepare_data.py reads/writes parquet. The verl image ships pyarrow; pin the
# recorded major version if a rebuilt image is ever missing it.
python3 -c 'import pyarrow' 2>/dev/null \
  || uv pip install --system --python "$(command -v python3)" \
       --index-url "${PTDBENCH_PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
       'pyarrow==20.0.0'

python3 "$HERE/fetch_assets.py"
