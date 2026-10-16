#!/usr/bin/env bash
# ptdbench_rank_distillation_collapse — prepare stage.
#
# Runs inside the `rank-distillation-cpu` Docker image (python:3.12.12-slim
# with /opt/venv containing the issue-profile packages).  This is the only
# stage that may access the network.
#
# Issue profile (default): the venv is already baked into the image; nothing
# extra to install.  Writes prepared/env so downstream stages pick up the
# correct PYTHONPATH (verl source lives at /workspace/verl).
#
# Training profile (PTDBENCH_PREPARE_PROFILE=training): downloads the ~18.8 GB
# model weights and datasets, installs the GPU package closure, and compiles
# flash-attn from source.  Requires Linux x86_64 with a CUDA 13-compatible
# driver and compiler toolchain.  NOTE: this profile has not been validated
# with a clean GPU rerun; see readme.md §validation for details.
set -euo pipefail

: "${PTDBENCH_WORKSPACE:?must point to the staged workspace}"
: "${PTDBENCH_PREPARED:?must point to the prepared artifact directory}"
: "${PTDBENCH_ENVIRONMENT:?must point to this task environment directory}"

profile=${PTDBENCH_PREPARE_PROFILE:-issue}
case "$profile" in issue|training) ;; *) echo 'unknown prepare profile' >&2; exit 2 ;; esac

mkdir -p "$PTDBENCH_PREPARED"

if [[ "$profile" == training ]]; then
    python3 "$PTDBENCH_ENVIRONMENT/fetch_assets.py" --groups build model dataset

    export UV_PYTHON_INSTALL_DIR="$PTDBENCH_PREPARED/uv-python"
    export UV_CACHE_DIR=${UV_CACHE_DIR:-$PTDBENCH_PREPARED/uv-cache}
    export UV_LINK_MODE=${UV_LINK_MODE:-copy}

    uv venv --python 3.12.12 --allow-existing "$PTDBENCH_PREPARED/venv"
    uv pip install --python "$PTDBENCH_PREPARED/venv/bin/python" --no-deps \
        -r "$PTDBENCH_ENVIRONMENT/requirements-training.txt"
    uv pip install --python "$PTDBENCH_PREPARED/venv/bin/python" \
        --no-deps --no-build-isolation \
        "$PTDBENCH_PREPARED/sources/flash_attn-2.8.3.post1.tar.gz"
    "$PTDBENCH_PREPARED/venv/bin/python" "$PTDBENCH_ENVIRONMENT/prepare_data.py"
    "$PTDBENCH_PREPARED/venv/bin/python" "$PTDBENCH_ENVIRONMENT/prepare_models.py"

    python_bin="$PTDBENCH_PREPARED/venv/bin/python"
    venv_dir="$PTDBENCH_PREPARED/venv"
else
    # Issue profile: the image's /opt/venv has everything needed.
    python_bin="/opt/venv/bin/python"
    venv_dir="/opt/venv"
fi

# Write the handoff file consumed by ptdbench-activate.sh in subsequent stages.
# Idempotent: replace on repeat prepare, never append duplicate entries.
{
    printf 'VIRTUAL_ENV=%s\n' "$venv_dir"
    printf 'PATH=%s/bin:%s\n' "$venv_dir" "$PATH"
    printf 'PYTHONPATH=%s/verl\n' "$PTDBENCH_WORKSPACE"
    printf 'HF_HUB_OFFLINE=1\nHF_DATASETS_OFFLINE=1\nWANDB_MODE=disabled\nPYTHONDONTWRITEBYTECODE=1\n'
    printf 'PTDBENCH_PREPARE_PROFILE=%s\n' "$profile"
} > "$PTDBENCH_PREPARED/env"

"$python_bin" - "$PTDBENCH_PREPARED" "$profile" <<'PY'
import json, sys
from pathlib import Path
import torch
Path(sys.argv[1], 'prepare_manifest.json').write_text(json.dumps({
    'profile': sys.argv[2],
    'python': sys.version.split()[0],
    'torch': torch.__version__,
    'models_and_datasets_downloaded': sys.argv[2] == 'training',
}, indent=2) + '\n')
PY
