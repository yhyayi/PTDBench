#!/usr/bin/env bash
# rank_distillation_collapse — prepare stage.
#
# Runs inside the `rank-distillation-cpu` Docker image (python:3.12.12-slim
# with /opt/venv containing the issue-profile packages).  This is the only
# stage that may access the network.
#
# Issue profile (default): the venv is already baked into the image; nothing
# extra to install.  Writes prepared/env so downstream stages pick up the
# correct PYTHONPATH (verl source lives at /workspace/verl).
#
# Training profile (AUTORLBENCH_PREPARE_PROFILE=training): downloads the ~18.8 GB
# model weights and datasets, installs the GPU package closure, and compiles
# flash-attn from source.  Requires Linux x86_64 with a CUDA 13-compatible
# driver and compiler toolchain.  NOTE: this profile has not been validated
# with a clean GPU rerun.
set -euo pipefail

: "${AUTORLBENCH_WORKSPACE:?must point to the staged workspace}"
: "${AUTORLBENCH_PREPARED:?must point to the prepared artifact directory}"
: "${AUTORLBENCH_ENVIRONMENT:?must point to this task environment directory}"

profile=${AUTORLBENCH_PREPARE_PROFILE:-issue}
case "$profile" in issue|training) ;; *) echo 'unknown prepare profile' >&2; exit 2 ;; esac

mkdir -p "$AUTORLBENCH_PREPARED"

if [[ "$profile" == training ]]; then
    python3 "$AUTORLBENCH_ENVIRONMENT/fetch_assets.py" --groups build model dataset

    export UV_PYTHON_INSTALL_DIR="$AUTORLBENCH_PREPARED/uv-python"
    export UV_CACHE_DIR=${UV_CACHE_DIR:-$AUTORLBENCH_PREPARED/uv-cache}
    export UV_LINK_MODE=${UV_LINK_MODE:-copy}

    uv venv --python 3.12.12 --allow-existing "$AUTORLBENCH_PREPARED/venv"
    uv pip install --python "$AUTORLBENCH_PREPARED/venv/bin/python" --no-deps \
        -r "$AUTORLBENCH_ENVIRONMENT/requirements-training.txt"
    uv pip install --python "$AUTORLBENCH_PREPARED/venv/bin/python" \
        --no-deps --no-build-isolation \
        "$AUTORLBENCH_PREPARED/sources/flash_attn-2.8.3.post1.tar.gz"
    "$AUTORLBENCH_PREPARED/venv/bin/python" "$AUTORLBENCH_ENVIRONMENT/prepare_data.py"
    "$AUTORLBENCH_PREPARED/venv/bin/python" "$AUTORLBENCH_ENVIRONMENT/prepare_models.py"

    python_bin="$AUTORLBENCH_PREPARED/venv/bin/python"
    venv_dir="$AUTORLBENCH_PREPARED/venv"
else
    # Issue profile: the image's /opt/venv has everything needed.
    python_bin="/opt/venv/bin/python"
    venv_dir="/opt/venv"
fi

# Write the handoff file consumed by autorlbench-activate.sh in subsequent stages.
# Idempotent: replace on repeat prepare, never append duplicate entries.
{
    printf 'VIRTUAL_ENV=%s\n' "$venv_dir"
    printf 'PATH=%s/bin:%s\n' "$venv_dir" "$PATH"
    printf 'PYTHONPATH=%s/verl\n' "$AUTORLBENCH_WORKSPACE"
    printf 'HF_HUB_OFFLINE=1\nHF_DATASETS_OFFLINE=1\nWANDB_MODE=disabled\nPYTHONDONTWRITEBYTECODE=1\n'
    printf 'AUTORLBENCH_PREPARE_PROFILE=%s\n' "$profile"
} > "$AUTORLBENCH_PREPARED/env"

"$python_bin" - "$AUTORLBENCH_PREPARED" "$profile" <<'PY'
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
