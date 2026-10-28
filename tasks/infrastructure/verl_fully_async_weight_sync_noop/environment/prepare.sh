#!/usr/bin/env bash
# verl_fully_async_weight_sync_noop -- what to prepare before training.
#
# Runs in a separate prepare-stage container: root, the
# `verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12` image, **the only
# networked stage**, and $PTDBENCH_WORKSPACE, $PTDBENCH_MODELS and
# $PTDBENCH_DATASETS are all writable. Offline mode for the later stages is set
# by the framework per stage; do not repeat it in metadata.environment.env,
# which applies to every stage and would cut off prepare's own network.
#
# Three things to prepare. The two asset paths are fixed by run.sh's own guards
# (MODEL_PATH / DATA_DIR default to $WORKSPACE/assets/...), so --dest must match
# them exactly or the script exits 2 on a path that does not exist.
#
# The fetch/dataset helpers are shipped in this directory rather than taken from
# the image: the image Dockerfiles under docker/images/ are a single pinned FROM
# line and bake no tooling, so a task that needs --revision/--expect pinning has
# to carry it. Same arrangement as tasks/reward/search_r1_nomask.
set -euo pipefail

# --- Base model ---------------------------------------------------------------
# The pin is the full commit sha of Qwen/Qwen3-1.7B. Every sha256 and byte size
# below was read back from this revision through hf-mirror and cross-checked
# against the files the recorded runs actually loaded.
python3 "$PTDBENCH_ENVIRONMENT/ptdbench_fetch.py" hf \
    --repo Qwen/Qwen3-1.7B \
    --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
    --type model \
    --allow LICENSE \
    --allow config.json \
    --allow generation_config.json \
    --allow merges.txt \
    --allow model.safetensors.index.json \
    --allow 'model-*.safetensors' \
    --allow tokenizer.json \
    --allow tokenizer_config.json \
    --allow vocab.json \
    --expect config.json=1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197 \
    --expect tokenizer.json=aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4 \
    --expect tokenizer_config.json=d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101 \
    --expect model-00001-of-00002.safetensors=169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5 \
    --expect model-00002-of-00002.safetensors=912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9 \
    --expect-size config.json=726 \
    --expect-size tokenizer.json=11422654 \
    --expect-size tokenizer_config.json=9732 \
    --expect-size model-00001-of-00002.safetensors=3441185608 \
    --expect-size model-00002-of-00002.safetensors=622329984 \
    --dest "$PTDBENCH_WORKSPACE/assets/Qwen3-1.7B"

# --- Dataset ------------------------------------------------------------------
# The recipe rebuilds verl's own GSM8K preprocessing (upstream's train/test
# split, 7473/1319 rows, the `#### ` answer instruction appended to each
# question) into the row schema RLHFDataset requires.
#
# These two checksums are environment-dependent and were taken from inside this
# task's image: the parquet bytes are reproducible for a fixed pandas/pyarrow
# (verified by building twice in the image and comparing), but a different
# pandas writes different bytes for the same rows. Re-read them in the image if
# the image is ever repinned; the row contents are unchanged either way.
python3 "$PTDBENCH_ENVIRONMENT/ptdbench_dataset.py" \
    --recipe gsm8k \
    --expect train.parquet=f2f3e1935904b74935f8db8822721979cd5b08827e0049db5bf5d0011eb0e27e \
    --expect-size train.parquet=3379875 \
    --expect test.parquet=ae5fbf081fcef21853f5d5652a6a9fb7dec1f4678cac2e85e11fe5bb65a794bc \
    --expect-size test.parquet=618541 \
    --dest "$PTDBENCH_WORKSPACE/assets/gsm8k"

# --- cupy ---------------------------------------------------------------------
# The image ships cupy-cuda12x 14.x against numpy 1.26, so `import cupy` raises
# ImportError. verl/checkpoint_engine/__init__.py wraps every backend import in
# try/except ImportError, so that one failed import silently removes several
# checkpoint-engine backends from the registry, and any run that asks for one of
# them dies with "Checkpoint engine <name> not registered" instead. Restoring a
# cupy that imports is part of making this release's checkpoint-engine registry
# complete; it is not specific to any one configuration of this task.
#
# This is done here rather than in the image because docker/images/*.Dockerfile
# is a single pinned FROM line shared by other tasks, and prepare is the only
# networked stage.
python3 -c 'import cupy' >/dev/null 2>&1 || \
    pip install --no-cache-dir "cupy-cuda12x<14"
python3 -c 'import cupy; print("cupy", cupy.__version__)'
