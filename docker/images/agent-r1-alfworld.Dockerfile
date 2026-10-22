# PTDBench environment `agent-r1-alfworld`.
# ptdbench-image: ptdbench/agent-r1-alfworld:vllm0.11-torch2.8-verl0.7
#
# Agent-R1 ALFWorld tool-use GRPO stack: Python 3.10, torch 2.8.0, vLLM 0.11.0,
# verl 0.7.0, flash-attn 2.8.3, transformers 4.57.1, ALFWorld 0.4.2,
# TextWorld 1.6.2, ray 2.48.0, numpy 1.26.4.
# Tasks: alfworld_tool_schema_mismatch.
#
# Why this is not one of the shared verl images: those are vLLM 0.17 / torch
# 2.10 / CUDA 12.9, many minor versions past the rollout API this workspace's
# Agent-R1 recipe is written against. The recorded 63-step curves this task
# grades against were produced on the stack below.
#
# The Agent-R1 recipe code is NOT in this image -- it ships in the task
# workspace under tool_schema/vendor/Agent-R1 and is put on PYTHONPATH by
# train.sh. Only the pinned Python stack lives here. The stack is installed
# from docker/agent-r1-runtime/ (the delivery's own requirements files), which
# is what pins it; this Dockerfile is just the recipe.
#
# UNVERIFIED: this image has never been built. It was written on a machine with
# no GPU and no reachable Docker Hub / GitHub / PyPI, so:
#   * the flash-attn wheel URL below follows Dao-AILab's release naming for
#     2.8.3 (cu12 / torch2.8 / cxx11abiTRUE / cp310, the build matching the
#     PyPI torch 2.8.0 wheels). If it 404s the build falls back to compiling
#     flash-attn 2.8.3 from its sdist, which is slow (~30-60 min) but produces
#     the same version. The wheel's expected sha256 is pinned either way.
#   * transitive dependency versions are not fully locked; the delivery recorded
#     the realized versions in the built image at /opt/build/pip-freeze-alfworld.txt.
FROM ubuntu:22.04@sha256:5ec03bb3441e8b0bf3b4f9cd4629a1ae763010dc3035bb8da3ae6cf026486401

ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# CUDA user-space libraries are supplied by the torch/vLLM wheels; the host
# still needs a compatible NVIDIA driver and container runtime. curl/ca-certificates
# are for the flash-attn wheel fetch; build-essential is its sdist fallback.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev \
        ca-certificates curl git build-essential libgomp1 libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*
RUN python3.10 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    VIRTUAL_ENV=/opt/venv \
    MAX_JOBS=4

COPY agent-r1-runtime/requirements.txt /opt/build/requirements.txt
COPY agent-r1-runtime/requirements-alfworld.txt /opt/build/requirements-alfworld.txt

# Pass 1: the main stack (torch must exist before flash-attn is touched).
RUN python -m pip install --no-cache-dir pip==25.3 setuptools==80.9.0 wheel==0.45.1 \
    && python -m pip install --no-cache-dir -r /opt/build/requirements.txt

# Pass 2: flash-attn 2.8.3 from Dao-AILab's prebuilt wheel (cu12 torch2.8 cp310,
# cxx11abiTRUE to match the PyPI torch 2.8.0 wheels), falling back to the sdist
# if the release URL is gone. The sha256 is verified in both directions.
RUN set -eux; \
    FA_WHEEL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl; \
    FA_SHA=75c51f34bb93c5a4438d6b767547ca6c22a9bfc6b29b5646dda98e943cd75d99; \
    if curl -fsSL --retry 3 -o /tmp/flash_attn.whl "$FA_WHEEL"; then \
        echo "$FA_SHA  /tmp/flash_attn.whl" | sha256sum -c -; \
        python -m pip install --no-cache-dir --no-deps /tmp/flash_attn.whl; \
        rm -f /tmp/flash_attn.whl; \
    else \
        echo "prebuilt flash-attn wheel unavailable; compiling 2.8.3 from sdist" >&2; \
        python -m pip install --no-cache-dir --no-build-isolation flash-attn==2.8.3; \
    fi; \
    python -m pip check

# Pass 3: ALFWorld + TextWorld, constrained so they do not move the main stack.
RUN python -m pip freeze > /opt/build/pip-freeze.txt \
    && python -m pip install --no-cache-dir \
        -c /opt/build/pip-freeze.txt -r /opt/build/requirements-alfworld.txt \
    && python -m pip check \
    && python -c 'import torch, vllm, verl, transformers, ray; \
print("torch", torch.__version__); print("vllm", vllm.__version__); \
print("verl", verl.__version__); print("transformers", transformers.__version__)' \
    && python -c 'import flash_attn; print("flash-attn", flash_attn.__version__)' \
    && python -c 'import alfworld, textworld, textworld.gym' \
    && python -m pip freeze > /opt/build/pip-freeze-alfworld.txt

# ---- PTDBench tooling: keep identical in every docker/images/*.Dockerfile ----
# Build context is docker/ (see docker/.dockerignore for what it may COPY).

RUN mkdir -p /ptdbench/prepared /ptdbench/models /ptdbench/datasets

# Every command the harness runs goes through `sh -lc`, and a login shell's
# /etc/profile may reset PATH. Record the image's own PATH so the hook can put
# back whatever it drops (/usr/local/cuda/bin, /root/.cargo/bin, ...).
RUN printf '%s\n' "$PATH" > /etc/ptdbench-image-path
COPY ptdbench-activate.sh /etc/profile.d/ptdbench-activate.sh
RUN set -eux; \
    sed -i 's/\r$//' /etc/profile.d/ptdbench-activate.sh; \
    chmod 0644 /etc/profile.d/ptdbench-activate.sh; \
    sh -n /etc/profile.d/ptdbench-activate.sh

ENV CUDA_DEVICE_ORDER=PCI_BUS_ID \
    TOKENIZERS_PARALLELISM=false

# The prepare-stage fetch/dataset tooling (ptdbench_fetch.py, ptdbench_dataset.py,
# ptdbench_datasets/) is deliberately NOT baked here. docker/install-ptdbench-tooling.sh
# installs it into /usr/local at the start of every prepare run, from the repo's
# current copy (mounted RO locally, uploaded on Cybertron). That keeps a fix to
# those files reaching running jobs without an image rebuild. Only the activate
# hook above stays baked -- it must exist in the agent/verifier containers too,
# which never run prepare-time provisioning.

# The harness bind-mounts a per-run copy of the task's workspace here.
WORKDIR /workspace
