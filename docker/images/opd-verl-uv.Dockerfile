# PTDBench environment `opd-verl-uv`.
# ptdbench-image: ptdbench/opd-verl-uv:uv-torch2.8-cu128
#
# CUDA 12.8 base that provides `uv` plus the PTDBench tooling. The actual
# torch 2.8.0+cu128 / vLLM 0.11.0 / flash-attn 2.8.1 stack is NOT baked here:
# environment/prepare.sh builds it into a uv venv under $PTDBENCH_PREPARED from
# the task's frozen environment/requirements.txt (via environment/setup_uv.sh),
# so the recorded run's exact package closure is reproduced, not the image's.
#
# Tasks: configuration/opd_bf16_actor_dtype.
#
# The base matches the ppo-critic-vimpo image (CUDA 12.8.1 / Ubuntu 24.04); the
# only additions are uv and the shared tooling block. The workspace verl source
# is NOT baked: on_policy_distillation.sh puts workspace/verl on PYTHONPATH.
FROM docker.m.daocloud.io/nvidia/cuda:12.8.1-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive

ENV CUDA_HOME=/usr/local/cuda \
    PYTHONUNBUFFERED=1 \
    PIP_ROOT_USER_ACTION=ignore

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        git-lfs \
        curl \
        wget \
        ca-certificates \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

# uv (>= 0.9, required by environment/setup_uv.sh) ships its own CPython, so the
# image does not depend on a system Python. prepare.sh points UV_PYTHON_INSTALL_DIR
# and the venv at $PTDBENCH_PREPARED.
COPY --from=ghcr.io/astral-sh/uv:0.9.2 /uv /uvx /usr/local/bin/

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
