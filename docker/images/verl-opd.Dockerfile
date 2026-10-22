# PTDBench environment `verl-opd`.
# ptdbench-image: ptdbench/verl-opd:vllm0.24-torch2.11-cu130
#
# verl (colocated top-k OPD recipe) + vLLM 0.24, torch 2.11 / CUDA 13.0,
# Ubuntu 24.04, flash-attn 2.8.3, transformers 5.5.3.
# Tasks: infrastructure/opd_candidate_gradient.
#
# This environment is newer than the shared `verl` image (vLLM 0.17 / torch
# 2.10 / CUDA 12.9): the OPD recipe under verl/trainer/opd needs the newer vLLM
# and CUDA. The dependency versions below are carried over verbatim from the
# raw task's own environment/Dockerfile; only the model-download and
# dataset-copy layers are dropped — those are now the prepare stage's job
# (environment/prepare.sh) and the task's workspace, not the image.
#
# Status: NOT YET BUILT. No upstream base is pinned by digest here because the
# task has not been run end to end on a built copy of this image; build and pin
# before recording outcomes against it.
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.12
ARG TORCH_VERSION=2.11.0
ARG TRANSFORMERS_VERSION=5.5.3
ARG FLASH_ATTENTION_VERSION=2.8.3
ARG VLLM_VERSION=0.24.0
ARG RAY_VERSION=2.56.1
ARG TENSORDICT_VERSION=0.10.0
ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git build-essential cmake ninja-build \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python

RUN curl --fail --location --retry 5 https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
    && python /tmp/get-pip.py \
    && rm /tmp/get-pip.py

RUN pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cu130 \
    && pip install \
       accelerate==1.14.0 codetiming==1.4.0 datasets==5.0.0 dill==0.4.1 \
       hydra-core==1.3.4 numpy==1.26.4 pandas==3.0.5 \
       pyarrow==25.0.0 pybind11==3.0.4 pylatexenc==2.10 \
       transformers==${TRANSFORMERS_VERSION} \
       ray[default]==${RAY_VERSION} tensordict==${TENSORDICT_VERSION} \
       fastapi==0.136.3 uvicorn==0.51.0 packaging==25.0 \
       mathruler==0.1.0 latex2sympy2_extended==1.11.0 math-verify==0.9.0 \
       liger-kernel==0.8.1 peft==0.19.1 torchdata==0.11.0 \
       wandb==0.28.1 tensorboard==2.21.0 TransferQueue==0.1.8 \
       swanlab==0.6.4 huggingface_hub==1.24.0

RUN MAX_JOBS=32 pip install --no-build-isolation flash-attn==${FLASH_ATTENTION_VERSION} \
    && pip install vllm==${VLLM_VERSION}

# The verl source tree is the task's own checkout and ships in the bind-mounted
# workspace (workspace/verl); opd.sh runs `python3 -m verl.trainer.main_ppo`
# from that directory, and the solution verifier sets PYTHONPATH to it. So verl
# is deliberately NOT pip-installed into the image.

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
