# PTDBench environment `search-r1-verl-vllm0.6.3-torch2.4.0-flashattn2.6.3-cu121-py3.10`.
# ptdbench-image: ptdbench:search-r1-verl-vllm0.6.3-torch2.4.0-flashattn2.6.3-cu121-py3.10
#
# Search-R1 training runtime: Python 3.10, torch 2.4.0, vLLM 0.6.3,
# flash-attn 2.6.3, transformers 4.46.3, ray 2.40.0, faiss-cpu 1.14.2,
# numpy 1.26.4, CUDA 12.1.
# Tasks: search_r1_zero_query_vector.
#
# Why this is not the shared verl image: that one is vLLM 0.17 / torch 2.10 /
# CUDA 12.9, many vLLM minors past the rollout API this workspace is written
# against; the recorded 32-step curves this task grades against were produced on
# the stack below.
#
# The Search-R1 framework (with veRL vendored in it) is NOT baked: it arrives
# with the task's bind-mounted workspace at /workspace. The `search-r1-verl-`
# lead is a human-facing family label, not a claim the framework lives in the
# image -- the image is just the pinned Python runtime.
#
# The stack is a prebuilt image published to ghcr.io/yhyayi/ptdbench and pinned
# by digest below; its build recipe (uv.lock-pinned) is maintained separately in
# docker-build/search-r1-verl/ (see docker/README.zh.md §2). The base bakes the
# full /opt/search-r1-runtime/.venv and its ENV (PATH, VIRTUAL_ENV), so python3
# in every stage is the pinned interpreter and nothing is reinstalled here.
FROM ghcr.io/yhyayi/ptdbench:search-r1-verl-vllm0.6.3-torch2.4-cu121@sha256:ec9414a9beaea910b834348e41d13000356046b14ccf88fd994685b755228552

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
