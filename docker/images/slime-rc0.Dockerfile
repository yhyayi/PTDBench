# PTDBench environment `slime-rc0`.
# ptdbench-image: ptdbench/slime:v0.5.0rc0-sglang0.5.0rc0-cu126
#
# slime v0.5.0rc0 stack: sglang 0.5.0rc0, torch 2.8 / CUDA 12.6, Ubuntu 22.04.
# Tasks: rlve_hyper/*.
#
# Kept separate from slime-v0.3.2 on purpose: the rlve tasks recorded their
# outcomes on this torch, and the retool / search-r1 recipes need sglang APIs
# (sglang.srt.constants.GPU_MEMORY_TYPE_CUDA_GRAPH) this sglang does not have.
#
# The upstream image is used as-is; the only thing added is the PTDBench
# tooling block below. The FROM is pinned by digest because the tag moves
# upstream and the tasks' recorded outcomes must not move with it.
FROM slimerl/slime:v0.5.0rc0-cu126@sha256:8685221e76f85e89b7ffe8e13a9e8c57041c15c63b3878bd2cecc8ae50deec11

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
