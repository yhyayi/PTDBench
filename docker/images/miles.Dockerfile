# PTDBench environment `miles`.
# ptdbench-image: ptdbench/miles:ngc2506-cu12
#
# NOT YET BUILT. The recorded A/B runs used the internal training image
# `llm-rl/ngc-2506:nvidia.11b60366` (NGC 25.06 base: CUDA 12, Megatron-LM under
# /root/Megatron-LM, sglang with the miles patch, Ray). The miles framework
# itself is NOT part of the image -- it ships in the task workspace -- so the
# image only needs the training/rollout stack (torch + CUDA + NCCL, Megatron-LM,
# sglang 0.5.16, ray) plus the PTDBench tooling block.
#
# Pin FROM to a digest before building (the internal tag moves, and the tasks'
# recorded outcomes must not move with it). The public miles base builds on
# `lmsysorg/sglang:v0.5.16`; the internal ngc-2506 image is the one the runs
# were measured on.
FROM llm-rl/ngc-2506:nvidia.11b60366

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
