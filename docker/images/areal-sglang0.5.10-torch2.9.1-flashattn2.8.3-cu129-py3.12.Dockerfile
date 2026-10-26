# PTDBench environment `areal-sglang0.5.10-torch2.9.1-flashattn2.8.3-cu129-py3.12`.
# ptdbench-image: ptdbench:areal-sglang0.5.10-torch2.9.1-flashattn2.8.3-cu129-py3.12
#
# AReaL official runtime image, sglang variant (`ghcr.io/areal-project/areal-runtime`).
# Stack: sglang 0.5.10.post1 (.post1 kept in the digest), torch 2.9.1+cu129,
# flash-attn 2.8.3, CUDA 12.9.1, cuDNN 9.10.2, NCCL 2.27.3, Python 3.12.
# Tasks: the AReaL bugfix-replay family (first one: GSM8K GRPO FP32-master lr
# mismatch, AReaL @ 3cf0dfbd -- idea #8 in 构建思路收集).
#
# The AReaL framework is NOT pinned by this image. The upstream image bakes its own
# release AReaL (v2.0.0) at /AReaL, but each task overrides it with a commit-pinned
# AReaL snapshot carried in its bind-mounted workspace (e.g. 3cf0dfbd for the GSM8K
# GRPO task, with .git stripped and the pre-#1634 README). The `areal-` lead is a
# human-facing label for the family of AReaL tasks that share this stack, exactly
# like the `verl-` lead -- it is not a claim that a single baked AReaL version is
# these tasks' ground truth.
#
# The upstream image is used as-is; the only thing added is the PTDBench tooling
# block below. The FROM is pinned by digest because the ghcr tag moves upstream and
# the tasks' recorded outcomes must not move with it. Changing the digest is a
# deliberate upgrade: re-run the tasks on this environment after it.
FROM ghcr.io/areal-project/areal-runtime:v2.0.0-sglang@sha256:7ce1a21eee7cbdba6d1ac3d5cb5e1162edc392b8d499c9d5d7e17a1cf79276e3

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
