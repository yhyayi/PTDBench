# PTDBench environment `vllm0.11.0-torch2.8.0-flashattn2.8.3-cu128-py3.12`.
# ptdbench-image: ptdbench:vllm0.11.0-torch2.8.0-flashattn2.8.3-cu128-py3.12
#
# veRL PPO-Critic / VIMPO runtime, the environment that produced the recorded
# ppo_critic runs:
#   Python       3.12
#   CUDA         12.8
#   torch        2.8.0 / torchvision 0.23.0 / torchaudio 2.8.0
#   vLLM         0.11.0
#   Ray          2.56.0
#   transformers 4.57.6
#   flash-attn   2.8.3.post1
#
# Tasks: ppo_critic/* (reward_parser_mismatch, excessive_actor_lr_collapse,
# proxy_reward_misalignment; aime25_single_sample_variance rides this image for
# its stdlib-only protocol check).
#
# This stack is DELIBERATELY NOT the shared verl image (vLLM 0.17 / torch 2.10
# / CUDA 12.9). The ppo_critic workspaces ship their own veRL + VIMPO recipe
# written against vLLM 0.11, and their recorded val-core / critic-reward
# baselines were produced on this exact stack; the solution grader reconstructs
# those baselines, so the stack must not move.
#
# The stack was originally assembled here from nvidia/cuda + pip; it is now a
# prebuilt image published to ghcr.io/yhyayi/ptdbench and pinned by digest below
# (the detailed build process is maintained separately -- see docker/README.zh.md
# §2 "Dockerfile 的两种形态"). The base already bakes the full /opt/venv stack and
# its ENV (PATH, VIRTUAL_ENV, PYTHONPATH=/workspace, VLLM_WORKER_MULTIPROC_METHOD,
# ...), so nothing is reinstalled here. The workspace veRL is NOT baked: the
# harness bind-mounts each task's workspace at /workspace and the baked
# PYTHONPATH=/workspace makes `import verl` resolve to that mounted source.
FROM ghcr.io/yhyayi/ptdbench:vllm0.11-torch2.8-cu128@sha256:6f7c302c0161a6c32d978faade165c86c153575baf028e729d7eebe44c06f683

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
