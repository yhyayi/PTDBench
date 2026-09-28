# PTDBench environment `verl`.
# ptdbench-image: ptdbench/verl:vllm0.17-torch2.10-cu129
#
# verl + vLLM 0.17, torch 2.10 / CUDA 12.9, Ubuntu 22.04.
# Tasks: reward_design/*, verl_hyper/*, verl_implementation/*.
#
# The upstream image is used as-is; the only thing added is the PTDBench
# tooling block below. The FROM is pinned by digest because the tag moves
# upstream and the tasks' recorded outcomes must not move with it. Changing the
# digest is a deliberate upgrade: re-run the tasks on this environment after it.
FROM verlai/verl:vllm017.latest@sha256:4c43bbf17e90284b1102008399240b25406e8d34fea178d86272231b333b7cb6

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

# Called by every task's environment/prepare.sh. The shims resolve `python3`
# at call time, so they run under the image's interpreter and its
# huggingface_hub / datasets.
COPY ptdbench_fetch.py /usr/local/libexec/ptdbench_fetch.py
RUN set -eux; \
    for kind in hf url; do \
        printf '#!/bin/sh\nexec python3 /usr/local/libexec/ptdbench_fetch.py %s "$@"\n' \
            "${kind}" > "/usr/local/bin/ptdbench_fetch_${kind}"; \
        chmod 0755 "/usr/local/bin/ptdbench_fetch_${kind}"; \
    done
COPY ptdbench_dataset.py /usr/local/libexec/ptdbench_dataset.py
COPY ptdbench_datasets /usr/local/libexec/ptdbench_datasets
RUN set -eux; \
    printf '#!/bin/sh\nexec python3 /usr/local/libexec/ptdbench_dataset.py "$@"\n' \
        > /usr/local/bin/ptdbench_dataset; \
    chmod 0755 /usr/local/bin/ptdbench_dataset

# The harness bind-mounts a per-run copy of the task's workspace here.
WORKDIR /workspace
