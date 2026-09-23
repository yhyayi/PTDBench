# PTDBench unified evaluation image.
# Each environment is built at /opt/conda/envs/<name> in its own stage
#
# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
#   DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile -t ptdbench/env:latest docker/
#
# docker/build.sh wraps exactly that, plus the registry check. BuildKit builds
# the env-* stages in parallel; `--target env-slime-rc0` builds just one, which
# is how you iterate on a single environment without waiting for the others.
#
# GitHub-only git proxy (host tunnel). The env-* RUN stages see the host's
# 127.0.0.1 only with --network=host:
#   ./docker/build.sh --network=host --build-arg GITHUB_HTTP_PROXY=http://127.0.0.1:8444
#
# ---------------------------------------------------------------------------
# What the build wants from the host, and what happens without it
# ---------------------------------------------------------------------------
# Two prefetch scripts fill two directories inside the build context, and each
# env-* stage COPYs its own share out:
#
#   ./docker/fetch-cuda-wheels.sh    docker/wheels/<env>
#   ./docker/fetch-git-mirrors.sh    docker/gitmirror/<env>
#
# Both are optional. The directories are committed (each holds a .gitkeep), and
# COPY of an empty directory is not an error, so a bare `docker build` on a
# fresh clone still works: torch falls back to pytorch.org, flash_attn compiles
# from the PyPI sdist, and build-env.sh clones its git pins from github itself.
# Optional, but not free -- that fallback is a couple of extra hours and needs
# the build to reach the network.
#
# Per environment, rather than one directory for all three, and that split is
# the point: what a COPY hashes into its stage's cache key is the subtree it
# names, so under one shared directory adding a slime-v0.3.1 wheel sent verl and
# slime-rc0 back to a full recompile too. Per-subdirectory is the whole of what
# buys that; the six named build contexts this used to go through bought nothing
# on top of it.
#
# If github.com:443 is filtered but :22 is not, the SSH rewrite now lives in
# fetch-git-mirrors.sh, on the host, where ssh has an agent and no conda openssl
# to collide with:
#   GITHUB_GIT_SSH=1 ./docker/fetch-git-mirrors.sh
#   ./docker/build.sh --target env-slime-v0.3.1
# Nothing about the image changes: the mirrors record github URLs, so pip still
# resolves git+https and `pip freeze` still says so.

# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=ubuntu:22.04

# ===========================================================================
# base -- the build environment, and nothing else
# ===========================================================================
# No framework, no CUDA toolkit, no Python packages. CUDA lives in the
# environments because its version differs between them; Python likewise. What
# is left is a compiler, a build system and a fetcher, which is what every
# environment needs and none of them disagree about. Measured 0.46 GB.
FROM ${BASE_IMAGE} AS base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        git-lfs \
        libnuma-dev \
        ninja-build \
        pkg-config \
        tzdata \
    ; \
    git lfs install --system --skip-repo; \
    rm -rf /var/lib/apt/lists/*

# micromamba drives the conda layer. It is a single binary with no Python of
# its own, so it cannot be confused with the environments it creates. The
# linux-64 2.0.5 build is vendored next to this Dockerfile (docker/micromamba)
# so the image build does not pull mambaorg/micromamba.
COPY micromamba /usr/local/bin/micromamba
RUN chmod 0755 /usr/local/bin/micromamba \
 && echo 'cdf5a2e18d213617b6c5c32d8763b4e8fc05f8173b4f4017997683cb984357eb  /usr/local/bin/micromamba' \
    | sha256sum -c -
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/bin:$PATH

# GPU architecture is the one thing the "build prefix == run prefix" property
# does NOT give us for free: the builder's card has nothing to do with the
# evaluation machine's, so the target has to be stated outright.
#
# 8.0 = A100, 9.0 = H100. No "+PTX": a PTX fallback would let an unsupported
# card JIT its way to a run that looks fine and reproduces different numerics,
# and for a benchmark that grades on reproducing a specific training failure,
# failing loudly on the wrong hardware is the better outcome. Widening this
# list is a deliberate act, and it costs build time linearly.
ENV TORCH_CUDA_ARCH_LIST="8.0;9.0"
ENV NVTE_CUDA_ARCHS="80;90"

# The fetch tooling (ptdbench_fetch.py, ptdbench_dataset.py) is deliberately
# NOT here, even though it is what every task's prepare.sh calls. It is runtime
# tooling: no build stage runs it, so putting it in `base` bought nothing and
# cost everything -- a one-character edit to it invalidated `base`, which
# invalidated all three env-* stages, which is hours of recompiling apex and
# transformer_engine to ship a different string. It lives at the end of `final`
# instead; see the block down there.
#
# build-env.sh is the opposite case and stays: the env-* stages are the only
# thing that runs it.
COPY build-env.sh /usr/local/libexec/ptdbench-build-env.sh
RUN set -eux; \
    sed -i 's/\r$//' /usr/local/libexec/ptdbench-build-env.sh; \
    chmod 0755 /usr/local/libexec/ptdbench-build-env.sh; \
    bash -n /usr/local/libexec/ptdbench-build-env.sh

# ===========================================================================
# env-<name> -- one stage per conda environment
# ===========================================================================
# Each is FROM base and builds only its own prefix, so BuildKit runs them in
# parallel and a change to one environment's requirements does not invalidate
# the others' cache.
#
# The stage names are load-bearing: docker/collect_envs.py asserts that every
# environment a task declares has a matching `AS env-<name>` here, so a task
# can never name an environment the image does not contain. It asserts the two
# prefetch COPY lines below for the same reason -- a stage missing them builds
# fine and quietly fetches everything over the network instead.
#
# There are deliberately no mounts of any kind on these stages -- not
# type=cache, not from=, not type=ssh. The two caches that used to be here
# cached nothing:
#
#   /root/.cache/pip    build-env.sh installs with --no-cache-dir on every
#                       pass, so this was never read and never written.
#   /opt/conda/pkgs     build-env.sh ends with `micromamba clean --all`, so
#                       each stage emptied the cache before the next could see
#                       it -- and the mount was a different filesystem than
#                       the prefix, so micromamba's hardlinks fell back to
#                       copies too.
#
# Both carried sharing=locked, which is an exclusive lock held for the whole
# RUN. All three stages named the same two mount IDs, so BuildKit ran them one
# after another: three multi-hour compiles serialised to buy two caches that
# were empty by construction. The "in parallel" above is what that cost.
#
# The host-prefetch directories used to arrive as `--mount=from=` on a shared
# named context, which had a subtler version of the same problem -- one mount of
# one shared context put its whole digest in all three cache keys. They are a
# plain COPY of a per-environment subdirectory now. A COPY is a layer, so each
# env-* image carries its own wheels and git objects; that costs disk in
# `ptdbench/stage:env-<name>` and nothing at all in `final`, which copies only
# /opt/conda/envs/<name> out. build-env.sh deletes both before it finishes
# anyway.
#
# So the caches that remain are the ones that were doing the work all along. A
# *download* cache belongs on the host, where it also survives
# `docker builder prune`: docker/wheels and docker/gitmirror are that. A *build*
# cache is ptdbench/stage:env-<name> plus PTDBENCH_PREBUILT_ENVS=1; see
# docker/build.sh. pip's own wheel cache is not an option even though it does
# hold compiled wheels (the upstream slime image's flash_attn comes out of
# exactly that, see fetch-cuda-wheels.sh): its key does not include
# TORCH_CUDA_ARCH_LIST or NVTE_CUDA_ARCHS, so after a change to either it would
# hand back a wheel built for the old architecture without saying so -- the
# silent-wrong-hardware outcome the arch list above exists to prevent.

FROM base AS env-verl
COPY envs/verl /opt/ptdbench/envs/verl
COPY wheels/verl    /opt/ptdbench/wheels/
COPY gitmirror/verl /opt/ptdbench/gitmirror/
# Re-declared per stage: ARG does not inherit across FROM. Empty default so a
# bare `docker build` is unchanged. Last, after everything it does not affect,
# so changing the proxy does not invalidate the environment tree or the
# prefetched directories.
ARG GITHUB_HTTP_PROXY=
RUN bash /usr/local/libexec/ptdbench-build-env.sh verl

FROM base AS env-slime-rc0
COPY envs/slime-rc0 /opt/ptdbench/envs/slime-rc0
COPY wheels/slime-rc0    /opt/ptdbench/wheels/
COPY gitmirror/slime-rc0 /opt/ptdbench/gitmirror/
ARG GITHUB_HTTP_PROXY=
RUN bash /usr/local/libexec/ptdbench-build-env.sh slime-rc0

# The second slime stack. Not a newer patch of slime-rc0 -- a different stack
# end to end (CUDA 13.0, torch 2.11, sglang 0.5.15, transformers 5.12,
# megatron-core 0.16), and slime's own API surface moved with it. The retool
# recipe imports sglang.srt.constants, which 0.5.0rc0 does not have; the rlve
# tasks' recorded outcomes are from 0.5.0rc0. Neither side can be moved, which
# is exactly what "give it its own environment name" is for. See
# docker/envs/README.md.
FROM base AS env-slime-v0.3.1
COPY envs/slime-v0.3.1 /opt/ptdbench/envs/slime-v0.3.1
COPY wheels/slime-v0.3.1    /opt/ptdbench/wheels/
COPY gitmirror/slime-v0.3.1 /opt/ptdbench/gitmirror/
ARG GITHUB_HTTP_PROXY=
RUN bash /usr/local/libexec/ptdbench-build-env.sh slime-v0.3.1

# ===========================================================================
# final -- the environments, side by side, at their own paths
# ===========================================================================
FROM base AS final

# Source path == destination path, for every environment. This is the line the
# whole design rests on; if it ever has to change, the previous version's
# failure mode comes back.
COPY --from=env-verl         /opt/conda/envs/verl         /opt/conda/envs/verl
COPY --from=env-slime-rc0    /opt/conda/envs/slime-rc0    /opt/conda/envs/slime-rc0
COPY --from=env-slime-v0.3.1 /opt/conda/envs/slime-v0.3.1 /opt/conda/envs/slime-v0.3.1

# The recorded closure of each environment, for the G1 diff against upstream.
COPY --from=env-verl         /opt/ptdbench/freeze/verl.txt         /opt/ptdbench/freeze/verl.txt
COPY --from=env-slime-rc0    /opt/ptdbench/freeze/slime-rc0.txt    /opt/ptdbench/freeze/slime-rc0.txt
COPY --from=env-slime-v0.3.1 /opt/ptdbench/freeze/slime-v0.3.1.txt /opt/ptdbench/freeze/slime-v0.3.1.txt

# The environment definitions travel with the image so that a container can be
# checked against them without the repository at hand.
COPY envs /opt/ptdbench/envs

# Where the prepare stage leaves its results, and the two download-cache roots
# it fetches into. Bind-mounted per run (writable in prepare, read-only in the
# agent and verifier stages); on the harbor path they are build-time only and
# get removed again. Created here so both are a mount point rather than
# something docker invents on first use. See ptdbench-activate.sh.
RUN mkdir -p /ptdbench/prepared /ptdbench/models /ptdbench/datasets

COPY ptdbench-activate.sh /etc/profile.d/ptdbench-activate.sh
RUN set -eux; \
    sed -i 's/\r$//' /etc/profile.d/ptdbench-activate.sh; \
    chmod 0644 /etc/profile.d/ptdbench-activate.sh; \
    sh -n /etc/profile.d/ptdbench-activate.sh

# Build-time smoke: import-only, and only what does not need a GPU driver.
# A stack that cannot import its own framework fails here rather than halfway
# through an evaluation. The GPU half -- torch.cuda.is_available(), an actual
# attention kernel, an actual TE layer -- is docker/smoke.sh, which runs the
# built image with --gpus and is the real G2 gate.
#
# flash_attn and transformer_engine are in this list because loading their
# extensions is exactly where a wheel built against the wrong torch shows up,
# and it shows up as an undefined symbol, with no GPU involved. A prebuilt
# flash_attn that predated torch 2.8's c10::cuda::SetDevice signature change
# reached a finished image once; nothing before this line would have caught it.
# Both import without a driver -- verified against the upstream images -- so
# this stays honest about being the driver-free half.
#
# Kept to one statement per `;` on purpose: `python3 -c` takes a single logical
# line, so a compound `if` block cannot follow other statements there -- it is a
# SyntaxError, not a skipped check. Anything conditional belongs in the shell
# loop below instead.
RUN set -eux; \
    for name in verl slime-rc0 slime-v0.3.1; do \
        echo "=== smoke: ${name} ==="; \
        PTDBENCH_STAGE=agent PTDBENCH_CONDA_ENV="${name}" sh -lc '\
            python3 -c "\
import sys, torch, flash_attn, transformer_engine.pytorch; \
import importlib.metadata as md; \
print(\" python\", sys.version.split()[0], sys.executable); \
print(\" torch \", torch.__version__, torch.__file__); \
print(\" flash_attn\", md.version(\"flash_attn\")); \
print(\" transformer_engine\", md.version(\"transformer_engine\")); \
"'; \
    done

# megatron.core belongs to the slime stacks only. It arrives there as the
# editable Megatron-LM tree in docker/envs/slime-*/sources.txt, and this is also
# the one check that exercises the activate hook's /root/Megatron-LM symlink.
# verl's environment has no megatron at all, so asking for it there would fail
# on a package that stack is not supposed to contain.
#
# fused_weight_gradient_mlp_cuda is apex's, and it is checked here rather than
# left to the GPU smoke because its absence is not a degraded mode: megatron's
# ColumnParallelLinear defaults to gradient_accumulation_fusion=True and raises
# at construction. An environment that built apex as pure Python (no
# --cpp_ext --cuda_ext) looks completely fine until the first layer is built --
# which, for slime_retool, is inside prepare.sh. Importing the extension needs
# no driver; constructing the layer does, so that half is docker/smoke.sh.
RUN set -eux; \
    for name in slime-rc0 slime-v0.3.1; do \
        echo "=== smoke: ${name} megatron + apex ==="; \
        PTDBENCH_STAGE=agent PTDBENCH_CONDA_ENV="${name}" sh -lc '\
            python3 -c "\
import megatron.core as m, fused_weight_gradient_mlp_cuda as f; \
print(\" megatron\", m.__file__); \
print(\" apex    \", f.__file__); \
"'; \
    done

ENV CUDA_DEVICE_ORDER=PCI_BUS_ID \
    TOKENIZERS_PARALLELISM=false

# ---------------------------------------------------------------------------
# The fetch tooling -- last on purpose
# ---------------------------------------------------------------------------
# Every task's prepare.sh calls these. They used to sit in `base`, which meant
# editing one of them invalidated `base` and with it all three env-* stages:
# hours of recompiling apex and transformer_engine to change a cache path. They
# are runtime tooling -- no build stage runs them -- so nothing above needs them
# and they belong after the expensive layers. Editing them now rebuilds these
# four layers and nothing else.
#
# Keep this block at the end. Anything inserted below it pays the same tax the
# move was made to remove.
#
# Note there is no python3 on the default PATH: the root conda prefix has no
# interpreter, on purpose. Each environment brings its own, so these shims
# resolve `python3` at call time and run under whichever environment is
# active -- which is also what gives them that environment's huggingface_hub.

# Weights and datasets at a pinned revision. Cache-first, sha256-verified,
# hard-fails on mismatch rather than re-downloading over a bad copy.
COPY ptdbench_fetch.py /usr/local/libexec/ptdbench_fetch.py
RUN set -eux; \
    for kind in hf url; do \
        printf '#!/bin/sh\nexec python3 /usr/local/libexec/ptdbench_fetch.py %s "$@"\n' \
            "${kind}" > "/usr/local/bin/ptdbench_fetch_${kind}"; \
        chmod 0755 "/usr/local/bin/ptdbench_fetch_${kind}"; \
    done

# Datasets that are *derived* rather than downloaded. verl's RLHFDataset does
# not accept upstream gsm8k or DAPO-Math-17k as published, so the fixed thing
# for those tasks is a transformation, not a revision. Same cache, same
# hard-fail-on-mismatch rule; see ptdbench_dataset.py.
#
# The recipe package ships whole because 011/022/023 must get byte-identical
# output -- one copy, not one per task.
COPY ptdbench_dataset.py /usr/local/libexec/ptdbench_dataset.py
COPY ptdbench_datasets /usr/local/libexec/ptdbench_datasets
RUN set -eux; \
    printf '#!/bin/sh\nexec python3 /usr/local/libexec/ptdbench_dataset.py "$@"\n' \
        > /usr/local/bin/ptdbench_dataset; \
    chmod 0755 /usr/local/bin/ptdbench_dataset

# The harness bind-mounts a per-run copy of the task's workspace here.
WORKDIR /workspace
