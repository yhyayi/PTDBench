#!/usr/bin/env bash
# Build the unified PTDBench evaluation image.
#
# One image, one tag. Every conda environment the task set needs is inside it;
# there is no per-framework tag to keep in sync, and an evaluation host pulls
# one thing.
#
# The build is long (hours) and disk-hungry (tens of GB): it compiles apex,
# transformer_engine and friends from source, once per environment. BuildKit
# runs the env-* stages in parallel, and `--target env-<name>` builds just one,
# which is how to iterate on a single environment without paying for the rest.
#
#   ./docker/build.sh                      # everything
#   ./docker/build.sh --target env-slime-rc0   # one environment, for debugging
#
# Flags are passed through to `docker build`, so --no-cache, --progress=plain
# and --build-arg all work as usual.
#
# ---------------------------------------------------------------------------
# Building the environments separately, then assembling
# ---------------------------------------------------------------------------
# Only `final` is the deliverable, so only `final` gets $PTDBENCH_IMAGE_TAG.
# A `--target env-<name>` build is an intermediate and is tagged
# ptdbench/stage:env-<name> instead. Two reasons that matters:
#
#   1. Tagging both with the same name means the second `--target` build
#      silently steals the tag from the first, leaving the first dangling as
#      <none> -- and leaves a tag named "the evaluation image" pointing at
#      something that is not one.
#   2. A tagged image survives `docker builder prune`. Hours of compiling that
#      exist only as a build-cache record do not.
#
# With the environments tagged, PTDBENCH_PREBUILT_ENVS=1 assembles the final
# image from them instead of rebuilding the stages:
#
#   ./docker/build.sh --target env-verl         # hours
#   ./docker/build.sh --target env-slime-rc0    # hours
#   ./docker/build.sh --target env-slime-v0.3.1 # hours
#   PTDBENCH_PREBUILT_ENVS=1 ./docker/build.sh   # minutes
#
# This is not just a shortcut. The env stages' cache keys include things that
# do not survive between `docker build` invocations -- the content of that
# environment's prefetch directories, the --network mode, the GITHUB_HTTP_PROXY
# build arg, and every layer of `base` above them. Any one of those differing
# sends both environments back to a full recompile even though the finished
# images are sitting right there. Reading them as images sidesteps the cache
# entirely.
#
# The trade is that the environments are then whatever those images contain,
# not what the current requirements.txt says. Rebuild the stage after changing
# an environment's closure; ./docker/smoke.sh is what checks the result.
#
# ---------------------------------------------------------------------------
# The two host prefetch directories
# ---------------------------------------------------------------------------
# Neither is fetched during the image build, because a cluster builder
# typically cannot reach the hosts involved:
#
#   ./docker/fetch-cuda-wheels.sh   docker/wheels/<env>     torch, flash_attn
#   ./docker/fetch-git-mirrors.sh   docker/gitmirror/<env>  every git+ pin
#
# Both live in the build context and each env-* stage COPYs its own share, so
# this wrapper has nothing to inject; it only reports what it found. Both are
# optional and both fall back to the network: missing torch wheels go to the
# PyTorch index, a missing flash_attn compiles, an absent mirror is cloned
# in-stage. The fallbacks are slow and repeat on every rebuild, which is what
# the directories exist to avoid.
#
# One subdirectory per environment rather than one shared directory: a COPY
# hashes the subtree it names into that stage's cache key, so sharing meant a
# new slime-v0.3.1 wheel recompiled verl as well.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Must match ptdbench/core/types.py's DEFAULT_IMAGE and docker/smoke.sh.
image_tag="${PTDBENCH_IMAGE_TAG:-ptdbench/env:latest}"
# Where intermediate stage builds are tagged. Anything that is not the final
# image is parked here rather than wearing the deliverable's name.
stage_repo="${PTDBENCH_STAGE_REPO:-ptdbench/stage}"

# Which stage is being built, read out of the caller's own flags. Both spellings
# docker accepts are handled; an unrecognised one means "final", which is what a
# bare `docker build` produces.
target=""
_expect_target=0
for arg in "$@"; do
    if ((_expect_target)); then
        target="$arg"
        _expect_target=0
    elif [[ "$arg" == "--target" ]]; then
        _expect_target=1
    elif [[ "$arg" == --target=* ]]; then
        target="${arg#--target=}"
    fi
done

if [[ -n "$target" && "$target" != "final" ]]; then
    image_tag="${stage_repo}:${target}"
fi

# Every environment the image is supposed to contain, read from the registry so
# this file does not need editing when one is added. Same source as smoke.sh.
mapfile -t ENVS < <(find docker/envs -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)

# Cheap, docker-free, milliseconds: every task that names a conda environment
# must ship the same closure for it, every closure must be fully pinned, and
# every environment must have a stage in the Dockerfile. A registry that has
# drifted fails here rather than after two hours of compiling.
python3 docker/collect_envs.py --check

# Syntax-check the scripts that only ever run inside a build stage. A typo in
# build-env.sh otherwise surfaces an hour in, after the conda solve.
bash -n docker/build-env.sh
sh -n docker/ptdbench-activate.sh
python3 -c "import ast, sys; [ast.parse(open(p).read(), p) for p in sys.argv[1:]]" \
    docker/ptdbench_fetch.py docker/ptdbench_dataset.py docker/ptdbench_datasets/*.py

# The prefetch directories, one per environment. Created when absent because
# each env-* stage COPYs its own, and COPY of a directory that is not there is a
# hard error; an empty one is exactly the "fetch it yourself" fallback. A fresh
# clone has them already -- each carries a committed .gitkeep -- so this only
# matters after someone deletes one, or adds an environment.
wheel_total=0
mirror_total=0
for name in "${ENVS[@]}"; do
    wheels_dir="$repo_root/docker/wheels/${name}"
    mirror_dir="$repo_root/docker/gitmirror/${name}"
    mkdir -p "$wheels_dir" "$mirror_dir"

    shopt -s nullglob
    env_wheels=("$wheels_dir"/*.whl)
    shopt -u nullglob
    wheel_total=$((wheel_total + ${#env_wheels[@]}))

    env_mirrors=0
    if [[ -f "${mirror_dir}/manifest.txt" ]]; then
        env_mirrors="$(awk 'NF' "${mirror_dir}/manifest.txt" | wc -l)"
    fi
    mirror_total=$((mirror_total + env_mirrors))

    echo "prefetched ${name}: ${#env_wheels[@]} wheel(s), ${env_mirrors} git mirror(s)"
done
if ((wheel_total == 0)); then
    echo "  no wheels: CUDA pins fall back to download.pytorch.org and" >&2
    echo "  flash_attn to compiling the PyPI sdist -- ./docker/fetch-cuda-wheels.sh" >&2
fi
if ((mirror_total == 0)); then
    echo "  no git mirrors: every git+ pin is cloned inside the build, on every" >&2
    echo "  rebuild -- ./docker/fetch-git-mirrors.sh" >&2
fi

# A named build context whose name matches a stage name replaces that stage, so
# `COPY --from=env-slime-rc0` reads the image instead of building the stage.
prebuilt_args=()
if [[ "${PTDBENCH_PREBUILT_ENVS:-0}" != "0" ]]; then
    if [[ -n "$target" && "$target" != "final" ]]; then
        echo "PTDBENCH_PREBUILT_ENVS has nothing to do for --target $target:" >&2
        echo "  it replaces the env-* stages, which only 'final' consumes." >&2
        exit 2
    fi
    missing=()
    for name in "${ENVS[@]}"; do
        stage_image="${stage_repo}:env-${name}"
        if docker image inspect "$stage_image" >/dev/null 2>&1; then
            prebuilt_args+=(--build-context "env-${name}=docker-image://${stage_image}")
        else
            missing+=("$stage_image")
        fi
    done
    if ((${#missing[@]})); then
        # Partial reuse would be worse than refusing: half the environments
        # would come from images and half would silently start a recompile,
        # which is exactly the surprise this flag exists to remove.
        echo "PTDBENCH_PREBUILT_ENVS=1 but these stage images do not exist:" >&2
        printf '  %s\n' "${missing[@]}" >&2
        echo "build them first, e.g.:" >&2
        for image in "${missing[@]}"; do
            echo "  ./docker/build.sh --target ${image##*:}" >&2
        done
        exit 2
    fi
    echo "prebuilt envs: ${ENVS[*]} (read from ${stage_repo}:env-*, stages not rebuilt)"
fi

# GITHUB_GIT_SSH moved to the host, and this is only here to say so. It used
# to forward an agent into the env-* RUN stages with --mount=type=ssh; the
# rewrite now happens in fetch-git-mirrors.sh, where ssh has the user's own
# agent, no BuildKit forwarding, and no conda openssl to collide with. The
# build then reads mirrors off disk and never talks to github at all, which is
# a better answer to a filtered :443 than tunnelling through :22 was.
if [[ "${GITHUB_GIT_SSH:-0}" != "0" ]]; then
    echo "GITHUB_GIT_SSH is a flag for the prefetch step now, not for the build:" >&2
    echo "  GITHUB_GIT_SSH=1 ./docker/fetch-git-mirrors.sh" >&2
    echo "then build as usual. The mirrors still record git+https URLs, so" >&2
    echo "nothing about the resulting image depends on how they were fetched." >&2
    exit 2
fi

echo "building $image_tag"
DOCKER_BUILDKIT=1 docker build \
    "${prebuilt_args[@]+"${prebuilt_args[@]}"}" \
    -f docker/Dockerfile -t "$image_tag" "$@" docker/

echo
echo "built $image_tag"
docker image inspect "$image_tag" --format 'size: {{.Size}} bytes'
echo
if [[ -n "$target" && "$target" != "final" ]]; then
    echo "this is an intermediate stage, not the evaluation image."
    echo "next: ./docker/build.sh --target env-<other>   (the remaining environments)"
    echo "      PTDBENCH_PREBUILT_ENVS=1 ./docker/build.sh   (assemble, minutes)"
else
    echo "next: ./docker/smoke.sh   (needs a GPU; the build only checks imports)"
    echo "      ./docker/warm-cache.sh   (needs network; fills models/ and datasets/)"
fi
