#!/usr/bin/env bash
# Build one conda environment at /opt/conda/envs/<name>. Run by the env-<name>
# stages in docker/Dockerfile, one stage per environment.
#
# The load-bearing property of this whole design is that the prefix this script
# installs into is byte-for-byte the prefix the packages will later run from.
# The final stage copies /opt/conda/envs/<name> to /opt/conda/envs/<name> --
# same path, no relocation -- so the shebangs, .pth finders, RPATHs, CUDA_HOME
# and conda-meta prefixes written here all stay true. Nothing in this script may
# assume otherwise: no staging directory, no `--prefix` that differs from the
# final location, no post-build `mv`.
#
#   $1  environment name; must match a directory under docker/envs/
set -euo pipefail

env_name="${1:?usage: build-env.sh <env-name>}"
env_dir="/opt/ptdbench/envs/${env_name}"
prefix="/opt/conda/envs/${env_name}"

[ -d "$env_dir" ] || { echo "no such environment: $env_dir" >&2; exit 1; }

# Optional GitHub-only git proxy. Cluster builders often cannot reach
# github.com, while PyPI/conda mirrors work unproxied. Do not set HTTPS_PROXY:
# that would send the Tsinghua index through the tunnel as well.
# Dockerfile ARG GITHUB_HTTP_PROXY; build with --network=host so 127.0.0.1 is
# the host tunnel. pip's git+https and git submodule both honor this.
#
# This only matters for repositories docker/fetch-git-mirrors.sh did not
# already mirror -- see the git mirrors section below. With the mirrors in
# place the build makes no git network calls at all.
if [ -n "${GITHUB_HTTP_PROXY:-}" ]; then
    echo "=== [${env_name}] git proxy for GitHub -> ${GITHUB_HTTP_PROXY}"
    for host in github.com codeload.github.com gist.github.com \
                raw.githubusercontent.com objects.githubusercontent.com; do
        git config --global "http.https://${host}.proxy" "$GITHUB_HTTP_PROXY"
    done
fi

# GITHUB_GIT_SSH is deliberately not handled here any more. It is a host-side
# flag now, read by docker/fetch-git-mirrors.sh, and the reason is that getting
# an ssh agent into this RUN required --mount=type=ssh. On the host it requires
# nothing, and it also stops needing the workaround this block used to carry:
# conda-forge openssl is 3.6, Ubuntu 22.04's /usr/bin/ssh is built against 3.0,
# and the mix dies with "OpenSSL version mismatch. Built against 30000020, you
# have 30600040" the moment the prefix lands on PATH. There is no conda prefix
# on the host, so there is nothing to work around.
#
#   GITHUB_GIT_SSH=1 ./docker/fetch-git-mirrors.sh   # then build as usual

echo "=== [${env_name}] conda layer"
# The interpreter, nvcc and the system build libs. Everything else is pip's.
micromamba create -y -f "${env_dir}/conda.yaml"
[ -d "$prefix" ] || { echo "micromamba did not create ${prefix}" >&2; exit 1; }

export PATH="${prefix}/bin:${PATH}"
export CONDA_PREFIX="$prefix"
# nvcc came from the conda layer, so CUDA_HOME is the prefix itself. Every
# compiled package below records this path; it survives the COPY intact.
export CUDA_HOME="$prefix"
export CUDA_PATH="$prefix"
export LD_LIBRARY_PATH="${prefix}/lib:${LD_LIBRARY_PATH:-}"
# conda-forge CUDA does not put headers in $PREFIX/include. They live under
# targets/<triple>/include, and the usual `conda activate` hook exports CPATH
# to match. This script never activates -- it only prepends PATH -- so without
# this, transformer-engine-torch's host C++ compile dies on
# `#include <cuda_runtime_api.h>` even though nvcc is on PATH.
# Official TE 2.12 ships no torch-2.10 wheel (PyPI is sdist-only), so this
# compile path is the one verl actually takes.
cuda_target="${prefix}/targets/x86_64-linux"
if [ -d "${cuda_target}/include" ]; then
    export CPATH="${cuda_target}/include${CPATH:+:$CPATH}"
    export CPLUS_INCLUDE_PATH="${cuda_target}/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
    export LIBRARY_PATH="${cuda_target}/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
    export LD_LIBRARY_PATH="${cuda_target}/lib:${LD_LIBRARY_PATH}"
    export CXXFLAGS="${CXXFLAGS:-} -I${cuda_target}/include"
    export CFLAGS="${CFLAGS:-} -I${cuda_target}/include"
fi

# Cluster builders typically cannot reach pypi.org or files.pythonhosted.org.
# Tsinghua is the default index. Do not pass PyTorch's --extra-index-url as a
# global extra index: that repository also lists regular PyPI projects with
# absolute files.pythonhosted.org links, and pip will follow them. CUDA wheels
# (+cuXXX) are installed in a separate --no-deps pass instead.
# Override with PIP_INDEX_URL if needed.
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"

# Build-time-only knobs for the source builds further down.
#   TORCH_CUDA_ARCH_LIST / NVTE_CUDA_ARCHS -- see the note in docker/Dockerfile.
#     Unlike paths, GPU architecture is *not* inherited from the build host, so
#     it has to be stated.
#   MAX_JOBS -- transformer-engine-torch and apex will otherwise spawn one compiler
#     per core and exhaust memory on a big builder.
export NVTE_FRAMEWORK=pytorch
export NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS:-80;90}"
# transformer-engine-torch's sdist first tries to fetch a GitHub Releases
# wheel. Official v2.12 has no torch-2.10 build (verl), and GitHub is
# unreachable from a cluster builder anyway (we do not set HTTPS_PROXY).
# Force the local compile against the pip-installed transformer-engine-cu12
# (or -cu13) wheel instead of hanging on that fetch.
export NVTE_PYTORCH_FORCE_BUILD=TRUE
export MAX_JOBS="${MAX_JOBS:-8}"

pip_install() {
    pip install --no-cache-dir --disable-pip-version-check "$@"
}

requirements="${env_dir}/requirements.txt"

# Package names that get their own install pass, so they can be excluded from
# the bulk one. Comments and blank lines are dropped; the first field is the
# name.
special_names() {
    sed -e 's/#.*//' "${env_dir}/build-flags.txt" | awk 'NF {print $1}'
}

# Resolve a package name to its pinned line in requirements.txt. The pin lives
# in exactly one place and is read from there, so build-flags.txt and
# build-first.txt never carry a version that could drift.
pin_for() {
    local name="${1//_/-}"
    sed -e 's/#.*//' "$requirements" \
      | awk -v want="$name" 'NF && $0 !~ /^-/ {
            line = $0
            split(line, parts, /[ @=<>!~[;]/)
            key = tolower(parts[1]); gsub(/_/, "-", key)
            if (key == tolower(want)) { print line; found = 1; exit }
        }
        END { if (!found) exit 1 }'
}

pytorch_index="$(awk '/^--extra-index-url[[:space:]]/ { print $2; exit }' "$requirements")"
mapfile -t cuda_pins < <(sed -e 's/#.*//' "$requirements" | awk 'NF && $0 ~ /\+cu[0-9]/ { print }')
# Host-prefetched wheels (docker/fetch-cuda-wheels.sh). The Dockerfile COPYs
# this environment's share of them -- docker/wheels/<env> -- to here. Missing
# +cuXXX files fall back to pytorch_index; missing flash_attn falls through to
# the bulk sdist compile.
wheels_dir="${PTDBENCH_WHEELS_DIR:-/opt/ptdbench/wheels}"

# This environment's torch, as major.minor -- e.g. 2.11 from 2.11.0+cu130.
# Used to reject a prebuilt wheel that was compiled against a different torch;
# see wheel_for_pin. Empty if this environment has no torch at all -- the `||`
# is load-bearing: pin_for exits 1 on a miss, and under `set -euo pipefail` a
# bare assignment would abort the build here with no message at all.
env_torch_mm="$(pin_for torch 2>/dev/null | sed -nE 's/^torch==([0-9]+\.[0-9]+)\..*/\1/p')" \
    || env_torch_mm=""

# This environment's CUDA major, from the same pin -- 13 from 2.11.0+cu130, 12
# from 2.10.0+cu129. The trailing digit of cuNNN is the minor and is dropped on
# purpose: cu128 binaries load fine on a 12.9 prefix (verl's flash_attn wheel
# depends on exactly that), while libcudart.so.12 and libcudart.so.13 are
# different sonames and nothing bridges them. Empty when there is no torch or
# the pin carries no +cuNNN -- same `||` reasoning as above, and empty is what
# disables the check.
env_cuda_major="$(pin_for torch 2>/dev/null | sed -nE 's/^torch==[^+]*\+cu([0-9]+)[0-9]$/\1/p')" \
    || env_cuda_major=""

# A pin like torch==2.10.0+cu129 matches torch-2.10.0+cu129-*.whl.
# flash_attn==2.8.3 matches flash_attn-2.8.3+cu128torch2.10-*.whl: the CUDA
# tag is in the wheel filename, not the pin. Try both _ and -; pip treats
# them as the same distribution name.
#
# The filename tags are checked even though each environment now has its own
# wheels directory, and the two guards are not redundant. The directories keep
# verl's flash_attn out of slime-v0.3.1's reach; the tags are what catches a
# file put in the wrong directory by hand or by an edit to fetch-cuda-wheels.sh.
# verl and slime-v0.3.1 both pin flash_attn==2.8.3, one for torch 2.10 and one
# for 2.11, so the version alone does not tell those two files apart at all.
# A wheel whose filename says `torchX.Y` is therefore only a candidate when X.Y
# is this environment's torch: loading an extension built against another one is
# the undefined-symbol failure this directory exists to avoid, and it would
# surface as a segfault mid-evaluation, not here.
#
# The same applies to CUDA, and the pin says even less about that: a
# +cu128torch2.11 and a +cu130torch2.11 file both answer `flash_attn==2.8.3`,
# which has no CUDA in it at all. Installing the cu128 one into a CUDA 13
# prefix produces a package that builds, installs and freezes cleanly and then
# dies at `import flash_attn` with libcudart.so.12 not found -- the nvcc-vs-torch
# assertion below does not catch it, because torch itself is fine.
wheel_for_pin() {
    local pin="$1"
    local name="${pin%%==*}"
    local ver="${pin#*==}"
    local candidate n tag cu
    name="${name%%[*}"
    for n in "$name" "${name//_/-}" "${name//-/_}"; do
        for candidate in \
            "${wheels_dir}/${n}-${ver}-"*.whl \
            "${wheels_dir}/${n}-${ver}+"*.whl; do
            [ -f "$candidate" ] || continue
            tag="$(printf '%s' "${candidate##*/}" | sed -nE 's/.*torch([0-9]+\.[0-9]+).*/\1/p')"
            if [ -n "$tag" ] && [ -n "$env_torch_mm" ] && [ "$tag" != "$env_torch_mm" ]; then
                echo "--- skip ${candidate##*/}: built for torch ${tag}, this env is ${env_torch_mm}" >&2
                continue
            fi
            # Major only, for the reason given where env_cuda_major is set.
            cu="$(printf '%s' "${candidate##*/}" | sed -nE 's/.*\+cu([0-9]+).*/\1/p')"
            cu="${cu%?}"
            if [ -n "$cu" ] && [ -n "$env_cuda_major" ] && [ "$cu" != "$env_cuda_major" ]; then
                echo "--- skip ${candidate##*/}: built for CUDA ${cu}.x, this env is ${env_cuda_major}.x" >&2
                continue
            fi
            printf '%s' "$candidate"
            return 0
        done
    done
    return 1
}

echo "=== [${env_name}] build prerequisites"
# torch and the build backend must exist before anything is compiled against
# them; see docker/envs/*/build-first.txt.
first_pins=()
while read -r name; do
    [ -n "$name" ] || continue
    first_pins+=("$(pin_for "$name")")
done < <(sed -e 's/#.*//' "${env_dir}/build-first.txt" | awk 'NF {print $1}')
other_first=()
for pin in "${first_pins[@]}"; do
    case "$pin" in
        *+cu*) ;;
        *) other_first+=("$pin") ;;
    esac
done
pip_install "${other_first[@]}"
if ((${#cuda_pins[@]})); then
    local_cuda_pins=()
    remote_cuda_pins=()
    for pin in "${cuda_pins[@]}"; do
        if wheel="$(wheel_for_pin "$pin")"; then
            echo "--- local ${pin} <- ${wheel##*/}"
            local_cuda_pins+=("$pin")
        else
            echo "--- missing ${pin}, will fetch from ${pytorch_index:-<no index>}"
            remote_cuda_pins+=("$pin")
        fi
    done
    if ((${#local_cuda_pins[@]})); then
        echo "=== [${env_name}] CUDA wheels from ${wheels_dir}"
        pip_install --no-index --find-links "$wheels_dir" --no-deps "${local_cuda_pins[@]}"
    fi
    if ((${#remote_cuda_pins[@]})); then
        [ -n "$pytorch_index" ] || {
            echo "CUDA wheels need --extra-index-url in ${requirements}" >&2
            exit 1
        }
        echo "=== [${env_name}] CUDA wheels from ${pytorch_index}"
        pip_install --index-url "$pytorch_index" --no-deps "${remote_cuda_pins[@]}"
    fi
fi

# Same directory, same lookup, for pins that are not +cuXXX -- flash_attn
# is the live case. PyPI only has an sdist, so a hit here is what stops the
# bulk pass from compiling it. A miss falls through to bulk unchanged.
prebuilt_pins=()
while read -r pin; do
    [ -n "$pin" ] || continue
    case "$pin" in
        --*) continue ;;
        *+cu*) continue ;;
    esac
    if wheel="$(wheel_for_pin "$pin")"; then
        echo "--- local ${pin} <- ${wheel##*/}"
        prebuilt_pins+=("$pin")
    fi
done < <(sed -e 's/#.*//' "$requirements" | awk 'NF')
if ((${#prebuilt_pins[@]})); then
    echo "=== [${env_name}] prebuilt wheels from ${wheels_dir}"
    pip_install --no-index --find-links "$wheels_dir" --no-deps "${prebuilt_pins[@]}"
fi

# Everything installed above went in through a --no-deps pass, and every pass
# below resolves dependencies. None of them carries the pins: the bulk file has
# the +cuXXX lines stripped along with the --extra-index-url that made them
# installable (see the sed further down), so `torch` survives only as a
# transitive requirement and any package asking for it may pull the plain PyPI
# build -- which is compiled against a different CUDA than the nvcc in this
# prefix. apex is where that surfaces, seven minutes into its compile:
#   RuntimeError: Cuda extensions are being compiled with a version of Cuda
#   that does not match the version used to compile Pytorch binaries.
#
# A constraints file hands those pins back to the later passes without handing
# back the install directive. pip applies constraints to transitive
# requirements too, and fetches nothing unless it was going to change the
# package anyway -- so no index lookup for a +cuXXX version the Tsinghua mirror
# does not have, and a package that genuinely demands a different torch fails
# at that pass with the conflict named, rather than silently winning.
#
# The recorded version is what is actually on disk, not the pin it was asked
# for. That difference is the whole point for flash_attn: its wheel carries the
# local tag `+cu130torch2.11`, and a constraint of `flash_attn==2.8.3` -- which
# is what requirements.txt says -- would still let a PyPI sdist build take its
# place, since both satisfy `==2.8.3`.
pinned_names=()
for pin in "${cuda_pins[@]+"${cuda_pins[@]}"}" \
           "${prebuilt_pins[@]+"${prebuilt_pins[@]}"}"; do
    name="${pin%%==*}"
    pinned_names+=("${name%%[*}")
done
if ((${#pinned_names[@]})); then
    constraints="/tmp/${env_name}-constraints.txt"
    python - "$constraints" "${pinned_names[@]}" <<'PY'
import sys
from importlib.metadata import PackageNotFoundError, version

out, names = sys.argv[1], sys.argv[2:]
with open(out, "w") as fh:
    for name in names:
        try:
            fh.write(f"{name}=={version(name)}\n")
        except PackageNotFoundError:
            sys.exit(f"{name} was installed by a --no-deps pass but is not on disk")
PY
    export PIP_CONSTRAINT="$constraints"
    echo "=== [${env_name}] constraints for the resolving passes"
    sed -e 's/^/--- /' "$constraints"
fi

# The constraints only reach passes that go through pip's resolver. This is the
# backstop, and it is cheap: assert that the torch on disk and the nvcc in this
# prefix agree about CUDA, at points where the answer still costs seconds to act
# on instead of a failed apex compile. Skipped for an environment with no torch.
assert_torch_cuda() {               # <where>
    python - "$1" <<'PY'
import re, subprocess, sys

import torch

where = sys.argv[1]
nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout
found = re.search(r"release (\d+\.\d+)", nvcc)
if found and torch.version.cuda != found.group(1):
    sys.exit(
        f"[{where}] torch {torch.__version__} is built for CUDA "
        f"{torch.version.cuda}, but nvcc in this prefix is {found.group(1)}. "
        "The pinned wheel was replaced -- search the log above for "
        "'Downloading torch-' to see which pass did it."
    )
PY
}

# pip's nvidia-* wheels (cudnn, cublas, ...) keep headers next to the .so,
# not under CUDA_HOME. The conda CPATH above is only cudart. TE includes
# ATen/cudnn/cudnn-wrapper.h which does `#include <cudnn.h>` -- that file is
# in site-packages/nvidia/cudnn/include, installed with nvidia-cudnn-cu12
# (or -cu13; the path is the same either way) in the build-first pass.
echo "=== [${env_name}] pip NVIDIA headers"
site_packages="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
shopt -s nullglob
for d in "${site_packages}/nvidia"/*/include "${site_packages}/nvidia"/*/*/include; do
    [ -d "$d" ] || continue
    echo "--- include ${d#"$site_packages/"}"
    export CPATH="${d}${CPATH:+:$CPATH}"
    export CPLUS_INCLUDE_PATH="${d}${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
    export CXXFLAGS="${CXXFLAGS:-} -I${d}"
    export CFLAGS="${CFLAGS:-} -I${d}"
done
for d in "${site_packages}/nvidia"/*/lib "${site_packages}/nvidia"/*/lib64; do
    [ -d "$d" ] || continue
    export LIBRARY_PATH="${d}${LIBRARY_PATH:+:$LIBRARY_PATH}"
    export LD_LIBRARY_PATH="${d}:${LD_LIBRARY_PATH:-}"
done
shopt -u nullglob
if [ -d "${site_packages}/nvidia/cudnn/include" ]; then
    export CUDNN_HOME="${site_packages}/nvidia/cudnn"
    # CPATH and CXXFLAGS above are compiler variables; cmake's find_path and
    # find_library do not read either. transformer_engine's cmake includes
    # 3rdparty/cudnn-frontend/cmake/cuDNN.cmake, which looks for cudnn.h and
    # libcudnn.so.9 through CUDNN_PATH / CUDNN_INCLUDE_PATH / CUDNN_LIBRARY_PATH
    # (and a $CUDA_HOME that has neither), so the same directory has to be
    # handed over a second time, in cmake's own vocabulary. Upstream never hit
    # this: its images carry a system cudnn that find_path locates unaided.
    export CUDNN_PATH="${site_packages}/nvidia/cudnn"
    export CUDNN_INCLUDE_PATH="${site_packages}/nvidia/cudnn/include"
    export CUDNN_LIBRARY_PATH="${site_packages}/nvidia/cudnn/lib"
fi

git_retry() {
    # Six attempts, not two or three: a github connect failure here costs 133
    # seconds (the kernel's, not git's -- there is no connect timeout knob), and
    # a run of two in a row was the observed normal, not the bad case.
    local attempt=1
    while :; do
        if "$@"; then return 0; fi
        if [ "$attempt" -ge 6 ]; then
            echo "--- gave up after ${attempt} attempts: $*" >&2
            return 1
        fi
        echo "--- attempt ${attempt} failed, retrying: $*" >&2
        sleep $((attempt * 10))
        attempt=$((attempt + 1))
    done
}

# A retried clone has to start from nothing each time -- git refuses to clone
# into the half-populated directory the failed attempt left behind.
git_clone_fresh() {                 # <url> <dir>
    rm -rf "$2"
    git clone --filter=blob:none "$1" "$2"
}

# Local patches for an editable source tree, from
# docker/envs/<env>/patches/<dir-name>/*.patch, applied in sorted order.
#
# Plain apply first, --3way only as the fallback. A patch generated against
# the pinned commit applies exactly, and the plain path needs nothing but the
# working tree. --3way is the useful failure mode when a pin moves: instead of
# git's context-line complaint it attempts a real merge and names the file it
# could not reconcile. It needs the pre-image blob in the object database --
# available here because `checkout --detach` above materialised it, but this
# is a --filter=blob:none clone, so do not count on it for a file the checkout
# did not touch.
#
# No patches directory is the normal case and is silently a no-op. An empty
# one is not: `patches/<dir>` existing with nothing in it means someone
# removed a patch without removing the directory, and a glob quietly expanding
# to nothing would hide that.
apply_source_patches() {            # <dir-name> <target>
    local dir="$1" target="$2" patch_dir="${env_dir}/patches/$1" p found=0
    [ -d "$patch_dir" ] || return 0
    for p in "$patch_dir"/*.patch; do
        [ -e "$p" ] || continue
        found=1
        echo "--- patch ${dir} <- ${p##*/}"
        # The `||` makes this compound command succeed as a whole, which is
        # exactly what suppresses `set -e` -- so the failure has to be raised
        # by hand. Without the explicit exit a patch that stopped applying
        # would print git's error and then build an unpatched tree anyway.
        git -C "$target" apply --verbose "$p" \
            || git -C "$target" apply --3way --verbose "$p" \
            || { echo "failed to apply ${p} to ${target}" >&2; return 1; }
    done
    [ "$found" -eq 1 ] || { echo "empty patch directory: ${patch_dir}" >&2; return 1; }
    # Leaves the tree dirty on purpose. `git -C src/<dir> diff` inside a built
    # image is then an honest answer to "how does this differ from the pin".
    git -C "$target" --no-pager diff --stat
}

# A full bare clone, and served to pip as a plain path rather than a file://
# URL. Both halves of that are load-bearing, and both were learned the hard
# way:
#
#   full, not --depth 1   pip does not simply clone and check out. If anything
#                         goes wrong it falls back to `git fetch origin <sha>`,
#                         and a fetch against a shallow source is refused --
#                         "shallow roots are not allowed to be updated", once
#                         per remote ref, thirteen hundred times, then failure.
#   path, not file://     a file:// URL makes git fork a git-upload-pack to
#                         serve a repository that is already on this disk, and
#                         that fork is what kept failing (EMFILE, then E2BIG).
#                         A plain path uses the local transport: hardlinks, no
#                         server process, and faster besides.
git_mirror() {                      # <url> <dir>
    git_retry git_clone_bare "$1" "$2"
}

git_clone_bare() {                  # <url> <dir>
    rm -rf "$2"
    git clone --quiet --bare "$1" "$2"
}

# Submodules recorded by <commit>. Only the first level: a nested submodule
# stays pip's problem, and none of the pins here has one.
git_mirror_submodules() {           # <mirror dir> <commit>
    local dir="$1" commit="$2" key url path sha sub
    while read -r key url; do
        [ -n "${key:-}" ] || continue
        case "$url" in
            https://*|http://*) ;;
            *) echo "--- submodule ${url}: not http(s), left to pip"; continue ;;
        esac
        path="$(git -C "$dir" config --blob "${commit}:.gitmodules" \
                    --get "${key%.url}.path")" || continue
        sha="$(git -C "$dir" ls-tree "$commit" "$path" | awk '$2 == "commit" { print $3 }')"
        [ -n "$sha" ] || continue
        sub="${mirror_root}/$(printf '%s' "$path" | tr '/' '-')"
        echo "--- submodule ${path} <- ${url}@${sha:0:12}"
        git_mirror "$url" "$sub"
        # The commit the superproject records is the one `git submodule update`
        # will ask for; a mirror without it would fail later and further away.
        git -C "$sub" cat-file -e "${sha}^{commit}" || {
            echo "${url} has no commit ${sha} (recorded for ${path})" >&2
            exit 1
        }
        git config --global "url.${sub}.insteadOf" "$url"
    done < <(git -C "$dir" config --blob "${commit}:.gitmodules" \
                 --get-regexp '^submodule\..*\.url$' || true)
}

echo "=== [${env_name}] git mirrors"
# pip clones a `name @ git+...` requirement itself and then runs
#   git submodule update --init --recursive
# inside the clone. Neither step retries in any way that helps: one flaky fetch
# of one submodule aborts the entire bulk pass, twenty minutes in. That is not
# hypothetical -- transformer_engine's 3rdparty/googletest is exactly where it
# happened, after the transformer_engine clone itself had already succeeded.
#
# So pip is pointed at local copies through url.<local>.insteadOf. Only the
# transport changes: the requirement string pip resolves is untouched, so
# direct_url.json and `pip freeze` still record the github URL and the pinned
# commit.
#
# The copies normally arrive prefetched: docker/fetch-git-mirrors.sh fills
# docker/gitmirror/<env> on the host and the Dockerfile COPYs it to
# $mirror_root -- manifest.txt is how this script learns what is in there. A
# bare `docker build` against an empty directory finds no manifest and falls
# back to cloning here, which is the old behaviour and still works; it is just
# slower, repeated on every rebuild, and needs the build to reach github.
mirror_root=/opt/ptdbench/gitmirror
manifest="${mirror_root}/manifest.txt"
mirrored=0
local_allowed=0

# Drop url.<this dest>.insteadOf rules this script wrote earlier. The
# Dockerfile stage dies after one run so leftover config never matters
# there; a debug-env container lives on, and a stale rewrite of a github
# URL onto a dest that prune just deleted is how the next `build` dies
# with "does not appear to be a git repository". Called at the start of
# this section (so a re-run is a no-op against last time's rules) and
# again at prune (so the rest of the container does not inherit them).
clear_mirror_insteadof() {
    git config --global --get-regexp '^url\..*\.insteadof$' 2>/dev/null \
        | awk -v prefix="url.${mirror_root}" '
            index($1, prefix "/") == 1 || $1 == prefix ".insteadof" { print $1 }
          ' \
        | sed 's/\.insteadof$//' \
        | while read -r section; do
              git config --global --remove-section "$section" || true
          done || true
}
clear_mirror_insteadof

# git >= 2.34.4 refuses to clone a submodule from a local path outright
# (CVE-2022-39253). What that protects against is a hostile .gitmodules naming
# a path on the build machine; here every local path is a bare mirror of the
# URL the pin itself names. Called once, from whichever of the two paths below
# first produces a local mirror.
allow_local_submodules() {
    [ "$local_allowed" -eq 0 ] || return 0
    local_allowed=1
    git config --global protocol.file.allow always
}

# What the prefetched directory covers: URL -> mirror directory. The clone loop
# below consults this and skips anything already here. Keyed by URL because
# that is what insteadOf matches on and what requirements.txt spells.
declare -A prefetched=()
if [ -f "$manifest" ]; then
    while IFS=$'\t' read -r url dir; do
        [ -n "${url:-}" ] && [ -n "${dir:-}" ] || continue
        [ -d "${mirror_root}/${dir}" ] || {
            echo "${manifest} names ${dir}, which is not in ${mirror_root}" >&2
            exit 1
        }
        allow_local_submodules
        mirrored=$((mirrored + 1))
        prefetched["$url"]="$dir"
        echo "--- prefetched ${dir} <- ${url}"
        git config --global "url.${mirror_root}/${dir}.insteadOf" "$url"
    done < "$manifest"
fi

while read -r line; do
    case "$line" in *@*git+*) ;; *) continue ;; esac
    spec="${line#*git+}"
    spec="${spec%%#*}"                      # drop #subdirectory=/#egg=
    spec="${spec%"${spec##*[![:space:]]}"}"
    url="${spec%@*}"
    rev="${spec##*@}"
    name="${line%%[[:space:]]*}"
    name="${name%%[*}"
    case "$url" in https://*|http://*) ;; *) continue ;; esac
    if ! printf '%s' "$rev" | grep -Eq '^[0-9a-fA-F]{40}$'; then
        echo "--- ${name}: ${rev} is not a full commit, left to pip"
        continue
    fi
    if [ -n "${prefetched[$url]:-}" ]; then
        # Checked against the pin rather than taken on faith: the manifest says
        # a mirror exists, not that it has the commit this file asks for. A
        # requirements.txt bumped without re-running fetch-git-mirrors.sh is
        # exactly the case that has to fail here, loudly, rather than let pip
        # resolve the pin against a mirror that predates it.
        git -C "${mirror_root}/${prefetched[$url]}" cat-file -e "${rev}^{commit}" || {
            echo "${url} mirror has no commit ${rev} (pinned by ${name})." >&2
            echo "re-run ./docker/fetch-git-mirrors.sh after moving a pin." >&2
            exit 1
        }
        continue
    fi
    # No prefetched mirrors, or a set that does not cover this URL: clone here,
    # the way this script always used to. Submodules included, for the reason
    # in the comment above.
    allow_local_submodules
    mirrored=$((mirrored + 1))
    echo "--- ${name} <- ${url}@${rev:0:12}"
    git_mirror "$url" "${mirror_root}/${name}"
    git -C "${mirror_root}/${name}" cat-file -e "${rev}^{commit}" || {
        echo "${url} has no commit ${rev} (pinned by ${name})" >&2
        exit 1
    }
    git_mirror_submodules "${mirror_root}/${name}" "$rev"
    git config --global "url.${mirror_root}/${name}.insteadOf" "$url"
done < <(sed -e 's/#.*//' "$requirements" | awk 'NF')
[ "$mirrored" -gt 0 ] || echo "--- no git requirements"

echo "=== [${env_name}] bulk requirements"
# --no-build-isolation so that anything compiled here builds against the torch
# installed above rather than a fresh one pip would fetch into a throwaway
# build environment -- a torch mismatch produces extensions that import and
# then segfault. Everything that is not a source build resolves to a wheel and
# never reaches a build backend at all.
#
# If this pass fails with a missing build backend (hatchling, setuptools-scm,
# ...), that package has no wheel for this platform: add the backend to
# build-first.txt. Do not switch the pass back to isolated builds.
bulk="/tmp/${env_name}-bulk.txt"
cp "$requirements" "$bulk"
while read -r name; do
    [ -n "$name" ] || continue
    escaped="$(printf '%s' "$name" | sed -e 's/[][\.*^$\/]/\\&/g' -e 's/_/[-_]/g')"
    sed -i -E "/^${escaped}([[:space:]]|\[|@|=)/Id" "$bulk"
done < <(special_names)
if ((${#prebuilt_pins[@]})); then
    for pin in "${prebuilt_pins[@]}"; do
        name="${pin%%==*}"
        name="${name%%[*}"
        escaped="$(printf '%s' "$name" | sed -e 's/[][\.*^$\/]/\\&/g' -e 's/_/[-_]/g')"
        sed -i -E "/^${escaped}([[:space:]]|\[|@|=)/Id" "$bulk"
    done
fi
# nvidia-cublas is already on disk from build-first. Leaving the pin in
# bulk puts it in the same resolve as torch's cuda-toolkit[cublas]==13.0.2
# extra (nvidia-cublas==13.1.0.3.*) -- a hard conflict on slime-v0.3.1,
# whose pin is 13.3.0.5 for TE 2.16.1. Strip it here; re-apply --no-deps
# after the source trees so a later extra cannot win.
if pin_for nvidia-cublas >/dev/null 2>&1; then
    sed -i -E '/^nvidia-cublas([[:space:]]|\[|@|=)/Id' "$bulk"
fi
# Index options and +cuXXX pins are handled above. Leaving --extra-index-url
# in this file would make pip chase files.pythonhosted.org for numpy et al.
# Dropping the pins is what makes $PIP_CONSTRAINT necessary; see the comment
# where it is written.
sed -i -E '/^--/d; /\+cu[0-9]/d' "$bulk"
pip_install --no-build-isolation -r "$bulk"
if [ -n "$env_torch_mm" ]; then assert_torch_cuda "bulk requirements"; fi

echo "=== [${env_name}] packages with build flags"
# One pass each, because pip has no way to attach per-requirement build flags
# inside a requirements file. Without these, apex installs as pure Python and
# Megatron silently falls back to a different layer norm.
while read -r line; do
    line="${line%%#*}"
    [ -n "${line// }" ] || continue
    read -r name rest <<<"$line"
    vars=()
    args=()
    seen_separator=0
    # shellcheck disable=SC2086
    eval "set -- ${rest}"
    for token in "$@"; do
        if [ "$token" = "--" ]; then seen_separator=1; continue; fi
        if [ "$seen_separator" -eq 1 ]; then args+=("$token"); else vars+=("$token"); fi
    done
    pin="$(pin_for "$name")"
    echo "--- ${name}"
    env "${vars[@]}" pip install --no-cache-dir --disable-pip-version-check \
        --no-build-isolation "${args[@]}" "$pin"
done < <(sed -e 's/#.*//' "${env_dir}/build-flags.txt" | awk 'NF')

echo "=== [${env_name}] editable source trees"
# Trees a task script names by absolute path, cloned *inside* the prefix so
# they travel with the env through the COPY. ptdbench-activate.sh links each to
# the path the scripts expect. Safe to relocate only because they are pure
# Python resolved at import time -- see docker/envs/*/sources.txt.
#
# No mirror bookkeeping here: these URLs are in the manifest too, so the
# insteadOf rules set above make this clone local and offline. Without
# prefetched mirrors it is a plain network clone, with git_retry around it.
#
# Patches: any docker/envs/<env>/patches/<dir>/*.patch is applied, in sorted
# order, between the checkout and the editable install. The upstream images we
# reproduce do the same thing (slimerl/slime:latest applies its own
# docker/patch/<ver>/megatron.patch on top of a vanilla clone), so a pinned
# commit alone does not reproduce them -- see docker/envs/README.md.
#
# Applied before `pip install -e` so the install sees the patched tree, and
# unconditionally rather than idempotently: the clone is fresh every build, so
# a patch that no longer applies is a pin/patch mismatch the build should fail
# on rather than skip.
while read -r dir url commit; do
    [ -n "${dir:-}" ] || continue
    target="${prefix}/src/${dir}"
    mkdir -p "${prefix}/src"
    echo "--- ${dir} <- ${url}@${commit:0:12}"
    git_retry git_clone_fresh "$url" "$target"
    git -C "$target" checkout --detach "$commit"
    apply_source_patches "$dir" "$target"
    pip_install --no-build-isolation -e "$target"
done < <(sed -e 's/#.*//' "${env_dir}/sources.txt" | awk 'NF')
if [ -n "$env_torch_mm" ]; then assert_torch_cuda "editable source trees"; fi

# transformer_engine_cu13 2.16.1's prebuilt libtransformer_engine.so needs
# cuBLAS 13.3 (cublasLtGroupedMatrixLayoutInit_internal). torch 2.11.0+cu130
# declares cuda-toolkit[cublas]==13.0.2, which pins nvidia-cublas==13.1.0.3,
# and the megatron-core editable install pulls that extra and would revert a
# first-pass 13.3. Re-apply the pin --no-deps so the wheel on disk is what
# requirements.txt says. No-op on environments that ship nvidia-cublas-cu12.
if pin="$(pin_for nvidia-cublas 2>/dev/null)"; then
    echo "=== [${env_name}] re-apply ${pin} (--no-deps)"
    pip_install --no-deps "$pin"
fi

echo "=== [${env_name}] prune"
# The git mirrors have served their purpose: everything built from them is
# installed into the prefix, and the prefix is all the final stage copies.
# Drop the insteadOf rules first either way -- they outlive the dest in a
# long-lived container, and the next clone of the same github URL would
# then be rewritten onto a path that no longer exists.
clear_mirror_insteadof
# The Dockerfile COPY is real files in a dying stage; drop them so they
# do not ride into the env-* layer. debug-env.sh's dest is a directory of
# host-mount symlinks: deleting it does not save space and leaves the
# next `build` with no dest and no manifest.
if [ -n "$(find "$mirror_root" -mindepth 1 -maxdepth 1 -type l -print -quit 2>/dev/null)" ]; then
    echo "--- keeping ${mirror_root} (debug-env host-mount dest)"
else
    rm -rf "${mirror_root:?}"
fi
micromamba clean --all --yes
# `clean --all` keeps the package directories this environment still
# references, and there is no cache mount under /opt/conda/pkgs any more (see
# the note in docker/Dockerfile), so those would ride along in this stage's
# layer -- several GB of files the prefix already has its own copy of. Removing
# them is safe precisely because the mount is gone: pkgs and the prefix are one
# filesystem now, so micromamba hardlinked rather than copied, and dropping the
# cache decrements a link count instead of deleting anything the prefix needs.
# The import smoke in `final` is what proves this held.
rm -rf /opt/conda/pkgs
find "$prefix" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true

echo "=== [${env_name}] freeze"
# Archived for G1: diffed against the upstream image's own `pip freeze`, with
# every difference required to have an explanation in docker/envs/README.md.
mkdir -p /opt/ptdbench/freeze
pip freeze > "/opt/ptdbench/freeze/${env_name}.txt"
echo "=== [${env_name}] done: $(wc -l < "/opt/ptdbench/freeze/${env_name}.txt") packages"
