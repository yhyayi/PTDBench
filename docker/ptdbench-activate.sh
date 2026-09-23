#!/usr/bin/env sh
# Installed as /etc/profile.d/ptdbench-activate.sh.
#
# Every command the harness runs goes through `sh -lc`, so this file is the one
# place that shapes the per-stage environment. POSIX sh, not bash: it is sourced
# by /bin/sh and has no reason to need more.
#
# It is short, and that is the point. The previous version was five sections of
# repair work -- rewriting sys.path through a site hook, symlinking relocated
# source trees back to their baked-in absolute paths, prepending a second copy
# of the console scripts, reordering the loader path so two cudnn builds would
# not fight. All of it existed because packages had been moved out of the prefix
# they were built in. Nothing is moved now, so those patches are gone.
# Activation is PATH, CUDA_HOME, and the loader path for pip's nvidia-* wheels
# (they keep .so files next to the package, not in $prefix/lib).
#
# What remains is three things that are genuinely per-stage or per-task:
#
#   1. which environment is active            PTDBENCH_CONDA_ENV
#   2. what prepare.sh left behind            $PTDBENCH_PREPARED/env
#   3. where a stray `pip install` may write  agent stage only
#
# The task's workspace is not this file's business: the harness bind-mounts a
# per-run copy of it at /workspace before the container starts.

# Nothing to do outside a PTDBench sandbox (a plain `docker run` for debugging,
# say), and nothing to do twice.
if [ -z "${PTDBENCH_STAGE:-}" ] || [ -n "${PTDBENCH_ACTIVATED:-}" ]; then
    return 0 2>/dev/null || :
fi
PTDBENCH_ACTIVATED=1
export PTDBENCH_ACTIVATED

PTDBENCH_PREPARED="${PTDBENCH_PREPARED:-/ptdbench/prepared}"
PTDBENCH_MODELS="${PTDBENCH_MODELS:-/ptdbench/models}"
PTDBENCH_DATASETS="${PTDBENCH_DATASETS:-/ptdbench/datasets}"
export PTDBENCH_PREPARED PTDBENCH_MODELS PTDBENCH_DATASETS

# --- 1. activate the environment --------------------------------------------
# The prefix is where the packages were built. CUDA_HOME is set because nvcc
# came from the conda layer and anything that JIT-compiles a kernel at runtime
# (torch.utils.cpp_extension, and through it a few Megatron and TE paths) looks
# it up rather than inferring it.
if [ -n "${PTDBENCH_CONDA_ENV:-}" ]; then
    _ptdbench_prefix="/opt/conda/envs/$PTDBENCH_CONDA_ENV"
    if [ ! -x "$_ptdbench_prefix/bin/python" ]; then
        echo "ptdbench: no such conda env '$PTDBENCH_CONDA_ENV' (looked in $_ptdbench_prefix)" >&2
        echo "ptdbench: the image has: $(ls /opt/conda/envs 2>/dev/null | tr '\n' ' ')" >&2
    else
        PATH="$_ptdbench_prefix/bin:$PATH"
        CONDA_PREFIX="$_ptdbench_prefix"
        CUDA_HOME="$_ptdbench_prefix"
        export PATH CONDA_PREFIX CUDA_HOME

        # pip nvidia-* wheels keep .so next to the package, not in
        # $prefix/lib. torch finds those via its own preload; transformer_engine
        # (and anything else that dlopens with DT_NEEDED and no RPATH) does not.
        # Inserted before $prefix/lib so the copies the wheels were built
        # against win over the conda toolkit stubs. Existing LD_LIBRARY_PATH
        # stays first: that is where --gpus puts libcuda.so.1.
        # $prefix/lib is appended, never prepended -- putting the conda toolkit
        # first would let its cudart disagree with pip's cublas/cudnn, which is
        # the failure the old section 4 was fighting.
        _ptdbench_nvidia_libs=""
        for _ptdbench_nlib in "$_ptdbench_prefix"/lib/python*/site-packages/nvidia/*/lib \
                              "$_ptdbench_prefix"/lib/python*/site-packages/nvidia/*/lib64; do
            [ -d "$_ptdbench_nlib" ] || continue
            _ptdbench_nvidia_libs="${_ptdbench_nvidia_libs:+$_ptdbench_nvidia_libs:}$_ptdbench_nlib"
        done
        LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}${_ptdbench_nvidia_libs:+$_ptdbench_nvidia_libs:}$_ptdbench_prefix/lib"
        export LD_LIBRARY_PATH
        unset _ptdbench_nvidia_libs _ptdbench_nlib

        # Source trees that a task's launch scripts name by absolute path -- see
        # docker/envs/*/sources.txt. They live inside the prefix, so they came
        # through the image build with it; this only puts them where the scripts
        # look. Safe because they are pure Python, resolved at import time.
        if [ -d "$_ptdbench_prefix/src" ]; then
            for _ptdbench_tree in "$_ptdbench_prefix"/src/*; do
                [ -d "$_ptdbench_tree" ] || continue
                _ptdbench_link="/root/$(basename "$_ptdbench_tree")"
                if [ -e "$_ptdbench_link" ] && [ ! -L "$_ptdbench_link" ]; then
                    echo "ptdbench: $_ptdbench_link exists and is not a symlink; leaving it" >&2
                    continue
                fi
                ln -sfn "$_ptdbench_tree" "$_ptdbench_link" 2>/dev/null || \
                    echo "ptdbench: could not link $_ptdbench_link -> $_ptdbench_tree" >&2
            done
            unset _ptdbench_tree _ptdbench_link
        fi
    fi
    unset _ptdbench_prefix
fi

# --- 2. what prepare.sh left behind ------------------------------------------
# The prepare stage records the paths it fetched to, plus whatever else the task
# needs exported, as KEY=VALUE lines. Plain assignments, not shell code: this
# file is sourced by every command in the agent and verifier stages, and a task
# package should not be able to run arbitrary shell in all of them.
#
# It lives under $PTDBENCH_PREPARED rather than in the workspace because the
# agent can write to the workspace, and the verifier reads this file too --
# grading must not consult anything the agent could have edited.
if [ -f "$PTDBENCH_PREPARED/env" ]; then
    while IFS= read -r _ptdbench_line || [ -n "$_ptdbench_line" ]; do
        case "$_ptdbench_line" in
            ''|'#'*) continue ;;
        esac
        _ptdbench_key="${_ptdbench_line%%=*}"
        # Reject anything that is not a bare NAME=..., so a malformed line
        # cannot turn into a command substitution when `export` parses it.
        case "$_ptdbench_key" in
            ''|*[!A-Za-z0-9_]*)
                echo "ptdbench: ignoring malformed line in prepared/env: $_ptdbench_line" >&2
                continue ;;
        esac
        export "$_ptdbench_line"
    done < "$PTDBENCH_PREPARED/env"
    unset _ptdbench_line _ptdbench_key
fi

# --- 3. where a stray install may write, agent stage only --------------------
# The conda prefixes are root-owned and read-only in practice. An agent that
# runs `pip install` anyway should land in its own workspace rather than half
# succeed against the shared environment -- and the verifier must not see it,
# because grading cannot depend on anything the agent installed.
if [ "$PTDBENCH_STAGE" = "agent" ]; then
    PYTHONUSERBASE="${PYTHONUSERBASE:-/workspace/.ptdbench-user}"
    PIP_USER=1
    export PYTHONUSERBASE PIP_USER
fi
