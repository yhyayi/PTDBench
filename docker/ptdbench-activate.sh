#!/usr/bin/env sh
# Installed as /etc/profile.d/ptdbench-activate.sh by every
# docker/images/*.Dockerfile.
#
# Every command the harness runs goes through `sh -lc`, so this file is the one
# place that shapes the per-stage environment. POSIX sh, not bash: it is sourced
# by /bin/sh and has no reason to need more.
#
# Each environment is its own upstream image, used as-is: the interpreter, the
# CUDA toolkit and the package set are whatever that image ships, so there is
# nothing to activate. What remains is three small things:
#
#   0. the image's PATH, if the login shell dropped part of it
#   1. what prepare.sh left behind            $PTDBENCH_PREPARED/env
#   2. where a stray `pip install` may write  agent stage only
#
# The task's workspace is not this file's business: the harness bind-mounts a
# per-run copy of it at /workspace before the container starts.

# --- 0. the image's PATH -----------------------------------------------------
# A login shell runs /etc/profile first, and on Debian that resets PATH to the
# stock list -- dropping whatever the image's ENV PATH added (/usr/local/cuda/bin,
# /root/.cargo/bin, ...). The Dockerfile recorded the image's PATH at build
# time; any entry missing now goes back in front, in the image's order. Where
# /etc/profile leaves PATH alone this changes nothing, and it is idempotent, so
# it runs outside the once-only guard below (a debugging `docker run ... sh -l`
# wants it too).
if [ -r /etc/ptdbench-image-path ]; then
    _ptdbench_image_path=""
    IFS= read -r _ptdbench_image_path < /etc/ptdbench-image-path || :
    _ptdbench_missing=""
    _ptdbench_ifs=$IFS
    IFS=:
    for _ptdbench_dir in $_ptdbench_image_path; do
        [ -n "$_ptdbench_dir" ] || continue
        case ":$_ptdbench_missing:$PATH:" in
            *":$_ptdbench_dir:"*) ;;
            *) _ptdbench_missing="${_ptdbench_missing:+$_ptdbench_missing:}$_ptdbench_dir" ;;
        esac
    done
    IFS=$_ptdbench_ifs
    if [ -n "$_ptdbench_missing" ]; then
        PATH="$_ptdbench_missing${PATH:+:$PATH}"
        export PATH
    fi
    unset _ptdbench_image_path _ptdbench_missing _ptdbench_ifs _ptdbench_dir
fi

# Nothing more to do outside a PTDBench sandbox (a plain `docker run` for
# debugging, say), and nothing to do twice.
if [ -z "${PTDBENCH_STAGE:-}" ] || [ -n "${PTDBENCH_ACTIVATED:-}" ]; then
    return 0 2>/dev/null || :
fi
PTDBENCH_ACTIVATED=1
export PTDBENCH_ACTIVATED

PTDBENCH_PREPARED="${PTDBENCH_PREPARED:-/ptdbench/prepared}"
PTDBENCH_MODELS="${PTDBENCH_MODELS:-/ptdbench/models}"
PTDBENCH_DATASETS="${PTDBENCH_DATASETS:-/ptdbench/datasets}"
export PTDBENCH_PREPARED PTDBENCH_MODELS PTDBENCH_DATASETS

# --- 1. what prepare.sh left behind ------------------------------------------
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

# --- 2. where a stray install may write, agent stage only --------------------
# An agent that runs `pip install` should land in its own workspace rather than
# modify the image's site-packages -- and the verifier must not see it, because
# grading cannot depend on anything the agent installed.
#
# PIP_BREAK_SYSTEM_PACKAGES is there for the Ubuntu 24.04 images, whose system
# interpreter is marked externally managed: pip refuses even a --user install
# there without it. With PIP_USER set the install still lands in the workspace
# user base, which is the thing that marker exists to protect.
if [ "$PTDBENCH_STAGE" = "agent" ]; then
    PYTHONUSERBASE="${PYTHONUSERBASE:-/workspace/.ptdbench-user}"
    PIP_USER=1
    PIP_BREAK_SYSTEM_PACKAGES=1
    export PYTHONUSERBASE PIP_USER PIP_BREAK_SYSTEM_PACKAGES
fi
