#!/usr/bin/env bash
# Build one conda environment by hand, inside a long-lived container, and then
# actually run a task in it.
#
# BEGIN usage
# usage: docker/debug-env.sh <command> [command ...]
#
# Commands run in the order given, so they compose:
#
#     ./docker/debug-env.sh up build smoke
#
#   base      build the base stage image                            (minutes)
#   up        start the debug container from it, with the harness's mounts
#   build     run docker/build-env.sh for the selected environment    (hours)
#   smoke     imports, prefix identity, CUDA arch list, megatron + apex
#   use       point the container at $PTDBENCH_TASK. Implied by every command
#             below, so it is rarely typed on its own -- switching tasks costs
#             four symlinks, not a container, and never the environment.
#   prepare   run the task's prepare.sh (downloads; the only networked stage)
#   retriever start the search-r1 tasks' local dense retrieval server and wait
#             for :8000. Optional -- run_qwen2.5_3B.sh starts it itself when it
#             is missing -- but paying the 60GiB index load once and then
#             iterating on `run` is why this is a separate command
#   run       run the task's training script (training.script), for real
#   all       up + build + smoke + prepare + run
#
#   shell     interactive shell, environment activated as the harness does it
#   logs      tail the last build log
#   status    what this invocation resolves to, and whether the env is built
#   reset     wipe the env prefix, keep the container (and its package cache)
#   commit    freeze the container as an env-<name> stage image
#   down      remove the container
#   clean     remove this task's run directory (through a container: its files
#             are root's). The other tasks under the same environment stay.
#
# --- which environment, which task ------------------------------------------
# The two knobs this script is steered with. Everything below them has a
# working default; these two decide what is being tested at all.
#
#   PTDBENCH_TASK           the *leaf* task directory -- the one that owns a
#                           metadata.yaml, not the family directory above it.
#                           Read on *every* command, not just `up`: it is the
#                           task switch.          default tasks/slime_retool
#   PTDBENCH_DEBUG_ENV      which environment to build. Defaults to whatever
#                           $PTDBENCH_TASK declares in environment.conda, so
#                           the task decides and the two cannot disagree. Set
#                           it explicitly for an environment no task names
#                           yet -- which is the case this script exists for.
#                                                 default <the task's>
#   PTDBENCH_TRAIN_SCRIPT   what `run` runs, relative to the workspace.
#                           Defaults to the task's training.script, so the
#                           task decides here too. Set it explicitly to run
#                           some other script in the same workspace.
#   PTDBENCH_TRAIN_ARGS     the arguments `run` passes it, word-split. Defaults
#                           to the task's training.script_args (rlve.sh takes
#                           three positional arguments). Set it to the empty
#                           string to pass none.
#   PTDBENCH_DEBUG_STAGE    the stage `shell` impersonates: agent, prepare or
#                           verifier. The activate hook keys every per-stage
#                           difference off it.    default agent
#
#     PTDBENCH_TASK=tasks/slime_search_r1  ./docker/debug-env.sh up prepare run
#     PTDBENCH_DEBUG_ENV=my-new-env        ./docker/debug-env.sh up build smoke
#
#   Two tasks that declare the same environment share one container, and
#   switching between them does not touch it:
#
#     PTDBENCH_TASK=tasks/slime_retool     ./docker/debug-env.sh up build
#     PTDBENCH_TASK=tasks/slime_search_r1  ./docker/debug-env.sh prepare run
#     PTDBENCH_TASK=tasks/slime_retool     ./docker/debug-env.sh run
#
#   Each keeps its own workspace copy, prepared/ and results/ side by side
#   under the runs root, so switching back does not re-download anything.
#   One task at a time per container, though: the switch is container-global
#   state, so do not `use` one task while another one's `run` is going. Two at
#   once means two containers (PTDBENCH_DEBUG_CONTAINER), both of which can
#   start from the same committed env image.
#
# --- where things live ------------------------------------------------------
#
#   PTDBENCH_BASE_IMAGE      the image `up` starts from. Defaults to the
#                            env-<name> stage image when `commit` has already
#                            made one (so down/up costs minutes, not hours),
#                            and to ptdbench/stage:base otherwise.
#   PTDBENCH_STAGE_REPO      where `base` and `commit` tag intermediates.
#                                                 default ptdbench/stage
#   PTDBENCH_DEBUG_CONTAINER container name. One container per environment --
#                            it is the environment that took hours to build,
#                            so that is what the default name carries, not
#                            the task.            default ptdbench-debug-<env>
#   PTDBENCH_DEBUG_RUNS_ROOT host directory holding one subdirectory per task
#                            (workspace/, prepared/, results/) plus the build
#                            logs. Per environment, like the container, and
#                            mounted whole -- which is what lets `use` switch
#                            tasks without a new mount table.
#                                                 default outputs/debug-<env>
#   PTDBENCH_DEBUG_FRESH=1   recopy this task's workspace instead of reusing
#                            the copy already there.
#   PTDBENCH_MODELS_DIR      the shared download caches. Deliberately outside
#   PTDBENCH_DATASETS_DIR    the run root: they have to survive across runs
#                            and across tasks, and they are the same default
#                            every other entry point uses.
#                                  defaults <repo>/models, <repo>/datasets
#
# --- host, GPUs and network -------------------------------------------------
# All four are fixed at `up`: a container's device list, mount table and
# environment cannot be changed afterwards. Changing one means `down`, `up`.
#
#   PTDBENCH_DEBUG_GPUS      --gpus value; "" on a builder with no driver.
#                                                            default all
#   PTDBENCH_DEBUG_SHM_SIZE  --shm-size. ray/sglang pass tensors through
#                            /dev/shm and NCCL opens its buffers there;
#                            docker's 64MB default dies on the first rollout
#                            with an opaque bus error.        default 16g
#   PTDBENCH_DEBUG_NETWORK   --network value. `host` is what makes a tunnel
#                            on the host's 127.0.0.1 reachable from inside,
#                            which is what GITHUB_HTTP_PROXY needs.
#   PTDBENCH_HF_ENDPOINT     HF mirror, for `prepare`.
#                                              default https://hf-mirror.com
#   PIP_INDEX_URL            pip mirror, for every stage -- `build` and the
#   PIP_TRUSTED_HOST         task's prepare.sh alike. Set on the container and
#                            on each exec, so changing it reaches a container
#                            that is already running.
#                     default https://pypi.tuna.tsinghua.edu.cn/simple
#
#   GITHUB_HTTP_PROXY        Passed through to build-env.sh with the same
#   MAX_JOBS                 meaning as in docker/build.sh. GITHUB_GIT_SSH is
#                            *not* one of these any more -- it belongs to
#                            ./docker/fetch-git-mirrors.sh now, and `up`
#                            rejects it rather than ignore it.
#
#     GITHUB_HTTP_PROXY=http://127.0.0.1:8444 PTDBENCH_DEBUG_NETWORK=host \
#         ./docker/debug-env.sh up build
# END usage
#
# ---------------------------------------------------------------------------
# What this is for
# ---------------------------------------------------------------------------
# docker/build.sh is the production path: `base` -> the env-* stages ->
# `final`. It is also the worst possible debugger. A build stage is a black
# box that dies after two hours with a compiler error and takes its filesystem
# with it; the environment that would have told you what went wrong never
# becomes an image. And the only way to find out whether the result can run a
# real experiment is to finish `final`, which needs *every* environment to
# build first.
#
# This script turns that inside out. It starts one container from the `base`
# stage image, mounts this directory into it, and then runs the exact same
# steps the build and the harness would run -- one at a time, interactively,
# against a filesystem that is still there when a step fails.
#
# ---------------------------------------------------------------------------
# The three properties that make this worth having
# ---------------------------------------------------------------------------
# 1. docker/ is bind-mounted at /opt/ptdbench/src, and `build` runs
#    /opt/ptdbench/src/build-env.sh -- the file in *your working tree*, not
#    the copy baked into the base image at /usr/local/libexec. Edit
#    build-env.sh, a requirements.txt, a conda.yaml, and re-run `build`. No
#    image rebuild, no cache invalidation, no waiting for `base`.
#
# 2. The container outlives the step. A failed compile leaves its build tree,
#    its config.log and its half-installed prefix sitting at
#    /opt/conda/envs/<name>, and `shell` gets you there. That is also the
#    cache: micromamba's package cache and every already-installed pin stay in
#    the container's own layer between `build` runs, so a retry after a
#    one-package fix does not re-download the world. (Which is why there are no
#    host cache mounts here -- the container *is* the cache. Note that a
#    *successful* build-env.sh ends with `micromamba clean --all`.)
#
# 3. The task is not baked into the container. A container's mount table,
#    device list and environment are fixed at creation -- which is precisely
#    why the four task-specific paths are *not* mounts. /workspace,
#    /ptdbench/prepared, /results and /ptdbench/environment are symlinks into
#    two mounts that are the same for every task:
#
#        /ptdbench/runs    the per-environment runs root, one subdirectory
#                          per task (workspace/, prepared/, results/)
#        /ptdbench/tasks   the repo's tasks/ tree, read-only
#
#    `use` re-points those four symlinks, and every task command implies it.
#    So trying the same environment against another task is seconds, and the
#    hours sitting in the container's writable layer are never at risk. The
#    paths a task sees are still the harness's exact paths -- that is the
#    reason for symlinks rather than simply exporting different variables:
#    prepare.sh bakes /workspace and /ptdbench/... absolute paths into
#    prepared/env, and the training scripts have them too.
#
#    The cost of that is the one rule this layout has: **one task at a time
#    per container**. The symlinks are container-global, so a `use` (or a
#    `prepare`, or a `run`) issued from a second shell while a training run is
#    going moves /workspace out from under it. Two tasks at once means two
#    containers -- PTDBENCH_DEBUG_CONTAINER, and they can start from the same
#    committed env image.
#
# ---------------------------------------------------------------------------
# What is faked, and what is not
# ---------------------------------------------------------------------------
# `base` does not contain the things `final` adds after the environments: the
# activate hook and the ptdbench_fetch / ptdbench_dataset shims. Without them
# no task's prepare.sh can run. `up` installs both from /opt/ptdbench/src, so
# they are the working-tree copies too -- see bootstrap_container.
#
# Everything else is the real path, deliberately:
#
#   * the prefix is /opt/conda/envs/<name>, the same path `final` copies to
#     and the same path the packages will run from. Build prefix == run
#     prefix, which is the property the whole image design rests on.
#   * every command after the build goes through `sh -lc` with
#     PTDBENCH_STAGE / PTDBENCH_CONDA_ENV set, because that is how the harness
#     runs commands, and the activate hook is where the per-stage differences
#     live.
#   * the mount table and the environment for `prepare` and `run` are the
#     harness's (see DockerRuntime._mounts and SandboxRuntime.standard_env),
#     which is what the repo-root manual_run_experiment.sh does against the
#     finished image. This script is that script, with the image replaced by
#     a container you built by hand. The one difference is the indirection
#     described above: the harness bind-mounts one task's directories onto
#     those paths, this mounts every task's and symlinks the current one --
#     so the paths match, and the per-stage read-only-ness does not (it never
#     did here; one long-lived container cannot remount per command).
#
# The one thing it cannot tell you is whether the environments still work
# *side by side* in one image -- no leaking prefixes, no shared
# /root/Megatron-LM pointing at the wrong tree. That is docker/smoke.sh
# against `final`, and it stays the gate. This is the loop you run before you
# get there.
#
# ---------------------------------------------------------------------------
# Adding an environment
# ---------------------------------------------------------------------------
#   1. docker/envs/<name>/{conda.yaml,requirements.txt,build-first.txt,
#      build-flags.txt,sources.txt}   (see docker/envs/README.md)
#   2. PTDBENCH_DEBUG_ENV=<name> ./docker/debug-env.sh up build smoke
#      -- iterate here until it is green. This is the part that used to cost
#      a full image build per attempt.
#   3. add the `FROM base AS env-<name>` stage to docker/Dockerfile, with its
#      three COPY lines (envs/<name>, wheels/<name>, gitmirror/<name>), and
#      commit docker/{wheels,gitmirror}/<name>/.gitkeep so the two prefetch
#      directories exist on a fresh clone   (collect_envs.py --check enforces
#      all of it)
#   4. ./docker/debug-env.sh commit  ->  ptdbench/stage:env-<name>, which
#      PTDBENCH_PREBUILT_ENVS=1 ./docker/build.sh then assembles into `final`
#      without rebuilding the stage.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# The task decides the environment, not the other way round: metadata.yaml's
# environment.conda is the single source of truth, the same one warm-cache.sh
# and collect_envs.py read. PTDBENCH_DEBUG_ENV overrides it for the case this
# script exists to serve -- a new environment that no task names yet.
#
# Note that this is resolved on *every* invocation, not just `up`. The task is
# what the four symlinks point at, and nothing about the container depends on
# it, so PTDBENCH_TASK is free to change between two commands.
task_dir="${PTDBENCH_TASK:-tasks/slime_retool}"
task_dir="${task_dir%/}"
task_id="$(basename "$task_dir")"

# Tasks nest to whatever depth their family needs (tasks/slime_retool, but
# also tasks/verl_hyper/qwen_dapo_hparam/task_hparam_overlong_023), so what
# this has to point at is the leaf that owns a metadata.yaml, not the family.
if [ ! -f "$task_dir/metadata.yaml" ]; then
    echo "error: $task_dir/metadata.yaml does not exist." >&2
    echo "       PTDBENCH_TASK must name the leaf task directory. Candidates under $task_dir:" >&2
    find "$task_dir" -name metadata.yaml -maxdepth 4 2>/dev/null \
        | sed -e 's#/metadata.yaml$##' -e 's/^/         /' >&2
    exit 2
fi

# tasks/ as a whole is what gets mounted, so the task has to live under it:
# that is the mount that stays the same while the task changes. The path
# inside the container is /ptdbench/tasks/<this>, nesting and all.
tasks_root="$repo_root/tasks"
case "$task_dir" in
    tasks/*) task_rel="${task_dir#tasks/}" ;;
    *) echo "error: PTDBENCH_TASK must be under tasks/ (got: $task_dir)" >&2
       echo "       the container mounts tasks/ read-only at /ptdbench/tasks and" >&2
       echo "       points /ptdbench/environment inside it; a task outside that" >&2
       echo "       tree has no path in the container." >&2
       exit 2 ;;
esac

base_image="${PTDBENCH_BASE_IMAGE:-}"
stage_repo="${PTDBENCH_STAGE_REPO:-ptdbench/stage}"

# GPUs from the start, even though build-env.sh does not need them: a
# container's mount table and device list are fixed at creation, and the same
# container has to serve `run` later. Set to "" on a build-only box with no
# driver.
gpus="${PTDBENCH_DEBUG_GPUS-all}"
shm_size="${PTDBENCH_DEBUG_SHM_SIZE:-16g}"

# --network=host is what makes a 127.0.0.1 tunnel on the host reachable from
# inside, which is the whole point of GITHUB_HTTP_PROXY. Same flag docker
# build.sh documents.
network="${PTDBENCH_DEBUG_NETWORK:-}"

hf_endpoint="${PTDBENCH_HF_ENDPOINT:-https://hf-mirror.com}"

# The prepare stage installs with pip too -- slime_search_r1 pulls faiss-gpu
# into $PTDBENCH_PREPARED/pysite -- and PyPI is where that fails here. Not the
# index: pypi.org/simple answers in half a second from inside the container.
# It is files.pythonhosted.org, which resolves and then stalls partway through
# a 135MB wheel. So the index has to move, and with it the file links it hands
# out.
#
# Same default and same reasoning as docker/build-env.sh, which has had this
# for the *build* half all along ("cluster builders typically cannot reach
# pypi.org"). The run half needs it for exactly the same reason, and this is
# the same pair of variables pip reads on its own -- no task script has to
# know about any of it.
pip_index_url="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
pip_trusted_host="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"
pip_env=(-e "PIP_INDEX_URL=$pip_index_url" -e "PIP_TRUSTED_HOST=$pip_trusted_host")

# Read the environment name, the prepare script and the training entry point
# out of the task card, with the repo's own YAML reader rather than a grep that
# approximates one. `script_args` prints one argument per line, so an argument
# with a space in it survives the round trip; the others print a single value.
read_task_field() {                 # <field>  -- conda | prepare | script | script_args
    python3 - "$task_dir" "$1" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, ".")
from ptdbench.core.types import DEFAULT_PREPARE  # noqa: E402
from ptdbench.utils.yaml import load_simple_yaml  # noqa: E402

task_dir, field = sys.argv[1], sys.argv[2]
metadata = load_simple_yaml(str(Path(task_dir) / "metadata.yaml")) or {}

if field in ("conda", "prepare"):
    environment = metadata.get("environment")
    if not isinstance(environment, dict):
        sys.exit(f"{task_dir}/metadata.yaml declares no environment:")
    if field == "conda":
        value = environment.get("conda")
        if not value:
            sys.exit(f"{task_dir} declares no environment.conda (legacy task?)")
        print(value)
    else:
        # The container sees environment/ at /ptdbench/environment, so what it
        # needs is the path relative to that directory. Same rule as warm-cache.sh.
        script = Path(task_dir) / str(environment.get("prepare") or DEFAULT_PREPARE)
        if not script.is_file():
            sys.exit(f"{task_dir}: {script} does not exist")
        print(script.relative_to(Path(task_dir) / "environment").as_posix())
    raise SystemExit(0)

training = metadata.get("training")
if not isinstance(training, dict):
    sys.exit(f"{task_dir}/metadata.yaml declares no training:")

if field == "script":
    value = training.get("script")
    if not value or not isinstance(value, str):
        sys.exit(
            f"{task_dir} declares no training.script.\n"
            f"       Add it to metadata.yaml -- the path of the launch script "
            f"relative to the task's workspace/ -- or set PTDBENCH_TRAIN_SCRIPT."
        )
    # Checked against the task's own workspace, which is the copy this script
    # copies from. A warning rather than an error: `run` executes the *run*
    # copy, which the agent stage is allowed to rewrite, so a path that is
    # missing here can still be the right one there.
    if not (Path(task_dir) / "workspace" / value).is_file():
        print(
            f"warning: {task_dir}/workspace/{value} does not exist",
            file=sys.stderr,
        )
    print(value)
else:
    args = training.get("script_args")
    if args in (None, ""):
        raise SystemExit(0)
    if not isinstance(args, list):
        sys.exit(f"{task_dir}: training.script_args must be a list")
    for arg in args:
        if not isinstance(arg, str):
            sys.exit(f"{task_dir}: training.script_args must be strings; quote {arg!r}")
        print(arg)
PY
}

if [ -n "${PTDBENCH_DEBUG_ENV:-}" ]; then
    env_name="$PTDBENCH_DEBUG_ENV"
else
    env_name="$(read_task_field conda)"
fi
[ -d "docker/envs/$env_name" ] || {
    echo "error: no such environment: docker/envs/$env_name" >&2
    echo "       the registry has: $(ls -d docker/envs/*/ | xargs -n1 basename | tr '\n' ' ')" >&2
    exit 2
}

container="${PTDBENCH_DEBUG_CONTAINER:-ptdbench-debug-$env_name}"

# What `up` starts from. `commit` freezes the built environment as
# <repo>:env-<name>, and once that exists it is a strictly better starting
# point than `base`: recreating the container -- which is still the only way to
# change the mount table, the device list or the proxy -- then costs seconds
# and keeps the environment. PTDBENCH_BASE_IMAGE pins it explicitly; `status`
# prints which one this invocation resolved to either way.
env_image="${stage_repo}:env-${env_name}"
base_image_pinned=1
if [ -z "$base_image" ]; then
    base_image_pinned=0
    if docker image inspect "$env_image" >/dev/null 2>&1; then
        base_image="$env_image"
    else
        base_image="${stage_repo}:base"
    fi
fi

# outputs/ is gitignored; models/ and datasets/ are the shared download caches
# every entry point defaults to, and they live outside the runs root on purpose
# -- they have to survive across runs and across tasks.
#
# The runs root is per *environment*, like the container, and holds one
# subdirectory per task. That is the whole task-switching mechanism: it is
# mounted whole at /ptdbench/runs, so a task that did not exist when the
# container was created still has a path inside it.
if [ -n "${PTDBENCH_DEBUG_RUN_ROOT:-}" ]; then
    echo "error: PTDBENCH_DEBUG_RUN_ROOT is PTDBENCH_DEBUG_RUNS_ROOT now, and it" >&2
    echo "       means something else: the root *above* the per-task directories," >&2
    echo "       one per environment, not one per (env, task) pair. Use" >&2
    echo "         PTDBENCH_DEBUG_RUNS_ROOT=$(dirname "$PTDBENCH_DEBUG_RUN_ROOT")" >&2
    echo "       and this task's files will be in its $task_id/ subdirectory." >&2
    exit 2
fi
runs_root="${PTDBENCH_DEBUG_RUNS_ROOT:-$repo_root/outputs/debug-$env_name}"
# Where this task's workspace copy, prepared/ and results/ live on the host.
# Named by task id rather than by the full nested path, for legibility; the
# marker file below catches the one case that makes ambiguous (two leaf tasks
# in different families with the same basename).
task_run_root="$runs_root/$task_id"
# Pre-existing per-(env, task) run roots from before the split. Moved rather
# than abandoned: a prepared workspace holds tens of GB of fetched weights.
legacy_run_root="$repo_root/outputs/debug-$env_name-$task_id"
models_dir="${PTDBENCH_MODELS_DIR:-$repo_root/models}"
datasets_dir="${PTDBENCH_DATASETS_DIR:-$repo_root/datasets}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
say() { printf '\n=== %s\n' "$*"; }

container_state() {                 # -> running | exited | ""
    docker inspect -f '{{.State.Status}}' "$container" 2>/dev/null || true
}

require_container() {
    case "$(container_state)" in
        running) ;;
        "")      echo "error: no container $container; run: $0 up" >&2; exit 2 ;;
        *)       say "starting $container"; docker start "$container" >/dev/null
                 bootstrap_container ;;
    esac
    require_runs_mount
}

# A container created before the runs root was mounted whole has one task
# bind-mounted straight onto /workspace, and no amount of symlinking inside it
# can change that. Refusing with the recipe is the only honest answer -- and
# the recipe keeps the environment, which is the thing worth keeping.
require_runs_mount() {
    docker exec "$container" test -d /ptdbench/runs 2>/dev/null && return 0
    cat >&2 <<EOF
error: $container predates the per-task layout: it has no /ptdbench/runs, so
       its /workspace is bind-mounted to a single task and cannot be switched.
       Recreate it without rebuilding the environment:

           $0 commit     # freeze the env as $env_image
           $0 down
           $0 up         # starts from that image; seconds, not hours

       ('$0 down' without the commit throws the built environment away.)
EOF
    exit 2
}


# The three things `final` adds after the environments and `base` therefore
# does not have. Re-applied on every `up`/start rather than baked once, so an
# edit to ptdbench-activate.sh or ptdbench_fetch.py takes effect on the next
# step with no rebuild and no container recreation.
#
# The shims resolve `python3` at call time, exactly as in the Dockerfile:
# there is no interpreter on the default PATH, and which one they get depends
# on which environment the activate hook put there.
#
# It also stands up /opt/ptdbench/gitmirror, which build-env.sh hardcodes and
# the Dockerfile fills with `COPY gitmirror/<env>`. Symlinks into the read-only
# docker/ mount rather than a bind mount of it, for one reason: build-env.sh
# both *reads* prefetched mirrors from that directory and *writes* fallback
# clones into it for any pin the manifest does not cover -- which is the normal
# case for the new environment this script exists to develop. A read-only bind
# mount would break the fallback; a read-write one would let a debug container
# write into your repo. Symlinks give a writable directory whose prefetched
# entries cost no disk.
bootstrap_container() {
    # A quoted heredoc on stdin rather than `bash -c '...'`: this script needs
    # both quote styles and a literal tab as IFS, and nesting those inside a
    # single-quoted argument is how it would get silently wrong.
    docker exec -i "$container" bash -euo pipefail -s -- "$env_name" <<'BOOTSTRAP'
env_name="$1"

install -m 0644 /opt/ptdbench/src/ptdbench-activate.sh \
    /etc/profile.d/ptdbench-activate.sh
for kind in hf url; do
    printf '#!/bin/sh\nexec python3 /opt/ptdbench/src/ptdbench_fetch.py %s "$@"\n' \
        "$kind" > "/usr/local/bin/ptdbench_fetch_${kind}"
    chmod 0755 "/usr/local/bin/ptdbench_fetch_${kind}"
done
printf '#!/bin/sh\nexec python3 /opt/ptdbench/src/ptdbench_dataset.py "$@"\n' \
    > /usr/local/bin/ptdbench_dataset
chmod 0755 /usr/local/bin/ptdbench_dataset
# Only the two caches: /ptdbench/prepared and /results are symlinks that
# `use` maintains, and an mkdir here would create the directory that stops
# the symlink from being made.
mkdir -p /ptdbench/models /ptdbench/datasets

src="/opt/ptdbench/src/gitmirror/${env_name}"
dst=/opt/ptdbench/gitmirror
mkdir -p "$dst"
# Only the symlinks are rebuilt. A real directory here is a fallback clone an
# earlier `build` made and is still perfectly good; throwing it away would mean
# re-cloning from github, which is the cost this directory exists to avoid.
find "$dst" -maxdepth 1 -type l -delete
rm -f "$dst/manifest.txt"
# Cleared and rewritten with the symlinks, so that repeated `up` runs do not
# stack duplicates and a pin dropped from the manifest does not keep an
# exception alive. The pattern is a value regex anchored at the mirror root, so
# it touches nothing else in the global config -- build-env.sh writes its
# insteadOf and protocol.file.allow entries there too. `|| true` because
# --unset-all exits 5 when nothing matches, which is the normal first run.
git config --global --unset-all safe.directory "^/opt/ptdbench/src/gitmirror/" || true
# The url.<mirror>.insteadOf rules from the previous `build`, too. build-env.sh
# writes one per pin it mirrors (prefetched or fallback-cloned) and never takes
# them back: in the Dockerfile a stage runs once and dies. Here the container
# lives on, and on the next `build` the stale rule redirects the fallback
# clone at its own destination -- which git_clone_bare has just rm -rf'd -- so
# git inits an empty repository and "clones" it from itself, and the run dies
# with "<url> has no commit <sha>" for a commit that is very much on github.
# `|| true` twice: --get-regexp exits 1 when nothing matches (the normal first
# run, and this shell is pipefail), and a section can already be gone.
git config --global --get-regexp '^url\..*\.insteadof$' 2>/dev/null \
    | awk '{ print $1 }' | sed 's/\.insteadof$//' \
    | while read -r section; do
          git config --global --remove-section "$section" || true
      done || true
if [ -f "$src/manifest.txt" ]; then
    cp "$src/manifest.txt" "$dst/manifest.txt"
    while IFS=$'\t' read -r url dir; do
        [ -n "${dir:-}" ] || continue
        [ -e "$dst/$dir" ] || ln -s "$src/$dir" "$dst/$dir"
        # Unlike the real build, these mirrors arrive over a bind mount and so
        # are owned by the host user, not by root. git refuses a repository it
        # does not own (CVE-2022-24765) and the refusal surfaces from inside
        # pip as an ordinary clone failure, which reads as "the mirror did not
        # work" rather than "git would not touch it". The Dockerfile has no
        # such problem: COPY makes everything root-owned.
        #
        # One exact entry per mirror, not safe.directory=*: the `<path>/*` glob
        # needs git 2.38 and the base image ships Ubuntu 22.04's 2.34.1, which
        # leaves `*` as the only wildcard that works -- and that disables the
        # check for every repository in the container, including the task
        # workspaces. The path is the resolved one, which also covers reaching
        # the mirror through the symlink above (verified).
        git config --global --add safe.directory "$src/$dir"
    done < "$src/manifest.txt"
fi
BOOTSTRAP
}

# ---------------------------------------------------------------------------
# use -- point the container at $PTDBENCH_TASK
# ---------------------------------------------------------------------------
# The task switch, and the reason `up` no longer owns the task. Two halves:
#
#   host side       make (or adopt) $runs_root/$task_id, with the workspace
#                   copy the agent stage is allowed to write to
#   container side  re-point the four task-specific paths at it
#
# Cheap enough to run before every task command rather than being remembered,
# which is what keeps `prepare` and `run` from ever acting on the task that
# happened to be selected last.
select_task() {
    require_container

    # One directory per task, adopted from the old per-(env, task) layout when
    # that is what is on disk -- a prepared workspace is tens of GB of fetched
    # weights and is not worth re-downloading over a rename.
    #
    # Only into the *default* runs root, and only as a same-filesystem rename.
    # Both restrictions are the same lesson: `mv` across devices is a copy, the
    # files inside were written by root (ptdbench_fetch lands weights
    # read-only), and a copy that cannot read them fails halfway with two
    # half-directories to sort out. Under outputs/ the rename is atomic and
    # needs no read permission at all.
    if [ ! -d "$task_run_root" ] && [ -d "$legacy_run_root" ] \
       && [ -z "${PTDBENCH_DEBUG_RUNS_ROOT:-}" ]; then
        mkdir -p "$runs_root"
        if [ "$(stat -c %d "$legacy_run_root")" != "$(stat -c %d "$runs_root")" ]; then
            echo "note: ${legacy_run_root#"$repo_root/"} is on another filesystem than" >&2
            echo "      ${runs_root#"$repo_root/"}, so it is left where it is. Move it with" >&2
            echo "      root's permissions if you want to keep its prepared workspace:" >&2
            echo "        sudo mv $legacy_run_root $task_run_root" >&2
        else
            say "adopting ${legacy_run_root#"$repo_root/"} -> ${task_run_root#"$repo_root/"}"
            mv "$legacy_run_root" "$task_run_root"
            # The build log was per run root and is per environment now.
            if [ -d "$task_run_root/logs" ]; then
                mkdir -p "$runs_root/logs"
                find "$task_run_root/logs" -maxdepth 1 -type f -name '*.log' \
                    -exec mv -n {} "$runs_root/logs/" \; 2>/dev/null || true
                rmdir "$task_run_root/logs" 2>/dev/null || true
            fi
        fi
    fi
    mkdir -p "$task_run_root"/{prepared,results} "$runs_root/logs" \
             "$models_dir" "$datasets_dir"

    # Directories are named by task id, and two families can hold leaves with
    # the same basename. The marker makes that collision loud instead of
    # letting the second task inherit the first one's prepared weights.
    local marker="$task_run_root/.task-dir" owner
    if [ -f "$marker" ]; then
        owner="$(cat "$marker")"
        if [ "$owner" != "$task_dir" ]; then
            echo "error: $task_run_root belongs to $owner, not $task_dir." >&2
            echo "       Two leaf tasks share the basename '$task_id'. Set" >&2
            echo "       PTDBENCH_DEBUG_RUNS_ROOT to keep them apart." >&2
            exit 2
        fi
    else
        printf '%s\n' "$task_dir" > "$marker"
    fi

    # The workspace is a per-task copy because the agent stage writes to it --
    # prepare.sh lands the weights there, the training script writes
    # checkpoints there. Never mount the task's own tree. An existing copy is
    # left alone, so switching back to a task does not undo its `prepare`;
    # PTDBENCH_DEBUG_FRESH=1 forces a clean one.
    if [ -d "$task_run_root/workspace" ] && [ "${PTDBENCH_DEBUG_FRESH:-0}" = 0 ]; then
        say "task $task_id: reusing workspace ${task_run_root#"$repo_root/"}/workspace"
        echo "    (PTDBENCH_DEBUG_FRESH=1 to recopy it from $task_dir/workspace)"
    else
        say "task $task_id: copying $task_dir/workspace -> ${task_run_root#"$repo_root/"}/workspace"
        # Written by root inside the container -- fetched weights are landed
        # read-only -- so the removal has to be done by root as well.
        if [ -d "$task_run_root/workspace" ]; then
            docker exec "$container" rm -rf "/ptdbench/runs/$task_id/workspace"
        fi
        cp -a "$task_dir/workspace" "$task_run_root/workspace"
    fi

    # The task's environment.conda and $PTDBENCH_DEBUG_ENV can disagree only
    # because the latter was set by hand. Say so: prepare.sh and the training
    # script will run against an environment the task never asked for.
    local declared
    declared="$(read_task_field conda 2>/dev/null || true)"
    if [ -n "$declared" ] && [ "$declared" != "$env_name" ]; then
        echo "warning: $task_id declares environment.conda=$declared, but this" >&2
        echo "         container is $env_name (PTDBENCH_DEBUG_ENV). Unset it to" >&2
        echo "         let the task choose, which also picks the right container." >&2
    fi

    docker exec -i "$container" bash -euo pipefail -s -- \
        "$task_id" "$task_rel" <<'SELECT'
task_id="$1"
task_rel="$2"
run="/ptdbench/runs/$task_id"

# ln -sfn is not enough on its own: when the path is an existing *directory*
# it links *inside* it instead of replacing it, which is exactly the shape the
# image's own WORKDIR /workspace and any mkdir leave behind. So the old entry
# goes first -- but a non-empty directory is never something this script made,
# and silently deleting it would be deleting someone's data.
relink() {                          # <path> <target>
    if [ -L "$1" ]; then
        rm -f "$1"
    elif [ -d "$1" ]; then
        rmdir "$1" 2>/dev/null || {
            echo "error: $1 is a non-empty directory, not a symlink." >&2
            echo "       This container predates the per-task layout, or" >&2
            echo "       something wrote there. Move it aside and retry." >&2
            exit 1
        }
    elif [ -e "$1" ]; then
        echo "error: $1 exists and is neither a symlink nor a directory" >&2
        exit 1
    fi
    ln -s "$2" "$1"
}

mkdir -p "$run/prepared" "$run/results"
[ -d "$run/workspace" ] || {
    echo "error: $run/workspace is missing inside the container" >&2
    exit 1
}
# The harness's four paths. Every one of them is per task, and every one of
# them is baked into something -- prepared/env, the training scripts,
# PYTHONUSERBASE in the activate hook -- so the paths stay and the targets move.
relink /workspace            "$run/workspace"
relink /ptdbench/prepared    "$run/prepared"
relink /results              "$run/results"
relink /ptdbench/environment "/ptdbench/tasks/$task_rel/environment"
# Read back by `status` and by the shell prompt banner: the selected task is
# container state now, and guessing it from the host would be a second source
# of truth.
printf '%s\n' "$task_id" > /opt/ptdbench/current-task
SELECT
}

cmd_use() {
    select_task
    cat <<EOF

  task       $task_id  ($task_dir)
  env        $env_name
  run dir    ${task_run_root#"$repo_root/"}
  next       $0 prepare
EOF
}

# Same entry path as the harness and as docker/smoke.sh: `sh -lc`, with the
# stage set per command. Calling /opt/conda/envs/<name>/bin/python directly
# would work and would test a path no evaluation ever takes.
#
# PTDBENCH_TASK_ID is per command for the same reason the paths are symlinks:
# the container's own environment was fixed when it was created, and the task
# was not yet decided then.
#
# `cd /workspace` inside the shell rather than `docker exec -w /workspace`:
# docker resolves the workdir through the symlink and the process would start
# in /ptdbench/runs/<task>/workspace, while the shell's own cd is logical and
# leaves PWD=/workspace. That matters because task scripts read $PWD -- slime's
# prepare.sh defaults DATA_DIR and MODEL_DIR to it and writes the result into
# prepared/env, which the verifier then reads. Under the harness that value is
# /workspace, so it is /workspace here.
#
# The pip mirror is per command for the same reason as the task: it was also
# set at creation (for the sake of a `docker exec` typed by hand), but passing
# it here means changing it is a variable on the next command rather than a
# `down` and an `up`.
exec_stage() {                      # <stage> <script>
    docker exec -e "PTDBENCH_STAGE=$1" -e "PTDBENCH_TASK_ID=$task_id" \
        "${pip_env[@]}" "$container" sh -lc "cd /workspace || exit 1
$2"
}

exec_stage_tty() {                  # <stage> <script>
    docker exec -it -e "PTDBENCH_STAGE=$1" -e "PTDBENCH_TASK_ID=$task_id" \
        "${pip_env[@]}" "$container" sh -lc "cd /workspace || exit 1
$2"
}

# Same shape again, detached. The one caller is `retriever`, which starts a
# process that has to outlive the exec -- `docker exec -d` is what makes the
# container, not this script's ssh session, the thing it is parented to.
exec_stage_detached() {             # <stage> <script>
    docker exec -d -e "PTDBENCH_STAGE=$1" -e "PTDBENCH_TASK_ID=$task_id" \
        "${pip_env[@]}" "$container" sh -lc "cd /workspace || exit 1
$2"
}


# The training entry point. Declared, not detected: metadata.yaml's
# training.script is the single source of truth, the same way environment.conda
# is for the environment, and the task's own instruction.md quotes the same
# path. The glob this used to do (examples/*/*_rl.sh, examples/*/run_*.sh)
# described slime's layout and nothing else -- rlve's entry point lives at
# scripts/training/<model>/rlve.sh and matched neither pattern, so `run` could
# not start those tasks at all without PTDBENCH_TRAIN_SCRIPT.
#
# The path is workspace-relative, which is both what the task card records and
# what PTDBENCH_TRAIN_SCRIPT has always meant.
find_train_script() {
    if [ -n "${PTDBENCH_TRAIN_SCRIPT:-}" ]; then
        printf '%s\n' "$PTDBENCH_TRAIN_SCRIPT"
        return
    fi
    read_task_field script
}

# The arguments that go with it, into the global $train_args. A few launch
# scripts are not runnable without them -- rlve.sh exits on `$# -lt 3` -- so the
# path alone is not enough to describe how a task starts.
#
# `+set` rather than `:-`: PTDBENCH_TRAIN_ARGS="" has to be able to mean "no
# arguments, ignore the task card", which is indistinguishable from unset under
# the usual idiom.
load_train_args() {                 # -> $train_args
    train_args=()
    local raw arg
    if [ -n "${PTDBENCH_TRAIN_ARGS+set}" ]; then
        # Word-split on purpose: this variable is a command line, not a path.
        for arg in $PTDBENCH_TRAIN_ARGS; do train_args+=("$arg"); done
        return
    fi
    # Not through a pipe or a process substitution: a failure in there is
    # invisible to `set -e`, and "the task card is unreadable" must not silently
    # become "this script takes no arguments".
    raw="$(read_task_field script_args)"
    [ -n "$raw" ] || return 0
    while IFS= read -r arg; do train_args+=("$arg"); done <<<"$raw"
}

# Single-quoting for the `sh -lc` on the other side of `docker exec`. The
# command is assembled as text there, so anything that is not quoted here is
# quoted by nobody.
shquote() {
    printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"
}


# ---------------------------------------------------------------------------
# base -- the stage image this whole thing starts from
# ---------------------------------------------------------------------------
# Roughly half a GB and minutes, not hours: no framework, no CUDA toolkit, no
# Python. build.sh tags a --target build as ptdbench/stage:<target> rather than
# as the evaluation image, which is where $base_image's default comes from.
cmd_base() {
    local built="${stage_repo}:base"
    if [ "$base_image_pinned" = 1 ] && [ "$built" != "$base_image" ]; then
        echo "note: build.sh tags the base stage $built, but this invocation" >&2
        echo "      points at PTDBENCH_BASE_IMAGE=$base_image" >&2
    fi
    say "building $built"
    PTDBENCH_STAGE_REPO="$stage_repo" ./docker/build.sh --target base
}

# ---------------------------------------------------------------------------
# up -- the container everything else runs in
# ---------------------------------------------------------------------------
# GITHUB_GIT_SSH used to be live here: it was forwarded into the container and
# ~/.ssh was mounted read-only, because build-env.sh rewrote https github URLs
# to ssh and BuildKit's --ssh has no docker-run equivalent. That rewrite is
# gone from build-env.sh -- it happens on the host in fetch-git-mirrors.sh now,
# and the build (and this script) read the resulting mirrors off disk. Silently
# ignoring the flag would be the worst outcome: it would look like it had done
# something while every pin was cloned from github anyway, which is exactly the
# slow path it exists to avoid. So it says so, the same way docker/build.sh
# does.
reject_git_ssh() {
    if [ "${GITHUB_GIT_SSH:-0}" != "0" ]; then
        echo "GITHUB_GIT_SSH is a flag for the prefetch step now, not for this:" >&2
        echo "  GITHUB_GIT_SSH=1 ./docker/fetch-git-mirrors.sh" >&2
        echo "then '$0 up build' as usual -- the mirrors it writes are what this" >&2
        echo "container installs from, and they record git+https URLs either way." >&2
        exit 2
    fi
}

cmd_up() {
    reject_git_ssh
    docker image inspect "$base_image" >/dev/null 2>&1 || {
        echo "error: image $base_image not found; build it first:" >&2
        echo "       $0 base" >&2
        exit 2
    }

    # An existing container is reused, never recreated. It holds the built
    # environment -- tens of GB and hours of compiling that live only in its
    # writable layer -- so `up` must not be able to throw that away by
    # accident. Recreating is `down` then `up`, which says so out loud.
    local state
    state="$(container_state)"
    if [ -n "$state" ]; then
        if [ "$state" != running ]; then
            say "starting the existing $container (was: $state)"
            docker start "$container" >/dev/null
        else
            say "$container is already up"
        fi
        bootstrap_container
        # Not a warning about the wrong task any more: pointing the container
        # at this one is now a supported operation, and this is it.
        select_task
        return
    fi

    mkdir -p "$runs_root/logs" "$models_dir" "$datasets_dir"

    docker rm -f "$container" >/dev/null 2>&1 || true

    local args=(
        docker run -d --name "$container"
        --shm-size="$shm_size"

        # --- the build half ------------------------------------------------
        # docker/ itself, so build-env.sh and the env registry are the
        # working-tree copies. Read-only: nothing in a build stage writes
        # here, and a container that could edit the repo is a bad debugger.
        -v "$repo_root/docker:/opt/ptdbench/src:ro"
        # build-env.sh hardcodes /opt/ptdbench/envs/<name> (the Dockerfile
        # COPYs it there), so the registry is mounted a second time at the
        # path it expects rather than the script being taught a variable.
        -v "$repo_root/docker/envs:/opt/ptdbench/envs:ro"

        # --- the run half --------------------------------------------------
        # Two task-independent mounts and two shared caches. The harness's
        # own four paths -- /workspace, /ptdbench/prepared, /results and
        # /ptdbench/environment -- are symlinks into the first two, made by
        # `use`, because a mount table cannot be changed after creation and
        # the task has to be changeable. See property 3 at the top.
        -v "$runs_root:/ptdbench/runs"
        -v "$tasks_root:/ptdbench/tasks:ro"
        -v "$models_dir:/ptdbench/models"
        -v "$datasets_dir:/ptdbench/datasets"

        # PTDBENCH_STAGE is deliberately NOT here. It is per-command: the
        # activate hook keys every per-stage difference off it, and a build
        # that inherited PTDBENCH_STAGE=agent would install into the agent's
        # private PYTHONUSERBASE instead of the prefix.
        #
        # PTDBENCH_TASK_ID is not here either, and for the new reason: it is
        # per task, this container serves all of them, and exec_stage passes
        # the current one. Everything that remains is either per environment
        # (which this container is) or the same for every task.
        -e "PTDBENCH_CONDA_ENV=$env_name"
        -e PTDBENCH_EVALUATION_LEVEL=solution
        -e PTDBENCH_WORKSPACE=/workspace
        -e PTDBENCH_ENVIRONMENT=/ptdbench/environment
        -e PTDBENCH_PREPARED=/ptdbench/prepared
        -e PTDBENCH_MODELS=/ptdbench/models
        -e PTDBENCH_DATASETS=/ptdbench/datasets
        -e PTDBENCH_RESULTS=/results
        -e "HF_ENDPOINT=$hf_endpoint"
        -e HF_HUB_DISABLE_XET=1
        # Both halves read these: build-env.sh for the environment, pip inside
        # prepare.sh for whatever a task installs. Set here as well as per
        # command so that a `docker exec` typed by hand behaves the same.
        -e "PIP_INDEX_URL=$pip_index_url"
        -e "PIP_TRUSTED_HOST=$pip_trusted_host"
        -e WANDB_MODE=offline
        -e WANDB_API_KEY=offline-placeholder
        -e WANDB_DIR=/results/experiment_logs
        # The image's WORKDIR is /workspace, which does not exist until a task
        # is selected -- and docker would create it as a real directory, in
        # the way of the symlink. `/` it is; exec_stage cds into /workspace
        # from inside the shell.
        -w /
    )

    # `if`, not `[ ... ] && ...`: under `set -e` a false short-circuit is a
    # failed command and would abort the script instead of skipping the flag.
    if [ -n "$gpus" ]; then args+=(--gpus "$gpus"); fi
    if [ -n "$network" ]; then args+=(--network "$network"); fi

    # Build-time knobs build-env.sh reads. Passed through only when set, so
    # the container's environment says what was actually asked for. (The two
    # pip variables are not in this list any more: they are always set, a few
    # lines above, because they have a default now.)
    local passthrough
    for passthrough in GITHUB_HTTP_PROXY MAX_JOBS; do
        if [ -n "${!passthrough:-}" ]; then
            args+=(-e "$passthrough=${!passthrough}")
        fi
    done

    args+=("$base_image" sleep infinity)

    say "starting $container from $base_image"
    if [ "$base_image" = "$env_image" ] && [ "$base_image_pinned" = 0 ]; then
        # Chosen because it exists, not because anyone said so on this
        # invocation -- and it is a *snapshot*, made by whichever `commit` ran
        # last. If a later `build` was never committed, its work is not in
        # here. Printing when it was made is what makes that checkable.
        echo "    (the committed env image, built $(docker image inspect \
            "$env_image" --format '{{.Created}}' 2>/dev/null | cut -c1-19);"
        echo "     PTDBENCH_BASE_IMAGE=${stage_repo}:base to start from a bare base instead)"
    fi
    "${args[@]}" >/dev/null
    bootstrap_container
    select_task

    local next="$0 build"
    if [ "$base_image" = "$env_image" ]; then
        next="$0 prepare   (the environment came with $env_image)"
    fi
    cat <<EOF

  container  $container
  env        $env_name
  task       $task_id
  runs root  $runs_root
  next       $next
EOF
}

# ---------------------------------------------------------------------------
# build -- build-env.sh, by hand, in the container
# ---------------------------------------------------------------------------
# The hours-long step, and the reason this file exists. Runs the working-tree
# build-env.sh with plain `bash`, not `sh -lc`: a build stage in the Dockerfile
# does not go through a login shell either, so the activate hook must not run
# here. It would put PTDBENCH_CONDA_ENV's (possibly half-built) prefix on PATH
# in front of the build, which is not what the stage sees.
#
# PTDBENCH_WHEELS_DIR is set here rather than at `up` because it is per
# environment -- docker/wheels/<env>, the same subdirectory the Dockerfile's
# `COPY wheels/<env>` hands the real stage. Setting it at exec time also means a
# container created before this was per-env still gets the right one.
cmd_build() {
    require_container
    # A previous `build` in this container may have pruned
    # /opt/ptdbench/gitmirror and left url.*.insteadOf behind. `up` already
    # rebuilds the dest from the host-mount mirrors and drops those rules;
    # `reset` + `build` does not go through `up`, so do it here too.
    bootstrap_container
    # Under the runs root, not under a task's directory: the environment is
    # what is being built, and the same log is the record for every task that
    # later runs in it.
    local log="$runs_root/logs/build-$env_name.log"
    mkdir -p "$runs_root/logs"
    say "building env $env_name in $container  (log: ${log#"$repo_root/"})"
    echo "    tail it from another shell with: $0 logs"
    # Not `docker exec -it`: this has to survive the terminal it was started
    # from, and a TTY here mangles the compiler output in the log.
    #
    # errexit is lifted around the pipeline on purpose. A failed build is the
    # expected case here and has something useful to say about where to look;
    # letting `set -e` kill the script mid-pipeline would skip all of it.
    set +e
    docker exec -e "PTDBENCH_WHEELS_DIR=/opt/ptdbench/src/wheels/$env_name" \
        "${pip_env[@]}" "$container" \
        bash /opt/ptdbench/src/build-env.sh "$env_name" 2>&1 | tee "$log"
    # PIPESTATUS, not $?, which is tee's -- and read immediately, because
    # every subsequent command resets it.
    local status=${PIPESTATUS[0]}
    set -e
    if [ "$status" -eq 0 ]; then
        say "built $env_name; next: $0 smoke"
    else
        echo
        echo "build failed ($status). The container is still up, with" >&2
        echo "everything the failed step left behind:" >&2
        echo "    $0 shell        # /opt/conda/envs/$env_name, build trees, pip logs" >&2
        echo "    $0 reset        # wipe the prefix and start the env over" >&2
        return "$status"
    fi
}

# ---------------------------------------------------------------------------
# smoke -- what the Dockerfile's final stage checks, plus the GPU half
# ---------------------------------------------------------------------------
# The driver-free import check is the Dockerfile's; the CUDA and arch-list half
# is docker/smoke.sh's. Both are here because at this point in the loop there
# is no `final` image to run either against, and a stack that imports but was
# compiled for the wrong card is the failure that otherwise surfaces hours
# later inside a training run.
#
# megatron/apex are checked only when this environment actually carries a
# Megatron-LM tree, read off the prefix rather than off a hardcoded list of
# environment names -- verl has no megatron at all, and a new environment
# should not have to be added here to be checked correctly.
cmd_smoke() {
    select_task
    say "smoke: $env_name"
    exec_stage agent '
        set -e
        python3 - <<PY
import os, sys, sysconfig, importlib.metadata as md
import torch, flash_attn, transformer_engine.pytorch  # noqa: F401

prefix = "/opt/conda/envs/" + os.environ["PTDBENCH_CONDA_ENV"]
# Build prefix == run prefix. Cheap, and the single most valuable assertion in
# the image design: anything resolving outside its own prefix means something
# was relocated.
assert sys.executable.startswith(prefix + "/"), sys.executable
assert sysconfig.get_paths()["purelib"].startswith(prefix + "/")
assert torch.__file__.startswith(prefix + "/"), torch.__file__
print(" python  ", sys.version.split()[0], sys.executable)
print(" torch   ", torch.__version__, "cuda", torch.version.cuda)
print(" flash_attn", md.version("flash_attn"))
print(" transformer_engine", md.version("transformer_engine"))

if not torch.cuda.is_available():
    print(" SKIP: no GPU visible to this container; the CUDA half is unchecked")
    raise SystemExit(0)
arches = torch.cuda.get_arch_list()
missing = [a for a in ("sm_80", "sm_90") if a not in arches]
assert not missing, f"built without {missing}; have {arches}"
torch.zeros(1).cuda()
print(" cuda ok ", arches)
PY
        # The editable source trees and the symlinks the activate hook makes
        # for them. slime assigns PYTHONPATH=/workspace:/root/Megatron-LM
        # inside its Ray runtime_env, where no outer export can reach it, so
        # the path has to be real and has to point at this env.
        if [ -d "/opt/conda/envs/$PTDBENCH_CONDA_ENV/src/Megatron-LM" ]; then
            test "$(readlink -f /root/Megatron-LM)" \
                 = "/opt/conda/envs/$PTDBENCH_CONDA_ENV/src/Megatron-LM" \
              || { echo "/root/Megatron-LM points elsewhere" >&2; exit 1; }
            PYTHONPATH=/root/Megatron-LM python3 - <<PY
import megatron.core as m
# apex built without --cpp_ext --cuda_ext installs as pure Python and this
# module is simply absent; megatron ColumnParallelLinear then raises at layer
# construction, which is well inside prepare.sh.
import fused_weight_gradient_mlp_cuda as f
print(" megatron", m.__file__)
print(" apex    ", f.__file__)
PY
        fi
        if command -v nvidia-smi >/dev/null 2>&1; then
            nvidia-smi -L | head -8
        else
            echo " no nvidia-smi in this container"
        fi
        df -h /dev/shm | tail -1
    '
    say "smoke passed; next: $0 prepare"
}

# ---------------------------------------------------------------------------
# prepare -- the task's own prepare.sh
# ---------------------------------------------------------------------------
# The only stage that is supposed to reach the network, and the only one with
# a writable $PTDBENCH_MODELS / $PTDBENCH_DATASETS. Everything it downloads is
# revision-pinned and sha256-checked, and lands in the shared cache outside
# the run root, so re-running this is cheap after the first time.
cmd_prepare() {
    select_task
    local script
    script="$(read_task_field prepare)"
    say "prepare: $task_id  (/ptdbench/environment/$script)"
    exec_stage prepare "bash /ptdbench/environment/$script"
    say "prepared; next: $0 run"
}

# ---------------------------------------------------------------------------
# retriever -- the search-r1 tasks' local dense retrieval server
# ---------------------------------------------------------------------------
# run_qwen2.5_3B.sh starts this itself when :8000 has no listener, so `run`
# works without ever typing this command. It is here for the case the debug loop
# is actually about: a cold start reads a 60GiB faiss index and shards it across
# eight GPUs, and paying that once and then iterating on `run` beats paying it
# inside every attempt. It also puts the server's own output in its own file
# instead of interleaved into train.log.
#
# Detached rather than backgrounded from here: the server has to outlive this
# exec, and in the harness's own layout it outlives whole stages.
#
# Worth remembering that this is the one thing about the debug container that is
# *not* like production. A long-lived container makes a persistent service easy;
# the harness's prepare stage is created and removed around prepare.sh, so
# nothing there can hold one. That asymmetry is why the auto-start lives in the
# training script rather than in prepare.sh -- this command is a convenience on
# top of it, never the thing that makes a run possible.
cmd_retriever() {
    select_task
    local port="${PTDBENCH_RETRIEVAL_PORT:-8000}"
    local log="$task_run_root/workspace/retrieval_server.log"

    # /dev/tcp is a bash feature and exec_stage runs `sh -lc`, which is dash
    # here -- hence the inner bash. Probed from inside the container, because
    # --network=host is optional in this script: the port is the container's,
    # not necessarily the host's.
    retriever_listening() {
        exec_stage agent "bash -c '(exec 3<>/dev/tcp/127.0.0.1/$port)' 2>/dev/null"
    }

    # The bracket trick is load-bearing: exec_stage's own `sh -lc` carries this
    # pattern in its command line, and a plain `pgrep -f retrieval_server.py`
    # would match that shell and report the server alive forever.
    retriever_alive() {
        exec_stage agent \
            "pgrep -f '[r]etrieval_server.py|[s]tart_retriver.sh' >/dev/null"
    }

    # Through the container: the file is written by root, and reading it from
    # the host depends on the run root's permissions.
    retriever_tail() {
        echo "       last 40 lines of ${log#"$repo_root/"}:" >&2
        exec_stage agent "tail -n 40 /workspace/retrieval_server.log" >&2 || true
    }

    if retriever_listening; then
        say "retriever already listening on :$port in $container"
        echo "    log: ${log#"$repo_root/"}"
        return 0
    fi

    say "starting the retriever in $container (task=$task_id)"
    echo "    log: ${log#"$repo_root/"}"
    # start_retriver.sh, not start_retriever.sh -- the file in the recipe is
    # spelled without the second 'e'. No shebang either, hence `bash`.
    exec_stage_detached agent \
        "exec setsid bash examples/search-r1/start_retriver.sh \
             >/workspace/retrieval_server.log 2>&1 </dev/null"

    # The port opens only after the index and the corpus are loaded, so waiting
    # for it is waiting for readiness. A cold corpus costs a datasets arrow
    # conversion of a 14GB jsonl on top of the index load; prepare.sh warms that
    # cache, and this timeout has to cover the case where it did not.
    local waited=0 timeout="${PTDBENCH_RETRIEVAL_TIMEOUT:-1800}"
    until retriever_listening; do
        if ! retriever_alive; then
            echo "error: the retriever exited before it started listening." >&2
            retriever_tail
            exit 1
        fi
        if [ "$waited" -ge "$timeout" ]; then
            echo "error: still no listener on :$port after ${timeout}s." >&2
            retriever_tail
            exit 1
        fi
        sleep 10
        waited=$((waited + 10))
        [ $((waited % 60)) -ne 0 ] || echo "    loading the index; ${waited}s elapsed"
    done
    say "retriever up on :$port; next: $0 run"
}

# ---------------------------------------------------------------------------
# run -- the experiment, for real
# ---------------------------------------------------------------------------
cmd_run() {
    select_task
    local script
    script="$(find_train_script)"
    load_train_args
    # Assembled here rather than inside the heredoc so that every argument is
    # quoted exactly once, and so the line that is echoed is the line that runs.
    local command arg
    command="bash $(shquote "$script")"
    for arg in ${train_args[@]+"${train_args[@]}"}; do
        command+=" $(shquote "$arg")"
    done
    local log="$task_run_root/results/experiment_logs/train.log"
    say "run: ${command#bash }"
    echo "    log: ${log#"$repo_root/"}"
    exec_stage agent "
        mkdir -p /results/experiment_logs
        $command 2>&1 | tee /results/experiment_logs/train.log
    "
}

# ---------------------------------------------------------------------------
# The small ones
# ---------------------------------------------------------------------------
cmd_shell() {
    select_task
    # agent, not prepare: this is the stage a human is usually impersonating,
    # and it is the one with the private PYTHONUSERBASE, so a `pip install`
    # typed here cannot damage the prefix being debugged.
    say "shell in $container (stage=${PTDBENCH_DEBUG_STAGE:-agent}, env=$env_name, task=$task_id)"
    exec_stage_tty "${PTDBENCH_DEBUG_STAGE:-agent}" bash
}

cmd_logs() {
    local log="$runs_root/logs/build-$env_name.log"
    [ -f "$log" ] || { echo "no build log at $log" >&2; exit 1; }
    tail -f "$log"
}

# A failed build usually leaves a prefix that micromamba will refuse to create
# over. Wiping just the prefix keeps the rest of the container -- the package
# cache above all -- so the retry is much faster than a fresh `up`.
cmd_reset() {
    require_container
    say "removing /opt/conda/envs/$env_name in $container"
    docker exec "$container" rm -rf "/opt/conda/envs/$env_name" \
                                    "/opt/ptdbench/freeze/$env_name.txt"
    echo "next: $0 build"
}

# Freeze the container into the image the Dockerfile's env-<name> stage would
# have produced. Layout matches, because nothing was relocated: the prefix is
# at /opt/conda/envs/<name> and the freeze manifest at
# /opt/ptdbench/freeze/<name>.txt, which is exactly what `final` COPYs out of
# that stage. So this is a drop-in for build.sh's prebuilt path:
#
#   PTDBENCH_PREBUILT_ENVS=1 ./docker/build.sh     # assembles final, minutes
#
# Bind-mounted paths are not part of a commit, so /opt/ptdbench/{src,envs},
# /workspace and the caches do not travel -- which is what we want; `final`
# COPYs the registry from the build context anyway.
cmd_commit() {
    require_container
    local image="$env_image"
    docker exec "$container" test -x "/opt/conda/envs/$env_name/bin/python" || {
        echo "error: /opt/conda/envs/$env_name has no interpreter; build it first" >&2
        exit 2
    }
    say "committing $container -> $image"
    docker commit "$container" "$image" >/dev/null
    docker image inspect "$image" --format 'size: {{.Size}} bytes'
    cat <<EOF

  This image stands in for the Dockerfile's env-$env_name stage:
      PTDBENCH_PREBUILT_ENVS=1 ./docker/build.sh
  It is what the container contains, not what requirements.txt says --
  rebuild the stage after changing the closure, and gate on docker/smoke.sh.

  It is also what this script's own '$0 up' starts from from now on, so
  '$0 down' no longer costs the build -- which is what makes changing a
  mount, a device or the proxy a cheap operation again.
EOF
}

cmd_down() {
    say "removing $container"
    docker rm -f "$container" >/dev/null 2>&1 || true
    echo "the runs root is untouched: $runs_root"
    echo "(the built environment went with the container; '$0 commit' first to keep it --"
    echo " the next '$0 up' then starts from $env_image and needs no build)"
}

# Everything under a task's run directory was written by root inside the
# container -- the fetched weights above all, which ptdbench_fetch lands
# read-only. A plain `rm -rf` from the host user's account fails on every one of
# them, so the removal is done by a container that is root, which is the only
# account that can. Nothing is mounted but the runs root.
#
# One task, not the whole runs root: its siblings are other tasks' prepared
# workspaces, and the point of the layout is that they survive each other.
cmd_clean() {
    [ -d "$task_run_root" ] || {
        echo "nothing to clean: $task_run_root does not exist"; return; }
    say "removing ${task_run_root#"$repo_root/"}"
    du -sh "$task_run_root" 2>/dev/null || true
    docker run --rm -v "$runs_root:/mnt/runs" "$base_image" \
        rm -rf "/mnt/runs/$task_id"
    echo "gone. Still under $runs_root:"
    ls -1 "$runs_root" 2>/dev/null | sed 's/^/  /'
    echo "the shared caches are untouched: $models_dir, $datasets_dir"
}

cmd_status() {
    local selected=""
    if [ "$(container_state)" = running ]; then
        selected="$(docker exec "$container" cat /opt/ptdbench/current-task 2>/dev/null || true)"
    fi
    # What `run` would start. Read here as well, because a task card without a
    # training.script fails at `run` and that is hours into a session; `status`
    # is where you can still see it coming. Errors are swallowed: `status` has
    # to keep printing the rest.
    local train
    train="$(find_train_script 2>/dev/null || true)"
    if [ -n "$train" ]; then
        load_train_args 2>/dev/null || train_args=()
        train="$train${train_args[*]+ ${train_args[*]}}"
    else
        train="<none: metadata declares no training.script>"
    fi
    cat <<EOF
  task        $task_dir  ($task_id)
  env         $env_name
  train       $train
  base image  $base_image  $(docker image inspect "$base_image" >/dev/null 2>&1 && echo "(present)" || echo "(MISSING: $0 base)")
  container   $container  $(container_state || true)${selected:+  (pointing at: $selected)}
  runs root   $runs_root
  run dir     $task_run_root
  models      $models_dir
  datasets    $datasets_dir
  gpus        ${gpus:-<none>}
  network     ${network:-<default bridge>}
EOF
    # The two prefetch directories, because their absence is silent and
    # expensive: build-env.sh falls back to download.pytorch.org for a missing
    # wheel and to a github clone for a missing mirror, and the only symptom is
    # that `build` takes an extra hour. Counted on the host, where you can
    # actually do something about it -- the fix is a fetch-*.sh run, not
    # anything inside the container.
    local wheels="docker/wheels/$env_name" mirrors="docker/gitmirror/$env_name"
    local n_wheels n_mirrors hint=
    n_wheels="$(find "$wheels" -maxdepth 1 -name '*.whl' 2>/dev/null | wc -l | tr -d ' ')"
    [ "$n_wheels" != 0 ] || hint="  <- empty; torch comes from pytorch.org. ./docker/fetch-cuda-wheels.sh"
    echo "  wheels      $wheels  ($n_wheels .whl)$hint"
    hint=
    n_mirrors="$(grep -c . "$mirrors/manifest.txt" 2>/dev/null || true)"
    [ -n "${n_mirrors:-}" ] || n_mirrors=0
    [ "$n_mirrors" != 0 ] || hint="  <- empty; every git+ pin clones from github. ./docker/fetch-git-mirrors.sh"
    echo "  gitmirror   $mirrors  ($n_mirrors pin(s))$hint"
    if [ "$(container_state)" = running ]; then
        echo "  prefix      $(docker exec "$container" sh -c \
            "test -x /opt/conda/envs/$env_name/bin/python && echo built || echo 'not built yet'")"
    fi
    # Which tasks already have a run directory under this environment, because
    # that is what a switch is free for -- the ones listed here keep their
    # prepared weights, and any other task pays one `prepare` first.
    if [ -d "$runs_root" ]; then
        local dir label="  tasks     "
        for dir in "$runs_root"/*/; do
            [ -d "$dir" ] || continue
            dir="$(basename "$dir")"
            [ "$dir" != logs ] || continue
            if [ "$dir" = "$task_id" ]; then
                echo "$label -> $dir  (selected by PTDBENCH_TASK)"
            else
                echo "$label    $dir"
            fi
            label="            "
        done
    fi
}

# Printed from the block at the top of this file rather than kept as a second
# copy of it. Two hand-maintained lists of the same flags drift, and the one
# that drifts is always the one further from the code -- so there is only one,
# and it is the one you are reading when you open the file.
usage() {
    sed -n '/^# BEGIN usage$/,/^# END usage$/p' "${BASH_SOURCE[0]}" \
        | sed -e '1d' -e '$d' -e 's/^# \{0,1\}//' >&2
    echo >&2
    echo "what this invocation resolves to: $0 status" >&2
    exit 2
}

[ $# -gt 0 ] || usage

for command in "$@"; do
    case "$command" in
        base)    cmd_base ;;
        up)      cmd_up ;;
        build)   cmd_build ;;
        smoke)   cmd_smoke ;;
        use)     cmd_use ;;
        prepare) cmd_prepare ;;
        retriever) cmd_retriever ;;
        run)     cmd_run ;;
        all)     cmd_up; cmd_build; cmd_smoke; cmd_prepare; cmd_run ;;
        shell)   cmd_shell ;;
        logs)    cmd_logs ;;
        status)  cmd_status ;;
        reset)   cmd_reset ;;
        commit)  cmd_commit ;;
        down)    cmd_down ;;
        clean)   cmd_clean ;;
        -h|--help|help) usage ;;
        *)       echo "error: unknown command: $command" >&2; usage ;;
    esac
done
