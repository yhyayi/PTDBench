"""Remote PTDBench ``prepare.sh`` execution on a Cybertron devspace.

PTDBench's prepare stage is the only networked stage: it downloads model weights
and datasets and, when a GPU is present, converts HF weights to mcore/torch_dist
in place. The local-docker eval path runs it in a container and uploads the
result (``ptdbench/evaluation/harbor_evaluator.py``), which is unusable against a
Cybertron devspace for three reasons:

* the ptdbench images are x86-64 CUDA images, so the prepare container cannot run
  on an arm64 workstation at all;
* ``CybertronEnvironment`` declares ``mounted=False`` -- the prepared dir, model
  cache and dataset cache reach the devspace as paths the backend creates, not as
  bind mounts, so there is nothing for a locally prepared tree to be mounted into;
* ``upload_dir`` tars the whole tree in memory, so shipping tens of GB of weights
  over the WAN per trial is not viable.

Running prepare *on* the devspace solves all three: weights are fetched straight
into the shared cluster cache at near-LAN speed, the GPU needed for the
torch_dist conversion is the trial's own, and nothing large crosses the WAN.

The shell built here is the same one ``scripts/cybertron_stage.py`` has used for
the manual staging path; it lives in the package so ``CybertronEnvironment`` can
reach it without importing from ``scripts/``. ``scripts/cybertron_preheat.py``
keeps its own copies of the export helpers on purpose: the runbook invokes it
with a bare ``python``, so it must stay importable without the harbor package.
"""

from __future__ import annotations

import shlex

# Stable in-container paths that ``CybertronEnvironment._setup_devspace``
# creates or symlinks. ``/workspace`` points at the per-run workspace on the
# persistent ``/user`` filesystem; ``/ptdbench/{models,datasets}`` point at the
# shared cluster cache. prepare writes weights into the workspace and reads the
# cache from ``/ptdbench/*``.
#
# ``prepared`` and ``environment`` deliberately sit *under* ``/workspace``
# rather than at PTDBench's canonical ``/ptdbench/{prepared,environment}``:
# ``/workspace`` is the symlink to the persistent filesystem, while
# ``/ptdbench`` is the ephemeral container rootfs.
REMOTE_WORKSPACE = "/workspace"
REMOTE_MODELS = "/ptdbench/models"
REMOTE_DATASETS = "/ptdbench/datasets"
REMOTE_PREPARED = "/workspace/.ptdbench-prepared"
REMOTE_ENVIRONMENT = "/workspace/.ptdbench-environment"

DEFAULT_HTTP_PROXY = "http://whitelist-proxy.cybertron.svc.cluster.local:7891"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"

_NO_PROXY = "localhost,127.0.0.1,.modelbest.co,.svc.cluster.local"

# The activate hook body, written into the prepared dir by the prepare recipe and
# auto-sourced by every non-login `bash -c` via BASH_ENV (the task.toml points
# BASH_ENV at $PTDBENCH_PREPARED/ptdbench-activate.sh). It exports what prepare.sh
# left in $PTDBENCH_PREPARED/env (only bare NAME=…, to block command substitution)
# and sets the pip --user base for the agent/rerun stages. No PATH restore: Harbor
# runs non-login shells, so /etc/profile never wipes the image's ENV PATH.
#
# KEEP IN SYNC with ptdbench/runtimes/base.py:PTDBENCH_ACTIVATE_BODY (the Docker
# backend injects that copy inline; this copy serves the Harbor/cybertron backend).
PTDBENCH_ACTIVATE_BODY = r"""if [ -n "${PTDBENCH_STAGE:-}" ]; then
    PTDBENCH_PREPARED="${PTDBENCH_PREPARED:-/ptdbench/prepared}"
    export PTDBENCH_PREPARED
    if [ -f "$PTDBENCH_PREPARED/env" ]; then
        while IFS= read -r _ptdbench_line || [ -n "$_ptdbench_line" ]; do
            case "$_ptdbench_line" in
                ''|'#'*) continue ;;
            esac
            _ptdbench_key="${_ptdbench_line%%=*}"
            case "$_ptdbench_key" in
                ''|*[!A-Za-z0-9_]*)
                    echo "ptdbench: ignoring malformed line in prepared/env: $_ptdbench_line" >&2
                    continue ;;
            esac
            export "$_ptdbench_line"
        done < "$PTDBENCH_PREPARED/env"
        unset _ptdbench_line _ptdbench_key
    fi
    if [ "$PTDBENCH_STAGE" = "agent" ] || [ "$PTDBENCH_STAGE" = "rerun" ]; then
        PYTHONUSERBASE="${PYTHONUSERBASE:-/workspace/.ptdbench-user}"
        PIP_USER=1
        PIP_BREAK_SYSTEM_PACKAGES=1
        export PYTHONUSERBASE PIP_USER PIP_BREAK_SYSTEM_PACKAGES
    fi
fi
"""


def proxy_export(http_proxy: str | None) -> str:
    """Shell ``export http(s)_proxy=… no_proxy=…;`` for the whitelist egress proxy.

    Cybertron nodes only reach the HF mirror through this proxy, and neither a
    BATCH ``--entry`` nor a non-interactive ssh inherits it from Teleport.
    Exported first so even DNS/connect for the mirror goes through it. Empty
    string when no proxy is configured.
    """
    if not http_proxy:
        return ""
    quoted = shlex.quote(http_proxy)
    return (
        f"export http_proxy={quoted} https_proxy={quoted} "
        f"HTTP_PROXY={quoted} HTTPS_PROXY={quoted} "
        f"no_proxy={_NO_PROXY} NO_PROXY={_NO_PROXY}; "
    )


def hf_export(hf_endpoint: str | None) -> str:
    """Shell ``export HF_ENDPOINT=… HF_HUB_DISABLE_XET=1;`` for the HF mirror.

    huggingface_hub reads HF_ENDPOINT at runtime; xet is disabled because the
    mirror does not serve it. Empty string when no endpoint is configured.
    """
    if not hf_endpoint:
        return ""
    return f"export HF_ENDPOINT={shlex.quote(hf_endpoint)} HF_HUB_DISABLE_XET=1; "


def build_prepare_recipe(
    *,
    workspace: str = REMOTE_WORKSPACE,
    models: str = REMOTE_MODELS,
    datasets: str = REMOTE_DATASETS,
    prepared: str = REMOTE_PREPARED,
    environment: str = REMOTE_ENVIRONMENT,
    script: str = "prepare.sh",
    http_proxy: str | None = None,
    hf_endpoint: str | None = None,
    task_env: dict[str, str] | None = None,
) -> str:
    """Shell that runs the uploaded task's prepare.sh.

    Runs on the devspace against the shared cache at ``/ptdbench/{models,datasets}``.
    Weights land in ``$PTDBENCH_WORKSPACE`` (=/workspace); the tiny ``prepared/env``
    file lands under it. The fetch/dataset tools ride along in the task's uploaded
    ``environment/`` and prepare.sh runs them directly via ``$PTDBENCH_ENVIRONMENT``.

    ``task_env`` carries the task's own ``[environment].env`` (``CUDA_VISIBLE_DEVICES``,
    ``PTDBENCH_PIP_INDEX_URL``, …). It is exported *before* the PTDBENCH_* paths
    on purpose: a generated Harbor task.toml also pins the *container* layout
    (``PTDBENCH_PREPARED=/ptdbench/prepared`` and friends), and letting that win
    here would point prepare at the ephemeral container rootfs instead of the
    persistent workspace this backend guarantees. A task can still override the
    proxy and mirror, which are exported earlier still.
    """
    task_exports = ""
    if task_env:
        pairs = " ".join(
            f"{key}={shlex.quote(str(value))}"
            for key, value in sorted(task_env.items())
        )
        task_exports = f"export {pairs}; "
    return (
        "set -euo pipefail; "
        "export PTDBENCH_STAGE=prepare WANDB_MODE=offline WANDB_SILENT=true; "
        f"{proxy_export(http_proxy)}"
        f"{hf_export(hf_endpoint)}"
        f"{task_exports}"
        # prepare is the ONLY networked stage. The generated task.toml carries the
        # agent/verifier offline switches (HF_HUB_OFFLINE etc.) and task_env above
        # replays them here, which is correct for docker -- there prepare finishes
        # before `harbor run` -- but wrong for this backend, where prepare runs
        # inside start(). Left set, huggingface_hub raises OfflineModeIsEnabled on
        # any repo that is not already cached, so a cold cache can never warm up.
        "unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE; "
        f"export PTDBENCH_WORKSPACE={shlex.quote(workspace)} "
        f"PTDBENCH_ENVIRONMENT={shlex.quote(environment)} "
        f"PTDBENCH_PREPARED={shlex.quote(prepared)} "
        f"PTDBENCH_MODELS={shlex.quote(models)} "
        f"PTDBENCH_DATASETS={shlex.quote(datasets)}; "
        f"mkdir -p {shlex.quote(prepared)}; "
        # Write the activate hook next to prepare's env output. The Dockerfiles no
        # longer bake it; agent/verifier auto-source it via BASH_ENV (non-login).
        f"cat > {shlex.quote(prepared + '/ptdbench-activate.sh')} "
        f"<<'PTDBENCH_ACTIVATE_EOF'\n{PTDBENCH_ACTIVATE_BODY}PTDBENCH_ACTIVATE_EOF\n"
        f"cd {shlex.quote(environment)}; "
        f"bash {shlex.quote(script)}"
    )
