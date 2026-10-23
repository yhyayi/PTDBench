#!/usr/bin/env bash
# Install PTDBench's prepare-stage fetch/dataset tooling into the image's
# /usr/local at runtime, from this directory's copies.
#
# These files used to be COPY'd into every docker/images/*.Dockerfile at build
# time. That baked a snapshot: a fix to ptdbench_fetch.py / ptdbench_dataset.py
# never reached a running job (cluster platform images can't be rebuilt on
# demand), and even locally it needed a full image rebuild. Instead the
# Dockerfiles now ship only the activate hook, and each prepare run invokes this
# installer first so the shims exec the repo's CURRENT tooling, not a stale bake.
#
# Run it with this script sitting next to ptdbench_fetch.py / ptdbench_dataset.py
# / ptdbench_datasets (i.e. the repo's docker/ dir, mounted RO in the local
# prepare container or uploaded to a Cybertron devspace). Overwrites
# unconditionally and is idempotent.
set -eu

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p /usr/local/libexec /usr/local/bin

# --- disable hf_xet --------------------------------------------------------
# huggingface_hub's HF_HUB_DISABLE_XET flag is unreliable (huggingface_hub#3266):
# in 0.34.x the download path keys off `find_spec("hf_xet")`, not the flag, so a
# xet-backed repo still routes through cas-*.xethub.hf.co -- which the cluster's
# whitelist egress proxy does not allow, so every HF fetch hangs/fails even with
# HF_HUB_DISABLE_XET=1 exported (see ptdbench_fetch.py). This hits both the model
# fetch (ptdbench_fetch_hf -> huggingface_hub) and the dataset recipes
# (ptdbench_dataset -> `datasets.load_dataset` -> huggingface_hub).
#
# The only reliable fix is to make `find_spec("hf_xet")` return None. `pip
# uninstall` is the clean way, but it cannot remove an hf_xet that was installed
# by conda/system packaging or that lives in a site-packages `python3 -m pip`
# does not own -- and on those images the earlier pip-only attempt silently left
# hf_xet in place. So: try pip, then if it is STILL importable, delete its files
# directly (these are throwaway prepare containers), then verify. The whole block
# is a no-op once hf_xet is gone, so it costs nothing on later prepares.
if python3 -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("hf_xet") else 1)' 2>/dev/null; then
    python3 -m pip uninstall -y hf-xet >/dev/null 2>&1 || true
    # Fallback: nuke the files wherever the import system still finds them.
    python3 - <<'PY' || true
import importlib.util, shutil
from pathlib import Path

spec = importlib.util.find_spec("hf_xet")
if spec is not None:
    dirs = {str(p) for p in (spec.submodule_search_locations or [])}
    if spec.origin and spec.origin not in ("namespace", "built-in"):
        dirs.add(str(Path(spec.origin).parent))
    for d in dirs:
        p = Path(d)
        if p.name == "hf_xet" and p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        # drop its *.dist-info / *.egg-info so pip/importlib stop advertising it
        for meta in list(p.parent.glob("hf_xet-*.dist-info")) + list(
            p.parent.glob("hf_xet-*.egg-info")
        ):
            shutil.rmtree(meta, ignore_errors=True)
PY
    if python3 -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("hf_xet") else 1)' 2>/dev/null; then
        echo "[install-ptdbench-tooling] WARN: hf_xet still importable after pip+file removal; xet-backed HF fetches may fail" >&2
    else
        echo "[install-ptdbench-tooling] removed hf_xet (hf#3266: HF_HUB_DISABLE_XET unreliable)"
    fi
fi

# --- pinlib hook ------------------------------------------------------------
# Tasks whose pinned assets were generated under a newer data stack than the
# devspace image's (e.g. qwen3vl_geo3k's parquet/PIL pins embed pyarrow 26 /
# datasets 4.4.1 / Pillow 13) need that exact stack importable by prepare's
# embedded python. The prepare recipe forces the parquet-writing subprocess's
# PYTHONPATH to the workspace verl checkout, so an env-var injection cannot
# reach it; a .pth in system site-packages can. The stack itself lives on the
# persistent shared fs (/user), only the one-line hook goes in the ephemeral
# /usr/local. Opt-in per task via prepare_env PTDBENCH_PINLIB=<dir>; with it
# unset any stale hook is removed so other tasks keep the image's own stack.
HOOK_SITE="$(python3 -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)"
if [ -n "${PTDBENCH_PINLIB:-}" ]; then
    if [ -d "$PTDBENCH_PINLIB" ] && [ -n "$HOOK_SITE" ]; then
        printf 'import sys; sys.path.insert(0, "%s")\n' "$PTDBENCH_PINLIB" \
            > "$HOOK_SITE/00-ptdbench-pinlib.pth"
        echo "[install-ptdbench-tooling] pinlib hook -> $HOOK_SITE/00-ptdbench-pinlib.pth"
    else
        echo "[install-ptdbench-tooling] WARN: PTDBENCH_PINLIB=$PTDBENCH_PINLIB not found; pins may fail" >&2
    fi
elif [ -n "$HOOK_SITE" ] && [ -f "$HOOK_SITE/00-ptdbench-pinlib.pth" ]; then
    rm -f "$HOOK_SITE/00-ptdbench-pinlib.pth"
    echo "[install-ptdbench-tooling] removed stale pinlib hook"
fi

# --- agent CLI staging (qwen-coder) -----------------------------------------
# qwen_code.py's install() bootstraps node through NVM from
# raw.githubusercontent.com and then `npm install -g` pulls the CLI tarball
# from cdn.npmmirror.com; neither host is on the devspace egress whitelist
# (curl 35 / ECONNRESET), so that install path cannot succeed on a devspace.
# The persistent shared fs carries a pre-assembled bundle (outputs/
# activate_qwen.sh documents how it was built); stage it like the pinlib
# hook: symlink into the ephemeral /usr/local during prepare, opt-in per
# task via prepare_env PTDBENCH_QWEN_CODE=<dir>. Unset -> no-op.
QWEN_PREFIX="${PTDBENCH_QWEN_CODE:-}"
if [ -n "$QWEN_PREFIX" ]; then
    if [ -x "$QWEN_PREFIX/node_modules/.bin/qwen" ]; then
        ln -sfn "$QWEN_PREFIX/node_modules/.bin/qwen" /usr/local/bin/qwen
        echo "[install-ptdbench-tooling] qwen-code -> /usr/local/bin/qwen"
        # cli-entry.js starts with `#!/usr/bin/env node`; a devspace that has
        # never run a node-based agent also needs a resolvable node.
        if ! command -v node >/dev/null 2>&1; then
            NODE_BIN=/user/linjiayin/node-v22.11.0-linux-x64/bin/node
            if [ -x "$NODE_BIN" ]; then
                ln -sfn "$NODE_BIN" /usr/local/bin/node
            else
                echo "[install-ptdbench-tooling] WARN: no node at $NODE_BIN for qwen-code" >&2
            fi
        fi
    else
        echo "[install-ptdbench-tooling] WARN: PTDBENCH_QWEN_CODE=$QWEN_PREFIX has no qwen CLI" >&2
    fi
fi

# --- fetch -----------------------------------------------------------------
# The shims resolve python3 at call time, so they run under the image's own
# interpreter and its huggingface_hub / datasets (same as the old Dockerfile).
cp "$DIR/ptdbench_fetch.py" /usr/local/libexec/ptdbench_fetch.py
for kind in hf url; do
    printf '#!/bin/sh\nexec python3 /usr/local/libexec/ptdbench_fetch.py %s "$@"\n' \
        "${kind}" > "/usr/local/bin/ptdbench_fetch_${kind}"
    chmod 0755 "/usr/local/bin/ptdbench_fetch_${kind}"
done

# --- dataset ---------------------------------------------------------------
cp "$DIR/ptdbench_dataset.py" /usr/local/libexec/ptdbench_dataset.py
rm -rf /usr/local/libexec/ptdbench_datasets
cp -a "$DIR/ptdbench_datasets" /usr/local/libexec/ptdbench_datasets
printf '#!/bin/sh\nexec python3 /usr/local/libexec/ptdbench_dataset.py "$@"\n' \
    > /usr/local/bin/ptdbench_dataset
chmod 0755 /usr/local/bin/ptdbench_dataset

echo "[install-ptdbench-tooling] installed fetch/dataset tooling from $DIR"
