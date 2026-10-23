#!/usr/bin/env bash
# Provision node + the Claude Code CLI on a Cybertron devspace.
#
# Run ON the devspace, after the two archives in this directory have been copied
# to $PAYLOAD (see README.md). Idempotent: safe to re-run on every devspace start.
#
#   bash provision.sh [PAYLOAD_DIR]
#
# PAYLOAD_DIR defaults to /user/$USER_NAME/devspace-cli, i.e. the per-cluster
# persistent disk, so the archives survive a devspace being stopped.
set -euo pipefail

PAYLOAD="${1:-/user/${CLUSTER_USER:-$(id -un)}/devspace-cli}"
NODE_TARBALL="$PAYLOAD/node-v22.11.0-linux-x64.tar.xz"
CC_TARBALL="$PAYLOAD/claude-code-2.1.300-linux-x64.tar.gz"
NODE_DIR="$PAYLOAD/node-v22.11.0-linux-x64"
CC_DIR="$PAYLOAD/claude-code"

# --- extract onto the persistent disk ---------------------------------------
# Each half needs its archive only until it has been extracted once; the
# extracted tree is what persists, so an already-provisioned cluster does not
# need the archives kept around.
if [ ! -x "$NODE_DIR/bin/node" ]; then
  [ -f "$NODE_TARBALL" ] || { echo "missing payload: $NODE_TARBALL (see README.md)" >&2; exit 1; }
  tar -xJf "$NODE_TARBALL" -C "$PAYLOAD"
fi
if [ ! -d "$CC_DIR/node_modules" ]; then
  [ -f "$CC_TARBALL" ] || { echo "missing payload: $CC_TARBALL (see README.md)" >&2; exit 1; }
  mkdir -p "$CC_DIR"
  tar -xzf "$CC_TARBALL" -C "$CC_DIR"
fi

# --- stage into /usr/local so the agent sandbox can execute it ---------------
# llrun's Landlock ruleset grants read+exec on /usr, /lib, /bin, /etc, /opt --
# but NOT on /user, and it checks the RESOLVED target, so a symlink from
# /usr/local/bin into /user is denied inside the sandbox. The binaries therefore
# have to be copied under /usr/local, which is the container's ephemeral overlay
# -- hence re-running this after every devspace start.
install -d /usr/local/lib
cp -a "$CC_DIR/node_modules/@anthropic-ai/claude-code-linux-x64" \
      /usr/local/lib/claude-code
chmod +x /usr/local/lib/claude-code/claude
ln -sfn /usr/local/lib/claude-code/claude /usr/local/bin/claude

cp -a "$NODE_DIR" /usr/local/lib/node22
ln -sfn /usr/local/lib/node22/bin/node /usr/local/bin/node
ln -sfn /usr/local/lib/node22/bin/npm /usr/local/bin/npm

# Package *metadata* is reachable via registry.npmmirror.com; package *content*
# is not (it redirects to cdn.npmmirror.com, which the egress whitelist blocks).
# Set the registry anyway so anything that only reads metadata behaves.
npm config set registry https://registry.npmmirror.com >/dev/null 2>&1 || true

echo "node:   $(node -v)"
echo "claude: $(claude --version)"
if command -v llrun >/dev/null 2>&1; then
  echo "sandboxed: $(llrun --rw /workspace -- claude --version)"
fi
