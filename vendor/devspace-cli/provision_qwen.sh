#!/usr/bin/env bash
# Expose the qwen-code CLI on a devspace, from the payload already on /user.
#
# Same egress + Landlock situation as provision_codex.sh (see its header):
# harbor's qwen_code.install() only stays offline if `command -v qwen` already
# resolves -- otherwise it curls the nvm installer from raw.githubusercontent
# .com, which this cluster's egress whitelist blocks (curl 35). The payload
# must be COPIED under /usr/local (not symlinked from /user).
#
# /usr/local is the container's ephemeral overlay: re-run after every devspace
# start. The payload on /user persists.
set -euo pipefail

PAYLOAD="${1:-/user/linjiayin/qwen-code}"
NODE_SRC="${NODE_SRC:-/user/linjiayin/node-v22.11.0-linux-x64}"

[ -d "$PAYLOAD/node_modules/@qwen-code/qwen-code" ] || {
  echo "missing qwen payload: $PAYLOAD/node_modules/@qwen-code/qwen-code" >&2; exit 1; }
[ -x "$NODE_SRC/bin/node" ] || { echo "missing node: $NODE_SRC/bin/node" >&2; exit 1; }

install -d /usr/local/lib/qwen
# node22 must be a REAL directory under /usr/local (Landlock resolves symlinks
# and only grants exec on /usr); chown because cp -a carries the uid from /user.
if [ -L /usr/local/lib/node22 ] || [ ! -x /usr/local/lib/node22/bin/node ]; then
  rm -rf /usr/local/lib/node22
  cp -a "$NODE_SRC" /usr/local/lib/node22
fi
chown -R root:root /usr/local/lib/node22
# Entry point is package.json's bin: {"qwen": "cli-entry.js"} (v0.25.2; the
# payload predates the bin/ layout codex uses).
cat > /usr/local/bin/qwen <<'EOF'
#!/bin/sh
exec /usr/local/lib/node22/bin/node \
  /usr/local/lib/qwen/node_modules/@qwen-code/qwen-code/cli-entry.js "$@"
EOF
# The directory MUST be called node_modules: the CLI locates its siblings and
# platform binaries via require.resolve through node_modules directories.
rm -rf /usr/local/lib/qwen/node_modules.tmp
cp -a "$PAYLOAD/node_modules" /usr/local/lib/qwen/node_modules.tmp
rm -rf /usr/local/lib/qwen/node_modules
mv /usr/local/lib/qwen/node_modules.tmp /usr/local/lib/qwen/node_modules
chown -R root:root /usr/local/lib/qwen

chmod +x /usr/local/bin/qwen

echo "qwen: $(qwen --version 2>&1 | head -1)"
if command -v llrun >/dev/null 2>&1; then
  # harbor runs the agent under llrun; exit 0 + a version-ish banner means the
  # sandbox exec works. `qwen --version` prints only "0.25.2" (no "qwen"
  # prefix), so grep for a digit rather than the tool name.
  _v=$(llrun --rw /workspace -- qwen --version 2>&1)
  if [ $? -eq 0 ] && echo "$_v" | grep -qE '^[0-9]+\.[0-9]+'; then
    echo "sandboxed: OK"
  else
    echo "SANDBOX CHECK FAILED:" >&2
    echo "$_v" | head -5 >&2
    exit 1
  fi
  unset _v
fi
