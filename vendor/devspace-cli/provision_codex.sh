#!/usr/bin/env bash
# Expose the codex CLI on a devspace, from the payload already on /user.
#
# Same egress situation as claude-code: `npm install -g @openai/codex` cannot
# work here because cdn.npmmirror.com (package contents) and
# raw.githubusercontent.com (the nvm installer) are both outside the whitelist.
# The node_modules tree is installed on a workstation and uploaded to
# /user/<user>/codex-cli once per cluster.
#
# The payload must be COPIED under /usr/local rather than symlinked from /user:
# harbor launches the agent through llrun, whose Landlock ruleset grants
# read+exec on /usr but not on /user, and it checks the resolved target. A
# wrapper pointing into /user fails with "Permission denied" inside the sandbox
# even though codex itself runs with --dangerously-bypass-approvals-and-sandbox
# (that flag only disables codex's *own* approval prompts, not llrun).
#
# /usr/local is the container's ephemeral overlay, so re-run after every
# devspace start; the payload on /user persists.
set -euo pipefail

PAYLOAD="${1:-/user/linjiayin/codex-cli}"
NODE_SRC="${NODE_SRC:-/user/linjiayin/node-v22.11.0-linux-x64}"

[ -d "$PAYLOAD/node_modules/@openai/codex" ] || {
  echo "missing codex payload: $PAYLOAD/node_modules/@openai/codex" >&2; exit 1; }
[ -x "$NODE_SRC/bin/node" ] || { echo "missing node: $NODE_SRC/bin/node" >&2; exit 1; }

install -d /usr/local/lib/codex
# node22 must be a REAL directory under /usr/local, not a symlink into /user:
# llrun's Landlock ruleset grants exec on /usr but not /user, and it resolves
# symlinks, so a linked interpreter fails with "Permission denied" inside the
# sandbox. provision.sh (claude-code) may have left a symlink here, so replace
# it. chown afterwards because cp -a carries the uid from /user (502/1000) and
# a non-root-owned interpreter is likewise refused under llrun.
if [ -L /usr/local/lib/node22 ] || [ ! -x /usr/local/lib/node22/bin/node ]; then
  rm -rf /usr/local/lib/node22
  cp -a "$NODE_SRC" /usr/local/lib/node22
fi
chown -R root:root /usr/local/lib/node22
# The directory MUST be called node_modules: codex.js locates its platform
# binary via require.resolve('@openai/codex-linux-x64/package.json'), which
# only walks node_modules directories. Any other name and it dies with
# "Missing optional dependency @openai/codex-linux-x64".
rm -rf /usr/local/lib/codex/node_modules.tmp
cp -a "$PAYLOAD/node_modules" /usr/local/lib/codex/node_modules.tmp
rm -rf /usr/local/lib/codex/node_modules
mv /usr/local/lib/codex/node_modules.tmp /usr/local/lib/codex/node_modules
chown -R root:root /usr/local/lib/codex

cat > /usr/local/bin/codex <<'EOF'
#!/bin/sh
exec /usr/local/lib/node22/bin/node \
  /usr/local/lib/codex/node_modules/@openai/codex/bin/codex.js "$@"
EOF
chmod +x /usr/local/bin/codex

echo "codex: $(codex --version 2>&1 | head -1)"
if command -v llrun >/dev/null 2>&1; then
  # Load-bearing: harbor runs the agent under llrun, so a codex that only works
  # unsandboxed fails every trial with NonZeroAgentExitCodeError and an empty
  # agent log. Grep for the version rather than reading the first line --
  # codex emits benign warnings first ("could not create PATH aliases" because
  # Landlock denies /root/.codex), then proceeds normally.
  if llrun --rw /workspace -- codex --version 2>&1 | grep -q 'codex-cli'; then
    echo "sandboxed: OK"
  else
    echo "SANDBOX CHECK FAILED:" >&2
    llrun --rw /workspace -- codex --version 2>&1 | head -5 >&2
    exit 1
  fi
fi
