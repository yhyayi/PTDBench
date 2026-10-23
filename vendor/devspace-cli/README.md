# devspace-cli — agent CLI payload for Cybertron devspaces

Vendored binaries that let a Harbor agent run on a Cybertron devspace. They are
here because **the devspace cannot download them itself**.

## Why these are vendored

A devspace has no direct egress at all; everything goes through
`whitelist-proxy.cybertron.svc.cluster.local:7891`, and that whitelist is narrow.
Measured from a devspace on `sichuan_train`:

| host | result |
|---|---|
| `hf-mirror.com` | 200 — this is why `prepare.sh` works |
| `registry.npmmirror.com` (package **metadata**) | 200 |
| `cdn.npmmirror.com` (package **content**) | blocked — every tarball redirects here |
| `npmmirror.com/mirrors/node` | blocked |
| `raw.githubusercontent.com` (nvm installer) | blocked |
| `downloads.claude.ai` (claude-code installer) | blocked |
| `llm-center.modelbest.co` | reachable — the model gateway itself is fine |

The images also ship **no node/npm and no system package manager**, so Harbor's
own agent installers cannot bootstrap:

- `claude-code` uses npm only on Alpine; elsewhere it curls `downloads.claude.ai`
- `codex` curls the nvm installer from GitHub first

`npm install` is no help either: metadata resolves, then the tarball fetch dies
with `ECONNRESET` against `cdn.npmmirror.com`.

So the CLI is built on a workstation (which does have egress) and shipped here.

## Contents

| file | sha256 |
|---|---|
| `node-v22.11.0-linux-x64.tar.xz` | `83bf07dd343002a26211cf1fcd46a9d9534219aad42ee02847816940bf610a72` |
| `claude-code-2.1.300-linux-x64.tar.gz` | `13da2a4f65772917814436a1d51729b800af14de00feaa47abb1877d344e30ab` |

The claude-code archive is an `npm install` tree built for the devspace's
platform rather than the workstation's:

```bash
npm install @anthropic-ai/claude-code \
  --os=linux --cpu=x64 --libc=glibc \
  --registry=https://registry.npmmirror.com
tar czf claude-code-<version>-linux-x64.tar.gz node_modules
```

The `--os/--cpu/--libc` overrides matter: the real binary lives in the
platform-specific optional dependency `@anthropic-ai/claude-code-linux-x64`, and
a plain install on macOS would fetch the darwin one.

## Usage

`/user` is a **per-cluster** persistent disk (`filesystemId` 339 on
sichuan_train, 43 on huhehaote, 218 on wuxi_train — they are separate
filesystems), so the payload has to be uploaded once per cluster, and it then
survives devspaces being stopped.

```bash
# once per cluster
tsh ssh --cluster=<teleport> root@<devspace-host> \
  'mkdir -p /user/<user>/devspace-cli'
for f in node-v22.11.0-linux-x64.tar.xz claude-code-2.1.300-linux-x64.tar.gz; do
  cat "$f" | tsh ssh --cluster=<teleport> root@<devspace-host> \
    "cat > /user/<user>/devspace-cli/$f"
done

# after every devspace start (the /usr/local half is ephemeral)
cat provision.sh | tsh ssh --cluster=<teleport> root@<devspace-host> \
  'bash -s /user/<user>/devspace-cli'
```

Once `claude` is on PATH, Harbor's installer is skipped entirely — its
`_installed_claude_satisfies_version` check is just `command -v claude`
(`harbor/src/harbor/agents/installed/claude_code.py`).

`provision.sh` copies the binaries under `/usr/local` rather than symlinking to
`/user`: the agent runs inside `llrun`'s Landlock sandbox, which grants
read+exec on `/usr` but not `/user`, and checks the *resolved* target.
