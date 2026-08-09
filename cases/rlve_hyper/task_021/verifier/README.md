# PTDBench standalone task verifier

This directory contains the complete runtime needed to evaluate one PTDBench
case. It runs an Agent, collects a structured diagnosis, asks an independent
LLM Judge to compare that diagnosis with the case oracle, and writes a binary
score plus a complete execution trace.

No repository-level Python package is required. Keep this directory beside the
case inputs:

```text
case/
├── instruction.md
├── workspace/
├── oracle/
│   ├── standard_answer.txt
│   └── task.yaml
└── verifier/
    ├── test.sh
    ├── run_evaluation.py
    ├── agent.Dockerfile
    ├── agent_prompt.txt
    ├── judge_prompt.txt
    └── README.md
```

## Evaluation pipeline

The verifier supports two execution paths:

- `standalone`: this verifier runs both Agent A and Judge B.
- `judge-only`: the central PTDBench evaluator runs Agent A, and this verifier
  grades the diagnosis already present in the staged workspace.

`--evaluation-mode auto` is the default. It selects `judge-only` only when the
central evaluator supplied both a non-empty staged `outputs/diagnosis.txt` and
`agent_trace.json`, proving that central Agent A already ran. If central
PTDBench did not run Agent A, auto selects `standalone`, so this task verifier
runs Agent A and Judge B without requiring `--evaluation-mode standalone`.

One invocation performs the following steps:

1. Create a timestamped run directory.
2. Copy either the original case workspace (`standalone`) or
   `$PTDBENCH_WORKSPACE` (`judge-only`) into that run directory.
3. In standalone mode, reuse or acquire the Agent Docker image.
4. Run Agent A locally through central PTDBench or in the standalone isolated
   container.
5. Require Agent A to write `workspace/outputs/diagnosis.txt`.
6. Read `oracle/standard_answer.txt` and the allowed location context from
   `oracle/task.yaml` on the host. The oracle is never mounted into Agent A's
   container.
7. Call Judge B directly over HTTP and require a strict binary JSON decision.
8. Write `metrics.json`, `result.json`, and the full Agent/Judge trajectory.

Agent A may modify any file in the staged workspace. The original case
workspace is not modified.

## Requirements

- Linux or macOS with Bash
- Python 3.8 or newer
- For standalone mode, a running Docker Engine with permission to run
  `docker build` and `docker run`
- For the first standalone run, internet access to download the Docker base
  image and Agent CLI packages
- API credentials for Agent A and Judge B

Confirm the local prerequisites:

```bash
python3 --version
docker version
docker info
```

If the executable bit was lost while downloading an archive:

```bash
chmod +x test.sh
```

`bash test.sh ...` also works without changing the executable bit.

## Clone and run

From a cloned repository, enter one case's verifier directory:

```bash
git clone <repository-url>
cd <repository>/cases/<family>/<case>/verifier
```

Run the local self-test first. It does not build an image, start a container,
or call an external API:

```bash
./test.sh --self-test
```

### Complete DeepSeek example

The following example uses DeepSeek through the Claude Code harness for Agent A
and the OpenAI-compatible DeepSeek endpoint for Judge B:

```bash
export DEEPSEEK_API_KEY='<your-api-key>'

./test.sh \
  --agent-harness claude \
  --agent-model deepseek-v4-pro \
  --agent-api-key-env DEEPSEEK_API_KEY \
  --agent-base-url https://api.deepseek.com/anthropic \
  --judge-provider openai-compatible \
  --judge-model deepseek-v4-pro \
  --judge-api-key-env DEEPSEEK_API_KEY \
  --judge-base-url https://api.deepseek.com
```

Do not place a key value in a command-line option. The
`--agent-api-key-env` and `--judge-api-key-env` options take environment
variable names, not secrets.

No `--agent-image` argument is needed in the normal workflow. On the first run,
if `ptdbench-agent-harness:latest` is not local, the verifier automatically
builds it from this directory's `agent.Dockerfile`. Later runs reuse that local
image. Concurrent verifier processes use a host lock so that only one process
builds or pulls a given image.

### Configure once, then run `test.sh` directly

Every normal CLI setting also has an environment-variable form:

```bash
export DEEPSEEK_API_KEY='<your-api-key>'
export PTDBENCH_AGENT_HARNESS=claude
export PTDBENCH_AGENT_MODEL=deepseek-v4-pro
export PTDBENCH_AGENT_API_KEY_ENV=DEEPSEEK_API_KEY
export PTDBENCH_AGENT_BASE_URL=https://api.deepseek.com/anthropic
export PTDBENCH_JUDGE_PROVIDER=openai-compatible
export PTDBENCH_JUDGE_MODEL=deepseek-v4-pro
export PTDBENCH_JUDGE_API_KEY_ENV=DEEPSEEK_API_KEY
export PTDBENCH_JUDGE_BASE_URL=https://api.deepseek.com

./test.sh
```

CLI options override the corresponding environment variables.

## Central PTDBench workflow

Run schema validation from the repository root, not by executing `schema.py`
directly:

```bash
cd <repository>

PYTHONDONTWRITEBYTECODE=1 \
python3 -m ptdbench validate cases/<family>/<case>
```

A valid package returns exit code `0` and:

```json
{
  "ok": true,
  "issues": []
}
```

For an OpenAI Codex Agent and an OpenAI-compatible Judge using the same model,
export the API key and use the normal central command:

```bash
export OPENAI_API_KEY='<your-api-key>'

PYTHONDONTWRITEBYTECODE=1 \
python3 -m ptdbench evaluate cases/<family>/<case> \
  --backend local \
  --agent-backend local \
  --agent codex \
  --model gpt-5.5 \
  --base-url https://api.openai.com/v1 \
  --api-key "$OPENAI_API_KEY" \
  --json
```

Central PTDBench deletes its temporary run directory by default. To retain the
complete central Agent trace, staged workspace, Judge trace, and reports, append:

```bash
--keep-artifacts --work-dir /absolute/writable/ptdbench-runs
```

The central evaluator runs Agent A first. Agent A must write:

```text
$PTDBENCH_WORKSPACE/outputs/diagnosis.txt
```

Before invoking the verifier, central PTDBench writes
`$PTDBENCH_RESULTS/agent_trace.json`. With `--backend local`, it also forwards
the central `--model`, `--base-url`, and `--api-key` to the verifier subprocess
as Judge defaults. Auto mode can fall back to the trace and parent environment
when an option was omitted. Judge B uses the Agent model unless
`PTDBENCH_JUDGE_MODEL` overrides it.

For better process-list hygiene, expand the key from a protected environment
variable instead of writing a literal secret:

```bash
export OPENAI_API_KEY='<your-api-key>'
```

To use a different Judge:

```bash
export PTDBENCH_JUDGE_PROVIDER=openai-compatible
export PTDBENCH_JUDGE_MODEL='<judge-model>'
export PTDBENCH_JUDGE_API_KEY_ENV=JUDGE_API_KEY
export PTDBENCH_JUDGE_BASE_URL='https://judge.example/v1'
export JUDGE_API_KEY='<judge-api-key>'
```

Central integration currently requires `--backend local`. The central Docker
backend does not forward host Judge secrets and normally disables verifier
network access. Judge-only mode never starts a nested Agent Docker container.

The verifier consumes the central contract variables:

```text
PTDBENCH_WORKSPACE
PTDBENCH_ORACLE
PTDBENCH_VERIFIER
PTDBENCH_RESULTS
PTDBENCH_CASE_ID
```

It writes the PTDBench contract outputs to:

```text
$PTDBENCH_RESULTS/result.json
$PTDBENCH_RESULTS/metrics.json
```

Exit code `0` means the validated binary Judge score is `1`. Score `0` or a
runtime error exits nonzero.

## Docker image behavior

The default image name is:

```text
ptdbench-agent-harness:latest
```

The verifier applies this policy:

- If the requested image already exists locally, reuse it.
- If the missing image is the default image, build it from
  `verifier/agent.Dockerfile`.
- If a missing custom image is supplied with `--agent-image`, pull it from its
  registry.
- If image acquisition fails, write failure artifacts and return a zero score.

To build or refresh the default image manually:

```bash
docker build --pull \
  --file agent.Dockerfile \
  --tag ptdbench-agent-harness:latest \
  .
```

To use an existing compatible local image:

```bash
./test.sh \
  --agent-image final-rl-agent:latest \
  <the remaining Agent and Judge options>
```

To use an image published in GitHub Container Registry:

```bash
docker pull ghcr.io/<owner>/ptdbench-agent-harness:v1

./test.sh \
  --agent-image ghcr.io/<owner>/ptdbench-agent-harness:v1 \
  <the remaining Agent and Judge options>
```

For a private GHCR package, authenticate before running:

```bash
printf '%s' "$GHCR_TOKEN" | \
  docker login ghcr.io --username <github-user> --password-stdin
```

The Dockerfile installs fixed versions of the supported Agent CLIs:

- Claude Code
- OpenAI Codex
- Gemini CLI
- OpenCode

## Agent harnesses

Select one Agent CLI with `--agent-harness`:

| Harness | Value | API base URL variable set inside the container |
| --- | --- | --- |
| OpenAI Codex | `codex` | `OPENAI_BASE_URL` |
| Claude Code | `claude` | `ANTHROPIC_BASE_URL` |
| Gemini CLI | `gemini` | `GOOGLE_GEMINI_BASE_URL` |
| OpenCode | `opencode` | `OPENAI_BASE_URL` |

The verifier maps the secret named by `--agent-api-key-env` to the variables
expected by the selected CLI. The environment-variable name is visible in the
run configuration, but its value is not serialized.

## Judge providers

Judge B is called directly by the host Python process. Supported values for
`--judge-provider` are:

- `openai-compatible`
- `anthropic`
- `gemini`

Endpoint construction:

- OpenAI-compatible: appends `/chat/completions` unless already present.
- Anthropic: appends `/v1/messages` unless already present.
- Gemini: appends `/models/<model>:generateContent`, or substitutes `{model}`
  in a custom base URL.

## CLI reference

```text
--evaluation-mode     auto | standalone | judge-only
--agent-harness       codex | claude | gemini | opencode
--agent-model         model name passed to Agent A
--agent-image         Docker image; defaults to ptdbench-agent-harness:latest
--agent-api-key-env   parent environment-variable name containing Agent key
--agent-base-url      optional API base URL for the Agent harness
--agent-network       Docker network; defaults to bridge
--agent-timeout       positive seconds, or 0/unset for unlimited

--judge-provider      openai-compatible | anthropic | gemini
--judge-model         model name passed to Judge B
--judge-api-key-env   parent environment-variable name containing Judge key
--judge-base-url      optional Judge API base URL
--judge-timeout       positive seconds, or 0/unset for unlimited

--run-id              optional safe output directory name
--self-test           local contract test; no Docker or API usage
```

The matching environment variables are:

```text
PTDBENCH_EVALUATION_MODE
PTDBENCH_AGENT_HARNESS
PTDBENCH_AGENT_MODEL
PTDBENCH_AGENT_IMAGE
PTDBENCH_AGENT_API_KEY_ENV
PTDBENCH_AGENT_BASE_URL
PTDBENCH_AGENT_NETWORK
PTDBENCH_AGENT_TIMEOUT
PTDBENCH_JUDGE_PROVIDER
PTDBENCH_JUDGE_MODEL
PTDBENCH_JUDGE_API_KEY_ENV
PTDBENCH_JUDGE_BASE_URL
PTDBENCH_JUDGE_TIMEOUT
PTDBENCH_WORKSPACE
PTDBENCH_ORACLE
PTDBENCH_VERIFIER
PTDBENCH_RESULTS
PTDBENCH_CASE_ID
```

## Timeouts

Agent and Judge execution are unlimited by default. Positive values retain
compatibility with deadline-based execution:

```bash
./test.sh \
  --agent-timeout 1800 \
  --judge-timeout 180 \
  <the remaining options>
```

`0` means unlimited.

## Results and complete trajectories

Standalone mode creates one directory directly under `verifier/`:

```text
<sanitized-agent-model>-<harness>-<UTC timestamp>/
```

Example:

```text
deepseek-v4-pro-claude-20260808T153000123456Z/
```

The directory contains:

```text
workspace/                       staged workspace modified by Agent A
workspace/outputs/diagnosis.txt  Agent A's scored diagnosis
agent_prompt.txt                 fully assembled Agent prompt
agent_stdout.txt
agent_stderr.txt
agent_trace.txt
external_agent_trace.json        central Agent trace in judge-only mode
agent_status.json
agent_image_status.json          reused, built, or pulled image status
agent_image_stdout.txt           Docker build/pull stdout
agent_image_stderr.txt           Docker build/pull stderr
judge_request.json               redacted request payload
judge_raw_response.txt
judge_result.json                validated Judge decision
metrics.json                     benchmark metric and Agent return code
result.json                      pass/fail result
run_config.json                  public runtime configuration
error.txt                        empty on success; failure detail otherwise
```

Inspect the newest run:

```bash
latest="$(ls -dt -- */ | sed -n '1p')"
printf 'Run directory: %s\n' "$latest"
python3 -m json.tool "${latest}/result.json"
python3 -m json.tool "${latest}/metrics.json"
python3 -m json.tool "${latest}/judge_result.json"
cat "${latest}/workspace/outputs/diagnosis.txt"
```

The process exits with code `0` only when the validated Judge score is `1`.
Score `0` and runtime failures exit with code `1`; inspect `result.json` and
`error.txt` to distinguish them.

## External writable result directory

For a read-only case package, place all runtime artifacts elsewhere:

```bash
export PTDBENCH_RESULTS=/absolute/writable/results

./test.sh <the remaining options>
```

The complete run is written to:

```text
$PTDBENCH_RESULTS/<run-id>/
```

PTDBench contract copies are also written to:

```text
$PTDBENCH_RESULTS/metrics.json
$PTDBENCH_RESULTS/result.json
```

## Isolation and secret handling

- In standalone mode, Agent A receives only the staged workspace,
  `instruction.md`, and its assembled prompt.
- In central mode, Agent A is isolated by PTDBench before this verifier starts.
- `oracle/` is never exposed to Agent A.
- The container drops Linux capabilities, enables `no-new-privileges`, and
  uses a temporary home directory.
- API key values are passed by environment, never by CLI argument.
- Known secret values are redacted from persisted stdout, stderr, requests,
  responses, and errors.
- `run_config.json` records API-key environment-variable names only.

Treat the Docker daemon, the selected model providers, and generated Agent
content as trusted infrastructure. Do not commit API keys to the repository.

## Troubleshooting

### Docker permission denied

Verify that the current user can access the Docker daemon:

```bash
docker info
```

Configure Docker access for the user or run the verifier in an environment
where Docker is available. Avoid running the whole verifier as root unless the
surrounding environment requires it.

### Default image build failed

Inspect:

```text
agent_image_status.json
agent_image_stdout.txt
agent_image_stderr.txt
error.txt
```

The first build requires network access to the base image registry, Debian
package mirrors, and npm.

### Custom image pull failed

Confirm the image reference and registry authentication:

```bash
docker login <registry>
docker pull <image-reference>
```

### Agent completed without a diagnosis

Inspect `agent_status.json`, `agent_trace.txt`, and `agent_stderr.txt`. A valid
run must create a non-empty regular file at:

```text
workspace/outputs/diagnosis.txt
```

### Judge returned invalid JSON

The verifier retries one time with a format-correction prompt. If both attempts
fail, inspect `judge_raw_response.txt`, `judge_request.json`, and `error.txt`.
