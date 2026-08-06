# PTDBench

PostTrainDiagnosisBench, or PTDBench, is a benchmark framework for community
submission of broken post-train tasks, environment specs, and verifier logic.

## Core idea

1. A case package provides a broken `workspace/` plus public instructions.
2. A specified model+agent edits a copied workspace.
3. The evaluator runs `verifier/test.sh` against the edited workspace.
4. The verifier writes structured results to `$PTDBENCH_RESULTS/result.json`.

## Case layout

```text
example/empty/
├── instruction.md
├── readme.md
├── metadata.yaml
├── workspace/
├── oracle/
├── environment/
│   └── Dockerfile
└── verifier/
    ├── test.sh
    ├── tests/
    └── resources/
```

## Visibility

Agent-visible:

- `instruction.md`
- a writable copy of `workspace/`
- the Docker image built from `environment/Dockerfile`

Verifier-only:

- `oracle/`
- `verifier/`
- `verifier/tests/`
- `verifier/resources/`

Human-review only:

- `readme.md`

Evaluator metadata:

- `metadata.yaml`

The Docker evaluator runs agent and verifier in separate container
invocations. The agent stage does not mount `oracle/` or `verifier/`.

## Verifier contract

`verifier/test.sh` is the single entrypoint. It should read:

- `PTDBENCH_WORKSPACE`
- `PTDBENCH_ORACLE`
- `PTDBENCH_VERIFIER`
- `PTDBENCH_RESULTS`
- `PTDBENCH_CASE_ID`

The verifier should exit `0` for success and non-zero for failure. It may also
write:

```json
{
  "passed": true,
  "score": 1.0,
  "metrics": {
    "hidden_pass_rate": 1.0
  }
}
```

to:

```text
$PTDBENCH_RESULTS/result.json
```

If no `result.json` exists, the evaluator falls back to the verifier exit code.

## CLI

Validate a case:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m ptdbench validate example/empty
```

Run only the verifier on the staged workspace with the local backend:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m ptdbench evaluate example/empty --backend local --keep-artifacts --json
```

Put run artifacts under a known directory for debugging:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m ptdbench evaluate example/empty \
  --backend local \
  --work-dir ./runs \
  --keep-artifacts \
  --json
```

Run an agent command inside the Docker environment, then verify:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m ptdbench evaluate example/empty \
  --backend docker \
  --agent shell \
  --agent-command "your-agent-command" \
  --timeout-s 3600 \
  --keep-artifacts \
  --json
```

Run Codex as the agent:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m ptdbench evaluate example/empty \
  --backend docker \
  --agent-backend local \
  --agent codex \
  --model gpt-5 \
  --json
```

The Codex adapter turns `instruction.md` into the initial prompt and runs
`codex exec` in the staged workspace. `--agent-backend local` is usually the
right first choice for CLI agents installed on the host, while `--backend docker`
keeps verifier execution isolated.

Agent options:

- `--agent codex|shell`
- `--agent-backend local|docker`
- `--model <name>`
- `--base-url <url>`
- `--api-key <key>`
- `--agent-command <command-or-executable>`
- `--agent-arg <arg>` for repeatable extra CLI flags

To add a new agent, put its runner under `ptdbench/agents/installed/`, register
it with the factory, and keep `evaluator.py` untouched.

The Docker backend builds from `environment/Dockerfile`, mounts the editable
workspace at `/workspace`, exposes public task text at `/case`, and writes
results to `/results`.

Run artifacts are grouped under `results/`. The main files are:

- `agent_trace.json`
- `result.json`
- `report.json`

Docker container network defaults:

- Docker agent stage: `bridge`, so API-backed agents can reach their model endpoint
- Docker verifier stage: `none`, so hidden grading logic stays offline by default

## Toy SWE Example

A runnable code-repair case lives in `example/swe_slugfix/`.

Validate the package:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m ptdbench validate example/swe_slugfix
```

Run the example with a fixing agent command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m ptdbench evaluate example/swe_slugfix \
  --backend local \
  --work-dir /private/tmp/ptdbench-swe-debug \
  --keep-artifacts \
  --json \
  --agent shell \
  --agent-command "<your repair command here>"
```
