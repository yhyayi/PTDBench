# PTDBench

PostTrainDiagnosisBench, or PTDBench, is a benchmark framework for community
submission of broken post-train tasks, environment specs, and verifier logic.

## Core idea

1. A task package provides a broken `workspace/` plus one or more evaluations.
2. A specified model+agent follows the selected evaluation's instruction.
3. The evaluator runs the selected `verifier/test.sh` against the workspace.
4. The verifier writes structured results to `$PTDBENCH_RESULTS/result.json`.

## Task layout

```text
task/
├── readme.md
├── metadata.yaml
├── workspace/
├── environment/
│   └── Dockerfile
└── evaluations/
    ├── issue/
    │   ├── instruction.md
    │   ├── oracle.md
    │   └── verifier/
    │       └── test.sh
    └── solution/
        ├── instruction.md
        ├── oracle/
        └── verifier/
            ├── test.sh
            ├── tests/        # optional
            └── resources/    # optional
```

The directory layout identifies the task format. A task may provide `issue`,
`solution`, or both under `evaluations/`. Legacy tasks with top-level
`instruction.md`, `oracle/`, and `verifier/` remain supported and are loaded as
a `solution` evaluation.

## Source layout

```text
ptdbench/
├── cli.py
├── core/                       # 稳定的数据模型
│   └── types.py
├── tasks/                      # task 加载、校验和 schema YAML
│   ├── loader.py
│   ├── validator.py
│   └── yaml.py
├── evaluation/                 # agent/verifier 流程编排
│   └── evaluator.py
├── runtimes/                   # 执行平台抽象和 provider
│   ├── base.py
│   ├── factory.py
│   └── providers/
│       ├── local.py
│       └── docker.py
├── agents/                     # agent 抽象、注册和内置适配器
│   ├── base.py
│   ├── factory.py
│   └── installed/
│       ├── codex.py
│       └── shell.py
└── utils/                      # 不带业务语义的通用工具
    └── filesystem.py
```

新增 agent 只需要实现 `agents/base.py` 约定的接口并在 agent factory 中注册；
新增执行平台只需要实现 `runtimes/base.py` 的 `SandboxRuntime` 并注册 provider。
两者都不需要修改 `evaluation/evaluator.py`。

## Visibility

Agent-visible for the selected evaluation:

- its `instruction.md`, exposed as `/task/instruction.md`
- a writable copy of `workspace/`
- the Docker image built from `environment/Dockerfile`

Verifier-only:

- the selected evaluation's `oracle.md` or `oracle/`
- the selected evaluation's `verifier/`

Human-review only:

- `readme.md`

Evaluator metadata:

- `metadata.yaml`

The Docker evaluator runs agent and verifier in separate container
invocations. The agent stage does not mount hidden oracle or verifier files.

## Runtime abstraction

PTDBench separates the task contract from the execution platform. The current
runtime implementations are:

- `local`: runs commands on the host and is intended only for development;
- `docker`: builds `environment/Dockerfile` and runs agent/verifier in separate
  containers.

The environment image is built once and can be reused by both stages. The two
stages still use different containers: the agent gets a writable workspace and
public instruction, while the verifier gets a read-only workspace plus the
selected hidden evaluation. Reusing one image does not mean sharing one
container, which is important for keeping hidden grading files inaccessible.

The platform boundary is `ptdbench.runtimes.base.SandboxRuntime`. A future runtime
such as Harbor or a remote cloud sandbox should implement this interface for
environment preparation, logical path mapping, and command execution. Agent
adapters only receive a runtime and an environment artifact; they do not import
Docker or host-process executors.

The low-level command implementation lives with the corresponding provider under
`ptdbench.runtimes.providers/`. New platform integrations should depend on
`SandboxRuntime`, not on the Docker or Local provider classes.

## Verifier contract

Each evaluation's `verifier/test.sh` is the single entrypoint. It should read:

- `PTDBENCH_WORKSPACE`
- `PTDBENCH_ORACLE`
- `PTDBENCH_VERIFIER`
- `PTDBENCH_RESULTS`
- `PTDBENCH_TASK_ID`
- `PTDBENCH_EVALUATION_LEVEL`

For an issue evaluation, the instruction may require the agent to write a
report such as `$PTDBENCH_RESULTS/agent_answer.md`. This is a task-level output
contract: the framework does not generate or recover that file from the agent's
final message. The issue verifier decides how to score it. `tests/` and
`resources/` are optional for both evaluation levels.

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

Validate a task:

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
  --evaluation-level solution \
  --backend docker \
  --agent-backend docker \
  --agent codex \
  --model gpt-5 \
  --reasoning-effort high \
  --json
```

The Codex adapter turns `instruction.md` into the initial prompt and runs
`codex exec` in the staged workspace. With `--agent-backend docker`, PTDBench
starts a fresh task container, installs `@openai/codex@latest`, and runs Codex
in that same container. The verifier runs in a separate clean container built
from the original task environment. Agent installation is not cached yet.
After installation, PTDBench creates a Codex model-provider entry from
`--base-url` and runs `codex login --with-api-key` via stdin. The credential is
not embedded in the command line or setup logs. The generated provider uses
the generic internal name `ptdbench`.
For the Docker backend, the container is the outer security boundary, so Codex
runs with `--dangerously-bypass-approvals-and-sandbox` to avoid an unsupported
nested bubblewrap user namespace. The local backend retains Codex's
`workspace-write` sandbox.

Agent options:

- `--evaluation-level issue|solution`
- `--agent codex|shell`
- `--agent-backend local|docker`
- `--model <name>`
- `--reasoning-effort minimal|low|medium|high|xhigh`
- `--base-url <url>`
- `--api-key <key>`
- `--agent-command <command-or-executable>`
- `--agent-arg <arg>` for repeatable extra CLI flags

`run.sh` defaults to `high`; override it without editing the script:

```bash
PTDBENCH_REASONING_EFFORT=medium ./run.sh
```

`run.sh` applies a 3600-second timeout to each long-running stage. Override it
with `PTDBENCH_TIMEOUT_S`. Terminal progress is quiet by default. Enable it
with `PTDBENCH_LOG_LEVEL=debug`; setup output is always streamed to
`results/agent_setup.stdout.log` and `results/agent_setup.stderr.log` while the
installation is running.

```bash
PTDBENCH_LOG_LEVEL=debug bash run.sh
```

Select the issue evaluation in the same way:

```bash
PTDBENCH_EVALUATION_LEVEL=issue ./run.sh
```

To add a new agent, put its runner under `ptdbench/agents/installed/`, register
it with the factory, and keep `evaluator.py` untouched. Its `run` method should
only construct a `CommandSpec` and call the supplied `SandboxRuntime`.

The Docker backend builds from `environment/Dockerfile`, mounts the editable
workspace at `/workspace`, exposes public task text at `/task`, and writes
results to `/results`. The verifier sees its selected evaluation at `/task`;
`PTDBENCH_ORACLE` and `PTDBENCH_VERIFIER` resolve the evaluation-specific
paths.

Run artifacts are grouped under `results/`. The main files are:

- `agent_trace.json`
- `result.json`
- `report.json`

Docker container network defaults:

- Docker agent stage: `bridge`, so API-backed agents can reach their model endpoint
- Docker verifier stage: `none`, so hidden grading logic stays offline by default

Docker Desktop does not automatically pass a host GUI/system proxy into a
container. If the host network requires Clash, opt in explicitly with a
host-reachable proxy URL (the port must match the Clash configuration):

```bash
export PTDBENCH_DOCKER_PROXY=http://host.docker.internal:7890
export PTDBENCH_DOCKER_NO_PROXY=localhost,127.0.0.1,host.docker.internal
./run.sh
```

Without `PTDBENCH_DOCKER_PROXY`, PTDBench leaves the default Docker bridge
network unchanged and does not require Clash.

## Toy SWE Example

A runnable code-repair task lives in `example/swe_slugfix/`.

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
