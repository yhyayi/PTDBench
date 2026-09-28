# PTDBench

PostTrainDiagnosisBench, or PTDBench, is a benchmark framework for community
submission of broken post-train tasks, environment specs, and verifier logic.

## Core idea

1. A task package provides a broken `workspace/` plus one or more evaluations.
2. A prepare stage runs the task's `environment/prepare.sh` to fetch models and
   datasets. It is the only stage with network access and the only one that
   sees the shared download cache.
3. A specified model+agent follows the selected evaluation's instruction.
4. The evaluator runs the selected `verifier/test.sh` against the workspace.
5. The verifier writes structured results to `$PTDBENCH_RESULTS/result.json`.

## Task layout

```text
task/
├── readme.md
├── metadata.yaml
├── workspace/
├── environment/         # optional
│   └── prepare.sh       # fetch models/datasets before the run
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
- a writable copy of `workspace/`, including whatever `prepare.sh` fetched into it
- the environment image named by `environment.image`
  (`docker/images/<name>.Dockerfile`, tagged as its `# ptdbench-image:` header says)
- the task's own `environment/` directory, read-only at `/ptdbench/environment`
- the prepare stage's output, read-only at `/ptdbench/prepared`

Verifier-only:

- the selected evaluation's `oracle.md` or `oracle/`
- the selected evaluation's `verifier/`

Human-review only:

- `readme.md`

Evaluator metadata:

- `metadata.yaml`

The Docker evaluator runs prepare, agent and verifier in separate container
invocations. The agent stage does not mount hidden oracle or verifier files, and
neither does the prepare stage: it executes a script the task package ships, so
there is no reason to let it read the answer.

## Runtime abstraction

PTDBench separates the task contract from the execution platform. The current
runtime implementations are:

- `local`: runs commands on the host and is intended only for development;
- `docker`: runs prepare/agent/verifier in separate containers, all from the
  image named by the task's `environment.image`. The image is built from
  `docker/images/<name>.Dockerfile` the first time it is missing; run
  `ptdbench build-images [NAME...] [--rebuild]` to build ahead of time or to
  pick up an edited Dockerfile (the tag is fixed, so edits are not detected).

The environment image is prepared once and reused by all three stages. The
stages still use different containers: prepare gets the network and the download
cache, the agent gets a writable workspace and public instruction, while the
verifier gets a read-only workspace plus the selected hidden evaluation. Reusing
one image does not mean sharing one container, which is important for keeping
hidden grading files inaccessible.

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
PYTHONDONTWRITEBYTECODE=1 python3 -m ptdbench validate tasks/slime_retool
```

Run only the verifier on the staged workspace with the local backend:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m ptdbench evaluate tasks/slime_retool --backend local --keep-artifacts --json
```

Put run artifacts under a known directory for debugging:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m ptdbench evaluate tasks/slime_retool \
  --backend local \
  --work-dir ./runs \
  --keep-artifacts \
  --json
```

Run an agent command inside the Docker environment, then verify:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m ptdbench evaluate tasks/slime_retool \
  --backend docker \
  --agent shell \
  --agent-command "your-agent-command" \
  --timeout-s 3600 \
  --keep-artifacts \
  --json
```

Run Codex as the agent:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m ptdbench evaluate tasks/slime_retool \
  --evaluation-level solution \
  --backend docker \
  --agent codex \
  --model gpt-5 \
  --reasoning-effort high \
  --json
```

The Codex adapter turns `instruction.md` into the initial prompt and runs
`codex exec` in the staged workspace. With `--backend docker`, PTDBench starts
a fresh task container, installs `@openai/codex@latest`, and runs Codex in that
same container. The verifier runs in a separate clean container built
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
- `--model <name>`
- `--judge-model <name>`
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

### Harbor batch evaluation

`harbor-evaluate` runs every task on its prebuilt environment image: the
generated task.toml sets `docker_image` to the tag declared in
`docker/images/<name>.Dockerfile`, so Harbor never builds a per-task image. For
each task PTDBench first runs `environment/prepare.sh` in that image (online,
same staging as `evaluate`), then starts one Harbor job with the prepared
workspace uploaded to `/workspace` and the prepare output, the task's
`environment/` and the model/dataset caches bind-mounted read-only. The verifier
shares the agent's container. Agent and verifier are offline for HF/W&B.

Harbor mounts are job-wide, so each task gets its own job; the per-task reports
are merged. It accepts one or more task directories, or a directory whose direct
children are tasks (not recursive):

```bash
python3 -m ptdbench harbor-evaluate \
  tasks/rlve_hyper/task_001 \
  tasks/rlve_hyper/task_008 \
  --agent codex \
  --model gpt-5.5
```

GPUs: Harbor's docker environment has no GPU option, so GPU tasks need the
Docker daemon's default runtime set to `nvidia` (or an equivalent compose
override on the Harbor side). `--mounts`/`--mounts-json` are managed by
PTDBench and rejected if forwarded.

Keep every task path before forwarded Harbor options. The included batch
script makes that boundary explicit: arguments before `--` are task paths and
arguments after it are passed to `harbor run`.

```bash
./run_harbor_multi.sh
./run_harbor_multi.sh tasks/rlve_hyper -- --n-attempts 2
```

Each trial keeps its singular `task_id`. A batch report uses `task_ids` and
also reports `n_tasks`, `n_completed_tasks`, and `missing_task_ids`. It passes
only when every submitted task produced at least one trial and every trial
passed.

To add a new agent, put its runner under `ptdbench/agents/installed/`, register
it with the factory, and keep `evaluator.py` untouched. Its `run` method should
only construct a `CommandSpec` and call the supplied `SandboxRuntime`.

The Docker backend resolves the image from the task's `environment.image`,
mounts the editable
workspace at `/workspace`, exposes public task text at `/task`, and writes
results to `/results`. The verifier sees its selected evaluation at `/task`;
`PTDBENCH_ORACLE` and `PTDBENCH_VERIFIER` resolve the evaluation-specific
paths. `--backend` selects the single runtime backend used by both agent and
verifier; their network policies remain independently configurable through
`--agent-network` and `--verifier-network`.

Run artifacts are grouped under `results/`. The main files are:

- `agent_trace.json`
- `result.json`
- `report.json`

Docker container network defaults:

- Docker agent stage: `bridge`, so API-backed agents can reach their model endpoint
- Docker verifier stage: `bridge`, so API-backed judges and external validation
  services are available; pass `--verifier-network none` for an offline verifier

Docker Desktop does not automatically pass a host GUI/system proxy into a
container. If the host network requires Clash, opt in explicitly with a
host-reachable proxy URL (the port must match the Clash configuration):

```bash
export PTDBENCH_DOCKER_PROXY=http://host.docker.internal:7890
export PTDBENCH_DOCKER_NO_PROXY=localhost,127.0.0.1,host.docker.internal
./run.sh
```

The proxy settings apply to both agent and verifier containers. Without
`PTDBENCH_DOCKER_PROXY`, PTDBench leaves the default Docker bridge network
unchanged and does not require Clash.
