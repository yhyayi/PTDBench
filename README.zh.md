# PTDBench

PTDBench（PostTrainDiagnosisBench）是一个面向社区提交的后训练诊断基准框架。
它用来收集“带 bug 的 post-train 任务 + 环境 + verifier”，让指定的 model/agent 在受控工作区里修复问题并接受评测。

## 核心流程

1. 提交一个 task package，里面包含公开说明、可写 workspace、隐藏 oracle、环境声明和 verifier。
2. 评测器把 `workspace/` 复制到 staging 目录。
3. **prepare 阶段**跑任务的 `environment/prepare.sh`：下模型、下数据集、写环境变量。
   这是三个阶段里唯一联网、也唯一能写共享下载缓存的一个（后两个阶段只读挂）。
4. 指定的 agent 在 staging workspace 上修改代码。
5. verifier 运行 `verifier/test.sh`，读取修复后的 workspace 并输出结果。
6. 评测器汇总 `report.json`，同时保留 agent 轨迹和 verifier 结果。

## 快速上手

下面以 `tasks/slime_retool` 为例；`tasks/` 下的任意一个任务目录都可以替换进去。

### 1. 先做校验

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m ptdbench validate tasks/slime_retool
```

### 2. 再跑本地 + Codex

本地开发时仍可使用 `local + codex`：

- `--backend local`：评测和 verifier 都在本机跑，仅建议用于开发和调试
- `--agent codex`：用 Codex 作为 agent

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m ptdbench evaluate tasks/slime_retool \
  --backend local \
  --agent codex \
  --model gpt-5.5 \
  --base-url "https://yeysai.com/v1" \
  --api-key "YOUR_API_KEY" \
  --work-dir /private/tmp/ptdbench-swe-debug \
  --keep-artifacts \
  --json
```

在 `--backend local` 下，中央 evaluator 会把 `--model`、`--base-url`
和 `--api-key` 通过 verifier 子进程环境转交给需要 LLM Judge 的
self-contained verifier。Codex task 默认让 Judge 使用同一个
OpenAI-compatible model、base URL 和 API key；task verifier 仍可用
`PTDBENCH_JUDGE_*` 环境变量覆盖 Judge 设置。该转交不会把 key 写进
`report.json` 或 agent trace。Docker verifier backend 不转交 key，以免
secret 出现在 Docker argv 中。

直接写 `--api-key` 可能被同机进程列表看到；在不要求完全复现上述命令时，
仍建议从受保护的环境变量展开：

```bash
--api-key "$OPENAI_API_KEY"
```

### Docker agent + verifier

`run.sh` 现在默认让 Codex agent 和 verifier 都在 Docker 中运行。每次评测都会从 task
基础镜像启动一个新的 agent 容器，在容器内安装最新 Codex，然后在同一个容器内执行 rollout；
verifier 使用另一个干净容器。当前版本不缓存 agent 安装镜像。

```bash
export OPENAI_API_KEY='你的 API key'
./run.sh
```

它等价于：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m ptdbench evaluate tasks/slime_retool \
  --backend docker \
  --agent codex \
  --model gpt-5.5 \
  --judge-model gpt-5.5 \
  --reasoning-effort high \
  --base-url "${OPENAI_BASE_URL:-https://yeysai.com/v1}" \
  --api-key-env OPENAI_API_KEY \
  --work-dir outputs/ptdbench-swe-debug \
  --keep-artifacts \
  --json
```

运行前需要确保 Docker daemon 已启动，并允许 Docker Desktop 访问当前仓库目录。
评测器按 `metadata.yaml` 的 `environment.image` 决定镜像：`image: verl` 对应
`docker/images/verl.Dockerfile`，镜像 tag 由 Dockerfile 头部的 `# ptdbench-image:` 一行声明（如 `ptdbench/verl:vllm0.17-torch2.10-cu129`）。本地没有这个镜像时
先构建再跑，已有就直接复用；prepare/agent/verifier 三个阶段用同一个镜像、各自起容器。
想提前构建，或者改了 Dockerfile 之后需要重建（只改工具脚本时 tag 不变，改动不会被自动发现），用：

```bash
python3 -m ptdbench build-images                  # 构建 docker/images/ 下所有缺失的镜像
python3 -m ptdbench build-images verl slime-rc0   # 只构建指定的
python3 -m ptdbench build-images --rebuild verl   # 已存在也重建
```

`--backend` 是整次评测唯一的 runtime backend，agent 和 verifier 始终使用同一个 backend；
两者的网络策略仍可分别通过 `--agent-network` 和 `--verifier-network` 配置。
task 镜像不需要预装 Codex，框架会在 agent 容器启动后安装 `@openai/codex@latest`。
安装完成后，框架会根据 `--base-url` 在容器内生成 Codex model provider 配置，并通过 stdin
执行 `codex login --with-api-key`；API key 不会写入命令参数或运行日志。例如默认的
`https://yeysai.com/v1` 会配置为使用 Responses API 的通用 `ptdbench` provider。
`run.sh` 默认使用 `high` 推理强度，可以通过环境变量覆盖而不必修改脚本：

完成任务的 agent 模型和 verifier judge 模型可以分别设置：

```bash
PTDBENCH_MODEL=gpt-5.5 PTDBENCH_JUDGE_MODEL=gpt-4.1 ./run.sh
```

```bash
PTDBENCH_REASONING_EFFORT=medium ./run.sh
```

CLI 支持 `minimal`、`low`、`medium`、`high` 和 `xhigh`。
Docker 已经作为 agent 的外层安全边界，因此 Docker backend 会使用 Codex 的
`--dangerously-bypass-approvals-and-sandbox`，避免 bubblewrap 在容器内创建嵌套 user namespace
失败。本地 backend 仍使用 Codex 的 `workspace-write` 沙盒。

### 通过 Harbor 运行

`evaluate` 和 `harbor-evaluate` 是两个并行入口：前者使用 PTDBench 原有的
`TaskEvaluator`，后者在线生成临时 Harbor task 并交给 Harbor CLI 执行，最后再映射回同一个
PTDBench 结果结构。因此不需要修改原有评测命令。

Harbor 直接用预构建镜像，不再逐任务构建：生成的 task.toml 把 `[environment] docker_image`
设成该环境 Dockerfile 头部声明的 tag。每个 task 的流程是：

1. PTDBench 在同一个镜像里先跑 `environment/prepare.sh`（联网，与 `evaluate` 的 prepare
   阶段完全一致），镜像不存在时先构建；
2. prepare 写过的 workspace 放进 Harbor task 的 `environment/`（不含 Dockerfile），Harbor
   起容器后把它上传到 `/workspace`；
3. prepare 产物、任务的 `environment/`、模型和数据集缓存以只读 bind mount 挂到
   `/ptdbench/prepared`、`/ptdbench/environment`、`/ptdbench/models`、`/ptdbench/datasets`；
4. verifier 与 agent 共用同一个容器（shared 模式），agent / verifier 阶段对 HF / W&B 离线。

Harbor 的挂载是 job 级的，所以每个 task 单独一个 `harbor run`，最后合并成一份报告。
`--mounts`/`--mounts-json` 由 PTDBench 管理，不能再透传。Harbor 的 docker 环境没有 GPU
选项，GPU 任务需要把 Docker daemon 的默认 runtime 设为 `nvidia`。

如果 Harbor 是当前仓库旁边的 clone，可以这样运行：

```bash
export OPENAI_API_KEY='你的 API key'
export OPENAI_BASE_URL='https://yeysai.com/v1'
./run_harbor_fit.sh
```

脚本默认使用 `tasks/slime_retool`、`solution`、Codex、`gpt-5.5` 和
`uv run --project harbor harbor`。也可以直接展开它对应的 CLI 命令：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m ptdbench harbor-evaluate tasks/slime_retool \
  --evaluation-level solution \
  --judge-model gpt-5.5 \
  --huggingface-mirror https://hf-mirror.com \
  --agent-network public \
  --verifier-network public \
  --harbor-command "uv run --project harbor harbor" \
  --harbor-temp-dir outputs/harbor-tmp \
  --agent codex \
  --model gpt-5.5 \
  --agent-kwarg reasoning_effort=high \
  --env docker \
  --jobs-dir outputs/harbor \
  --yes \
  --debug
```

`--evaluation-level issue` 和
`solution` 分别选择 task package 中对应的两套 instruction/oracle/verifier，仍然只会运行
选中的一套，不会离线复制成两个数据集。`--agent-network` 和 `--verifier-network` 用于生成
Harbor task 中对应阶段的 `network_mode`。

`--harbor-command` 是可拆分的命令字符串，也可以直接写成 `"harbor"`。运行前需确保该命令
在当前 shell 中可用，并按 Harbor 的要求准备 agent、Docker 和模型凭据。

`--huggingface-mirror`、`--models-dir`、`--datasets-dir` 只作用于 prepare 阶段和挂载，
含义与 `evaluate` 相同。

除了上述 PTDBench 转换参数，其余参数会原样传给 `harbor run`。参数名、别名和行为均保持
Harbor 原义，因此 Harbor 增加普通 CLI 参数时不需要同步修改 adapter：

```bash
python3 -m ptdbench harbor-evaluate tasks/slime_retool \
  --evaluation-level solution \
  --judge-model gpt-4.1 \
  --agent codex \
  --model gpt-5.5 \
  --env daytona \
  --n-concurrent 8 \
  --max-retries 2
```

`--path/-p` 和 `--mounts/--mounts-json` 由 adapter 管理：前者始终指向在线生成的临时 Harbor task，后者是 prepare 产物和缓存的只读挂载。
`--harbor-temp-dir` 设置这些临时 task 的父目录；每次运行仍会创建并只清理一个随机唯一的
子目录。未设置时使用操作系统临时目录。
`--jobs-dir/-o` 和 `--job-name` 可以按 Harbor 原生方式使用；没有提供时 adapter 使用 `jobs`
和一个唯一 job name，以便可靠读取结果。临时 task 在运行后自动清理，Harbor job 正常保留。
使用 `--config/-c` 时，配置文件中的 `jobs_dir` 和 `job_name` 也会保持 Harbor 原义，不会被
adapter 覆盖。
`HarborEvaluator.evaluate()` 每个 task 对应一个 Harbor job，并统一返回
`HarborEvaluationReport`。批量时用户给的 `--job-name` 会按 task 加后缀。默认只有一个 trial；`--n-attempts` 或多 model 产生多个 trial 时，
它们仍然包含在同一个 report 中。

`harbor-evaluate` 也可以一次提交多个 task。多个显式路径必须写在 Harbor 原生参数之前；也可以
只传一个父目录，此时只展开它的直接子目录，不递归扫描更深层级。task 按顺序逐个 prepare、
逐个提交 `harbor run`；某个 task 的 prepare 失败只让它出现在 `missing_task_ids` 里。
同一 task 的多次 trial（`--n-attempts`）并发仍由 Harbor 原生的 `--n-concurrent` 负责：

```bash
python3 -m ptdbench harbor-evaluate \
  tasks/rlve_hyper/task_001 \
  tasks/rlve_hyper/task_008 \
  --agent codex \
  --model gpt-5.5
```

也可以直接运行多任务脚本。无参数时它使用上面两个 task；传一个集合目录时会运行该目录的
直接子 task。脚本参数中的 `--` 用来分隔任务路径和额外 Harbor 参数：

```bash
./run_harbor_multi.sh
./run_harbor_multi.sh tasks/rlve_hyper -- --n-attempts 2
```

批次报告顶层使用 `task_ids`，并包含 `n_tasks`、`n_completed_tasks` 和
`missing_task_ids`；每条 trial 仍使用单数 `task_id`。只有每个提交的 task 都至少产生一条
trial 且所有 trial 均通过时，批次的 `passed` 才为 `true`。

Python API 同样只区分 PTDBench 转换配置和 Harbor 原生 argv：

```python
HarborEvaluationConfig(
    evaluation_level="solution",
    judge_model="gpt-4.1",
    harbor_args=(
        "--agent", "codex",
        "--model", "gpt-5.5",
        "--env", "docker",
        "--agent-kwarg", "reasoning_effort=high",
    ),
)
```

`run_harbor_fit.sh` 和 `run_harbor_multi.sh` 可通过 `PTDBENCH_TASK`、`PTDBENCH_MODEL`、`PTDBENCH_EVALUATION_LEVEL`、
`PTDBENCH_JUDGE_MODEL`、`PTDBENCH_REASONING_EFFORT`、`PTDBENCH_BACKEND`、
`PTDBENCH_AGENT_NETWORK`、`PTDBENCH_VERIFIER_NETWORK`、`PTDBENCH_HARBOR_COMMAND`、`PTDBENCH_HARBOR_TEMP_DIR`、
`PTDBENCH_TIMEOUT_MULTIPLIER`、`PTDBENCH_N_CONCURRENT` 和 `PTDBENCH_JOBS_DIR` 覆盖默认值；
额外 CLI 参数可以直接追加在脚本命令后。

Docker 默认使用 bridge 网络直连外网；宿主机开启 Clash 不会自动让容器使用 Clash。
如果当前网络必须通过 Clash，先确认 Clash 对 Docker Desktop 可访问（通常开启局域网访问），
然后显式设置宿主机可访问的代理地址。常见配置如下，端口以 Clash 实际配置为准：

```bash
export PTDBENCH_DOCKER_PROXY=http://host.docker.internal:7890
export PTDBENCH_DOCKER_NO_PROXY=localhost,127.0.0.1,host.docker.internal
./run.sh
```

Harbor 的容器环境变量直接使用其原生参数。例如 Colima 中只有 HTTPS 需要经过宿主机代理时：

```bash
bash run_harbor_fit.sh \
  --agent-env HTTPS_PROXY=http://192.168.5.2:7890 \
  --verifier-env HTTPS_PROXY=http://192.168.5.2:7890
```

不设置 `PTDBENCH_DOCKER_PROXY` 时，框架不会修改容器网络或代理环境，适用于本身可以直连外网的环境。

如果你只想先看 verifier 是否能跑通，也可以先挂一个空 agent：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m ptdbench evaluate tasks/slime_retool \
  --backend local \
  --agent shell \
  --agent-command true \
  --keep-artifacts \
  --json
```

## 目录结构

```text
tasks/your_task/
├── instruction.md
├── readme.md
├── metadata.yaml
├── workspace/
├── oracle/
├── environment/         # 可选
│   └── prepare.sh       # 跑训练前要准备的东西（下模型、下数据集）
└── verifier/
    ├── test.sh
    ├── tests/
    └── resources/
```

## 可见性约定

- `instruction.md`：agent 可见
- `workspace/`：agent 可写
- `environment/`：任务自己的环境文件（主要是 `prepare.sh`）；
  三个阶段都以**只读**挂在 `/ptdbench/environment`
- `oracle/`：仅 verifier 可见
- `verifier/`、`verifier/tests/`、`verifier/resources/`：仅 verifier 可见
- `readme.md`：仅人类审核用
- `metadata.yaml`：提交元数据

## 评测输出

一次 run 的结果目录里通常保留三类文件：

- `agent_trace.json`：agent 的完整调用轨迹
- `result.json`：verifier 自己写出的结构化结果
- `report.json`：PTDBench 汇总后的总报告

## 元数据 schema

`metadata.yaml` 需要包含以下顶层字段：

- `title`
- `submitter`
- `license`
- `created_date`
- `domain`
- `model`
- `dataset`
- `training`
- `environment`
- `outcome`

其中建议结构如下：

```yaml
title: 'rlve_hyper hyperparameter diagnosis: task_001'
submitter:
  name: benchmark_evaluation
license: MIT
created_date: '2026-08-08'
domain: post_train
model:
  name: Qwen2.5-0.5B-Instruct
dataset:
  name: RLVE held-out environments
training:
  framework: RLVE / slime + GRPO
  algo: grpo
  script: scripts/training/Qwen2.5-0.5B-Instruct/rlve.sh
environment:
  image: slime-rc0
outcome:
  failure_type: training_collapse
  failure_reason: '--attention-dropout=0.5 keeps held-out eval moving the wrong way
```

### 字段说明

- `submitter.name`：提交者名称
- `model.name`：出问题时所用模型或系统名称
- `dataset.name`：训练或数据来源名称
- `training.framework`：训练框架
- `training.algo`：训练算法
- `training.script`：训练启动脚本，写成相对 `workspace/` 的路径
- `training.script_args`：启动脚本的位置参数，列表，一项一个参数
- `environment`：运行环境，见下文 environment 一节
- `outcome.failure_type`：失败类型
- `outcome.failure_reason`：失败原因说明

### training.script：任务怎么跑起来

`training.script` 是"这个任务的训练入口是哪个脚本"的唯一出处，任务卡上记的就是它，
`instruction.md` 里写给 agent 看的也应该是同一条路径。路径相对
`workspace/`，因为容器里 workspace 挂在 `/workspace`，训练脚本是在那儿跑的。

启动脚本需要位置参数才能跑起来时（RLVE 的 `rlve.sh` 要 `WANDB_PROJECT`、
`RUN_NAME` 和一个占位的环境列表 token），用 `script_args` 一并写上，否则任务卡上
记的那条命令是跑不通的：

```yaml
training:
  framework: RLVE / slime + GRPO
  algo: grpo
  script: scripts/training/Qwen2.5-0.5B-Instruct/rlve.sh
  script_args:
    - ptdbench
    - solution_rerun
    - _
```

只有真的没有训练脚本的任务（比如 `framework: none` 的纯代码修复）可以不写这两个
字段；声明了训练框架却没写 `script`，校验会给一条 warning。

### environment

```yaml
environment:
  image: verl                       # docker/images/ 下的 <name>.Dockerfile
  prepare: environment/prepare.sh   # 可选，默认就是这个路径
  env:                              # 可选，逐任务不同的环境变量
    CUDA_VISIBLE_DEVICES: "0,1"
```

- `image`：必填，用哪个环境镜像。可选值就是 `docker/images/` 下 `*.Dockerfile` 的文件名
  （目前是 `verl`、`slime-rc0`、`slime-v0.3.2`），没有第二份常量表。
  镜像 tag 写在 Dockerfile 头部 `# ptdbench-image: <tag>`，描述的是栈（框架、推理引擎、CUDA），
  多个任务共用，不带 hash；构建上下文是 `docker/`。
- 要加新环境就加一个 `docker/images/<name>.Dockerfile`：`FROM` 必须钉 digest
  （`image:tag@sha256:...`），文件末尾照抄其他 Dockerfile 里
  `# ---- PTDBench tooling ...` 之后的那一段（`ptdbench_fetch_*`、`ptdbench_dataset`、
  激活 hook），测试会检查每个 Dockerfile 的这一段完全一致。
- 一个镜像是多个任务共享的，镜像内容就是这些任务的 ground truth 环境。改 Dockerfile
  等于同时改所有用它的任务；需要不同依赖的任务应该用新的镜像名，而不是改已有的。
- `prepare`：不写就按约定路径找，找得到才算数。写了则以写的为准，
  但必须待在 `environment/` 里——只有那个目录会被挂进容器。
- `env`：值**必须是字符串**。YAML 里 `CUDA_VISIBLE_DEVICES: 0,1` 会读成字符串，
  但 `: 0` 会读成整数——后者会被拒掉而不是悄悄转成 `"0"`，因为"只用 0 号卡"和
  "写错了"必须能分清。另外不允许设 `PTDBENCH_*`，那是 harness 自己的通道。
- 旧的 `conda`、`requirements`、`dockerfile` 字段已经不支持，校验会直接报错。
- agent 阶段 `pip install` 会装到用户目录（`PIP_USER=1`），不会改动镜像本身的
  site-packages。

`environment/prepare.sh`：跑训练前要准备的一切。它跑在一个独立的 **prepare 阶段**
容器里，是三个阶段里**唯一联网**的一个，也是唯一挂到共享下载缓存的一个：

```bash
#!/usr/bin/env bash
set -euo pipefail

ptdbench_fetch_hf \
    --repo LIF1014/ptdbench-Qwen2.5-0.5B-Instruct \
    --revision e967ddaecb746b9cea032fc1ae80c29a9a49bb73 \
    --allow config.json --allow tokenizer.json --allow model.safetensors \
    --expect model.safetensors=fdf756fa...288fb7fe \
    --expect-size model.safetensors=988097824 \
    --dest "$PTDBENCH_WORKSPACE/verl/Qwen2.5-0.5B-Instruct"

ptdbench_dataset --recipe gsm8k --dest "$PTDBENCH_WORKSPACE/gsm8k_verl"
```

脚本可以用的变量和命令：

| 名字 | 是什么 |
| --- | --- |
| `$PTDBENCH_WORKSPACE` | agent 会拿到的 workspace，**prepare 阶段可写**；权重和数据集下到这里 |
| `$PTDBENCH_PREPARED` | 交给后两个阶段的产物目录；往 `$PTDBENCH_PREPARED/env` 写 `KEY=VALUE` 行即可给 agent/verifier 传环境变量 |
| `$PTDBENCH_MODELS` | 跨 run、跨任务共享的模型下载缓存；`--models-dir` 指定，默认仓库根的 `models/`。prepare 可写，agent/verifier 只读挂 |
| `$PTDBENCH_DATASETS` | 同上，数据集用；`--datasets-dir` 指定，默认仓库根的 `datasets/` |
| `$PTDBENCH_ENVIRONMENT` | 任务自己的 `environment/` 目录（只读） |
| `ptdbench_fetch_hf` | 从 HF 仓库按 allow 列表取文件，带 sha256/size 硬校验，走缓存 |
| `ptdbench_fetch_url` | 同上，但取单个 URL |
| `ptdbench_dataset` | 跑镜像内置的数据集配方（`docker/ptdbench_datasets/`），产物同样走缓存 |

`$PTDBENCH_MODELS` / `$PTDBENCH_DATASETS` 是缓存，不是交付路径。任务脚本要引用的
永远是 `--dest` 写出去的那份：缓存目录可能被清理或换机器，只有 `--dest` 那份会跟着
workspace 交给 agent（Harbor 后端里它随 workspace 一起上传到 `/workspace`）。

`$PTDBENCH_PREPARED/env` 是被当成**数据**逐行读取的，不会被 `.` 进任何 shell——
否则一个任务包就能在 verifier 前面跑任意命令。

校验不过是**硬失败**，不会降级成"重新下一遍"——那会把一次数据损坏变成一行看不见
的重试日志。

评测机没网的话，先在有网的机器上跑一次
`docker/warm-cache.sh [--models-dir DIR] [--datasets-dir DIR]`，它会把每个任务的
`prepare.sh` 各跑一遍填满这两个目录；之后把它们搬到评测机上，用 `--models-dir`
和 `--datasets-dir` 指过去就能离线评测。评测机上也要有对应的镜像，可以在有网的机器上
`build-images` 之后 `docker save` / `docker load` 过去。

### 失败类型

当前建议使用以下几类：

- `training_collapse`：训练直接崩坏，输出异常
- `training_failure`：训练没崩，但指标不涨
- `reward_over_optimization`：训练 reward 涨了，但验证不涨
- `normal_training`：训练和验证都正常

## 新任务怎么加

最简单的方式是参考 `tasks/` 下同类的现有任务，例如 `tasks/slime_retool`。

### 你需要补的东西

1. `instruction.md`
   - 给 agent 看的公开任务说明
   - 要写清楚“要修什么”“期望行为是什么”“不要改什么”

2. `readme.md`
   - 只给人类看
   - 用来解释任务背景、为什么这么设 verifier、你希望审核者重点看哪里

3. `metadata.yaml`
   - 填完整元数据
   - 至少把 `title`、`submitter`、`license`、`created_date`、`domain`、`model`、`dataset`、`training`、`environment`、`outcome` 写全
   - 训练类任务还要写 `training.script`（必要时加 `script_args`），否则没人知道这个任务该跑哪个脚本

4. `workspace/`
   - 放有 bug 的可写工作区
   - 这是 agent 真正会改的地方
   - 对 SWE 任务来说，最好保持最小可复现

5. `oracle/`
   - 放隐藏参考实现、正确数据、正确模型输出，或者 verifier 需要的真值
   - agent 不应该看到这里

6. `metadata.yaml` 里的 `environment.image`，以及可选的 `environment/prepare.sh`
   - 从 `docker/images/` 里挑一个现成的镜像；都不合适再新加一个 Dockerfile
   - 模型、数据集等运行前要下载的东西写进 `prepare.sh`

7. `verifier/test.sh`
   - 评测入口，只保留一个主脚本
   - 负责读取环境变量、启动测试、写 `result.json`

8. `verifier/tests/`
   - 放具体测试代码
   - 对 SWE 任务通常就是单测或集成测试

9. `verifier/resources/`
   - 放 verifier 专用资源
   - 比如隐藏样例、额外配置、对照数据等

### 新任务的注意点

- `instruction.md` 要能单独读懂，不要依赖 `readme.md`
- `readme.md` 不要泄露 verifier 逻辑
- `workspace/` 里的内容必须是 agent 可改的版本
- `oracle/` 里可以放正确答案，但不要被 agent stage 挂进去
- `verifier/test.sh` 要尽量稳定，别依赖不受控的外部状态
- 如果 verifier 里要 import workspace 代码，记得把 `PTDBENCH_WORKSPACE` 加进 `sys.path`
- 本框架现在的运行轨迹会保存在 `results/agent_trace.json`

## verifier 约定

`verifier/test.sh` 是单一入口，建议读取这些环境变量：

- `PTDBENCH_WORKSPACE`
- `PTDBENCH_ORACLE`
- `PTDBENCH_VERIFIER`
- `PTDBENCH_RESULTS`
- `PTDBENCH_TASK_ID`

`verifier` 需要：

- 成功时退出码为 `0`
- 失败时退出非 `0`
- 如有结构化结果，写入 `$PTDBENCH_RESULTS/result.json`

## 常用命令

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m ptdbench validate tasks/slime_retool
PYTHONDONTWRITEBYTECODE=1 python3 -m ptdbench evaluate tasks/slime_retool --backend local --agent codex --model gpt-5.5 --json
```
