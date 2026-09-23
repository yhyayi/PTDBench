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

## 先跑 example

当前最值得先跑的是 `example/swe_slugfix`，它是一个最小可运行的 SWE 修复任务。
workspace 里有一个 `toy_swe.slugify.slugify()` 的 bug，agent 需要把它修好，让 verifier 测试通过。

### 1. 先做校验

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m ptdbench validate example/swe_slugfix
```

### 2. 再跑本地 + Codex

本地开发时仍可使用 `local + codex`：

- `--backend local`：评测和 verifier 都在本机跑，仅建议用于开发和调试
- `--agent codex`：用 Codex 作为 agent

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m ptdbench evaluate example/swe_slugfix \
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
PYTHONDONTWRITEBYTECODE=1 python3 -m ptdbench evaluate example/swe_slugfix \
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
评测器按 `metadata.yaml` 的 `environment` 段决定环境：写 `conda` 的任务直接用统一镜像
（`--image` / `PTDBENCH_IMAGE_TAG` 可覆盖），写 `dockerfile` 的 legacy 任务仍然逐任务构建
基础镜像。两种情况都分别启动 agent/verifier 容器。
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

如果 Harbor 是当前仓库旁边的 clone，可以这样运行：

```bash
export OPENAI_API_KEY='你的 API key'
export OPENAI_BASE_URL='https://yeysai.com/v1'
./run_harbor.sh
```

脚本默认使用 `example/swe_slugfix`、`solution`、`separate`、Codex、`gpt-5.5` 和
`uv run --project harbor harbor`。也可以直接展开它对应的 CLI 命令：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m ptdbench harbor-evaluate example/swe_slugfix \
  --evaluation-level solution \
  --judge-model gpt-5.5 \
  --debian-mirror https://mirrors.tuna.tsinghua.edu.cn \
  --verifier-mode separate \
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

`--verifier-mode separate` 是默认值，表示 agent 结束后由 Harbor 使用独立 verifier 环境；
需要复用同一个环境时改成 `--verifier-mode shared`。`--evaluation-level issue` 和
`solution` 分别选择 task package 中对应的两套 instruction/oracle/verifier，仍然只会运行
选中的一套，不会离线复制成两个数据集。`--agent-network` 和 `--verifier-network` 用于生成
Harbor task 中对应阶段的 `network_mode`。

`--harbor-command` 是可拆分的命令字符串，也可以直接写成 `"harbor"`。运行前需确保该命令
在当前 shell 中可用，并按 Harbor 的要求准备 agent、Docker 和模型凭据。

`--debian-mirror` 是任务物化参数。设置后，adapter 会在生成的 Dockerfile 中替换 Debian
软件源，确保 Harbor 启动容器后安装 agent 依赖时也使用该镜像；不设置则保持任务原始软件源。
`run_harbor.sh` 默认使用 `https://mirrors.tuna.tsinghua.edu.cn`，可通过
`PTDBENCH_DEBIAN_MIRROR` 覆盖。

除了上述 PTDBench 转换参数，其余参数会原样传给 `harbor run`。参数名、别名和行为均保持
Harbor 原义，因此 Harbor 增加普通 CLI 参数时不需要同步修改 adapter：

```bash
python3 -m ptdbench harbor-evaluate example/swe_slugfix \
  --evaluation-level solution \
  --judge-model gpt-4.1 \
  --agent codex \
  --model gpt-5.5 \
  --env daytona \
  --n-concurrent 8 \
  --max-retries 2
```

`--path/-p` 是唯一由 adapter 管理的 Harbor 参数，它始终指向在线生成的临时 Harbor task。
`--harbor-temp-dir` 设置这些临时 task 的父目录；每次运行仍会创建并只清理一个随机唯一的
子目录。未设置时使用操作系统临时目录。
`--jobs-dir/-o` 和 `--job-name` 可以按 Harbor 原生方式使用；没有提供时 adapter 使用 `jobs`
和一个唯一 job name，以便可靠读取结果。临时 task 在运行后自动清理，Harbor job 正常保留。
使用 `--config/-c` 时，配置文件中的 `jobs_dir` 和 `job_name` 也会保持 Harbor 原义，不会被
adapter 覆盖。
`HarborEvaluator.evaluate()` 一次对应一个 Harbor job，并统一返回
`HarborEvaluationReport`。默认只有一个 trial；`--n-attempts` 或多 model 产生多个 trial 时，
它们仍然包含在同一个 report 中。

`harbor-evaluate` 也可以一次提交多个 task。多个显式路径必须写在 Harbor 原生参数之前；也可以
只传一个父目录，此时只展开它的直接子目录，不递归扫描更深层级。所有 task 会被物化到同一个
临时 Harbor dataset，并通过一次 `harbor run` 提交。并发调度由 Harbor 原生的
`--n-concurrent` 负责，PTDBench 不另开线程或进程：

```bash
python3 -m ptdbench harbor-evaluate \
  tasks/raw/rlve_hyper/task_001 \
  tasks/raw/rlve_hyper/task_002 \
  --agent codex \
  --model gpt-5.5 \
  --n-concurrent 2
```

也可以直接运行多任务脚本。无参数时它使用上面两个 task；传一个集合目录时会运行该目录的
直接子 task。脚本参数中的 `--` 用来分隔任务路径和额外 Harbor 参数：

```bash
./run_harbor_multi.sh
./run_harbor_multi.sh tasks/raw/rlve_hyper -- --n-attempts 2
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

`run_harbor.sh` 和 `run_harbor_fit.sh` 可通过 `PTDBENCH_TASK`、`PTDBENCH_MODEL`、`PTDBENCH_EVALUATION_LEVEL`、
`PTDBENCH_JUDGE_MODEL`、`PTDBENCH_REASONING_EFFORT`、`PTDBENCH_BACKEND`、
`PTDBENCH_AGENT_NETWORK`、`PTDBENCH_VERIFIER_NETWORK`、`PTDBENCH_VERIFIER_MODE`、
`PTDBENCH_DEBIAN_MIRROR`、`PTDBENCH_HARBOR_COMMAND`、`PTDBENCH_HARBOR_TEMP_DIR`、
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
bash run_harbor.sh \
  --agent-env HTTPS_PROXY=http://192.168.5.2:7890 \
  --verifier-env HTTPS_PROXY=http://192.168.5.2:7890
```

不设置 `PTDBENCH_DOCKER_PROXY` 时，框架不会修改容器网络或代理环境，适用于本身可以直连外网的环境。

如果你只想先看 verifier 是否能跑通，也可以先挂一个空 agent：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m ptdbench evaluate example/swe_slugfix \
  --backend local \
  --agent shell \
  --agent-command true \
  --keep-artifacts \
  --json
```

## 目录结构

```text
example/your_task/
├── instruction.md
├── readme.md
├── metadata.yaml
├── workspace/
├── oracle/
├── environment/
│   ├── requirements.txt # 新形态：该 conda 环境的完整依赖闭包
│   ├── prepare.sh       # 新形态：跑训练前要准备的东西（下模型、下数据集）
│   └── Dockerfile      # legacy 形态
└── verifier/
    ├── test.sh
    ├── tests/
    └── resources/
```

## 可见性约定

- `instruction.md`：agent 可见
- `workspace/`：agent 可写
- `environment/`：环境声明（`requirements.txt` / `prepare.sh`，legacy 任务是 `Dockerfile`）；
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
title: Toy slugify repair
submitter:
  name: ptdbench
license: MIT
created_date: 2026-08-06
domain: code
model:
  name: codex
dataset:
  name: toy-swe
training:
  framework: none
  algo: none
environment:
  dockerfile: environment/Dockerfile
outcome:
  failure_type: implementation_bug
  failure_reason: slugify only handles spaces and does not normalize punctuation
```

### 字段说明

- `submitter.name`：提交者名称
- `model.name`：出问题时所用模型或系统名称
- `dataset.name`：训练或数据来源名称
- `training.framework`：训练框架
- `training.algo`：训练算法
- `training.script`：训练启动脚本，写成相对 `workspace/` 的路径
- `training.script_args`：启动脚本的位置参数，列表，一项一个参数
- `environment`：运行环境，两种形态二选一，见下一节
- `outcome.failure_type`：失败类型
- `outcome.failure_reason`：失败原因说明

### training.script：任务怎么跑起来

`training.script` 是"这个任务的训练入口是哪个脚本"的唯一出处：`docker/debug-env.sh
run` 读它来跑训练，`instruction.md` 里写给 agent 看的也应该是同一条路径。路径相对
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

### environment 的两种形态

**形态一：统一镜像 + conda 环境（新任务用这个）**

```yaml
environment:
  conda: verl                                  # docker/envs/ 下的目录名
  requirements: environment/requirements.txt   # 可选，默认就是这个路径
  prepare: environment/prepare.sh              # 可选，默认就是这个路径
  env:                                         # 可选，逐任务不同的环境变量
    CUDA_VISIBLE_DEVICES: "0,1"
```

- `conda`：用镜像里预建的哪个 conda 环境。可选值就是 `docker/envs/` 下的目录名
  （目前是 `verl` 和 `slime`），镜像本来就是照它构建的，所以这里没有第二份常量表。
  要加新环境得先加 `docker/envs/<name>/` 和 `docker/Dockerfile` 里对应的
  `env-<name>` stage。
- 用 conda 而不是 venv，是因为 slime-rc0 编译时对的是 CUDA 12.6、verl 对的是 12.9、
  slime-v0.3.1 对的是 13.0，**一个 venv 装不下三个 nvcc**（何况跨了大版本）。每个环境都在自己的构建 stage 里落在
  `/opt/conda/envs/<name>`，再原样 COPY 进最终镜像——路径一致，shebang、`.pth`、
  RPATH、`conda-meta` 里记的 prefix 才都还成立。
- `requirements` / `prepare`：不写就按约定路径找，找得到才算数。写了则以写的为准，
  但必须待在 `environment/` 里——只有那个目录会被挂进（或 COPY 进）容器。
- `requirements.txt` 是**这个 conda 环境的完整闭包**，不是"额外再装的东西"。
  同一个 `conda` 名下的所有任务必须给出**完全一致**的一份，`docker/build.sh` 会校验。
  取并集是不行的：加一个任务就会悄悄改掉另一个任务的环境，连带改掉它的 ground truth。
- `env`：值**必须是字符串**。YAML 里 `CUDA_VISIBLE_DEVICES: 0,1` 会读成字符串，
  但 `: 0` 会读成整数——后者会被拒掉而不是悄悄转成 `"0"`，因为"只用 0 号卡"和
  "写错了"必须能分清。另外不允许设 `PTDBENCH_*`，那是 harness 自己的通道。
- 不写 `dockerfile`。两个都写会报错，因为运行时只有一个生效，不该让读者猜。

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
永远是 `--dest` 写出去的那份——Harbor 后端在构建期跑完 prepare.sh 就会把这两个目录
`rm -rf` 掉，直接引用 `/ptdbench/models/...` 的任务在那条路径上会找不到文件。

`$PTDBENCH_PREPARED/env` 是被当成**数据**逐行读取的，不会被 `.` 进任何 shell——
否则一个任务包就能在 verifier 前面跑任意命令。

校验不过是**硬失败**，不会降级成"重新下一遍"——那会把一次数据损坏变成一行看不见
的重试日志。

评测机没网的话，先在有网的机器上跑一次
`docker/warm-cache.sh [--models-dir DIR] [--datasets-dir DIR]`，它会把每个任务的
`prepare.sh` 各跑一遍填满这两个目录；之后把它们搬到评测机上，用 `--models-dir`
和 `--datasets-dir` 指过去就能离线评测。

**形态二：逐任务 Dockerfile（legacy，`tasks/raw/` 下还没迁的任务用）**

```yaml
environment:
  dockerfile: environment/Dockerfile
```

字段缺省时按约定路径 `environment/Dockerfile` 算。

### 失败类型

当前建议使用以下几类：

- `training_collapse`：训练直接崩坏，输出异常
- `training_failure`：训练没崩，但指标不涨
- `reward_over_optimization`：训练 reward 涨了，但验证不涨
- `normal_training`：训练和验证都正常

## 新任务怎么加

最简单的方式是直接参考 `example/empty` 和 `example/swe_slugfix`。

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

6. `environment/Dockerfile`
   - 放运行环境依赖
   - 如果任务未来要跑 Docker backend，这里尤其重要

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

### 结合 `example/swe_slugfix`

这个 example 里：

- `workspace/toy_swe/slugify.py`：故意写坏的实现
- `oracle/toy_swe/slugify.py`：正确实现参考
- `verifier/tests/test_slugify.py`：隐藏测试入口
- `verifier/test.sh`：负责把 `workspace` 加到 `sys.path` 并运行 unittest
- `instruction.md`：只告诉 agent 修复 `slugify`
- `readme.md`：给人类审核者看的背景说明

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

## 示例

### `example/empty`

这是一个空模板 task，适合作为社区提交的起点。
它保留了完整目录骨架，但具体内容仍然是占位符。

### `example/swe_slugfix`

这是一个可运行的 toy SWE 修复任务，也是目前最推荐先参考的 example。
它用来验证：

- 框架能否跑通 `local + codex`
- workspace / oracle / verifier 的隔离是否正确
- 结果目录是否足够干净、可回放

## 常用命令

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m ptdbench validate example/swe_slugfix
PYTHONDONTWRITEBYTECODE=1 python3 -m ptdbench evaluate example/swe_slugfix --backend local --agent codex --model gpt-5.5 --json
```
