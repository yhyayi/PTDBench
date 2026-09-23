# Package task reference

## 必填输入

缺一项就停。

### 任务位置

- 来源工作区路径
- 目标目录：`tasks/<family>/<task_id>/`，或顶层 `tasks/<task_id>/`
- `task_id`、title
- 模型名称、数据集名称、框架、算法

`failure_reason` 由过滤后的 diff 写成一句话，并带上用户给出的指标现象。

### 两次 commit

- git 仓库路径
- buggy commit（workspace 对齐这一次）
- baseline commit（预期正确状态）
- 要排除的无关 hunk（没有就留空）

归因命令：

```bash
git diff <buggy> <baseline>
```

只保留代码与启动脚本，以及它们直接引用的训练设定、超参、奖励与数据处理。排除：

- `logs/`、`wandb/`、`outputs/`、`*.log`
- checkpoint、`*.safetensors`、`*.bin`、`*.pt`、`*.ckpt`
- 训练集与验证集数据文件（parquet、jsonl 数据等）

### 本次训练如何启动

- 启动脚本相对 `workspace/` 的路径（就是 `metadata.training.script`）
- 位置参数（就是 `metadata.training.script_args`）与必要环境变量
- 容器内模型路径
- 容器内训练集路径、验证集路径
- 步数与评测间隔（被固定时要写进 `solution/instruction.md`）
- 失败运行 logs 在来源工作区中的路径
- 来源是否包含 wandb；有则保留

### 环境与资产

- conda 环境名：`docker/envs/` 下的目录名，目前 `verl`、`slime-rc0`、`slime-v0.3.1`
- 该环境的 `requirements.txt` 来源任务（同名环境下必须逐字一致）
- 要用的卡：`CUDA_VISIBLE_DEVICES` / `NVIDIA_VISIBLE_DEVICES`
- 每份 HF 资产：repo、40 位 commit sha revision、`--type`（`model` 或 `dataset`）、`--allow` 清单、需校验文件的 SHA-256 与字节数、`--dest`
- 用内置配方的数据集：recipe 名（`dapo_math_17k`、`gsm8k`、`search_r1_nq_hotpotqa`、`wiki18_e5`）、产物文件的 SHA-256 与字节数、`--dest`
- 单个 URL 的资产：URL、SHA-256、字节数、是否 `--extract`
- 是否需要 `docker-compose.yaml`（多卡或大 `/dev/shm`，且要走 harbor 后端）

不收集基础镜像：统一镜像由 `docker/` 构建，任务不再自带 Dockerfile。

### 评分键（用户本次输入）

评测集键始终要给：

- 评测集键名称
- 该键出现处：文字 log、离线 wandb，或两者
- 一条含该键与数字的真实 log 样例
- `final_step`
- `historical_final_baseline`
- `required_final_score`
- 可选：`score_mode`（`best` 或 `final`，默认 `best`）、`required_final_score_max`

评分方式可以不给。不给时，分数全部看评测集键。

用户指定评分方式时，每一项参考分都要给键名、门槛、权重、出现位置和一条 log 样例，并给出及格规则。参考分可以是训练分，也可以是学习率等其他日志分数。填写示例见 [examples.md](examples.md)。

### 历史证据

- 那次失败运行的评测点序列（step、值），用来写 `historical_evidence_manifest.json` 和校准门槛

## 原样复制

复制后 `cmp` 必须一致，换行保持 LF。

| 目标 | 来源 |
| --- | --- |
| `evaluations/issue/instruction.md` | `tasks/verl_implementation/llama_dapo_implementation/task_monkey_patch_011/evaluations/issue/instruction.md` |
| `evaluations/issue/verifier/judge_prompt.md` | 同上目录 `verifier/judge_prompt.md` |
| `evaluations/issue/verifier/judge.py` | 同上目录 `verifier/judge.py` |
| `evaluations/issue/verifier/test.sh` | 同上目录 `verifier/test.sh` |
| `evaluations/solution/verifier/test.sh` | `.../task_monkey_patch_011/evaluations/solution/verifier/test.sh` |

`tasks/slime_retool` 里这五份与 `task_monkey_patch_011` 逐字节相同，拿哪边都行。

`environment/docker-compose.yaml` 需要时照抄 `tasks/slime_retool/environment/docker-compose.yaml`。

## issue/oracle.md

```markdown
# Canonical issue oracle

Reconciled from the task's frozen evidence and the diff between `<buggy>` and `<baseline>`.

Expected verdict: **unhealthy**.

## Root cause

- `<path-or-symbol>`
  - kind: `<symbol|hparam|script>`
  - observed: `<buggy side>`
  - expected: `<baseline side>`

## Observable impact

- Primary source evaluation metric: `<metric_key>`.
- <来自 logs 的现象，不另立为根因>

## Trigger conditions

- <diff 里的触发条件>

## Documentation reconciliation

- <哪些 log 现象是症状>
```

多于一个 Root cause 时，在该节末尾加上：

```text
Listed causes are all essential. Identifying only part of them must receive partial root_cause_score.
```

## issue/verifier/README.md

固定说明这一档怎么评，照 `tasks/slime_retool/evaluations/issue/verifier/README.md` 写：judge 读 `$PTDBENCH_RESULTS/agent_answer.md` 比对隐藏的 `oracle.md`，权重 0.2 / 0.6 / 0.2，及格 0.8，需要 `PTDBENCH_JUDGE_MODEL`、`OPENAI_BASE_URL`、`OPENAI_API_KEY`，结果写 `$PTDBENCH_RESULTS/result.json`。

## issue/verifier/gold.yaml（可选）

只给 reviewer 看，机器评分不读它。写了就必须和 `oracle.md` 一致：

```yaml
category: <implementation|hyperparameter|reward>
framework: <framework>
task: <task_id>
verdict: <healthy|unhealthy>
root_cause:
  - <一条根因，和 oracle.md 的 Root cause 对应>
evidence_metrics:
  - <一条可观测证据>
fix:
  must:
    - <必须做到的修复动作>
```

## solution/instruction.md

开头至「The dataset, held-out set, ... are fixed」沿用 `task_monkey_patch_011` 的 `evaluations/solution/instruction.md`。其后改成用户本次输入：

```markdown
The launch script is:

```text
<launch-script-relative-path>
```

<用户给的参数或调用范例>

The effective run must use:

- model `<容器内模型路径>`
- train data `<训练集路径>`
- validation data `<验证集路径>`
- <被固定的步数与评测间隔，例如 `--num-rollout 100`、`--eval-interval 10`>
- `LOG_DIR=$PTDBENCH_RESULTS/experiment_logs`
- `WANDB_MODE=offline`
- `WANDB_DIR=$PTDBENCH_RESULTS/experiment_logs`

Create `$PTDBENCH_RESULTS/experiment_logs` first. Offline W&B data must be
created below `$PTDBENCH_RESULTS/experiment_logs/wandb`. The run is scored only
from files under `$PTDBENCH_RESULTS/experiment_logs`; logs written anywhere
else are ignored. Do not copy, edit, or fabricate log metrics. Otherwise the
run is scored 0.
```

## solution/oracle/README.md

用同一份过滤后的 diff 填写。机器评分段落只重复用户本次的键、门槛与权重。

```markdown
# Canonical solution oracle

## Root cause

- `<path-or-symbol>`
  - kind: `<kind>`
  - observed: `<buggy>`
  - expected: `<baseline>`

## Observable impact

- Primary source evaluation metric: `<metric_key>`.

## Trigger conditions

- <同 issue oracle>

## Required remediation

- `<path>`: 改回 baseline 的 `<expected>`。

Machine grading reads only the new experiment logs.
<按 grader_spec 写明键、门槛、权重与及格规则>
```

## solution/oracle/task.yaml

```yaml
task_id: <task_id>
category: <family 或 recipe 分类>
framework: <framework>
launch_script: <相对 workspace/ 的启动脚本>
model: <model name>
dataset: <dataset name>
algo: <algo>
expected_verdict: <healthy|unhealthy>
evidence_kind: <historical_run|dirty_agent_failure>
buggy_commit: <sha>
baseline_commit: <sha>
eval_metric_key: <metric_key>
last_eval_step: <final_step>
causes:
  - path: <repo-relative path>
    kind: <symbol|hparam|script>
    observed: <buggy>
    expected: <baseline>
key_files:
  - <diff 命中的文件>
external_assets_not_in_git:
  - <由 prepare.sh 交付的资产，一句话说明>
```

同一框架已有更完整的 `task.yaml` 时（例如 `task_monkey_patch_011` 的 hollow-file 形态），沿用该骨架，仍须把本次 diff 的每一项根因写进去。

## solution/oracle/standard_answer.txt

judge 的硬判据，纯文本，固定抬头：

```text
STANDARD_ANSWER_V1
CASE_ID: <task_id>
BENCHMARK_GROUP: <B-...>
EXPECTED_VERDICT: <healthy|unhealthy>
EXPECTED_ATTRIBUTION: <一句话，diff 指向的根因；healthy 时写 none 及理由>
PRIMARY_METRIC: <metric_key>

REQUIRED_ALL:
- <每一项必须点到的东西；多根因就逐条列>

GLOBAL_JUDGE_POLICY:
- The score must be exactly 0 or 1.
- Agent A must explicitly classify the run as healthy or unhealthy.
- Every REQUIRED_ALL item must be satisfied.
- A verdict mismatch forces score 0.
- Equivalent technical wording in any language is acceptable.
- task.yaml may clarify targets and aliases but may not relax REQUIRED_ALL.
```

`healthy` 任务另加一条：把良性但少见的设定报成缺陷即 score 0。

## solution/oracle/historical_evidence_manifest.json

那次失败运行的指标序列，reviewer 用它复核门槛：

```json
{
  "schema_version": 1,
  "evidence_kind": "dirty_agent_failure",
  "task_id": "<task_id>",
  "family": "<family>",
  "metrics": {
    "metric_status": "log_reconstructed",
    "expected_metric_key": "<metric_key>",
    "actual_metric_key": "<metric_key>",
    "eval_points": [
      { "step": 0, "eval_score": 0.0307, "eval_key": "<metric_key>", "truncate_ratio": null }
    ]
  }
}
```

## grader_spec.json

```json
{
  "schema_version": 2,
  "task_id": "<task_id>",
  "metric_key": "<用户给的评测键>",
  "final_step": 0,
  "historical_final_baseline": "<用户给的值>",
  "required_final_score": "<用户给的值>",
  "log_roots": ["experiment_logs"]
}
```

可选键：`score_mode`（`best` 或 `final`，`grade_submission.py` 默认 `best`）、`required_final_score_max`（上限，挡掉编造出来的离谱值）。数值一律写成字符串，由 `Decimal` 读。

未指定评分方式时不要追加其他分数键。用户指定了参考分时，在同一份 `grader_spec.json` 里按**扁平三件套**追加，不要用 `components` / `pass_rule`：

```json
{
  "training_reward_key": "<参考分键>",
  "required_training_reward": "<门槛>",
  "validation_weight": "0.8",
  "training_weight": "0.2"
}
```

现有任务用过的几组命名：`training_reward_key` / `required_training_reward` / `training_weight`（`reward_design/*`、`rlve_hyper/task_012`），`learning_rate_key` / `required_learning_rate` / `learning_rate_weight`（`rlve_hyper/task_008`、`task_019`）。评测集那一侧一律是 `validation_weight`。取名照参考分的性质来，并在 `grade_submission.py` 的必填键清单里列全。

`grade_submission.py` 读取同层 `../resources/grader_spec.json`。用样例字符串自测：必须抽出用户给的键与样例中的数字。输出：

- `experiment_metrics.json`
- `metrics.json`
- `result.json`：`{"passed": <bool>, "score": <number>, "metrics": {...}}`

`resources/README.md` 说明入口是 `test.sh`，分数只来自 `$PTDBENCH_RESULTS/experiment_logs`，并列出与 spec 相同的键和门槛。`verifier/README.md` 再写一层：这个键读的是哪个空间（score 还是 accuracy）、`score_mode` 为什么这么选、门槛对着历史曲线的哪几个点校准、最薄的余量是多少。

## environment/requirements.txt

该 conda 环境的**完整闭包**，第一行注明是哪个环境、对应哪个上游镜像：

```text
# PTDBench conda env `<conda-name>`.
#
# <这个闭包是从哪来的：wandb 记录的 pip freeze、上游 requirements、还是镜像自带>
--extra-index-url https://download.pytorch.org/whl/cu129
<pkg>==<version>
...
```

同一 `conda` 名下的所有任务给出的这份文件必须**逐字一致**，`python3 docker/collect_envs.py --check`（`docker/build.sh` 会跑）不一致就 fail。直接从已有任务复制，不要取并集——加一个任务就会悄悄改掉另一个任务的环境，连带改掉它的 ground truth。真要换依赖就开新环境：加 `docker/envs/<name>/` 和 `docker/Dockerfile` 里对应的 `env-<name>` stage。

不要把框架本身（verl / slime）写进去：它来自任务 workspace 里的 checkout。

## environment/prepare.sh

```bash
#!/usr/bin/env bash
# <task_id> —— 训练前要准备的东西。
#
# 跑在独立的 prepare 阶段容器里：root、已激活 `<conda-name>` 环境、**唯一有网的
# 阶段**，而且 $PTDBENCH_WORKSPACE、$PTDBENCH_MODELS 和 $PTDBENCH_DATASETS 都可写。
set -euo pipefail

# --- 基座模型 ---------------------------------------------------------------
# <为什么 dest 必须正好是这条路径：启动脚本第几行的哪个守卫在找它>
ptdbench_fetch_hf \
    --repo <org/repo> \
    --revision <40 位 commit sha> \
    --allow config.json \
    --allow tokenizer.json \
    --allow model.safetensors \
    --expect model.safetensors=<sha256> \
    --expect-size model.safetensors=<bytes> \
    --dest "$PTDBENCH_WORKSPACE/<模型相对路径>"

# --- 数据集 -----------------------------------------------------------------
ptdbench_dataset \
    --recipe <dapo_math_17k|gsm8k|search_r1_nq_hotpotqa|wiki18_e5> \
    --expect train.parquet=<sha256> \
    --expect-size train.parquet=<bytes> \
    --expect test.parquet=<sha256> \
    --expect-size test.parquet=<bytes> \
    --dest "$PTDBENCH_WORKSPACE/<数据集相对路径>"
```

三个命令：

| 命令 | 用途 | 必给 |
| --- | --- | --- |
| `ptdbench_fetch_hf` | HF repo 按 allow 列表取 | `--repo`、`--revision`（40 位 sha）、`--dest`；`--type dataset` 取数据集仓 |
| `ptdbench_fetch_url` | 单个 URL | `--url`、`--sha256`、`--dest`；压缩档加 `--extract` |
| `ptdbench_dataset` | 镜像内置配方 | `--recipe`、`--dest` |

三个都支持可重复的 `--expect PATH=SHA256` 与 `--expect-size PATH=BYTES`，每份资产至少校验一个主文件。校验不过是**硬失败**，不降级重试。

可用变量：

| 名字 | 是什么 |
| --- | --- |
| `$PTDBENCH_WORKSPACE` | agent 会拿到的 workspace，prepare 阶段可写；资产下到这里 |
| `$PTDBENCH_PREPARED` | 产物目录；往 `$PTDBENCH_PREPARED/env` 写 `KEY=VALUE` 行可给 agent/verifier 传环境变量（按数据逐行读，不会被 source） |
| `$PTDBENCH_MODELS` / `$PTDBENCH_DATASETS` | 跨 run 共享的下载缓存，agent/verifier 阶段只读；**不要让任务脚本直接引用**，harbor 后端跑完 prepare 就把它们删掉 |
| `$PTDBENCH_ENVIRONMENT` | 任务自己的 `environment/`（只读） |

要在 prepare 阶段做格式转换（例如 HF → Megatron torch_dist）时，先探测 CUDA 是否可用，不可用就**跳过而不是报错**——harbor 后端把 prepare 跑成 build-time 层，那里没有 GPU，留给训练脚本自己的守卫在 agent 阶段补做。

## metadata.yaml

```yaml
title: '<family> diagnosis: <task_id>'
submitter:
  name: benchmark_evaluation
license: MIT
created_date: '<YYYY-MM-DD>'
domain: post_train
model:
  name: <model name>
dataset:
  name: <dataset name>
training:
  framework: <framework>
  algo: <algo>
  script: <相对 workspace/ 的启动脚本路径>
  script_args:          # 只有启动脚本要位置参数时才写
    - '<arg>'
environment:
  conda: <verl|slime-rc0|slime-v0.3.1>
  env:
    CUDA_VISIBLE_DEVICES: "0,1"
    NVIDIA_VISIBLE_DEVICES: "0,1"
outcome:
  failure_type: <见下>
  failure_reason: <由过滤后的 diff 写成的一句话，并带上用户给出的指标现象>
```

`failure_type` 只要求非空。`README.zh.md` 建议的几类是 `training_collapse`、`training_failure`、`reward_over_optimization`、`normal_training`；现有任务另外用过 `implementation_bug`、`reward_hacking`、`rollout_loss_mask`、`implementation_contract_regression`、`run_health_unknown`（无注入缺陷、只跑 issue 档的任务）。能对上就用现成的，对不上再自拟。

validator 会拒掉的写法：

- `conda` 和 `dockerfile` 同时写
- `conda` 不在 `docker/envs/` 里
- `script` 是绝对路径、含 `..`，或指向 `workspace/` 里不存在的文件
- 有 `script_args` 但没有 `script`；`script_args` 不是列表；列表项不是字符串（YAML 里 `- 0` 是 int，要写 `- '0'`）
- `env` 的值不是字符串（`CUDA_VISIBLE_DEVICES: 0` 会被拒），或设了 `PTDBENCH_*`

声明了 `framework` 却没写 `script` 只给 warning，但任务卡和 `docker/debug-env.sh run` 都靠它，别省。

## readme.md 要有的节

- Case overview：evaluation key、family、schema version `2`、levels `issue` 与 `solution`、模型、数据集、框架、主要指标键
- Canonical failure reason（healthy 任务写 Canonical verdict）
- Workspace and hidden evaluations（workspace 为历史快照；issue 对 `oracle.md` 做 LLM judge；solution 只读新 log）
- Source and preserved context：buggy / baseline commit、过滤后的根因文件
- Original failure evidence（来自既有 logs 的现象）
- Canonical expected outcome：verdict 与 diff 根因、修复目标
- Evaluation design：issue 权重 0.2 / 0.6 / 0.2、及格 0.8；solution 规则与 `grader_spec.json` 一致，含门槛怎么校准的
- 启动脚本路径，以及 workspace 内失败 logs 的位置
- Environment requirement

```markdown
## Environment requirement

The task runs on the unified PTDBench image, in the `<conda-name>` conda
environment (`environment: conda: <conda-name>`) — <这个 stack 的一句话描述>。
<为什么不是另一个环境>。See `docker/envs/README.md`.

`environment/prepare.sh` delivers <几份资产，逐条一句话>，all pinned by commit
sha and checksummed.

The recipe needs **<N> GPUs**; `metadata.yaml` declares them through
`environment.env`.
```

不要再写 External dataset asset 那种镜像下载表。资产的大小与 SHA-256 只出现在 `prepare.sh`。

校验值或 revision 未给齐时，在 Environment requirement 里写一行 `Publication status: unpublished`，并在 `prepare.sh` 对应位置留明确待填字段。
