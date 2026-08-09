# PTDBench

PTDBench（PostTrainDiagnosisBench）是一个面向社区提交的后训练诊断基准框架。
它用来收集“带 bug 的 post-train 任务 + 环境 + verifier”，让指定的 model/agent 在受控工作区里修复问题并接受评测。

## 核心流程

1. 提交一个 case package，里面包含公开说明、可写 workspace、隐藏 oracle、环境镜像和 verifier。
2. 评测器把 `workspace/` 复制到 staging 目录。
3. 指定的 agent 在 staging workspace 上修改代码。
4. verifier 运行 `verifier/test.sh`，读取修复后的 workspace 并输出结果。
5. 评测器汇总 `report.json`，同时保留 agent 轨迹和 verifier 结果。

## 先跑 example

当前最值得先跑的是 `example/swe_slugfix`，它是一个最小可运行的 SWE 修复任务。
workspace 里有一个 `toy_swe.slugify.slugify()` 的 bug，agent 需要把它修好，让 verifier 测试通过。

### 1. 先做校验

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m ptdbench validate example/swe_slugfix
```

### 2. 再跑本地 + Codex

当前框架已经支持的主流程就是 `local + codex`：

- `--backend local`：评测和 verifier 都在本机跑
- `--agent codex`：用 Codex 作为 agent
- `--agent-backend local`：Codex 在本机直接启动

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m ptdbench evaluate example/swe_slugfix \
  --backend local \
  --agent-backend local \
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
self-contained verifier。Codex case 默认让 Judge 使用同一个
OpenAI-compatible model、base URL 和 API key；task verifier 仍可用
`PTDBENCH_JUDGE_*` 环境变量覆盖 Judge 设置。该转交不会把 key 写进
`report.json` 或 agent trace。Docker verifier backend 不转交 key，以免
secret 出现在 Docker argv 中。

直接写 `--api-key` 可能被同机进程列表看到；在不要求完全复现上述命令时，
仍建议从受保护的环境变量展开：

```bash
--api-key "$OPENAI_API_KEY"
```

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
example/your_case/
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

## 可见性约定

- `instruction.md`：agent 可见
- `workspace/`：agent 可写
- `environment/Dockerfile`：用于构建运行环境
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

- `case_id`
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
case_id: swe_slugfix_001
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
  run_command: n/a
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
- `training.run_command`：训练命令
- `environment.dockerfile`：环境 Dockerfile 路径
- `outcome.failure_type`：失败类型
- `outcome.failure_reason`：失败原因说明

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
   - 至少把 `case_id`、`title`、`submitter`、`license`、`created_date`、`domain`、`model`、`dataset`、`training`、`environment`、`outcome` 写全

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
- `PTDBENCH_CASE_ID`

`verifier` 需要：

- 成功时退出码为 `0`
- 失败时退出非 `0`
- 如有结构化结果，写入 `$PTDBENCH_RESULTS/result.json`

## 示例

### `example/empty`

这是一个空模板 case，适合作为社区提交的起点。
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
PYTHONDONTWRITEBYTECODE=1 python3 -m ptdbench evaluate example/swe_slugfix --backend local --agent-backend local --agent codex --model gpt-5.5 --json
```
