---
name: package-task
description: >-
  Packages a buggy training workspace into a PTDBench task with environment
  (shared image + prepare.sh), issue and solution evaluations, workspace,
  metadata.yaml, and readme.md.
  Use when the user asks to package, build, or convert a workspace into the
  task format under tasks/, for any model, framework, or dataset.
---

# Package a training workspace into a PTDBench task

把有问题的训练工作区收成 `tasks/<family>/<task_id>/`（新任务也可以直接放 `tasks/<task_id>/`）。适用所有模型、框架与数据集。不产生 `agent_trace.json`。

环境走**共享环境镜像**：任务不自带 Dockerfile，`metadata.environment.image` 写 `docker/images/` 下某个 `<name>.Dockerfile` 的名字，`environment/` 只有 `prepare.sh`（唯一联网阶段，下模型下数据）。

细节模板见 [reference.md](reference.md)。填好的输入示例见 [examples.md](examples.md)。仓库侧的权威说明是 `README.zh.md`（元数据 schema、prepare 阶段）、`docker/README.zh.md`（环境镜像）和 `ptdbench/tasks/validator.py`（校验规则）。

## 用户必填输入

缺一项就停。

- 来源工作区路径、目标目录、`task_id`、title
- 模型名称、数据集名称、框架、算法
- git 仓库路径、buggy commit、baseline commit；无关 hunk 没有则留空
- 启动脚本相对 `workspace/` 的路径、位置参数、必要环境变量
- 容器内模型路径、训练集路径、验证集路径
- 失败运行 logs 的路径；来源有 wandb 则保留
- 环境镜像名（`docker/images/` 下 `<name>.Dockerfile` 的 `<name>`，现有 `verl`、`slime-rc0`、`slime-v0.3.2`、`python3.11-slim`）
- 任务要用的卡（写进 `metadata.environment.env` 的 `CUDA_VISIBLE_DEVICES` / `NVIDIA_VISIBLE_DEVICES`）
- 每份资产：HF repo、40 位 commit sha revision、`--allow` 清单、需校验文件的 SHA-256 与字节数、`--dest` 落点；用内置配方时给 recipe 名与校验值
- 评测集键、出现位置（文字 log、离线 wandb 或两者）、一条含该键与数字的 log 样例
- `final_step`、`historical_final_baseline`、`required_final_score`

评分方式可省略。省略时分数全部看评测集键。若指定，则列出每一项参考分的键名、门槛、权重、出现位置、log 样例，以及及格规则；这些参考分不限于训练分。

`score_mode`（`best` 或 `final`）与 `required_final_score_max` 可省略，省略时按 `best` 且不设上限。

## 目录

```text
<task_id>/
├── environment/prepare.sh
├── environment/docker-compose.yaml          # 需要多卡或大 shm 且要走 harbor 时才有
├── evaluations/issue/instruction.md
├── evaluations/issue/oracle.md
├── evaluations/issue/verifier/judge_prompt.md
├── evaluations/issue/verifier/judge.py
├── evaluations/issue/verifier/test.sh
├── evaluations/issue/verifier/README.md
├── evaluations/issue/verifier/gold.yaml     # 可选，只给 reviewer 看
├── evaluations/solution/instruction.md
├── evaluations/solution/oracle/README.md
├── evaluations/solution/oracle/task.yaml
├── evaluations/solution/oracle/standard_answer.txt
├── evaluations/solution/oracle/historical_evidence_manifest.json
├── evaluations/solution/verifier/test.sh
├── evaluations/solution/verifier/README.md
├── evaluations/solution/verifier/resources/grader_spec.json
├── evaluations/solution/verifier/resources/README.md
├── evaluations/solution/verifier/tests/grade_submission.py
├── workspace/
├── metadata.yaml
└── readme.md
```

没有 `environment/Dockerfile`、`environment/requirements.txt`，也没有 `environment/hydrate-assets.sh`。`tasks/` 下已经没有任何任务留着它们（`tasks/reward_design/*` 那两个最后的 legacy 任务已于 2026-09-23 迁完）；新任务不要照抄。

## 流程

1. 对齐 buggy commit 的代码。跑 `git diff <buggy> <baseline>`，只保留代码与启动脚本（含它们直接引用的训练设定、超参、奖励与数据处理）。排除 `logs/`、`wandb/`、`outputs/`、`*.log`、checkpoint、权重与数据文件的不同。用户标成无关的 hunk 也排除。过滤后没有实质变更就停。
2. 原样复制 issue 的 `instruction.md`、`judge_prompt.md`、`judge.py`、`test.sh`，以及 solution 的 `verifier/test.sh`。来源见 [reference.md](reference.md)。复制后用 `cmp` 确认与来源一致，并保持 LF。
3. 用过滤后的 diff 重写 `evaluations/issue/oracle.md`、`evaluations/solution/oracle/README.md`、`task.yaml`、`standard_answer.txt`。`metadata.yaml` 的 `failure_reason` 与 `readme.md` 的 Canonical failure reason 也按这份 diff 写成几句话，并写上用户给出的指标现象。用户若另写了失败原因，不作为根因依据。
4. 用用户本次输入重写 `solution/instruction.md`、`grader_spec.json`、`resources/README.md`、`grade_submission.py`，并把失败运行的指标序列落成 `oracle/historical_evidence_manifest.json`。
5. 组 workspace、`environment/prepare.sh`、`metadata.yaml`、`readme.md`。
6. 用用户给的 log 样例跑一次提取，确认能抽出所填的键与数字。抽不到就改 `grade_submission.py`，直到样例通过。
7. 跑 `python3 -m ptdbench validate <task_dir>`，必须零 error。有 docker 和网络的机器上再用 `./docker/warm-cache.sh` 跑一遍 `prepare.sh`（它会按需构建镜像，并把全部任务的 prepare 各跑一次）。

## Issue 归因

根因只来自第 1 步的 diff。每一项写路径或符号、kind、observed（buggy）、expected（baseline）。`failure_reason` 由该 diff 写成一句话，不照抄用户自拟的摘要。

log、wandb、指标曲线只写进 Observable impact，不写成根因。

多于一个实质变更时，`oracle.md` 必须写上：

```text
Listed causes are all essential. Identifying only part of them must receive partial root_cause_score.
```

这句对齐 `judge_prompt.md`（多个根因只点出一部分要给部分分）。权重维持来源文件的 0.2 / 0.6 / 0.2，及格线 0.8。不要写「点出任一项即满分」。

`gold.yaml` 只是给 reviewer 的对照，机器评分读 `oracle.md`。写了就必须和 `oracle.md` 的 verdict、root_cause 一致。

## Solution

`instruction.md` 保留「在 `/workspace` 修好、前台跑完、不用网络、不 detach、数据与模型固定、log 只认 `$PTDBENCH_RESULTS/experiment_logs`、禁止编造指标」。启动脚本、参数、模型路径、训练集路径、验证集路径改成用户本次输入。步数与评测间隔被固定时，一并写进「The effective run must use」。

`grader_spec.json` 写入用户本次给的 `task_id`、评测集键、`final_step`、`historical_final_baseline`、`required_final_score`、`log_roots: ["experiment_logs"]`，需要时加 `score_mode` 与 `required_final_score_max`。用户指定了评分方式时，参考分按扁平的 `<name>_key` / `required_<name>` / `<name>_weight` 三件套写，不要用 `components` / `pass_rule`。

`grade_submission.py` 从这份 spec 读键与门槛，按用户给的 log 样例撰写提取。键出现在文字 log、离线 wandb，或两者，以用户输入为准。不要复制旧 task 的 `grade_submission.py`，也不要套用旧的 log 格式分类。

未指定评分方式时，分数全部看评测集：按 `score_mode` 取该键的最高有限值（`best`）或 `final_step` 那一点（`final`），达到 `required_final_score` 则 score 为 1，否则为 0。用户指定了参考分时，按该输入的权重与及格规则计算，不另行假设必须两项都过线。

输出 `experiment_metrics.json`、`metrics.json`、`result.json`（含 `passed` 与 `score`）。`resources/README.md` 用同一组键与门槛说明上述规则；`verifier/README.md` 再写一遍门槛怎么来的（对着历史曲线的哪几个点校准）。

`solution/oracle/README.md`、`task.yaml`、`standard_answer.txt` 按同一份 diff 重写。修复动作是把那些变更改回 baseline。`historical_evidence_manifest.json` 记那次失败运行的指标序列，是 reviewer 复核门槛的依据。

## Workspace

放入 buggy commit 的完整代码与启动脚本。保留用户指出的失败运行 logs。来源里有 wandb 就一并放入。

排除 `.git`、`__pycache__`、`agent_trace.json`、权重、checkpoint，以及将由 `prepare.sh` 下载的训练集与验证集文件。workspace 里没有那次失败运行的 log 就停。

启动脚本在 workspace 里的路径必须等于 `metadata.training.script`，否则 `validate` 报 error。

## Environment

`environment.image` 从 `docker/images/` 现有的镜像里挑，按框架与版本对上任务录制时的栈。镜像是多个任务共享的，内容就是它们的 ground truth 环境：不要为新任务改已有的 Dockerfile。现有镜像都不合适时，新加一个 `docker/images/<name>.Dockerfile`（`FROM` 钉 digest，末尾照抄其他 Dockerfile 的 PTDBench tooling 段），做法见 `docker/README.zh.md`。只缺一两个纯 Python 包的，在 `prepare.sh` 里 `pip install --target` 到 `$PTDBENCH_WORKSPACE` 下，再往 `$PTDBENCH_PREPARED/env` 写一行 `PYTHONPATH=...`（照 `tasks/reward_design/task_evaluator` 的做法；prepare 容器对镜像 site-packages 的改动带不到后两个阶段）。

`prepare.sh` 跑在独立的 prepare 阶段容器里：root、镜像自带的解释器、**唯一有网的阶段**，`$PTDBENCH_WORKSPACE` 可写。用 `ptdbench_fetch_hf` / `ptdbench_fetch_url` / `ptdbench_dataset` 三个命令取资产，每一份都要 `--expect` 与 `--expect-size` 硬校验，HF revision 必须是 40 位 commit sha。`--dest` 写到 `$PTDBENCH_WORKSPACE/<启动脚本实际读的路径>`，不要让任务去引用 `$PTDBENCH_MODELS` / `$PTDBENCH_DATASETS`——那两个只是缓存，harbor 后端跑完 prepare 就删。

要用的卡写在 `metadata.environment.env` 里，值必须是带引号的字符串。不在 `prepare.sh` 里设 `CUDA_VISIBLE_DEVICES`，也不写 `PTDBENCH_*`。

需要多卡或大 `/dev/shm` 且要走 harbor 后端时才加 `environment/docker-compose.yaml`；它对原生 docker 路径无效，照抄现有 slime 任务那份即可。

校验值或 revision 未给齐时，`prepare.sh` 保留明确待填字段，并在 `readme.md` 标成未发布。

## metadata.yaml 与 readme.md

字段结构见 [reference.md](reference.md)，可对照 `tasks/slime_retool` 与 `tasks/verl_implementation/llama_dapo_implementation/task_monkey_patch_011`。内容用本次输入与 diff 填写。

`training.script` 写相对 `workspace/` 的启动脚本路径，需要位置参数时加 `script_args`（列表，一项一个参数，全部带引号）。`environment` 写 `image` 与 `env`；旧的 `conda` / `requirements` / `dockerfile` 字段会被 validator 拒掉。

`readme.md` 的 Environment requirement 节写明用哪个环境镜像、为什么是它、`prepare.sh` 交付了哪几份资产、要几张卡。不要再写 External dataset asset 那种镜像下载表：资产大小与 SHA-256 只出现在 `prepare.sh`。

## 完成前检查

- 骨架齐全，且没有 `agent_trace.json`、没有 `environment/Dockerfile`、没有 `requirements.txt`、没有 `hydrate-assets.sh`
- issue 四档与 solution `test.sh` 与来源 `cmp` 一致
- oracle 根因与过滤后的 diff 一致；多根因时含部分分那句；写了 `gold.yaml` 则与 `oracle.md` 一致
- `grade_submission.py` 能从样例抽出用户给的键，且 spec 里的键它全都读
- workspace 含启动脚本与 logs；有 wandb 来源时 workspace 内也有
- `metadata.training.script` 指向的文件在 `workspace/` 里真的存在
- `metadata.environment.image` 是 `docker/images/` 下已有的名字
- `prepare.sh` 里每份资产都有 `--expect` 与 `--expect-size`，HF revision 都是 40 位 sha
- `python3 -m ptdbench validate <task_dir>` 零 error
