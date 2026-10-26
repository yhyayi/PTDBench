# verl_grpo_uid_after_repeat

| | |
| --- | --- |
| **Author** | benchmark_evaluation |
| **Domain** | `algorithm` |
| **Tags** | `grpo` `advantage` `uid` `group-baseline` `verl` `silent-degeneration` `issue-only` |
| **Framework** | verl `0.9.0.dev` @ `6f2aa24abfc01cc72811b376aeb3936503885096`，GRPO |
| **Model** | Llama-3.2-1B-Instruct |
| **Dataset** | DAPO-Math-17k |
| **Environment** | `verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12` 镜像；`prepare.sh` 交付权重与 parquet |
| **Resources** | 2 × H800 |
| **Recorded runs** | defect / control 各一次，2026-10-16 |
| **Recorded runtime** | 约 13.5 min（60 步） |

## Task abstract

`uid` 的赋值被挪到了 `batch.repeat()` **之后**，于是每条采样回复都拿到一个新 UUID
而不是继承其 prompt 的。`uid` 是 GRPO 分组的键，每次分组都退化成单元素，而
`len(group) == 1` 的分支会**静默**把 `id2mean` 置 0、`id2std` 置 1 而不是报错。
组基线从此不再被减去，GRPO 退化为无基线的 REINFORCE。

曲线看起来基本正常：60 步内 held-out 准确率与对照组在噪声范围内
（0.0570 vs 0.0592）。诊断只能在优势统计里看到。

本任务是 **issue-only**。

## The defect

```text
verl/trainer/ppo/ray_trainer.py
  -  uid assignment above  `gen_batch = self._get_gen_batch(batch)`
  +  uid assignment below  `batch = batch.union(gen_batch_output)`
```

移动是逐字的，注释一并搬走，所以暂存树里训练循环仍只有一处 `uid` 赋值、文件里仍有
两个 `uuid.uuid4()` 调用点，与上游一致。

### Why nothing catches it

- **计数查不出来。** 赋值语句的数量、位置之外的形式都没变，只有相对
  `batch.repeat` 的顺序错了。
- **不抛异常。** `len(group) == 1` 是 `compute_grpo_outcome_advantage` 里一条合法的
  分支，只是它的回退值让基线消失。
- **held-out 不塌。** 60 步预算下 defect 臂甚至略高于起点（0.0454 → 0.0570），
  与对照组（0.0592）差距在采样噪声内。靠准确率找缺陷的答案什么也没找到。

## Evidence

两臂都跑了，除 `uid` 移位外完全相同；对照臂的 `ray_trainer.py` 与 pristine upstream
逐字节一致（sha256 `e636912ae2be46b6ff7152f6473feab3af5db962f744903dc55f7d26d58d7d73`）。

| metric | defect 臂 | control 臂（= upstream） |
| --- | --- | --- |
| `critic/advantages/min` < 0 的步数 | **0 / 60** | 29 / 60 |
| 最负的 `critic/advantages/min` | **0.000000** | −0.935413 |
| 非零 `critic/advantages/max` 的取值 | **只有一个：`0.9999990463`** | 三个：`1.2076`、`1.6202`、`2.4749` |
| `actor/pg_loss` 范围 | **[−0.0674, 0]**，从不为正 | [−0.0139, +0.0366] |
| `actor/entropy` 首 → 末 | 4.81 → **1.31** | 4.42 → 2.99 |
| `val-core/…/acc/mean@1` @ 0/20/40/60 | .0454 / .0437 / .0476 / .0570 | .0454 / .0542 / .0504 / .0592 |
| 有奖励的步数 | 33 / 60 | 29 / 60 |

两条可读的事实，**只靠 defect 日志就够**：

1. **`critic/advantages/min` 在全部 60 步上恰好为 `0.0`。** 组内同时存在正确与错误
   回复时，GRPO 不可能产出全非负的优势批次；对照组 60 步里有 29 步为负。
2. **每个有奖励的步，`critic/advantages/max` 逐位相同**：`0.9999990463256836`，
   即 fp32 的 `1/(1+1e-6)`——原始奖励 `1.0` 除以被替换的 `id2std = 1.0` 再加
   `epsilon = 1e-6` 保护。对照组有三个不同的最大值，每个对应一种组内构成。

## Reference answer

### `issue` — the diagnosis

把 `uid` 赋值移回 `self._get_gen_batch(batch)` 之上。诊断必须点出赋值相对于
`batch.repeat` 的**位置**。

只说"组大小为 1"或"优势没有归一化"而没有定位到 `uid` 赋值的位置，属于不完整，
不得拿满 `root_cause_score`。症状在日志里一 grep 就有，原因还隔着一层静默分支。

必须判不过的归因：`train_batch_size`、`rollout.n`、学习率、奖励函数、缺少 KL loss、
以及裸的"held-out 准确率低"——这些都和健康配方一致。

### `solution`

不存在。见下。

## Verification

共享 LLM judge 比对 `$PTDBENCH_RESULTS/agent_answer.md` 与隐藏 `oracle.md`，
verdict / root cause / evidence 权重 0.2 / 0.6 / 0.2；通过要求 verdict_score == 1.0 且 root_cause_score、evidence_score 均 > 0.8，不是加权总分过 0.8。单一根因，
`oracle.md` 不含部分分条款。

**为什么这是 issue-only：** 两臂都跑完了，缺的不是数据，而是**这个配方在 60 步预算
下没有任何指标能分开两臂**。held-out 0.0570 vs 0.0592，训练奖励 0.0161 vs 0.0156。
按其他任务的形状去读 `experiment_logs` 只会读到噪声。

真正能分开两臂的量是结构性不变量而非分数。若将来加 solution 档，应当直接读新 run
的 metric JSONL：

- `min(critic/advantages/min) < 0`（跨评分步）
- `critic/advantages/max` 出现多于一个不同取值
- 每个固定步都在（否则 3 步的 run 因为没有有奖励的步而 `max` 合法为 0 也能过）

这与仓库现有 `grade_submission.py` 形状不同，所以没有强行拼上去。"延长预算直到
准确率分开"没有被测量过，不应假设；2 GPU 上 60 步是 13.5 分钟，400 步标定对约需
3 GPU 小时。

## Running it

defect 臂，在 `workspace/` 下：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash examples/grpo_trainer/run_llama3_2_1b_instruct_dapo_math_fsdp.sh
```

control 臂：另复制一份 workspace，把 `verl/trainer/ppo/ray_trainer.py` 里那段四行
`batch.non_tensor_batch["uid"] = ...` 移回 `gen_batch = self._get_gen_batch(batch)`
之上，跑同一条命令。不涉及启动脚本或配置改动，两份日志只差这个缺陷。用
`sha256sum` 核对修复后文件应为
`e636912ae2be46b6ff7152f6473feab3af5db962f744903dc55f7d26d58d7d73`。对照臂的代码与
日志不要放进 `workspace/`。

逐 step 指标落在 `logs/metrics_<timestamp>.jsonl`，**这才是要读的文件**：启动脚本
通过 `VERL_FILE_LOGGER_PATH` 开了 verl 的 `file` backend，因为 `console` backend 的
指标行打在 TaskRunner Ray actor 内，而 Ray 对该 actor stdout 的转发并不可靠——本镜像
上第一次录制就产出了零条 `step:N` 的日志，而 Ray worker 日志里有全部 61 条。`file`
backend 不缓冲、不依赖 Ray。

让 run 跑完前先看 step 1：`critic/advantages/min` 决定整个实验。

镜像用 `verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12`
（`docker/images/verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12.Dockerfile`），
verl + vLLM 0.17、torch 2.10 /
CUDA 12.9 / Ubuntu 22.04——同一 commit 的兄弟任务在这上面录过。CPU-only 的
`rank-distillation-cpu` 跑不了 vLLM rollout，slime 镜像是另一个框架。

`environment/prepare.sh` 交付 Llama-3.2-1B-Instruct 权重到
`workspace/verl/Llama-3.2-1B-instruct`（注意小写 `i`，启动脚本就是这么找的）以及
DAPO-Math-17k 的 train/test parquet 到 `workspace/DAPO-Math-17k_verl`，均按 commit
sha 与校验和钉死。其非注释行与兄弟任务逐字节相同，两个任务共用一份数据集缓存。

`environment/` 以只读挂进 agent 容器（`/ptdbench/environment`），所以里面只有
`prepare.sh`——任何会泄露注入缺陷的记录都不能放进 `environment/`。

2 GPU，由 `metadata.yaml` 的 `environment.env` 声明。
