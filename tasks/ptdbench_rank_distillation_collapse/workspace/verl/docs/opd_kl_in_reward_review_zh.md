# verl 中 KL-in-Reward OPD 实现审查

审查对象是 `/workspace/verl` 中的 `verl 0.9.0.dev`。这里的 OPD 不是原有的
`distillation.*` 独立教师集群路径，而是新增的 `opd.*` 共置教师路径：教师作为
actor resource pool 内的 forward-only engine worker，与学生轮流使用同一组 GPU。

## 结论

数学核心基本成立，但当前实现还不能判定为端到端正确。

- `top_k=0` 的 sampled-token OPD、四种 top-k candidate strategy、稀疏
  `log_softmax + gather`、PPO ratio/clipping 和 packed tensor 对齐，在 CPU 单测覆盖
  范围内是自洽的。
- `student_p` 在候选集等于全词表时还原
  `grad KL(p_student || p_teacher)`；严格 top-k 且 `is_norm=True` 时，advantage、
  PPO ratio 和带梯度的 score 都使用候选集条件分布，因而在固定候选集、
  old policy 处给出 `grad KL(p_student(.|C) || p_teacher(.|C))` 的精确梯度。
  `is_norm=False` 时仍是对 full-vocab reverse-KL 的稀疏近似；sampled-token 路径不受影响。
  `teacher_p` 和 `none` 是受支持的加权策略，但不是标准 reverse-KL 梯度。
- 仓库自带 GPU smoke 在默认 legacy trainer (`trainer.use_v1=False`) 的第一步即失败，
  目前没有通过完整训练链的证据。
- `opd.grpo_outcome_weight` 已暴露在配置和文档中，但没有任何运行时代码读取；在
  `opd_token_kl_plus_grpo` 下调整它不会改变结果。这是确定的功能缺口。

## 支持的参数

### OPD 专用参数

定义位置：[verl/workers/config/opd.py](../verl/workers/config/opd.py:40)，默认 Hydra 配置见
[opd.yaml](../verl/trainer/config/opd/opd.yaml:7)。

| 参数 | 默认值 | 当前行为 |
|---|---:|---|
| `opd.enable` | `False` | 打开共置 OPD。开启后必须设置教师路径，并使用 OPD advantage estimator。 |
| `opd.teacher_model_path` | `null` | 教师 HF/本地模型路径；`enable=True` 时必填。教师输入直接使用学生 token id。 |
| `opd.top_k` | `0` | `0`：只对 rollout 采样 token 计算 OPD；`>0`：每个 response position 建立 top-k 候选集。不能为负。 |
| `opd.strategy` | `only_stu` | `only_stu`、`only_tch`、`intersection`、`union`。`top_k=0` 时被忽略。 |
| `opd.reward_weight_mode` | `student_p` | `student_p`：按 student probability 加权；`teacher_p`：按 teacher probability 加权；`none`：有效候选均匀权重。`is_norm=False` 时使用原始 full-vocab probability，`is_norm=True` 时使用候选集内重归一化 probability。 |
| `opd.is_norm` | `False` | 仅在 `top_k>0` 时生效。开启后 student/teacher 在有效候选集内各自归一化；advantage、actor PPO ratio 和 score gradient 使用同一条件分布。`top_k=0` 不受影响。 |
| `opd.outside_bucket` | `False` | 将候选集之外的全部 token 聚合成一个 bucket，并把它纳入 coarse reverse-KL。仅支持 `top_k>0`、`is_norm=False`、`reward_weight_mode=student_p`。不增加模型前向。 |
| `opd.teacher_temperature` | `1.0` | 教师 logits 除以该值后再 `log_softmax`；必须大于 0。学生侧使用 rollout temperature。 |
| `opd.chunk_size` | `4096` | 稀疏 vocab reduction 的 packed row chunk 大小；必须大于 0，越小峰值显存越低但前向更慢。 |
| `opd.kl_estimator` | `k1` | 目前只实现 `k1`，即使用 student/teacher 的 log-ratio；`is_norm=True` 时输入为候选集内重归一化后的 log-prob。其他值启动时 `NotImplementedError`。 |
| `opd.grpo_outcome_weight` | `1.0` | 设计意图是控制 `opd_token_kl_plus_grpo` 的 task-reward GRPO 项，但当前未被读取，实际无效。 |

### 必须配套的通用参数

| 参数 | 要求 |
|---|---|
| `algorithm.adv_estimator` | 必须是 `opd_token_kl`（纯 OPD）或 `opd_token_kl_plus_grpo`（OPD + task reward）。 |
| `algorithm.use_kl_in_reward` | 必须为 `False`。该开关是普通 reference-policy KL reward；共置 OPD 已经自己产生 KL 信号，二者同时打开会被拒绝。 |
| `actor_rollout_ref.actor.strategy` | 只支持 `fsdp` / `fsdp2`；Megatron、VeOmni 等后端会在启动校验时拒绝。 |
| `actor_rollout_ref.model.use_remove_padding` | 必须为 `True`；OPD 依赖 packed no-padding 路径。 |
| `actor_rollout_ref.model.use_fused_kernels` | 必须为 `False`；fused log-prob kernel 不物化原始 vocab logits，无法建立候选集。 |
| `actor_rollout_ref.teacher.*` | 使用 `dp_teacher` 配置组。教师为 forward-only，通常开启 `fsdp_config.param_offload=True` 和 dynamic batch，并设置 `log_prob_max_token_len_per_gpu` 或 micro-batch size。 |
| `trainer.v1.trainer_mode` | 使用 v1 trainer 时必须是 `sync`；异步更新可能在 sparse old-policy passes 之间改变学生权重。 |
| `uid` | `opd_token_kl_plus_grpo` 沿用 GRPO group baseline，必须有 uid；纯 `opd_token_kl` 不需要 group baseline。 |
| reward pipeline | 即使纯 OPD 不使用 task reward，当前 trainer 仍会执行 `extract_reward` 并读取 `rm_scores`；数据/默认 reward 配置仍需能产出该字段。 |

教师和学生的 vocab size 会检查是否相等，但当前只比较 `vocab_size`，没有比较 tokenizer 的
token-to-id 映射、special token 或 chat template。使用不同 tokenizer 时，即使词表大小相同，
也可能把同一个 id 当成不同 token 来蒸馏。

## 数学与数据流

对每个 response position `t`，代码维护候选 id 集 `C_t`，以及旧学生策略和教师策略在
这些 id 上的 log-prob：

```text
delta[t,k] = log p_student_old(t,k) - log p_teacher(t,k)
w[t,k]     = p_student_old(t,k) or p_teacher(t,k), depending on reward_weight_mode
             (candidate-renormalized if is_norm=True)
A[t,k]     = -w[t,k] * delta[t,k]
r_kl[t]   = sum_k A[t,k]
```

其中 `score` 由 `reward_weight_mode` 决定；概率权重是否在候选集内归一化由
`is_norm` 控制。实现位于
[advantage.py](../verl/trainer/opd/advantage.py:148)。padding 和 response mask 会把无效
position/candidate 置零；`intersection` 在某个 position 为空时返回全零，不产生 NaN。

当 `top_k>0 && is_norm=True` 时，`delta` 改为候选集条件分布的 log-ratio：

```text
log p_student_C[t,k]
  = log p_student[t,k] - logsumexp_j(log p_student[t,j]), j in valid C_t
log p_teacher_C[t,k]
  = log p_teacher[t,k] - logsumexp_j(log p_teacher[t,j]), j in valid C_t
delta[t,k] = log p_student_C[t,k] - log p_teacher_C[t,k]
```

student 和 teacher 分别归一化；`student_p`/`teacher_p` 直接取对应 log-prob 的
`exp()` 作为权重。`opd/student_mass`
和 `opd/teacher_mass` 仍报告归一化前的 full-vocab 概率在候选集上的覆盖质量。

### sampled-token (`top_k=0`)

候选集只有 rollout 实际采样的 token，`W=1`，因此 `w=1`：

```text
A = -(log p_student_old(sampled) - log p_teacher(sampled))
```

`is_norm` 在该路径上被忽略，避免单元素集合重归一化后 student/teacher 都变成
`log 1=0` 而丢失 sampled-token 信号。

driver 把该值写到二维 `opd_token_rewards`，随后
`opd_token_kl` 直接把它当 advantage；`opd_token_kl_plus_grpo` 再加上 group-normalized
task reward。

### top-k (`top_k>0`)

候选集及所需前向如下：

| strategy | 候选集 | 每步额外前向 | candidate 宽度 |
|---|---|---:|---:|
| `only_stu` | 学生 top-k | 学生 old-policy pass + 教师在学生 id 上 gather | `k` |
| `only_tch` | 教师 top-k | 教师 top-k prepass + 学生在教师 id 上 gather | `k` |
| `intersection` | 学生/教师 top-k 的 id 交集 | 学生 top-k pass + 教师 top-k/gather pass | `k`，通过 mask 过滤 |
| `union` | 两个 top-k 的 id 并集，去重通过 mask 实现 | 上述两次 + 学生在教师 id 上的第二次 gather | `2k`，重复槽位无效 |

`only_tch` 的教师 top-k 会在学生 old-policy pass 之前预计算，从而把学生在教师 id 上的
gather piggyback 到同一次学生前向。`union` 是唯一需要第三次前向的策略。

top-k 模式下，driver **故意把二维 `opd_token_rewards` 置零**，因为完整 KL 梯度由
三维 candidate PPO 项承载，避免把同一个 KL 梯度计算两遍。candidate 项由
[losses.py](../verl/trainer/opd/losses.py:112) 计算：

```text
is_norm=False:
  ratio[t,k] = p_student_current(t,k) / p_student_old(t,k)
is_norm=True:
  ratio[t,k] = p_student_current(t,k|C_t) / p_student_old(t,k|C_t)
L_opd        = sum_{t,k} clipped_PPO(-A[t,k], ratio[t,k])
```

当前策略会在 actor update 中重新对保存的 candidate ids 计算 log-prob，因此
`ppo_epochs>1` 或 mini-batch 变化时 `ratio` 仍然有效。`is_norm=True` 时，current/old
两侧都通过 masked `logsumexp` 条件化，current 侧保留 autograd，所以 score 为
`grad log p_student_current(k|C_t)`，而不再是 full-vocab `grad log p_student_current(k)`。

当 `outside_bucket=True` 时，`is_norm=False` 的 candidate 集合会再增加一个聚合项：

```text
p_student_out = 1 - sum_{k in C_t} p_student(k)
p_teacher_out = 1 - sum_{k in C_t} p_teacher(k)
A_out         = -p_student_out * log(p_student_out / p_teacher_out)
```

目标是把 `C_t` 中的每个 token 和所有 outside token 组成的分区上的 KL。它保留两侧
candidate mass 的差异，但仍然忽略 outside bucket 内部的 token-level 条件分布差异，
因此是 full-vocab KL 的 coarse approximation，而不是完整 KL。actor 侧从可微的
candidate log-prob 计算 `log(1 - sum_C p_current)`，使用与普通 candidate 相同的 PPO
ratio/clipping；不创建 outside sentinel token，也不需要额外 forward。

### 稀疏 logits kernel

[kernels.py](../verl/trainer/opd/kernels.py:35) 只保存 logits、candidate ids 和每行
`logsumexp`，按 `chunk_size` 重算 softmax；不会创建完整的 `[B,T,V]` log-prob 张量。
只有 candidate gather 输出带梯度，top-k ids/log-probs 都是 detached。

## 函数调用链

### 启动阶段

```text
verl.trainer.main_ppo
  -> run_ppo
  -> RayPPOTrainer / v1 trainer __init__
  -> need_colocated_opd(config)
  -> omega_conf_to_dataclass(config.opd) -> OPDConfig
  -> _validate_opd_config()
  -> init_workers()
  -> ActorRolloutRefWorker(..., opd_config=config.opd)
  -> init_model()
       ├─ actor TrainingWorker
       └─ teacher TrainingWorker (forward_only, same actor resource pool)
```

Hydra 注册入口是
[ppo_trainer.yaml](../verl/trainer/config/ppo_trainer.yaml:23) 的
`teacher@actor_rollout_ref.teacher: dp_teacher` 和
`opd@opd: opd`。教师模型配置在
[dp_teacher.yaml](../verl/trainer/config/teacher/dp_teacher.yaml:16)。

### legacy trainer (`trainer.use_v1=False`)

```text
fit()
  -> rollout generate
  -> reward / extract_reward
  -> _balance_batch
  -> _opd_teacher_prepass()       # 仅 only_tch
  -> _prepare_opd_old_log_prob()
  -> _compute_old_log_prob()
       -> actor_rollout_wg.compute_log_prob()
       -> FSDPEngineWithLMHead.prepare_model_inputs/outputs()
       -> sparse_log_probs()
  -> _compute_opd()
       ├─ actor old pass 中的 student top-k / student-at-teacher-id
       ├─ actor_rollout_wg.compute_teacher_opd()
       ├─ build_opd_candidates()
       └─ opd_advantage()
  -> compute_advantage()
       -> compute_opd_token_kl_advantage()
          或 compute_opd_token_kl_plus_grpo_advantage()
  -> _update_actor()
       -> actor_rollout_wg.update_actor()
       -> TrainingWorker.train_mini_batch()
       -> FSDPEngine.forward_step()
       -> opd_ppo_loss()
       -> backward / optimizer.step
```

主要实现位置：

- worker dispatch：[engine_workers.py](../verl/workers/engine_workers.py:796)
- teacher forward：[engine_workers.py](../verl/workers/engine_workers.py:804)
- packed candidate 对齐：[transformer_impl.py](../verl/workers/engine/fsdp/transformer_impl.py:1221)
- sparse output：[transformer_impl.py](../verl/workers/engine/fsdp/transformer_impl.py:1420)
- driver orchestration：[ray_trainer.py](../verl/trainer/ppo/ray_trainer.py:1321)
- advantage registry：[core_algos.py](../verl/trainer/ppo/core_algos.py:1124)

### v1 trainer (`trainer.use_v1=True`)

v1 使用同一套 `build_opd_candidates` / `opd_advantage` / `opd_ppo_loss`，但 batch 经过
TransferQueue 存取：

```text
train_step()
  -> _opd_teacher_prepass()
  -> _prepare_opd_old_log_prob()
  -> _compute_old_log_prob(opd_meta)
  -> _compute_opd() -> tq.kv_batch_put(opd_*)
  -> _compute_advantage() -> tq.kv_batch_put(advantages/returns)
  -> _update_actor() -> update_actor() -> opd_ppo_loss()
```

v1 明确拒绝异步 trainer mode；legacy 路径也只在 FSDP/FSDP2 上实现。

## 已发现的问题与风险

### P1：默认 legacy OPD smoke 在第一步失败

仓库自带 [run_opd_smoke.sh](../tests/special_e2e/opd/run_opd_smoke.sh:1) 的默认配置使用
`trainer.use_v1=False`。现有日志 `/tmp/opd_smoke/0.log` 显示在
`top_k=0` 的第一轮 teacher pass 失败：

```text
ray.exceptions.RayTaskError(KeyError): KeyError: 'prompts'
ray_trainer.py:1440 -> _compute_opd_forward()
padding.py:146   -> no_padding_2_padding()
```

原因是 `_compute_opd_forward()` 只选取“存在的” `_OPD_BASE_KEYS`，实际 generated batch
没有 `prompts` 字段；但 `no_padding_2_padding()` 无条件执行 `data["prompts"]`。因此
至少有一类真实数据/agent-loop batch 无法进入 actor update，GPU 端到端链路尚未打通。

建议让 `no_padding_2_padding()` 在缺少 `prompts` 时从 `input_ids`、`attention_mask`、
`responses` 推导 prompt/response 长度，或在 `_compute_opd_forward()` 明确构造所需的
prompt tensor；修复后应重新跑 0 / only_stu / only_tch / intersection / union 五个 case。

### P1：`grpo_outcome_weight` 是 dead configuration

字段只出现在 [OPDConfig](../verl/workers/config/opd.py:67)、YAML 和注释中；
`compute_opd_token_kl_plus_grpo_advantage()` 直接计算
`outcome_advantages + opd_token_rewards`，调用方也没有把该权重应用到 outcome term。
所以设置 `opd.grpo_outcome_weight=0`、`0.5` 或 `2` 都不会改变 task-reward 梯度。

建议把权重作为 estimator 的显式输入，或在 trainer 组装 `AlgoConfig`/advantage kwargs
时传入并只缩放 GRPO outcome 项，不要缩放 candidate KL 项。

### P2：tokenizer 兼容性检查不足

[engine_workers.py](../verl/workers/engine_workers.py:645) 只比较 student/teacher
`vocab_size`。相同大小不代表 token id 映射、special token 或 chat template 相同；不兼容
时会静默地对错误 token 做蒸馏。建议启动时比较 tokenizer vocab/hash 和关键 special ids，
或明确要求用户提供同一 tokenizer 并在配置层拒绝无法验证的组合。

### P2：诊断指标名称与数值含义不完全一致

`opd/overlap_ratio` 现在按每个 response 位置计算
`|student_topk ∩ teacher_topk| / top_k`，因此所有 `top_k > 0` 的 strategy 都表示
student/teacher top-k 的交集比例，并且数值范围为 `[0, 1]`。`top_k=0` 是 sampled-token
特殊路径，没有配置 top-k 集合，指标保留单候选路径的 `1.0` 行为。

此外，top-k 的 `opd/kl_mean` 是候选集上的加权 log-ratio，候选集不完整时可能为负，
不能直接当作非负的完整 KL；candidate probability mass 仍应单独结合
`opd/student_mass` 和 `opd/teacher_mass` 观察。

### P3：teacher forward 有额外计算开销

teacher 是 forward-only，但 `prepare_model_outputs()` 在 OPD gather 之外仍会按默认路径
计算普通 sampled-token `log_probs`；调用方随后只取 `opd_*` 字段。这不会改变梯度结果，
但会增加 teacher 的计算/内存和输出搬运成本。可为 teacher pass 增加只产出 sparse candidate
字段的模式。

## 当前可依赖的验证结果

在本工作区执行：

```text
pytest -q tests/trainer/test_opd_advantage_on_cpu.py \
  tests/trainer/test_opd_loss_on_cpu.py \
  tests/workers/test_opd_gather_on_cpu.py \
  tests/workers/test_opd_padding_on_cpu.py
36 passed in 9.68s

pytest -q tests/special_sanity/test_config_docs.py
1 passed

ruff check <OPD 相关 Python 文件>
All checks passed!
```

这些测试覆盖数学 oracle、四种候选集合、空 intersection、sampled-token 退化、稀疏
gather backward、response padding round-trip 和 YAML 文档格式；不覆盖 Ray worker
dispatch、真实模型 forward、vLLM 交互、GPU 显存、tokenizer 映射或 GRPO outcome weight。
现有 GPU smoke 失败，因此在修复 P1 问题前，不建议宣称 OPD 已完成可用。

## 最小运行配置

```bash
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=opd_token_kl \
  algorithm.use_kl_in_reward=False \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.use_fused_kernels=False \
  opd.enable=True \
  opd.teacher_model_path=/path/to/teacher \
  opd.top_k=8 \
  opd.strategy=only_stu \
  opd.reward_weight_mode=student_p \
  opd.is_norm=False \
  actor_rollout_ref.teacher.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.teacher.log_prob_max_token_len_per_gpu=4096
```

该命令还需要正常的 `data.*`、rollout、trainer 和 reward 配置。纯 OPD 也要能让现有
reward pipeline 生成 `rm_scores`；这只是 trainer 的数据契约，并不代表 task reward 会
参与 `opd_token_kl` 的最终 advantage。
