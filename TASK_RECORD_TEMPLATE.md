# 【已构建 / 在建】<一句话任务标题，点明故障现象>

> @<负责人>  ·  `tasks/<task_dir>`

**一句话**：<症状的量级（放大几倍 / 从 A 到 B）+ 对训练的端到端影响。例：续训重载后 critic value head 未清零，warmup 期 value_loss 放大约 77×（7.20 vs 0.094）；同等 rollout 预算下 held-out pass rate buggy 从 0.078 跌到 0.043、fixed 从 0.080 升到 0.169。>

**设定**

<算法> / <模型> / <数据集> / <框架 + 版本>

镜像：`<image@sha256:...>`

训练耗时：<如 1.5h>

资源：<如 1 x A800>

**Bug**

- 症状：<训练者直接观察到的异常指标/现象，以及"正确应是什么">
- 根因：<文件 → 类/函数/行，一句话说清错在哪、为何错>
- 为何复现触发：<什么条件下才暴露；容易被误判成什么>
- 修复：<具体代码改动 + 改动行数；为何这样改最小且正确>

**评测（两级）**

- `issue` 级（复现 + 诊断）：decider = <判定条件>；`agent_answer.md` 需指出 <必须命中的要点>，且**不得** <常见误判>。
- `solution` 级（应用修复）：decider = <离线检查 + 端到端指标阈值，含最小真实工作量要求，如 ≥1 step / ≥1 rollout>。
- 指标 / 阈值：`<metric>`，阈值 **<x>**（buggy <方向> 阈值，fixed <方向> 阈值）。
- 防退化：<verifier 如何防止"靠不报指标/空跑"骗过阈值>。

**关键结果**

| 指标 | buggy | fixed |
|---|---|---|
| <decider 指标> | <v> | <v> |
| <辅助指标> | <v> | <v> |
| <端到端指标 起点> | <v> | <v> |
| <端到端指标 终点> | <v> | <v> |

<统计显著性 / 置信区间 / 与理论推导的吻合等补充说明。>

---

<!--
模板说明（提交前删除本注释块）：
- 标题前缀用【已构建】或【在建】区分任务状态。
- 字段顺序与飞书「任务清单」保持一致，便于直接粘贴归档。
- 「一句话」只写量级与影响，细节留给下面各节。
- 内容应可从任务目录的 metadata.yaml / readme.md 提炼：
    设定 ← algorithm/model/dataset/framework/image/resources
    Bug  ← symptom/root_cause/residue_path/fix_summary
    评测 ← evaluations + metric/threshold + verifier 防退化说明
    关键结果 ← measured.*
- 「在建」任务允许只填到 issue 级、关键结果可留空或写预期。
-->
