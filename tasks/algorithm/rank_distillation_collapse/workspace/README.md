# On-policy token-ranking distillation

本工作区包含一个数学推理模型的蒸馏训练实现，以及两次训练的配置、日志和生成样本。
学生模型为 DeepSeek-R1-Distill-Qwen-1.5B，冻结教师为 DeepSeek-R1-Distill-Qwen-7B。
任务要求见评测系统提供的 instruction。

## 方法动机

教师的输出分布同时表达了候选 token 的偏好顺序和置信程度。较小的学生未必能够重现
教师的全部概率值，但教师认为哪些候选更好、哪些更差，可能仍然是有用的监督。
这个方法尝试只迁移候选的相对顺序，避免把教师的概率幅值作为拟合目标。

训练时由当前学生生成回复，教师在学生实际访问的前缀上提供监督，以减少离线蒸馏中
训练前缀与推理前缀的差异。每一轮固定生成的回复和教师候选顺序，进行一次学生参数更新，
随后重新生成下一轮数据。

## 定义和目标

令输入为 $x$，当前学生策略为 $\pi_\theta$，冻结教师为 $p_T$。首先采样

$$y\sim\pi_\theta(\cdot\mid x),\qquad s_t=(x,y_{<t}).$$

在每个前缀 $s_t$ 上，教师从共同 tokenizer 的有效词表中选出 $K$ 个候选，按教师分数
从高到低排列：

$$C_t=(v_{t,1},\ldots,v_{t,K})=\operatorname{TopKOrder}(p_T(\cdot\mid s_t)).$$

记学生原始 logits 为 $z_\theta(v\mid s_t)$，相邻候选的分数差为

$$\Delta_{t,i}=z_\theta(v_{t,i}\mid s_t)-z_\theta(v_{t,i+1}\mid s_t).$$

在这两个候选之间，学生选择教师更偏好候选的条件概率是

$$r_{t,i}=\frac{\pi_\theta(v_{t,i}\mid s_t)}
{\pi_\theta(v_{t,i}\mid s_t)+\pi_\theta(v_{t,i+1}\mid s_t)}
=\sigma(\Delta_{t,i}).$$

训练使用相邻 logistic 排序损失，先在每条回复内平均，再在 batch 中平均：

$$L(\theta)=\frac{1}{|B|}\sum_{(x,y)\in B}
\frac{1}{|y|(K-1)}\sum_{t=1}^{|y|}\sum_{i=1}^{K-1}
-\log\sigma(\Delta_{t,i}).$$

实现中的等价表达为 `softplus(score_next - score_previous)`。相邻约束全部满足时，
通过传递性可以得到整个候选列表的顺序。教师分数只用于选候选和排序；正温度缩放不改变
教师排序，因此不会改变这一目标的候选顺序。学生分数使用原始温度 1，不随采样温度缩放。

## 记录的实验

两组均为同一初始学生、DAPO-Math-17k、全参数 AdamW 更新、学习率 `1e-6`、batch 32、
每题一条回复、$K=8$、seed 42、200 步。训练回复上限为 7168 tokens，验证上限 31744，
每 20 步评测 AIME25/AIME26/AMC23，每题采样 16 次。学生使用自身聊天模板的思考前缀。

| 实验 | 教师输出头 | 相邻教师同分候选 |
|---|---|---|
| A | BF16 计算 | 保留，按 token ID 稳定排序 |
| B | FP32 矩阵乘 | 屏蔽同分候选对，分母仍为 K−1 |

A 在训练前进行了评测，B 没有单独记录 step 0。两组均训练到 step 200。

## 代码和证据

- `opd_train/run_recorded.sh A|B`：训练入口；`opd.sh`：底层配置组装。
- `verl/`：完整框架源码、配置和公开测试；`verl/verl/trainer/opd/`：蒸馏目标实现。
- `logs/run_A.log`、`logs/run_B.log`：脱敏后的完整控制台日志。
- `logs/metrics.csv`：上述日志解析出的全部 step 指标，数值未修改。
- `evidence/validation_samples.jsonl`：原始验证文件中选取的生成片段；不同 step 的问题可能不同。
- `evidence/fixed_prefix_metrics.json`：A 组不同 checkpoint 在相同 3 条正常前缀上的 CPU FP32
  观测。每条取前 128 个回复 token，每隔 8 个位置采样，共 48 个位置；不是独立 benchmark。

数据质量说明：历史日志的 `opd/topk_overlap` 与 `opd/top1_agreement` 有已知仪表记录问题，
恒为 1，不能作为模型一致性的证据。`actor/opd_candidate_top1_agreement` 是另一项候选集内
指标。所附源码包含后续的 metadata 保留修正；训练目标未因此改变。其他历史字段和
固定前缀观测保持原值。精确的启动时源码快照未保存，不能将当前单元测试通过等同于
对历史运行每条分布式路径的验证。

## 使用环境

prepare 阶段创建 uv 环境，通过 `AUTORLBENCH_PREPARED/env` 提供解释器。issue 默认模式
只安装 CPU 诊断依赖；检查代码、日志或做小规模张量实验不需要下载模型或使用 GPU。
在 agent 阶段保持离线，不执行 `pip install` / `uv sync`。

完整训练复现需要在 prepare 阶段显式启用 training profile。准备完成后：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash opd_train/run_recorded.sh A
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash opd_train/run_recorded.sh B
```

历史运行分别使用独占的 8 张 A800 80GB GPU，约 11 小时。实际内存、速度和结果受硬件、
数值库与采样影响；issue 评测不要求重跑训练。需要长时间运行时应使用独立会话并保存日志。
