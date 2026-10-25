---
name: audit-task
description: 审核 PTDBench 任务包（tasks/ 下）的质量。当用户要求"审核/审一下某个任务的质量"、"看看这个任务质量如何"、判断任务该保留还是移除时使用。产出固定结构的中文报告：白话一句话、ground truth 准确性、区分度（泄漏）、verifier 强度、成本、结论与修复清单。全程只读，不动文件。
---

# audit-task — PTDBench 任务质量审核

把一次任务质量审拆成固定流程：摸底 → ground truth → 区分度（泄漏）→ verifier → 成本 → 结论。核心判断始终是三个问题：**症状和原因对得上吗？因果链清晰吗？是真实生产中自然的 bug 吗？**

## 0. 原则

- **只读审计**。修复等用户说，用户习惯"修掉1-4"式跟进——所以修复清单要编号、可指认。
- 报告用中文，给 benchmark 维护者看。结论必须落在三档之一：**保留 / 条件保留（挂修复项）/ 移除**。
- 每个论断配文件链接（`文件:行号`），证据都是读出来的，不是推断出来的。
- 被泄漏污染的记录运行（agent_trace.json 里的通过记录）**不能当难度证据**——先审泄漏再审难度。

## 1. 摸底：固定清单读任务包

```
python3 -m ptdbench validate tasks/<type>/<id>     # 结构校验先跑
```

| 读什么 | 看什么 |
| --- | --- |
| `metadata.yaml` | title/type/image/env 声明的 GPU 数 |
| `readme.md` | 维护者文档：宣称根因、冻结设置表、判定合同、资产 pin、作者自己的备注（往往暴露已知问题） |
| `workspace/` | agent 可见快照：缺陷代码/脚本、冻结日志、证据文件、vendored 框架 |
| `environment/prepare.sh` | 资产下载与校验；**头注释是否点名缺陷** |
| `evaluations/issue/` | instruction.md（agent 被问什么）、oracle.md（标准答案）、verifier/judge.py + judge_prompt.md |
| `evaluations/solution/` | instruction.md（含"禁止作弊"条款）、oracle/、verifier/tests/ 判定脚本 + resources/grader_spec.json |
| `agent_trace.json`（任务根，若提交） | 记录运行：模型、后端、时长、returncode、docker exec argv |

家族一致性：judge.py 与家族参考任务 diff（通常应完全一致）；judge_prompt.md 是否任务定制。

**Bash 教训**：任务 id 会匹配文件路径里的自己——`grep -i oom` 先命中任务名而不是报错；要用精确错误模式（`torch.OutOfMemoryError`、`RayTaskError`）。跨命令 cd 会残留，一律用绝对路径。

## 2. Ground truth 准确性（最重要的一维）

### a. 症状和原因对得上吗

- 找到**真实**症状：冻结日志里崩溃发生在哪个阶段哪个函数（例：OOM 落在 `actor_rollout_update_actor()` → 指向激活内存，而非 rollout 或 ref）。宣称根因必须能解释这个位置。
- **定量验算**：拿实测数字做数量级估算（例：无 checkpointing 激活 ≈ 层数 × S × (H×~6 + I×~3) × 2B；和日志里 allocated 对得上吗）。oracle 的机制解释是实测可核的还是编的。
- agent 拿不拿得到完整证据链：宣称的缺陷设置在 config dump / 代码 / 日志里 grep 得到吗（`grep -o "ulysses_sequence_parallel_size[^,]*" 日志 | sort | uniq -c`）。
- readme 叙事有没有**讲过头**：宣称的触发条件比日志证据更极端（例：说"回复填满 30720"，日志里已完成步的 max 才 7425——触发其实是"异常长回复出现"，不需要到顶）。

### b. 因果链清晰吗

- 把完整因果链写出来，逐环标注：**有直接日志证据 / 需要反推**。
- 需要反推的缺口是好考题还是证据缺失？好缺口 = 现实诊断中确实需要同样的推理（配置不变而结果突变 → 数据依赖触发）；坏缺口 = agent 无从推理只能猜。
- 根因是**单因子还是合取**（多个条件共同成立才触发）？oracle 有没有承认合取结构？judge 的部分给分规则和它匹配吗？
- issue 和 solution 是不是在测**同一个**东西（solution 允许的掩蔽性修法会不会绕开 issue 认定的根因——这是设计取舍，要点名，不是必错）。

### c. 是真实生产中自然的 bug 吗

- bug 类别是否属于该领域高发事故：**配置遗漏**（默认值常规负载无害、极端负载致命——所以能通过冒烟测试）、数据依赖触发、间歇性故障、版本错配、隐式契约破坏、训练-推理不一致。
- 事故时间线是否和真实事故一致（冒烟通过 → 正式跑暴露 → 随机某步炸）。
- 任务化收敛有没有失真：症状被"提纯"（现实里同类事故可能炸在多个位置，任务收敛到一处）算合理，判据是**指向性变清晰而因果不失真**。

### d. 验证完备性（工程侧）

- 有没有**对照/qualification 运行**（authoring/ 目录、修复后跑通的记录）？"修复有效"是实测还是机制推断？
- 触发是确定性还是随机？随机的话复现方差有没有被表征（修好后仍偶发炸 = false-fail 风险；不修也可能短跑不炸 = false-pass 风险）。
- 唯一记录的缺陷运行是不是真的：真实时间戳、连续的步骤序号、failure context trap、真实的栈和显存数字（手造日志通常在步骤连续性或数字自洽性上露馅）。

## 3. 区分度与泄漏（按渠道逐条查）

PTDBench 已确认的泄漏渠道，逐条过：

1. **任务 id 本身是不是答案**（id 含缺陷关键词，如 `context_parallel_oom`）→ `grep -rn "<id>" workspace/ | wc -l`：冻结日志/wandb 里的任务 id 和宿主路径出现次数。
2. **`PTDBENCH_TASK_ID` 环境变量**被导进 agent 容器（框架级，查 agent_trace.json 的 docker exec argv 里有没有 `-e PTDBENCH_TASK_ID=`）。id=答案时这条致命。
3. **prepare.sh 头注释点名缺陷**（Harbor-local 下 environment/ 只读挂载、agent 可见；docker-native/cybertron 不可见——注明受影响的部署路径）。
4. **数据集/模型仓库名**在共享下载缓存可见（`LIF1014/ptdbench-*` 命名）——泄漏的是家族/姊妹任务名还是本任务缺陷，分级。
5. **agent_trace.json 提交在任务根目录**（仓库卫生 + 内部路径外泄）。
6. instruction.md / 冻结文件里有没有"小提示"式剧透（qwen3vl 的"diagnostic inputs"备注是前车之鉴）。

泄漏定级：**任务名=答案或日志路径含答案 → 修复前该任务不出可信数据（issue、solution 全污染）**；头注释/仓库名 → Harbor-local 部署下污染。

## 4. Verifier 强度

### 确定性判定器（solution 级常见）

- 鲁棒性细节：ANSI strip、symlink 拒绝、二进制/.wandb 跳过、原子写。
- 正则质量：边界锚定够不够（`(?:^|\s)step:(\d+)` 能免疫 `timing_s/step:130`、`global_step:78` 假阳性——grep 复核时先被这些骗的是人不是判定器）。
- **判定漏洞（重点）**：逐条对照"instruction 声称 vs 判定器实际检查"——声称"跑满 100 步"但判定器只要 1 个 step 行；`TOTAL_TRAINING_STEPS` 这类环境变量在启动脚本里可覆盖（`${VAR:-default}`）而判定器不查。然后**自己构造一条作弊路径**：不改缺陷、最小代价拿满分的方法是什么（例：短跑 5 步、有缺陷配置 79 步才炸 → 通过）。
- 替代修法是否也通过（掩蔽性修复 vs 根因修复），readme 是否如实标注 "not graded"。

### LLM judge（issue 级常见）

- judge.py 家族一致性（应逐字节相同，diff 验证）。
- judge_prompt 是通用模板还是任务定制部分给分规则；宽松度和根因合取结构是否匹配（合取根因 + 唯一答案 rubric = 误伤；单因子根因 + 宽松 rubric = 软区分度）。
- judge 把 agent 产出当不可信内容（防 prompt 注入）的条款在不在。

## 5. 成本

- 每级 GPU·小时。**issue 是不是诊断型**（agent 只读代码+日志、judge 走 API）→ 可以 0 卡跑。
- **镜像默认已有**（Cybertron devspace / Harbor），关于镜像/资产下载的 TODO 占位符（revision/sha256）可在"其他"节点名、不挂修复；**但其他类型的 TODO**（缺失配置、未完成的逻辑）仍需挂修复项。
- vendored 框架的仓库重量（`git ls-files workspace/verl | wc -l`）。

## 6. 报告模板（固定）

```markdown
# <task-id> 质量审计

**一句话**：……（白话讲清 bug 是什么、修法是什么）

## ground truth 准确性：……（症状—原因匹配【含定量验算】、因果链【逐环证据/反推】、
  自然性、验证完备性）
## 区分度：……（泄漏逐条列出：渠道、出现次数、受影响的部署路径、难度证据是否有效）
## verifier 强度：……（鲁棒性 + 作弊路径 + 替代修法）
## 成本：……
## 其他：……（仓库卫生等）
## 结论与修复清单：保留 / 条件保留 / 移除；修复项编号排序，
  每项标注 [文件操作] / [判定器代码] / [需 GPU 运行] / [需补充信息]；
  最后一句说明修复可行性：能否一键完成（所有信息齐全），还是需要作者补充外部信息（资产坐标、真实运行数据等）
```

### 判断尺度（用已审任务校准）

- 泄漏：任务名=答案、日志路径含答案 → **条件保留**，堵泄漏是前置项。
- ground truth：无对照运行 + 随机触发 → 条件保留，挂 qualification 项。
- 判定漏洞：存在不改缺陷即满分的路径 → **必须修**（这条单独足以否决现状可信度）。
- 内核好（真实事故形态、症状指向性强、因果链可推理）但工程粗糙 → 条件保留 + 修复清单。
- 内核虚构（症状与宣称根因对不上、证据链断裂到只能猜）→ 移除。
