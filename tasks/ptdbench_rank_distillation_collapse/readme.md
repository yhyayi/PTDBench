# On-policy 排序蒸馏的生成分布退化

这是一个 **issue-only** 的 PTDBench 任务。来源为两次真实的数学推理蒸馏运行；
训练能够完成且数值有限，但验证准确率和生成质量显著退化。任务关注目标函数设计的
一个可严格验证的缺口，以及 agent 能否区分数学结论、实验观察和未验证的因果解释。

本文件面向维护者，不向 agent 暴露。agent 可见的方法介绍和数学定义位于
`workspace/README.md`；其中只提供方法动机、公式、配置和证据位置，不提供标准诊断。
任务不包含源出版物或其 PDF，也不依赖 agent 检索外部材料。

## 方法与问题

方法希望迁移教师的候选偏好顺序，而不要求较小学生精确拟合教师的置信程度。
每步让学生生成回复，冻结教师在学生实际访问的每个前缀上选出有序 top-K 候选，
学生通过相邻 logistic 约束学习该顺序。对 batch、回复位置和 K−1 个候选对取平均：

$$L=\frac{1}{|B|}\sum_{(x,y)\in B}\frac{1}{|y|(K-1)}
\sum_{t,i}\operatorname{softplus}\left(z_\theta(v_{t,i+1}\mid x,y_{<t})-
z_\theta(v_{t,i}\mid x,y_{<t})\right).$$

**被考查的问题：** 内部排序目标对整个候选集合的共同 logit 平移不敏感。若只将候选
logits 减去一个常数、保持其他词表 logits 不变，则目标不变，而学生给候选集的总概率
可以趋近零。因此，排序 loss 改善与实际生成分布退化可以同时发生。独立 logit 坐标下
集外直接梯度为零，不代表共享参数网络的集外输出在更新中不变。

这是一项目标设计/稳定性诊断，不将公式实现错误作为预设答案。数学不变性能够证明，
但“不惩罚退化方向”不等于“优化器必然选择退化方向”。目前没有单因素消融将全部失败
唯一归因于全参数更新、长 rollout 或某个超参数。oracle 会奖励准确的机制解释和证据，
不会要求 agent 宣称已完成这种因果证明。

## 原始实验和保留的上下文

学生/教师为 DeepSeek-R1-Distill-Qwen-1.5B/7B，DAPO-Math-17k、全参数 AdamW、
lr=1e-6、batch=32、K=8、seed=42、训练回复上限 7168、200 步。A 使用 BF16 教师头，
B 使用 FP32 教师头及同分候选屏蔽。两组各使用 8×A800 80GB，约 11 小时。

| 观测 | A | B |
|---|---:|---:|
| teacher top-8 学生总概率，step 1→200 | 0.93194→0.000667 | 0.93266→0.000378 |
| 输出熵，step 1→200 | 1.02176→11.75231 | 0.99482→11.80022 |
| AIME25，step 100 | 0% | 0% |
| 三个验证集，step 140 | 全为 0% | 全为 0% |

B 的 teacher tie rate 已降到百万分之一级，但仍发生相近的失稳。固定前缀的独立小型
观测进一步发现 A 的排序 loss 从 0.9054 降到 0.6575，同时候选质量从 0.9912 降到
0.7747、熵从 0.2490 升到 2.7804。该观测仅为 3 条正常轨迹、48 个位置的 CPU FP32
检查，不作为完整 benchmark 结果。

`workspace/` 保留完整 verl 框架源码、训练入口、配置、公开测试、两组完整脱敏日志、
401 条 step 指标、生成片段以及固定前缀的原始指标。没有权重、数据集、缓存、隐藏测试、
oracle 或参考修复。原始路径和追踪服务账号已脱敏；训练数值未修改。

历史 top-K/top-1 全词表一致率有独立的仪表错误，已在 agent 文档明确标为不可用，
避免把它作为本任务预设的第二个隐藏问题。当前快照包含后续 metadata 修复，排序损失
保持原样。未保存精确的启动时源码；`environment/source_manifest.json` 记录来源和
封装前文件哈希。这一差异必须保留在复现声明中。

## uv 环境和 prepare

不使用 Docker 或 conda。`metadata.yaml` 使用 `conda: none`，通过 uv 创建 Python
3.12.12 环境。prepare 是唯一允许下载依赖、模型和数据的阶段。

**默认 issue profile** 安装带哈希锁定的 CPU PyTorch、NumPy、PyYAML 及其依赖。
agent 能离线阅读代码/日志，直接导入损失文件运行张量探针。由于 issue 不要求重训，
默认不下载约 18.8 GB 的权重，也不安装 GPU 训练依赖。

```bash
# 这些路径由评测系统指定；preparation 和后续阶段必须在相同路径看到 prepared。
export PTDBENCH_ENVIRONMENT="$TASK/environment"
export PTDBENCH_WORKSPACE="$STAGED_WORKSPACE"
export PTDBENCH_PREPARED="$PREPARED_DIR"
bash "$PTDBENCH_ENVIRONMENT/prepare.sh"
```

环境传递文件为 `$PTDBENCH_PREPARED/env`。重复 prepare 会替换该文件，不重复追加。
uv 管理的 Python 位于 prepared 下，不依赖宿主的个人 Python 安装。

**完整训练 profile** 在 prepare 阶段使用 `PTDBENCH_PREPARE_PROFILE=training`。
它下载固定版本资源、重建 DAPO/AIME26 的原始行内容，安装记录环境的 package closure，
并从固定 source release 编译 flash-attn。需要 Linux x86_64、可兼容 CUDA 13 的驱动和
CUDA 编译工具链，以及足够的磁盘和内存。`requirements-training.txt` 是记录环境的冻结
快照；因为原框架的通用依赖范围与实际环境不同，使用 `--no-deps`，源码通过 PYTHONPATH
加载。**本次没有对该 GPU profile 做干净环境重训验收。**

```bash
PTDBENCH_PREPARE_PROFILE=training bash "$PTDBENCH_ENVIRONMENT/prepare.sh"
# 后续离线阶段，在激活 prepared 的环境后：
cd "$PTDBENCH_WORKSPACE"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash opd_train/run_recorded.sh A
# B 使用相同配置与初始模型，切换教师精度和 tie 处理。
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash opd_train/run_recorded.sh B
```

下载器使用逐文件 SHA-256 与字节数双重检查，不依赖个人登录。损坏缓存或校验失败会
立即失败，不降级继续。缓存只用于下载复用；交付文件复制到 workspace/prepared，
不使用指向宿主缓存的软链接。可在联网 prepare 中填充 `PTDBENCH_MODELS` 和
`PTDBENCH_DATASETS`，转移缓存后使用 `fetch_assets.py --offline` 验证和交付。
uv 的依赖缓存/解释器也须单独保留；只有模型缓存不足以重建 Python 环境。

DAPO 固定来源的提示结尾为 `\\boxed{{}}`，prepare 将其还原成记录运行中的 `\\boxed{}`。
17917 行的内容、顺序和答案均通过 canonical JSON SHA-256 验证。AIME26 从固定 JSONL
构造相同 schema，30 行也验证 canonical hash。Parquet 序列化字节可因库版本变化，
因此重建后验证行语义，源下载文件仍验证精确字节。

固定版本的学生 tokenizer 配置在上游没有追加 `<think>`；记录运行使用了追加
`<think>\n` 的模板。prepare 将上游原件保存为 `tokenizer_config.upstream.json`，
再生成记录版本的 `tokenizer_config.json`，不修改已校验的下载原件，重复 prepare
不会因派生文件与原件哈希不同而失败。token→ID 映射不变。

## Issue 评测协议

公开 instruction 要求 agent 将诊断写入 `$PTDBENCH_RESULTS/agent_answer.md`。禁止
网络访问、依赖安装和读取隐藏任务材料；允许离线 CPU 探针。本任务不含 solution 目录，
不要求参考修复，也不按“某个补丁能否训练达标”评分。

`evaluations/issue/verifier/test.sh` 为唯一正式 verifier 入口。`judge.py` 和
`judge_prompt.md` 原样复用已有 PTDBench 任务，按 verdict/root_cause/evidence
三维评分，权重 0.2/0.6/0.2，通过阈值 0.8。隐藏 `oracle.md` 是共享 judge 实际读取的
标准答案，`gold.yaml` 是同内容的结构化摘要；评分上限同时写入 oracle，避免只修改
未被代码读取的 YAML。不评分修复建议。

隐藏语义案例涵盖完整机制解释、泛泛的“缺正则”、错误的 teacher 精度归因、
只归因于监控，以及过度因果断言。后四类不应通过。它们留给未来正式 LLM 验收，
本次没有假装用关键词规则替代语义 judge。

正式 LLM judge 需要平台提供可信调用通道和 `PTDBENCH_JUDGE_MODEL`、`OPENAI_BASE_URL`、
`OPENAI_API_KEY`。agent 不应获得 judge 凭据。若 verifier 进程所在环境禁止网络，
平台须在可信宿主侧运行 judge 或提供允许的本地服务；本任务不自行开启公网。
缺少配置会写 `passed=false, score=null, status=unavailable`，不伪造通过结果。

## 验证与已知限制

用户明确要求本次**不进行 LLM-judge 验收**。因此不提交虚构的 agent 推理轨迹或
语义评分结果，也不宣称“正确答案通过、错误答案失败”已得到真实 LLM 验证。
`validation/status.json` 记录实际执行情况。

本地验证包括 schema、shell 语法、体积/泄漏检查、CPU uv prepare、数值不变性、
证据一致性、缓存校验和 judge 输入/输出契约。mock judge 只验证输出结构、失败行为及
加权阈值，不能代替真实语义判断。未运行的 LLM 端到端评测、干净 GPU 复现和因果消融
明确标记为未运行。

本次实际结果：schema 严格校验通过；任务契约/数学/证据测试 **14/14 通过**；CPU uv
prepare 的首次、重复和 `UV_OFFLINE=1` 缓存模式均通过；A/B launcher dry-run 通过；
源数据下载、离线缓存校验、17917/30 行语义校验和学生模板重建通过。资源和源码均已做
体积、个人路径、隐藏文件泄漏及缓存文件检查。

现有 PTDBench 框架测试为 **59/60 通过**。唯一失败是
`test_docker_stream_process_writes_live_stage_logs`：本机 `/etc/profile` 无条件执行
磁盘配额查询，向测试的 `sh -lc` 标准输出插入配额信息，导致严格字符串断言失败。
这是已定位的宿主环境干扰；未修改框架测试或宿主 profile 来掩盖它。

本地已有 PTDBench checkout 支持 `--backend local`，但其旧版本 local backend
不会自动执行 prepare，也不提供文件系统/网络安全隔离。开发者应先运行 prepare 并传入
环境；正式评测必须由外层平台保证阶段隔离。uv 只管理 Python，不承担沙箱职责。

验证命令（在 PTDBench 仓库执行，`TASK` 指向本任务）：

```bash
python3 -m ptdbench validate "$TASK"
"$PTDBENCH_PREPARED/venv/bin/python" -m unittest discover \
  -s "$TASK/evaluations/issue/verifier/tests" -v
python3 -m unittest discover -s tests -v
```

## 资源、版本及 license

任务代码保持 Apache-2.0；模型为 MIT。下表列出 training profile 的每个外部文件。
数据集的上游仓库/card license 与题目原始权利分开说明；本任务包不重新分发数据或权重。
机器可读清单为 `environment/assets.json`，CPU Python 依赖的所有允许发行物哈希在
`environment/requirements.txt`。

| 资源 | 固定来源 URL | revision | 字节数 | SHA-256 | license |
|---|---|---|---:|---|---|
| DeepSeek-R1-Distill-Qwen-1.5B/LICENSE | [下载](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/resolve/73d63fa6f97d4285b49bdbedf6a161ceba1e55ad/LICENSE) | `73d63fa6f97d4285b49bdbedf6a161ceba1e55ad` | 1064 | `1c8f573e830ca9b3ebfeb7ace1823146e22b66f99ee223840e7637c9e745e1c7` | MIT |
| DeepSeek-R1-Distill-Qwen-1.5B/config.json | [下载](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/resolve/73d63fa6f97d4285b49bdbedf6a161ceba1e55ad/config.json) | `73d63fa6f97d4285b49bdbedf6a161ceba1e55ad` | 679 | `37bd455e9679d2959536270fed49d25cc7c290a64f6e52abb97c71345a9cee41` | MIT |
| DeepSeek-R1-Distill-Qwen-1.5B/generation_config.json | [下载](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/resolve/73d63fa6f97d4285b49bdbedf6a161ceba1e55ad/generation_config.json) | `73d63fa6f97d4285b49bdbedf6a161ceba1e55ad` | 181 | `72cbec1015da9ed03ad025483005cbf6481403abf947adfd54e504d1a66a2126` | MIT |
| DeepSeek-R1-Distill-Qwen-1.5B/model.safetensors | [下载](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/resolve/73d63fa6f97d4285b49bdbedf6a161ceba1e55ad/model.safetensors) | `73d63fa6f97d4285b49bdbedf6a161ceba1e55ad` | 3554214621 | `58858233513d76b8703e72eed6ce16807b523328188e13329257fb9594462945` | MIT |
| DeepSeek-R1-Distill-Qwen-1.5B/tokenizer.json | [下载](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/resolve/73d63fa6f97d4285b49bdbedf6a161ceba1e55ad/tokenizer.json) | `73d63fa6f97d4285b49bdbedf6a161ceba1e55ad` | 7031660 | `88145e3c3249adc2546ede277e9819d6e405e19072456e4b521cbc724bd60773` | MIT |
| DeepSeek-R1-Distill-Qwen-1.5B/tokenizer_config.json | [下载](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/resolve/73d63fa6f97d4285b49bdbedf6a161ceba1e55ad/tokenizer_config.json) | `73d63fa6f97d4285b49bdbedf6a161ceba1e55ad` | 3061 | `8c726f2ebf532f1d60437b17dd8da4f380adba3608e59e8448b3fb30c9115068` | MIT |
| DeepSeek-R1-Distill-Qwen-7B/LICENSE | [下载](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/resolve/916b56a44061fd5cd7d6a8fb632557ed4f724f60/LICENSE) | `916b56a44061fd5cd7d6a8fb632557ed4f724f60` | 1064 | `1c8f573e830ca9b3ebfeb7ace1823146e22b66f99ee223840e7637c9e745e1c7` | MIT |
| DeepSeek-R1-Distill-Qwen-7B/config.json | [下载](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/resolve/916b56a44061fd5cd7d6a8fb632557ed4f724f60/config.json) | `916b56a44061fd5cd7d6a8fb632557ed4f724f60` | 680 | `8290d3c1e385e2616572eb6926f7e2745f901d1c13da95970326a8a73f6c379c` | MIT |
| DeepSeek-R1-Distill-Qwen-7B/generation_config.json | [下载](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/resolve/916b56a44061fd5cd7d6a8fb632557ed4f724f60/generation_config.json) | `916b56a44061fd5cd7d6a8fb632557ed4f724f60` | 181 | `72cbec1015da9ed03ad025483005cbf6481403abf947adfd54e504d1a66a2126` | MIT |
| DeepSeek-R1-Distill-Qwen-7B/model-00001-of-000002.safetensors | [下载](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/resolve/916b56a44061fd5cd7d6a8fb632557ed4f724f60/model-00001-of-000002.safetensors) | `916b56a44061fd5cd7d6a8fb632557ed4f724f60` | 8606596466 | `8b27be8d43b81d0cd19b3113ef88caf1a61c2fc602de77e3cce97aedcec184b4` | MIT |
| DeepSeek-R1-Distill-Qwen-7B/model-00002-of-000002.safetensors | [下载](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/resolve/916b56a44061fd5cd7d6a8fb632557ed4f724f60/model-00002-of-000002.safetensors) | `916b56a44061fd5cd7d6a8fb632557ed4f724f60` | 6624675384 | `1780a9502f56f592cb4d6262c8c76ca8ad36a86b0257e052c672d87f49707e80` | MIT |
| DeepSeek-R1-Distill-Qwen-7B/model.safetensors.index.json | [下载](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/resolve/916b56a44061fd5cd7d6a8fb632557ed4f724f60/model.safetensors.index.json) | `916b56a44061fd5cd7d6a8fb632557ed4f724f60` | 28090 | `788266ec033788abcf34b2cb8fe156ce300ae05d94d3d54c60f2ecffd8927a1a` | MIT |
| DeepSeek-R1-Distill-Qwen-7B/tokenizer.json | [下载](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/resolve/916b56a44061fd5cd7d6a8fb632557ed4f724f60/tokenizer.json) | `916b56a44061fd5cd7d6a8fb632557ed4f724f60` | 7031660 | `88145e3c3249adc2546ede277e9819d6e405e19072456e4b521cbc724bd60773` | MIT |
| DeepSeek-R1-Distill-Qwen-7B/tokenizer_config.json | [下载](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/resolve/916b56a44061fd5cd7d6a8fb632557ed4f724f60/tokenizer_config.json) | `916b56a44061fd5cd7d6a8fb632557ed4f724f60` | 3071 | `8ac8c85fb242563c2260baec0909debd69d718af6a0b3d90e6cab62b4d341cd5` | MIT |
| datasets/dapo-math-17k-processed.parquet | [下载](https://raw.githubusercontent.com/thunlp/OPD/ac26e38d6f1572eb027597b48a9f4e01f6915ef8/datasets/dapo-math-17k-processed.parquet) | `ac26e38d6f1572eb027597b48a9f4e01f6915ef8` | 3386777 | `500bd8c45eca355b98f9ba6f3213194a72bd42c73c5e9569c6fbbb1b51bd0b39` | Apache-2.0 (upstream BytedTsinghua-SIA/DAPO-Math-17k); processed conversion license not separately declared |
| datasets/test_data/AIME25/test.parquet | [下载](https://raw.githubusercontent.com/thunlp/OPD/ac26e38d6f1572eb027597b48a9f4e01f6915ef8/datasets/test_data/AIME25/test.parquet) | `ac26e38d6f1572eb027597b48a9f4e01f6915ef8` | 13112 | `14ee9a0e8d79cba76c62247bc5ebee1647e023eccf60e31994e279113777dad2` | n/a: hosting snapshot has no separate dataset license; competition problem statements retain original rights |
| datasets/test_data/AMC23/test.parquet | [下载](https://raw.githubusercontent.com/thunlp/OPD/ac26e38d6f1572eb027597b48a9f4e01f6915ef8/datasets/test_data/AMC23/test.parquet) | `ac26e38d6f1572eb027597b48a9f4e01f6915ef8` | 21835 | `0f933978194b73fc091bd667af7d9056d3d745c9d9e3577b5adcad77f8f1ae7c` | n/a: hosting snapshot has no separate dataset license; competition problem statements retain original rights |
| AIME26 source JSONL | [下载](https://modelscope.cn/api/v1/datasets/evalscope/aime26/repo?Revision=12ae93b463353acd4d272dbbc7dbecbcfb902642&FilePath=aime2026.jsonl) | `12ae93b463353acd4d272dbbc7dbecbcfb902642` | 13101 | `52822957957a3f577d1e9706c36a66a8108a3f99b6aff424cfb72dff0094a9ee` | Apache-2.0 (dataset card); underlying competition questions retain original rights |
| flash-attn 2.8.3.post1 source | [下载](https://files.pythonhosted.org/packages/01/7a/92a46e7cd6bbb4d7b2855a457c3b855df54a97af5656d98fc92e58e61065/flash_attn-2.8.3.post1.tar.gz) | `2.8.3.post1` | 8451657 | `55d5103ed846da8b56e0797acf4bde07dee4b1c7e8907fcfc6699c203030c348` | BSD-3-Clause |

