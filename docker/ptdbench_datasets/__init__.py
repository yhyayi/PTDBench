"""把上游资产转成任务要的形状：verl 格式的 parquet，或者拼好的检索索引。

为什么需要这一步：任务的 launch 脚本要 `/workspace/<name>_verl/{train,test}.parquet`，
而 verl 的 `RLHFDataset` 要求每行有

    prompt                    chat messages 列表（不是字符串）
    data_source               精确字符串，决定路由到哪个打分器
    reward_model.ground_truth 真值

（见 workspace/verl/utils/dataset/rl_dataset.py:313 和
  workspace/verl/utils/reward_score/__init__.py:127,142）

上游 `openai/gsm8k` 是 question/answer 两列的原始格式，直接喂进去会在
`example["prompt"]` 上崩。所以"换成标准数据集"= 标准数据集 + 这里这份确定性配方。

slime 那边的两份（`search_r1_nq_hotpotqa` / `wiki18_e5`）是同一类事情的另外两个
形状：前者要把两个 QA 集按 Search-R1 的模板合成 parquet，后者要把上游切成两块的
FAISS 索引 `cat` 回一个能读的文件。共同点都是"下载来的字节 ≠ 任务要的字节"。

## 为什么这个包住在 docker/ 底下，又为什么叫 ptdbench_datasets

它是**镜像的一部分**，和 `ptdbench_fetch.py` 同一类东西：任务的 `environment/prepare.sh`
在容器里调用它，而容器里没有这个仓库。docker/ 是镜像的构建上下文，所以放这里
（`COPY ptdbench_datasets /usr/local/libexec/ptdbench_datasets`）。

名字里的前缀不是装饰：这里的配方要 `from datasets import load_dataset`（HF 那个库），
包一旦叫 `datasets`，它自己就会把那个库遮掉，而报错会发生在 `load_dataset` 上，
离原因很远。

为什么不是每个任务的 `environment/` 各带一份：011 / 022 / 023 三个任务要的是
**同一份 DAPO-Math-17k_verl**，slime_search_r1 / slime_search_r1_nomask 要的是
**同一份 nq_hotpotqa_train 和同一份 wiki-18**。复制几份就有几份可以各自漂移的
配方，而这些任务的 ground truth 依赖于它们拿到同样的字节——这和 §3.2 里
requirements.txt "必须一致而不是取并集"是同一条理由。

## 两种调用方式

容器内（prepare.sh 走这条，带缓存，见 docker/ptdbench_dataset.py）：

    ptdbench_dataset --recipe gsm8k --dest "$PTDBENCH_WORKSPACE/gsm8k_verl"

宿主机上手工跑一份（会额外打印 sha256 + size）：

    python3 -m ptdbench_datasets.gsm8k --out-dir /some/where/gsm8k_verl

运行环境：需要 `datasets` + `pandas` + `pyarrow` + `huggingface_hub`，并且要能联网
访问 HF。镜像里两个 conda 环境（verl / slime）都有；仓库自带的 `/usr/bin/python3`
(3.10.12) 都没有。`wiki18_e5` 还要 ~86GB 的临时磁盘，见那份配方的 docstring。
dapo_math_17k / gsm8k 到目前为止**只做过语法检查和划分规则的离线单测，没有真正
端到端跑过**；search_r1_nq_hotpotqa 的产物离线核对过行数、列和前 3610 行的来源。
第一次跑请人盯着，对一下打印出来的行数和 schema。
"""
