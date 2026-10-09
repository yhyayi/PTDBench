#!/usr/bin/env bash
# ppo_critic/aime25_single_sample_variance —— 训练前要准备的东西。
#
# 跑在独立的 prepare 阶段容器里：root、已激活环境、**唯一有网的阶段**，
# $PTDBENCH_WORKSPACE / $PTDBENCH_PREPARED / $PTDBENCH_MODELS / $PTDBENCH_DATASETS 都可写。
#
# 这个任务是四个 ppo_critic 里的特例：它改的是**评测协议**（config/eval_config.yaml
# 的 samples_per_question），不跑训练、也不跑真评测来判分，所以 prepare 阶段
# **没有要下载的东西**——这是刻意的，不是漏写。理由逐条：
#
#   * 模型：不需要。两级判分都不加载权重——
#       - solution/verifier 只跑 `run_eval.py --protocol-only`（run_eval.py:93-97：
#         纯打印协议 JSON，不加载 checkpoint、不起 vLLM）+ 对 original_eval/ 证据做
#         `sha256sum -c`；两步都只用 Python 标准库。
#       - issue/verifier 是 LLM judge，比对文本，不碰模型。
#   * 数据集/证据：不需要下。受控重复采样的全部证据已随包携带且被判分校验：
#       workspace/ptdbench_case/original_eval/{n1,n32,notes,SHA256SUMS.txt}。
#   * 可选的「完整 AIME25 重跑」要一个**外部 PPO-Critic checkpoint**，经环境变量
#     CHECKPOINT_PATH 传入（run_eval.py:116-161 要 checkpoint 目录里有 config.json 和
#     model.safetensors.index.json）。那是一个训练产出的大 checkpoint、不是 HF 上的
#     基座模型，没有可 ptdbench_fetch_hf 的仓；它是否提供由操作者决定，本任务判分不依赖它。
#
# 结论：prepare 阶段是一个有据可查的 no-op。保留这份文件是为了和另外三个 ppo_critic
# 任务的 environment/ 布局一致，并把「为什么不下任何东西」写在它该在的地方。
set -euo pipefail

echo "[prepare] aime25_single_sample_variance: nothing to fetch."
echo "[prepare]   grading is stdlib-only (run_eval.py --protocol-only + sha256sum -c);"
echo "[prepare]   preserved evidence ships in ptdbench_case/original_eval/;"
echo "[prepare]   the optional full rerun takes an external checkpoint via CHECKPOINT_PATH."
