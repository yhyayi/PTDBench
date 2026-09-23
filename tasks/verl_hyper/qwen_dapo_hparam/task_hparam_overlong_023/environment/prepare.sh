#!/usr/bin/env bash
# verl_hyper/qwen_dapo_hparam/task_hparam_overlong_023 —— 训练前要准备的东西。
#
# 由 environment/{Dockerfile,assets.yaml} 迁移而来。跑在独立的 prepare 阶段容器里：
# root、已激活 `verl` 环境、**唯一有网的阶段**。
set -euo pipefail

# --- 基座模型 ---------------------------------------------------------------
# examples/grpo_trainer/run_qwen2_5_0_5b_instruct_dapo_math_fsdp.sh:27-33 按顺序找
# /models/Qwen2.5-0.5B-Instruct → /workspace/verl/Qwen2.5-0.5B-Instruct → 一个
# REDACTED 的占位路径。第一个候选在新方案里没人挂，所以必须正好落在第二个上；
# 改一个字就会掉到那个占位路径，然后在一个和真实原因无关的地方报错。
#
# allow 列表原样来自 legacy Dockerfile：同一份权重在这个仓里还存着别的格式，
# 不筛的话下载量翻倍，而多出来的字节没有任何东西会读。
ptdbench_fetch_hf \
    --repo LIF1014/ptdbench-Qwen2.5-0.5B-Instruct \
    --revision e967ddaecb746b9cea032fc1ae80c29a9a49bb73 \
    --allow config.json \
    --allow generation_config.json \
    --allow LICENSE \
    --allow merges.txt \
    --allow README.md \
    --allow tokenizer_config.json \
    --allow tokenizer.json \
    --allow vocab.json \
    --allow model.safetensors \
    --expect model.safetensors=fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe \
    --expect-size model.safetensors=988097824 \
    --dest "$PTDBENCH_WORKSPACE/verl/Qwen2.5-0.5B-Instruct"

# --- 数据集 -----------------------------------------------------------------
# 同一个脚本 :38-56 要 /workspace/DAPO-Math-17k_verl/{train,test}.parquet，
# workspace/ 里没有 git-tracked 的这份数据，少了它任务直接起不来。
#
# 不是一次下载而是一次**构建**：上游 BytedTsinghua-SIA/DAPO-Math-17k 的 parquet
# 是 17917 条不重复的题首尾相接复制了 100 遍（1791700 行），不带 train/test。
# 配方先按 extra_info.index 收成 17917 行，再切成 verl 要的两份。规则写死在
# docker/ptdbench_datasets/dapo_math_17k.py 里
#   sha256("ptdbench-dapo-split-v1:" + extra_info.index) % 10 == 0 进 test
# 与行序、库版本都无关，重跑是同一份。011 / 022 用的是同一份产物（同一个缓存条目）。
#
# 历史产物（已弃用，留档）：LIF1014/ptdbench-task-hparam-overlong-023-dataset
#   revision ab385ea0b4fd9a8c16680917660d442121a6b5eb，切成 base64 分片存的；
#   train.parquet sha256 2897263b99e9b4af7523edeb5ec1ab9cac6e1bba63f0bfa78d4cd9356426ca5b (5885886)
#   test.parquet  sha256 f860d1c3bfb0ce4a937b5e91bb43549a0a7ded51b1725c58c9c10f9f075d2aeb (675271)
# 新配方**不会**复刻这两个 digest，也不会复刻 16125/1792 那条边界——那次划分的 seed
# 任务里任何地方都没记录。
#
# !! 因此 grader 必须重算。(2026-09-13 查证)
# evaluations/solution/verifier/resources/grader_spec.json 里
#   historical_final_baseline: "0.0011173184357541898"
# 是 2/1790 的 float64 精确表示（逐位相同），而历史日志
#   workspace/logs/qwen2_5_0_5b_instruct_dapo_math_fsdp_20260720_120218.log:1503-1504
# 记的 val 是 dataset len: 1792 / filter dataset len: 1790。也就是说 baseline 的含义是
# "1790 题里做对 2 题"，required_final_score: "0.045" 也是这个 1790 题验证集上的绝对
# 准确率门槛。换了 test.parquet 这两个数字就失效，必须重跑基线实验后重填，否则
# pass/fail 衡量的不是 agent 修得对不对。（重跑已由用户显式推后。）
#
# 行数和 schema 由配方自己断言。下面的 digest 挡的是"缓存里那份被动过"：
# 2026-09-23 用 revision 65877096c24f 去重后写出，train 16109 / test 1808。
ptdbench_dataset \
    --recipe dapo_math_17k \
    --expect test.parquet=52596fd6ade0d49db3a49ef0c9c89f88aeb323c17fa5ee23fb0c774d7fba3255 \
    --expect-size test.parquet=331284 \
    --expect train.parquet=fc4cad54f2d15c622025b5e3764e91d2a79145e75358a703658daa66fe47d321 \
    --expect-size train.parquet=2735462 \
    --dest "$PTDBENCH_WORKSPACE/DAPO-Math-17k_verl"
