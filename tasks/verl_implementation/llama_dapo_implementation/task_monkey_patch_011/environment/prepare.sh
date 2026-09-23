#!/usr/bin/env bash
# verl_implementation/llama_dapo_implementation/task_monkey_patch_011 —— 训练前准备。
#
# 由 environment/{Dockerfile,assets.yaml} 迁移而来。跑在独立的 prepare 阶段容器里：
# root、已激活 `verl` 环境、**唯一有网的阶段**。
set -euo pipefail

# --- 基座模型 ---------------------------------------------------------------
# examples/grpo_trainer/run_llama3_2_1b_instruct_dapo_math_fsdp.sh:30-38 按顺序找
# /models/… → /workspace/verl/Llama-3.2-1B-instruct → 一个 REDACTED 的占位路径。
# 第一个候选在新方案里没人挂，所以必须正好落在第二个上。注意大小写：目录名是
# `Llama-3.2-1B-instruct`（小写 i），和上游仓名不一致，但脚本里写的就是这个。
#
# allow 列表原样来自 legacy Dockerfile。
ptdbench_fetch_hf \
    --repo LIF1014/ptdbench-Llama-3.2-1B-Instruct \
    --revision 32f9e721f796d3c639ab64d2f58ceeaa2c3213b9 \
    --allow config.json \
    --allow generation_config.json \
    --allow LICENSE.txt \
    --allow README.md \
    --allow USE_POLICY.md \
    --allow special_tokens_map.json \
    --allow tokenizer_config.json \
    --allow tokenizer.json \
    --allow tokenizer.model \
    --allow model.safetensors \
    --expect model.safetensors=1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f \
    --expect-size model.safetensors=2471645608 \
    --dest "$PTDBENCH_WORKSPACE/verl/Llama-3.2-1B-instruct"

# --- 数据集 -----------------------------------------------------------------
# 同一个脚本 :44-62 要 /workspace/DAPO-Math-17k_verl/{train,test}.parquet。
# 和 022 / 023 是**同一份产物**，也就是同一个缓存条目——三个任务的 ground truth 依赖
# 于它们拿到同样的字节，所以配方只有一份（docker/ptdbench_datasets/dapo_math_17k.py），
# 不是每个任务各抄一遍。上游 parquet 是 17917 条题复制了 100 遍，划分规则见那个文件。
#
# 历史产物（已弃用，留档）：legacy Dockerfile 从
#   https://hf-mirror.com/datasets/LIF1014/ptdbench-llama-dapo-implementation-task-monkey-patch-011-dataset
#   revision c3a84bacaadea32cc7a2b9d81162678ac2cb684c 的 artifacts/workspace-assets.tar.gz
#   sha256 e6ff5139b9fd4a4c9cca8d29d1d50faaea2220098e9d97829a737fb50fd314db (5676512)
# 解包到 /opt/ptdbench-assets，agent 启动时由 hydrate-assets.sh 铺到 /workspace。
# 本机无网，那个 tarball 没下到过，内容未经核对。
#
# !! grader 必须重算。(2026-09-13 查证)
# evaluations/solution/verifier/resources/grader_spec.json 里
#   historical_final_baseline: "0.04022346368715084"
# 是 72/1790 的 float64 精确表示，那个 1790 是历史 test split（1792 行）过完长度过滤
# 剩下的数量。新划分不会复刻 16125/1792 那条边界（seed 任务里没有任何地方记录），
# 所以 baseline 和 required_final_score "0.07" 这个绝对门槛都要重跑基线后重填。
# （重跑已由用户显式推后。）
#
# digest 与 023 相同：三个任务共用一份去重后的产物。
ptdbench_dataset \
    --recipe dapo_math_17k \
    --expect test.parquet=52596fd6ade0d49db3a49ef0c9c89f88aeb323c17fa5ee23fb0c774d7fba3255 \
    --expect-size test.parquet=331284 \
    --expect train.parquet=fc4cad54f2d15c622025b5e3764e91d2a79145e75358a703658daa66fe47d321 \
    --expect-size train.parquet=2735462 \
    --dest "$PTDBENCH_WORKSPACE/DAPO-Math-17k_verl"
