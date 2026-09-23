#!/usr/bin/env bash
# verl_implementation/verl_reward_implementation/task_engine_base_025 —— 训练前准备。
#
# 由 environment/{Dockerfile,assets.yaml} 迁移而来。跑在独立的 prepare 阶段容器里：
# root、已激活 `verl` 环境、**唯一有网的阶段**。
set -euo pipefail

# --- 基座模型 ---------------------------------------------------------------
# examples/grpo_trainer/run_qwen2_5_0.5b_instruct_fsdp.sh:28-36 按顺序找
# /models/Qwen2.5-0.5B-Instruct → /workspace/verl/Qwen2.5-0.5B-Instruct → 占位路径。
# 第一个候选在新方案里没人挂，所以必须正好落在第二个上。
#
# allow 列表原样来自 legacy Dockerfile；和 023 是同一个仓同一个 revision，所以缓存
# 命中，两个任务只下一次。
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
# 同一个脚本 :41-59 要 /workspace/gsm8k_verl/{train,test}.parquet，而 workspace/ 里
# **没有** git-tracked 的这份数据（2026-09-13 核实）。少了它任务起不来。
#
# 上游 openai/gsm8k (config=main) 是 question/answer 两列的原始格式，verl 的
# RLHFDataset 直接喂会在 example["prompt"] 上崩，所以这里同样是一次**构建**而不是
# 一次下载。配方见 docker/ptdbench_datasets/gsm8k.py。
#
# 这个任务**不需要**自定义划分：历史产物就是 train 7473 / test 1319，正好是上游 main
# 自带的官方 split，没有任何人为边界。配方沿用上游 split，并对这两个行数做硬校验，
# 对不上直接失败。
#
# grader 侧因此也不受影响（和 011/022/023 不同，那三个必须重算）：
# historical_final_baseline 是 "0.0"，不携带任何验证集规模信息；
# required_final_score "0.4" 是 gsm8k 官方 test 上的绝对准确率门槛，继续成立。
#
# 历史产物（已弃用，留档）：legacy Dockerfile 从
#   https://hf-mirror.com/datasets/LIF1014/ptdbench-verl-reward-implementation-task-engine-base-025-dataset
#   revision ec9d2e88d7729dd73c81715750e8a34ad9718c49 的 artifacts/workspace-assets.tar.gz
#   sha256 7211ce9e85dae7bb431d951ab4970706e29113d06f9e5e6ec5f24c64a7fcf449 (2490115)
# 解包到 /opt/ptdbench-assets，agent 启动时由 hydrate-assets.sh 铺到 /workspace。
# 本机无网，那个 tarball 没下到过，内容未经核对。
#
# 没有 --expect/--expect-size：产物还没生成过。行数校验够挡住"上游变了"，挡不住
# "缓存里那份被动过"；第一次跑完把 `python3 -m ptdbench_datasets.gsm8k` 打印的
# sha256/size 填回来。
ptdbench_dataset \
    --recipe gsm8k \
    --dest "$PTDBENCH_WORKSPACE/gsm8k_verl"
