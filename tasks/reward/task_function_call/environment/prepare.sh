#!/usr/bin/env bash
# 训练前要准备的东西。
#
# 跑在独立的 prepare 阶段容器里：root、已激活 `verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12` 环境、**唯一有网的阶段**，
# $PTDBENCH_WORKSPACE、$PTDBENCH_PREPARED、$PTDBENCH_MODELS、$PTDBENCH_DATASETS
# 都可写。
#
# 这个脚本只负责把模型和数据集放到启动脚本会去读的位置。为什么是这两个仓、为什么
# 不装 pyext、以及迁移时补齐了什么，都记在 readme.md 里；维护者看那份，不要把答案
# 性质的背景写回这里——本文件对 agent 可见（`ptdbench/agents/base.py` 把
# environment/ 只读挂进 /ptdbench/environment）。
set -euo pipefail

# --- 基座模型 ---------------------------------------------------------------
# examples/grpo_trainer/run_qwen2_5_1.5b_instruct_taco_fsdp.sh:20 的
#   MODEL_PATH="${MODEL_PATH:-${VERL_ROOT}/Qwen2.5-1.5B-Instruct}"
# 里 VERL_ROOT 是启动脚本上溯两级，容器里就是 /workspace，所以 dest 必须正好是
# $PTDBENCH_WORKSPACE/Qwen2.5-1.5B-Instruct。solution/instruction.md 也把
# MODEL_PATH=/workspace/Qwen2.5-1.5B-Instruct 写进了 "The effective run must use"，
# 历史 log :83 记的 vLLM model 同样是这条路径——三处必须一致。
#
# 官方仓，整仓下载（fetch 不再按 --allow 取子集，也不再校验 sha256）。
python3 "$PTDBENCH_ENVIRONMENT/ptdbench_fetch.py" \
    --repo Qwen/Qwen2.5-1.5B-Instruct \
    --dest "$PTDBENCH_WORKSPACE/Qwen2.5-1.5B-Instruct"

# --- 数据集 -----------------------------------------------------------------
# 同一个脚本 :21-22 要 $VERL_ROOT/taco_verl/{train,test}.parquet，历史 log
# :1599/:1605 记的也是 /workspace/taco_verl/{train,test}.parquet。
#
# 没有内置配方：上游是本地的 TACO EASY（按 picture_num != 0 丢图文题），
# 8368 train / 184 test，这个筛法和划分只存在于发布好的快照里，没有可复算的配方。
# 所以走"下一个钉死的归档"而不是 ptdbench_dataset。
#
# 归档是一个 tar.gz，里面是 workspace/taco_verl/*.parquet。ptdbench_fetch_hf
# 不会解包，所以下到一个临时目录再自己展开；展开完两份 parquet 逐个校验。
#
# 校验值取自仓里的 artifacts/MANIFEST.json，与 readme.md "External dataset asset"
# 表里的两行一致。
TACO_STAGING="$PTDBENCH_WORKSPACE/.ptdbench-taco-archive"
rm -rf "$TACO_STAGING"

python3 "$PTDBENCH_ENVIRONMENT/ptdbench_fetch.py" \
    --type dataset \
    --repo LIF1014/ptdbench-verl-coding-tasks-function-call-dataset \
    --cache-as ptdbench/verl-asset-2 \
    --dest "$TACO_STAGING"

echo "[prepare] unpacking taco_verl -> $PTDBENCH_WORKSPACE/taco_verl"
tar -xzf "$TACO_STAGING/artifacts/workspace-assets.tar.gz" \
    -C "$TACO_STAGING" workspace/taco_verl

# mv 而不是 cp：同一个文件系统上是一次 rename，省掉 667MB 的第二份拷贝。
rm -rf "$PTDBENCH_WORKSPACE/taco_verl"
mv "$TACO_STAGING/workspace/taco_verl" "$PTDBENCH_WORKSPACE/taco_verl"
rm -rf "$TACO_STAGING"

# 展开后的两份 parquet 自己校一遍（fetch 本身已不再校验），训练真正读它们。
# 对不上就硬失败。
(
    cd "$PTDBENCH_WORKSPACE/taco_verl"
    sha256sum -c - <<'TACO_DIGESTS'
e364fa69d973f8e8644b2fa4d7dc6ec405d5b4adff64002c6ad52e7c787c6eaa  test.parquet
7f25abdfa9e0f5baa173d153a32973fbe2cc8736503a35408b8ebb8ae2bc5706  train.parquet
TACO_DIGESTS
    [ "$(stat -c %s test.parquet)" = 61082676 ]
    [ "$(stat -c %s train.parquet)" = 606016648 ]
)
