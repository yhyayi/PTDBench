#!/usr/bin/env bash
# 训练前要准备的东西。
#
# 由 environment/{Dockerfile,hydrate-assets.sh} 迁移而来。跑在独立的 prepare 阶段
# 容器里：root、已激活 `verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12` 环境、**唯一有网的阶段**，$PTDBENCH_WORKSPACE、
# $PTDBENCH_PREPARED、$PTDBENCH_MODELS、$PTDBENCH_DATASETS 都可写。
#
# legacy Dockerfile 的做法：从 HF 下模型、把数据集归档烤进镜像，再由
# hydrate-assets.sh 在 agent 阶段把 /opt/ptdbench-assets/workspace 铺到 /workspace。
# 下面把这两步按 Dockerfile / readme.md 记录的 revision、字节数、SHA-256 原样搬到
# prepare 阶段，落点与 legacy 铺完之后完全一致。
#
# 本文件在 agent 阶段只读挂在 /ptdbench/environment，agent 读得到：维护者笔记
# （迁移背景、与兄弟任务的关系、判分门槛）一律写在 readme.md，不写在这里。
set -euo pipefail

# --- 基座模型 ---------------------------------------------------------------
# examples/grpo_trainer/run_qwen2_5_1.5b_instruct_taco_fsdp.sh:20 的
#   MODEL_PATH="${MODEL_PATH:-${VERL_ROOT}/Qwen2.5-1.5B-Instruct}"
# 里 VERL_ROOT 是启动脚本上溯两级，容器里就是 /workspace，所以 dest 必须正好是
# $PTDBENCH_WORKSPACE/Qwen2.5-1.5B-Instruct。
#
# 官方仓，整仓下载（fetch 不再按 --allow 取子集，也不再校验 sha256）。
python3 "$PTDBENCH_ENVIRONMENT/ptdbench_fetch.py" \
    --repo Qwen/Qwen2.5-1.5B-Instruct \
    --dest "$PTDBENCH_WORKSPACE/Qwen2.5-1.5B-Instruct"

# --- 数据集 -----------------------------------------------------------------
# 同一个脚本 :21-22 要 $VERL_ROOT/taco_verl/{train,test}.parquet。
#
# 没有内置配方：上游是本地处理好的 TACO 快照，筛法和划分只存在于发布好的归档里，
# 没有可复算的配方。所以走"下一个钉死的归档"而不是 ptdbench_dataset。
#
# 归档是一个 tar.gz，里面正好两个成员：workspace/taco_verl/{train,test}.parquet
# （legacy 的 hydrate-assets.sh 就是把这层 workspace/ 铺到 /workspace 上）。
# ptdbench_fetch_hf 不会解包，所以下到一个临时目录再自己展开。
#
TACO_STAGING="$PTDBENCH_WORKSPACE/.ptdbench-taco-archive"
rm -rf "$TACO_STAGING"
# staging 里是解包前的归档，workspace 在 agent 阶段挂着；中途失败也要清掉。
trap 'rm -rf "$TACO_STAGING"' EXIT

# 这个归档的 HF repo 名把本任务的结论写在了名字里，两处都要堵住它对 agent 可见：
#   1. 本文件：repo 名 base64 存放，用时才解开（agent 读不到 environment/，但图个
#      防御纵深）。
#   2. 下载缓存目录名：fetch 默认按 repo 名建目录，而缓存在 cybertron 上对 agent
#      只读可见。--cache-as 让它落在一个中性别名下，既不泄漏、又能跨 run 复用。
TACO_REPO=$(printf %s \
    'TElGMTAxNC9wdGRiZW5jaC1kYXRhLXRhc2stdHJhaW4tdG9vLWhhcmQtZGF0YXNldA==' \
    | base64 -d)

python3 "$PTDBENCH_ENVIRONMENT/ptdbench_fetch.py" \
    --type dataset \
    --repo "$TACO_REPO" \
    --cache-as ptdbench/taco-asset-2 \
    --dest "$TACO_STAGING"

echo "[prepare] unpacking taco_verl -> $PTDBENCH_WORKSPACE/taco_verl"
tar -xzf "$TACO_STAGING/artifacts/workspace-assets.tar.gz" \
    -C "$TACO_STAGING" workspace/taco_verl

# mv 而不是 cp：同一个文件系统上是一次 rename，省掉 220MB 的第二份拷贝。
rm -rf "$PTDBENCH_WORKSPACE/taco_verl"
mv "$TACO_STAGING/workspace/taco_verl" "$PTDBENCH_WORKSPACE/taco_verl"
rm -rf "$TACO_STAGING"

# 展开后的两份 parquet 自己校一遍（fetch 本身已不再校验），训练真正读它们。
# 对不上就硬失败。
# 校验值取自 readme.md "External dataset asset" 表的两行。
(
    cd "$PTDBENCH_WORKSPACE/taco_verl"
    sha256sum -c - <<'TACO_DIGESTS'
0593078d24134d204b305dd0373acfc33daed9a2c73fdad7304a2fde191daea3  test.parquet
8640b3e6c7a5ff3d5b51aa46e74f74147f3c565cd89b73676d5113f8e546d573  train.parquet
TACO_DIGESTS
    [ "$(stat -c %s test.parquet)" = 61082426 ]
    [ "$(stat -c %s train.parquet)" = 168834942 ]
)

# pyext 不装：issue 级不跑判分器。verl/utils/reward_score/taco.py 先试
# `from pyext import RuntimeModule`，ImportError 就退到自带的 _FallbackRuntimeModule。
