#!/usr/bin/env bash
# task_train_too_hard —— 训练前要准备的东西。
#
# 由 environment/{Dockerfile,hydrate-assets.sh} 迁移而来。跑在独立的 prepare 阶段
# 容器里：root、已激活 `verl` 环境、**唯一有网的阶段**，$PTDBENCH_WORKSPACE、
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
# repo / revision / allow 列表 / 校验值全部原样来自 legacy Dockerfile 的
# MODEL_REPOSITORY、MODEL_REVISION、allow_patterns、MODEL_WEIGHT_SHA256 与那一行
# `test "$(stat -c %s ...)" = 3087467144`。这个快照仓是 verl + TACO 这一族任务共用
# 的（见 readme.md "External model asset"），不是本任务专有。
ptdbench_fetch_hf \
    --repo LIF1014/ptdbench-verl-implementation-torch-functional \
    --revision 36a4c226ef70aef2c38df173a7cf86a47ad3ad00 \
    --allow config.json \
    --allow generation_config.json \
    --allow LICENSE \
    --allow merges.txt \
    --allow README.md \
    --allow tokenizer_config.json \
    --allow tokenizer.json \
    --allow vocab.json \
    --allow model.safetensors \
    --expect model.safetensors=dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee \
    --expect-size model.safetensors=3087467144 \
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
# 归档校验值取自 legacy Dockerfile 的 DATASET_ARCHIVE_SHA256 / DATASET_ARCHIVE_SIZE。
TACO_STAGING="$PTDBENCH_WORKSPACE/.ptdbench-taco-archive"
rm -rf "$TACO_STAGING"

ptdbench_fetch_hf \
    --type dataset \
    --repo LIF1014/ptdbench-data-task-train-too-hard-dataset \
    --revision 22a1fee0ea2527f514c4115f6c792216a99d7e6f \
    --allow artifacts/workspace-assets.tar.gz \
    --expect artifacts/workspace-assets.tar.gz=eddeece8c2828e21b581d5168dd4604470165cb061a53e68d50b26d49ce6d916 \
    --expect-size artifacts/workspace-assets.tar.gz=167612568 \
    --dest "$TACO_STAGING"

echo "[prepare] unpacking taco_verl -> $PTDBENCH_WORKSPACE/taco_verl"
tar -xzf "$TACO_STAGING/artifacts/workspace-assets.tar.gz" \
    -C "$TACO_STAGING" workspace/taco_verl

# mv 而不是 cp：同一个文件系统上是一次 rename，省掉 220MB 的第二份拷贝。
rm -rf "$PTDBENCH_WORKSPACE/taco_verl"
mv "$TACO_STAGING/workspace/taco_verl" "$PTDBENCH_WORKSPACE/taco_verl"
rm -rf "$TACO_STAGING"

# 展开后的字节自己再校一遍。ptdbench_fetch_hf 校的是归档，这里校的是训练真正读的
# 那两个文件——中间隔着一次 tar，值得各校各的。对不上就硬失败，不重下。
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

# pyext 不装：verl/utils/reward_score/taco.py 先试 `from pyext import RuntimeModule`，
# ImportError 就退到自带的 _FallbackRuntimeModule（完整可用，不是半截）。依据与取舍
# 记在 readme.md "Environment requirement"，不在这里展开。
