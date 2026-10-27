#!/usr/bin/env bash
# verl_implementation/context_parallel_oom —— 训练前要准备的东西。
#
# 由 torch_functional 的 environment/ 原样复用（同一份模型仓、同一份 TACO 归档、
# 同一组 SHA-256）。缺陷不在这些资产里，而在启动脚本的 context parallel：
# ulysses_sequence_parallel_size 默认为 1。跑在独立的 prepare 阶段容器里：root、
# 已激活 `verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12` 环境、**唯一有网的阶段**，$PTDBENCH_WORKSPACE、
# $PTDBENCH_PREPARED、$PTDBENCH_MODELS、$PTDBENCH_DATASETS 都可写。
#
# legacy Dockerfile 的做法：装一份打过补丁的 pyext、从 HF 下模型和数据集归档烤进
# 镜像、再由 hydrate-assets.sh 在 agent 阶段把 /opt/ptdbench-assets/workspace 铺到
# /workspace。下面把"下模型 + 下数据集"按 Dockerfile / readme.md 记录的 revision、
# 字节数、SHA-256 原样补回到 prepare 阶段；pyext 那一节**故意不装**，原因见末尾。
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
# 没有内置配方：上游是本地的 TACO EASY（按 picture_num != 0 丢图文题），
# 8368 train / 184 test，这个筛法和划分只存在于发布好的快照里，没有可复算的配方。
# 所以走"下一个钉死的归档"而不是 ptdbench_dataset。两份 parquet 的字节与
# task_function_call 相同（readme.md "External dataset asset" 表），但归档发布在
# 本任务专用的 dataset 仓里，归档本身的 SHA-256 / 字节数与 task_function_call 不同。
#
# 归档是一个 tar.gz，里面是 workspace/taco_verl/*.parquet（legacy 的
# hydrate-assets.sh 就是把这层 workspace/ 铺到 /workspace 上）。ptdbench_fetch_hf
# 不会解包，所以下到一个临时目录再自己展开；展开完两份 parquet 逐个校验。
#
TACO_STAGING="$PTDBENCH_WORKSPACE/.ptdbench-taco-archive"
rm -rf "$TACO_STAGING"

python3 "$PTDBENCH_ENVIRONMENT/ptdbench_fetch.py" \
    --type dataset \
    --repo LIF1014/ptdbench-verl-implementation-torch-functional-dataset \
    --cache-as ptdbench/verl-asset-1 \
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
# 校验值取自 readme.md "External dataset asset" 表的两行。
(
    cd "$PTDBENCH_WORKSPACE/taco_verl"
    sha256sum -c - <<'TACO_DIGESTS'
e364fa69d973f8e8644b2fa4d7dc6ec405d5b4adff64002c6ad52e7c787c6eaa  test.parquet
7f25abdfa9e0f5baa173d153a32973fbe2cc8736503a35408b8ebb8ae2bc5706  train.parquet
TACO_DIGESTS
    [ "$(stat -c %s test.parquet)" = 61082676 ]
    [ "$(stat -c %s train.parquet)" = 606016648 ]
)