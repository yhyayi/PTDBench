#!/usr/bin/env bash
# 训练前要准备的东西。
#
# 跑在独立的 prepare 阶段容器里：root、已激活 `verl` 环境、**唯一有网的阶段**，
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
# 这个仓是本任务专用的快照（仓里的 PTDBENCH.md 明写 "Do not reuse this repository
# as a shared family checkpoint"）。
#
# allow 列表按仓里实际存在的文件写死，.gitattributes 那种没人读的不取。
ptdbench_fetch_hf \
    --repo LIF1014/ptdbench-verl-coding-tasks-function-call \
    --revision 8d94247a55f318abf15ceb596b30eb2208fdab29 \
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

ptdbench_fetch_hf \
    --type dataset \
    --repo LIF1014/ptdbench-verl-coding-tasks-function-call-dataset \
    --revision 11329cf7797c5450a65d1700e0215588b1ca6901 \
    --allow artifacts/workspace-assets.tar.gz \
    --expect artifacts/workspace-assets.tar.gz=e2b3defb2d289c2ead9fbba6a2248fabcfab2cb3875dd5f010fa3bef6015231d \
    --expect-size artifacts/workspace-assets.tar.gz=471681991 \
    --dest "$TACO_STAGING"

echo "[prepare] unpacking taco_verl -> $PTDBENCH_WORKSPACE/taco_verl"
tar -xzf "$TACO_STAGING/artifacts/workspace-assets.tar.gz" \
    -C "$TACO_STAGING" workspace/taco_verl

# mv 而不是 cp：同一个文件系统上是一次 rename，省掉 667MB 的第二份拷贝。
rm -rf "$PTDBENCH_WORKSPACE/taco_verl"
mv "$TACO_STAGING/workspace/taco_verl" "$PTDBENCH_WORKSPACE/taco_verl"
rm -rf "$TACO_STAGING"

# 展开后的字节自己再校一遍。ptdbench_fetch_hf 校的是归档，这里校的是训练真正读的
# 那两个文件——中间隔着一次 tar，值得各校各的。对不上就硬失败，不重下。
(
    cd "$PTDBENCH_WORKSPACE/taco_verl"
    sha256sum -c - <<'TACO_DIGESTS'
e364fa69d973f8e8644b2fa4d7dc6ec405d5b4adff64002c6ad52e7c787c6eaa  test.parquet
7f25abdfa9e0f5baa173d153a32973fbe2cc8736503a35408b8ebb8ae2bc5706  train.parquet
TACO_DIGESTS
    [ "$(stat -c %s test.parquet)" = 61082676 ]
    [ "$(stat -c %s train.parquet)" = 606016648 ]
)
