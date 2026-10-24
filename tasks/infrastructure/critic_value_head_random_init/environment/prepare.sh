#!/usr/bin/env bash
# slime_critic_value_head_random_init —— 训练前要准备的东西。
#
# 跑在独立的 prepare 阶段容器里：root、`slime-v0.3.2-sglang0.5.15-torch2.11.0-flashattn2.8.3-cu129-py3.12` 镜像、**唯一有网的阶段**，
# $PTDBENCH_WORKSPACE、$PTDBENCH_PREPARED、$PTDBENCH_MODELS、$PTDBENCH_DATASETS 都可写。
#
# 两份资产，都从 HF 下载到 workspace 里。目录名由 run.sh 的默认值决定，**一个字都
# 不能改**（原始工作区用的是 /models/Qwen3-1.7B 和 /datasets/endless-terminals 这两个
# 外部挂载；打包时把 run.sh 的 MODEL_DIR / ET_SRC 默认值改成了 workspace 相对路径，
# 这样 harness 里不需要任何额外挂载）：
#
#   run.sh:MODEL_DIR  ->  $WS/Qwen3-1.7B        （$WS 容器里就是 /workspace）
#   run.sh:ET_SRC     ->  $WS/endless-terminals （再由 `python3 -m et.build_dataset
#                                                 --src "$ET_SRC"` 转写 + 校验）
#
# slime 源码树已经随 workspace 一起打包进来了（workspace/slime/，就是镜像里
# /root/slime 的那份 commit a3f50097），run.sh 用 SLIME_HOME=$WS/slime 把它放到
# PYTHONPATH 最前面、优先于镜像里的安装——所以这里**不**下载 slime 本体，镜像只负责
# 提供 Megatron-LM / torch / ray / sglang 等运行时。
#
# 第一次 run.sh 还会把全部 2492 个 Endless Terminals 任务转写并验证一遍（~40min CPU，
# 写到 ET_DATA_ROOT=/tmp/et_prompts，被后续每次运行复用）——那一步属于 agent 阶段的
# run.sh，不在 prepare 里做；prepare 只把原始任务目录放到位。
#
# ============================== 待补 / TODO ==================================
# 原始任务用的是本地挂载，没有记录任何 HF 坐标，而多 GB 的 sha256 无法在打包机上
# 现算。下面每一处 TODO 都要在任务端到端跑起来之前补上真实值：
#   * Qwen3-1.7B：确认发布仓（公开仓是 Qwen/Qwen3-1.7B）、40 位 commit sha、
#     以及至少一个主权重文件的 sha256 / 字节数。
#   * Endless Terminals：补上承载 2492 个 task_* 目录的 HF dataset 仓、40 位 revision、
#     以及一个代表性文件（或归档）的 sha256 / 字节数。
# 在补全之前，`python3 -m ptdbench validate` 仍然通过（environment/ 与 prepare.sh
# 对校验器是可选的），但 prepare 阶段会因为占位 revision 而失败。
# ============================================================================
set -euo pipefail

# --- 基座模型（run.sh 的 --hf-checkpoint / MODEL_DIR） -----------------------
# dest 必须正好是 $PTDBENCH_WORKSPACE/Qwen3-1.7B：run.sh 的
#   MODEL_DIR="${MODEL_DIR:-${WS}/Qwen3-1.7B}"
# 以及 Stage A / Stage B 两份 megatron role YAML 的 load 字段都读这条路径。
ptdbench_fetch_hf \
    --repo Qwen/Qwen3-1.7B \
    --revision TODO_QWEN3_1_7B_40CHAR_COMMIT_SHA \
    --allow config.json \
    --allow generation_config.json \
    --allow tokenizer_config.json \
    --allow tokenizer.json \
    --allow vocab.json \
    --allow merges.txt \
    --allow model.safetensors \
    --expect model.safetensors=TODO_QWEN3_1_7B_MODEL_SAFETENSORS_SHA256 \
    --expect-size model.safetensors=TODO_QWEN3_1_7B_MODEL_SAFETENSORS_BYTES \
    --dest "$PTDBENCH_WORKSPACE/Qwen3-1.7B"

# --- Endless Terminals 原始任务集（run.sh 的 ET_SRC） ------------------------
# run.sh 的
#   ET_SRC="${ET_SRC:-${WS}/endless-terminals}"
# 指向原始的 task_* 目录树；`python3 -m et.build_dataset --src "$ET_SRC"` 把每个任务的
# Dockerfile 转写成 setup.sh（安装命令 shim 成 no-op），再逐个验证能否被接纳。
# dest 必须正好是 $PTDBENCH_WORKSPACE/endless-terminals。
ptdbench_fetch_hf \
    --type dataset \
    --repo TODO_ENDLESS_TERMINALS_HF_DATASET_REPO \
    --revision TODO_ENDLESS_TERMINALS_40CHAR_REVISION \
    --allow '*' \
    --expect TODO_ENDLESS_TERMINALS_REPRESENTATIVE_FILE=TODO_SHA256 \
    --expect-size TODO_ENDLESS_TERMINALS_REPRESENTATIVE_FILE=TODO_BYTES \
    --dest "$PTDBENCH_WORKSPACE/endless-terminals"
