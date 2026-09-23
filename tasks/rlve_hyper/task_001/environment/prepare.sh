#!/usr/bin/env bash
# rlve_hyper/task_001 — 训练前要准备的东西。
#
# 由 environment/Dockerfile 的 MODEL_* ARG 和 environment/assets.yaml 迁移而来。
# 跑在独立的 prepare 阶段容器里：root、已激活 `slime` 环境、**唯一有网的阶段**。
set -euo pipefail

# --- 基座模型 ---------------------------------------------------------------
# scripts/training/Qwen2.5-0.5B-Instruct/rlve.sh 的 resolve_ckpt_dir() 按顺序找
# /workspace/<name> → /models/<name> → …，所以 dest 必须正好是这个目录名，
# 改一个字就会掉到后面的候选里，最后拿一个不存在的路径去 --hf-checkpoint。
#
# allow 列表是从 legacy Dockerfile 原样搬过来的：这个仓里同一份权重还存了别的
# 格式，不筛的话下载量翻倍，而多出来的字节没有任何东西会读。
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
    --dest "$PTDBENCH_WORKSPACE/Qwen2.5-0.5B-Instruct"

# --- 数据集：故意不下载 ------------------------------------------------------
# legacy Dockerfile 从一个 tarball 取 workspace/{assets,data}，agent 启动时由
# hydrate-assets.sh 铺到 /workspace 上。那 15 个文件在 git 里都已经是 tracked 的，
# 且逐字节相同（只差行尾：tarball 是 CRLF，git 是 LF；JSON 里那是无意义空白）。
# 也就是说 legacy 那条路其实是把 git 里的 LF 换成了 CRLF。不挂它是净收益。
# 完整的比对记录见被本文件取代的 environment/assets.yaml（git 历史里）。

# --- mcore (torch_dist) 格式的 ref 权重（--ref-load） ------------------------
# 上面那份是 HF 格式，喂给 --hf-checkpoint；--ref-load 要的是 mcore 的
# dist-checkpoint，没有上游仓，只能由上面那份现场转出来。
#
# 目录名必须正好是 <HF 目录名>_torch_dist：rlve.sh:80 用
#   RLVE_REF_LOAD="$(resolve_ckpt_dir "Qwen2.5-0.5B-Instruct_torch_dist" ...)"
# 去找，找不到就退回默认值 /workspace/<name>，然后把这个不存在的路径直接交给
# megatron，报错出现在 slime/utils/arguments.py:971，离原因很远。
#
# 和 slime_retool 那份不同，**这里没有兜底**。retool 的训练脚本自己带
# `if [ ! -d "$DIST_MODEL_DIR" ]` 守卫，prepare 少做一次只是慢；
# rlve.sh 没有这个守卫（见 :79-92，拿到路径就直接进 CKPT_ARGS），所以这一步在
# prepare 阶段做不成，agent 阶段就没有第二次机会。下面的 else 分支因此写成显式
# 警告而不是静默跳过。
#
# 产物不保证逐字节可复现：dist-checkpoint 的分片布局跟 world size 走，
# WORLD_SIZE 钉死成 1 得到 pp=1 的单一布局，换机器换卡数也不会变形状
# （convert_hf_to_torch_dist.py 用 WORLD_SIZE 推 pp size）。0.5B 的权重不到 1GB。
#
# rotary-base 不在这里覆盖：retool 那边要显式写 5000000，是因为它的 RL 脚本在
# MODEL_ARGS 之后又传了一次；rlve.sh 全程没有第二个 --rotary-base，
# scripts/models/qwen2.5-0.5B.sh 里的 1000000 就是训练实际用的值。source 同一份
# MODEL_ARGS 就够了，抄一遍反而会和 recipe 漂移。
#
# slime 的源码树在**任务工作区**里，不在镜像里——这个任务自带一棵。
SLIME_DIR="$PTDBENCH_WORKSPACE"
HF_MODEL_DIR="$PTDBENCH_WORKSPACE/Qwen2.5-0.5B-Instruct"
DIST_MODEL_DIR="$PTDBENCH_WORKSPACE/Qwen2.5-0.5B-Instruct_torch_dist"

# 转换要 CUDA + NCCL（dist.init_process_group + torch.cuda.set_device）。
# prepare 阶段按 metadata 的 NVIDIA_VISIBLE_DEVICES 拿得到卡；harbor 后端把
# prepare 跑成 build-time 的 RUN 层，那里没有 GPU。
if python3 -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    echo "[prepare] converting $HF_MODEL_DIR -> $DIST_MODEL_DIR (WORLD_SIZE=1)"
    # MODEL_ARGS 直接 source 任务自带的那份，不在这里抄一遍。
    # shellcheck source=/dev/null
    source "$SLIME_DIR/scripts/models/qwen2.5-0.5B.sh"
    mkdir -p "$DIST_MODEL_DIR"
    WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
    PYTHONPATH="/root/Megatron-LM:$SLIME_DIR" \
    python3 "$SLIME_DIR/tools/convert_hf_to_torch_dist.py" \
        "${MODEL_ARGS[@]}" \
        --hf-checkpoint "$HF_MODEL_DIR" \
        --save "$DIST_MODEL_DIR"
else
    echo "[prepare] WARNING: no CUDA device visible, skipping the torch_dist conversion." >&2
    echo "[prepare] rlve.sh has no fallback guard for this -- the run will fail at" >&2
    echo "[prepare] slime/utils/arguments.py with a --ref-load path that does not exist." >&2
    echo "[prepare] Re-run prepare on a host where the task's GPUs are visible." >&2
fi
