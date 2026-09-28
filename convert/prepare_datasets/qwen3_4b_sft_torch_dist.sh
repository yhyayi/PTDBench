#!/usr/bin/env bash
# font-info/qwen3-4b-sft-SGLang-RL  ->  font-info_qwen3-4b-sft_torch_dist/
#
# slime_retool 的 RL 脚本要一份 mcore (torch_dist) 格式的 ref 权重（--ref-load）。
#
# **这个脚本现在是备用件。** 正式路径是任务自己的
# tasks/slime_retool/environment/prepare.sh：它在 prepare 阶段现场转，往
# $PTDBENCH_WORKSPACE 底下写（那个阶段可写），训练脚本里那段
# `if [ ! -d "$DIST_MODEL_DIR" ]` 守卫自然跳过。留着这份是为了能脱离 harness
# 单独转一次——调试转换本身、或者给一台机器预热产物的时候。
#
# ## 和 prepare_datasets 下那两个脚本的**关键区别**
#
# gsm8k.py / dapo_math_17k.py 是纯 CPU、纯 python 的确定性变换，跑两遍出同样的字节。
# **这个不是。** tools/convert_hf_to_torch_dist.py 要 CUDA + NCCL + megatron
# （它 `dist.init_process_group(backend="nccl")` 并且 `torch.cuda.set_device`），
# 所以：
#
#   1. 必须在**有 GPU 的机器**上、在 slime 环境镜像里跑，不能在 CI 的 CPU 机器上跑；
#   2. 产物**不保证逐字节可复现**——dist-checkpoint 的分片布局取决于 world size，
#      元数据里还可能带非确定性内容。记下来的 sha256 的含义是"就是这一份构建
#      产物"，不是"从上游能重算出来的确定值"。这一点和两个 parquet 脚本不同，
#      别混为一谈。
#
# 为了把 (2) 的不确定性压到最小，这里把 WORLD_SIZE 钉死成 1：转换脚本会用
# WORLD_SIZE 推 pipeline_model_parallel_size（convert_hf_to_torch_dist.py:62-79），
# 钉死 1 就得到 pp=1 的单一布局，换机器换卡数也不会变形状。4B 的 bf16 权重约 8GB，
# 单卡放得下。
#
# ## rotary-base 必须是 5000000
#
# scripts/models/qwen3-4B.sh 里默认是 1000000，但 RL 脚本在 MODEL_ARGS **之后**
# 又传了一次 `--rotary-base 5000000`（retool_qwen3_4b_rl.sh:74 和 84），后者生效。
# 转换和训练必须用同一个值，否则 ref 权重的位置编码和 actor 对不上。这里显式写死，
# 不从环境继承，免得哪天被 MODEL_ARGS_ROTARY_BASE 悄悄改掉。
#
# ## 用法（在有 GPU 的宿主机上）
#
# 本脚本不在镜像里，bind-mount 进去跑：
#
#   docker run --rm --gpus '"device=0"' \
#     -e PTDBENCH_STAGE=agent \
#     -v "$PWD/convert:/convert:ro" \
#     -v "$PWD/tasks/slime_retool/workspace:/workspace:ro" \
#     -v /path/to/models:/models \
#     ptdbench/slime:v0.3.2-sglang0.5.15-cu130 \
#     sh -lc 'HF_MODEL_DIR=/models/font-info_qwen3-4b-sft \
#             SAVE_DIR=/models/font-info_qwen3-4b-sft_torch_dist \
#             SLIME_DIR=/workspace \
#             bash /convert/prepare_datasets/qwen3_4b_sft_torch_dist.sh'
#
# **SLIME_DIR 必须给。** slime 的源码树不在镜像里——它在任务的工作区里，每个用
# slime 的任务自带一棵。镜像只提供 slime 的依赖和 /root/Megatron-LM。
#
# HF 侧的权重自己先下好（脚本不会替你下）：
#   hf download font-info/qwen3-4b-sft-SGLang-RL --local-dir /path/to/models/font-info_qwen3-4b-sft

set -euo pipefail

HF_MODEL_DIR=${HF_MODEL_DIR:?需要指向已下好的 font-info/qwen3-4b-sft-SGLang-RL 目录}
SAVE_DIR=${SAVE_DIR:?需要指定 torch_dist 产物的输出目录（必须可写）}
# slime 源码树：不在镜像里，在任务工作区里（tasks/slime_retool/workspace）。
SLIME_DIR=${SLIME_DIR:?需要指向 slime 源码树，通常是 tasks/slime_retool/workspace}
MEGATRON_LM_DIR=${MEGATRON_LM_DIR:-/root/Megatron-LM}

ROTARY_BASE=5000000   # 见上面说明，不要改成从环境读

if [ ! -d "$HF_MODEL_DIR" ] || [ -z "$(ls -A "$HF_MODEL_DIR" 2>/dev/null)" ]; then
    echo "!! HF_MODEL_DIR 不存在或为空：$HF_MODEL_DIR" >&2
    echo "   先 hf download font-info/qwen3-4b-sft-SGLang-RL --local-dir \"$HF_MODEL_DIR\"" >&2
    exit 1
fi
if [ ! -f "$SLIME_DIR/tools/convert_hf_to_torch_dist.py" ]; then
    echo "!! 在 $SLIME_DIR 下找不到 tools/convert_hf_to_torch_dist.py" >&2
    exit 1
fi
if [ -d "$SAVE_DIR" ] && [ -n "$(ls -A "$SAVE_DIR" 2>/dev/null)" ]; then
    echo "!! SAVE_DIR 非空：$SAVE_DIR（不覆盖已有产物，请先挪走）" >&2
    exit 1
fi

# MODEL_ARGS 直接 source 任务自带的那份，不在这里抄一遍——抄了就会和 recipe 漂移。
# shellcheck source=/dev/null
source "$SLIME_DIR/scripts/models/qwen3-4B.sh"

mkdir -p "$SAVE_DIR"

echo "[convert] $HF_MODEL_DIR -> $SAVE_DIR (torch_dist, WORLD_SIZE=1, rotary-base=$ROTARY_BASE)"
WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
PYTHONPATH="$MEGATRON_LM_DIR:$SLIME_DIR" \
python3 "$SLIME_DIR/tools/convert_hf_to_torch_dist.py" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "$HF_MODEL_DIR" \
    --rotary-base "$ROTARY_BASE" \
    --save "$SAVE_DIR"

echo
echo "========================================================================"
echo "产物清单（sha256 + size，留档用）："
echo "========================================================================"
# torch_dist 是一棵目录树，不是单文件。真要记校验值的话，挑几个稳定的关键文件
# 就够了：tracker 文件 + release/ 下的 .metadata。
( cd "$SAVE_DIR" && find . -type f | sort | while read -r f; do
    printf '  %s  %s  %s\n' "$(sha256sum "$f" | cut -d' ' -f1)" "$(stat -c %s "$f")" "${f#./}"
done )
echo
echo "注意：分片布局跟 WORLD_SIZE 走，这些值是"就是这一份产物"，不是能从上游"
echo "重算出来的确定值。正式流程里这棵树由 prepare 阶段现场转，不做校验。"
