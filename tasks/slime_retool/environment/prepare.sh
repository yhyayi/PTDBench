#!/usr/bin/env bash
# slime_retool —— 训练前要准备的东西。
#
# 跑在独立的 prepare 阶段容器里：root、`slime-v0.3.2` 镜像、**唯一有网的
# 阶段**，而且 $PTDBENCH_WORKSPACE、$PTDBENCH_MODELS 和 $PTDBENCH_DATASETS 都可写。
#
# tasks/slime_retool_length_hack/environment/prepare.sh 是同一份：那个任务跑的是
# 同一个 recipe，只有 reward_func 里多一段 length bonus。健康/不健康这一对的差异
# 必须只有注入的那个缺陷，所以改这里要连那边一起改。
#
# 四份资产，三份下载、一份现场转。目录名由 examples/retool/retool_qwen3_4b_rl.sh
# 的 DATA_DIR/MODEL_DIR 守卫决定（两者默认都是 $PWD，容器里就是 /workspace），
# **一个字都不能改**：改了守卫就命中不到，脚本会安静地拿一个不存在的路径去
# --hf-checkpoint，错误出现在 megatron 里，离原因很远。
set -euo pipefail

# --- 训练集 -----------------------------------------------------------------
# --prompt-data + --input-key prompt + --label-key label 读的是 jsonl。
# 注意这和 verl 那几个任务的 DAPO-Math-17k_verl **不是同一份**：那边是 parquet
# schema。不要为了"统一"把两者合并——服务的是两个框架的两套 loader。
ptdbench_fetch_hf \
    --repo zhuzilin/dapo-math-17k \
    --type dataset \
    --revision 2e65612930298bde4c5d58fd97b3f23a483aaff9 \
    --allow dapo-math-17k.jsonl \
    --expect dapo-math-17k.jsonl=cc9c39c2aa19177abe9464741e121cf4cac90fd25484ef3cdf86535101e3a5b6 \
    --expect-size dapo-math-17k.jsonl=10490834 \
    --dest "$PTDBENCH_WORKSPACE/zhuzilin_dapo-math-17k"

# --- 验证集 -----------------------------------------------------------------
# EVAL_ARGS 里的 `--eval-prompt-data aime "$EVAL_DATA"`，也就是主指标 eval/aime
# 的来源。30 道题，判分脚本认的就是这条曲线。
ptdbench_fetch_hf \
    --repo zhuzilin/aime-2024 \
    --type dataset \
    --revision 1c625e328db94ec7ef7ff169016b097c468d60b9 \
    --allow aime-2024.jsonl \
    --expect aime-2024.jsonl=5d4685dd504391b0ec00a9b73c379a80ba4da804829a113b1d4fe630308d47cc \
    --expect-size aime-2024.jsonl=13029 \
    --dest "$PTDBENCH_WORKSPACE/zhuzilin_aime-2024"

# --- SFT 过的 HF checkpoint（--hf-checkpoint） -------------------------------
# allow 列表挡掉 .gitattributes，其余都是 tokenizer / config，模型自己要读。
# 只对两个分片记 sha256：revision 已经把小文件钉死了，逐个记只是让任务包难写。
ptdbench_fetch_hf \
    --repo font-info/qwen3-4b-sft-SGLang-RL \
    --revision 6de9b0b89be3deddd78e9245670164ed9145ded4 \
    --allow LICENSE \
    --allow added_tokens.json \
    --allow chat_template.jinja \
    --allow config.json \
    --allow generation_config.json \
    --allow merges.txt \
    --allow model-00001-of-00002.safetensors \
    --allow model-00002-of-00002.safetensors \
    --allow model.safetensors.index.json \
    --allow special_tokens_map.json \
    --allow tokenizer.json \
    --allow tokenizer_config.json \
    --allow vocab.json \
    --expect model-00001-of-00002.safetensors=9164ce24732079f2515a3c8c29d6537abf8b56d2a64915f1572b6fa6ab4719ee \
    --expect-size model-00001-of-00002.safetensors=4862372272 \
    --expect model-00002-of-00002.safetensors=35dac1e2a58d6f1cd9d193ce11070d29536d6822eaf0f44dba9eb39a3c0fe405 \
    --expect-size model-00002-of-00002.safetensors=3960522232 \
    --dest "$PTDBENCH_WORKSPACE/font-info_qwen3-4b-sft"

# --- mcore (torch_dist) 格式的 ref 权重（--ref-load） ------------------------
# 没有上游仓，只能由上面那份现场转出来。原 recipe 在训练脚本里顺手转
# （retool_qwen3_4b_rl.sh:68-76），这里提前做掉，把 GPU 小时留给训练本身。
#
# 两个坑：
#
#   * 产物**不保证逐字节可复现**——dist-checkpoint 的分片布局跟 world size 走。
#     WORLD_SIZE 钉死成 1 就得到 pp=1 的单一布局，换机器换卡数也不会变形状
#     （convert_hf_to_torch_dist.py:62-79 用 WORLD_SIZE 推 pp size）。
#     4B 的 bf16 权重约 8GB，单卡放得下。
#   * --rotary-base 必须是 5000000，不是 scripts/models/qwen3-4B.sh 里的默认
#     1000000。RL 脚本在 MODEL_ARGS **之后**又传了一次（:74 和 :84），后者生效；
#     转换和训练用同一个值，ref 权重的位置编码才和 actor 对得上。显式写死，
#     不从环境继承，免得哪天被 MODEL_ARGS_ROTARY_BASE 悄悄改掉。
#
# slime 的源码树在**任务工作区**里，不在镜像里——这个任务自带一棵。
SLIME_DIR="$PTDBENCH_WORKSPACE"
HF_MODEL_DIR="$PTDBENCH_WORKSPACE/font-info_qwen3-4b-sft"
DIST_MODEL_DIR="$PTDBENCH_WORKSPACE/font-info_qwen3-4b-sft_torch_dist"

# 转换要 CUDA + NCCL（dist.init_process_group + torch.cuda.set_device）。
# prepare 阶段按 metadata 的 NVIDIA_VISIBLE_DEVICES 拿得到卡，正常路径下条件是
# 满足的（harbor 后端也是在 `harbor run` 之前用同一套 docker prepare 阶段跑）；
# 但机器上没卡、或者 docker 没配 GPU runtime 时就拿不到。所以这里**跳过而不是报错**：训练脚本自己的
# `if [ ! -d "$DIST_MODEL_DIR" ]` 守卫会在 agent 阶段补做，那时 /workspace 仍然
# 可写。少做一次是慢，做不成才是错。
if python3 -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    echo "[prepare] converting $HF_MODEL_DIR -> $DIST_MODEL_DIR (WORLD_SIZE=1, rotary-base=5000000)"
    # MODEL_ARGS 直接 source 任务自带的那份，不在这里抄一遍——抄了就会和 recipe 漂移。
    # shellcheck source=/dev/null
    source "$SLIME_DIR/scripts/models/qwen3-4B.sh"
    mkdir -p "$DIST_MODEL_DIR"
    WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
    PYTHONPATH="/root/Megatron-LM:$SLIME_DIR" \
    python3 "$SLIME_DIR/tools/convert_hf_to_torch_dist.py" \
        "${MODEL_ARGS[@]}" \
        --hf-checkpoint "$HF_MODEL_DIR" \
        --rotary-base 5000000 \
        --save "$DIST_MODEL_DIR"
else
    echo "[prepare] no CUDA device visible; leaving the torch_dist conversion to the"
    echo "[prepare] training script's own guard (examples/retool/retool_qwen3_4b_rl.sh)."
fi
