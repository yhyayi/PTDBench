#!/usr/bin/env bash
# 训练前准备。
#
# 这个文件按设计是只读挂给 agent 的（PTDBENCH_ENVIRONMENT_DIR）——它是"环境里装了
# 什么"的说明，不是答案。所以这里不要写任务标识、缺陷位置、评测指标或阈值；
# 维护者笔记写进 ../readme.md。
#
# 跑在独立的 prepare 阶段容器里：root、已激活 `verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12` 环境、**唯一有网的阶段**。
set -euo pipefail

# --- 基座模型 ---------------------------------------------------------------
# examples/grpo_trainer/run_llama3_2_1b_instruct_dapo_math_fsdp.sh:31-39 按顺序找
# /models/… → /workspace/verl/Llama-3.2-1B-instruct → workspace 内的同名目录。
# 第一个候选没人挂，所以必须正好落在第二个上。注意大小写：目录名是
# `Llama-3.2-1B-instruct`（小写 i），和上游仓名不一致，但脚本里写的就是这个。
ptdbench_fetch_hf \
    --repo LIF1014/ptdbench-Llama-3.2-1B-Instruct \
    --revision 32f9e721f796d3c639ab64d2f58ceeaa2c3213b9 \
    --allow config.json \
    --allow generation_config.json \
    --allow LICENSE.txt \
    --allow README.md \
    --allow USE_POLICY.md \
    --allow special_tokens_map.json \
    --allow tokenizer_config.json \
    --allow tokenizer.json \
    --allow tokenizer.model \
    --allow model.safetensors \
    --expect model.safetensors=1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f \
    --expect-size model.safetensors=2471645608 \
    --dest "$PTDBENCH_WORKSPACE/verl/Llama-3.2-1B-instruct"

# --- 数据集 -----------------------------------------------------------------
# 同一个脚本 :42-60 要 /workspace/DAPO-Math-17k_verl/{train,test}.parquet。
# 配方只有一份（docker/ptdbench_datasets/dapo_math_17k.py），由共用这份数据的几个
# 任务复用同一个缓存条目——它们的 ground truth 依赖于拿到同样的字节，所以不是每个
# 任务各抄一遍。上游 parquet 是 17917 条题复制了 100 遍，划分规则见那个文件。
ptdbench_dataset \
    --recipe dapo_math_17k \
    --expect test.parquet=52596fd6ade0d49db3a49ef0c9c89f88aeb323c17fa5ee23fb0c774d7fba3255 \
    --expect-size test.parquet=331284 \
    --expect train.parquet=fc4cad54f2d15c622025b5e3764e91d2a79145e75358a703658daa66fe47d321 \
    --expect-size train.parquet=2735462 \
    --dest "$PTDBENCH_WORKSPACE/DAPO-Math-17k_verl"
