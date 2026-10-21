#!/usr/bin/env bash
# infrastructure/opd_candidate_gradient —— 训练前要准备的东西。
#
# 跑在独立的 prepare 阶段容器里：root、`verl-opd` 镜像、**唯一有网的阶段**，
# $PTDBENCH_WORKSPACE / $PTDBENCH_PREPARED / $PTDBENCH_MODELS / $PTDBENCH_DATASETS
# 都可写。
#
# 两个 checkpoint 都从 HF 下到 workspace 里。目录名由 opd.sh 的默认值决定，不能改：
#   opd.sh:52  ACTOR_MODEL_PATH   = $WORKSPACE_ROOT/model/DeepSeek-R1-Distill-Qwen-1.5B
#   opd.sh:53  TEACHER_MODEL_PATH = $WORKSPACE_ROOT/model/JustRL-DeepSeek-1.5B
# $WORKSPACE_ROOT 默认是 opd.sh 所在目录（容器里就是 /workspace），所以 dest 必须正好
# 落在 $PTDBENCH_WORKSPACE/model/<name>。
#
# 训练 / 验证数据集随任务一起发布在 workspace/datasets 下（opd.sh:54-55 读的也是
# $WORKSPACE_ROOT/datasets/...），不在这里下载。
#
# revision、sha256、字节数取自原始任务：
#   * revision 来自 environment/Dockerfile 的 snapshot_download 调用；
#   * model.safetensors 的 sha256 来自同一处的显式校验；
#   * model.safetensors 的字节数来自 readme.md "Assets and licenses" 表（该表的
#     "Size"/"SHA-256" 两列就是对着各自那一个 model.safetensors 记的，和 Dockerfile
#     校验的文件一致）。
# 两个 1.5B 模型都只有一个大权重文件 model.safetensors，其余 config / tokenizer 都是
# 几 KB，所以只对 model.safetensors 校 sha256 + size（原始 Dockerfile 也只校它）。
set -euo pipefail

# --- 学生模型（actor） ------------------------------------------------------
ptdbench_fetch_hf \
    --repo deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --revision 73d63fa6f97d4285b49bdbedf6a161ceba1e55ad \
    --allow config.json \
    --allow generation_config.json \
    --allow tokenizer_config.json \
    --allow tokenizer.json \
    --allow model.safetensors \
    --expect model.safetensors=58858233513d76b8703e72eed6ce16807b523328188e13329257fb9594462945 \
    --expect-size model.safetensors=3554214621 \
    --dest "$PTDBENCH_WORKSPACE/model/DeepSeek-R1-Distill-Qwen-1.5B"

# --- 教师模型 ---------------------------------------------------------------
# 必须和学生共用同一套 tokenizer / vocabulary（opd.sh:51 的注释）。
ptdbench_fetch_hf \
    --repo hbx/JustRL-DeepSeek-1.5B \
    --revision 0637e4096c789c67f9eecbe8355e0bdeddede1c2 \
    --allow config.json \
    --allow generation_config.json \
    --allow tokenizer_config.json \
    --allow tokenizer.json \
    --allow model.safetensors \
    --expect model.safetensors=1bedcdf243c4e2e633fa04c05617809cf6a1bbc1a07221609035f6347efecffb \
    --expect-size model.safetensors=3554214752 \
    --dest "$PTDBENCH_WORKSPACE/model/JustRL-DeepSeek-1.5B"
