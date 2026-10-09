#!/usr/bin/env bash
# ppo_critic/proxy_reward_misalignment —— 训练前要准备的东西。
#
# 由 bundle 的 environment/Dockerfile 迁移而来。跑在独立的 prepare 阶段容器里：
# root、已激活 `ppo-critic-vimpo` 环境、**唯一有网的阶段**，$PTDBENCH_WORKSPACE、
# $PTDBENCH_PREPARED、$PTDBENCH_MODELS、$PTDBENCH_DATASETS 都可写。
#
# legacy Dockerfile 只做了三件事：装 cu128/torch2.8/vllm0.11 依赖栈、把 workspace
# 烤进镜像、再 `pip install -e /workspace --no-deps`。它**没有下基座模型**——
# run_buggy.sh:6 的 `: "${MODEL_PATH:?...}"` 把 Qwen3-4B-Base 作为必填外部变量留给了
# 调用方，recipe/vimpo/run_vimpo_critic.sh:34 的默认值是 canonical `Qwen/Qwen3-4B-Base`。
# 这份 prepare.sh 把那一步按 house 约定补回来：下基座模型到
# $PTDBENCH_WORKSPACE/Qwen3-4B-Base（= 容器里的 /workspace/Qwen3-4B-Base，正是
# solution/instruction.md "The effective run must use" 里写死的 MODEL_PATH）。
#
# 不需要装任何任务专属包：reward 函数（ptdbench_case/type3_length_reward_v2_fn.py 与
# 参考 oracle/fixed_reward_fn.py）只 import workspace 自带的
# verl.utils.reward_score.default_compute_score，其依赖（mathruler 等）已在
# docker/images/ppo-critic-vimpo.Dockerfile 的 44 个 runtime 包里。
#
# 也不需要 `pip install -e /workspace`：verl 的 __version__ 读自 verl/version/version
# 文件（不查已安装包元数据），镜像里 PYTHONPATH=/workspace，所以 `import verl` /
# `python3 -m recipe.vimpo.main_vimpo` 直接走 /workspace 下的源码。editable install
# 即便在 prepare 里做了，也不会从 prepare 容器带到 agent/train 容器（site-packages
# 在镜像层，不随 $PTDBENCH_WORKSPACE 持久化），所以这里刻意不做。
set -euo pipefail

# --- 基座模型 ---------------------------------------------------------------
# dest 必须正好是 $PTDBENCH_WORKSPACE/Qwen3-4B-Base，三处对齐：
#   run_buggy.sh:6        要 MODEL_PATH
#   run_vimpo_critic.sh:178  actor_rollout_ref.model.path="${MODEL_PATH}"
#   solution/instruction.md:29  MODEL_PATH=/workspace/Qwen3-4B-Base
#
# repo 默认用 recipe 自己的默认仓 canonical Qwen/Qwen3-4B-Base；允许用环境变量覆盖，
# 因为 canonical Qwen 仓现在是 xet-only，on-cluster 经 whitelist-proxy 取 cas-bridge
# CDN 会被挡（见 harbor 侧 HF-xet 处理），届时可把 QWEN3_4B_BASE_REPO 指向一个
# 已 staged 的镜像仓（rlve_hyper / verl_hyper 就是这么用 LIF1014/ptdbench-* 镜像仓的）。
#
# revision 已钉死为 canonical Qwen/Qwen3-4B-Base 在 2026-08-18（logs/observed_buggy_run.log
# 里原始 run 的日期）的 main commit 906bfd4b4dc7f14ee4320094d8b41684abff8539。这条 sha 由
# 本机（mac，走 hf-mirror.com）下全仓后逐分片核验得到，不是编的；ptdbench_fetch_hf 只接受
# 40 位 commit sha（分支/标签不算钉死），git commit sha 即内容寻址，取到它就等价于取到当时
# 那棵文件树。仍可被环境变量覆盖：若把 QWEN3_4B_BASE_REPO 指向上面说的 staged 镜像仓，镜像仓
# commit 不同，必须同时用 QWEN3_4B_BASE_REVISION 覆盖成镜像仓里的那条。
#
# 不写 --allow：整仓取（Qwen3-4B-Base 仓里只有 config / tokenizer / 分片 safetensors，
# 没有多余大文件），省得猜分片数写漏某一片。
# 三片权重 safetensors 另记 --expect / --expect-size（sha256 + 字节数）：revision 对
# canonical 仓已足够内容寻址，这几行是 big-shard 的双保险——QWEN3_4B_BASE_REPO 被覆盖成
# 镜像仓（此时 revision 已不是 canonical 那条）时仍能校验到权重字节一致（同一模型分片内容
# 相同），也对齐 house 对大分片记 --expect 的做法。数值同样来自本机逐分片核验。
: "${QWEN3_4B_BASE_REPO:=Qwen/Qwen3-4B-Base}"
: "${QWEN3_4B_BASE_REVISION:=906bfd4b4dc7f14ee4320094d8b41684abff8539}"

ptdbench_fetch_hf \
    --repo "$QWEN3_4B_BASE_REPO" \
    --revision "$QWEN3_4B_BASE_REVISION" \
    --dest "$PTDBENCH_WORKSPACE/Qwen3-4B-Base" \
    --expect model-00001-of-00003.safetensors=4c807e2503d68ae373d508689d00a41f4b33f33c2536da97ab81a20caddc1241 \
    --expect model-00002-of-00003.safetensors=f4707585548b2fc75a6b1d732e8465c62040a8699903c32850781beeb9b27826 \
    --expect model-00003-of-00003.safetensors=c7b1aa8fb672de2e00423c99876926022e50b18d4f0d140670788510a27f9965 \
    --expect-size model-00001-of-00003.safetensors=3957900840 \
    --expect-size model-00002-of-00003.safetensors=3987450520 \
    --expect-size model-00003-of-00003.safetensors=99630640

# --- 数据集：不下载 ---------------------------------------------------------
# 受控训练集与留出验证集都很小且已随包携带（git-tracked），不从 HF 取：
#   ptdbench_case/assets/train_proxy_fixed16_x64.parquet   （长度代理奖励的受控训练集）
#   ptdbench_case/assets/aime32.parquet                    (~9 KB)
#   ptdbench_case/assets/math500_32.parquet                (~17 KB)
# run_buggy.sh:29-31 直接按 $ROOT/ptdbench_case/assets/*.parquet 引用它们。
echo "[prepare] datasets ship in ptdbench_case/assets/ (embedded, tiny); nothing to fetch"
