#!/usr/bin/env bash
# reward_design/task_function_call —— 训练前要准备的东西。
#
# 由 environment/{Dockerfile,hydrate-assets.sh} 迁移而来。跑在独立的 prepare 阶段
# 容器里：root、已激活 `verl` 环境、**唯一有网的阶段**，$PTDBENCH_WORKSPACE、
# $PTDBENCH_PREPARED、$PTDBENCH_MODELS、$PTDBENCH_DATASETS 都可写。
#
# 和 reward_design/task_evaluator 是同一个 recipe（同一份 taco.py 的两种注入），
# 两边的资产内容逐字节相同，只是各自发布在自己的 HF 仓里。改这里要连那边一起看。
# **但两边的 prepare.sh 不是一份**：末尾 pyext 那一节只有 task_evaluator 有，原因
# 写在下面"关于 pyext"一段。
#
# !! 迁移时补齐的东西（2026-09-23 核实）：legacy Dockerfile 只做了两件事——装一个
# 打过补丁的 pyext，再把 workspace/ 烤进镜像。它**既没有下模型，也没有下数据集**，
# 而 workspace/ 里 git-tracked 的内容里两者都没有。也就是说照着那份 Dockerfile 起
# 出来的容器，跑 examples/grpo_trainer/run_qwen2_5_1.5b_instruct_taco_fsdp.sh 会在
# MODEL_PATH 和 TRAIN_FILE 上直接失败。readme.md 的 "External dataset asset" 一节
# 也自陈 "The current Dockerfile does not download or verify a published external
# dataset revision"。下面这两份资产是按该节记录的 runtime 路径、字节数和 SHA-256
# 对回 HF 上已发布的那两个仓补上的。
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
# as a shared family checkpoint"），所以不和 task_evaluator 共用一个 repo；
# 两边的 model.safetensors 字节相同，但 revision 各是各的。
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
# 所以走"下一个钉死的归档"而不是 ptdbench_dataset。readme.md 记的 184 条里有
# 41 条 call-based、143 条 stdin——这个比例正是本任务的观测证据，换一份数据就对不上了。
#
# 归档是一个 tar.gz，里面是 workspace/taco_verl/*.parquet（legacy 的
# hydrate-assets.sh 就是把这层 workspace/ 铺到 /workspace 上）。ptdbench_fetch_hf
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

# --- 关于 pyext：这里**故意**不装 ---------------------------------------------
# verl/utils/reward_score/taco.py:19-32 先试 `from pyext import RuntimeModule`，
# ImportError 就退到自带的 _FallbackRuntimeModule（`types` 在 :11 已 import，这条
# 退路是完整可用的，不是半截）。:354 的 `_runtime_module().from_string(...)` 在
# stdin 路径上每条样例都会走到，所以装不装 pyext 是会落到奖励上的选择，不是摆设。
#
# 选择不装，依据是这个任务自己的历史运行记录：
#   workspace/logs/wandb/run-20260821_153903-js8e27l6/files/requirements.txt
# 是 282 行，里面**没有** pyext；而 task_evaluator 的那份
#   workspace/logs/wandb/offline-run-20260824_163828-5bjvsmv5/files/requirements.txt
# 是 283 行，:4 记着 pyext==0.7。两份排序后逐行 diff 的差异正好只有这一个包。
# 也就是说这两次 ground-truth 运行用的不是同一套栈，而 legacy 的那份
# environment/Dockerfile 是打包时两边复制的同一份，对本任务而言它装 pyext 这件事
# 与被记录的运行不符。本 benchmark 的门槛是按历史曲线校的
# （readme.md：val 峰值 ≥ 0.125、critic/rewards/mean > 0.14），所以以 wandb 的
# 冻结清单为准，不以那份 Dockerfile 为准。
#
# 两者不是等价实现，这也是必须各自对齐的原因：pyext 的 from_string 是
# `exec` 进一个临时 globals 再把其中不在 pyext 自身 globals 里的名字 copy 进新
# module，于是函数的 __globals__ 和 module.__dict__ 是两个对象；fallback 直接
# `exec(..., module.__dict__)`，两者是同一个。对 TACO 取 `getattr(tmp_sol, 'code')`
# 这一种用法来说结果一样，但不是同一段语义。
#
# 要改回去的话：把 task_evaluator/environment/prepare.sh 末尾那一节整段搬过来即可
# （装进 $PTDBENCH_PREPARED/pysite 并往 $PTDBENCH_PREPARED/env 写 PYTHONPATH）。
# 不要改 docker/images/verl.Dockerfile——那个镜像是所有 verl 任务共用的。
