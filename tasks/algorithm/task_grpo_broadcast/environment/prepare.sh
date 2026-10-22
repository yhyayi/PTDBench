#!/usr/bin/env bash
# reward_design/task_grpo_broadcast —— 训练前要准备的东西。
#
# 由 environment/{Dockerfile,hydrate-assets.sh} 迁移而来。跑在独立的 prepare 阶段
# 容器里：root、已激活 `verl` 环境、**唯一有网的阶段**，$PTDBENCH_WORKSPACE、
# $PTDBENCH_PREPARED、$PTDBENCH_MODELS、$PTDBENCH_DATASETS 都可写。
#
# 本任务是 reward_design/task_function_call 的拷贝（readme.md：
# "The copy started from task_function_call"）。注入的缺陷有两处：
# verl/trainer/ppo/core_algos.py 里 compute_grpo_outcome_advantage 把每条样本的
# GRPO advantage 摊在序列轴上再广播到整个 batch；以及 verl/utils/reward_score/taco.py
# 里 _to_bool_results 对可能为空的 ndarray 调 .item(0)。模型与数据集的字节和
# task_function_call / torch_functional 逐字节相同，只是各自发布在本任务用到的 HF 仓里，
# revision 各是各的——所以这份 prepare.sh 的资产校验值与 task_function_call 不共享。
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
# 官方仓的固定 revision。allow 列表里的每个文件（含 model.safetensors）与历史 run
# 用的快照逐字节相同，校验值不变。
ptdbench_fetch_hf \
    --repo Qwen/Qwen2.5-1.5B-Instruct \
    --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
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
# 归档校验值取自 legacy Dockerfile 的 DATASET_ARCHIVE_SHA256 / DATASET_ARCHIVE_SIZE。
TACO_STAGING="$PTDBENCH_WORKSPACE/.ptdbench-taco-archive"
rm -rf "$TACO_STAGING"

ptdbench_fetch_hf \
    --type dataset \
    --repo LIF1014/ptdbench-reward-design-task-grpo-broadcast-dataset \
    --revision bf47407d7372856d5d09b6d16dae3c1570393692 \
    --allow artifacts/workspace-assets.tar.gz \
    --expect artifacts/workspace-assets.tar.gz=338c5c49dcf3d999709842b8a20042349f4b81530a9cbf55b1d637e92b5010ae \
    --expect-size artifacts/workspace-assets.tar.gz=472377173 \
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

# --- 关于 pyext：这里**故意**不装 ---------------------------------------------
# verl/utils/reward_score/taco.py 先试 `from pyext import RuntimeModule`，ImportError
# 就退到自带的 _FallbackRuntimeModule（完整可用，不是半截），stdin 路径每条样例都会
# 走到，所以装不装 pyext 会落到奖励上。
#
# 选择不装，依据是本任务 wandb 冻结清单：
#   workspace/logs/wandb/*/files/requirements.txt 里没有 pyext。
# 本 benchmark 的门槛按历史曲线校（readme.md：val 峰值 ≥ 0.125、
# critic/rewards/mean > 0.14），以 wandb 清单为准，不以 legacy Dockerfile 为准。
# （reward_design/task_function_call 出于同一依据也不装；task_evaluator 装，因为
# 它那份 requirements.txt 里有 pyext==0.7。）
#
# 要改回去的话：把 reward_design/task_evaluator/environment/prepare.sh 末尾那一节整段
# 搬过来即可（装进 $PTDBENCH_PREPARED/pysite 并往 $PTDBENCH_PREPARED/env 写 PYTHONPATH）。
# 不要改 docker/images/verl.Dockerfile——那个镜像是所有 verl 任务共用的。
