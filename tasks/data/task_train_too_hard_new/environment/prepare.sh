#!/usr/bin/env bash
# 训练前要准备的东西。
#
# 由 environment/{Dockerfile,hydrate-assets.sh} 迁移而来。跑在独立的 prepare 阶段
# 容器里：root、已激活 `verl` 环境、**唯一有网的阶段**，$PTDBENCH_WORKSPACE、
# $PTDBENCH_PREPARED、$PTDBENCH_MODELS、$PTDBENCH_DATASETS 都可写。
#
# legacy Dockerfile 的做法：从 HF 下模型、把数据集归档烤进镜像，再由
# hydrate-assets.sh 在 agent 阶段把 /opt/ptdbench-assets/workspace 铺到 /workspace。
# 下面把这两步按 Dockerfile / readme.md 记录的 revision、字节数、SHA-256 原样搬到
# prepare 阶段，落点与 legacy 铺完之后完全一致。
#
# 本文件在 agent 阶段只读挂在 /ptdbench/environment，agent 读得到：维护者笔记
# （迁移背景、与兄弟任务的关系、判分门槛）一律写在 readme.md，不写在这里。
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
# 没有内置配方：上游是本地处理好的 TACO 快照，筛法和划分只存在于发布好的归档里，
# 没有可复算的配方。所以走"下一个钉死的归档"而不是 ptdbench_dataset。
#
# 归档是一个 tar.gz，里面正好两个成员：workspace/taco_verl/{train,test}.parquet
# （legacy 的 hydrate-assets.sh 就是把这层 workspace/ 铺到 /workspace 上）。
# ptdbench_fetch_hf 不会解包，所以下到一个临时目录再自己展开。
#
# 归档校验值取自 legacy Dockerfile 的 DATASET_ARCHIVE_SHA256 / DATASET_ARCHIVE_SIZE。
TACO_STAGING="$PTDBENCH_WORKSPACE/.ptdbench-taco-archive"
# 缓存根必须和 --dest 分开、且不能是 --dest 的子目录：materialize() 在拷贝前会
# 先 rmtree(dest)，dest 套着缓存的话，它删掉的正是自己马上要拷的源
# （FileNotFoundError: .../snapshots）。所以这里是 staging 的兄弟目录。
TACO_CACHE="$PTDBENCH_WORKSPACE/.ptdbench-taco-cache"
rm -rf "$TACO_STAGING" "$TACO_CACHE"
# 两个目录都带 repo 名或归档，而 workspace 在 agent 阶段是挂着的；中途失败时
# 必须照样清掉，否则现场就留着一个名字即答案的目录。trap 覆盖非零退出和中断。
trap 'rm -rf "$TACO_STAGING" "$TACO_CACHE"' EXIT

# 这个归档的 HF repo 名把本任务的结论直接写在了名字里，所以它不能以明文出现在
# 任何 agent 读得到的地方。三条泄漏路径：
#
#   1. 这个文件：agent 阶段 environment/ 只读挂在 /ptdbench/environment，
#      prepare.sh 的正文 agent 读得到。所以 repo 名以 base64 存放，用时才解开
#      ——明文不进源码，解出来是真名，revision / sha256 / size 校验全部照旧。
#   2. 下载缓存的目录名：fetch 先落到 $PTDBENCH_DATASETS 的 HF 缓存布局
#      （hf_repo_dir: datasets--<owner>--<repo>，repo 名变成目录名），再硬链到
#      --dest；而 $PTDBENCH_DATASETS 在 agent 阶段同样只读挂着。把这一条 fetch
#      的缓存根指到上面那个跑完即删的目录，agent 阶段就看不到。
#   3. prepare.log：它在 $_logs_dir 里，而那个目录以 --rw 挂进 agent 容器的
#      /logs（agent 的答案也写在那里）。fetch 会打 "[fetch] downloading <repo>"，
#      失败时 traceback 还会打出缓存路径——注意那是 datasets--A--B 这种双横线
#      形式，和 repo 的 A/B 形式不是同一个串。所以 sed 匹配两者共有的那段
#      （repo 名去掉 owner 的部分），一条规则盖住两种形式。
#
# 代价：这一条不再复用跨 run 的数据集缓存，每次 prepare 重下 160MB。上面的模型
# 那条不受影响，仍走共享缓存。
TACO_REPO=$(printf %s \
    'TElGMTAxNC9wdGRiZW5jaC1kYXRhLXRhc2stdHJhaW4tdG9vLWhhcmQtZGF0YXNldA==' \
    | base64 -d)
TACO_SLUG="${TACO_REPO#*/}"

PTDBENCH_DATASETS="$TACO_CACHE" \
ptdbench_fetch_hf \
    --type dataset \
    --repo "$TACO_REPO" \
    --revision 22a1fee0ea2527f514c4115f6c792216a99d7e6f \
    --allow artifacts/workspace-assets.tar.gz \
    --expect artifacts/workspace-assets.tar.gz=eddeece8c2828e21b581d5168dd4604470165cb061a53e68d50b26d49ce6d916 \
    --expect-size artifacts/workspace-assets.tar.gz=167612568 \
    --dest "$TACO_STAGING" \
    2>&1 | sed -e "s|$TACO_SLUG|<dataset>|g" -e "s|${TACO_REPO%%/*}|<owner>|g"
# PIPESTATUS 而不是 $?：$? 是 sed 的退出码，fetch 失败会被它吞掉。
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[prepare] dataset fetch failed" >&2; exit 1; }

echo "[prepare] unpacking taco_verl -> $PTDBENCH_WORKSPACE/taco_verl"
tar -xzf "$TACO_STAGING/artifacts/workspace-assets.tar.gz" \
    -C "$TACO_STAGING" workspace/taco_verl

# mv 而不是 cp：同一个文件系统上是一次 rename，省掉 220MB 的第二份拷贝。
rm -rf "$PTDBENCH_WORKSPACE/taco_verl"
mv "$TACO_STAGING/workspace/taco_verl" "$PTDBENCH_WORKSPACE/taco_verl"
rm -rf "$TACO_STAGING"

# 展开后的字节自己再校一遍。ptdbench_fetch_hf 校的是归档，这里校的是训练真正读的
# 那两个文件——中间隔着一次 tar，值得各校各的。对不上就硬失败，不重下。
# 校验值取自 readme.md "External dataset asset" 表的两行。
(
    cd "$PTDBENCH_WORKSPACE/taco_verl"
    sha256sum -c - <<'TACO_DIGESTS'
0593078d24134d204b305dd0373acfc33daed9a2c73fdad7304a2fde191daea3  test.parquet
8640b3e6c7a5ff3d5b51aa46e74f74147f3c565cd89b73676d5113f8e546d573  train.parquet
TACO_DIGESTS
    [ "$(stat -c %s test.parquet)" = 61082426 ]
    [ "$(stat -c %s train.parquet)" = 168834942 ]
)

# pyext 不装：issue 级不跑判分器。verl/utils/reward_score/taco.py 先试
# `from pyext import RuntimeModule`，ImportError 就退到自带的 _FallbackRuntimeModule。
