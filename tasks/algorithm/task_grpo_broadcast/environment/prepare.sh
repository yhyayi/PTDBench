#!/usr/bin/env bash
# 训练前要准备的东西。
#
# 由 environment/{Dockerfile,hydrate-assets.sh} 迁移而来。跑在独立的 prepare 阶段
# 容器里：root、已激活 `verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12` 环境、**唯一有网的阶段**，$PTDBENCH_WORKSPACE、
# $PTDBENCH_PREPARED、$PTDBENCH_MODELS、$PTDBENCH_DATASETS 都可写。
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
# 所以走"下一个钉死的归档"而不是 ptdbench_dataset。
#
# 归档是一个 tar.gz，里面是 workspace/taco_verl/*.parquet（legacy 的
# hydrate-assets.sh 就是把这层 workspace/ 铺到 /workspace 上）。ptdbench_fetch_hf
# 不会解包，所以下到一个临时目录再自己展开；展开完两份 parquet 逐个校验。
#
# 归档校验值取自 legacy Dockerfile 的 DATASET_ARCHIVE_SHA256 / DATASET_ARCHIVE_SIZE。
TACO_STAGING="$PTDBENCH_WORKSPACE/.ptdbench-taco-archive"
rm -rf "$TACO_STAGING"
# staging 里是解包前的归档，workspace 在 agent 阶段挂着；中途失败也要清掉。
trap 'rm -rf "$TACO_STAGING"' EXIT

# 这个归档的 HF repo 名把本任务的结论写在了名字里，所以它不能以明文出现在任何
# agent 读得到的地方。三处都堵上：
#
#   1. 这个文件：repo 名以 base64 存放，用时才解开。
#   2. 下载缓存的目录名：fetch 默认按 repo 名建 datasets--<owner>--<repo>，而
#      $PTDBENCH_DATASETS 是共享缓存。--cache-as 让它落在一个中性别名下，既不
#      泄漏、又仍然跨 run 复用（早先的做法是把缓存根指到跑完即删的目录，结果
#      每个配置都要重下 450MB，把 hf-mirror 打到 429）。
#   3. fetch 自己的输出：sed 盖住 owner/repo 和 datasets--owner--repo 两种形式
#      ——失败时的 traceback 打的是后者。
#
# 解出来是真名，revision / sha256 / size 校验全部照旧。
TACO_REPO=$(printf %s \
    'TElGMTAxNC9wdGRiZW5jaC1yZXdhcmQtZGVzaWduLXRhc2stZ3Jwby1icm9hZGNhc3QtZGF0YXNldA==' \
    | base64 -d)
TACO_SLUG="${TACO_REPO#*/}"

ptdbench_fetch_hf \
    --type dataset \
    --repo "$TACO_REPO" \
    --cache-as ptdbench/taco-assets-algorithm \
    --revision bf47407d7372856d5d09b6d16dae3c1570393692 \
    --allow artifacts/workspace-assets.tar.gz \
    --expect artifacts/workspace-assets.tar.gz=338c5c49dcf3d999709842b8a20042349f4b81530a9cbf55b1d637e92b5010ae \
    --expect-size artifacts/workspace-assets.tar.gz=472377173 \
    --dest "$TACO_STAGING" \
    2>&1 | sed -e "s|$TACO_SLUG|<dataset>|g" -e "s|${TACO_REPO%%/*}|<owner>|g"
# PIPESTATUS 而不是 $?：$? 是 sed 的退出码，fetch 失败会被它吞掉。
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[prepare] dataset fetch failed" >&2; exit 1; }

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
#
# 要改回去的话：照其他 verl 任务 prepare.sh 末尾装 pyext 的那一节
# （装进 $PTDBENCH_PREPARED/pysite 并往 $PTDBENCH_PREPARED/env 写 PYTHONPATH）。
# 不要改 docker/images/verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12.Dockerfile——那个镜像是所有 verl 任务共用的。
