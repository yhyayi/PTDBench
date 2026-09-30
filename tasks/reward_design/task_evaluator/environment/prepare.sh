#!/usr/bin/env bash
# reward_design/task_evaluator —— 训练前要准备的东西。
#
# 由 environment/{Dockerfile,hydrate-assets.sh} 迁移而来。跑在独立的 prepare 阶段
# 容器里：root、已激活 `verl` 环境、**唯一有网的阶段**，$PTDBENCH_WORKSPACE、
# $PTDBENCH_PREPARED、$PTDBENCH_MODELS、$PTDBENCH_DATASETS 都可写。
#
# 和 reward_design/task_function_call 是同一个 recipe（同一份 taco.py 的两种注入），
# 两边的资产内容逐字节相同，只是各自发布在自己的 HF 仓里。改这里要连那边一起看。
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
# 历史 log :79 记的 vLLM model 同样是这条路径——三处必须一致。
#
# 这个仓是本任务专用的快照（仓里的 PTDBENCH.md 明写 "Do not reuse this repository
# as a shared family checkpoint"），所以不和 task_function_call 共用一个 repo；
# 两边的 model.safetensors 字节相同，但 revision 各是各的。
#
# allow 列表按仓里实际存在的文件写死，.gitattributes 那种没人读的不取。
ptdbench_fetch_hf \
    --repo LIF1014/ptdbench-verl-coding-task-evaluator \
    --revision 39b3e1ddfff50c16220ea43b87193a9dd4e03d8b \
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
# :1550/:1556 记的也是 /workspace/taco_verl/{train,test}.parquet。
#
# 没有内置配方：上游是本地的 TACO EASY（按 picture_num != 0 丢图文题），
# 8368 train / 184 test，这个筛法和划分只存在于发布好的快照里，没有可复算的配方。
# 所以走"下一个钉死的归档"而不是 ptdbench_dataset。
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
    --repo LIF1014/ptdbench-verl-coding-task-evaluator-dataset \
    --revision 1e49d4a87c839b4b1d5b7a104280f189bc026f6e \
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

# --- pyext ------------------------------------------------------------------
# verl/utils/reward_score/taco.py:18-31 先试 `from pyext import RuntimeModule`，
# ImportError 就退到自带的 _FallbackRuntimeModule。两条路都能跑，但历史那次运行
# 是**装了** pyext 的（workspace/logs/wandb/offline-run-20260824_163828-5bjvsmv5/
# files/requirements.txt:4 记着 pyext==0.7），而这个任务的评分门槛是按那次运行的
# 曲线定的，所以这里把它补回去，不靠 fallback。:203/:247 两处
# `_runtime_module().from_string(...)` 每条样例都会走到，所以这不是摆设。
#
# 注意 task_function_call **没有**这一节：它那次运行的冻结清单（282 行）里没有
# pyext，两份排序后 diff 的差异正好只有这一个包。legacy 的那份 Dockerfile 是两边
# 复制的同一份，对不上各自的记录——以 wandb 的冻结清单为准。理由写在
# ../../task_function_call/environment/prepare.sh 末尾。
#
# 为什么不写进 docker/images/verl.Dockerfile：那个镜像是**所有 verl 任务共用的**。
# 加一个包等于改掉
# verl_implementation 那三个任务的运行环境，而它们的 outcome 是按没有 pyext 的
# 那套栈记下来的。
#
# 为什么要打补丁：pyext 0.7 里 inspect.getargspec 用了 11 处，而 Python 3.11 把它
# 删了（原 verl conda 环境是 3.12.13）。legacy Dockerfile 用
# 的就是这条 sed，原样搬过来。注意它也会改到那段 IPython 分支——原 verl conda 环境里
# 没有 IPython，那段被自己的
# try/except ImportError 短路掉；哪天闭包里真出现了 IPython，下面的 import 冒烟会
# 当场失败，而不是等到 agent 阶段。
#
# 为什么装到 $PTDBENCH_PREPARED 而不是直接 pip 进 env：prepare 是独立的一次性容器，
# 装进镜像 site-packages 的东西出了这个容器就没了。$PTDBENCH_PREPARED 是唯一会被后两个
# 阶段只读挂进去的目录。约 40KB。
PYEXT_SITE="$PTDBENCH_PREPARED/pysite"
PYEXT_MARKER="$PYEXT_SITE/pyext-0.7.dist-info"
if [ ! -d "$PYEXT_MARKER" ]; then
    if [ -e "$PYEXT_SITE" ]; then
        echo "[prepare] $PYEXT_SITE is stale or incomplete; reinstalling from scratch"
        rm -rf "$PYEXT_SITE"
    fi

    PYEXT_SRC="$PTDBENCH_WORKSPACE/.ptdbench-pyext-src"
    rm -rf "$PYEXT_SRC"
    mkdir -p "$PYEXT_SRC"

    # 钉 sha256 而不是钉版本号：sdist 的地址可变、字节应当不变。
    PYEXT_SHA=e618a635be4e8c4a84b443cc27186601ad052efe35f01839815f2a2cf3947cdf

    # vendor/ 里带了一份逐字节相同的 sdist（sha 同上），**优先用它**：Cybertron 这类
    # 集群只有白名单出口代理（`whitelist-proxy…:7891`，hf-mirror 放行、pypi.tuna 不放行），
    # 直接下 tsinghua 会在 TLS 握手上被重置（`SSLEOFError`）。vendored 文件让 prepare
    # 在无外网时也能过；留着下面的 URL 兜底给出口开放的环境。两条路都按 $PYEXT_SHA 校验。
    PYEXT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PYEXT_VENDORED="$PYEXT_DIR/vendor/pyext-0.7.tar.gz"
    if [ -f "$PYEXT_VENDORED" ]; then
        echo "[prepare] using vendored pyext sdist ($PYEXT_VENDORED)"
        cp "$PYEXT_VENDORED" "$PYEXT_SRC/pyext-0.7.tar.gz"
        echo "$PYEXT_SHA  $PYEXT_SRC/pyext-0.7.tar.gz" | sha256sum -c -
    else
        ptdbench_fetch_url \
            --url https://pypi.tuna.tsinghua.edu.cn/packages/b0/be/9b6005ac644aaef022527ce49617263379e49dbdbd433d1d3dd66d71f570/pyext-0.7.tar.gz \
            --sha256 "$PYEXT_SHA" \
            --size 7772 \
            --dest "$PYEXT_SRC/pyext-0.7.tar.gz"
    fi

    tar -xzf "$PYEXT_SRC/pyext-0.7.tar.gz" -C "$PYEXT_SRC" --strip-components=1
    sed -i 's/inspect\.getargspec/inspect.getfullargspec/g' "$PYEXT_SRC/pyext.py"

    echo "[prepare] installing patched pyext==0.7 into $PYEXT_SITE"
    # --no-build-isolation：pyext 的 setup.py 起手就 `import pyext`，隔离出来的构建
    # 环境里没有刚打好补丁的那份，会重新拿到未打补丁的语义。
    python3 -m pip install --no-cache-dir --no-deps --no-build-isolation \
        --target "$PYEXT_SITE" "$PYEXT_SRC"
    rm -rf "$PYEXT_SRC"
else
    echo "[prepare] pyext already present in $PYEXT_SITE"
fi

# 免 GPU 的冒烟：补丁没打全的话，这一行当场炸在 prepare 阶段，而不是在 400 步训练
# 的第一次算奖励时。
PYTHONPATH="$PYEXT_SITE" python3 -c 'from pyext import RuntimeModule; RuntimeModule.from_string("ptdbench_smoke", "", "def f():\n    return 1\n")'

# 后两个阶段靠这一行看见它。启动脚本对 PYTHONPATH 是**追加**不是赋值
# （run_qwen2_5_1.5b_instruct_taco_fsdp.sh:142 `export PYTHONPATH="${VERL_ROOT}:${PYTHONPATH:-}"`），
# 所以这条路径会一路传进训练进程和它拉起的 reward worker。
if ! grep -qs "^PYTHONPATH=" "$PTDBENCH_PREPARED/env"; then
    echo "PYTHONPATH=$PYEXT_SITE" >> "$PTDBENCH_PREPARED/env"
fi
