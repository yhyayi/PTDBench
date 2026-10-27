#!/usr/bin/env bash
# slime_search_r1_nomask —— 训练前要准备的东西。
#
# 和 tasks/slime_search_r1/environment/prepare.sh 是同一份：两个任务跑的是同一个
# recipe（examples/search-r1/run_qwen2.5_3B.sh），要的资产也完全一样。改这里要连
# 那边一起改——容器环境或资产上的任何差别，都会让两边的运行不可比。
#
# 跑在独立的 prepare 阶段容器里：root、`slime-v0.3.2-sglang0.5.15-torch2.11.0-flashattn2.8.3-cu129-py3.12` 镜像、**唯一有网的
# 阶段**，而且 $PTDBENCH_WORKSPACE、$PTDBENCH_MODELS 和 $PTDBENCH_DATASETS 都可写。
# agent / verifier 阶段的断网由框架按阶段设（runtimes/base.py 里 _network_policy_env 的
# HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE / HF_DATASETS_OFFLINE），任务的
# metadata.environment.env 里**不要**再写一遍：那份 env 三个阶段一视同仁，会把
# prepare 自己要用的网也关掉。
set -euo pipefail

# ---------------------------------------------------------------------------
# faiss-gpu —— 只有 search-r1 这两个任务要
# ---------------------------------------------------------------------------
# 为什么不写进 docker/images/slime-v0.3.2-sglang0.5.15-torch2.11.0-flashattn2.8.3-cu129-py3.12.Dockerfile：那个镜像是**四个 slime 任务
# 共用的**，加一个包等于改掉
# slime_retool / slime_retool_length_hack 的运行环境。它们的 outcome 是按没有
# faiss 的那套栈记下来的，不该被另一个任务的需求动到。
#
# 为什么是 GPU 版而不是 faiss-cpu（原 environment/Dockerfile 里写的是 faiss-cpu，
# 那是个从没生效过的 `import yaml || pip install ...` 守卫）：
# local_dense_retriever/retrieval_server.py:211-214 走的是
# faiss.GpuMultipleClonerOptions + faiss.index_cpu_to_all_gpus，CPU 构建里没有这
# 两个符号；recipe 也确实按 --faiss_gpu 起服务（run_qwen2.5_3B.sh:43）。版本取
# 1.15.0，是历史运行 wandb freeze 里记的那个。
#
# 为什么是 faiss-gpu-cuvs 而不是 faiss-gpu：faiss 1.15 的两个 GPU 轮子对应两套
# CUDA 布局，而这个环境是 CUDA 13。faiss-gpu 走的是 CUDA 12 的 per-component
# 布局，import 时按 `nvidia.cuda_runtime` / `nvidia.cublas` / `nvidia.curand` 三
# 个 nvidia-*-cu12 模块找 .so；本环境一个都没有（原 slime-v0.3.1 环境钉的是
# torch==2.11.0+cu130 和不带后缀的 nvidia-cuda-runtime==13.0.96 一系，装出来是
# nvidia/cu13/lib/ 一个目录）。硬塞一套 cu12 轮子进来也不行：retrieval_server.py
# 同一个进程里既 import faiss 又 import torch，两份 libcurand.so.10（cu12 的和
# cu13 的 soname 一样）会在同一个 CUDA context 里撞车。faiss-gpu-cuvs 就是为
# CUDA 13 编的那一个，走 nvidia/cu13/lib/。
#
# 为什么 cuVS 一系钉在 26.6 而不是最新的 26.8：libcuvs-cu13 26.8 要
# nvidia-nvjitlink>=13.3，原 slime-v0.3.1 环境钉死在 13.0.88，装上去
# import 会死在 `undefined symbol: __nvJitLinkCreate_13_3`。26.6 的约束是
# >=13.0，正好落在 faiss-gpu-cuvs 1.15.0 自己要的 >=26.06,<27 里。
#
# 为什么 libraft / librmm / rapids-logger 要逐个写出来：--no-deps 是故意的（见
# 下一段），而 libcuvs.so 的 DT_NEEDED 链是 rapids-logger -> librmm -> libraft
# -> libcuvs，faiss/__init__.py 也按这个顺序 preload。少一环就是一句
# "cannot open shared object file"。
#
# 为什么装到 $PTDBENCH_PREPARED 而不是直接 pip 进 env：prepare 是**独立的一次性
# 容器**（evaluator.py 的 _run_prepare 建了就关），装进镜像 site-packages 的东西出了这个容器就
# 没了。$PTDBENCH_PREPARED 是唯一会被后两个阶段只读挂进去的目录。约 640MB。
FAISS_SITE="$PTDBENCH_PREPARED/pysite"
# 为什么要走集群 PyPI 镜像：cybertron 的 whitelist-proxy 能和 files.pythonhosted.org
# （PyPI 的 fastly CDN）握上手，但下这几个几百 MB 的 RAPIDS cu13 轮子时会被半途
# reset（SSLEOFError: UNEXPECTED_EOF_WHILE_READING），和 HF xet cas-bridge 被挡是
# 同一类。集群自建的 proxying 源直接把 pythonhosted 的 wheel 代理过来，稳得多。
# 默认用京算（cluster=beijing）的源；换集群时用 PTDBENCH_PIP_INDEX_URL /
# PTDBENCH_PIP_TRUSTED_HOST 覆盖，别把某个集群的地址写死进 task。设成空串可强制
# 回退到原来的直连 pythonhosted 行为。
PIP_INDEX_URL="${PTDBENCH_PIP_INDEX_URL-https://pypi.mb1.bj3.paratera.com/root/pypi/+simple}"
PIP_TRUSTED_HOST="${PTDBENCH_PIP_TRUSTED_HOST-pypi.mb1.bj3.paratera.com}"
PIP_INDEX_ARGS=()
if [ -n "$PIP_INDEX_URL" ]; then
    PIP_INDEX_ARGS+=(--index-url "$PIP_INDEX_URL")
    [ -n "$PIP_TRUSTED_HOST" ] && PIP_INDEX_ARGS+=(--trusted-host "$PIP_TRUSTED_HOST")
fi

# nvidia cu13 五件（nvjitlink / cublas / curand / cusolver / nvrtc）装进系统
# dist-packages，不进 $FAISS_SITE——faiss loader 在 import 时用 `nvidia.cu13`
# 拿路径，而 dist-packages 里那个 0 字节的 nvidia/__init__.py stub 让 nvidia 成了
# regular package，--target 装出来的 $FAISS_SITE/nvidia 永远参与不进
# `import nvidia.cu13` 的解析，loader 只认 dist-packages 那份 __path__[0]。镜像里确
# 实有 nvidia/ 目录，但那是 torch 2.11.0+cu129 带的 cu12 布局（nvidia/nvjitlink/lib/
# libnvJitLink.so.12 一系）；faiss-gpu-cuvs 要的 nvidia/cu13/lib/ 下的一套 .so，镜像的
# cu13 目录只装了 cudart/nvvm/nvblas。前三个（.13 soname 的 nvjitlink / cublas /
# curand）缺了，import 死在 "failed to load libnvJitLink.so.13"——那是 faiss
# loader 自己 RTLD_GLOBAL 预载的五件里的三件。
#
# 后两个缺得更深：libcuvs.so 的 DT_NEEDED 里有 libcusolver.so.12 和
# libnvrtc.so.13（镜像只有 .so.11 / .so.12，torch 的 cu12 布局里没有这两个
# soname），预载完五件之后 rapids 的 load.py 去 dlopen libcuvs.so，缺了就
# **静默**返回 None，最后 _swigfaiss.abi3.so 报 "libcuvs.so: cannot open shared
# object file"——离原因隔着三层。cusolver 轮子的版本号是 12.3.5.3 而不是
# 13.x：CUDA 13 这条线里 cusolver 的**库版本**就是 12（soname .so.12），
# unsuffixed 的 nvidia-cusolver 到 12.3.5.3 为止，装出来照样落在
# nvidia/cu13/lib/（其余四件同款布局，已对着 RECORD 逐个核过）。nvrtc 的发行名
# 带 cuda- 前缀（nvidia-cuda-nvrtc，13.4.92）。
#
# 这五个与镜像里已有的 *-cu12 同名物是不同的发行名（nvidia-nvjitlink vs
# nvidia-nvjitlink-cu12 一系），pip 元数据不冲突，soname 也不撞（.13 vs .12），
# retrieval server 进程里 torch（cu12 轮子）和 faiss（cu13 目录）两套各走各的，
# 同进程 import 两者 + GPU cloner 冒烟已过。cuda-runtime 不装：镜像的
# nvidia-cuda-runtime==13.3.29 已经把 libcudart.so.13 放在 cu13/lib/ 里了。这一步
# 写在 PIP_INDEX_ARGS 定义之后才能吃到集群的镜像源——unsuffixed 的 cu13 轮子
# 不在 pypi.org 上。它写的是 prepare 所在容器的系统目录，devspace 重建后由守卫
# 重装（本任务只在 attach 模式的 devspace 上跑，prepare/agent/verifier 同容器）。
# 版本取 nvjitlink 13.4.92 / cublas 13.8.2.12 / curand 10.4.4.72 / cusolver
# 12.3.5.3 / cuda-nvrtc 13.4.92，libcuvs 26.6 要的 __nvJitLinkCreate_13_0 符号
# 它们都有。
NV13_SYS="/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib"
cu13_ready() {
    local soname
    for soname in libnvJitLink.so.13 libcublas.so.13 libcurand.so.10 \
                  libcusolver.so.12 libnvrtc.so.13; do
        [ -e "$NV13_SYS/$soname" ] || return 1
    done
    return 0
}
if ! cu13_ready; then
    echo "[prepare] installing CUDA-13 nvidia runtime components into system dist-packages"
    python3 -m pip install --no-cache-dir --no-deps --no-user \
        --retries 10 --timeout 120 \
        ${PIP_INDEX_ARGS[@]+"${PIP_INDEX_ARGS[@]}"} \
        nvidia-nvjitlink==13.4.92 \
        nvidia-cublas==13.8.2.12 \
        nvidia-curand==10.4.4.72 \
        nvidia-cusolver==12.3.5.3 \
        nvidia-cuda-nvrtc==13.4.92
else
    echo "[prepare] CUDA-13 nvidia runtime components already present"
fi
# 守卫认的是这一版的 dist-info，不是 `faiss/` 目录：认目录的话，一棵装了一半的、
# 或者上一版 cu12 的树会被当成装好了，然后在 agent 阶段才炸。
FAISS_MARKER="$FAISS_SITE/faiss_gpu_cuvs-1.15.0.dist-info"
if [ ! -d "$FAISS_MARKER" ]; then
    if [ -e "$FAISS_SITE" ]; then
        echo "[prepare] $FAISS_SITE is stale or incomplete; reinstalling from scratch"
        rm -rf "$FAISS_SITE"
    fi
    echo "[prepare] installing faiss-gpu-cuvs==1.15.0 (CUDA 13) into $FAISS_SITE${PIP_INDEX_URL:+ via $PIP_INDEX_URL}"
    # --no-user: some images set PIP_USER=1 (or pip.conf [install] user=true),
    # which is mutually exclusive with --target ("Can not combine '--user' and
    # '--target'"). Same guard as reward_design/task_evaluator's pyext install.
    python3 -m pip install --no-cache-dir --no-deps --no-user \
        --retries 10 --timeout 120 \
        ${PIP_INDEX_ARGS[@]+"${PIP_INDEX_ARGS[@]}"} \
        --target "$FAISS_SITE" \
        faiss-gpu-cuvs==1.15.0 \
        libcuvs-cu13==26.6.0 \
        libraft-cu13==26.6.0 \
        librmm-cu13==26.6.0 \
        rapids-logger==0.2.3
else
    echo "[prepare] faiss already present in $FAISS_SITE"
fi

# libcuvs.so 的 RPATH 写死的是**相对 pysite** 的路径：
# $ORIGIN:$ORIGIN/../../nvidia/cu13/lib:$ORIGIN/../../nvidia/nccl/lib:$ORIGIN/../../libcuvs_cu13.libs。
# 系统安装（上一段）够不着它的 cu13 / nccl 两个元素；rapids 的 load.py（libcuvs/
# load.py 的 _load_wheel_installation）在 dlopen libcuvs.so 失败时**静默**返回
# None，然后 _swigfaiss.abi3.so 死在 "libcuvs.so: cannot open shared object
# file"。这里在 pysite 里铺一层 symlink 指回系统目录，让 RPATH 那两个元素命中：
# $FAISS_SITE/nvidia/cu13/lib -> dist-packages/nvidia/cu13/lib（五件 + cudart 都
# 在那），nccl 同理（镜像 torch cu12 布局带的，ldconfig 本来就解析得了，铺上这层
# 让它不依赖 ldconfig）。$FAISS_SITE/nvidia 没有 __init__.py，PEP 420 下不参与
# `import nvidia.cu13` 的解析（dist-packages 的 regular package 永远优先）——
# 这层纯粹是给动态链接器看的。每次 run 的 $FAISS_SITE 都是全新目录，所以每遍
# prepare 都要重建（ln -sfn 幂等）。
NVROOT="$(python3 -c 'import nvidia.cu13, os; print(os.path.dirname(nvidia.cu13.__path__[0]))')"
mkdir -p "$FAISS_SITE/nvidia/cu13" "$FAISS_SITE/nvidia/nccl"
ln -sfn "$NVROOT/cu13/lib" "$FAISS_SITE/nvidia/cu13/lib"
ln -sfn "$NVROOT/nccl/lib" "$FAISS_SITE/nvidia/nccl/lib"

# 后两个阶段靠这一行看见它。注意 slime 的启动脚本对 PYTHONPATH 是**赋值**不是追加
# （`export PYTHONPATH=/workspace:/root/Megatron-LM`，且在 ray runtime_env 里再写
# 一遍），所以这一行进不了训练进程——不要紧，训练侧不 import faiss，import 它的是
# 单独起的那个 retrieval server，它跑在普通 shell 里，看得见。
if ! grep -qs "^PYTHONPATH=" "$PTDBENCH_PREPARED/env"; then
    echo "PYTHONPATH=$FAISS_SITE" >> "$PTDBENCH_PREPARED/env"
fi

python3 "$PTDBENCH_ENVIRONMENT/ptdbench_fetch.py" \
    --repo Qwen/Qwen2.5-3B \
    --dest "$PTDBENCH_WORKSPACE/Qwen2.5-3B"

# 检索用的 e5 编码器 
python3 "$PTDBENCH_ENVIRONMENT/ptdbench_fetch.py" \
    --repo intfloat/e5-base-v2 \
    --dest "$PTDBENCH_WORKSPACE/e5-base-v2"

# 训练集 / 验证集
python3 "$PTDBENCH_ENVIRONMENT/ptdbench_dataset.py" \
    --recipe search_r1_nq_hotpotqa \
    --dest "$PTDBENCH_WORKSPACE/nq_hotpotqa_train"

# 检索语料 + FAISS 索引 
python3 "$PTDBENCH_ENVIRONMENT/ptdbench_dataset.py" \
    --recipe wiki18_e5 \
    --dest "$PTDBENCH_WORKSPACE/wiki-18"

# --- 预热语料的 arrow 缓存 --------------------------------------------------
# 检索服务的 load_corpus 走的是
# `datasets.load_dataset("json", data_files=..., split="train", num_proc=4)`
# （local_dense_retriever/retrieval_server.py:34-36），也就是每次冷启动都要把那份
# 14GB 的 jsonl 转成一棵 arrow 树。转换本身是 prepare 该干的活：它只读已经备好的
# 语料、不要 GPU、结果能跨 run 复用，而放到 agent 阶段就是在计时的重跑里白等十几
# 分钟。
#
# 进程留不住，缓存留得住——这是 prepare 在检索服务这件事上唯一能帮上的地方。服务
# 本身不在这里起，原因见文件末尾。
#
# 两个目录而不是一个，是被读写权限逼出来的：
#
#   * $PTDBENCH_DATASETS 跨 run 存活，但 agent / verifier 阶段是只读挂载的，而
#     datasets 即便是缓存命中也要在缓存目录里开一个 .lock，只读就直接报错。
#   * $PTDBENCH_WORKSPACE 两个阶段都可写，但它是 per-run 的副本，每次新 run 都是
#     空的。
#
# 所以真身建在 datasets 根下，再硬链一份进 workspace：同一个文件系统上这份拷贝
# 不花字节（prepare.sh 其余几步的 materialize 也是这么做的），而 agent 阶段写的
# 锁文件是新文件，落在 workspace 那一侧，碰不到跨 run 的那份。
#
# 指纹对得上才算命中：datasets 的 config id 里有 data_files 的路径和 mtime。路径
# 两个阶段都是 /workspace/wiki-18/wiki-18.jsonl（容器里的路径一致），mtime 跟着
# 硬链走的是同一个 inode，所以跨 run 也稳。万一哪天对不上，代价只是服务自己再转
# 一次，不会报错——这也是下面整段失败也只警告不退出的原因。
HF_CACHE_PERSISTENT="$PTDBENCH_DATASETS/hf_datasets_cache"
HF_CACHE_RUN="$PTDBENCH_WORKSPACE/.hf_datasets_cache"
CORPUS_JSONL="$PTDBENCH_WORKSPACE/wiki-18/wiki-18.jsonl"

warm_corpus_cache() {
    mkdir -p "$HF_CACHE_PERSISTENT"
    # num_proc 和 retrieval_server.py 那行一字不差：转换的产物路径里带着它，写不
    # 一样就是白转一遍。
    HF_DATASETS_CACHE="$HF_CACHE_PERSISTENT" \
    CORPUS_JSONL="$CORPUS_JSONL" \
    python3 - <<'PY'
import os

import datasets

corpus = os.environ["CORPUS_JSONL"]
print(f"[prepare] building the arrow cache for {corpus}", flush=True)
rows = datasets.load_dataset("json", data_files=corpus, split="train", num_proc=4)
print(f"[prepare] corpus cached: {len(rows)} rows", flush=True)
PY
    # -a 保 mtime（指纹要用），-l 让这份拷贝只花 inode 不花字节。跨文件系统时 -l
    # 会失败，那就老实拷一份：慢，但不能因此没有缓存。
    rm -rf "$HF_CACHE_RUN"
    cp -al "$HF_CACHE_PERSISTENT" "$HF_CACHE_RUN" 2>/dev/null \
        || cp -a "$HF_CACHE_PERSISTENT" "$HF_CACHE_RUN"
}

if warm_corpus_cache; then
    echo "[prepare] corpus arrow cache staged at $HF_CACHE_RUN"
else
    # 缓存是优化，不是资产。失败就让检索服务自己在启动时转，慢十几分钟而已；这里
    # 退出会把整个 prepare 连着已经下好的 90GB 一起判成失败。
    echo "[prepare] warming the corpus arrow cache failed; the retrieval server" >&2
    echo "[prepare] will do the conversion itself on startup (slower, still correct)." >&2
    rm -rf "$HF_CACHE_RUN"
fi

# 后两个阶段要看见这个缓存，指的是 workspace 那一份（只有它可写）。和上面 faiss
# 的 PYTHONPATH 同一个机制：prepared/env 是 KEY=VALUE，由 harness 在每个阶段 export
# （ptdbench/runtimes/base.py:PTDBENCH_ACTIVATE_BODY）。
if ! grep -qs "^HF_DATASETS_CACHE=" "$PTDBENCH_PREPARED/env"; then
    echo "HF_DATASETS_CACHE=$HF_CACHE_RUN" >> "$PTDBENCH_PREPARED/env"
fi

# mcore (torch_dist) 格式的 ref 权重
SLIME_DIR="$PTDBENCH_WORKSPACE"
HF_MODEL_DIR="$PTDBENCH_WORKSPACE/Qwen2.5-3B"
DIST_MODEL_DIR="$PTDBENCH_WORKSPACE/Qwen2.5-3B_torch_dist"

CONVERT_GPU="$(
python3 <<'PY' 2>/dev/null
import sys

try:
    import torch
except Exception:
    sys.exit(0)
if not torch.cuda.is_available():
    sys.exit(0)

best, most_free = None, 0
for index in range(torch.cuda.device_count()):
    try:
        free, _total = torch.cuda.mem_get_info(index)
    except Exception:
        # 卡满到连 context 都建不起来，就是不能用。
        continue
    if free > most_free:
        best, most_free = index, free

# 12GiB：够 6GB 权重 + 建模型的峰值，还留一半余量。宁可跳过也不要跑到一半 OOM。
if best is not None and most_free >= (12 << 30):
    print(best)
PY
)" || CONVERT_GPU=""

if [ -n "$CONVERT_GPU" ]; then
    echo "[prepare] converting $HF_MODEL_DIR -> $DIST_MODEL_DIR (GPU $CONVERT_GPU, WORLD_SIZE=1)"
    # shellcheck source=/dev/null
    source "$SLIME_DIR/scripts/models/qwen2.5-3B.sh"
    mkdir -p "$DIST_MODEL_DIR"
    if CUDA_VISIBLE_DEVICES="$CONVERT_GPU" WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
       PYTHONPATH="/root/Megatron-LM:$SLIME_DIR" \
       python3 "$SLIME_DIR/tools/convert_hf_to_torch_dist.py" \
           "${MODEL_ARGS[@]}" \
           --hf-checkpoint "$HF_MODEL_DIR" \
           --save "$DIST_MODEL_DIR"; then
        echo "[prepare] torch_dist reference ready at $DIST_MODEL_DIR"
    else
        # 半截的 dist-checkpoint 比没有糟糕得多：训练脚本的守卫只看目录存不存在、
        # 空不空，转了一半的树会被它当成做好了，然后 --ref-load 读出错的权重。
        echo "[prepare] conversion failed; dropping $DIST_MODEL_DIR so the training" >&2
        echo "[prepare] script's own guard redoes it (run_qwen2.5_3B.sh:79-86)." >&2
        rm -rf "$DIST_MODEL_DIR"
    fi
else
    echo "[prepare] no GPU with at least 12GiB free (or no GPU at all); leaving the"
    echo "[prepare] torch_dist conversion to the training script's own guard"
    echo "[prepare] (examples/search-r1/run_qwen2.5_3B.sh:79-86)."
fi

echo "[prepare] assets staged under $PTDBENCH_WORKSPACE"
echo "[prepare] the retrieval server is not started here; run_qwen2.5_3B.sh brings it"
echo "[prepare] up itself (examples/search-r1/start_retriver.sh) and waits for :8000."
