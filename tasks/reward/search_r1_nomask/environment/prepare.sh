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
# "cannot open shared object file"。CUDA 侧的依赖（cublas / curand / nvjitlink /
# nccl）不写，是因为环境里已经有了，多装一份就回到上面那个撞车问题。
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

# 后两个阶段靠这一行看见它。注意 slime 的启动脚本对 PYTHONPATH 是**赋值**不是追加
# （`export PYTHONPATH=/workspace:/root/Megatron-LM`，且在 ray runtime_env 里再写
# 一遍），所以这一行进不了训练进程——不要紧，训练侧不 import faiss，import 它的是
# 单独起的那个 retrieval server，它跑在普通 shell 里，看得见。
if ! grep -qs "^PYTHONPATH=" "$PTDBENCH_PREPARED/env"; then
    echo "PYTHONPATH=$FAISS_SITE" >> "$PTDBENCH_PREPARED/env"
fi

# ---------------------------------------------------------------------------
# 权重 / 检索模型 / 数据集 / 检索索引
# ---------------------------------------------------------------------------
# 五份资产，四份在这里备好、一份现场转。目录名和文件名由
# examples/search-r1/run_qwen2.5_3B.sh 的 DATA_DIR / MODEL_DIR 守卫决定（两者默认
# 都是 $PWD，容器里就是 /workspace），**一个字都不能改**：
#
#   * 改了名字，run_qwen2.5_3B.sh:74-77 的守卫命中不到，它会在 agent 阶段自己去
#     `hf download`，而那个阶段是断网的（runtimes/base.py 的 _network_policy_env 按阶段打开
#     HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE / HF_DATASETS_OFFLINE），报错离原因很远。
#   * 同样的道理对 run_qwen2.5_3B.sh:62 起手就跑的那次 prepare_data.sh 也成立：
#     它靠 `-f train.parquet` / `-f e5_Flat.index` 跳过，下面这几步就是那个脚本的
#     钉死版本（钉 revision、校字节数、产物进缓存跨 run 复用）。
#
# 磁盘：缓存建成之后这五份加起来约 90GB，其中 wiki-18 一家占 78GB。第一次构建
# 峰值还要再多 ~20GB（见 ptdbench_datasets/wiki18_e5.py）。materialize 走硬链，
# 所以只要 $PTDBENCH_DATASETS / $PTDBENCH_MODELS 和 run 目录在同一个文件系统上，
# 第二个 run 起就不再花这些字节。

# --- actor / ref 的 HF 权重（--hf-checkpoint） ------------------------------
# allow 列表挡掉 .gitattributes 和 README，其余 config / tokenizer 模型自己要读。
# 只对两个分片记 sha256：revision 已经把小文件钉死了，逐个记只是让任务包难写。
ptdbench_fetch_hf \
    --repo Qwen/Qwen2.5-3B \
    --revision 3aab1f1954e9cc14eb9509a215f9e5ca08227a9b \
    --allow LICENSE \
    --allow config.json \
    --allow generation_config.json \
    --allow merges.txt \
    --allow model-00001-of-00002.safetensors \
    --allow model-00002-of-00002.safetensors \
    --allow model.safetensors.index.json \
    --allow tokenizer.json \
    --allow tokenizer_config.json \
    --allow vocab.json \
    --expect model-00001-of-00002.safetensors=f9558df91d3b89b4826e4db37439edb52f1d62a4fd602685013e7ca6b9f60f8f \
    --expect-size model-00001-of-00002.safetensors=3968658944 \
    --expect model-00002-of-00002.safetensors=51410930d5cf19a998fdb17ef0c46e4d9ace72c97a975a3331395a8a500f5edb \
    --expect-size model-00002-of-00002.safetensors=2203268048 \
    --dest "$PTDBENCH_WORKSPACE/Qwen2.5-3B"

# --- 检索用的 e5 编码器 -----------------------------------------------------
# retrieval_server.py:44-51 拿 --retriever_model 直接喂 AutoModel/AutoTokenizer。
# 上游给的是仓库名 `intfloat/e5-base-v2`，那在 agent 阶段等于一次联网下载；
# start_retriver.sh 因此指向下面这个本地目录（两个 search-r1 任务同步改过）。
# onnx/ 和 openvino/ 两棵子树不取：同一份权重的另外两种格式，这里没人读。
ptdbench_fetch_hf \
    --repo intfloat/e5-base-v2 \
    --revision f52bf8ec8c7124536f0efb74aca902b2995e5bcd \
    --allow config.json \
    --allow model.safetensors \
    --allow special_tokens_map.json \
    --allow tokenizer.json \
    --allow tokenizer_config.json \
    --allow vocab.txt \
    --expect model.safetensors=d0d559c47d5f71b1d280b13b62a2657f3e3bc70c0786f9ab91a36545e6a8f693 \
    --expect-size model.safetensors=437955512 \
    --dest "$PTDBENCH_WORKSPACE/e5-base-v2"

# --- 训练集 / 验证集（--prompt-data / --eval-prompt-data） -------------------
# nq + hotpotqa 合出来的 {train,test}.parquet。配方是 Search-R1 那两个 merge 脚本
# 的等价实现，prompt 模板逐字一致；行顺序也一致——主指标 eval/nq_test 取的是
# `$TEST_DATA@[0:3000]`，靠的就是 test.parquet 前 3610 行全是 NQ。
# 不给 --expect：parquet 的字节随 pandas/pyarrow 版本漂，配方自己断言行数和 schema。
ptdbench_dataset \
    --recipe search_r1_nq_hotpotqa \
    --dest "$PTDBENCH_WORKSPACE/nq_hotpotqa_train"

# --- 检索语料 + FAISS 索引 --------------------------------------------------
# 上游把索引切成两块 part_*，拼起来才是 e5_Flat.index（60GiB）；corpus 那份文件名
# 叫 wiki-18.jsonl.gz，里面其实是个单成员的 tar（配方里有详细说明，照上游
# prepare_data.sh 的 `gzip -df` 解会得到一份读不了的 jsonl）。这一步是整个任务里
# 最贵的一步，第一次要下 70GB 左右；之后是缓存命中 + 硬链。
#
# 中途失败不用怕重下：配方把分片和拼好的索引放在 $PTDBENCH_DATASETS/scratch 底下，
# 那个目录不随 staging 一起删。
ptdbench_dataset \
    --recipe wiki18_e5 \
    --dest "$PTDBENCH_WORKSPACE/wiki-18" \
    --expect-size e5_Flat.index=64559075373 \
    --expect-size wiki-18.jsonl=14393573105

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

# --- mcore (torch_dist) 格式的 ref 权重（--ref-load） ------------------------
# 没有上游仓，只能由上面那份现场转出来。原 recipe 在训练脚本里顺手转
# （run_qwen2.5_3B.sh:79-86），这里提前做掉，把 GPU 小时留给训练本身。
#
# 产物**不保证逐字节可复现**——dist-checkpoint 的分片布局跟 world size 走。
# WORLD_SIZE 钉死成 1 就得到 pp=1 的单一布局，换机器换卡数也不会变形状
# （convert_hf_to_torch_dist.py 用 WORLD_SIZE 推 pp size）。3B 的 bf16 权重约
# 6GB，单卡放得下。MODEL_ARGS 直接 source 任务自带的那份，不在这里抄一遍——抄了
# 就会和 recipe 漂移；训练脚本转换时也没有额外覆盖 --rotary-base，所以这里同样
# 不覆盖，ref 权重的位置编码才和 actor 对得上。
SLIME_DIR="$PTDBENCH_WORKSPACE"
HF_MODEL_DIR="$PTDBENCH_WORKSPACE/Qwen2.5-3B"
DIST_MODEL_DIR="$PTDBENCH_WORKSPACE/Qwen2.5-3B_torch_dist"

# 转换要 CUDA + NCCL（dist.init_process_group + torch.cuda.set_device），而且要一张
# **空闲显存够用**的卡：3B 的 bf16 权重约 6GB，Megatron/TE 建模型时还要一截余量。
# 两种情况拿不到卡，都跳过而不是报错：
#
#   * harbor 后端把 prepare 跑成 build-time 的 RUN 层，那里既没有任务 env 也没有 GPU；
#   * 共享机器上八张卡可能都被别的作业占着（这不是假设，实测过：整机每张卡只剩
#     9GB，默认落在 GPU 0 上就是一个 OutOfMemoryError）。
#
# 训练脚本自己的 `if [ ! -d "$DIST_MODEL_DIR" ]` 守卫会在 agent 阶段补做，那时
# /workspace 仍然可写、而且整机是留给这次训练的。少做一次是慢，做不成才是错。
#
# 挑卡而不是认准 0 号：prepare 看得到全部八张，哪张空是运行时才知道的。
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

# 检索服务本身不在这里起：prepare 是一个建了就关的一次性容器
# （evaluation/evaluator.py 的 _run_prepare），进程活不到 agent 阶段；而且这个阶段是被前台
# 等返回的，不 detach 地起一个只会挂到超时。harbor 那条路上 prepare 还可能是
# build-time 的 RUN 层，那里连 GPU 都没有。
#
# 所以起服务这件事放在了训练脚本自己手里：run_qwen2.5_3B.sh 的 retrieval 段落在
# :8000 上探不到监听时会自己后台拉起 start_retriver.sh 并等它就绪。prepare 在这件
# 事上的贡献是上面那段 arrow 缓存预热——把冷启动里最慢的一段提前付掉。
echo "[prepare] assets staged under $PTDBENCH_WORKSPACE"
echo "[prepare] the retrieval server is not started here; run_qwen2.5_3B.sh brings it"
echo "[prepare] up itself (examples/search-r1/start_retriver.sh) and waits for :8000."
