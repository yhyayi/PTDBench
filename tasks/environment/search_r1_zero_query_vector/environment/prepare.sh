#!/usr/bin/env bash
# 本任务跑起来之前要准备的东西。
#
# 跑在独立的 prepare 阶段容器里：root、`search-r1-verl` 镜像、**唯一有网的阶段**，
# 而且 $PTDBENCH_WORKSPACE、$PTDBENCH_MODELS、$PTDBENCH_DATASETS 都可写。
# agent / verifier 阶段的断网由框架按阶段设（runtimes/base.py 里 _network_policy_env 的
# HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE / HF_DATASETS_OFFLINE），任务的
# metadata.environment.env 里**不要**再写一遍：那份 env 三个阶段一视同仁，会把
# prepare 自己要用的网也关掉。
#
# 三件事：两个模型（约 3.5GB）下载到 workspace，一个 FAISS 索引（37MB）就地重建。
# 数据本身（train/test.parquet + corpus.jsonl，6.9MB）随任务包走，不在这里下。
# 目标目录名由 workspace/run.py 和 workspace/serve_retriever.py 的 SHARED 常量决定：
#
#   shared/models/Qwen2.5-1.5B-Instruct   actor / ref 的初始权重（BASE_MODEL）
#   shared/models/e5-base-v2              检索用的查询/文档编码器
#   shared/data/{train,test}.parquet      已随包
#   shared/data/corpus.jsonl              已随包
#   shared/data/e5_Flat.index             这里重建
#
# run.py 启动前会逐个检查这五条路径（run.py 的 `required` 列表），少一个就拒绝开跑，
# 所以路径写错不会拖到训练中途才炸。
set -euo pipefail

ENVIRONMENT_DIR="${PTDBENCH_ENVIRONMENT:-$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
DATA_DIR="$PTDBENCH_WORKSPACE/shared/data"
MODEL_DIR="$PTDBENCH_WORKSPACE/shared/models"

# ---------------------------------------------------------------------------
# 随包数据的完整性
# ---------------------------------------------------------------------------
# 这三个文件是 git 里的，不是下载来的，所以这里不是"校验下载"而是"校验 checkout"：
# LFS 没装好、或者谁手工动过 parquet，症状会是训练到第一个 eval 才报行数不对。
# 顺带也把 data_manifest.json 和实际文件绑在一起——manifest 是 readme 和 oracle
# 引用的那份出处，漂了就都对不上。
echo "[prepare] verifying the data shipped in the task package"
python3 - "$ENVIRONMENT_DIR/data_manifest.json" "$DATA_DIR" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path, data_dir = Path(sys.argv[1]), Path(sys.argv[2])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
for relative, expected in manifest["shipped_in_workspace"].items():
    path = data_dir.parent.parent / relative
    if not path.is_file():
        raise SystemExit(f"[prepare] missing shipped data file: {path}")
    size = path.stat().st_size
    if size != expected["size"]:
        raise SystemExit(
            f"[prepare] {path}: {size} bytes, manifest says {expected['size']}"
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected["sha256"]:
        raise SystemExit(
            f"[prepare] {path}: sha256 {digest}, manifest says {expected['sha256']}"
        )
    print(f"[prepare]   ok {relative}")
PY

# ---------------------------------------------------------------------------
# actor / ref 的初始权重
# ---------------------------------------------------------------------------
# revision 是交付包 README 记下的那次实验用的提交；sha256 来自
# MODEL_SHA256SUMS，和 results/baseline32/experiment.json 的 input_sha256 对得上
# （那一条写的是 model.safetensors 的 dd924a11…），也就是说这里下到的权重和产生
# 那条 15.6% 曲线的权重是同一份文件。
#
# allow 列表挡掉 .gitattributes 和 README；剩下的 config / tokenizer 都是
# transformers 自己要读的。只对 model.safetensors 记 sha256：revision 已经把小文件
# 钉死了，逐个记只是让任务包难写。
ptdbench_fetch_hf \
    --repo Qwen/Qwen2.5-1.5B-Instruct \
    --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
    --allow config.json \
    --allow generation_config.json \
    --allow merges.txt \
    --allow model.safetensors \
    --allow tokenizer.json \
    --allow tokenizer_config.json \
    --allow vocab.json \
    --expect model.safetensors=dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee \
    --dest "$MODEL_DIR/Qwen2.5-1.5B-Instruct"

# ---------------------------------------------------------------------------
# 检索编码器
# ---------------------------------------------------------------------------
# 交付包没记 e5 的提交号（README 里写明了这点），但 f52bf8ec… 这个 revision 下到的
# model.safetensors 的 sha256 和交付包里那份逐字节相同（d0d559c4…），而且
# tasks/slime_search_r1_nomask 已经按这个 revision 钉了同一个文件——两个任务因此共用
# 一条缓存。sha 不对就停下来核版本，别拿另一版编码器去配这个索引：文档向量和查询
# 向量必须出自同一套权重，否则检索质量会无声下降。
#
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
    --dest "$MODEL_DIR/e5-base-v2"

# ---------------------------------------------------------------------------
# FAISS 索引
# ---------------------------------------------------------------------------
# 为什么不随包：它是 corpus.jsonl 加一个钉死的编码器就能算出来的东西，37MB，是它
# 自己来源文件的五倍大。为什么不让 agent 阶段建：那是计时的阶段，而这一步既不需要
# agent 做任何决定、也能跨 run 复用。
#
# 真身放在 $PTDBENCH_DATASETS 下跨 run 存活，再硬链一份进 workspace；同一个文件系统
# 上这份拷贝不花字节（和其他任务的 materialize 一样），跨文件系统时退化成真拷贝。
# 这样第二个 run 起就不再花这几分钟的编码时间。
#
# 用 .complete 标记而不是认索引文件本身存在：write_index 是在 build_index.py 里
# 原子替换的，但缓存目录可能是上一次被 kill 的 prepare 留下的半截状态，认文件存在
# 会把它当成建好了。
# $PTDBENCH_DATASETS is the empty string when the harness has no dataset cache
# configured (the local backend can run that way). Falling through with an empty
# value would build the cache at /search_r1_hotpotqa_e5; keep it in the run's own
# workspace instead, where it is at least correct, just not reused.
INDEX_CACHE="${PTDBENCH_DATASETS:-$PTDBENCH_WORKSPACE/.index-cache}/search_r1_hotpotqa_e5"
INDEX_CACHED="$INDEX_CACHE/e5_Flat.index"
INDEX_MARKER="$INDEX_CACHE/.complete"

if [ ! -f "$INDEX_MARKER" ] || [ ! -s "$INDEX_CACHED" ]; then
    rm -rf "$INDEX_CACHE"
    mkdir -p "$INDEX_CACHE"
    echo "[prepare] building $INDEX_CACHED from corpus.jsonl"
    python3 "$ENVIRONMENT_DIR/build_index.py" \
        --corpus "$DATA_DIR/corpus.jsonl" \
        --model "$MODEL_DIR/e5-base-v2" \
        --out "$INDEX_CACHED"
    touch "$INDEX_MARKER"
else
    echo "[prepare] reusing the cached index at $INDEX_CACHED"
fi

rm -f "$DATA_DIR/e5_Flat.index"
cp -al "$INDEX_CACHED" "$DATA_DIR/e5_Flat.index" 2>/dev/null \
    || cp -a "$INDEX_CACHED" "$DATA_DIR/e5_Flat.index"

# 检索服务不在这里起：prepare 是一个建了就关的一次性容器
# （evaluation/evaluator.py 的 _run_prepare），进程活不到 agent 阶段；而且这个阶段是被
# 前台等返回的，不 detach 地起一个只会挂到超时。这件事交给 workspace/run.py：它在
# 自己的前台进程里拉起 serve_retriever.py、等 /openapi.json 就绪、结束时再收掉。
#
# 语料的 arrow 缓存也不在这里预热（slime_search_r1_nomask 的 prepare 做了那件事）：
# 那边的语料是 14GB，冷启动要转十几分钟；这里是 6.6MB，datasets 转换是秒级的。
echo "[prepare] assets staged under $PTDBENCH_WORKSPACE/shared"
echo "[prepare] next step is the agent's: python3 run.py --check validates them without a GPU"
