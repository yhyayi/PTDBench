"""PeterJinGo/wiki-18-e5-index + wiki-18-corpus -> wiki-18/{e5_Flat.index,wiki-18.jsonl}

tasks/slime_search_r1 和 tasks/slime_search_r1_nomask 共用这一份产物：recipe 的
检索服务要它（examples/search-r1/start_retriver.sh -> local_dense_retriever/
retrieval_server.py），而 run_qwen2.5_3B.sh 的 retrieval 段落（96 行起）在 :8000
上探不到监听时会自己拉起那个服务、并等它就绪，等不到就退出。

## 为什么是"配方"而不是 ptdbench_fetch_hf

两份上游资产下下来都不是能直接用的字节：

  * 索引被切成了两块 40GB 级的 `part_*`，谁也读不了，要 `cat` 成一个文件；
  * corpus 那份**文件名是 `wiki-18.jsonl.gz`，里面却是个 tar**。

第二条是个坑，值得写下来：它是单成员的 GNU tar.gz，成员名还带着打包那台机器上的
路径残留（见 CORPUS_MEMBER）。照上游 prepare_data.sh:110-113 的 `gzip -df` 解，
得到的是 tar 归档本身——前 100 字节是成员名，第 124 字节起是 base-256 编码的成员
大小，首字节 0x80。那个文件叫 wiki-18.jsonl，检索服务把它当 json 数据集读的时候
会死在 `UnicodeDecodeError: invalid start byte 0x80`，而报错离原因很远。所以这里
走 tarfile 抽成员，并且把成员名和大小都钉死——上游哪天重新打一次包，这里要立刻
停下来，而不是产出一份读不了的 corpus。

拼索引的规则仍然抄自 examples/search-r1/prepare_data.sh:100-113。

## 体量：这是本仓库最大的一份资产

    part_aa            42,949,672,960 B   (40 GiB)
    part_ab            21,609,402,413 B   (20 GiB)
    e5_Flat.index      64,559,075,373 B   (拼出来的，60 GiB)
    wiki-18.jsonl.gz    5,123,307,260 B   (其实是 tar.gz)
    wiki-18.jsonl      14,393,573,105 B   (抽出来的)

下载一块、追加一块、立刻删掉那一块，所以峰值占用约 86GB 而不是 150GB。缓存条目
建成之后是 ~79GB，每个 run 靠硬链拿到它（materialize），不会再复制一遍——前提是
$PTDBENCH_DATASETS 和 run 目录在同一个文件系统上。不在的话 ptdbench_dataset 会
退化成真复制，79GB 一次，留意一下。

## 中途失败：产物建在 staging 外面

ptdbench_dataset 的重试语义是"没有 .ptdbench-complete 标记就整棵删掉重来"。对
几分钟就能跑完的 parquet 配方没问题，对这一份不是：corpus 那一步出错，等于把已经
下好拼好的 60GiB 索引也一起丢掉，再下一遍 70GB。

所以两份产物先建在 $PTDBENCH_DATASETS/scratch/wiki18_e5 里，两份都齐了才
`shutil.move` 进 staging（同一个 datasets 根，是一次 rename）。失败时 scratch 会
**故意留着**，下一次重跑看到大小正确的 e5_Flat.index 就跳过拼接。代价是失败后盘上
留着几十 GB，路径会打印出来，不想要就自己删。

半拼好的索引不会被当成成品：跳过的条件是字节数完全相等，差一个字节就重拼。这条
不能松——faiss 不会拒绝一棵截断的索引，它只会返回错的近邻。

## 校验

只校大小，不算 sha256：上游没有公布拼好之后的哈希，而对 64GB 逐字节算一遍每次
构建要多花十几分钟，挡住的还是"截断"——那个大小就能看出来。分片本身由 revision
钉死，下载过程由 hf_hub_download 自己校验。

用法：
    python3 -m ptdbench_datasets.wiki18_e5 --out-dir ./prepared/wiki-18
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
from pathlib import Path

from ._common import hf_cache_root

REPO_INDEX = "PeterJinGo/wiki-18-e5-index"
REVISION_INDEX = "a4d31160a035f30764604f4827cd8f1d0315eb86"
# 顺序就是 `cat part_*` 的 shell glob 顺序；拼反了 faiss 仍然能读出一个头，
# 然后给出完全错误的检索结果。
PARTS = (("part_aa", 42949672960), ("part_ab", 21609402413))
INDEX_NAME = "e5_Flat.index"
INDEX_BYTES = sum(size for _, size in PARTS)

REPO_CORPUS = "PeterJinGo/wiki-18-corpus"
REVISION_CORPUS = "69c1c00ffe7c5554c68d8548355cb22e46aabc51"
CORPUS_GZ = "wiki-18.jsonl.gz"
CORPUS_GZ_BYTES = 5123307260
CORPUS_NAME = "wiki-18.jsonl"
# 归档里唯一那个成员的名字。是打包那台机器上的路径残留，不是笔误——见 docstring。
CORPUS_MEMBER = "data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl"
CORPUS_BYTES = 14393573105

# retrieval_server.py:34-41 把这个 jsonl 当 datasets 的 json 数据集读，
# generate_with_search.py:51 取的是 document["contents"]。
CORPUS_KEYS = ("id", "contents")

# 缓存键的一部分（见 docker/ptdbench_dataset.py）。改了配方就要改这个数。
# 仍然是 1：v1 从来没有产出过带 .ptdbench-complete 的条目（上面那个 tar 的坑让它
# 每次都死在 corpus 这一步），所以没有旧缓存需要作废。
RECIPE_VERSION = "1"

_CHUNK = 32 << 20


def scratch_dir(out_dir: Path) -> Path:
    """下载和拼接的工作目录，**在 staging 外面**（理由见 docstring）。

    宿主机手工跑（`python3 -m ptdbench_datasets.wiki18_e5`）时没有
    PTDBENCH_DATASETS，那就退回 out_dir 底下——那种用法本来就是自己挑目录。
    """
    root = os.environ.get("PTDBENCH_DATASETS")
    base = Path(root) if root else out_dir
    return base / "scratch" / "wiki18_e5"


def _done(path: Path, size: int) -> bool:
    """这一步的产物是不是已经完整地躺在那里了。

    只认字节数完全相等。">= size" 或者 "存在即可" 都会把上一次截断的结果放行。
    """
    return path.is_file() and path.stat().st_size == size


def _download(repo: str, revision: str, filename: str, expected: int, into: Path) -> Path:
    # 在 import 之前：见 _common.hf_cache_root()。40GB 本体靠下面的 local_dir 落在
    # 宿主机上，这里管的是 huggingface_hub 自己的元数据/etag 缓存。
    hf_cache_root()
    from huggingface_hub import hf_hub_download

    print(f"下载 {repo}@{revision[:12]} {filename} ({expected / 1e9:.1f} GB) ...", flush=True)
    path = Path(
        hf_hub_download(
            repo_id=repo,
            filename=filename,
            repo_type="dataset",
            revision=revision,
            # 直接写成普通文件，不在 blob 缓存里再留一份 40GB。断点续传的记账
            # （.cache/huggingface/download/*.incomplete）也留在这里，所以重跑
            # 接着下而不是从头下。
            local_dir=str(into),
        )
    )
    actual = path.stat().st_size
    if actual != expected:
        raise SystemExit(f"{filename} 下来是 {actual} 字节，配方记的是 {expected}。上游变了或者下载被截断。")
    return path


def build_index(scratch: Path) -> Path:
    """按 prepare_data.sh 的 `cat part_* > e5_Flat.index` 拼索引。"""
    index = scratch / INDEX_NAME
    if _done(index, INDEX_BYTES):
        print(f"  {INDEX_NAME} 已经在 {scratch} 里而且大小对得上，跳过拼接", flush=True)
        return index

    with index.open("wb") as sink:
        for name, size in PARTS:
            part = _download(REPO_INDEX, REVISION_INDEX, name, size, scratch)
            print(f"  追加 {name} -> {index.name}", flush=True)
            with part.open("rb") as source:
                shutil.copyfileobj(source, sink, _CHUNK)
            # 追加完立刻删：两块加起来 60GiB，留着就等于把峰值占用翻一倍。
            part.unlink()
    actual = index.stat().st_size
    if actual != INDEX_BYTES:
        raise SystemExit(f"{INDEX_NAME} 拼出来是 {actual} 字节，期望 {INDEX_BYTES}。")
    print(f"  -> {index} ({actual} 字节)", flush=True)
    return index


def _check_first_line(corpus: Path) -> None:
    with corpus.open("r", encoding="utf-8") as handle:
        first = handle.readline()
    if not first.strip():
        raise SystemExit(f"{CORPUS_NAME} 第一行是空的")
    row = json.loads(first)
    missing = [key for key in CORPUS_KEYS if key not in row]
    if missing:
        raise SystemExit(
            f"{CORPUS_NAME} 第一行缺字段 {missing}，检索服务按 {list(CORPUS_KEYS)} 取文档"
            f"（retrieval_server.py:39-41 + generate_with_search.py:51）"
        )


def build_corpus(scratch: Path) -> Path:
    """从 wiki-18.jsonl.gz 里抽出那个 jsonl 成员。

    上游那个 .gz 是 tar.gz，不是 gzip 过的 jsonl——见模块 docstring。
    """
    corpus = scratch / CORPUS_NAME
    if _done(corpus, CORPUS_BYTES):
        print(f"  {CORPUS_NAME} 已经在 {scratch} 里而且大小对得上，跳过解包", flush=True)
        _check_first_line(corpus)
        return corpus

    gz = _download(REPO_CORPUS, REVISION_CORPUS, CORPUS_GZ, CORPUS_GZ_BYTES, scratch)
    print(f"  从 {CORPUS_GZ} 里抽 {CORPUS_MEMBER} -> {corpus.name}", flush=True)
    # `r|gz` 是流式的：读一遍、边解边写。`r:gz` 为了先建成员表会把 14GB 整个过
    # 一遍，然后 extractfile 再过第二遍。
    with tarfile.open(gz, "r|gz") as archive:
        member = archive.next()
        while member is not None and not member.isfile():
            member = archive.next()
        if member is None:
            raise SystemExit(f"{CORPUS_GZ} 里一个常规文件成员都没有")
        if member.name != CORPUS_MEMBER:
            raise SystemExit(
                f"{CORPUS_GZ} 里的成员是 {member.name!r}，配方记的是 {CORPUS_MEMBER!r}。"
                f"上游重新打过包了，对一下内容再改配方。"
            )
        if member.size != CORPUS_BYTES:
            raise SystemExit(
                f"{CORPUS_MEMBER} 在归档里声明 {member.size} 字节，配方记的是 {CORPUS_BYTES}。"
            )
        source = archive.extractfile(member)
        with corpus.open("wb") as sink:
            shutil.copyfileobj(source, sink, _CHUNK)
    gz.unlink()

    actual = corpus.stat().st_size
    if actual != CORPUS_BYTES:
        raise SystemExit(f"{CORPUS_NAME} 抽出来是 {actual} 字节，期望 {CORPUS_BYTES}。")
    _check_first_line(corpus)
    print(f"  -> {corpus} ({actual} 字节)", flush=True)
    return corpus


def build(out_dir: Path, *, allow_count_mismatch: bool = False) -> list[Path]:
    """拼索引 + 抽 corpus，两份都齐了才搬进 out_dir；返回写出的路径。

    `allow_count_mismatch` 在这份配方里没有用武之地（这里没有"行数"这种东西，
    校的是字节数，而字节数对不上就是资产坏了，没有"确认后放行"的余地）。参数留着
    是因为 ptdbench_dataset 对所有配方调用同一个签名。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = scratch_dir(out_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    print(f"[wiki18_e5] 工作目录 {scratch}（失败会留着，下次接着用）", flush=True)

    built = [build_index(scratch), build_corpus(scratch)]

    written = []
    for source in built:
        target = out_dir / source.name
        # 同一个 datasets 根下面，move 是一次 rename；退化成跨盘复制也还是对的。
        shutil.move(str(source), str(target))
        written.append(target)

    # 到这里两份产物都已经搬走了，剩下的只有 hf_hub_download 的记账目录。
    shutil.rmtree(scratch, ignore_errors=True)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--allow-count-mismatch", action="store_true")
    args = ap.parse_args()

    written = build(args.out_dir, allow_count_mismatch=args.allow_count_mismatch)
    print()
    for path in written:
        print(f"    --expect-size {path.name}={path.stat().st_size} \\")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
