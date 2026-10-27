"""RUC-NLPIR/FlashRAG_datasets -> nq_hotpotqa_train/{train,test}.parquet

tasks/slime_search_r1 和 tasks/slime_search_r1_nomask 共用这一份产物（两个任务的
launch 脚本都要 /workspace/nq_hotpotqa_train/{train,test}.parquet，见
examples/search-r1/run_qwen2.5_3B.sh:64-66）。

## 这份配方是什么

它是 recipe 自带的那两个合并脚本的等价实现：

    workspace/Search-R1/scripts/data_process/qa_search_train_merge.py
    workspace/Search-R1/scripts/data_process/qa_search_test_merge.py

那两个脚本走 `datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', <source>)`，
也就是**不钉 revision**、而且只能在有网的进程里跑。prepare 是唯一有网的阶段，
但产物要能跨 run 复用（workspace 每次评测都是新的一份副本），所以这里按
ptdbench_datasets 的惯例重建：钉死 revision、断言行数、用 pandas 写 parquet。

prompt 模板、字段名、`extra_info.index` 的计数方式都是从那两个脚本**逐字**抄来的
（见下面 make_prefix / build_rows 的注释）。改这里必须同时改那边，反过来也一样——
它们的产物要是不一致，agent 在 /workspace 里手动跑一遍 prepare_data.sh 就会得到
和判分基线不同的数据。

## 和那两个脚本的唯一差别：不带 metadata 列

上游 hotpotqa 的每行带一个 `metadata`（supporting_facts + 十篇 context 全文，占
569MB 里的绝大部分），nq 的行则根本没有这个键。合并脚本靠 `datasets` 的 schema
对齐去吃这个差异；这里换成 pandas 之后，一列"一半是 struct、一半是缺失"既要多写
一段补空逻辑，又会把 500MB 训练侧根本不读的字节搬进每个 run 的 workspace。

slime 侧只读 `--input-key prompt` 和 `--label-key reward_model`
（run_qwen2.5_3B.sh 的 ROLLOUT_ARGS），reward 只用 `reward_model.ground_truth.target`
（examples/search-r1/qa_em_format.py），所以丢掉 metadata 不改变任何训练信号或
判分依据。保留 id / question / golden_answers 是因为它们两边都有，而且 dump 出来
排查问题时有用。

## 行数

    train = nq/train.jsonl 79168 + hotpotqa/train.jsonl 90447 = 169615
    test  = nq/test.jsonl   3610 + hotpotqa/dev.jsonl    7405 =  11015

顺序**不能换**：主指标是 `eval/nq_test`，而 recipe 取的是
`--eval-prompt-data nq_test "$TEST_DATA@[0:3000]"`（run_qwen2.5_3B.sh 的 ROLLOUT_ARGS），
靠的就是 test.parquet 的前 3610 行全是 NQ。先 hotpotqa 后 nq 的话，指标名还叫
nq_test，评的却是别的集合，而且不会报错。

hotpotqa 的测试集取 dev.jsonl，因为上游 hotpotqa 没有 test split——合并脚本的
test / dev / train 优先级在 qa_search_test_merge.py:56-63，这里照搬。

用法：
    python3 -m ptdbench_datasets.search_r1_nq_hotpotqa --out-dir ./prepared/nq_hotpotqa_train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ._common import hf_cache_root, write_split

REPO = "RUC-NLPIR/FlashRAG_datasets"
# 钉死 revision：上游是个 91 个文件的合集仓，随时可能加数据集或改某一份。
# 判分基线是按这一份测出来的。
REVISION = "bcafb8dd07d453be3cbeeeb3f78be1841bddf92c"

# 顺序即产物里的行顺序，见模块 docstring。合并脚本的默认值是 `nq,hotpotqa`
# （prepare_data.sh:34-35 的 TRAIN_SOURCES / TEST_SOURCES）。
SOURCES = ("nq", "hotpotqa")

# 上游 README 的 configs 块把每个 config 映射到这些 jsonl，所以直接取文件和走
# load_dataset(REPO, config) 读到的是同一批字节。
FILES = {
    "train": {"nq": "nq/train.jsonl", "hotpotqa": "hotpotqa/train.jsonl"},
    "test": {"nq": "nq/test.jsonl", "hotpotqa": "hotpotqa/dev.jsonl"},
}

# 每个来源、每个 split 的行数。对不上就是上游变了，必须停下来——
# required_final_score 是按这批数据测出来的。
EXPECTED = {
    "train": {"nq": 79168, "hotpotqa": 90447},
    "test": {"nq": 3610, "hotpotqa": 7405},
}

# 缓存键的一部分（见 docker/ptdbench_dataset.py）。**改了配方就要改这个数**：
# prompt 模板、列的取舍、行顺序，任何一处变了，上一次缓存下来的 parquet 就不再是
# 这份配方的产物，而先跑的那个任务会一直拿到旧字节。
RECIPE_VERSION = "1"


def make_prefix(question: str) -> str:
    """逐字抄自 qa_search_{train,test}_merge.py 的 make_prefix(template_type='base')。

    模板里的每一个空格都进了历史运行的 prompt，而 reward 的 format 状态机
    （qa_em_format.py）认的是 <think>/<search>/<information>/<answer> 这套标签，
    所以这段文本是数据的一部分，不是注释。
    """
    return (
        "Answer the given question. "
        "You must conduct reasoning inside <think> and </think> first every time you get new information. "
        "After reasoning, if you find you lack some knowledge, you can call a search engine by "
        "<search> query </search> and it will return the top searched results between <information> "
        "and </information>. "
        "You can search as many times as your want. "
        "If you find no further external knowledge needed, you can directly provide the answer inside "
        "<answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. "
        f"Question: {question}\n"
    )


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_rows(source: str, split_name: str, raw: list[dict]) -> list[dict]:
    """把一个来源的原始行转成 verl 风格的行。

    `index` 按**来源内**从 0 计数，不是合并后的全局序号：合并脚本是先各自
    `.map(with_indices=True)` 再 concatenate 的，所以 hotpotqa 的 index 从 0 重来。
    照抄这个行为，不是它更合理，而是产物要和那两个脚本一致。
    """
    rows = []
    for idx, item in enumerate(raw):
        question = item["question"].strip()
        if not question:
            raise SystemExit(f"{source}/{split_name} 第 {idx} 行的 question 是空的")
        # 合并脚本是就地改 example['question'] 再拼模板的，改过的问题也会留在产物里。
        if question[-1] != "?":
            question += "?"
        golden = item.get("golden_answers")
        if golden is None:
            raise SystemExit(
                f"{source}/{split_name} 第 {idx} 行没有 golden_answers，"
                f"reward_model.ground_truth.target 取的就是这个键"
            )
        rows.append(
            {
                "id": item.get("id"),
                "question": question,
                "golden_answers": golden,
                "data_source": source,
                "prompt": [{"role": "user", "content": make_prefix(question)}],
                "ability": "fact-reasoning",
                "reward_model": {"style": "rule", "ground_truth": {"target": golden}},
                "extra_info": {"split": split_name, "index": idx},
            }
        )
    return rows


def build(out_dir: Path, *, allow_count_mismatch: bool = False) -> list[Path]:
    """下载上游 jsonl、重建两个 parquet；返回写出的路径。

    和 main() 分开的理由同 dapo_math_17k.build()：容器里的 `ptdbench_dataset`
    要在缓存目录里调它，然后把产物硬链到任务要的位置。
    """
    # 在 import 之前：见 _common.hf_cache_root()。jsonl 本体靠下面的 local_dir
    # 落在宿主机上，这里管的是 huggingface_hub 自己的元数据/etag 缓存，不设就写进
    # 容器可写层。
    hf_cache_root()
    from huggingface_hub import hf_hub_download

    # 原始 jsonl 下到 out_dir 底下再删掉：`local_dir` 让 hf_hub_download 直接写成
    # 普通文件，不在别处再留一份 blob 缓存。缓存条目是靠改名生效的，所以中途失败
    # 留下的这棵树会被 ptdbench_dataset 整个丢掉。
    raw_dir = out_dir / "_raw"
    written = []
    for split_name in ("train", "test"):
        rows: list[dict] = []
        for source in SOURCES:
            filename = FILES[split_name][source]
            print(f"下载 {REPO}@{REVISION[:12]} {filename} ...")
            path = Path(
                hf_hub_download(
                    repo_id=REPO,
                    filename=filename,
                    repo_type="dataset",
                    revision=REVISION,
                    local_dir=str(raw_dir),
                )
            )
            raw = read_jsonl(path)
            want = EXPECTED[split_name][source]
            if len(raw) != want:
                message = f"{filename} 有 {len(raw)} 行，配方记的是 {want} 行"
                if not allow_count_mismatch:
                    raise SystemExit(f"{message}。上游变了，确认后加 --allow-count-mismatch。")
                print(f"!! {message}（--allow-count-mismatch 已开，继续）")
            rows.extend(build_rows(source, split_name, raw))
            # 读完就删：hotpotqa/train.jsonl 一个文件 569MB，而缓存条目里留着它
            # 等于每个 run 的 workspace 都要多搬一份没人读的字节。
            path.unlink()
        path = write_split(rows, out_dir, split_name)
        print(f"  {split_name}: {len(rows)} 行 -> {path}")
        written.append(path)

    # `_raw` 下面只剩 hf_hub_download 的记账目录（.cache/huggingface）。留在这里
    # 会跟着产物一起被硬链进 workspace。
    if raw_dir.exists():
        import shutil

        shutil.rmtree(raw_dir)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--allow-count-mismatch", action="store_true")
    args = ap.parse_args()

    build(args.out_dir, allow_count_mismatch=args.allow_count_mismatch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
