"""openai/gsm8k -> gsm8k_verl/{train,test}.parquet（task_engine_base_025 用）

**不需要自定义 split。** 历史产物的形状是 train (7473, 5) / test (1319, 5)，正好是上游
`openai/gsm8k` config `main` 自带的 train/test 划分，所以这里直接沿用上游 split，
没有任何我们自己发明的边界。

目标 schema 取自历史 agent_trace 里真实 dump 出来的那一份（025 的 agent 跑过
`pd.read_parquet(...).head(3).to_dict('records')`）：

    data_source   'openai/gsm8k'        <- reward_score/__init__.py:127 精确匹配这个串
    prompt        [{'content': <question> + ' ' + INSTRUCTION, 'role': 'user'}]
    ability       'math'
    reward_model  {'ground_truth': '72', 'style': 'rule'}
    extra_info    {'answer': <完整解答>, 'index': i, 'question': <原始问题>, 'split': 'train'}

也就是 verl 自己 examples/data_preprocess/gsm8k.py 的输出（那个目录没被 vendored 进
任务 workspace，所以在这里重建）。

用法：
    python3 -m ptdbench_datasets.gsm8k --out-dir ./prepared/gsm8k_verl
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from ._common import hf_datasets_cache, report, write_split

REPO = "openai/gsm8k"
CONFIG = "main"
DATA_SOURCE = "openai/gsm8k"  # 必须逐字符等于这个，否则打分器路由不到 gsm8k

# verl 原版 gsm8k 预处理追加的指令后缀，和历史产物里观察到的完全一致。
INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'

# 历史产物的行数。上游 main 的官方 split 就是这个，对不上说明上游变了，必须停下来。
EXPECTED = {"train": 7473, "test": 1319}

# 缓存键的一部分；改了配方就要改这个数。理由同 dapo_math_17k.RECIPE_VERSION。
RECIPE_VERSION = "1"

_SOLUTION_RE = re.compile(r"#### (\-?[0-9\.\,]+)")


def extract_ground_truth(answer: str) -> str:
    """gsm8k 的答案末尾是 `#### <数字>`，真值就是那个数字（去掉千分位逗号）。"""
    m = _SOLUTION_RE.search(answer)
    if m is None:
        raise ValueError(f"答案里没有 '#### <数字>'：{answer[-120:]!r}")
    return m.group(1).replace(",", "")


def build_rows(split_name: str, ds) -> list[dict]:
    rows = []
    for idx, item in enumerate(ds):
        question_raw = item["question"]
        answer_raw = item["answer"]
        rows.append(
            {
                "data_source": DATA_SOURCE,
                "prompt": [{"content": f"{question_raw} {INSTRUCTION}", "role": "user"}],
                "ability": "math",
                "reward_model": {
                    "ground_truth": extract_ground_truth(answer_raw),
                    "style": "rule",
                },
                "extra_info": {
                    "answer": answer_raw,
                    "index": idx,
                    "question": question_raw,
                    "split": split_name,
                },
            }
        )
    return rows


def build(out_dir: Path, *, allow_count_mismatch: bool = False) -> list[Path]:
    """下载上游、重建 verl schema、写出两个 parquet；返回写出的路径。

    和 main() 分开的理由同 dapo_math_17k.build()：容器里的 `ptdbench_dataset`
    要在缓存目录里调它，然后把产物硬链到任务要的位置。
    """
    # 在 import 之前：见 _common.hf_cache_root()。cache_dir 不传的话上游会下到
    # ~/.cache/huggingface，那是容器可写层，宿主机 datasets/ 里什么都不会出现，
    # 而且每次重建配方都要重下一遍。
    cache_dir = hf_datasets_cache()
    from datasets import load_dataset

    print(f"下载 {REPO} (config={CONFIG}) ...")
    dataset = load_dataset(REPO, CONFIG, cache_dir=str(cache_dir))

    written = []
    for split_name in ("train", "test"):
        ds = dataset[split_name]
        want = EXPECTED[split_name]
        if len(ds) != want:
            msg = f"{split_name}: 上游 {len(ds)} 行，历史产物是 {want} 行"
            if not allow_count_mismatch:
                raise SystemExit(
                    f"{msg}。上游数据变了，grader 里的 baseline 可能不再可比。\n"
                    f"确认无误后加 --allow-count-mismatch 重跑。"
                )
            print(f"!! {msg}（--allow-count-mismatch 已开，继续）")

        rows = build_rows(split_name, ds)
        path = write_split(rows, out_dir, split_name)
        print(f"  {split_name}: {len(rows)} 行 -> {path}")
        written.append(path)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument(
        "--allow-count-mismatch",
        action="store_true",
        help="上游行数和历史产物不一致时仍然继续（默认直接失败）",
    )
    args = ap.parse_args()

    report(build(args.out_dir, allow_count_mismatch=args.allow_count_mismatch))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
