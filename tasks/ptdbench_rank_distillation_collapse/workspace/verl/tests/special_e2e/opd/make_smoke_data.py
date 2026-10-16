# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Build a tiny GSM8K-shaped parquet pair for the colocated-OPD smoke test.

Same schema as ``examples/data_preprocess/gsm8k.py`` (data_source / prompt / ability /
reward_model / extra_info) so ``verl.trainer.main_ppo`` and the gsm8k rule reward work
unchanged. The arithmetic is generated locally -- the smoke test only needs the pipeline
to run, not a meaningful learning signal.
"""

import argparse
import os

import datasets

INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'
DATA_SOURCE = "openai/gsm8k"


def _rows(split: str, n: int, seed: int) -> list[dict]:
    rng = __import__("random").Random(seed)
    rows = []
    for idx in range(n):
        a, b, c = rng.randint(2, 30), rng.randint(2, 30), rng.randint(2, 9)
        answer = a * c + b
        question = (
            f"A shop sells notebooks in packs of {a}. Maya buys {c} packs and then finds "
            f"{b} more notebooks in her bag. How many notebooks does she have in total?"
        )
        rows.append(
            {
                "data_source": DATA_SOURCE,
                "prompt": [{"role": "user", "content": question + " " + INSTRUCTION}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": str(answer)},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": f"{a} * {c} = {a * c}\n{a * c} + {b} = {answer}\n#### {answer}",
                    "question": question,
                },
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir", default=os.path.expanduser("~/data/gsm8k_smoke"))
    parser.add_argument("--n_train", type=int, default=256)
    parser.add_argument("--n_test", type=int, default=32)
    args = parser.parse_args()

    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)
    for split, n, seed in (("train", args.n_train, 0), ("test", args.n_test, 1)):
        path = os.path.join(save_dir, f"{split}.parquet")
        datasets.Dataset.from_list(_rows(split, n, seed)).to_parquet(path)
        print(f"wrote {n} rows to {path}")


if __name__ == "__main__":
    main()
