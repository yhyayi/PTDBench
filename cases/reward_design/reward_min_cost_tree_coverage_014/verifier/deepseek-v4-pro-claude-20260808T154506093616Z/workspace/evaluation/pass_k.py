import os
import json
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", type = str, required = True)
parser.add_argument("--eval-name", type = str, default = "LiveCodeBench-v6")
parser.add_argument("--eval-config", type = str, default = "data/BENCHMARKS/LiveCodeBench/evaluation_config.json")
args = parser.parse_args()

with open(args.eval_config, "r") as fin :
    n_samples_per_eval_prompt = json.load(fin)["n_samples_per_eval_prompt"]

with open(os.path.join(args.path, "evaluation_result/{}.json".format(args.eval_name)), "r") as fin :
    results = json.load(fin)
accuracy = results["accuracy"]
assert len(accuracy) % n_samples_per_eval_prompt == 0
num_prompts = len(accuracy) // n_samples_per_eval_prompt

for k in range(1, n_samples_per_eval_prompt + 1) :
    Sum = 0.0
    for index in range(0, len(accuracy), n_samples_per_eval_prompt) :
        assert all(x in (0, 1) for x in accuracy[index : index + n_samples_per_eval_prompt])
        passed = sum(accuracy[index : index + n_samples_per_eval_prompt])
        assert 0 <= passed <= n_samples_per_eval_prompt
        Sum += 1.0 - (math.comb(n_samples_per_eval_prompt - passed, k) / math.comb(n_samples_per_eval_prompt, k) if n_samples_per_eval_prompt - passed >= k else 0.0)
    results["Pass@{}".format(k)] = Sum / num_prompts

with open(os.path.join(args.path, "evaluation_result/{}.json".format(args.eval_name)), "w") as fout :
    json.dump(results, fout, indent = 2)