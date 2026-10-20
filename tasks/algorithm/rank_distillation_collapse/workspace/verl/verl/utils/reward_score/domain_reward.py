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

"""Mixed math/code reward for verl's custom reward API, dispatched by sample domain.

Math samples go to :mod:`boxed_match`. Code samples extract the last ```python block of the
final answer and must pass every unit test in ``ground_truth`` (prime_code's
``{"inputs", "outputs", "fn_name"}`` JSON). Both branches return the same keys, because
the reward extra info of one batch is stacked key by key.
"""

import json
import os
import re
import subprocess
import sys

from verl.utils.reward_score.boxed_match import reward_func as math_reward_func

# Validation sets carry no extra_info.domain, so their data sources are listed here.
CODE_DATA_SOURCES = {"LCBv5", "codeforces"}
CODE_DATA_SOURCE_PREFIXES = ("deepcoder_",)

# LiveCodeBench's per-test limit, in whole seconds since the runner uses signal.alarm; the
# whole run is capped at per test * tests + slack.
PER_TEST_TIMEOUT = int(os.environ.get("CODE_REWARD_PER_TEST_TIMEOUT", 6))
TOTAL_TIMEOUT_CAP = float(os.environ.get("CODE_REWARD_TOTAL_TIMEOUT_CAP", 120))

_CODE_BLOCK_RE = re.compile(r"```(?:python|py|Python)?[ \t]*\n(.*?)```", re.S)


def is_code_sample(data_source, extra_info=None) -> bool:
    domain = (extra_info or {}).get("domain")
    if domain is not None:
        return domain == "code"
    return data_source in CODE_DATA_SOURCES or str(data_source).startswith(CODE_DATA_SOURCE_PREFIXES)


def extract_code(solution_str: str) -> str | None:
    """The last fenced code block of the final answer (after ``</think>`` when the model thought)."""
    answer = solution_str.rsplit("</think>", 1)[-1]
    blocks = _CODE_BLOCK_RE.findall(answer)
    return blocks[-1] if blocks else None


def run_unit_tests(tests: dict, code: str) -> list:
    """Per-test results of prime_code's runner: ``True`` passed, ``False`` wrong, negative codes for errors.

    Runs in a fresh interpreter instead of a forked one: reward workers live inside vLLM engine
    processes, where ``os.fork`` (multiprocessing) is unsafe while filelock changes descriptor
    ownership. The cold start dominates the cost, so a code sample without a fenced block never
    gets here.
    """
    n_tests = len(tests["inputs"])
    payload = json.dumps({"tests": tests, "code": code, "timeout": PER_TEST_TIMEOUT})
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "verl.utils.reward_score.domain_reward", "--once"],
            input=payload,
            capture_output=True,
            text=True,
            timeout=min(PER_TEST_TIMEOUT * n_tests + 5, TOTAL_TIMEOUT_CAP),
        )
    except subprocess.TimeoutExpired:
        return [-1] * n_tests
    if proc.returncode != 0 or not proc.stdout.strip():
        return [-1] * n_tests
    return json.loads(proc.stdout.strip().splitlines()[-1])


def code_reward(solution_str: str, ground_truth) -> dict:
    tests = ground_truth if isinstance(ground_truth, dict) else json.loads(ground_truth)
    code = extract_code(solution_str)
    if code is None:
        return {"score": 0.0, "format_score": 0.0, "acc": False, "extracted_gt": "", "pred": ""}
    results = run_unit_tests(tests, code)
    passed = len(results) == len(tests["inputs"]) and all(r is True for r in results)
    return {"score": float(passed), "format_score": 1.0, "acc": passed, "extracted_gt": "", "pred": ""}


def reward_func(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if is_code_sample(data_source, extra_info):
        return code_reward(solution_str, ground_truth)
    return math_reward_func(data_source, solution_str, ground_truth, extra_info=extra_info, **kwargs)


if __name__ == "__main__":
    if "--once" in sys.argv:
        # One request on stdin, one result line on stdout. run_test directly: this process is the
        # isolation boundary, so the extra multiprocessing layer inside check_correctness (a
        # Manager plus a forked child, seconds of startup per call) is skipped. The parent's
        # subprocess timeout covers the hangs that layer used to catch.
        from verl.utils.reward_score.prime_code.testing_util import run_test

        request = json.loads(sys.stdin.read())
        try:
            results, _ = run_test(
                in_outs=request["tests"], test=request["code"], debug=False, timeout=request["timeout"]
            )
        except Exception:
            results = [-1] * len(request["tests"]["inputs"])
        sys.stdout.write(json.dumps(list(results)) + "\n")
