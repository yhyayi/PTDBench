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

from concurrent.futures import ThreadPoolExecutor

import pytest

from verl.utils.reward_score.boxed_match import reward_func


@pytest.mark.parametrize(
    ("solution", "ground_truth"),
    [
        (r"The answer is \boxed{142}.", "142.0"),
        (r"The answer is \boxed{-4}.", "-4.0"),
        (r"The answer is \boxed{1/2}.", r"\frac{1}{2}"),
    ],
)
def test_reward_func_accepts_equivalent_math_answers(solution, ground_truth):
    result = reward_func("math", solution, ground_truth)

    assert result["score"] == 1.0
    assert result["format_score"] == 1.0
    assert result["acc"] is True


def test_reward_func_reports_format_and_uses_last_boxed_answer():
    wrong = reward_func("math", r"First \boxed{142}, finally \boxed{143}.", "142.0")
    missing = reward_func("math", "The answer is 142.", "142.0")

    assert wrong["score"] == 0.0
    assert wrong["format_score"] == 1.0
    assert wrong["pred"] == "143"
    assert missing["score"] == 0.0
    assert missing["format_score"] == 0.0


def test_reward_func_accepts_any_ground_truth_candidate():
    result = reward_func("math", r"The answer is \boxed{142}.", ["141", "142.0"])

    assert result["score"] == 1.0
    assert result["extracted_gt"] == "142.0"


def test_reward_func_rejects_empty_ground_truth_candidates():
    result = reward_func("math", r"The answer is \boxed{}.", [])

    assert result["score"] == 0.0
    assert result["format_score"] == 0.0


def test_reward_func_can_run_in_reward_worker_thread():
    with ThreadPoolExecutor(max_workers=1) as executor:
        correct = executor.submit(reward_func, "math", r"The answer is \boxed{x+1}.", "1+x").result(timeout=10)
        wrong = executor.submit(reward_func, "math", r"The answer is \boxed{143}.", "142.0").result(timeout=10)

    assert correct["score"] == 1.0
    assert wrong["score"] == 0.0
