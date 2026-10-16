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

"""Boxed math reward compatible with verl's custom reward API."""

from verl.utils.reward_score.ttrl_math import compute_score as compute_math_score


def reward_func(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Grade the last boxed answer with ttrl_math's math-equivalence rules."""
    ground_truths = ground_truth if isinstance(ground_truth, (list, tuple)) else [ground_truth]
    if not ground_truths:
        return {
            "score": 0.0,
            "format_score": 0.0,
            "acc": False,
            "extracted_gt": ground_truth,
            "pred": "",
        }

    results = [compute_math_score(solution_str, str(answer)) for answer in ground_truths]
    return max(results, key=lambda result: float(result["score"]))
