"""
Normal mathematical reward function used by the actor-LR collapse case.

This case intentionally keeps the reward pipeline correct so that the
observed failure can be attributed to the actor optimization setting,
rather than to reward parsing or reward-model behavior.
"""

from verl.utils.reward_score import default_compute_score


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
):
    return default_compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )
