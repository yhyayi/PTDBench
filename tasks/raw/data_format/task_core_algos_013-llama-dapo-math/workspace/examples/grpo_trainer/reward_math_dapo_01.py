"""0/1 reward wrapper around math_dapo.compute_score (+1/-1 -> 1/0)."""

from __future__ import annotations

from typing import Any, Optional

from verl.utils.reward_score.math_dapo import compute_score as _math_dapo_compute_score


def compute_score(
    data_source: str | None = None,
    solution_str: str = "",
    ground_truth: str = "",
    extra_info: Any = None,
    *,
    strict_box_verify: bool = False,
    pause_tokens_index: Optional[list[int]] = None,
    **kwargs,
) -> dict:
    """Same extract/verify as math_dapo, but score is 1.0 correct / 0.0 incorrect."""
    del data_source, extra_info, kwargs  # unused; kept for reward-manager call signature
    out = _math_dapo_compute_score(
        solution_str=solution_str,
        ground_truth=ground_truth,
        strict_box_verify=strict_box_verify,
        pause_tokens_index=pause_tokens_index,
    )
    # math_dapo returns score in {+1.0, -1.0}; map to {1.0, 0.0}
    out = dict(out)
    out["score"] = 1.0 if out.get("score", 0.0) > 0 else 0.0
    return out
