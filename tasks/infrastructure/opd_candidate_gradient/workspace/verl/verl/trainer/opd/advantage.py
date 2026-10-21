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

"""Driver-side assembly of the OPD candidate set, reward and advantage.

Everything here runs on CPU tensors of shape ``(B, T, W)``; no logits involved.

Policy-gradient signal
----------------------
For the reverse-KL policy gradient at one position, over a candidate set ``C``
with candidate normalization enabled::

    p_s^C(v) = p_s(v) / sum_{u in C} p_s(u)
    grad KL(p_s^C || p_t^C)
        = sum_{v in C} p_s^C(v) * (log p_s^C(v) - log p_t^C(v)) * grad log p_s^C(v)

which gives the per-candidate advantage
``A_v = -p_s^C(v) * (log p_s^C(v) - log p_t^C(v))``. That is what
:func:`opd_advantage` returns for ``student_p``, and the OPD actor loss applies
it to the matching ``grad log p_s^C(v)`` score. Thus, for a fixed candidate set,
``student_p`` plus ``is_norm`` gives the exact conditional reverse-KL policy
gradient at the old policy. Without candidate normalization, a strict top-k set
remains a sparse approximation to the full-vocabulary reverse-KL gradient.
When ``outside_bucket`` is enabled, the complement of the candidate set is added
as one aggregate outcome. With ``student_p`` this gives the exact reverse-KL
gradient after coarsening the vocabulary into the selected ids plus that bucket.
``is_norm`` only applies to ``top_k > 0`` and does not alter sampled-token OPD.
The token-level reward is ``sum_v A_v``, i.e. the negative weighted log-ratio at
that position.

For student_p/teacher_p, w_v is the corresponding raw full-vocabulary
probability when normalization is disabled, and the candidate-renormalized
probability when it is enabled. none uses 1 / |C|.
"""

import torch

from verl.workers.config import OPDConfig

__all__ = ["build_opd_candidates", "opd_advantage", "opd_topk_overlap_ratio"]


def _renormalize_log_probs(log_probs: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Renormalize log-probabilities over valid candidate slots per position."""
    masked = log_probs.masked_fill(~valid, -torch.inf)
    log_norm = torch.logsumexp(masked, dim=-1, keepdim=True)
    # An empty intersection has no candidate distribution. Keep it finite and
    # let the validity mask below make the resulting advantage exactly zero.
    log_norm = torch.where(torch.isfinite(log_norm), log_norm, torch.zeros_like(log_norm))
    return torch.where(valid, log_probs - log_norm, torch.zeros_like(log_probs))


def _complement_log_prob(log_probs: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Return ``log(1 - sum(exp(log_probs[valid])))`` stably.

    The upper clamp only handles floating-point sums that round to one (or just
    above it). For a true full-vocabulary candidate set both models receive the
    same epsilon-sized bucket, so its KL contribution and gradient are zero.
    """
    log_inside_mass = torch.logsumexp(log_probs.masked_fill(~valid, -torch.inf), dim=-1)
    eps = torch.finfo(log_probs.dtype).eps
    log_inside_mass = log_inside_mass.clamp_max(torch.log(log_inside_mass.new_tensor(1.0 - eps)))
    log_half = -torch.log(log_inside_mass.new_tensor(2.0))
    return torch.where(
        log_inside_mass < log_half,
        torch.log1p(-torch.exp(log_inside_mass)),
        torch.log(-torch.expm1(log_inside_mass)),
    )


def _isin_last_dim(elements: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    """Exact batched membership without materializing an ``(..., K, K)`` comparison.

    ``torch.topk`` always returns unique token ids, but equal logits at its boundary
    make probability-threshold membership ambiguous. Sorting ids and using
    ``searchsorted`` keeps intersection/union exact with ``O(K)`` auxiliary memory.
    """
    if elements.shape[:-1] != candidates.shape[:-1]:
        raise ValueError(
            f"membership tensors must share leading dimensions, got {elements.shape} and {candidates.shape}"
        )
    if candidates.shape[-1] == 0:
        return torch.zeros_like(elements, dtype=torch.bool)

    sorted_candidates = candidates.sort(dim=-1).values
    insertion = torch.searchsorted(sorted_candidates, elements)
    in_bounds = insertion < sorted_candidates.shape[-1]
    insertion = insertion.clamp_max(sorted_candidates.shape[-1] - 1)
    return in_bounds & (sorted_candidates.gather(-1, insertion) == elements)


def opd_topk_overlap_ratio(
    student_topk_ids: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    response_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``|student_top-k ∩ teacher top-k| / k`` per response position.

    The two top-k tensors contain token ids with shape ``(B, T, K)``.  Membership
    is compared by id rather than by score, so ties at the top-k boundary are
    handled exactly like candidate construction.  Padding positions are zeroed
    when ``response_mask`` is supplied.
    """
    if student_topk_ids.shape != teacher_topk_ids.shape:
        raise ValueError(
            "student and teacher top-k tensors must have the same shape, "
            f"got {student_topk_ids.shape} and {teacher_topk_ids.shape}"
        )
    overlap = _isin_last_dim(student_topk_ids, teacher_topk_ids).sum(dim=-1).float()
    ratio = overlap / max(student_topk_ids.shape[-1], 1)
    if response_mask is not None:
        ratio = ratio * response_mask
    return ratio


def build_opd_candidates(
    config: OPDConfig,
    *,
    responses: torch.Tensor,
    old_log_probs: torch.Tensor,
    s_topk_ids: torch.Tensor | None = None,
    s_topk_log_probs: torch.Tensor | None = None,
    s_on_t_log_probs: torch.Tensor | None = None,
    t_on_s_log_probs: torch.Tensor | None = None,
    t_topk_ids: torch.Tensor | None = None,
    t_topk_log_probs: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Assemble the per-strategy candidate set.

    All inputs are response-framed: ids/log-probs are ``(B, T, k)``, per-token
    quantities ``(B, T)``. ``s_*`` come from the student's ``compute_log_prob``
    pass, ``t_*`` from the teacher pass.

    Returns:
        dict with ``cand_ids`` ``(B, T, W)``, ``student_log_probs`` ``(B, T, W)``
        (the ``old`` student values, i.e. the PPO ratio denominator),
        ``teacher_log_probs`` ``(B, T, W)`` and ``cand_mask`` ``(B, T, W)`` bool.
    """
    strategy = config.strategy

    if config.top_k == 0:
        # Sampled-token OPD: the single candidate is the rolled-out token itself.
        return {
            "cand_ids": responses.unsqueeze(-1),
            "student_log_probs": old_log_probs.unsqueeze(-1),
            "teacher_log_probs": t_on_s_log_probs.unsqueeze(-1)
            if t_on_s_log_probs.dim() == 2
            else t_on_s_log_probs[..., :1],
            "cand_mask": torch.ones_like(responses, dtype=torch.bool).unsqueeze(-1),
        }

    if strategy == "only_stu":
        # Student's own top-k; teacher evaluated at those ids.
        return {
            "cand_ids": s_topk_ids,
            "student_log_probs": s_topk_log_probs,
            "teacher_log_probs": t_on_s_log_probs,
            "cand_mask": torch.ones_like(s_topk_ids, dtype=torch.bool),
        }

    if strategy == "only_tch":
        # Teacher's own top-k; student evaluated at those ids.
        return {
            "cand_ids": t_topk_ids,
            "student_log_probs": s_on_t_log_probs,
            "teacher_log_probs": t_topk_log_probs,
            "cand_mask": torch.ones_like(t_topk_ids, dtype=torch.bool),
        }

    if strategy == "intersection":
        # Student top-k ids that also sit inside the teacher's actual top-k id set.
        # Comparing ids matters when logits tie at the top-k boundary.
        mask = _isin_last_dim(s_topk_ids, t_topk_ids)
        return {
            "cand_ids": s_topk_ids,
            "student_log_probs": s_topk_log_probs,
            "teacher_log_probs": t_on_s_log_probs,
            "cand_mask": mask,
        }

    if strategy == "union":
        # Student top-k, then teacher top-k ids not already in the student's set
        # so each id is counted once. Use ids rather than a probability threshold:
        # the latter is wrong when several tokens tie at the k-th logit.
        t_not_in_s = ~_isin_last_dim(t_topk_ids, s_topk_ids)
        return {
            "cand_ids": torch.cat([s_topk_ids, t_topk_ids], dim=-1),
            "student_log_probs": torch.cat([s_topk_log_probs, s_on_t_log_probs], dim=-1),
            "teacher_log_probs": torch.cat([t_on_s_log_probs, t_topk_log_probs], dim=-1),
            "cand_mask": torch.cat([torch.ones_like(s_topk_ids, dtype=torch.bool), t_not_in_s], dim=-1),
        }

    raise ValueError(f"unknown opd.strategy {strategy!r}")


def opd_advantage(
    config: OPDConfig,
    *,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    cand_mask: torch.Tensor,
    response_mask: torch.Tensor,
    cand_ids: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Per-candidate advantage and the token-level KL reward.

    Accepts (and ignores) ``cand_ids`` so a :func:`build_opd_candidates` result can be
    splatted straight in.

    Args:
        student_log_probs: ``(B, T, W)`` old student log-probs on the candidates.
        teacher_log_probs: ``(B, T, W)`` teacher log-probs on the same candidates.
        cand_mask: ``(B, T, W)`` bool, which candidate slots are part of the set.
        response_mask: ``(B, T)`` valid response positions.

    Returns:
        dict with ``opd_advantages`` ``(B, T, W)``, ``opd_token_rewards`` ``(B, T)``
        (= negative expected KL, the KL-in-reward signal), ``opd_cand_mask``, plus
        ``kl``/candidate-count/``student_mass``/``teacher_mass`` diagnostics ``(B, T)``.
        With ``outside_bucket=True``, it also returns the old student outside
        log-probability and its scalar advantage for the actor loss.
    """
    # Promote to at least fp32; never downcast, so fp64 callers (tests) keep their precision.
    dtype = torch.promote_types(torch.promote_types(student_log_probs.dtype, teacher_log_probs.dtype), torch.float32)
    student_log_probs = student_log_probs.to(dtype)
    teacher_log_probs = teacher_log_probs.to(dtype)
    valid = cand_mask & response_mask.bool().unsqueeze(-1)

    # Keep full-vocabulary log-probs for diagnostics. For top-k OPD, is_norm
    # optionally changes the log-ratio and probability weights to distributions
    # truncated and renormalized on this candidate set.
    student_mass = (student_log_probs.exp() * valid).sum(dim=-1) * response_mask
    teacher_mass = (teacher_log_probs.exp() * valid).sum(dim=-1) * response_mask
    ratio_student_log_probs = student_log_probs
    ratio_teacher_log_probs = teacher_log_probs
    use_candidate_norm = config.top_k > 0 and config.is_norm
    if use_candidate_norm:
        ratio_student_log_probs = _renormalize_log_probs(student_log_probs, valid)
        ratio_teacher_log_probs = _renormalize_log_probs(teacher_log_probs, valid)

    # k1 estimator: the plain log-ratio, which is what the reverse-KL gradient needs.
    kl = ratio_student_log_probs - ratio_teacher_log_probs

    mode = config.reward_weight_mode
    if config.top_k == 0:
        # Sampled-token OPD has exactly one candidate, so its weight is always 1.
        weights = valid.to(dtype)
    elif mode == "student_p":
        weights = (ratio_student_log_probs if use_candidate_norm else student_log_probs).exp()
    elif mode == "teacher_p":
        weights = (ratio_teacher_log_probs if use_candidate_norm else teacher_log_probs).exp()
    else:
        # none means no probability preference, i.e. uniform over valid ids.
        weights = valid.to(dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
    weights = torch.where(valid, weights, torch.zeros_like(weights))

    advantages = -weights * torch.where(valid, kl, torch.zeros_like(kl))
    token_rewards = advantages.sum(dim=-1) * response_mask

    num_valid = valid.sum(dim=-1)
    out = {
        "opd_advantages": advantages,
        "opd_token_rewards": token_rewards,
        "opd_cand_mask": valid,
        "opd_kl": -token_rewards,
        # This is the number of valid reward candidates. The user-facing
        # `opd/overlap_ratio` metric compares the two model top-k id sets via
        # `opd_topk_overlap_ratio` instead.
        "opd_overlap": num_valid.float() * response_mask,
        "opd_student_mass": student_mass,
        "opd_teacher_mass": teacher_mass,
    }

    use_outside_bucket = config.top_k > 0 and not use_candidate_norm and getattr(config, "outside_bucket", False)
    if use_outside_bucket:
        student_outside_log_prob = _complement_log_prob(student_log_probs, valid)
        teacher_outside_log_prob = _complement_log_prob(teacher_log_probs, valid)
        outside_log_ratio = student_outside_log_prob - teacher_outside_log_prob
        outside_student_mass = student_outside_log_prob.exp() * response_mask
        outside_teacher_mass = teacher_outside_log_prob.exp() * response_mask
        outside_advantage = -outside_student_mass * outside_log_ratio

        token_rewards = token_rewards + outside_advantage
        out.update(
            {
                "opd_token_rewards": token_rewards,
                "opd_kl": -token_rewards,
                "opd_outside_advantages": outside_advantage,
                "opd_outside_old_log_probs": student_outside_log_prob,
                "opd_outside_kl": -outside_advantage,
                "opd_outside_student_mass": outside_student_mass,
                "opd_outside_teacher_mass": outside_teacher_mass,
            }
        )
    return out
