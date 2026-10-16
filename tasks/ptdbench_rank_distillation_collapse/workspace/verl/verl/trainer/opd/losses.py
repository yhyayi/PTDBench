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

"""Actor loss for colocated on-policy distillation.

Two additive terms:

* the standard 2D PPO term on the sampled tokens, driven by ``advantages``
  (outcome-level: zeros for pure distillation, GRPO advantage for the
  ``opd_token_kl_plus_grpo`` estimator). Delegated to :func:`ppo_loss`, so
  entropy bonus / KL loss / clip metrics behave exactly as in plain PPO.
* the OPD candidate term, a clipped PPO objective over the ``W`` candidate ids
  per position with the per-candidate advantages built by
  :func:`verl.trainer.opd.advantage.opd_advantage`. Summing over ``W`` collapses
  it to a per-token loss, so every ``loss_agg_mode`` keeps working unchanged.

When ``opd.top_k == 0`` there is no candidate term: the single candidate is the
sampled token, so the 2D term already *is* the OPD gradient and the driver folds
the KL reward straight into ``advantages``. When ``is_norm`` is enabled for a
top-k set, both sides of the PPO ratio are conditioned on that fixed candidate
set, so the ratio and its score gradient are for ``p(. | C)`` rather than the
full-vocabulary ``p``. When ``outside_bucket`` is enabled, a second scalar PPO
term represents the aggregate probability of every token outside ``C``.
"""

import torch
from tensordict import TensorDict

from verl.trainer.opd.advantage import (
    BUCKET_DEN_SUFFIX,
    BUCKET_NUM_SUFFIX,
    OPD_ACTOR_BUCKET_SPAN,
    _complement_log_prob,
    _renormalize_log_probs,
    binned_mean_sums,
    bucket_mean_sums,
)
from verl.trainer.ppo.core_algos import agg_loss
from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig, OPDConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import no_padding_2_padding, response_from_nested

#: Student-entropy (nats) bin edges for the entropy-stratified metrics; the last bin is open-ended.
STUDENT_ENTROPY_EDGES = (0.0, 0.1, 0.5, 1.0, 2.0)

__all__ = ["opd_ppo_loss", "opd_full_vocab_loss", "opd_direct_topk_loss"]


def opd_direct_topk_loss(config: ActorConfig, opd_config: OPDConfig, model_output, data: TensorDict, dp_group=None):
    """Direct O3PD/JSD gradients, with response means and global-batch normalization."""
    from verl.trainer.opd.order_only import direct_topk_loss

    scores = _candidate_log_probs_to_response(model_output["opd_cand_log_probs"], data)
    mask = _to_response_width(data["response_mask"], scores.shape[1]).bool()
    # Mask before evaluating a nonlinear objective, including invalid padded values.
    scores = scores.to(torch.promote_types(scores.dtype, torch.float32)).masked_fill(~mask[..., None], 0)
    teacher = None
    if opd_config.objective == "topk_jsd" or opd_config.mask_teacher_ties:
        teacher = _to_response_width(data["opd_teacher_topk_log_probs"], scores.shape[1])
        teacher = teacher.masked_fill(~mask[..., None], 0)
    per_token = direct_topk_loss(scores, opd_config.objective, teacher, opd_config.mask_teacher_ties)
    # The engine counts real (nonempty) responses globally before splitting microbatches.
    batch_size = tu.get_non_tensor_data(data, "opd_valid_batch_size", default=None)
    dp_size = tu.get_non_tensor_data(data, "dp_size", default=1)
    loss = agg_loss(
        per_token, mask, "seq-mean-token-mean", dp_size=dp_size, global_batch_size=batch_size
    )
    metrics = {}
    with torch.no_grad():
        delta = scores[..., :-1] - scores[..., 1:]
        i, j = torch.triu_indices(scores.shape[-1], scores.shape[-1], offset=1, device=scores.device)
        diagnostics = {
            "actor/opd_adjacent_accuracy": (delta > 0).float().mean(-1),
            "actor/opd_pair_inversion_rate": (scores[..., i] < scores[..., j]).float().mean(-1),
            "actor/opd_candidate_top1_agreement": (scores.argmax(-1) == 0).float(),
            "actor/opd_margin_mean": delta.mean(-1),
            "actor/opd_student_mass": scores.exp().sum(-1),
        }
        lengths = mask.sum(-1)
        seq_loss = (per_token * mask).sum(-1) / lengths.clamp_min(1)
        metrics["actor/opd_direct_loss" + BUCKET_NUM_SUFFIX] = Metric(
            value=seq_loss.sum().item(), aggregation=AggregationType.SUM
        )
        metrics["actor/opd_direct_loss" + BUCKET_DEN_SUFFIX] = Metric(
            value=(lengths > 0).sum().item(), aggregation=AggregationType.SUM
        )
        # SUM num/den across microbatches and ranks; the driver resolves weighted means.
        for name, value in diagnostics.items():
            metrics[name + BUCKET_NUM_SUFFIX] = Metric(
                value=(value * mask).sum().item(), aggregation=AggregationType.SUM
            )
            metrics[name + BUCKET_DEN_SUFFIX] = Metric(value=mask.sum().item(), aggregation=AggregationType.SUM)
    return opd_config.loss_coef * loss, metrics


def _opd_candidate_loss(
    config: ActorConfig,
    opd_config: OPDConfig,
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    cand_mask: torch.Tensor,
    response_mask: torch.Tensor,
    outside_old_log_probs: torch.Tensor | None = None,
    outside_advantages: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Clipped PPO objective over the ``W`` candidates, collapsed to ``(B, T)``.

    Args:
        log_probs: ``(B, T, W)`` current-policy log-probs at the candidate ids.
        old_log_probs: ``(B, T, W)`` same, under the policy that built the batch.
        advantages: ``(B, T, W)`` per-candidate advantage.
        cand_mask: ``(B, T, W)`` valid candidate slots.
        response_mask: ``(B, T)`` valid positions.
        outside_old_log_probs: ``(B, T)`` old-policy log-probability of the
            aggregate vocabulary complement.
        outside_advantages: ``(B, T)`` advantage of that aggregate outcome.
    """
    clip_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio

    cand_mask_bool = cand_mask.bool()
    cand_mask = cand_mask_bool.to(log_probs.dtype)
    use_outside_bucket = outside_old_log_probs is not None or outside_advantages is not None
    if use_outside_bucket and (outside_old_log_probs is None or outside_advantages is None):
        raise ValueError("outside_old_log_probs and outside_advantages must be provided together")
    outside_log_probs = _complement_log_prob(log_probs, cand_mask_bool) if use_outside_bucket else None

    if opd_config is not None and opd_config.top_k > 0 and opd_config.is_norm:
        # The driver-side advantage is built from p_old(.|C) and the teacher's
        # p_tch(.|C). Condition the actor's current and old log-probs on the
        # same fixed candidate set so the PPO ratio and its gradient use the
        # candidate distribution as well. The old side is data, hence detached
        # in practice; the current side remains differentiable through the
        # masked log-sum-exp.
        log_probs = _renormalize_log_probs(log_probs, cand_mask_bool)
        old_log_probs = _renormalize_log_probs(old_log_probs, cand_mask_bool)

    negative_approx_kl = torch.clamp(log_probs - old_log_probs, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_low, 1 + clip_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # Sum rather than average over candidates: each advantage already includes its
    # probability (or configured uniform) weight.
    loss_mat = (pg_losses * cand_mask).sum(dim=-1)

    num_cand = cand_mask.sum().clamp_min(1.0)
    metrics = {
        "actor/opd_clipfrac": (torch.gt(pg_losses2, pg_losses1).to(log_probs.dtype) * cand_mask).sum() / num_cand,
        "actor/opd_ppo_kl": (-negative_approx_kl * cand_mask).sum() / num_cand,
    }

    if use_outside_bucket:
        outside_old_log_probs = outside_old_log_probs.to(outside_log_probs.dtype)
        outside_advantages = outside_advantages.to(outside_log_probs.dtype)
        outside_negative_approx_kl = torch.clamp(
            outside_log_probs - outside_old_log_probs,
            min=-20.0,
            max=20.0,
        )
        outside_ratio = torch.exp(outside_negative_approx_kl)
        outside_pg_losses1 = -outside_advantages * outside_ratio
        outside_pg_losses2 = -outside_advantages * torch.clamp(outside_ratio, 1 - clip_low, 1 + clip_high)
        outside_pg_losses = torch.maximum(outside_pg_losses1, outside_pg_losses2)
        outside_mask = response_mask.to(outside_log_probs.dtype)
        loss_mat = loss_mat + outside_pg_losses * outside_mask

        num_outside = outside_mask.sum().clamp_min(1.0)
        metrics.update(
            {
                "actor/opd_outside_clipfrac": (
                    torch.gt(outside_pg_losses2, outside_pg_losses1).to(outside_log_probs.dtype) * outside_mask
                ).sum()
                / num_outside,
                "actor/opd_outside_ppo_kl": (-outside_negative_approx_kl * outside_mask).sum() / num_outside,
            }
        )
    loss = agg_loss(
        loss_mat=loss_mat,
        loss_mask=response_mask,
        loss_agg_mode=config.loss_agg_mode,
        **config.global_batch_info,
    )
    return loss, metrics


def _candidate_log_probs_to_response(model_output: torch.Tensor, data: TensorDict) -> torch.Tensor:
    """Restore candidate log-probs in the response frame for both trainer data paths."""
    response_mask = data["response_mask"]
    if response_mask.is_nested:
        # V1 may carry a bucket-level max_response_len that is wider than this
        # mini-batch. Jagged offsets are the authoritative response frame.
        return response_from_nested(model_output, response_mask).to_padded_tensor(0)
    return no_padding_2_padding(model_output, data)


def _to_response_width(tensor: torch.Tensor, target_length: int, padding=0) -> torch.Tensor:
    """Materialize a response tensor and discard TransferQueue bucket padding."""
    tensor = tensor.to_padded_tensor(padding) if tensor.is_nested else tensor
    if tensor.shape[1] < target_length:
        raise ValueError(
            f"OPD response tensor is shorter than candidate log-probs: {tensor.shape[1]} < {target_length}"
        )
    return tensor[:, :target_length]


def opd_ppo_loss(config: ActorConfig, opd_config: OPDConfig, model_output, data: TensorDict, dp_group=None):
    """PPO loss plus the OPD candidate term. Signature matches :func:`ppo_loss`."""
    policy_loss, metrics = ppo_loss(config=config, model_output=model_output, data=data, dp_group=dp_group)

    if opd_config.top_k == 0:
        # Sampled-token OPD: fully covered by the 2D term above.
        return policy_loss, metrics

    cand_log_probs = _candidate_log_probs_to_response(model_output["opd_cand_log_probs"], data)
    response_length = cand_log_probs.shape[1]
    response_mask = _to_response_width(data["response_mask"], response_length).bool()

    metric_aggregation = (
        AggregationType.SUM
        if (
            data["dp_size"] > 1
            or data["batch_num_tokens"] is not None
            or data["global_batch_size"] is not None
            or config.loss_scale_factor is not None
        )
        else AggregationType.MEAN
    )

    opd_loss, opd_metrics = _opd_candidate_loss(
        config=config,
        opd_config=opd_config,
        log_probs=cand_log_probs,
        old_log_probs=_to_response_width(data["opd_old_log_probs"], response_length).to(cand_log_probs.dtype),
        advantages=_to_response_width(data["opd_advantages"], response_length).to(cand_log_probs.dtype),
        cand_mask=_to_response_width(data["opd_cand_mask"], response_length),
        response_mask=response_mask,
        outside_old_log_probs=(
            _to_response_width(data["opd_outside_old_log_probs"], response_length)
            if getattr(opd_config, "outside_bucket", False)
            else None
        ),
        outside_advantages=(
            _to_response_width(data["opd_outside_advantages"], response_length)
            if getattr(opd_config, "outside_bucket", False)
            else None
        ),
    )

    policy_loss = policy_loss + opd_loss
    metrics.update(Metric.from_dict(opd_metrics, aggregation=AggregationType.MEAN))
    metrics["actor/opd_loss"] = Metric(value=opd_loss, aggregation=metric_aggregation)
    return policy_loss, metrics


def _full_vocab_diff_metrics(
    diag: torch.Tensor, log_probs: torch.Tensor, mask: torch.Tensor, num_tokens: torch.Tensor, scorers: dict[str, float]
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Full-vocabulary Direct-OPD diagnostics, as (mean, max) metric dicts plus per-token tensors.

    ``scorers`` maps the target's roles to their coefficients, in diag column order. With
    ``log q = log S0 + (sum_k w_k log T_k - log T0) / beta - log Z`` (one ``T_k = T+`` with
    ``w = 1`` unless several teachers form a consensus): the teacher shift is
    ``sum_k w_k log T_k - log T0``, the student shift ``log p_theta - log S0`` and the target
    shift ``teacher shift / beta - log Z``, so ``opd_sampled_k1`` is exactly student shift minus
    target shift. ``opd_absorption`` is the least-squares slope of the student shift on the
    target shift through the origin: 0 at S0, 1 once the student carries the whole target shift.
    ``opd_teacher_gain`` is ``KL(p_theta || T0) - sum_k w_k KL(p_theta || T_k)``, the
    full-vocabulary expectation of the teacher shift under the student. Top-1 agreements compare
    argmax ids: ``target_ref`` is how often the target keeps S0's greedy token, and
    ``<teacher>_teacher_ref`` how often RL left T0's greedy token unchanged.
    A consensus also reports each teacher's own shift and, for every pair, the Pearson
    correlation and sign agreement of their shifts.
    """
    from verl.trainer.opd.full_vocab import DIAG_LOG_Z, DIAG_TEACHER_ENTROPY, diag_layout

    layout = diag_layout(len(scorers))
    roles = list(scorers)

    def column(block: str, role: str) -> torch.Tensor:
        return diag[..., layout[block].start + roles.index(role)]

    inv_beta = -scorers["teacher_ref"]
    teachers = [role for role in scorers if role not in ("teacher_ref", "ref")]
    weights = {role: scorers[role] / inv_beta for role in teachers}
    shifts = {role: column("scorer_log_prob", role) - column("scorer_log_prob", "teacher_ref") for role in teachers}
    teacher_shift = sum(weights[role] * shifts[role] for role in teachers)
    log_z = diag[..., DIAG_LOG_Z]
    student_shift = log_probs - column("scorer_log_prob", "ref")
    target_shift = inv_beta * teacher_shift - log_z
    teacher_gain = column("scorer_kl", "teacher_ref") - sum(weights[r] * column("scorer_kl", r) for r in teachers)
    teacher_entropy = sum(weights[r] * column("scorer_entropy", r) for r in teachers)

    def token_mean(x: torch.Tensor) -> torch.Tensor:
        return (x * mask).sum() / num_tokens

    def agree(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return token_mean((a == b).to(mask.dtype))

    log_z_mean = token_mean(log_z)
    teacher_shift_mean = token_mean(teacher_shift)
    target_argmax = diag[..., layout["target_argmax"]]
    ref_argmax = column("scorer_argmax", "ref")
    means = {
        "actor/opd_log_z_mean": log_z_mean,
        "actor/opd_log_z_std": token_mean((log_z - log_z_mean).square()).sqrt(),
        "actor/opd_teacher_shift_mean": teacher_shift_mean,
        "actor/opd_teacher_shift_abs_mean": token_mean(teacher_shift.abs()),
        "actor/opd_teacher_shift_std": token_mean((teacher_shift - teacher_shift_mean).square()).sqrt(),
        "actor/opd_teacher_shift_pos_frac": token_mean((teacher_shift > 0).to(mask.dtype)),
        "actor/opd_student_shift_mean": token_mean(student_shift),
        "actor/opd_student_shift_abs_mean": token_mean(student_shift.abs()),
        "actor/opd_target_shift_mean": token_mean(target_shift),
        "actor/opd_target_shift_abs_mean": token_mean(target_shift.abs()),
        "actor/opd_residual_abs_mean": token_mean((student_shift - target_shift).abs()),
        "actor/opd_shift_sign_agree": agree(student_shift > 0, target_shift > 0),
        "actor/opd_absorption": (student_shift * target_shift * mask).sum()
        / (target_shift.square() * mask).sum().clamp_min(1e-8),
        "actor/opd_teacher_gain": token_mean(teacher_gain),
        # reduce_metrics takes the min of any key containing "min", so no "minus" here
        "actor/opd_target_entropy_delta_ref": token_mean(
            diag[..., DIAG_TEACHER_ENTROPY] - column("scorer_entropy", "ref")
        ),
        "actor/opd_teacher_entropy_delta_ref": token_mean(teacher_entropy - column("scorer_entropy", "teacher_ref")),
        "actor/opd_top1_agree_student_ref": agree(diag[..., layout["student_argmax"]], ref_argmax),
        "actor/opd_top1_agree_target_ref": agree(target_argmax, ref_argmax),
    }
    for role in teachers:
        means[f"actor/opd_top1_agree_{role}_teacher_ref"] = agree(
            column("scorer_argmax", role), column("scorer_argmax", "teacher_ref")
        )
    per_token = {
        "actor/opd_teacher_shift": teacher_shift,
        "actor/opd_teacher_shift_pos_frac": (teacher_shift > 0).to(mask.dtype),
        "actor/opd_student_shift": student_shift,
        "actor/opd_target_shift": target_shift,
        "actor/opd_teacher_gain": teacher_gain,
        "actor/opd_log_z": log_z,
    }
    if len(teachers) > 1:
        names = {role: role.removeprefix("teacher_") for role in teachers}
        centered = {}
        for role, shift in shifts.items():
            means[f"actor/opd_teacher_shift_mean/{names[role]}"] = token_mean(shift)
            means[f"actor/opd_teacher_shift_abs_mean/{names[role]}"] = token_mean(shift.abs())
            centered[role] = shift - token_mean(shift)
        for i, a in enumerate(teachers):
            for b in teachers[i + 1 :]:
                pair = f"{names[a]}-{names[b]}"
                var = token_mean(centered[a].square()) * token_mean(centered[b].square())
                means[f"actor/opd_teacher_shift_corr/{pair}"] = token_mean(
                    centered[a] * centered[b]
                ) / var.sqrt().clamp_min(1e-12)
                agree = (shifts[a] > 0) == (shifts[b] > 0)
                means[f"actor/opd_teacher_shift_sign_agree/{pair}"] = token_mean(agree.to(log_z.dtype))
    maxes = {
        "actor/opd_log_z_abs_max": (log_z.abs() * mask).max(),
        "actor/opd_teacher_shift_abs_max": (teacher_shift.abs() * mask).max(),
    }
    return means, maxes, per_token


def opd_full_vocab_loss(config: ActorConfig, opd_config: OPDConfig, model_output, data: TensorDict, dp_group=None):
    """PPO loss plus the full-vocabulary reverse KL. Signature matches :func:`ppo_loss`.

    ``model_output["opd_rkl"]`` is the exact per-token ``KL(p_theta || q)`` computed by the
    student's fused forward (``verl/trainer/opd/full_vocab.py``); its gradient is the analytic
    reverse-KL gradient, so it is aggregated like any other per-token loss rather than routed
    through a PPO ratio. The 2D PPO term carries only the outcome advantage (zero for pure
    distillation, GRPO for ``opd_token_kl_plus_grpo``).
    """
    from verl.trainer.opd.full_vocab import (
        DIAG_TEACHER_ENTROPY,
        DIAG_TEACHER_LOG_PROB,
        diag_layout,
        opd_full_vocab_scorers,
    )

    policy_loss, metrics = ppo_loss(config=config, model_output=model_output, data=data, dp_group=dp_group)

    rkl = _candidate_log_probs_to_response(model_output["opd_rkl"], data)
    response_length = rkl.shape[1]
    response_mask = _to_response_width(data["response_mask"], response_length).bool()

    full_vocab_loss = agg_loss(
        loss_mat=rkl,
        loss_mask=response_mask,
        loss_agg_mode=config.loss_agg_mode,
        **config.global_batch_info,
    )
    policy_loss = policy_loss + opd_config.full_vocab.loss_coef * full_vocab_loss

    with torch.no_grad():
        diag = _candidate_log_probs_to_response(model_output["opd_diag"], data)
        log_probs = _candidate_log_probs_to_response(model_output["log_probs"], data).detach()
        mask = response_mask.to(rkl.dtype)
        num_tokens = mask.sum().clamp_min(1.0)
        scorers = opd_full_vocab_scorers(opd_config)
        layout = diag_layout(len(scorers))

        def token_mean(x: torch.Tensor) -> torch.Tensor:
            return (x * mask).sum() / num_tokens

        # Under the diff target the "teacher" distribution is the combined q, not T+.
        entropy_key = "actor/opd_target_entropy" if opd_config.full_vocab_diff else "actor/opd_teacher_entropy"
        rkl = rkl.detach()
        student_entropy = diag[..., layout["student_entropy"]]
        top1_student_target = (diag[..., layout["student_argmax"]] == diag[..., layout["target_argmax"]]).to(mask.dtype)
        opd_metrics = {
            "actor/opd_full_rkl": token_mean(rkl),
            entropy_key: token_mean(diag[..., DIAG_TEACHER_ENTROPY]),
            # the sampled-token estimate of the same KL, for comparison with sampled OPD
            "actor/opd_sampled_k1": token_mean(log_probs - diag[..., DIAG_TEACHER_LOG_PROB]),
            "actor/opd_student_entropy": token_mean(student_entropy),
            "actor/opd_top1_agree_student_target": token_mean(top1_student_target),
            "actor/opd_sampled_is_target_top1": token_mean(
                (diag[..., layout["label"]] == diag[..., layout["target_argmax"]]).to(mask.dtype)
            ),
        }
        per_token = {
            "actor/opd_full_rkl": rkl,
            "actor/opd_student_entropy": student_entropy,
            entropy_key: diag[..., DIAG_TEACHER_ENTROPY],
            "actor/opd_top1_agree_student_target": top1_student_target,
        }
        entropies = {}
        for i, role in enumerate(scorers):
            kl = diag[..., layout["scorer_kl"].start + i]
            entropies[role] = diag[..., layout["scorer_entropy"].start + i]
            opd_metrics[f"actor/opd_kl_to_{role}"] = token_mean(kl)
            opd_metrics[f"actor/opd_entropy_{role}"] = token_mean(entropies[role])
            per_token[f"actor/opd_kl_to_{role}"] = kl
            per_token[f"actor/opd_entropy_{role}"] = entropies[role]
        entropy_deltas = {"actor/opd_student_entropy_delta_target": student_entropy - diag[..., DIAG_TEACHER_ENTROPY]}
        if "ref" in entropies:
            # entropy drift of the student away from its initial state S0
            entropy_deltas["actor/opd_student_entropy_delta_ref"] = student_entropy - entropies["ref"]
        for key, value in entropy_deltas.items():
            opd_metrics[key] = token_mean(value)
            per_token[key] = value
        opd_metrics["actor/opd_student_entropy_std"] = token_mean(
            (student_entropy - opd_metrics["actor/opd_student_entropy"]).square()
        ).sqrt()
        max_metrics = {}
        if opd_config.full_vocab_diff:
            diff_means, max_metrics, diff_per_token = _full_vocab_diff_metrics(
                diag, log_probs, mask, num_tokens, scorers
            )
            opd_metrics.update(diff_means)
            per_token.update(diff_per_token)

        seqlen_names = (
            "actor/opd_full_rkl",
            "actor/opd_kl_to_ref",
            "actor/opd_teacher_shift",
            "actor/opd_student_shift",
            "actor/opd_teacher_gain",
        )
        span = tu.get_non_tensor_data(data, key="opd_bucket_span", default=OPD_ACTOR_BUCKET_SPAN)
        bucket_sums = bucket_mean_sums(per_token, mask, seqlen_names=seqlen_names, span=span)
        stratified = (
            "actor/opd_full_rkl",
            "actor/opd_kl_to_ref",
            "actor/opd_teacher_shift",
            "actor/opd_student_shift",
            "actor/opd_teacher_gain",
        )
        bucket_sums.update(
            binned_mean_sums(
                {key: per_token[key] for key in stratified if key in per_token},
                student_entropy,
                STUDENT_ENTROPY_EDGES,
                mask,
                "student_entropy",
                frac_key="actor/opd_token_frac",
            )
        )

        special_ids = tu.get_non_tensor_data(data, key="opd_special_token_ids", default=None)
        if special_ids:
            labels = diag[..., layout["label"]]
            ids = torch.as_tensor(list(special_ids), dtype=labels.dtype, device=labels.device)
            special = torch.isin(labels, ids).to(mask.dtype) * mask
            # emitted every micro-batch as num/den pairs, so the key set stays fixed when none occur
            bucket_sums["actor/opd_special_token_frac" + BUCKET_NUM_SUFFIX] = special.sum().item()
            bucket_sums["actor/opd_special_token_frac" + BUCKET_DEN_SUFFIX] = mask.sum().item()
            for key, value in per_token.items():
                if key.endswith(("_shift", "_rkl")) or key == "actor/opd_kl_to_ref":
                    name = "actor/opd_special_token_" + key.removeprefix("actor/opd_")
                    bucket_sums[name + BUCKET_NUM_SUFFIX] = (value * special).sum().item()
                    bucket_sums[name + BUCKET_DEN_SUFFIX] = special.sum().item()

    metric_aggregation = (
        AggregationType.SUM
        if (
            data["dp_size"] > 1
            or data["batch_num_tokens"] is not None
            or data["global_batch_size"] is not None
            or config.loss_scale_factor is not None
        )
        else AggregationType.MEAN
    )
    metrics.update(Metric.from_dict(opd_metrics, aggregation=AggregationType.MEAN))
    metrics.update(Metric.from_dict(max_metrics, aggregation=AggregationType.MAX))
    # summed over micro-batches; the driver divides each pair in resolve_bucket_means
    metrics.update(Metric.from_dict(bucket_sums, aggregation=AggregationType.SUM))
    metrics["actor/opd_full_loss"] = Metric(value=full_vocab_loss.detach(), aggregation=metric_aggregation)
    return policy_loss, metrics
