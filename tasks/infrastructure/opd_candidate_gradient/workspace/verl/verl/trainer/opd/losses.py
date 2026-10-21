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

from verl.trainer.opd.advantage import _complement_log_prob, _renormalize_log_probs
from verl.trainer.ppo.core_algos import agg_loss
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig, OPDConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import no_padding_2_padding, response_from_nested

__all__ = ["opd_ppo_loss"]


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
