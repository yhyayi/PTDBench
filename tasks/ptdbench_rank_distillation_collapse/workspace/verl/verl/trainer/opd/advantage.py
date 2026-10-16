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

Sampled-token OPD (``top_k == 0``) can clamp its advantage
``log p_t - log p_s`` to ``[sampled_adv_clip_min, sampled_adv_clip_max]``.

Direct-OPD (``teacher_ref_model_path`` set, ``top_k == 0``) replaces the student
baseline with the pre-RL teacher: the advantage is ``log p_t - log p_t_ref`` at
the sampled token, while ``opd_kl`` keeps reporting ``log p_s - log p_t`` and
``opd_kl_to_teacher_ref`` reports ``log p_s - log p_t_ref``.

Multi-Diff OPD (``teachers`` set) keeps one shared reference and several post-RL
teachers ``t_k``. :func:`multi_diff_teacher_log_probs` folds them into one effective
teacher ``t_eff`` with ``t_eff - t_ref`` equal to the routed (``hard``) or summed
(``sum``) scaled shift, so the Direct-OPD path above runs unchanged on ``t_eff``.
"""

from collections.abc import Sequence

import torch

from verl.workers.config import OPDConfig

__all__ = [
    "binned_mean_sums",
    "bucket_mean_sums",
    "build_opd_candidates",
    "direct_opd_metrics",
    "multi_diff_metrics",
    "multi_diff_teacher_log_probs",
    "opd_advantage",
    "opd_kl_bucket_metrics",
    "opd_topk_overlap_ratio",
    "resolve_bucket_means",
    "route_teacher_log_probs",
    "token_set_shift_metrics",
]

#: Width of the KL-metric buckets along the response, in tokens.
OPD_KL_BUCKET_SIZE = 1024

#: Response span covered by the actor-side bucket metrics. Every DP rank and micro-batch must
#: report the same metric keys, so the buckets are fixed rather than sized to the micro-batch;
#: positions and lengths past the span fall into the last bucket.
OPD_ACTOR_BUCKET_SPAN = 32768
BUCKET_NUM_SUFFIX = "@num"
BUCKET_DEN_SUFFIX = "@den"


def bucket_mean_sums(
    values: dict[str, torch.Tensor],
    response_mask: torch.Tensor,
    seqlen_names: Sequence[str] | None = None,
    bucket_size: int = OPD_KL_BUCKET_SIZE,
    span: int = OPD_ACTOR_BUCKET_SPAN,
) -> dict[str, float]:
    """Numerators and denominators of bucketed means of each ``(B, T)`` tensor in ``values``.

    ``<name>_by_position/<lo>-<hi>`` averages over the tokens at response positions in
    ``[lo, hi)``; ``<name>_by_seqlen/<lo>-<hi>``, only for the names in ``seqlen_names`` (all
    when None), averages each response's own token mean over the responses whose length is in
    ``[lo, hi)``. Each key is emitted twice, with ``BUCKET_NUM_SUFFIX`` and ``BUCKET_DEN_SUFFIX``,
    for every bucket of the span, empty or not: summed over micro-batches and averaged over DP
    ranks, their ratio is the exact batch mean, which :func:`resolve_bucket_means` takes once the
    metrics are reduced.
    """
    seqlen_names = set(values if seqlen_names is None else seqlen_names)
    num_buckets = max(-(-span // bucket_size), 1)
    mask = response_mask.to(torch.float64)
    positions = torch.arange(mask.shape[-1], device=mask.device)
    pos_onehot = torch.nn.functional.one_hot((positions // bucket_size).clamp_max(num_buckets - 1), num_buckets)
    lengths = mask.sum(dim=-1)
    len_onehot = torch.nn.functional.one_hot(
        (lengths.long() // bucket_size).clamp_max(num_buckets - 1), num_buckets
    ) * (lengths > 0).unsqueeze(-1)
    pos_onehot, len_onehot = pos_onehot.to(mask.dtype), len_onehot.to(mask.dtype)

    rows = [mask.sum(dim=0) @ pos_onehot, len_onehot.sum(dim=0)]
    groups = []
    for name, value in values.items():
        masked = value.detach().to(mask.dtype) * mask
        rows.append(masked.sum(dim=0) @ pos_onehot)
        groups.append((f"{name}_by_position", 0))
        if name in seqlen_names:
            rows.append((masked.sum(dim=-1) / lengths.clamp_min(1.0)) @ len_onehot)
            groups.append((f"{name}_by_seqlen", 1))
    stacked = torch.stack(rows).tolist()
    dens, nums = stacked[:2], stacked[2:]

    metrics = {}
    for (prefix, den_row), num in zip(groups, nums, strict=True):
        for b in range(num_buckets):
            key = f"{prefix}/{b * bucket_size}-{(b + 1) * bucket_size}"
            metrics[key + BUCKET_NUM_SUFFIX] = num[b]
            metrics[key + BUCKET_DEN_SUFFIX] = dens[den_row][b]
    return metrics


def binned_mean_sums(
    values: dict[str, torch.Tensor],
    key: torch.Tensor,
    edges: Sequence[float],
    response_mask: torch.Tensor,
    bin_name: str,
    frac_key: str,
) -> dict[str, float]:
    """Numerator/denominator pairs like :func:`bucket_mean_sums`, with tokens binned by ``key``.

    Bin ``i`` holds the tokens with ``edges[i] <= key < edges[i + 1]``; the last bin is
    open-ended and tokens below ``edges[0]`` join the first. Per bin ``<lo>-<hi>``:
    ``<frac_key>_by_<bin_name>`` is its share of the tokens, ``<value>_by_<bin_name>`` the token
    mean of each value and ``<value>_abs_share_by_<bin_name>`` the bin's share of the value's
    total absolute sum, i.e. how concentrated the value is in that bin.
    """
    mask = response_mask.to(torch.float64)
    boundaries = torch.as_tensor(list(edges[1:]), dtype=torch.float64, device=mask.device)
    onehot = torch.nn.functional.one_hot(
        torch.bucketize(key.detach().to(torch.float64), boundaries, right=True), len(edges)
    ).to(mask.dtype) * mask.unsqueeze(-1)
    rows = [onehot.sum(dim=(0, 1))]
    totals = [mask.sum()]
    for value in values.values():
        value = value.detach().to(mask.dtype)
        rows += [(value.unsqueeze(-1) * onehot).sum(dim=(0, 1)), (value.abs().unsqueeze(-1) * onehot).sum(dim=(0, 1))]
        totals.append((value.abs() * mask).sum())
    stacked = torch.stack(rows).tolist()
    totals = torch.stack(totals).tolist()
    counts = stacked[0]

    labels = [f"{lo:g}-{hi:g}" for lo, hi in zip(edges, [*edges[1:], float("inf")], strict=True)]
    metrics = {}
    for b, label in enumerate(labels):
        metrics[f"{frac_key}_by_{bin_name}/{label}{BUCKET_NUM_SUFFIX}"] = counts[b]
        metrics[f"{frac_key}_by_{bin_name}/{label}{BUCKET_DEN_SUFFIX}"] = totals[0]
        for i, name in enumerate(values):
            metrics[f"{name}_by_{bin_name}/{label}{BUCKET_NUM_SUFFIX}"] = stacked[1 + 2 * i][b]
            metrics[f"{name}_by_{bin_name}/{label}{BUCKET_DEN_SUFFIX}"] = counts[b]
            metrics[f"{name}_abs_share_by_{bin_name}/{label}{BUCKET_NUM_SUFFIX}"] = stacked[2 + 2 * i][b]
            metrics[f"{name}_abs_share_by_{bin_name}/{label}{BUCKET_DEN_SUFFIX}"] = totals[1 + i]
    return metrics


def resolve_bucket_means(metrics: dict) -> dict:
    """Replace each reduced ``@num``/``@den`` pair of :func:`bucket_mean_sums` by its ratio, in place.

    Buckets with no tokens (or no responses) are dropped.
    """
    for key in [k for k in metrics if k.endswith(BUCKET_NUM_SUFFIX)]:
        base = key[: -len(BUCKET_NUM_SUFFIX)]
        num = metrics.pop(key)
        den = metrics.pop(base + BUCKET_DEN_SUFFIX)
        if den > 0:
            metrics[base] = num / den
    return metrics


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


def opd_kl_bucket_metrics(
    opd_kl: torch.Tensor,
    response_mask: torch.Tensor,
    bucket_size: int = OPD_KL_BUCKET_SIZE,
) -> dict[str, float]:
    """KL-mean diagnostics bucketed along the response, logged as two groups.

    ``opd/kl_by_position/<lo>-<hi>`` averages the token-level KL over every token
    whose position in its response falls in ``[lo, hi)``; it tracks how the KL
    evolves as generation progresses. ``opd/kl_by_seqlen/<lo>-<hi>`` averages over
    whole responses whose length falls in ``[lo, hi)``, where each response
    contributes the mean KL over its own tokens so long responses do not dominate;
    it tracks how the KL differs between short and long responses. Buckets span
    ``[0, T)`` in steps of ``bucket_size`` and empty buckets are omitted.
    """
    mask = response_mask.bool()
    weights = response_mask.to(opd_kl.dtype)
    seq_len = opd_kl.shape[-1]
    positions = torch.arange(seq_len, device=opd_kl.device).unsqueeze(0)
    lengths = weights.sum(dim=-1).detach()
    # Per-response mean KL. The clamp only guards empty responses, which the
    # `lengths > 0` check below then excludes from every bucket.
    resp_kl_mean = ((opd_kl * weights).sum(dim=-1) / lengths.clamp_min(1.0)).detach()

    metrics: dict[str, float] = {}
    for lo in range(0, seq_len, bucket_size):
        hi = lo + bucket_size
        pos_bucket = mask & (positions >= lo) & (positions < hi)
        if pos_bucket.any():
            metrics[f"opd/kl_by_position/{lo}-{hi}"] = (
                torch.masked_select(opd_kl, pos_bucket).mean().detach().item()
            )
        len_bucket = (lengths >= lo) & (lengths < hi) & (lengths > 0)
        if len_bucket.any():
            metrics[f"opd/kl_by_seqlen/{lo}-{hi}"] = resp_kl_mean[len_bucket].mean().item()
    return metrics


def direct_opd_metrics(
    opd_out: dict[str, torch.Tensor],
    response_mask: torch.Tensor,
    teacher_entropy: torch.Tensor | None = None,
    teacher_ref_entropy: torch.Tensor | None = None,
    bucket_size: int = OPD_KL_BUCKET_SIZE,
    prefix: str = "opd/",
    with_buckets: bool = True,
) -> dict[str, float]:
    """Direct-OPD diagnostics of the teacher shift ``log p_t - log p_t_ref`` at the sampled tokens.

    ``opd_out`` is an :func:`opd_advantage` result computed with ``teacher_ref_log_probs``.
    ``opd/teacher_shift_*`` are token-level statistics, except ``seq_sum_mean``, which sums
    the shift over each response and averages over responses. Along the response, buckets of
    ``bucket_size`` tokens report the token-mean shift, its positive fraction and the bucket's
    share of the per-response sum, so the ``seq_sum_by_position`` buckets add up to
    ``seq_sum_mean``. ``teacher_shift_by_seqlen`` instead buckets whole responses by length and
    averages each response's own mean shift, as ``opd/kl_by_seqlen`` does for the KL.
    The optional ``(B, T)`` entropies are the full-vocabulary entropies of
    the teacher and the reference teacher on the student's samples. ``prefix`` replaces the
    leading ``opd/`` of every key, e.g. ``opd/math/`` for one Multi-Diff OPD domain.
    """
    mask = response_mask.bool()
    shift = opd_out["opd_teacher_shift"]
    valid_shift = torch.masked_select(shift, mask)
    if valid_shift.numel() == 0:
        return {}
    masked_shift = shift * response_mask.to(shift.dtype)
    has_tokens = mask.any(dim=-1)
    metrics = {
        f"{prefix}teacher_shift_mean": valid_shift.mean().item(),
        f"{prefix}teacher_shift_std": valid_shift.std(unbiased=False).item(),
        f"{prefix}teacher_shift_abs_mean": valid_shift.abs().mean().item(),
        f"{prefix}teacher_shift_pos_frac": (valid_shift > 0).float().mean().item(),
        f"{prefix}teacher_shift_seq_sum_mean": masked_shift.sum(dim=-1)[has_tokens].mean().item(),
        f"{prefix}kl_to_teacher_ref": torch.masked_select(opd_out["opd_kl_to_teacher_ref"], mask).mean().item(),
    }

    positions = torch.arange(shift.shape[-1], device=shift.device)
    lengths = mask.sum(dim=-1)
    resp_shift_mean = masked_shift.sum(dim=-1) / lengths.clamp_min(1)
    for lo in range(0, shift.shape[-1], bucket_size) if with_buckets else ():
        hi = lo + bucket_size
        len_bucket = has_tokens & (lengths >= lo) & (lengths < hi)
        if len_bucket.any():
            metrics[f"{prefix}teacher_shift_by_seqlen/{lo}-{hi}"] = resp_shift_mean[len_bucket].mean().item()
        in_bucket = (positions >= lo) & (positions < hi)
        bucket_shift = torch.masked_select(shift, mask & in_bucket)
        if bucket_shift.numel() == 0:
            continue
        metrics[f"{prefix}teacher_shift_by_position/{lo}-{hi}"] = bucket_shift.mean().item()
        metrics[f"{prefix}teacher_shift_pos_frac_by_position/{lo}-{hi}"] = (bucket_shift > 0).float().mean().item()
        metrics[f"{prefix}teacher_shift_seq_sum_by_position/{lo}-{hi}"] = (
            (masked_shift * in_bucket).sum(dim=-1)[has_tokens].mean().item()
        )

    if teacher_entropy is not None and teacher_ref_entropy is not None:
        teacher_entropy = torch.masked_select(teacher_entropy, mask)
        teacher_ref_entropy = torch.masked_select(teacher_ref_entropy, mask)
        metrics[f"{prefix}teacher_entropy"] = teacher_entropy.mean().item()
        metrics[f"{prefix}teacher_ref_entropy"] = teacher_ref_entropy.mean().item()
        metrics[f"{prefix}teacher_entropy_minus_ref"] = (teacher_entropy - teacher_ref_entropy).mean().item()
    return metrics


def route_teacher_log_probs(teacher_log_probs: torch.Tensor, route_idx: torch.Tensor) -> torch.Tensor:
    """Pick each sample's teacher from a ``(K, B, T)`` stack: ``out[b] = teacher_log_probs[route_idx[b], b]``."""
    if teacher_log_probs.dim() != 3 or route_idx.shape != teacher_log_probs.shape[1:2]:
        raise ValueError(
            f"expected a (K, B, T) teacher stack and a (B,) route index, got {tuple(teacher_log_probs.shape)} "
            f"and {tuple(route_idx.shape)}"
        )
    index = route_idx.to(device=teacher_log_probs.device, dtype=torch.long)
    index = index.view(1, -1, 1).expand(1, *teacher_log_probs.shape[1:])
    return teacher_log_probs.gather(0, index).squeeze(0)


def multi_diff_teacher_log_probs(
    teacher_log_probs: torch.Tensor,
    teacher_ref_log_probs: torch.Tensor,
    route_idx: torch.Tensor,
    *,
    route_mode: str,
    adv_scales: Sequence[float],
) -> torch.Tensor:
    """The ``(B, T)`` effective teacher whose shift over the shared reference is the Multi-Diff OPD signal.

    ``teacher_log_probs`` is the ``(K, B, T)`` stack of post-RL teachers, ``teacher_ref_log_probs``
    the ``(B, T)`` reference and ``adv_scales`` the ``K`` per-teacher multipliers ``s_k``:

    - ``hard``: ``t_ref + s_r (t_r - t_ref)`` with ``r = route_idx[b]``; exactly ``t_r`` when ``s_r == 1``.
    - ``sum``: ``t_ref + sum_k s_k (t_k - t_ref)``.

    Passed to :func:`opd_advantage` as the teacher next to ``teacher_ref_log_probs``, the
    per-token advantage becomes that scaled shift.
    """
    if len(adv_scales) != teacher_log_probs.shape[0]:
        raise ValueError(f"need one scale per teacher, got {len(adv_scales)} for {teacher_log_probs.shape[0]}")
    scales = torch.as_tensor(adv_scales, dtype=teacher_log_probs.dtype, device=teacher_log_probs.device)
    if route_mode == "hard":
        routed = route_teacher_log_probs(teacher_log_probs, route_idx)
        scale = scales[route_idx.to(device=scales.device, dtype=torch.long)].unsqueeze(-1)
        scaled = teacher_ref_log_probs + scale * (routed - teacher_ref_log_probs)
        return torch.where(scale == 1, routed, scaled)
    if route_mode == "sum":
        shifts = teacher_log_probs - teacher_ref_log_probs.unsqueeze(0)
        return teacher_ref_log_probs + (scales.view(-1, 1, 1) * shifts).sum(dim=0)
    raise ValueError(f"unknown opd.route_mode {route_mode!r}")


def multi_diff_metrics(
    opd_out: dict[str, torch.Tensor],
    response_mask: torch.Tensor,
    route_idx: torch.Tensor,
    teacher_names: Sequence[str],
    *,
    teacher_log_probs: torch.Tensor,
    teacher_ref_log_probs: torch.Tensor,
    sample_valid: torch.Tensor | None = None,
    teacher_entropy: torch.Tensor | None = None,
    teacher_ref_entropy: torch.Tensor | None = None,
) -> dict[str, float]:
    """Per-domain Multi-Diff OPD diagnostics under ``opd/<teacher name>/``.

    A domain is the samples routed to one teacher. Besides the :func:`direct_opd_metrics` of
    the applied shift (without position buckets), each domain reports ``frac`` (its share of
    the ``sample_valid`` samples, i.e. excluding batch padding), ``response_len_mean``,
    ``kl_mean`` (``log p_s - log p_t_eff``) and ``raw_shift_abs_mean`` (``|t_k - t_ref|`` of
    its own teacher before any scale or sum). ``teacher_log_probs`` and the optional
    ``teacher_entropy`` are ``(K, B, T)`` stacks aligned with ``teacher_names``.
    """
    if sample_valid is None:
        sample_valid = torch.ones_like(route_idx, dtype=torch.bool)
    sample_valid = sample_valid.bool()
    num_valid = int(sample_valid.sum().item())
    mask = response_mask.bool()
    metrics: dict[str, float] = {}
    for k, name in enumerate(teacher_names):
        prefix = f"opd/{name}/"
        in_domain = sample_valid & (route_idx == k)
        metrics[f"{prefix}frac"] = in_domain.sum().item() / max(num_valid, 1)
        domain_mask = mask & in_domain.unsqueeze(-1)
        if not domain_mask.any():
            continue
        metrics[f"{prefix}response_len_mean"] = domain_mask.sum(dim=-1)[in_domain].float().mean().item()
        metrics[f"{prefix}kl_mean"] = torch.masked_select(opd_out["opd_kl"], domain_mask).mean().item()
        raw_shift = teacher_log_probs[k] - teacher_ref_log_probs
        metrics[f"{prefix}raw_shift_abs_mean"] = torch.masked_select(raw_shift, domain_mask).abs().mean().item()
        metrics.update(
            direct_opd_metrics(
                opd_out,
                domain_mask.to(response_mask.dtype),
                teacher_entropy[k] if teacher_entropy is not None else None,
                teacher_ref_entropy,
                prefix=prefix,
                with_buckets=False,
            )
        )
    return metrics


def token_set_shift_metrics(
    shift: torch.Tensor,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    token_ids: Sequence[int],
    prefix: str = "opd/special_token_",
) -> dict[str, float]:
    """Shift statistics on the sampled tokens whose id is in ``token_ids`` (e.g. chat-template tokens).

    Reports the fraction of response tokens in the set and the mean and mean absolute shift
    over them, to compare against the overall ``teacher_shift_*``.
    """
    mask = response_mask.bool()
    if not token_ids or not mask.any():
        return {}
    ids = torch.as_tensor(list(token_ids), dtype=responses.dtype, device=responses.device)
    in_set = torch.isin(responses, ids) & mask
    metrics = {f"{prefix}frac": (in_set.sum() / mask.sum()).item()}
    if in_set.any():
        set_shift = torch.masked_select(shift, in_set)
        metrics[f"{prefix}shift_mean"] = set_shift.mean().item()
        metrics[f"{prefix}shift_abs_mean"] = set_shift.abs().mean().item()
    return metrics


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
    t_ref_on_s_log_probs: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Assemble the per-strategy candidate set.

    All inputs are response-framed: ids/log-probs are ``(B, T, k)``, per-token
    quantities ``(B, T)``. ``s_*`` come from the student's ``compute_log_prob``
    pass, ``t_*`` from the teacher pass and ``t_ref_*`` from the Direct-OPD
    reference-teacher pass.

    Returns:
        dict with ``cand_ids`` ``(B, T, W)``, ``student_log_probs`` ``(B, T, W)``
        (the ``old`` student values, i.e. the PPO ratio denominator),
        ``teacher_log_probs`` ``(B, T, W)`` and ``cand_mask`` ``(B, T, W)`` bool,
        plus ``teacher_ref_log_probs`` ``(B, T, 1)`` for Direct-OPD.
    """
    strategy = config.strategy

    def _sampled(log_probs: torch.Tensor) -> torch.Tensor:
        return log_probs.unsqueeze(-1) if log_probs.dim() == 2 else log_probs[..., :1]

    if config.top_k == 0:
        # Sampled-token OPD: the single candidate is the rolled-out token itself.
        out = {
            "cand_ids": responses.unsqueeze(-1),
            "student_log_probs": old_log_probs.unsqueeze(-1),
            "teacher_log_probs": _sampled(t_on_s_log_probs),
            "cand_mask": torch.ones_like(responses, dtype=torch.bool).unsqueeze(-1),
        }
        if config.direct_opd:
            if t_ref_on_s_log_probs is None:
                raise ValueError("Direct-OPD requires the reference teacher's sampled-token log-probs")
            out["teacher_ref_log_probs"] = _sampled(t_ref_on_s_log_probs)
        return out

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
    teacher_ref_log_probs: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Per-candidate advantage and the token-level KL reward.

    Accepts (and ignores) ``cand_ids`` so a :func:`build_opd_candidates` result can be
    splatted straight in.

    Args:
        student_log_probs: ``(B, T, W)`` old student log-probs on the candidates.
        teacher_log_probs: ``(B, T, W)`` teacher log-probs on the same candidates.
        cand_mask: ``(B, T, W)`` bool, which candidate slots are part of the set.
        response_mask: ``(B, T)`` valid response positions.
        teacher_ref_log_probs: ``(B, T, 1)`` pre-RL teacher log-probs at the sampled
            token. Direct-OPD only; also returns ``opd_teacher_shift`` and
            ``opd_kl_to_teacher_ref`` (``log p_s - log p_t_ref``), both ``(B, T)``.

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

    kl_rewards = (-weights * torch.where(valid, kl, torch.zeros_like(kl))).sum(dim=-1) * response_mask
    adv_log_ratio = kl
    if teacher_ref_log_probs is not None:
        if config.top_k != 0:
            raise ValueError("Direct-OPD (teacher_ref_log_probs) is only implemented for top_k=0")
        adv_log_ratio = teacher_ref_log_probs.to(dtype) - teacher_log_probs
    advantages = -weights * torch.where(valid, adv_log_ratio, torch.zeros_like(adv_log_ratio))
    token_rewards = advantages.sum(dim=-1) * response_mask
    teacher_shift = token_rewards

    clip_diagnostics = {}
    if config.top_k == 0 and config.sampled_adv_clip_enabled:
        clip_min, clip_max = config.sampled_adv_clip_min, config.sampled_adv_clip_max
        clipped_low = (token_rewards < clip_min) if clip_min is not None else torch.zeros_like(valid[..., 0])
        clipped_high = (token_rewards > clip_max) if clip_max is not None else torch.zeros_like(valid[..., 0])
        token_rewards = token_rewards.clamp(min=clip_min, max=clip_max) * response_mask
        advantages = token_rewards.unsqueeze(-1)
        clip_diagnostics = {
            "opd_adv_clipped_low": clipped_low.to(dtype) * response_mask,
            "opd_adv_clipped_high": clipped_high.to(dtype) * response_mask,
        }

    num_valid = valid.sum(dim=-1)
    out = {
        "opd_advantages": advantages,
        "opd_token_rewards": token_rewards,
        "opd_cand_mask": valid,
        # Unclipped, so `opd/kl_mean` still tracks the true sampled KL.
        "opd_kl": -kl_rewards,
        **clip_diagnostics,
        # This is the number of valid reward candidates. The user-facing
        # `opd/overlap_ratio` metric compares the two model top-k id sets via
        # `opd_topk_overlap_ratio` instead.
        "opd_overlap": num_valid.float() * response_mask,
        "opd_student_mass": student_mass,
        "opd_teacher_mass": teacher_mass,
    }
    if teacher_ref_log_probs is not None:
        # Unclipped, like `opd_kl`.
        out["opd_teacher_shift"] = teacher_shift
        student_ref_log_ratio = student_log_probs - teacher_ref_log_probs.to(dtype)
        out["opd_kl_to_teacher_ref"] = (
            weights * torch.where(valid, student_ref_log_ratio, torch.zeros_like(student_ref_log_ratio))
        ).sum(dim=-1) * response_mask

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
