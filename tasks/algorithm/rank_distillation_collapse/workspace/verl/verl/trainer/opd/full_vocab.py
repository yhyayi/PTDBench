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

"""Full-vocabulary reverse KL for colocated on-policy distillation.

The frozen scorers never project onto the vocabulary on their own forward pass:
they export the hidden state that feeds their LM head (``[n, H]``, H << V), and
the student rebuilds the target distribution chunk by chunk inside its own
forward::

    log q   = log_softmax( sum_i coef_i * log_softmax(h_i W_i^T / tau_t) )
    rkl_t   = sum_v p_theta(v) * (log p_theta(v) - log q(v))

Plain OPD uses one scorer (the teacher, ``coef=+1``). Full-vocabulary
Direct-OPD plugs in ``[S0 (+1), T+ (+1/beta), T0 (-1/beta)]`` with no kernel change,
i.e. ``q ∝ S0 * (T+ / T0)^(1/beta)``. Several post-RL teachers ``T_k`` of the same ``T0``
replace ``T+`` by their consensus ``[T_k (+w_k/beta)]`` with ``sum_k w_k = 1``.

:class:`FullVocabRKLFunction` fuses the student's LM head with the sampled-token
log-prob, the entropy and the reverse KL. Forward and backward both stream over
row chunks and recompute every ``[chunk, V]`` buffer, so no ``[N, V]`` tensor
ever exists on either side; the gradient w.r.t. the student logits is::

    d rkl / d z = p * (log p - log q - rkl)

Doing the recompute inside one autograd function (instead of
``torch.utils.checkpoint`` around a ``lm_head`` call) keeps FSDP's view of the
LM head weight a single plain use, exactly like ``FusedLinearForPPOFunction``.
"""

from collections.abc import Mapping
from typing import Optional

import torch

__all__ = [
    "FULL_VOCAB_SCORERS",
    "FrozenScorer",
    "FullVocabTarget",
    "FullVocabRKLFunction",
    "consensus_teacher_role",
    "diag_layout",
    "full_vocab_rkl",
    "full_vocab_scorers",
    "load_lm_head_weight",
    "opd_full_vocab_scorers",
]

# Frozen scorers of each ``opd.full_vocab.target`` and their coefficient in ``log q`` at
# ``beta=1``. A role names the ``opd_hidden_<role>`` field and the staged frozen head; ``ref``
# is the frozen initial student S0 scored by the reference-policy engine. The order is the
# column order of the per-scorer diagnostics.
FULL_VOCAB_SCORERS = {
    "rkl": {"teacher": 1.0},
    "diff": {"teacher": 1.0, "teacher_ref": -1.0, "ref": 1.0},
}

def consensus_teacher_role(name: str) -> str:
    """Scorer role of the post-RL teacher ``name`` in the full-vocabulary consensus target."""
    return f"teacher_{name}"


def full_vocab_scorers(
    target: str, beta: float = 1.0, teacher_weights: Optional[Mapping[str, float]] = None
) -> dict[str, float]:
    """Scorer coefficients of ``target``; ``beta`` divides the teacher shift of ``diff``.

    ``teacher_weights`` (``diff`` only) replaces the single ``teacher`` by one
    :func:`consensus_teacher_role` per post-RL teacher, whose shifts form the consensus
    ``sum_k w_k log T_k - log T0`` with the weights normalized to sum to 1.
    """
    scorers = dict(FULL_VOCAB_SCORERS[target])
    if teacher_weights is not None:
        if target != "diff":
            raise ValueError(f"teacher_weights only applies to the diff target, got {target!r}")
        total = sum(teacher_weights.values())
        consensus = {consensus_teacher_role(name): w / total for name, w in teacher_weights.items()}
        scorers = {**consensus, "teacher_ref": scorers["teacher_ref"], "ref": scorers["ref"]}
    if target == "diff":
        for role in scorers:
            if role != "ref":
                scorers[role] /= beta
    return scorers


def opd_full_vocab_scorers(opd_config) -> dict[str, float]:
    """The full-vocabulary scorer coefficients selected by an ``OPDConfig``."""
    weights = None
    if opd_config.full_vocab_consensus:
        weights = dict(zip(opd_config.teacher_names, opd_config.consensus_teacher_weights, strict=True))
    return full_vocab_scorers(opd_config.full_vocab.target, opd_config.full_vocab.beta, teacher_weights=weights)

# Columns of the per-token diagnostics output: the target's entropy, its log-prob of the
# sampled token, its log-normalizer ``log Z`` (0 for a single unit-coefficient scorer), then
# each scorer's own log-prob of the sampled token from column DIAG_SCORER_LOG_PROB on.
DIAG_TEACHER_ENTROPY = 0
DIAG_TEACHER_LOG_PROB = 1
DIAG_LOG_Z = 2
DIAG_SCORER_LOG_PROB = 3


def diag_layout(num_scorers: int) -> dict[str, int | slice]:
    """Columns of the diagnostics output for ``num_scorers`` frozen scorers.

    After the ``DIAG_*`` columns and the scorers' sampled-token log-probs come the sampled token
    id, the student's entropy and argmax id and the target's argmax id, then, one column per
    scorer in scorer order, the full-vocabulary ``KL(p_student || scorer)``, the scorer's entropy
    and its argmax id. Ids are stored as fp32, exact for vocabularies below 2**24.
    """
    k = num_scorers
    base = DIAG_SCORER_LOG_PROB + k
    return {
        "scorer_log_prob": slice(DIAG_SCORER_LOG_PROB, base),
        "label": base,
        "student_entropy": base + 1,
        "student_argmax": base + 2,
        "target_argmax": base + 3,
        "scorer_kl": slice(base + 4, base + 4 + k),
        "scorer_entropy": slice(base + 4 + k, base + 4 + 2 * k),
        "scorer_argmax": slice(base + 4 + 2 * k, base + 4 + 3 * k),
        "width": base + 4 + 3 * k,
    }


class FrozenScorer:
    """One frozen model's contribution to the target distribution.

    A plain class rather than a dataclass or tuple: FSDP2 casts forward inputs by
    walking pytrees, and must not cast (i.e. copy) the frozen head or hidden.

    Args:
        hidden: ``(N, H)`` LM-head input of the frozen model, row-aligned with the
            student's packed logits rows. Rows outside ``row_mask`` are never read.
        head: ``(V, H)`` LM head weight of the frozen model.
        coef: weight of this model's log-probs in the target.
    """

    __slots__ = ("hidden", "head", "coef")

    def __init__(self, hidden: torch.Tensor, head: torch.Tensor, coef: float = 1.0):
        self.hidden = hidden
        self.head = head
        self.coef = float(coef)


class FullVocabTarget:
    """Everything the student forward needs to score its logits against the frozen target.

    Args:
        scorers: frozen models combined into ``log q`` (see module docstring).
        row_mask: ``(N,)`` bool, rows that carry a distillation target. Other rows get
            ``rkl = 0`` and never touch the frozen heads.
        temperature: temperature applied to every frozen model's logits.
        chunk_size: rows per ``[chunk, V]`` buffer.
    """

    __slots__ = ("scorers", "row_mask", "temperature", "chunk_size")

    def __init__(
        self,
        scorers: list[FrozenScorer],
        row_mask: torch.Tensor,
        temperature: float = 1.0,
        chunk_size: int = 256,
    ):
        if not scorers:
            raise ValueError("FullVocabTarget needs at least one frozen scorer")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        self.scorers = scorers
        self.row_mask = row_mask
        self.temperature = float(temperature)
        self.chunk_size = int(chunk_size)


def _log_softmax_(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """In-place fp32 ``log_softmax(logits / temperature)``; ``logits`` must already be a fresh buffer."""
    if temperature != 1.0:
        logits.div_(temperature)
    return logits.sub_(torch.logsumexp(logits, dim=-1, keepdim=True))


def _student_log_probs(hidden: torch.Tensor, weight: torch.Tensor, temperature: float) -> torch.Tensor:
    return _log_softmax_((hidden @ weight.t()).float(), temperature)


def _target_log_probs(
    scorers: list[FrozenScorer],
    rows: torch.Tensor | slice,
    temperature: float,
    device_type: str,
    labels: Optional[torch.Tensor] = None,
    student_probs: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """``(n, V)`` fp32 target log-probs at the given packed rows. Never part of the graph.

    With ``(n, 1)`` ``labels``, also returns the ``(n,)`` log-normalizer of the combination and
    the ``(n, num_scorers)`` log-prob each scorer assigns to its label; otherwise both are None.
    With the ``(n, V)`` ``student_probs`` of the same rows, also returns ``(n, 3 * num_scorers)``
    per-scorer statistics: the cross-entropy ``-sum p_student log scorer``, the scorer's entropy
    and its argmax id, each block in scorer order; otherwise None.
    """
    total = None
    label_log_probs = [] if labels is not None else None
    stats = [[], [], []] if student_probs is not None else None
    log_z = None
    with torch.no_grad(), torch.autocast(device_type=device_type, enabled=False):
        for scorer in scorers:
            hidden = scorer.hidden[rows].to(scorer.head.dtype)
            log_probs = _log_softmax_((hidden @ scorer.head.t()).float(), temperature)
            if label_log_probs is not None:
                label_log_probs.append(log_probs.gather(-1, labels).squeeze(-1))
            if stats is not None:
                stats[0].append(-(student_probs * log_probs).sum(dim=-1))
                stats[1].append(-(log_probs.exp().mul_(log_probs)).sum(dim=-1))
                stats[2].append(log_probs.argmax(dim=-1).float())
            if scorer.coef != 1.0:
                log_probs.mul_(scorer.coef)
            total = log_probs if total is None else total.add_(log_probs)
        if len(scorers) > 1 or scorers[0].coef != 1.0:
            log_z = torch.logsumexp(total, dim=-1, keepdim=True)
            total.sub_(log_z)
            log_z = log_z.squeeze(-1)
    scorer_stats = torch.stack([s for block in stats for s in block], dim=-1) if stats is not None else None
    if labels is None:
        return total, None, None, scorer_stats
    if log_z is None:
        log_z = total.new_zeros(total.shape[0])
    return total, log_z, torch.stack(label_log_probs, dim=-1), scorer_stats


def _masked_rows(row_mask: torch.Tensor, start: int, end: int) -> tuple[torch.Tensor | slice, torch.Tensor]:
    """Local index of the target rows in ``[start, end)`` (a full slice when every row qualifies)."""
    local = row_mask[start:end].nonzero().squeeze(-1)
    if local.numel() == end - start:
        return slice(None), local
    return local, local


class FullVocabRKLFunction(torch.autograd.Function):
    """Sampled-token log-prob, entropy and full-vocabulary reverse KL from the student hidden.

    Inputs are flattened ``(N, H)`` / ``(N,)``. Outputs are ``(N,)`` fp32 ``log_probs``,
    ``entropy`` and ``rkl`` (all differentiable), plus ``(N, diag_layout(num_scorers)["width"])``
    detached diagnostics (see the ``DIAG_*`` columns and :func:`diag_layout`).
    """

    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        temperature: float,
        target: FullVocabTarget,
    ):
        ctx.set_materialize_grads(False)
        n = hidden.shape[0]
        device = hidden.device
        hidden_c = hidden.to(weight.dtype)
        row_mask = target.row_mask.to(device=device, dtype=torch.bool)
        assert row_mask.shape == (n,), f"row_mask {tuple(row_mask.shape)} does not match {n} rows"
        for scorer in target.scorers:
            assert scorer.hidden.shape[0] == n, (
                f"frozen hidden has {scorer.hidden.shape[0]} rows, student has {n}; the packed frames are misaligned"
            )
            assert scorer.head.shape[0] == weight.shape[0], (
                f"frozen vocab {scorer.head.shape[0]} != student vocab {weight.shape[0]}"
            )

        log_probs = torch.zeros(n, dtype=torch.float32, device=device)
        entropy = torch.zeros(n, dtype=torch.float32, device=device)
        rkl = torch.zeros(n, dtype=torch.float32, device=device)
        layout = diag_layout(len(target.scorers))
        diag = torch.zeros(n, layout["width"], dtype=torch.float32, device=device)

        chunk = target.chunk_size
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            chunk_labels = labels[start:end].unsqueeze(-1)
            logp = _student_log_probs(hidden_c[start:end], weight, temperature)
            p = logp.exp()
            chunk_entropy = -(p * logp).sum(dim=-1)
            entropy[start:end] = chunk_entropy
            log_probs[start:end] = logp.gather(-1, chunk_labels).squeeze(-1)

            sel, local = _masked_rows(row_mask, start, end)
            if local.numel() == 0:
                continue
            target_labels = chunk_labels[sel]
            student_entropy = chunk_entropy[sel]
            logq, log_z, scorer_log_probs, scorer_stats = _target_log_probs(
                target.scorers,
                local + start,
                target.temperature,
                device.type,
                labels=target_labels,
                student_probs=p[sel],
            )
            # rkl = sum p (log p - log q) = -H(p) - sum p log q
            chunk_rkl = -student_entropy - (p[sel] * logq).sum(dim=-1)
            teacher_log_prob = logq.gather(-1, target_labels).squeeze(-1)
            teacher_entropy = -(logq.exp() * logq).sum(dim=-1)
            rows = local + start
            rkl[rows] = chunk_rkl
            diag[rows, DIAG_TEACHER_ENTROPY] = teacher_entropy
            diag[rows, DIAG_TEACHER_LOG_PROB] = teacher_log_prob
            diag[rows, DIAG_LOG_Z] = log_z
            diag[rows, layout["scorer_log_prob"]] = scorer_log_probs
            diag[rows, layout["label"]] = target_labels.squeeze(-1).float()
            diag[rows, layout["student_entropy"]] = student_entropy
            diag[rows, layout["student_argmax"]] = logp[sel].argmax(dim=-1).float()
            diag[rows, layout["target_argmax"]] = logq.argmax(dim=-1).float()
            # KL(p || scorer) = -H(p) + cross-entropy; the other two blocks are stored as computed.
            k = len(target.scorers)
            scorer_stats[:, :k] -= student_entropy.unsqueeze(-1)
            diag[rows, layout["scorer_kl"].start : layout["scorer_argmax"].stop] = scorer_stats
            del logp, p, logq

        ctx.save_for_backward(hidden_c, weight, labels, row_mask, entropy, rkl)
        ctx.hidden_dtype = hidden.dtype
        ctx.temperature = temperature
        ctx.target = target
        ctx.mark_non_differentiable(diag)
        return log_probs, entropy, rkl, diag

    @staticmethod
    def backward(
        ctx,
        d_log_probs: Optional[torch.Tensor],
        d_entropy: Optional[torch.Tensor],
        d_rkl: Optional[torch.Tensor],
        _d_diag: Optional[torch.Tensor],
    ):
        hidden, weight, labels, row_mask, entropy, rkl = ctx.saved_tensors
        target: FullVocabTarget = ctx.target
        temperature = ctx.temperature
        need_hidden, need_weight = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        if (d_log_probs is None and d_entropy is None and d_rkl is None) or not (need_hidden or need_weight):
            return None, None, None, None, None

        n = hidden.shape[0]
        device = hidden.device
        d_hidden = torch.zeros_like(hidden) if need_hidden else None
        d_weight = torch.zeros_like(weight) if need_weight else None
        if d_rkl is not None:
            d_rkl = d_rkl.float() * row_mask.float()

        chunk = target.chunk_size
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            logp = _student_log_probs(hidden[start:end], weight, temperature)
            p = logp.exp()

            # d/dz of  g_lp * logp[y]  +  g_H * H  +  g_R * rkl  is
            #   p * ((g_R - g_H) * logp - g_lp - g_H * H - g_R * rkl)  -  g_R * p * logq  +  g_lp * onehot(y)
            coef_logp = torch.zeros(end - start, dtype=torch.float32, device=device)
            bias = torch.zeros_like(coef_logp)
            if d_entropy is not None:
                g_h = d_entropy[start:end].float()
                coef_logp -= g_h
                bias -= g_h * entropy[start:end]
            if d_log_probs is not None:
                bias -= d_log_probs[start:end].float()
            g_r = None
            if d_rkl is not None:
                g_r = d_rkl[start:end]
                coef_logp += g_r
                bias -= g_r * rkl[start:end]

            d_logits = logp.mul_(coef_logp.unsqueeze(-1)).add_(bias.unsqueeze(-1)).mul_(p)
            if g_r is not None:
                sel, local = _masked_rows(row_mask, start, end)
                if local.numel() > 0:
                    logq, _, _, _ = _target_log_probs(target.scorers, local + start, target.temperature, device.type)
                    logq.mul_(p[sel]).mul_(g_r[sel].unsqueeze(-1))
                    if isinstance(sel, slice):
                        d_logits.sub_(logq)
                    else:
                        d_logits.index_add_(0, local, logq, alpha=-1.0)
                    del logq
            if d_log_probs is not None:
                d_logits.scatter_add_(-1, labels[start:end].unsqueeze(-1), d_log_probs[start:end].float().unsqueeze(-1))
            if temperature != 1.0:
                d_logits.div_(temperature)
            d_logits = d_logits.to(weight.dtype)
            del logp, p

            if need_hidden:
                d_hidden[start:end] = d_logits @ weight
            if need_weight:
                d_weight.addmm_(d_logits.t(), hidden[start:end])

        if d_hidden is not None:
            d_hidden = d_hidden.to(ctx.hidden_dtype)
        return d_hidden, d_weight, None, None, None


def full_vocab_rkl(
    hidden_states: torch.Tensor,
    vocab_weights: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    target: FullVocabTarget,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Leading dims of ``hidden_states`` (e.g. the packed ``(1, N, H)``) are kept on every output.

    Returns:
        ``log_probs``, ``entropy``, ``rkl`` of shape ``hidden_states.shape[:-1]`` and the
        detached ``diag`` of shape ``(*hidden_states.shape[:-1], diag_layout(num_scorers)["width"])``.
    """
    lead = hidden_states.shape[:-1]
    flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    flat_labels = labels.reshape(-1).to(torch.int64)
    log_probs, entropy, rkl, diag = FullVocabRKLFunction.apply(
        flat_hidden, vocab_weights, flat_labels, float(temperature), target
    )
    return log_probs.view(lead), entropy.view(lead), rkl.view(lead), diag.view(*lead, diag.shape[-1])


def load_lm_head_weight(model_path: str, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Read a frozen model's ``(V, H)`` LM head straight from its safetensors checkpoint.

    Falls back to the input embedding for tied-embedding models. Reading the file sidesteps
    gathering an FSDP-sharded, CPU-offloaded parameter, and the frozen weight is identical.
    """
    import json
    import os

    from safetensors import safe_open

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
    else:
        single = os.path.join(model_path, "model.safetensors")
        if not os.path.exists(single):
            raise FileNotFoundError(f"No safetensors checkpoint under {model_path}")
        with safe_open(single, framework="pt") as f:
            weight_map = {key: "model.safetensors" for key in f.keys()}

    def _pick(suffix: str) -> Optional[str]:
        names = [k for k in weight_map if k.endswith(suffix) and "mtp" not in k and "visual" not in k]
        return min(names, key=len) if names else None

    name = _pick("lm_head.weight") or _pick("embed_tokens.weight")
    if name is None:
        raise KeyError(f"Neither lm_head.weight nor embed_tokens.weight found in {model_path}")
    with safe_open(os.path.join(model_path, weight_map[name]), framework="pt") as f:
        weight = f.get_tensor(name)
    return weight.to(dtype).contiguous()
