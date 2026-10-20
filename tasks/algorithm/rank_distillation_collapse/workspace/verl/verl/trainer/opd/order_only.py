"""Order-only and matched-candidate JSD distillation primitives.

Scores are ordered by the frozen teacher, never by the student. Both objectives
are invariant to a shared additive normalizer, so candidate log-probabilities
can be used in place of logits. Generation temperature must not scale them.
"""

import math

import torch
import torch.nn.functional as F


def direct_topk_loss(
    scores: torch.Tensor, objective: str, teacher_scores=None, mask_teacher_ties: bool = False
) -> torch.Tensor:
    """Return unreduced per-position losses; accumulate in at least fp32.

    With ``mask_teacher_ties``, adjacent pairs whose ``teacher_scores`` are equal contribute
    zero, while the mean still divides by ``K - 1``.
    """
    if scores.shape[-1] < 2:
        raise ValueError("Direct top-K loss requires at least two candidates")
    scores = scores.to(torch.promote_types(scores.dtype, torch.float32))
    if objective == "adjacent_logistic":
        pairs = F.softplus(scores[..., 1:] - scores[..., :-1])
        if mask_teacher_ties:
            if teacher_scores is None or teacher_scores.shape != scores.shape:
                raise ValueError("Tie masking requires teacher scores with the same shape as student scores")
            teacher_scores = teacher_scores.detach()
            pairs = pairs * (teacher_scores[..., :-1] > teacher_scores[..., 1:]).to(pairs.dtype)
        return pairs.mean(-1)
    if objective != "topk_jsd":
        raise ValueError(f"Unknown direct objective {objective!r}")
    if teacher_scores is None or teacher_scores.shape != scores.shape:
        raise ValueError("JSD requires teacher scores with the same shape as student scores")
    log_p = scores.log_softmax(-1)
    log_q = teacher_scores.detach().to(scores.dtype).log_softmax(-1)
    log_m = torch.logaddexp(log_p, log_q) - math.log(2)
    return 0.5 * ((log_p.exp() * (log_p - log_m)).sum(-1) + (log_q.exp() * (log_q - log_m)).sum(-1))


@torch.no_grad()
def topk_agreement_metrics(
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    student_topk_ids: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> dict[str, float]:
    """Top-K diagnostics on the complete, pre-update rollout batch.

    Scores on both sides are unscaled logits or log-probabilities at the
    *teacher's* ordered candidates. The two ID tensors instead contain each
    model's own top-K over the common tokenizer vocabulary (not padded LM-head
    rows), sorted by score then ID. Only valid response positions enter means.

    Kendall tau-b uses the actual scores, retaining ties in either model. A
    position with either side constant has undefined tau and is excluded from
    that mean; valid_fraction exposes the denominator. With no defined tau,
    omit that metric rather than logging an invented zero or a NaN.
    """
    shape = student_scores.shape
    if len(shape) < 2 or shape[-1] < 2:
        raise ValueError("Top-K diagnostics require scores shaped (..., K), K >= 2")
    if any(x.shape != shape for x in (teacher_scores, student_topk_ids, teacher_topk_ids)):
        raise ValueError("Student/teacher diagnostic tensors must have identical shapes")
    if response_mask.shape != shape[:-1]:
        raise ValueError("Diagnostic mask must match the response positions")
    mask = response_mask.bool()
    count = mask.sum().item()
    k = shape[-1]
    metrics = {
        "opd/diagnostic_top_k": float(k),
        "opd/diagnostic_tokens": float(count),
        "opd/kendall_tau_valid_fraction": 0.0,
    }
    if not count:
        return metrics
    # Filter first: padding may contain nonfinite values or meaningless IDs.
    dtype = torch.promote_types(student_scores.dtype, torch.float32)
    student = student_scores[mask].to(dtype)
    teacher = teacher_scores[mask].to(dtype)
    student_ids, teacher_ids = student_topk_ids[mask], teacher_topk_ids[mask]
    metrics["opd/jsd_topk"] = direct_topk_loss(student, "topk_jsd", teacher).mean().item()
    overlap = (teacher_ids[..., :, None] == student_ids[..., None, :]).any(-1).to(dtype).mean(-1)
    metrics["opd/topk_overlap"] = overlap.mean().item()
    # Each first ID is that model's full-valid-vocabulary argmax, not a candidate argmax.
    metrics["opd/top1_agreement"] = (student_ids[:, 0] == teacher_ids[:, 0]).to(dtype).mean().item()

    i, j = torch.triu_indices(k, k, offset=1, device=student.device)
    student_sign = (student[:, i] - student[:, j]).sign()
    teacher_sign = (teacher[:, i] - teacher[:, j]).sign()
    numerator = (student_sign * teacher_sign).sum(-1)
    denominator = (
        student_sign.ne(0).sum(-1).to(dtype) * teacher_sign.ne(0).sum(-1).to(dtype)
    ).sqrt()
    defined = denominator > 0
    metrics["opd/kendall_tau_valid_fraction"] = defined.to(dtype).mean().item()
    if defined.any():
        metrics["opd/kendall_tau_topk"] = (numerator[defined] / denominator[defined]).mean().item()
    return metrics


def use_fp32_lm_head(module: torch.nn.Module) -> torch.nn.Module:
    """Make the (possibly FSDP-wrapped) model's LM head return exact fp32 products.

    Upcasting bf16 hidden states and weights is lossless; disabling autocast and TF32
    keeps the product itself in full fp32, so logits are no longer rounded to the bf16
    grid. The weight is read at call time, after FSDP has unsharded it.
    """
    heads = [m for name, m in module.named_modules() if name.split(".")[-1] == "lm_head"]
    if len(heads) != 1 or not isinstance(heads[0], torch.nn.Linear):
        raise ValueError(f"Expected exactly one nn.Linear lm_head, found {len(heads)}")
    head = heads[0]

    def forward(hidden: torch.Tensor) -> torch.Tensor:
        allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                bias = None if head.bias is None else head.bias.float()
                return F.linear(hidden.float(), head.weight.float(), bias)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32

    head.forward = forward
    return head


def common_token_ids(student_tokenizer, teacher_tokenizer, student_vocab: int, teacher_vocab: int) -> list[int]:
    """Allow padded LM heads of different sizes, but never different token IDs."""
    student = student_tokenizer.get_vocab()
    if student != teacher_tokenizer.get_vocab():
        raise ValueError("Direct top-K distillation requires identical tokenizer token-to-ID mappings")
    for key in ("bos_token_id", "eos_token_id", "pad_token_id", "all_special_ids"):
        if getattr(student_tokenizer, key) != getattr(teacher_tokenizer, key):
            raise ValueError(f"Teacher/student tokenizer mismatch: {key}")
    ids = sorted(set(student.values()))
    if not ids or ids[0] < 0 or ids[-1] >= min(student_vocab, teacher_vocab):
        raise ValueError("Tokenizer IDs exceed student or teacher LM-head size")
    return ids


@torch.no_grad()
def ordered_topk(logits: torch.Tensor, valid_ids: torch.Tensor, k: int, chunk_size: int):
    """Top-K over valid tokens, descending score then ascending token ID.

Only boundary ties require scanning the vocabulary again. Sorts of the selected
K elements are stable; no perturbation of scores or full-vocabulary sort is used.
"""
    valid_ids = valid_ids.to(device=logits.device, dtype=torch.long)
    if k < 2 or k > valid_ids.numel() or chunk_size < 1:
        raise ValueError("Invalid K/chunk size for ordered_topk")
    if valid_ids.ndim != 1 or (valid_ids[1:] <= valid_ids[:-1]).any():
        raise ValueError("valid_ids must be strictly increasing")
    if valid_ids[0] < 0 or valid_ids[-1] >= logits.shape[-1]:
        raise ValueError("Candidate ID outside LM-head size")
    values, indices = [], []
    for block in logits.split(chunk_size):
        scores = block.index_select(-1, valid_ids)
        # Read K+1 to detect ties spanning the cutoff (unless K is the whole set).
        top_values, top_indices = scores.topk(min(k + 1, scores.shape[-1]), dim=-1)
        selected = top_indices[:, :k].clone()
        if scores.shape[-1] > k:
            boundary_tie = top_values[:, k] == top_values[:, k - 1]
            if boundary_tie.any():
                tied_scores = scores[boundary_tie]
                cutoff = top_values[boundary_tie, k - 1, None]
                positions = torch.arange(scores.shape[-1], device=scores.device)
                tied_ids = torch.where(tied_scores == cutoff, positions, scores.shape[-1])
                smallest = tied_ids.topk(k, dim=-1, largest=False, sorted=True).values
                num_greater = (top_values[boundary_tie, :k] > cutoff).sum(-1, keepdim=True)
                slots = torch.arange(k, device=scores.device).expand_as(smallest)
                replacement = smallest.gather(-1, (slots - num_greater).clamp_min(0))
                selected[boundary_tie] = torch.where(slots >= num_greater, replacement, selected[boundary_tie])
        # selected indices have the same order as token IDs because valid_ids is sorted.
        selected = selected.sort(dim=-1).values
        chosen_scores = scores.gather(-1, selected)
        permutation = chosen_scores.argsort(dim=-1, descending=True, stable=True)
        selected = selected.gather(-1, permutation)
        values.append(scores.gather(-1, selected))
        indices.append(valid_ids[selected])
    if not values:
        return logits.new_empty((0, k)), valid_ids.new_empty((0, k))
    return torch.cat(values), torch.cat(indices)
