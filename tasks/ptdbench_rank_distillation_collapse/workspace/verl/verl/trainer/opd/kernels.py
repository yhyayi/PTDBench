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

"""Sparse log-softmax kernels for colocated on-policy distillation.

OPD only ever needs ``log_softmax(logits)`` at a handful of candidate ids per
position, never the full ``(N, vocab)`` matrix. These helpers exploit

    log_softmax(x)[i] == x[i] - logsumexp(x)

so the vocab-sized reduction streams in row chunks. Backward reuses the original
logits and never retains a full-vocabulary fp32 log-softmax matrix -- see
:class:`_SparseLogSoftmaxGather`.
"""

import torch

__all__ = ["sparse_log_probs"]


def _chunked_logsumexp(logits: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """``logsumexp(logits, dim=-1)`` in fp32, streamed over rows.

    Args:
        logits: ``(N, V)`` logits, any float dtype.
        chunk_size: rows reduced per step. Bounds the transient fp32 buffer to
            ``chunk_size * V``; it does not change the result.

    Returns:
        ``(N,)`` tensor in ``_accum_dtype(logits)``.
    """
    n = logits.shape[0]
    dtype = _accum_dtype(logits)
    out = torch.empty(n, dtype=dtype, device=logits.device)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        out[start:end] = torch.logsumexp(logits[start:end].to(dtype), dim=-1)
    return out


def _accum_dtype(tensor: torch.Tensor) -> torch.dtype:
    """fp32 for bf16/fp16/fp32 logits; never downcast an fp64 caller."""
    return torch.promote_types(tensor.dtype, torch.float32)


class _SparseLogSoftmaxGather(torch.autograd.Function):
    """``log_softmax(logits, -1).gather(-1, ids)`` without an ``(N, V)`` saved tensor.

    Forward saves only ``logits`` (already alive as the caller's activation),
    ``ids`` and the ``(N,)`` ``log_z``. Backward re-derives the softmax row by
    row, so peak extra memory is one fp32 chunk instead of a full fp32 copy of
    the logits -- which is what ``torch.logsumexp``/``F.log_softmax`` would keep.

    Gradient of ``y[i,k] = x[i, ids[i,k]] - log_z[i]``::

        dL/dx[i,v] = sum_k dL/dy[i,k] * (1{v == ids[i,k]} - softmax(x[i])[v])
    """

    @staticmethod
    def forward(ctx, logits: torch.Tensor, ids: torch.Tensor, log_z: torch.Tensor, chunk_size: int):
        gathered = torch.gather(logits, dim=-1, index=ids).to(log_z.dtype) - log_z.unsqueeze(-1)
        ctx.save_for_backward(logits, ids, log_z)
        ctx.chunk_size = chunk_size
        return gathered

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        logits, ids, log_z = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        grad_logits = torch.zeros_like(logits)
        grad_out = grad_out.to(log_z.dtype)
        grad_sum = grad_out.sum(dim=-1)  # (N,)

        for start in range(0, logits.shape[0], chunk_size):
            end = min(start + chunk_size, logits.shape[0])
            # -softmax(x) * sum_k dL/dy   (the log_z term, shared across k)
            probs = (logits[start:end].to(log_z.dtype) - log_z[start:end].unsqueeze(-1)).exp_()
            probs.mul_(-grad_sum[start:end].unsqueeze(-1))
            # +dL/dy[i,k] at each candidate id; scatter_add handles duplicate ids
            probs.scatter_add_(-1, ids[start:end], grad_out[start:end])
            grad_logits[start:end] = probs.to(grad_logits.dtype)

        return grad_logits, None, None, None


def sparse_log_probs(
    logits: torch.Tensor,
    *,
    top_k: int = 0,
    candidate_ids: torch.Tensor | None = None,
    need_kth: bool = False,
    chunk_size: int = 4096,
    valid_token_ids: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Top-k and/or gathered log-probs from packed logits.

    Args:
        logits: ``(N, V)`` packed logits, already temperature-scaled.
        top_k: if > 0, also return the row-wise top-k ids and their log-probs.
        candidate_ids: ``(N, K)`` ids to gather. Only this output carries grad.
        need_kth: if True, also return the k-th largest log-prob per row, used as
            a membership threshold ("is id v inside this model's top-k?").
        chunk_size: rows per vocab reduction step; bounds peak memory only.
        valid_token_ids: optional sorted tokenizer IDs for deterministic teacher
            top-K (descending logits, ties by ascending ID). Excludes padded LM
            head rows from selection; log-probs still use the full normalizer.

    Returns:
        dict with any of ``topk_ids`` ``(N, k)`` int64, ``topk_log_probs`` ``(N, k)``
        fp32, ``kth_log_prob`` ``(N,)`` fp32, ``cand_log_probs`` ``(N, K)`` fp32.
        Everything except ``cand_log_probs`` is detached.
    """
    assert logits.dim() == 2, f"expected packed (N, V) logits, got {tuple(logits.shape)}"
    n, vocab_size = logits.shape
    if top_k > vocab_size:
        raise ValueError(f"opd.top_k={top_k} exceeds vocab size {vocab_size}.")

    with torch.no_grad():
        log_z = _chunked_logsumexp(logits, chunk_size)  # (N,) fp32

    out: dict[str, torch.Tensor] = {}

    if top_k > 0:
        with torch.no_grad():
            if valid_token_ids is None:
                topk_logits, topk_ids = torch.topk(logits, k=top_k, dim=-1)
            else:
                from verl.trainer.opd.order_only import ordered_topk

                topk_logits, topk_ids = ordered_topk(logits, valid_token_ids, top_k, chunk_size)
            # log_softmax is monotonic, so topk(logits) == topk(log_softmax(logits))
            topk_log_probs = topk_logits.to(log_z.dtype) - log_z.unsqueeze(-1)
            out["topk_ids"] = topk_ids
            out["topk_log_probs"] = topk_log_probs
            if need_kth:
                out["kth_log_prob"] = topk_log_probs[:, -1]
    elif need_kth:
        raise ValueError("need_kth requires top_k > 0.")

    if candidate_ids is not None:
        assert candidate_ids.shape[0] == n, (
            f"candidate_ids rows {candidate_ids.shape[0]} != logits rows {n}; "
            "the packed candidate tensor is misaligned with the logits."
        )
        out["cand_log_probs"] = _SparseLogSoftmaxGather.apply(logits, candidate_ids.to(torch.int64), log_z, chunk_size)

    return out
