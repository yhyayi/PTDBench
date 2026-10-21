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

"""Sparse log-probability helpers for colocated on-policy distillation."""

import torch

from verl.utils.torch_functional import logprobs_from_logits

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


def sparse_log_probs(
    logits: torch.Tensor,
    *,
    top_k: int = 0,
    candidate_ids: torch.Tensor | None = None,
    need_kth: bool = False,
    chunk_size: int = 4096,
) -> dict[str, torch.Tensor]:
    """Top-k and/or gathered log-probs from packed logits.

    Args:
        logits: ``(N, V)`` packed logits, already temperature-scaled.
        top_k: if > 0, also return the row-wise top-k ids and their log-probs.
        candidate_ids: ``(N, K)`` ids to gather. Only this output carries grad.
        need_kth: if True, also return the k-th largest log-prob per row, used as
            a membership threshold ("is id v inside this model's top-k?").
        chunk_size: rows per vocab reduction step; bounds peak memory only.

    Returns:
        dict with any of ``topk_ids`` ``(N, k)`` int64, ``topk_log_probs`` ``(N, k)``
        fp32, ``kth_log_prob`` ``(N,)`` fp32, ``cand_log_probs`` ``(N, K)`` fp32.
        Everything except ``cand_log_probs`` is detached.
    """
    assert logits.dim() == 2, f"expected packed (N, V) logits, got {tuple(logits.shape)}"
    n, vocab_size = logits.shape
    if top_k > vocab_size:
        raise ValueError(f"opd.top_k={top_k} exceeds vocab size {vocab_size}.")

    out: dict[str, torch.Tensor] = {}

    if top_k > 0:
        with torch.no_grad():
            log_z = _chunked_logsumexp(logits, chunk_size)  # (N,) fp32
            topk_logits, topk_ids = torch.topk(logits, k=top_k, dim=-1)
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
        candidate_ids = candidate_ids.to(torch.int64)
        width = candidate_ids.shape[-1]
        if n == 0:
            out["cand_log_probs"] = torch.empty(
                (0, width),
                dtype=_accum_dtype(logits),
                device=logits.device,
            )
        else:
            out["cand_log_probs"] = torch.stack(
                [
                    logprobs_from_logits(
                        logits=logits,
                        labels=candidate_ids[..., i],
                        inplace_backward=False,
                    )
                    for i in range(width)
                ],
                dim=-1,
            )

    return out
