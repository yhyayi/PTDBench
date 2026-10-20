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
"""CPU tests for the sparse log-softmax gather used by colocated OPD.

The point of the custom autograd Function is to avoid keeping an ``(N, vocab)``
fp32 tensor alive for backward. These tests pin it to the reference
``log_softmax(...).gather(...)`` in both directions.
"""

import pytest
import torch

from verl.trainer.opd.kernels import sparse_log_probs


def _logits(n, v, seed, dtype=torch.float32):
    return torch.randn(n, v, generator=torch.Generator().manual_seed(seed), dtype=dtype)


@pytest.mark.parametrize("chunk_size", [1, 3, 1024])
def test_gather_matches_log_softmax_forward_and_backward(chunk_size):
    n, v, k = 7, 23, 4
    logits = _logits(n, v, 0, torch.float64).requires_grad_(True)
    ref_logits = logits.detach().clone().requires_grad_(True)
    ids = torch.randint(0, v, (n, k), generator=torch.Generator().manual_seed(1))
    grad_out = _logits(n, k, 2, torch.float64)

    out = sparse_log_probs(logits, candidate_ids=ids, chunk_size=chunk_size)["cand_log_probs"]
    ref = ref_logits.log_softmax(-1).gather(-1, ids)
    torch.testing.assert_close(out, ref, atol=1e-12, rtol=1e-10)

    out.backward(grad_out)
    ref.backward(grad_out)
    torch.testing.assert_close(logits.grad, ref_logits.grad, atol=1e-12, rtol=1e-10)


def test_duplicate_ids_accumulate():
    """A repeated id must receive the sum of its slots' gradients, not the last one."""
    n, v = 3, 9
    logits = _logits(n, v, 3, torch.float64).requires_grad_(True)
    ref_logits = logits.detach().clone().requires_grad_(True)
    ids = torch.tensor([[2, 2, 5], [0, 0, 0], [8, 1, 8]])
    grad_out = _logits(n, 3, 4, torch.float64)

    out = sparse_log_probs(logits, candidate_ids=ids, chunk_size=2)["cand_log_probs"]
    ref = ref_logits.log_softmax(-1).gather(-1, ids)
    torch.testing.assert_close(out, ref, atol=1e-12, rtol=1e-10)
    out.backward(grad_out)
    ref.backward(grad_out)
    torch.testing.assert_close(logits.grad, ref_logits.grad, atol=1e-12, rtol=1e-10)


def test_topk_matches_reference_and_reports_kth():
    n, v, k = 5, 17, 4
    logits = _logits(n, v, 5)
    out = sparse_log_probs(logits, top_k=k, need_kth=True, chunk_size=2)

    ref = logits.float().log_softmax(-1).topk(k, dim=-1)
    torch.testing.assert_close(out["topk_ids"], ref.indices)
    torch.testing.assert_close(out["topk_log_probs"], ref.values, atol=1e-6, rtol=1e-5)
    # kth_log_prob is the membership threshold used by intersection/union.
    torch.testing.assert_close(out["kth_log_prob"], ref.values[:, -1], atol=1e-6, rtol=1e-5)


def test_topk_and_gather_in_one_pass():
    """`union` needs both from a single forward; they must not disturb each other."""
    n, v, k = 4, 13, 3
    logits = _logits(n, v, 6, torch.float64).requires_grad_(True)
    ids = torch.randint(0, v, (n, 2), generator=torch.Generator().manual_seed(7))
    out = sparse_log_probs(logits, top_k=k, candidate_ids=ids, need_kth=True, chunk_size=3)

    ref_lp = logits.detach().log_softmax(-1)
    torch.testing.assert_close(out["cand_log_probs"], ref_lp.gather(-1, ids), atol=1e-12, rtol=1e-10)
    torch.testing.assert_close(out["topk_ids"], ref_lp.topk(k, dim=-1).indices)
    # top-k is a no_grad diagnostic; only the gather carries gradient.
    assert out["cand_log_probs"].requires_grad
    assert not out["topk_log_probs"].requires_grad


def test_int32_candidate_ids_are_accepted():
    """Ids can arrive as int32 from the packed path; gather needs int64."""
    logits = _logits(3, 11, 8)
    ids = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32)
    out = sparse_log_probs(logits, candidate_ids=ids, chunk_size=4)
    ref = logits.float().log_softmax(-1).gather(-1, ids.long())
    torch.testing.assert_close(out["cand_log_probs"], ref, atol=1e-6, rtol=1e-5)


def test_rejects_oversized_top_k_and_kth_without_top_k():
    logits = _logits(2, 5, 9)
    with pytest.raises(ValueError, match="exceeds vocab size"):
        sparse_log_probs(logits, top_k=6)
    with pytest.raises(ValueError, match="need_kth requires top_k"):
        sparse_log_probs(logits, top_k=0, need_kth=True)


def test_empty_batch_keeps_shapes():
    logits = torch.zeros(0, 7)
    out = sparse_log_probs(logits, top_k=2, candidate_ids=torch.zeros(0, 3, dtype=torch.int64))
    assert out["topk_ids"].shape == (0, 2)
    assert out["cand_log_probs"].shape == (0, 3)
