from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


workspace = Path(os.environ["PTDBENCH_WORKSPACE"])
sys.path.insert(0, str(workspace / "verl"))

from verl.trainer.opd import kernels  # noqa: E402
from verl.utils import torch_functional  # noqa: E402


def _emulated_optimized_logprobs(logits: torch.Tensor, labels: torch.Tensor, **_: object) -> torch.Tensor:
    """Model the packed-label contract of the optimized CUDA operator."""
    flat = labels.reshape(-1)
    if flat.numel() and flat.stride(0) != 1:
        flat = torch.as_strided(flat, (flat.numel(),), (1,), storage_offset=flat.storage_offset())
    return -F.cross_entropy(logits, flat.to(torch.long), reduction="none")


def _run_case() -> None:
    n, vocab, width = 9, 37, 4
    torch.manual_seed(9103)
    logits = torch.randn(n, vocab, dtype=torch.float64, requires_grad=True)
    reference_logits = logits.detach().clone().requires_grad_(True)
    candidate_ids = torch.randint(0, vocab, (n, width), dtype=torch.long)
    candidate_ids[1, 0] = candidate_ids[1, 1]
    weights = torch.randn(n, width, dtype=torch.float64)

    had_kernel_alias = hasattr(kernels, "logprobs_from_logits")
    original_kernel_alias = getattr(kernels, "logprobs_from_logits", None)
    original_utility = torch_functional.logprobs_from_logits
    kernels.logprobs_from_logits = _emulated_optimized_logprobs
    torch_functional.logprobs_from_logits = _emulated_optimized_logprobs
    try:
        result = kernels.sparse_log_probs(
            logits,
            top_k=6,
            candidate_ids=candidate_ids,
            need_kth=True,
            chunk_size=3,
        )
    finally:
        torch_functional.logprobs_from_logits = original_utility
        if had_kernel_alias:
            kernels.logprobs_from_logits = original_kernel_alias
        else:
            del kernels.logprobs_from_logits

    expected = reference_logits.log_softmax(-1).gather(-1, candidate_ids)
    loss = (result["cand_log_probs"] * weights).sum()
    expected_loss = (expected * weights).sum()
    loss.backward()
    expected_loss.backward()

    torch.testing.assert_close(result["cand_log_probs"], expected, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(logits.grad, reference_logits.grad, atol=1e-10, rtol=1e-10)
    assert result["topk_ids"].shape == (n, 6)
    assert result["kth_log_prob"].shape == (n,)
    assert not result["topk_log_probs"].requires_grad


def _run_real_cuda_smoke() -> None:
    if not torch.cuda.is_available():
        return

    torch.manual_seed(9104)
    logits = torch.randn(8, 257, device="cuda", dtype=torch.float32, requires_grad=True)
    ids = torch.randint(0, 257, (8, 3), device="cuda")
    out = kernels.sparse_log_probs(logits, top_k=3, candidate_ids=ids)["cand_log_probs"]
    ref = logits.detach().clone().requires_grad_(True)
    expected = ref.log_softmax(-1).gather(-1, ids)
    out.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(out, expected, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(logits.grad, ref.grad, atol=2e-4, rtol=2e-4)


if __name__ == "__main__":
    _run_case()
    _run_real_cuda_smoke()
    print("candidate forward/backward regression passed")
