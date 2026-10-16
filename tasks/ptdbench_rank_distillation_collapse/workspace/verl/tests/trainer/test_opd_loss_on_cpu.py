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
"""CPU gradient coverage for the sparse candidate-token OPD loss."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData

from verl.trainer.opd.advantage import opd_advantage
from verl.trainer.opd.kernels import sparse_log_probs
from verl.trainer.opd.losses import (
    _candidate_log_probs_to_response,
    _opd_candidate_loss,
    _to_response_width,
)


def test_candidate_loss_matches_analytic_policy_gradient_at_ratio_one():
    n, vocab, width = 4, 9, 3
    logits = torch.randn(
        n, vocab, generator=torch.Generator().manual_seed(123), dtype=torch.float64, requires_grad=True
    )
    candidate_ids = torch.tensor([[0, 3, 7], [1, 4, 8], [2, 3, 5], [0, 6, 8]])
    advantages = torch.randn(n, width, generator=torch.Generator().manual_seed(456), dtype=torch.float64)
    cand_mask = torch.tensor([[True, True, True], [True, False, True], [True, True, False], [True, True, True]])

    current = sparse_log_probs(logits, candidate_ids=candidate_ids, chunk_size=2)["cand_log_probs"]
    old = current.detach().clone()
    config = SimpleNamespace(
        clip_ratio=0.2,
        clip_ratio_low=None,
        clip_ratio_high=None,
        loss_agg_mode="token-sum",
        global_batch_info={
            "dp_size": 1,
            "batch_num_tokens": None,
            "global_batch_size": None,
            "loss_scale_factor": None,
        },
    )
    loss, _ = _opd_candidate_loss(
        config=config,
        opd_config=None,
        log_probs=current.view(1, n, width),
        old_log_probs=old.view(1, n, width),
        advantages=advantages.view(1, n, width),
        cand_mask=cand_mask.view(1, n, width),
        response_mask=torch.ones(1, n, dtype=torch.bool),
    )
    (gradient,) = torch.autograd.grad(loss, logits)

    masked_advantages = advantages * cand_mask
    probs = logits.detach().softmax(-1)
    expected = probs * masked_advantages.sum(-1, keepdim=True)
    expected.scatter_add_(-1, candidate_ids, -masked_advantages)
    torch.testing.assert_close(gradient, expected, atol=1e-10, rtol=1e-8)


def test_normalized_candidate_loss_matches_conditional_reverse_kl_gradient():
    """With is_norm, the candidate PPO score is exactly grad KL(p_s(.|C)||p_t(.|C))."""
    n, vocab, width = 4, 9, 3
    logits = torch.randn(
        n, vocab, generator=torch.Generator().manual_seed(789), dtype=torch.float64, requires_grad=True
    )
    teacher_logits = torch.randn(n, vocab, generator=torch.Generator().manual_seed(987), dtype=torch.float64)
    teacher_log_probs = teacher_logits.log_softmax(-1)
    candidate_ids = torch.tensor([[0, 3, 7], [1, 4, 8], [2, 3, 5], [0, 6, 8]])
    cand_mask = torch.tensor([[[True, True, True], [True, False, True], [True, False, False], [False, False, False]]])

    current = sparse_log_probs(logits, candidate_ids=candidate_ids, chunk_size=2)["cand_log_probs"]
    old = current.detach().clone()
    teacher = teacher_log_probs.gather(-1, candidate_ids).detach()
    advantage = opd_advantage(
        SimpleNamespace(top_k=width, is_norm=True, reward_weight_mode="student_p"),
        student_log_probs=old.view(1, n, width),
        teacher_log_probs=teacher.view(1, n, width),
        cand_mask=cand_mask,
        response_mask=torch.ones(1, n),
    )["opd_advantages"]

    config = SimpleNamespace(
        clip_ratio=0.2,
        clip_ratio_low=None,
        clip_ratio_high=None,
        loss_agg_mode="token-sum",
        global_batch_info={
            "dp_size": 1,
            "batch_num_tokens": None,
            "global_batch_size": None,
            "loss_scale_factor": None,
        },
    )
    loss, _ = _opd_candidate_loss(
        config=config,
        opd_config=SimpleNamespace(top_k=width, is_norm=True),
        log_probs=current.view(1, n, width),
        old_log_probs=old.view(1, n, width),
        advantages=advantage,
        cand_mask=cand_mask,
        response_mask=torch.ones(1, n, dtype=torch.bool),
    )
    (gradient,) = torch.autograd.grad(loss, logits, retain_graph=True)

    conditional_kl = logits.sum() * 0.0
    for row in range(n):
        valid_ids = candidate_ids[row, cand_mask[0, row]]
        if valid_ids.numel() == 0:
            continue
        student_cond = logits[row, valid_ids].log_softmax(-1)
        teacher_cond = teacher_logits[row, valid_ids].log_softmax(-1)
        conditional_kl = conditional_kl + (student_cond.exp() * (student_cond - teacher_cond)).sum()
    torch.testing.assert_close(loss.detach(), conditional_kl.detach(), atol=1e-10, rtol=1e-8)
    (expected,) = torch.autograd.grad(conditional_kl, logits)
    torch.testing.assert_close(gradient, expected, atol=1e-10, rtol=1e-8)
    assert torch.isfinite(gradient).all()
    assert (gradient[-1] == 0).all()


def test_outside_bucket_loss_matches_coarsened_reverse_kl_gradient():
    """Selected ids plus their complement reproduce the coarse-vocabulary KL gradient."""
    n, vocab, width = 4, 9, 3
    logits = torch.randn(
        n, vocab, generator=torch.Generator().manual_seed(159), dtype=torch.float64, requires_grad=True
    )
    teacher_logits = torch.randn(n, vocab, generator=torch.Generator().manual_seed(265), dtype=torch.float64)
    teacher_log_probs = teacher_logits.log_softmax(-1)
    candidate_ids = torch.tensor([[0, 3, 7], [1, 4, 8], [2, 3, 5], [0, 6, 8]])
    cand_mask = torch.tensor([[True, True, True], [True, False, True], [True, True, False], [False, False, False]])

    current = sparse_log_probs(logits, candidate_ids=candidate_ids, chunk_size=2)["cand_log_probs"]
    old = current.detach().clone()
    teacher = teacher_log_probs.gather(-1, candidate_ids).detach()
    advantage_out = opd_advantage(
        SimpleNamespace(top_k=width, is_norm=False, reward_weight_mode="student_p", outside_bucket=True),
        student_log_probs=old.view(1, n, width),
        teacher_log_probs=teacher.view(1, n, width),
        cand_mask=cand_mask.view(1, n, width),
        response_mask=torch.ones(1, n),
    )

    config = SimpleNamespace(
        clip_ratio=0.2,
        clip_ratio_low=None,
        clip_ratio_high=None,
        loss_agg_mode="token-sum",
        global_batch_info={
            "dp_size": 1,
            "batch_num_tokens": None,
            "global_batch_size": None,
            "loss_scale_factor": None,
        },
    )
    loss, _ = _opd_candidate_loss(
        config=config,
        opd_config=SimpleNamespace(top_k=width, is_norm=False),
        log_probs=current.view(1, n, width),
        old_log_probs=old.view(1, n, width),
        advantages=advantage_out["opd_advantages"],
        cand_mask=cand_mask.view(1, n, width),
        response_mask=torch.ones(1, n, dtype=torch.bool),
        outside_old_log_probs=advantage_out["opd_outside_old_log_probs"],
        outside_advantages=advantage_out["opd_outside_advantages"],
    )
    (gradient,) = torch.autograd.grad(loss, logits, retain_graph=True)

    student_log_probs = logits.log_softmax(-1)
    coarse_kl = logits.sum() * 0.0
    per_row_kl = []
    for row in range(n):
        valid_ids = candidate_ids[row, cand_mask[row]]
        student_inside = student_log_probs[row, valid_ids]
        teacher_inside = teacher_log_probs[row, valid_ids]
        inside_kl = (student_inside.exp() * (student_inside - teacher_inside)).sum()
        student_outside = 1.0 - student_inside.exp().sum()
        teacher_outside = 1.0 - teacher_inside.exp().sum()
        row_kl = inside_kl + student_outside * (student_outside.log() - teacher_outside.log())
        coarse_kl = coarse_kl + row_kl
        per_row_kl.append(row_kl.detach())

    torch.testing.assert_close(loss.detach(), coarse_kl.detach(), atol=1e-10, rtol=1e-8)
    torch.testing.assert_close(
        advantage_out["opd_token_rewards"].squeeze(0),
        -torch.stack(per_row_kl),
        atol=1e-10,
        rtol=1e-8,
    )
    (expected,) = torch.autograd.grad(coarse_kl, logits)
    torch.testing.assert_close(gradient, expected, atol=1e-10, rtol=1e-8)
    assert torch.isfinite(gradient).all()


def test_v1_candidate_restore_ignores_wider_bucket_metadata():
    response_mask = torch.nested.as_nested_tensor(
        [torch.ones(2, dtype=torch.bool), torch.ones(1, dtype=torch.bool)], layout=torch.jagged
    )
    # Full sequence lengths are 5 and 3. As in the engine, rows are already rolled
    # left so the response predictions occupy [-resp_len-1:-1].
    values = torch.arange(8 * 2, dtype=torch.float32).reshape(8, 2)
    model_output = torch.nested.nested_tensor_from_jagged(values, offsets=torch.tensor([0, 5, 8]))
    data = TensorDict(
        {
            "response_mask": response_mask,
            "max_response_len": NonTensorData(32),
        },
        batch_size=[2],
    )

    restored = _candidate_log_probs_to_response(model_output, data)
    assert restored.shape == (2, 2, 2)
    torch.testing.assert_close(restored[0], values[2:4])
    torch.testing.assert_close(restored[1, 0], values[6])
    assert (restored[1, 1] == 0).all()


def test_response_width_discards_transfer_queue_bucket_padding():
    bucketed = torch.arange(2 * 32 * 3).reshape(2, 32, 3)
    aligned = _to_response_width(bucketed, target_length=16)
    assert aligned.shape == (2, 16, 3)
    torch.testing.assert_close(aligned, bucketed[:, :16])

    with pytest.raises(ValueError, match="shorter than candidate"):
        _to_response_width(bucketed[:, :8], target_length=16)
