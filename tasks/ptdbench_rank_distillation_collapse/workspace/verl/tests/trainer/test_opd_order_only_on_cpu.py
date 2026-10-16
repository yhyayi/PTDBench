"""Numerical contracts for O3PD and its matched-candidate JSD baseline."""

import math
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.opd.kernels import sparse_log_probs
from verl.trainer.opd.losses import opd_direct_topk_loss
from verl.trainer.opd.order_only import common_token_ids, direct_topk_loss, ordered_topk, use_fp32_lm_head
from verl.trainer.ppo.core_algos import agg_loss
from verl.utils import tensordict_utils as tu
from verl.workers.config.opd import OPDConfig
from verl.workers.utils.padding import response_from_nested, response_to_full_nested


def config(**kwargs):
    return OPDConfig(enable=True, teacher_model_path="unused", **kwargs)


def test_probability_formula_and_sparse_gradient():
    z = torch.randn(3, 17, dtype=torch.float64, generator=torch.Generator().manual_seed(1), requires_grad=True)
    ids = torch.tensor([[2, 6, 4, 8], [1, 9, 7, 4], [0, 8, 5, 3]])
    logp = sparse_log_probs(z, candidate_ids=ids, chunk_size=2)["cand_log_probs"]
    actual = direct_topk_loss(logp, "adjacent_logistic")
    p = z.softmax(-1).gather(-1, ids)
    reference = -(p[:, :-1] / (p[:, :-1] + p[:, 1:])).log().mean(-1)
    torch.testing.assert_close(actual, reference)
    grad = torch.autograd.grad(actual.sum(), z, retain_graph=True)[0]
    expected = torch.autograd.grad(reference.sum(), z)[0]
    torch.testing.assert_close(grad, expected)
    outside = torch.ones_like(z, dtype=torch.bool).scatter_(-1, ids, False)
    torch.testing.assert_close(grad[outside], torch.zeros_like(grad[outside]), atol=1e-15, rtol=0)


def test_equal_reversed_and_extreme_pairs():
    scores = torch.tensor([[0., 0.], [-2., 2.], [10000., -10000.]], requires_grad=True)
    loss = direct_topk_loss(scores, "adjacent_logistic")
    torch.testing.assert_close(loss[0], torch.tensor(math.log(2)))
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    assert torch.isfinite(scores.grad).all()
    updated = scores.detach() - 0.1 * scores.grad
    assert (updated[1, 0] - updated[1, 1]) > (scores[1, 0] - scores[1, 1])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("k", [2, 3, 8])
def test_topk_ties_matches_stable_oracle(dtype, k):
    # Many interior/boundary ties, noncontiguous valid IDs, invalid head-tail rows dominate.
    logits = torch.randint(-2, 3, (31, 21), generator=torch.Generator().manual_seed(0)).to(dtype)
    logits[:, -1] = 100
    valid = torch.tensor([0, 2, 4, 5, 6, 9, 12, 15])
    values, ids = ordered_topk(logits, valid, k, 7)
    rank = logits[:, valid].argsort(dim=-1, descending=True, stable=True)[:, :k]
    torch.testing.assert_close(ids, valid[rank])
    torch.testing.assert_close(values, logits.gather(-1, ids))


def test_teacher_temperature_and_monotone_invariance():
    teacher = torch.randn(4, 19, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
    student = torch.randn(4, 19, dtype=torch.float64)
    valid = torch.arange(19)
    _, original = ordered_topk(teacher, valid, 8, 3)
    expected = direct_topk_loss(student.gather(-1, original), "adjacent_logistic")
    for transformed in [teacher / t for t in (0.25, 0.5, 1, 2, 4, 8)] + [teacher.exp(), teacher**3]:
        _, ids = ordered_topk(transformed, valid, 8, 3)
        torch.testing.assert_close(ids, original)
        torch.testing.assert_close(direct_topk_loss(student.gather(-1, ids), "adjacent_logistic"), expected)


def test_jsd_reference_gradients_and_detached_teacher():
    s = torch.randn(5, 8, dtype=torch.float64, requires_grad=True)
    t = torch.randn(5, 8, dtype=torch.float64, requires_grad=True)
    actual = direct_topk_loss(s, "topk_jsd", t)
    p, q = s.softmax(-1), t.detach().softmax(-1)
    m = (p + q) / 2
    expected = ((p * (p.log() - m.log())).sum(-1) + (q * (q.log() - m.log())).sum(-1)) / 2
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), s, retain_graph=True)[0], torch.autograd.grad(expected.sum(), s)[0]
    )
    assert torch.autograd.grad(actual.sum(), t, allow_unused=True)[0] is None
    assert (actual >= 0).all()
    torch.testing.assert_close(direct_topk_loss(s, "topk_jsd", s), torch.zeros(5, dtype=s.dtype), atol=1e-15, rtol=0)
    assert not torch.allclose(actual, direct_topk_loss(s, "topk_jsd", t / 4))


def test_global_sequence_mean_microbatch_and_dp_equivalence():
    scores = torch.randn(5, 7, 8, dtype=torch.float64, requires_grad=True)
    mask = torch.arange(7)[None, :] < torch.tensor([1, 7, 3, 0, 5])[:, None]
    token = direct_topk_loss(scores, "adjacent_logistic")
    expected = torch.stack([token[b, mask[b]].mean() for b in [0, 1, 2, 4]]).mean()
    accumulated = sum(
        agg_loss(token[a:b], mask[a:b], "seq-mean-token-mean", global_batch_size=4, dp_size=2)
        for a, b in [(0, 1), (1, 3), (3, 5)]
    ) / 2  # FSDP's DP gradient averaging
    torch.testing.assert_close(accumulated, expected)
    torch.testing.assert_close(
        torch.autograd.grad(accumulated, scores, retain_graph=True)[0], torch.autograd.grad(expected, scores)[0]
    )


def test_tokenizer_compatible_with_different_padded_heads():
    def tokenizer(vocab):
        return SimpleNamespace(
            get_vocab=lambda: vocab, bos_token_id=0, eos_token_id=3, pad_token_id=0, all_special_ids=[0, 3]
        )
    tok = tokenizer({"a": 0, "b": 1, "end": 3})
    assert common_token_ids(tok, tok, 4, 9) == [0, 1, 3]
    with pytest.raises(ValueError, match="token-to-ID"):
        common_token_ids(tok, tokenizer({"a": 1, "b": 0, "end": 3}), 4, 9)
    with pytest.raises(ValueError, match="LM-head"):
        common_token_ids(tok, tok, 3, 9)


@pytest.mark.parametrize("objective", ["adjacent_logistic", "topk_jsd"])
def test_packed_alignment_and_actor_loss(objective):
    input_ids = torch.nested.as_nested_tensor([torch.arange(5), torch.arange(7)], layout=torch.jagged)
    response_mask = torch.nested.as_nested_tensor([torch.ones(2), torch.ones(3)], layout=torch.jagged)
    ids = torch.nested.as_nested_tensor(
        [torch.tensor([[4, 2], [3, 1]]), torch.tensor([[1, 3], [2, 4], [0, 3]])], layout=torch.jagged
    )
    full_ids = response_to_full_nested(ids, input_ids=input_ids, response_mask=response_mask)
    # Engine shifts token-framed candidate IDs onto preceding prediction rows.
    prediction_ids = full_ids.values().roll(-1, 0)
    torch.testing.assert_close(prediction_ids[2], ids[0][0])
    torch.testing.assert_close(prediction_ids[8], ids[1][0])
    z = torch.randn(12, 6, requires_grad=True)
    gathered = sparse_log_probs(z, candidate_ids=prediction_ids)["cand_log_probs"]
    # Outputs remain prediction-framed; response_from_nested applies the causal offset.
    output = torch.nested.nested_tensor_from_jagged(gathered, input_ids.offsets())
    aligned = response_from_nested(output, response_mask)
    torch.testing.assert_close(aligned[0][0], z[2].log_softmax(-1)[ids[0][0]])
    torch.testing.assert_close(aligned[1][0], z[8].log_softmax(-1)[ids[1][0]])
    data = TensorDict({"response_mask": response_mask}, batch_size=2)
    tu.assign_non_tensor(data, dp_size=1, opd_valid_batch_size=2)
    if objective == "topk_jsd":
        data["opd_teacher_topk_log_probs"] = torch.nested.as_nested_tensor(
            [torch.randn(2, 2), torch.randn(3, 2)], layout=torch.jagged
        )
    cfg = config(objective=objective, top_k=2, strategy="only_tch")
    actual, metrics = opd_direct_topk_loss(None, cfg, {"opd_cand_log_probs": output}, data)
    response = response_from_nested(output, response_mask)
    expected = torch.stack([
        direct_topk_loss(
            response[b], objective,
            data["opd_teacher_topk_log_probs"][b] if objective == "topk_jsd" else None,
        ).mean() for b in range(2)
    ]).mean()
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(z.grad).all()
    assert "actor/opd_adjacent_accuracy@num" in metrics


def test_masked_nan_and_empty_actor_loss():
    data = TensorDict({"response_mask": torch.zeros(2, 3)}, batch_size=2)
    tu.assign_non_tensor(data, dp_size=1, opd_valid_batch_size=1)
    # Test the padded adapter independently of the packed forward.
    from unittest.mock import patch
    scores = torch.full((2, 3, 2), float("nan"), requires_grad=True)
    with patch("verl.trainer.opd.losses._candidate_log_probs_to_response", return_value=scores):
        loss, _ = opd_direct_topk_loss(
            None, config(objective="adjacent_logistic", top_k=2, strategy="only_tch"),
            {"opd_cand_log_probs": scores}, data,
        )
    assert loss.item() == 0
    loss.backward()
    assert (scores.grad == 0).all()


@pytest.mark.parametrize("kwargs", [
    {"top_k": 1, "strategy": "only_tch"},
    {"top_k": 8, "strategy": "union"},
    {"top_k": 8, "strategy": "only_tch", "teacher_ref_model_path": "ref"},
    {"top_k": 8, "strategy": "only_tch", "loss_coef": float("nan")},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        config(objective="adjacent_logistic", **kwargs)


def test_tie_masking_drops_only_tied_teacher_pairs():
    student = torch.randn(4, 5, 8, dtype=torch.float64, requires_grad=True)
    teacher = torch.tensor([9., 8., 8., 7., 5., 5., 5., 1.], dtype=torch.float64).expand(4, 5, 8)
    actual = direct_topk_loss(student, "adjacent_logistic", teacher, mask_teacher_ties=True)
    keep = torch.tensor([1., 0., 1., 1., 0., 0., 1.], dtype=torch.float64)
    expected = (torch.nn.functional.softplus(student[..., 1:] - student[..., :-1]) * keep).sum(-1) / 7
    torch.testing.assert_close(actual, expected)
    grad = torch.autograd.grad(actual.sum(), student)[0]
    # Candidate 5 appears only in the tied pairs (4, 5) and (5, 6).
    assert (grad[..., 5] == 0).all()
    distinct = torch.arange(8, 0, -1, dtype=torch.float64).expand(4, 5, 8)
    torch.testing.assert_close(
        direct_topk_loss(student, "adjacent_logistic", distinct, mask_teacher_ties=True),
        direct_topk_loss(student, "adjacent_logistic"),
    )
    with pytest.raises(ValueError, match="teacher scores"):
        direct_topk_loss(student, "adjacent_logistic", mask_teacher_ties=True)


def test_tie_masking_through_actor_loss():
    response_mask = torch.nested.as_nested_tensor([torch.ones(2), torch.ones(3)], layout=torch.jagged)
    scores = torch.randn(2, 3, 4, requires_grad=True)
    teacher = torch.nested.as_nested_tensor(
        [torch.tensor([[3., 3., 2., 1.], [4., 3., 2., 1.]]), torch.tensor([[1., 1., 1., 0.]] * 3)],
        layout=torch.jagged,
    )
    data = TensorDict({"response_mask": response_mask, "opd_teacher_topk_log_probs": teacher}, batch_size=2)
    tu.assign_non_tensor(data, dp_size=1, opd_valid_batch_size=2)
    from unittest.mock import patch
    cfg = config(objective="adjacent_logistic", top_k=4, strategy="only_tch", mask_teacher_ties=True)
    with patch("verl.trainer.opd.losses._candidate_log_probs_to_response", return_value=scores):
        actual, _ = opd_direct_topk_loss(None, cfg, {"opd_cand_log_probs": scores}, data)
    padded = teacher.to_padded_tensor(0)
    per_token = direct_topk_loss(scores, "adjacent_logistic", padded, mask_teacher_ties=True)
    expected = (per_token[0, :2].mean() + per_token[1, :3].mean()) / 2
    torch.testing.assert_close(actual, expected)


def test_tie_and_fp32_head_configuration():
    assert not config(objective="adjacent_logistic", top_k=8, strategy="only_tch").mask_teacher_ties
    config(objective="adjacent_logistic", top_k=8, strategy="only_tch", mask_teacher_ties=True, teacher_fp32_head=True)
    config(objective="topk_jsd", top_k=8, strategy="only_tch", teacher_fp32_head=True)
    with pytest.raises(ValueError, match="mask_teacher_ties"):
        config(objective="topk_jsd", top_k=8, strategy="only_tch", mask_teacher_ties=True)
    with pytest.raises(ValueError, match="teacher_fp32_head"):
        config(objective="legacy", top_k=8, strategy="only_tch", teacher_fp32_head=True)


def test_fp32_lm_head_is_exact_product_under_bf16_autocast():
    torch.manual_seed(0)

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Linear(16, 32, bias=False)
            self.lm_head = torch.nn.Linear(32, 64, bias=False)

        def forward(self, x):
            return self.lm_head(self.model(x))

    net = Tiny().to(torch.bfloat16)
    x = torch.randn(10, 16).to(torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        before = net(x)
    assert before.dtype == torch.bfloat16
    use_fp32_lm_head(net)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        after = net(x)
        hidden = net.model(x)
    assert after.dtype == torch.float32
    reference = hidden.double() @ net.lm_head.weight.double().t()
    torch.testing.assert_close(after.double(), reference, rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError, match="lm_head"):
        use_fp32_lm_head(torch.nn.Sequential(torch.nn.Linear(2, 2)))
