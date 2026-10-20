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
"""CPU tests for the colocated OPD advantage.

The load-bearing property is that full-vocabulary OPD equals the exact reverse-KL
gradient, while strict top-k sets deliver the specified sparse policy-gradient
signal. Strategy membership and masking are checked against a brute-force oracle.
"""

import pytest
import torch

from verl.trainer.opd.advantage import (
    binned_mean_sums,
    bucket_mean_sums,
    build_opd_candidates,
    direct_opd_metrics,
    opd_advantage,
    opd_kl_bucket_metrics,
    opd_topk_overlap_ratio,
    resolve_bucket_means,
)
from verl.workers.config.opd import OPDConfig


def _config(**kwargs) -> OPDConfig:
    kwargs.setdefault("enable", True)
    kwargs.setdefault("teacher_model_path", "/dev/null")
    return OPDConfig(**kwargs)


def _logits(shape, seed):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


@pytest.mark.parametrize("is_norm", [False, True])
def test_full_vocab_matches_exact_reverse_kl_gradient(is_norm):
    """With the candidate set = the whole vocab, OPD reproduces ``grad KL(p_s||p_t)``."""
    b, t, v = 2, 3, 7
    s_logits = _logits((b, t, v), 0).requires_grad_(True)
    t_log_probs = _logits((b, t, v), 1).log_softmax(-1)
    response_mask = torch.ones(b, t)

    s_log_probs = s_logits.log_softmax(-1)
    out = opd_advantage(
        _config(top_k=v, is_norm=is_norm),
        student_log_probs=s_log_probs,
        teacher_log_probs=t_log_probs,
        cand_mask=torch.ones(b, t, v, dtype=torch.bool),
        response_mask=response_mask,
    )
    # The PPO surrogate at ratio == 1 is -sum_k A_k * log p_s(k); its gradient is the update.
    surrogate = -(out["opd_advantages"].detach() * s_log_probs).sum()
    (opd_grad,) = torch.autograd.grad(surrogate, s_logits, retain_graph=True)

    kl = (s_log_probs.exp() * (s_log_probs - t_log_probs)).sum(-1).sum()
    (exact_grad,) = torch.autograd.grad(kl, s_logits)
    torch.testing.assert_close(opd_grad, exact_grad, atol=1e-10, rtol=1e-8)

    # The token reward is exactly the negative KL, i.e. "KL in reward".
    exact_kl = (s_log_probs.exp() * (s_log_probs - t_log_probs)).sum(-1)
    torch.testing.assert_close(out["opd_token_rewards"], -exact_kl.detach(), atol=1e-10, rtol=1e-8)


def test_masked_subset_uses_full_vocab_probability_when_is_norm_disabled():
    """With is_norm disabled, student_p uses raw full-vocab probabilities."""
    b, t, v = 2, 3, 8
    s_log_probs = _logits((b, t, v), 2).log_softmax(-1)
    t_log_probs = _logits((b, t, v), 3).log_softmax(-1)
    mask = torch.zeros(b, t, v, dtype=torch.bool)
    mask[..., :4] = True
    mask[0, 0, 3] = False  # ragged candidate counts across positions

    out = opd_advantage(
        _config(top_k=4),
        student_log_probs=s_log_probs,
        teacher_log_probs=t_log_probs,
        cand_mask=mask,
        response_mask=torch.ones(b, t),
    )
    kl = s_log_probs - t_log_probs
    probs = s_log_probs.exp() * mask
    weights = probs
    torch.testing.assert_close(out["opd_advantages"], -(weights * kl * mask))
    # Nothing leaks in from outside the candidate set.
    assert (out["opd_advantages"][~mask] == 0).all()
    torch.testing.assert_close(out["opd_token_rewards"], -(weights * kl * mask).sum(-1))


def test_outside_bucket_adds_complement_mass_term():
    student = torch.tensor([[[0.5, 0.3, 0.2]]], dtype=torch.float64).log()
    teacher = torch.tensor([[[0.4, 0.35, 0.25]]], dtype=torch.float64).log()
    valid = torch.tensor([[[True, True, False]]])

    out = opd_advantage(
        _config(top_k=2, outside_bucket=True),
        student_log_probs=student,
        teacher_log_probs=teacher,
        cand_mask=valid,
        response_mask=torch.ones(1, 1),
    )

    inside = (student[..., :2].exp() * (student[..., :2] - teacher[..., :2])).sum(-1)
    outside = torch.tensor([[0.2 * torch.log(torch.tensor(0.2 / 0.25, dtype=torch.float64))]])
    torch.testing.assert_close(out["opd_outside_student_mass"], torch.tensor([[0.2]], dtype=torch.float64))
    torch.testing.assert_close(out["opd_outside_teacher_mass"], torch.tensor([[0.25]], dtype=torch.float64))
    torch.testing.assert_close(out["opd_outside_kl"], outside)
    torch.testing.assert_close(out["opd_token_rewards"], -(inside + outside))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"top_k": 0}, "top_k > 0"),
        ({"top_k": 2, "is_norm": True}, "is_norm=False"),
        ({"top_k": 2, "reward_weight_mode": "teacher_p"}, "reward_weight_mode='student_p'"),
    ],
)
def test_outside_bucket_rejects_incompatible_config(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _config(outside_bucket=True, **kwargs)


def test_outside_bucket_is_zero_for_full_vocabulary():
    b, t, v = 1, 2, 5
    student = _logits((b, t, v), 20).log_softmax(-1)
    teacher = _logits((b, t, v), 21).log_softmax(-1)
    out = opd_advantage(
        _config(top_k=v, outside_bucket=True),
        student_log_probs=student,
        teacher_log_probs=teacher,
        cand_mask=torch.ones(b, t, v, dtype=torch.bool),
        response_mask=torch.ones(b, t),
    )
    assert torch.isfinite(out["opd_outside_advantages"]).all()
    assert (out["opd_outside_advantages"] == 0).all()


def test_is_norm_renormalizes_log_ratio_over_top_k_candidates_only():
    """`is_norm` changes the log-ratio to candidate-conditioned probabilities."""
    student = torch.tensor([[[-0.2, -1.1, -2.3, -3.7]]], dtype=torch.float64)
    teacher = torch.tensor([[[-1.4, -0.4, -2.0, -3.2]]], dtype=torch.float64)
    valid = torch.tensor([[[True, True, False, False]]])
    response_mask = torch.ones(1, 1)

    out = opd_advantage(
        _config(top_k=2, is_norm=True),
        student_log_probs=student,
        teacher_log_probs=teacher,
        cand_mask=valid,
        response_mask=response_mask,
    )
    s_norm = student[..., :2] - torch.logsumexp(student[..., :2], dim=-1, keepdim=True)
    t_norm = teacher[..., :2] - torch.logsumexp(teacher[..., :2], dim=-1, keepdim=True)
    weights = torch.softmax(student[..., :2], dim=-1)
    expected = -(weights * (s_norm - t_norm))
    expected = torch.cat([expected, torch.zeros(1, 1, 2, dtype=torch.float64)], dim=-1)
    torch.testing.assert_close(out["opd_advantages"], expected)
    # Diagnostics continue to expose the original full-vocabulary candidate mass.
    torch.testing.assert_close(out["opd_student_mass"], student[..., :2].exp().sum(-1))
    torch.testing.assert_close(out["opd_teacher_mass"], teacher[..., :2].exp().sum(-1))


def test_is_norm_controls_teacher_probability_weights():
    student = torch.tensor([[[-0.3, -1.2, -2.0]]], dtype=torch.float64)
    teacher = torch.tensor([[[-1.5, -0.4, -1.8]]], dtype=torch.float64)
    valid = torch.tensor([[[True, True, False]]])
    kwargs = dict(
        student_log_probs=student,
        teacher_log_probs=teacher,
        cand_mask=valid,
        response_mask=torch.ones(1, 1),
    )

    raw = opd_advantage(_config(top_k=2, reward_weight_mode="teacher_p", is_norm=False), **kwargs)
    normalized = opd_advantage(_config(top_k=2, reward_weight_mode="teacher_p", is_norm=True), **kwargs)

    raw_expected = -(teacher.exp() * (student - teacher) * valid).sum(-1)
    s_norm = student[..., :2] - torch.logsumexp(student[..., :2], dim=-1, keepdim=True)
    t_norm = teacher[..., :2] - torch.logsumexp(teacher[..., :2], dim=-1, keepdim=True)
    norm_expected = -(torch.softmax(teacher[..., :2], dim=-1) * (s_norm - t_norm)).sum(-1)
    torch.testing.assert_close(raw["opd_token_rewards"], raw_expected)
    torch.testing.assert_close(normalized["opd_token_rewards"], norm_expected)


def test_is_norm_is_ignored_for_sampled_token_opd():
    """The sampled-token path keeps its original log-ratio regardless of `is_norm`."""
    student = torch.tensor([[[-0.2]]], dtype=torch.float64)
    teacher = torch.tensor([[[-1.4]]], dtype=torch.float64)
    valid = torch.ones(1, 1, 1, dtype=torch.bool)
    kwargs = dict(
        student_log_probs=student,
        teacher_log_probs=teacher,
        cand_mask=valid,
        response_mask=torch.ones(1, 1),
    )
    plain = opd_advantage(_config(top_k=0, is_norm=False), **kwargs)
    normalized = opd_advantage(_config(top_k=0, is_norm=True), **kwargs)
    torch.testing.assert_close(normalized["opd_advantages"], plain["opd_advantages"])
    torch.testing.assert_close(normalized["opd_token_rewards"], plain["opd_token_rewards"])


def test_sampled_adv_clip_clamps_extremes_and_keeps_raw_kl():
    student = torch.tensor([[[-0.1], [-8.0], [-1.0], [-0.5]]], dtype=torch.float64)
    teacher = torch.tensor([[[-10.0], [-0.1], [-1.5], [-30.0]]], dtype=torch.float64)
    response_mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    out = opd_advantage(
        _config(top_k=0, sampled_adv_clip_min=-5.0, sampled_adv_clip_max=3.0),
        student_log_probs=student,
        teacher_log_probs=teacher,
        cand_mask=torch.ones(1, 4, 1, dtype=torch.bool),
        response_mask=response_mask,
    )
    expected = torch.tensor([[-5.0, 3.0, -0.5, 0.0]], dtype=torch.float64)
    torch.testing.assert_close(out["opd_token_rewards"], expected)
    torch.testing.assert_close(out["opd_advantages"].squeeze(-1), expected)
    raw_kl = (student - teacher).squeeze(-1) * response_mask
    torch.testing.assert_close(out["opd_kl"], raw_kl)
    torch.testing.assert_close(out["opd_adv_clipped_low"], torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64))
    torch.testing.assert_close(out["opd_adv_clipped_high"], torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float64))


def test_sampled_adv_clip_one_sided_and_ignored_for_top_k():
    student = torch.tensor([[[-0.1], [-8.0]]], dtype=torch.float64)
    teacher = torch.tensor([[[-10.0], [-0.1]]], dtype=torch.float64)
    kwargs = dict(
        student_log_probs=student,
        teacher_log_probs=teacher,
        cand_mask=torch.ones(1, 2, 1, dtype=torch.bool),
        response_mask=torch.ones(1, 2),
    )
    low_only = opd_advantage(_config(top_k=0, sampled_adv_clip_min=-5.0), **kwargs)
    torch.testing.assert_close(low_only["opd_token_rewards"], torch.tensor([[-5.0, 7.9]], dtype=torch.float64))
    assert (low_only["opd_adv_clipped_high"] == 0).all()

    plain = opd_advantage(_config(top_k=1, reward_weight_mode="none"), **kwargs)
    clipped_cfg = opd_advantage(
        _config(top_k=1, reward_weight_mode="none", sampled_adv_clip_min=-1.0, sampled_adv_clip_max=1.0), **kwargs
    )
    torch.testing.assert_close(clipped_cfg["opd_advantages"], plain["opd_advantages"])
    assert "opd_adv_clipped_low" not in clipped_cfg


def test_direct_opd_uses_teacher_shift_and_keeps_student_kl():
    """Direct-OPD's advantage is log p_t - log p_t_ref; `opd_kl` still reports log p_s - log p_t."""
    responses = torch.tensor([[3, 5, 7]])
    old_log_probs = torch.tensor([[-0.3, -1.0, -2.0]], dtype=torch.float64)
    teacher = torch.tensor([[-0.5, -0.2, -9.0]], dtype=torch.float64)
    teacher_ref = torch.tensor([[-1.5, -0.4, -1.0]], dtype=torch.float64)
    response_mask = torch.tensor([[1.0, 1.0, 0.0]])
    cfg = _config(top_k=0, teacher_ref_model_path="/dev/null")

    cands = build_opd_candidates(
        cfg,
        responses=responses,
        old_log_probs=old_log_probs,
        t_on_s_log_probs=teacher,
        t_ref_on_s_log_probs=teacher_ref,
    )
    out = opd_advantage(cfg, response_mask=response_mask, **cands)

    shift = (teacher - teacher_ref) * response_mask
    torch.testing.assert_close(out["opd_token_rewards"], shift)
    torch.testing.assert_close(out["opd_advantages"].squeeze(-1), shift)
    torch.testing.assert_close(out["opd_teacher_shift"], shift)
    torch.testing.assert_close(out["opd_kl"], (old_log_probs - teacher) * response_mask)
    kl_to_teacher_ref = (old_log_probs - teacher_ref) * response_mask
    torch.testing.assert_close(out["opd_kl_to_teacher_ref"], kl_to_teacher_ref)
    torch.testing.assert_close(out["opd_kl_to_teacher_ref"], out["opd_kl"] + out["opd_teacher_shift"])

    clipped = opd_advantage(
        _config(top_k=0, teacher_ref_model_path="/dev/null", sampled_adv_clip_min=-0.5, sampled_adv_clip_max=0.5),
        response_mask=response_mask,
        **cands,
    )
    torch.testing.assert_close(clipped["opd_token_rewards"], torch.tensor([[0.5, 0.2, 0.0]], dtype=torch.float64))
    torch.testing.assert_close(clipped["opd_teacher_shift"], shift)
    torch.testing.assert_close(clipped["opd_kl_to_teacher_ref"], kl_to_teacher_ref)


def test_direct_opd_metrics_report_teacher_shift_and_entropies():
    responses = torch.tensor([[3, 5, 7, 9], [2, 4, 6, 8], [1, 1, 1, 1]])
    old_log_probs = torch.full((3, 4), -1.0, dtype=torch.float64)
    teacher = torch.tensor(
        [[-0.5, -0.2, -1.0, -2.0], [-0.1, -3.0, -0.3, -0.7], [-1.0, -1.0, -1.0, -1.0]], dtype=torch.float64
    )
    teacher_ref = torch.tensor(
        [[-1.5, -0.4, -0.5, -9.0], [-0.2, -1.0, -0.3, -0.1], [-2.0, -2.0, -2.0, -2.0]], dtype=torch.float64
    )
    # The last row is an all-masked (padding) sample and must not dilute per-response means.
    response_mask = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    cfg = _config(top_k=0, teacher_ref_model_path="/dev/null")
    cands = build_opd_candidates(
        cfg,
        responses=responses,
        old_log_probs=old_log_probs,
        t_on_s_log_probs=teacher,
        t_ref_on_s_log_probs=teacher_ref,
    )
    out = opd_advantage(cfg, response_mask=response_mask, **cands)
    teacher_entropy = torch.tensor([[1.0, 2.0, 3.0, 99.0], [4.0, 5.0, 99.0, 99.0], [99.0] * 4])
    teacher_ref_entropy = torch.tensor([[0.5, 1.0, 1.5, 0.0], [2.0, 2.5, 0.0, 0.0], [0.0] * 4])
    metrics = direct_opd_metrics(out, response_mask, teacher_entropy, teacher_ref_entropy, bucket_size=2)

    mask = response_mask.bool()
    shift = (teacher - teacher_ref)[mask]
    assert metrics["opd/teacher_shift_mean"] == pytest.approx(shift.mean().item())
    assert metrics["opd/teacher_shift_std"] == pytest.approx(shift.std(unbiased=False).item())
    assert metrics["opd/teacher_shift_abs_mean"] == pytest.approx(shift.abs().mean().item())
    assert metrics["opd/teacher_shift_pos_frac"] == pytest.approx((shift > 0).double().mean().item())
    seq_sums = ((teacher - teacher_ref) * response_mask).sum(dim=-1)[:2]
    assert metrics["opd/teacher_shift_seq_sum_mean"] == pytest.approx(seq_sums.mean().item())
    assert metrics["opd/kl_to_teacher_ref"] == pytest.approx((old_log_probs - teacher_ref)[mask].mean().item())

    # Only row 0 position 2 is valid in [2, 4): shift -0.5 there, row 1 contributes 0 to the sum.
    assert metrics["opd/teacher_shift_by_position/2-4"] == pytest.approx(-0.5)
    assert metrics["opd/teacher_shift_pos_frac_by_position/2-4"] == pytest.approx(0.0)
    assert metrics["opd/teacher_shift_seq_sum_by_position/2-4"] == pytest.approx(-0.25)
    bucket_sums = [v for k, v in metrics.items() if k.startswith("opd/teacher_shift_seq_sum_by_position/")]
    assert len(bucket_sums) == 2
    assert sum(bucket_sums) == pytest.approx(metrics["opd/teacher_shift_seq_sum_mean"])

    # Rows 0 and 1 have lengths 3 and 2, both in [2, 4); each contributes its own mean shift.
    resp_means = seq_sums / response_mask[:2].sum(dim=-1)
    assert metrics["opd/teacher_shift_by_seqlen/2-4"] == pytest.approx(resp_means.mean().item())
    assert "opd/teacher_shift_by_seqlen/0-2" not in metrics

    assert metrics["opd/teacher_entropy"] == pytest.approx(3.0)
    assert metrics["opd/teacher_ref_entropy"] == pytest.approx(1.5)
    assert metrics["opd/teacher_entropy_minus_ref"] == pytest.approx(1.5)

    assert direct_opd_metrics(out, torch.zeros_like(response_mask)) == {}
    assert "opd/teacher_entropy" not in direct_opd_metrics(out, response_mask)


def test_direct_opd_is_identical_to_sampled_opd_when_disabled():
    responses = torch.tensor([[3, 5]])
    old_log_probs = torch.tensor([[-0.3, -1.0]], dtype=torch.float64)
    teacher = torch.tensor([[-0.5, -0.2]], dtype=torch.float64)
    cands = build_opd_candidates(
        _config(top_k=0), responses=responses, old_log_probs=old_log_probs, t_on_s_log_probs=teacher
    )
    assert "teacher_ref_log_probs" not in cands
    out = opd_advantage(_config(top_k=0), response_mask=torch.ones(1, 2), **cands)
    torch.testing.assert_close(out["opd_token_rewards"], teacher - old_log_probs)
    assert "opd_teacher_shift" not in out
    assert "opd_kl_to_teacher_ref" not in out


def test_direct_opd_requires_sampled_token_mode():
    with pytest.raises(ValueError, match="only implemented for opd.top_k=0"):
        _config(top_k=4, teacher_ref_model_path="/dev/null")


def test_sampled_adv_clip_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="sampled_adv_clip_min must be <"):
        _config(top_k=0, sampled_adv_clip_min=1.0, sampled_adv_clip_max=-1.0)


@pytest.mark.parametrize("is_norm", [False, True])
def test_empty_candidate_set_is_zero_not_nan(is_norm):
    """``intersection`` can be empty at a position; that must be a no-op, not a NaN."""
    b, t, v = 1, 2, 5
    s_log_probs = _logits((b, t, v), 4).log_softmax(-1)
    t_log_probs = _logits((b, t, v), 5).log_softmax(-1)
    mask = torch.ones(b, t, v, dtype=torch.bool)
    mask[0, 1] = False

    out = opd_advantage(
        _config(top_k=v, strategy="intersection", is_norm=is_norm),
        student_log_probs=s_log_probs,
        teacher_log_probs=t_log_probs,
        cand_mask=mask,
        response_mask=torch.ones(b, t),
    )
    assert torch.isfinite(out["opd_advantages"]).all()
    assert (out["opd_advantages"][0, 1] == 0).all()
    assert out["opd_token_rewards"][0, 1] == 0


def test_response_mask_zeroes_padding_positions():
    b, t, v = 2, 4, 6
    s_log_probs = _logits((b, t, v), 6).log_softmax(-1)
    t_log_probs = _logits((b, t, v), 7).log_softmax(-1)
    response_mask = torch.ones(b, t)
    response_mask[:, 2:] = 0

    out = opd_advantage(
        _config(top_k=v),
        student_log_probs=s_log_probs,
        teacher_log_probs=t_log_probs,
        cand_mask=torch.ones(b, t, v, dtype=torch.bool),
        response_mask=response_mask,
    )
    assert (out["opd_advantages"][:, 2:] == 0).all()
    assert (out["opd_token_rewards"][:, 2:] == 0).all()
    assert not (out["opd_token_rewards"][:, :2] == 0).any()


def _oracle(b, t, v, k, seed):
    """Build a full-vocab student/teacher pair plus the exact per-strategy id sets."""
    s_log_probs = _logits((b, t, v), seed).log_softmax(-1)
    t_log_probs = _logits((b, t, v), seed + 100).log_softmax(-1)
    s_ids = s_log_probs.topk(k, dim=-1).indices
    t_ids = t_log_probs.topk(k, dim=-1).indices
    return s_log_probs, t_log_probs, s_ids, t_ids


def _gather(log_probs, ids):
    return log_probs.gather(-1, ids)


def test_topk_overlap_ratio_is_student_teacher_intersection_over_k():
    student_ids = torch.tensor([[[0, 1, 2], [3, 4, 5]]])
    teacher_ids = torch.tensor([[[2, 1, 7], [8, 9, 10]]])
    response_mask = torch.tensor([[1.0, 0.0]])

    ratio = opd_topk_overlap_ratio(student_ids, teacher_ids, response_mask)

    torch.testing.assert_close(ratio, torch.tensor([[2.0 / 3.0, 0.0]]))


@pytest.mark.parametrize("strategy", ["only_stu", "only_tch", "intersection", "union"])
def test_strategy_candidate_sets_match_brute_force(strategy):
    """The assembled (ids, mask) pair selects exactly the ids the strategy names."""
    b, t, v, k = 2, 3, 11, 4
    s_lp, t_lp, s_ids, t_ids = _oracle(b, t, v, k, seed=8)
    responses = torch.zeros(b, t, dtype=torch.int64)
    config = _config(top_k=k, strategy=strategy)

    cands = build_opd_candidates(
        config,
        responses=responses,
        old_log_probs=_gather(s_lp, responses.unsqueeze(-1)).squeeze(-1),
        s_topk_ids=s_ids,
        s_topk_log_probs=_gather(s_lp, s_ids),
        s_on_t_log_probs=_gather(s_lp, t_ids),
        t_on_s_log_probs=_gather(t_lp, s_ids),
        t_topk_ids=t_ids,
        t_topk_log_probs=_gather(t_lp, t_ids),
    )
    for i in range(b):
        for j in range(t):
            got = {int(x) for x, m in zip(cands["cand_ids"][i, j], cands["cand_mask"][i, j], strict=True) if m}
            s_set, t_set = {int(x) for x in s_ids[i, j]}, {int(x) for x in t_ids[i, j]}
            want = {
                "only_stu": s_set,
                "only_tch": t_set,
                "intersection": s_set & t_set,
                "union": s_set | t_set,
            }[strategy]
            assert got == want, f"{strategy} at ({i},{j}): {got} != {want}"
    # Log-probs must stay paired with their ids after the concat/dedupe in `union`.
    valid = cands["cand_mask"]
    torch.testing.assert_close(
        cands["student_log_probs"][valid], _gather(s_lp, cands["cand_ids"])[valid], atol=1e-12, rtol=1e-12
    )
    torch.testing.assert_close(
        cands["teacher_log_probs"][valid], _gather(t_lp, cands["cand_ids"])[valid], atol=1e-12, rtol=1e-12
    )


def test_full_k_makes_every_strategy_identical():
    """At ``k == vocab`` all four sets are the whole vocab, so they must agree exactly."""
    b, t, v = 2, 2, 6
    s_lp, t_lp, s_ids, t_ids = _oracle(b, t, v, v, seed=9)
    responses = torch.zeros(b, t, dtype=torch.int64)
    rewards = {}
    for strategy in ("only_stu", "only_tch", "intersection", "union"):
        cands = build_opd_candidates(
            _config(top_k=v, strategy=strategy),
            responses=responses,
            old_log_probs=_gather(s_lp, responses.unsqueeze(-1)).squeeze(-1),
            s_topk_ids=s_ids,
            s_topk_log_probs=_gather(s_lp, s_ids),
            s_on_t_log_probs=_gather(s_lp, t_ids),
            t_on_s_log_probs=_gather(t_lp, s_ids),
            t_topk_ids=t_ids,
            t_topk_log_probs=_gather(t_lp, t_ids),
        )
        out = opd_advantage(_config(top_k=v, strategy=strategy), response_mask=torch.ones(b, t), **cands)
        rewards[strategy] = out["opd_token_rewards"]
    exact = -(s_lp.exp() * (s_lp - t_lp)).sum(-1)
    for strategy, reward in rewards.items():
        torch.testing.assert_close(reward, exact, atol=1e-10, rtol=1e-8, msg=f"{strategy} disagrees")


def test_top_k_zero_uses_the_sampled_token():
    b, t, v = 2, 3, 9
    s_lp, t_lp, _, _ = _oracle(b, t, v, 1, seed=10)
    responses = torch.randint(0, v, (b, t), generator=torch.Generator().manual_seed(11))
    old_log_probs = _gather(s_lp, responses.unsqueeze(-1)).squeeze(-1)

    config = _config(top_k=0)
    cands = build_opd_candidates(
        config,
        responses=responses,
        old_log_probs=old_log_probs,
        t_on_s_log_probs=_gather(t_lp, responses.unsqueeze(-1)),
    )
    assert cands["cand_ids"].shape == (b, t, 1)
    torch.testing.assert_close(cands["cand_ids"].squeeze(-1), responses)
    out = opd_advantage(config, response_mask=torch.ones(b, t), **cands)
    # A single candidate renormalizes to weight 1, so the reward is the plain log-ratio.
    expected = -(old_log_probs - _gather(t_lp, responses.unsqueeze(-1)).squeeze(-1))
    torch.testing.assert_close(out["opd_token_rewards"], expected, atol=1e-10, rtol=1e-8)


@pytest.mark.parametrize("strategy", ["intersection", "union"])
def test_strategy_membership_uses_ids_when_boundary_log_probs_tie(strategy):
    """Equal k-th logits must not turn threshold membership into the wrong id set."""
    s_ids = torch.tensor([[[0, 1]]])
    t_ids = torch.tensor([[[0, 2]]])
    # Every selected log-prob ties, so probability-threshold membership would
    # incorrectly claim that token 1 is in T and token 2 is in S.
    tied = torch.full((1, 1, 2), -1.0)
    cands = build_opd_candidates(
        _config(top_k=2, strategy=strategy),
        responses=torch.zeros(1, 1, dtype=torch.int64),
        old_log_probs=torch.zeros(1, 1),
        s_topk_ids=s_ids,
        s_topk_log_probs=tied,
        s_on_t_log_probs=tied,
        t_on_s_log_probs=tied,
        t_topk_ids=t_ids,
        t_topk_log_probs=tied,
    )
    got = {int(token) for token, valid in zip(cands["cand_ids"][0, 0], cands["cand_mask"][0, 0], strict=True) if valid}
    want = {0} if strategy == "intersection" else {0, 1, 2}
    assert got == want


def test_none_weight_mode_is_uniform_over_valid_candidates():
    student = torch.tensor([[[-0.1, -2.0, -7.0]]], dtype=torch.float64)
    teacher = torch.tensor([[[-0.5, -0.2, -3.0]]], dtype=torch.float64)
    valid = torch.tensor([[[True, True, False]]])
    out = opd_advantage(
        _config(top_k=3, reward_weight_mode="none"),
        student_log_probs=student,
        teacher_log_probs=teacher,
        cand_mask=valid,
        response_mask=torch.ones(1, 1),
    )
    expected = torch.tensor([[[-0.2, 0.9, 0.0]]], dtype=torch.float64)
    torch.testing.assert_close(out["opd_advantages"], expected)
    torch.testing.assert_close(out["opd_token_rewards"], expected.sum(-1))


def test_kl_bucket_metrics_split_by_position_and_seqlen():
    """Position buckets average tokens; seqlen buckets average per-response means."""
    # Two responses of lengths 2 and 4 in a padded frame of width 6.
    opd_kl = torch.tensor(
        [
            [1.0, 3.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 4.0, 6.0, 8.0, 0.0, 0.0],
        ]
    )
    response_mask = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        ]
    )
    metrics = opd_kl_bucket_metrics(opd_kl, response_mask, bucket_size=2)

    # Position buckets: token-level means over [0,2), [2,4), [4,6).
    assert metrics["opd/kl_by_position/0-2"] == pytest.approx((1.0 + 3.0 + 2.0 + 4.0) / 4)
    assert metrics["opd/kl_by_position/2-4"] == pytest.approx((6.0 + 8.0) / 2)
    assert "opd/kl_by_position/4-6" not in metrics  # empty bucket is omitted

    # Seqlen buckets: per-response means 2.0 (length 2, in [2,4)) and 5.0 (length 4, in [4,6)).
    assert "opd/kl_by_seqlen/0-2" not in metrics  # no response shorter than 2
    assert metrics["opd/kl_by_seqlen/2-4"] == pytest.approx(2.0)
    assert metrics["opd/kl_by_seqlen/4-6"] == pytest.approx(5.0)


def test_kl_bucket_metrics_cover_full_response_length():
    """Buckets span [0, T) so the last partial bucket is still reported."""
    opd_kl = torch.ones(1, 5)
    response_mask = torch.ones(1, 5)
    metrics = opd_kl_bucket_metrics(opd_kl, response_mask, bucket_size=2)
    assert set(metrics) == {
        "opd/kl_by_position/0-2",
        "opd/kl_by_position/2-4",
        "opd/kl_by_position/4-6",
        "opd/kl_by_seqlen/4-6",
    }
    assert all(value == pytest.approx(1.0) for value in metrics.values())


def test_bucket_mean_sums_split_micro_batches_into_exact_means():
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 20.0, 30.0, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0, 0.0]])
    whole = bucket_mean_sums({"v": values}, mask, bucket_size=2, span=6)
    # Two micro-batches summed key by key, as AggregationType.SUM does, give the same ratios.
    parts = [bucket_mean_sums({"v": values[i : i + 1]}, mask[i : i + 1], bucket_size=2, span=6) for i in range(2)]
    assert set(parts[0]) == set(parts[1]) == set(whole)
    summed = resolve_bucket_means({k: parts[0][k] + parts[1][k] for k in whole})

    assert summed["v_by_position/0-2"] == pytest.approx((1 + 2 + 10 + 20) / 4)
    assert summed["v_by_position/2-4"] == pytest.approx((3 + 4 + 30) / 3)
    assert summed["v_by_position/4-6"] == pytest.approx(5.0)
    # Lengths 5 and 3 fall in [4, 6) and [2, 4); each response contributes its own mean.
    assert summed["v_by_seqlen/4-6"] == pytest.approx(3.0)
    assert summed["v_by_seqlen/2-4"] == pytest.approx(20.0)
    assert "v_by_seqlen/0-2" not in summed
    assert summed == resolve_bucket_means(dict(whole))

    only_position = bucket_mean_sums({"v": values}, mask, seqlen_names=(), bucket_size=2, span=6)
    assert not [k for k in only_position if "_by_seqlen/" in k]


def test_binned_mean_sums_stratify_by_key():
    key = torch.tensor([[0.05, 0.3, 3.0, 0.0], [0.7, 1.5, 0.1, 9.0]])
    values = torch.tensor([[1.0, -2.0, 4.0, 99.0], [3.0, -1.0, 5.0, 99.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0]])
    out = resolve_bucket_means(binned_mean_sums({"v": values}, key, (0.0, 0.1, 0.5, 1.0, 2.0), mask, "h", "frac"))

    # Bins: [0, 0.1) {1}, [0.1, 0.5) {-2, 5}, [0.5, 1) {3}, [1, 2) {-1}, [2, inf) {4}; masked tokens ignored.
    assert out["frac_by_h/0-0.1"] == pytest.approx(1 / 6)
    assert out["frac_by_h/0.1-0.5"] == pytest.approx(2 / 6)
    assert out["v_by_h/0.1-0.5"] == pytest.approx(1.5)
    assert out["v_by_h/2-inf"] == pytest.approx(4.0)
    assert out["v_abs_share_by_h/0.1-0.5"] == pytest.approx(7 / 16)
    assert sum(v for k, v in out.items() if k.startswith("frac_by_h/")) == pytest.approx(1.0)
