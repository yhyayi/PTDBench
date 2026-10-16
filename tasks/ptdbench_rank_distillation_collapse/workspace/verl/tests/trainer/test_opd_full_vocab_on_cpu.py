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
"""CPU tests for the full-vocabulary OPD kernel.

The chunked autograd function must reproduce a dense autograd reference exactly
(values and both weight/hidden gradients), independent of the chunk size, for
the single-teacher target and for the multi-scorer target reserved for
full-vocabulary Direct-OPD.
"""

import json
import os

import pytest
import torch
import torch.nn.functional as F

from verl.trainer.opd.advantage import resolve_bucket_means
from verl.trainer.opd.full_vocab import (
    DIAG_LOG_Z,
    DIAG_SCORER_LOG_PROB,
    DIAG_TEACHER_ENTROPY,
    DIAG_TEACHER_LOG_PROB,
    FULL_VOCAB_SCORERS,
    FrozenScorer,
    FullVocabTarget,
    diag_layout,
    full_vocab_rkl,
    full_vocab_scorers,
    load_lm_head_weight,
)

N, H, V = 11, 6, 37


def _close(actual, expected):
    # The kernel upcasts logits to fp32 by design; the reference runs in fp64.
    torch.testing.assert_close(actual.to(expected.dtype), expected, rtol=1e-5, atol=1e-5)


def _inputs(seed: int, num_scorers: int = 1, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(N, H, generator=g, dtype=dtype)
    weight = torch.randn(V, H, generator=g, dtype=dtype)
    labels = torch.randint(0, V, (N,), generator=g)
    frozen = [
        (torch.randn(N, H, generator=g, dtype=dtype), torch.randn(V, H, generator=g, dtype=dtype))
        for _ in range(num_scorers)
    ]
    row_mask = torch.rand(N, generator=g) > 0.3
    row_mask[0] = False
    row_mask[1] = True
    return hidden, weight, labels, frozen, row_mask


def _dense_reference(hidden, weight, labels, frozen, coefs, row_mask, temperature, teacher_temperature):
    logp = F.log_softmax(hidden @ weight.t() / temperature, dim=-1)
    p = logp.exp()
    entropy = -(p * logp).sum(-1)
    log_probs = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    scorer_logps = [F.log_softmax(h @ w.t() / teacher_temperature, dim=-1) for h, w in frozen]
    combo = sum(c * lp for lp, c in zip(scorer_logps, coefs, strict=True))
    logq = F.log_softmax(combo, dim=-1)
    keep = row_mask.to(p.dtype)
    rkl = (p * (logp - logq)).sum(-1) * keep
    teacher_entropy = -(logq.exp() * logq).sum(-1) * keep
    teacher_log_prob = logq.gather(-1, labels.unsqueeze(-1)).squeeze(-1) * keep
    log_z = torch.logsumexp(combo, dim=-1) * keep
    scorer_label = torch.stack([lp.gather(-1, labels.unsqueeze(-1)).squeeze(-1) for lp in scorer_logps], -1)
    extra = {
        "label": labels.to(p.dtype),
        "student_entropy": entropy,
        "student_argmax": logp.argmax(-1).to(p.dtype),
        "target_argmax": logq.argmax(-1).to(p.dtype),
        "scorer_kl": torch.stack([(p * (logp - lp)).sum(-1) for lp in scorer_logps], -1),
        "scorer_entropy": torch.stack([-(lp.exp() * lp).sum(-1) for lp in scorer_logps], -1),
        "scorer_argmax": torch.stack([lp.argmax(-1).to(p.dtype) for lp in scorer_logps], -1),
    }
    extra = {k: (v * keep if v.dim() == 1 else v * keep.unsqueeze(-1)).detach() for k, v in extra.items()}
    scorer_label = scorer_label * keep.unsqueeze(-1)
    return log_probs, entropy, rkl, teacher_entropy, teacher_log_prob, log_z, scorer_label, extra


def _check_diag(diag, ref):
    layout = diag_layout(ref[6].shape[-1])
    assert diag.shape[-1] == layout["width"]
    _close(diag[:, DIAG_TEACHER_ENTROPY], ref[3])
    _close(diag[:, DIAG_TEACHER_LOG_PROB], ref[4])
    _close(diag[:, DIAG_LOG_Z], ref[5])
    _close(diag[:, layout["scorer_log_prob"]], ref[6])
    for name, expected in ref[7].items():
        _close(diag[:, layout[name]], expected)


def _run(seed, chunk_size, coefs=(1.0,), temperature=1.0, teacher_temperature=1.0, upstream=(0.7, -0.3, 1.3)):
    hidden, weight, labels, frozen, row_mask = _inputs(seed, num_scorers=len(coefs))

    h_ref = hidden.clone().requires_grad_(True)
    w_ref = weight.clone().requires_grad_(True)
    ref = _dense_reference(h_ref, w_ref, labels, frozen, coefs, row_mask, temperature, teacher_temperature)

    h_k = hidden.clone().requires_grad_(True)
    w_k = weight.clone().requires_grad_(True)
    target = FullVocabTarget(
        scorers=[FrozenScorer(h, w, c) for (h, w), c in zip(frozen, coefs, strict=True)],
        row_mask=row_mask,
        temperature=teacher_temperature,
        chunk_size=chunk_size,
    )
    out = full_vocab_rkl(h_k, w_k, labels, temperature, target)

    g = torch.Generator().manual_seed(seed + 1)
    weights = [torch.randn(N, generator=g, dtype=hidden.dtype) * u for u in upstream]
    loss_ref = sum((w * t).sum() for w, t in zip(weights, ref[:3], strict=True) if w.abs().sum() > 0)
    loss_k = sum((w * t).sum() for w, t in zip(weights, out[:3], strict=True) if w.abs().sum() > 0)
    loss_ref.backward()
    loss_k.backward()
    return ref, out, (h_ref.grad, w_ref.grad), (h_k.grad, w_k.grad)


@pytest.mark.parametrize("chunk_size", [1, 3, 4, N, 64])
def test_matches_dense_reference(chunk_size):
    ref, out, ref_grads, k_grads = _run(seed=0, chunk_size=chunk_size)
    log_probs, entropy, rkl, diag = out
    _close(log_probs, ref[0])
    _close(entropy, ref[1])
    _close(rkl, ref[2])
    _check_diag(diag, ref)
    _close(k_grads[0], ref_grads[0])
    _close(k_grads[1], ref_grads[1])


def test_temperatures():
    ref, out, ref_grads, k_grads = _run(seed=1, chunk_size=4, temperature=0.7, teacher_temperature=1.6)
    _close(out[2], ref[2])
    _close(k_grads[0], ref_grads[0])
    _close(k_grads[1], ref_grads[1])


def test_multi_scorer_target():
    # The full-vocabulary Direct-OPD target: log q ∝ log p_S0 + log p_T+ - log p_T0.
    ref, out, ref_grads, k_grads = _run(seed=2, chunk_size=5, coefs=(1.0, 1.0, -1.0))
    _close(out[2], ref[2])
    _check_diag(out[3], ref)
    _close(k_grads[0], ref_grads[0])
    _close(k_grads[1], ref_grads[1])


def test_diff_target_degenerates_to_plain_opd_without_teacher_shift():
    # T+ == T0 leaves q = S0: full-vocabulary Direct-OPD must coincide with plain
    # full-vocabulary OPD against S0, values and gradients alike.
    hidden, weight, labels, frozen, row_mask = _inputs(seed=8, num_scorers=1)
    s0_hidden, s0_head = frozen[0]

    def run(scorers):
        h = hidden.clone().requires_grad_(True)
        w = weight.clone().requires_grad_(True)
        target = FullVocabTarget(scorers=scorers, row_mask=row_mask, chunk_size=4)
        _, _, rkl, _ = full_vocab_rkl(h, w, labels, 1.0, target)
        rkl.sum().backward()
        return rkl, h.grad, w.grad

    plain = run([FrozenScorer(s0_hidden, s0_head)])
    diff = run(
        [
            FrozenScorer(s0_hidden, s0_head, coef=1.0),
            FrozenScorer(s0_hidden, s0_head, coef=1.0),
            FrozenScorer(s0_hidden, s0_head, coef=-1.0),
        ]
    )
    for got, expected in zip(diff, plain, strict=True):
        _close(got, expected)


def test_full_vocab_scorers_beta():
    assert full_vocab_scorers("diff") == FULL_VOCAB_SCORERS["diff"]
    assert full_vocab_scorers("diff", 2.0) == {"teacher": 0.5, "teacher_ref": -0.5, "ref": 1.0}
    assert list(full_vocab_scorers("diff", 2.0)) == list(FULL_VOCAB_SCORERS["diff"])
    assert full_vocab_scorers("rkl", 2.0) == FULL_VOCAB_SCORERS["rkl"]


@pytest.mark.parametrize("beta", [0.5, 1.0, 3.0])
def test_diff_beta_gradient_is_the_squared_residual_gradient(beta):
    # sum_v sg[p](beta log p/S0 - log T+/T0)^2 has the gradient 2 beta^2 grad KL(p || q),
    # q ∝ S0 (T+/T0)^(1/beta).
    roles = list(FULL_VOCAB_SCORERS["diff"])
    coefs = full_vocab_scorers("diff", beta)
    hidden, weight, labels, frozen, row_mask = _inputs(seed=10, num_scorers=len(roles))

    h_k = hidden.clone().requires_grad_(True)
    w_k = weight.clone().requires_grad_(True)
    target = FullVocabTarget(
        [FrozenScorer(h, w, coefs[role]) for role, (h, w) in zip(roles, frozen, strict=True)],
        row_mask=row_mask,
        chunk_size=4,
    )
    _, _, rkl, _ = full_vocab_rkl(h_k, w_k, labels, 1.0, target)
    (2 * beta**2 * rkl.sum()).backward()

    h_ref = hidden.clone().requires_grad_(True)
    w_ref = weight.clone().requires_grad_(True)
    logp = F.log_softmax(h_ref @ w_ref.t(), dim=-1)
    logs = {role: F.log_softmax(h @ w.t(), dim=-1) for role, (h, w) in zip(roles, frozen, strict=True)}
    residual = beta * (logp - logs["ref"]) - (logs["teacher"] - logs["teacher_ref"])
    loss = (logp.exp().detach() * residual.square()).sum(-1)
    (loss * row_mask.to(loss.dtype)).sum().backward()

    _close(h_k.grad, h_ref.grad)
    _close(w_k.grad, w_ref.grad)


@pytest.mark.parametrize("upstream", [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])
def test_unused_outputs(upstream):
    ref, out, ref_grads, k_grads = _run(seed=3, chunk_size=3, upstream=upstream)
    _close(k_grads[0], ref_grads[0])
    _close(k_grads[1], ref_grads[1])


def test_rkl_gradient_is_the_reverse_kl_gradient():
    # The rkl output alone drives exactly the analytic d KL(p||q)/dz = p (log p - log q - KL).
    hidden, weight, labels, frozen, _ = _inputs(seed=4)
    row_mask = torch.ones(N, dtype=torch.bool)
    h = hidden.clone().requires_grad_(True)
    target = FullVocabTarget([FrozenScorer(*frozen[0])], row_mask=row_mask, chunk_size=4)
    _, _, rkl, _ = full_vocab_rkl(h, weight, labels, 1.0, target)
    rkl.sum().backward()

    logp = F.log_softmax(hidden @ weight.t(), dim=-1)
    logq = F.log_softmax(frozen[0][0] @ frozen[0][1].t(), dim=-1)
    p = logp.exp()
    kl = (p * (logp - logq)).sum(-1, keepdim=True)
    d_logits = p * (logp - logq - kl)
    _close(h.grad, d_logits @ weight)


def test_masked_rows_never_read_frozen_hidden():
    hidden, weight, labels, frozen, row_mask = _inputs(seed=5)
    poisoned = frozen[0][0].clone()
    poisoned[~row_mask] = float("nan")
    target = FullVocabTarget([FrozenScorer(poisoned, frozen[0][1])], row_mask=row_mask, chunk_size=3)
    h = hidden.clone().requires_grad_(True)
    _, _, rkl, diag = full_vocab_rkl(h, weight, labels, 1.0, target)
    rkl.sum().backward()
    assert torch.isfinite(rkl).all() and torch.isfinite(diag).all() and torch.isfinite(h.grad).all()
    assert (rkl[~row_mask] == 0).all()


def test_keeps_leading_dims_and_bf16_head():
    hidden, weight, labels, frozen, row_mask = _inputs(seed=6, dtype=torch.float32)
    target = FullVocabTarget(
        [FrozenScorer(frozen[0][0].bfloat16(), frozen[0][1].bfloat16())], row_mask=row_mask, chunk_size=4
    )
    h = hidden.unsqueeze(0).requires_grad_(True)
    log_probs, entropy, rkl, diag = full_vocab_rkl(h, weight, labels.unsqueeze(0), 1.0, target)
    assert log_probs.shape == entropy.shape == rkl.shape == (1, N)
    assert diag.shape == (1, N, diag_layout(1)["width"])
    rkl.sum().backward()
    assert h.grad.shape == h.shape and h.grad.dtype == torch.float32


def _loss_case(coefs, seed=7):
    """Kernel outputs and actor-loss inputs for two packed sequences (prompt 2 + response 3, prompt 1 +
    response 2); packed row ``p`` predicts token ``p + 1``."""
    from tensordict import TensorDict
    from tensordict.tensorclass import NonTensorData

    seq_offsets = torch.tensor([0, 5, 8])
    resp_lens = [3, 2]
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(8, H, generator=g, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(V, H, generator=g, dtype=torch.float64)
    frozen = [
        (torch.randn(8, H, generator=g, dtype=torch.float64), torch.randn(V, H, generator=g, dtype=torch.float64))
        for _ in coefs
    ]
    labels = torch.randint(0, V, (8,), generator=g)
    row_mask = torch.zeros(8, dtype=torch.bool)
    row_mask[1:4] = True
    row_mask[5:7] = True
    target = FullVocabTarget(
        [FrozenScorer(h, w, c) for (h, w), c in zip(frozen, coefs, strict=True)], row_mask=row_mask, chunk_size=3
    )
    log_probs, entropy, rkl, diag = full_vocab_rkl(hidden, weight, labels, 1.0, target)

    def nested(values):
        return torch.nested.nested_tensor_from_jagged(values, seq_offsets)

    def response(tensors):
        return torch.nested.as_nested_tensor(tensors, layout=torch.jagged)

    data = TensorDict(
        {
            "response_mask": response([torch.ones(n, dtype=torch.int64) for n in resp_lens]),
            "old_log_probs": response([log_probs[1:4].detach(), log_probs[5:7].detach()]),
            "advantages": response([torch.zeros(n, dtype=torch.float32) for n in resp_lens]),
            "prompts": torch.nested.as_nested_tensor([torch.zeros(2), torch.zeros(1)], layout=torch.jagged),
            "responses": torch.nested.as_nested_tensor([torch.zeros(3), torch.zeros(2)], layout=torch.jagged),
            "dp_size": NonTensorData(1),
            "batch_num_tokens": NonTensorData(None),
            "global_batch_size": NonTensorData(None),
        },
        batch_size=[2],
    )
    model_output = {"log_probs": nested(log_probs), "opd_rkl": nested(rkl), "opd_diag": nested(diag)}
    return hidden, labels, frozen, row_mask, log_probs, rkl, model_output, data


def _actor_config():
    from verl.workers.config import ActorConfig

    return ActorConfig(strategy="fsdp", rollout_n=1, ppo_micro_batch_size_per_gpu=1, loss_agg_mode="token-mean")


def test_full_vocab_loss_adds_token_mean_reverse_kl():
    from verl.trainer.opd.losses import opd_full_vocab_loss
    from verl.workers.config import OPDConfig, OPDFullVocabConfig

    hidden, _, _, row_mask, _, rkl, model_output, data = _loss_case(coefs=(1.0,))
    opd_config = OPDConfig(
        enable=True, teacher_model_path="/dev/null", full_vocab=OPDFullVocabConfig(enable=True, loss_coef=0.5)
    )
    loss, metrics = opd_full_vocab_loss(_actor_config(), opd_config, model_output, data)

    expected = 0.5 * rkl[row_mask].mean()
    torch.testing.assert_close(loss.double(), expected.detach().double(), rtol=1e-6, atol=1e-6)
    (grad,) = torch.autograd.grad(loss, hidden, retain_graph=True)
    (expected_grad,) = torch.autograd.grad(expected, hidden)
    torch.testing.assert_close(grad, expected_grad, rtol=1e-6, atol=1e-6)
    assert {"actor/opd_full_rkl", "actor/opd_sampled_k1", "actor/opd_teacher_entropy"} <= set(metrics)
    assert "actor/opd_target_entropy" not in metrics and "actor/opd_log_z_mean" not in metrics


@pytest.mark.parametrize("beta", [1.0, 2.5])
def test_full_vocab_diff_loss_metrics(beta):
    from verl.trainer.opd.losses import opd_full_vocab_loss
    from verl.workers.config import OPDConfig, OPDFullVocabConfig

    roles = list(FULL_VOCAB_SCORERS["diff"])
    coefs = tuple(full_vocab_scorers("diff", beta).values())
    _, labels, frozen, row_mask, log_probs, _, model_output, data = _loss_case(coefs=coefs, seed=9)
    opd_config = OPDConfig(
        enable=True,
        teacher_model_path="/dev/null",
        teacher_ref_model_path="/dev/null",
        full_vocab=OPDFullVocabConfig(enable=True, target="diff", beta=beta),
    )
    _, metrics = opd_full_vocab_loss(_actor_config(), opd_config, model_output, data)
    value = {key: float(metric.aggregate()) for key, metric in metrics.items()}

    scorer_logps = {role: F.log_softmax(h @ w.t(), dim=-1) for role, (h, w) in zip(roles, frozen, strict=True)}
    combo = sum(c * scorer_logps[role] for role, c in zip(roles, coefs, strict=True))
    at_label = {role: lp.gather(-1, labels.unsqueeze(-1)).squeeze(-1) for role, lp in scorer_logps.items()}
    log_z = torch.logsumexp(combo, dim=-1)[row_mask]
    teacher_shift = (at_label["teacher"] - at_label["teacher_ref"])[row_mask]
    student_shift = (log_probs.detach() - at_label["ref"])[row_mask]
    target_shift = teacher_shift / beta - log_z

    def approx(key, expected):
        assert value[key] == pytest.approx(float(expected), rel=1e-5, abs=1e-6), key

    approx("actor/opd_log_z_mean", log_z.mean())
    approx("actor/opd_log_z_std", log_z.std(unbiased=False))
    approx("actor/opd_log_z_abs_max", log_z.abs().max())
    approx("actor/opd_teacher_shift_mean", teacher_shift.mean())
    approx("actor/opd_teacher_shift_abs_mean", teacher_shift.abs().mean())
    approx("actor/opd_student_shift_mean", student_shift.mean())
    approx("actor/opd_student_shift_abs_mean", student_shift.abs().mean())
    approx("actor/opd_absorption", (student_shift * target_shift).sum() / target_shift.square().sum())
    # k1 = log p - log q is exactly the student shift minus the target shift.
    approx(
        "actor/opd_sampled_k1",
        value["actor/opd_student_shift_mean"]
        - value["actor/opd_teacher_shift_mean"] / beta
        + value["actor/opd_log_z_mean"],
    )
    assert "actor/opd_target_entropy" in metrics and "actor/opd_teacher_entropy" not in metrics

    for role, lp in scorer_logps.items():
        approx(f"actor/opd_entropy_{role}", -(lp.exp() * lp).sum(-1)[row_mask].mean())
    teacher_argmax = scorer_logps["teacher"].argmax(-1)[row_mask]
    teacher_ref_argmax = scorer_logps["teacher_ref"].argmax(-1)[row_mask]
    approx("actor/opd_top1_agree_teacher_teacher_ref", (teacher_argmax == teacher_ref_argmax).double().mean())
    approx("actor/opd_teacher_gain", value["actor/opd_kl_to_teacher_ref"] - value["actor/opd_kl_to_teacher"])
    approx(
        "actor/opd_student_entropy_delta_ref",
        value["actor/opd_student_entropy"] - value["actor/opd_entropy_ref"],
    )

    # Both responses fit in the first bucket, so it reproduces the overall means.
    resolved = resolve_bucket_means(dict(value))
    assert not [k for k in resolved if k.endswith(("@num", "@den"))]
    assert resolved["actor/opd_teacher_shift_by_position/0-1024"] == pytest.approx(teacher_shift.mean().item())
    assert resolved["actor/opd_full_rkl_by_position/0-1024"] == pytest.approx(value["actor/opd_full_rkl"])
    seq_means = [teacher_shift[:3].mean(), teacher_shift[3:].mean()]
    assert resolved["actor/opd_teacher_shift_by_seqlen/0-1024"] == pytest.approx(sum(seq_means).item() / 2)
    assert "actor/opd_teacher_shift_by_position/1024-2048" not in resolved
    fracs = [v for k, v in resolved.items() if k.startswith("actor/opd_token_frac_by_student_entropy/")]
    assert sum(fracs) == pytest.approx(1.0)


def test_full_vocab_loss_metric_keys_do_not_depend_on_data():
    """Every DP rank and micro-batch must report the same keys for Metric.aggregate_dp."""
    from verl.trainer.opd.losses import opd_full_vocab_loss
    from verl.workers.config import OPDConfig, OPDFullVocabConfig

    coefs = tuple(full_vocab_scorers("diff", 1.0).values())
    opd_config = OPDConfig(
        enable=True,
        teacher_model_path="/dev/null",
        teacher_ref_model_path="/dev/null",
        full_vocab=OPDFullVocabConfig(enable=True, target="diff"),
    )
    keys = []
    for seed in (3, 4):
        *_, model_output, data = _loss_case(coefs=coefs, seed=seed)
        data.set_non_tensor("opd_bucket_span", 4096)
        data.set_non_tensor("opd_special_token_ids", [0, 1, 2])
        keys.append(set(opd_full_vocab_loss(_actor_config(), opd_config, model_output, data)[1]))
    assert keys[0] == keys[1]
    assert "actor/opd_teacher_shift_by_position/3072-4096@num" in keys[0]
    assert "actor/opd_teacher_shift_by_position/4096-5120@num" not in keys[0]
    assert "actor/opd_special_token_teacher_shift@den" in keys[0]


def test_diff_target_config_validation():
    from verl.workers.config import OPDConfig, OPDFullVocabConfig

    # The diff target validates: teacher_ref_model_path is the T0 scorer.
    diff_config = OPDConfig(
        enable=True,
        teacher_model_path="/dev/null",
        teacher_ref_model_path="/dev/null",
        full_vocab=OPDFullVocabConfig(enable=True, target="diff"),
    )
    assert diff_config.full_vocab_diff and diff_config.direct_opd
    with pytest.raises(ValueError, match="must be selected together"):
        OPDConfig(enable=True, teacher_model_path="/dev/null", full_vocab=OPDFullVocabConfig(enable=True, target="diff"))
    with pytest.raises(ValueError, match="must be selected together"):
        OPDConfig(
            enable=True,
            teacher_model_path="/dev/null",
            teacher_ref_model_path="/dev/null",
            full_vocab=OPDFullVocabConfig(enable=True, target="rkl"),
        )
    with pytest.raises(ValueError, match="beta must be > 0"):
        OPDFullVocabConfig(enable=True, target="diff", beta=0.0)
    with pytest.raises(ValueError, match="beta only applies to target='diff'"):
        OPDFullVocabConfig(enable=True, target="rkl", beta=2.0)
    with pytest.raises(NotImplementedError, match="only implemented for target='diff'"):
        OPDConfig(
            enable=True,
            teachers=[{"name": "math", "path": "/dev/null"}],
            full_vocab=OPDFullVocabConfig(enable=True, target="rkl"),
        )


def _consensus_config(teachers=("justrl", "deepscaler"), beta=1.0, teacher_weights=None, **kwargs):
    from verl.workers.config import OPDConfig, OPDFullVocabConfig

    entries = kwargs.pop("teacher_entries", None) or [{"name": n, "path": f"/dev/null/{n}"} for n in teachers]
    return OPDConfig(
        enable=True,
        teacher_ref_model_path="/dev/null/t0",
        teachers=entries,
        full_vocab=OPDFullVocabConfig(enable=True, target="diff", beta=beta, teacher_weights=teacher_weights),
        **kwargs,
    )


def test_consensus_config_validation():
    from verl.workers.config import OPDConfig, OPDFullVocabConfig

    cfg = _consensus_config()
    assert cfg.full_vocab_consensus and cfg.multi_diff and cfg.full_vocab_diff
    assert cfg.teacher_names == ("justrl", "deepscaler")
    assert cfg.consensus_teacher_weights == (0.5, 0.5)
    weighted = _consensus_config(teacher_weights={"justrl": 3.0})
    assert weighted.consensus_teacher_weights == pytest.approx((0.75, 0.25))
    legacy = OPDConfig(
        enable=True,
        teacher_model_path="/dev/null",
        teacher_ref_model_path="/dev/null",
        full_vocab=OPDFullVocabConfig(enable=True, target="diff"),
    )
    assert legacy.full_vocab_diff and not legacy.full_vocab_consensus

    with pytest.raises(ValueError, match="leave tags empty"):
        _consensus_config(teacher_entries=[{"name": "a", "path": "/x", "tags": ["math"]}, {"name": "b", "path": "/y"}])
    with pytest.raises(ValueError, match="teacher_weights instead"):
        _consensus_config(domain_adv_scale={"justrl": 2.0})
    with pytest.raises(ValueError, match="must be identifiers"):
        _consensus_config(teachers=("just-rl", "deepscaler"))
    with pytest.raises(ValueError, match="unknown teachers"):
        _consensus_config(teacher_weights={"other": 1.0})
    with pytest.raises(ValueError, match="must not all be zero"):
        _consensus_config(teacher_weights={"justrl": 0.0, "deepscaler": 0.0})
    with pytest.raises(ValueError, match=">= 0"):
        _consensus_config(teacher_weights={"justrl": -1.0})
    with pytest.raises(ValueError, match="replaces opd.teacher_model_path"):
        _consensus_config(teacher_model_path="/dev/null")
    with pytest.raises(ValueError, match="consensus of several opd.teachers"):
        OPDConfig(
            enable=True,
            teacher_model_path="/dev/null",
            teacher_ref_model_path="/dev/null",
            full_vocab=OPDFullVocabConfig(enable=True, target="diff", teacher_weights={"a": 1.0}),
        )
    with pytest.raises(ValueError, match="teacher_weights only applies to target='diff'"):
        OPDFullVocabConfig(enable=True, target="rkl", teacher_weights={"a": 1.0})


def test_full_vocab_scorers_consensus():
    from verl.trainer.opd.full_vocab import consensus_teacher_role, opd_full_vocab_scorers

    scorers = full_vocab_scorers("diff", 1.0, teacher_weights={"a": 1.0, "b": 3.0})
    assert scorers == {"teacher_a": 0.25, "teacher_b": 0.75, "teacher_ref": -1.0, "ref": 1.0}
    assert list(scorers) == ["teacher_a", "teacher_b", "teacher_ref", "ref"]
    assert full_vocab_scorers("diff", 2.0, teacher_weights={"a": 1.0, "b": 1.0}) == {
        "teacher_a": 0.25,
        "teacher_b": 0.25,
        "teacher_ref": -0.5,
        "ref": 1.0,
    }
    with pytest.raises(ValueError, match="only applies to the diff target"):
        full_vocab_scorers("rkl", teacher_weights={"a": 1.0})

    assert opd_full_vocab_scorers(_consensus_config(beta=0.5)) == {
        consensus_teacher_role("justrl"): 1.0,
        consensus_teacher_role("deepscaler"): 1.0,
        "teacher_ref": -2.0,
        "ref": 1.0,
    }
    from verl.workers.config import OPDConfig, OPDFullVocabConfig

    legacy = OPDConfig(
        enable=True,
        teacher_model_path="/dev/null",
        teacher_ref_model_path="/dev/null",
        full_vocab=OPDFullVocabConfig(enable=True, target="diff", beta=2.0),
    )
    assert opd_full_vocab_scorers(legacy) == full_vocab_scorers("diff", 2.0)


def test_consensus_of_identical_teachers_is_the_single_teacher_target():
    hidden, weight, labels, frozen, row_mask = _inputs(seed=11, num_scorers=3)
    (t_h, t_w), (t0_h, t0_w), (s0_h, s0_w) = frozen

    def run(scorers):
        h = hidden.clone().requires_grad_(True)
        w = weight.clone().requires_grad_(True)
        target = FullVocabTarget(scorers=scorers, row_mask=row_mask, chunk_size=4)
        _, _, rkl, diag = full_vocab_rkl(h, w, labels, 1.0, target)
        rkl.sum().backward()
        return rkl, diag[:, :DIAG_SCORER_LOG_PROB], h.grad, w.grad

    single = run([FrozenScorer(t_h, t_w, 1.0), FrozenScorer(t0_h, t0_w, -1.0), FrozenScorer(s0_h, s0_w, 1.0)])
    consensus = run(
        [
            FrozenScorer(t_h, t_w, 0.5),
            FrozenScorer(t_h, t_w, 0.5),
            FrozenScorer(t0_h, t0_w, -1.0),
            FrozenScorer(s0_h, s0_w, 1.0),
        ]
    )
    for got, expected in zip(consensus, single, strict=True):
        _close(got, expected)


@pytest.mark.parametrize("beta", [1.0, 0.5])
def test_full_vocab_consensus_loss_metrics(beta):
    from verl.trainer.opd.full_vocab import opd_full_vocab_scorers
    from verl.trainer.opd.losses import opd_full_vocab_loss

    opd_config = _consensus_config(beta=beta, teacher_weights={"justrl": 1.0, "deepscaler": 3.0})
    scorers = opd_full_vocab_scorers(opd_config)
    roles = list(scorers)
    _, labels, frozen, row_mask, log_probs, _, model_output, data = _loss_case(
        coefs=tuple(scorers.values()), seed=12
    )
    _, metrics = opd_full_vocab_loss(_actor_config(), opd_config, model_output, data)
    value = {key: float(metric.aggregate()) for key, metric in metrics.items()}

    scorer_logps = {role: F.log_softmax(h @ w.t(), dim=-1) for role, (h, w) in zip(roles, frozen, strict=True)}
    combo = sum(c * scorer_logps[role] for role, c in scorers.items())
    at_label = {role: lp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)[row_mask] for role, lp in scorer_logps.items()}
    log_z = torch.logsumexp(combo, dim=-1)[row_mask]
    shift = {name: at_label[f"teacher_{name}"] - at_label["teacher_ref"] for name in ("justrl", "deepscaler")}
    teacher_shift = 0.25 * shift["justrl"] + 0.75 * shift["deepscaler"]
    student_shift = log_probs.detach()[row_mask] - at_label["ref"]
    target_shift = teacher_shift / beta - log_z

    def approx(key, expected):
        assert value[key] == pytest.approx(float(expected), rel=1e-5, abs=1e-6), key

    approx("actor/opd_teacher_shift_mean", teacher_shift.mean())
    approx("actor/opd_teacher_shift_abs_mean", teacher_shift.abs().mean())
    approx("actor/opd_absorption", (student_shift * target_shift).sum() / target_shift.square().sum())
    approx(
        "actor/opd_sampled_k1",
        value["actor/opd_student_shift_mean"]
        - value["actor/opd_teacher_shift_mean"] / beta
        + value["actor/opd_log_z_mean"],
    )
    for name, s in shift.items():
        approx(f"actor/opd_teacher_shift_mean/{name}", s.mean())
        approx(f"actor/opd_teacher_shift_abs_mean/{name}", s.abs().mean())
    a, b = shift["justrl"] - shift["justrl"].mean(), shift["deepscaler"] - shift["deepscaler"].mean()
    corr = (a * b).mean() / (a.square().mean() * b.square().mean()).sqrt()
    approx("actor/opd_teacher_shift_corr/justrl-deepscaler", corr)
    approx(
        "actor/opd_teacher_shift_sign_agree/justrl-deepscaler",
        ((shift["justrl"] > 0) == (shift["deepscaler"] > 0)).double().mean(),
    )


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("sharded", [False, True])
def test_load_lm_head_weight(tmp_path, tied, sharded):
    from safetensors.torch import save_file

    head = torch.randn(V, H)
    embed = torch.randn(V, H)
    tensors = {"model.embed_tokens.weight": embed, "model.norm.weight": torch.ones(H)}
    if not tied:
        tensors["lm_head.weight"] = head
        tensors["mtp.lm_head.weight"] = torch.zeros(V, H)
    if sharded:
        names = sorted(tensors)
        files = {name: f"model-{i % 2:05d}.safetensors" for i, name in enumerate(names)}
        for file in set(files.values()):
            save_file({n: tensors[n] for n in names if files[n] == file}, os.path.join(tmp_path, file))
        with open(os.path.join(tmp_path, "model.safetensors.index.json"), "w") as f:
            json.dump({"weight_map": files}, f)
    else:
        save_file(tensors, os.path.join(tmp_path, "model.safetensors"))

    loaded = load_lm_head_weight(str(tmp_path), dtype=torch.float32)
    torch.testing.assert_close(loaded, embed if tied else head)
