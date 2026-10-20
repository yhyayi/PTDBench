"""Top-K diagnostics: numerical definitions, masks, ties, and forward/driver wiring."""

import math
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.opd.order_only import direct_topk_loss, topk_agreement_metrics
from verl.utils import tensordict_utils as tu


def metrics(student, teacher, mask=None, student_ids=None, teacher_ids=None):
    if teacher_ids is None:
        teacher_ids = torch.arange(student.shape[-1]).expand_as(student).long()
    if student_ids is None:
        student_ids = teacher_ids.gather(-1, student.argsort(dim=-1, descending=True, stable=True))
    if mask is None:
        mask = torch.ones(student.shape[:-1], dtype=torch.bool)
    return topk_agreement_metrics(student, teacher, student_ids, teacher_ids, mask)


@pytest.mark.parametrize("student,teacher,expected", [
    ([4, 3, 2, 1], [4, 3, 2, 1], 1.0),
    ([1, 2, 3, 4], [4, 3, 2, 1], -1.0),
    ([4, 3, 3, 1], [4, 3, 2, 1], math.sqrt(5 / 6)),
    ([4, 3, 3, 1], [4, 3, 3, 1], 1.0),
    ([4, 4, 2, 1], [4, 3, 3, 1], 0.8),
])
def test_tau_b_handles_both_sides_ties(student, teacher, expected):
    result = metrics(torch.tensor([student], dtype=torch.float64), torch.tensor([teacher], dtype=torch.float64))
    assert result["opd/kendall_tau_topk"] == pytest.approx(expected)
    assert result["opd/kendall_tau_valid_fraction"] == 1


def test_constant_scores_have_undefined_tau_and_report_valid_fraction():
    student = torch.tensor([[4., 3., 2.], [1., 1., 1.], [4., 3., 2.]])
    teacher = torch.tensor([[4., 3., 2.], [4., 3., 2.], [1., 1., 1.]])
    result = metrics(student, teacher)
    assert result["opd/kendall_tau_topk"] == pytest.approx(1)
    assert result["opd/kendall_tau_valid_fraction"] == pytest.approx(1 / 3)
    undefined = metrics(student[1:], teacher[1:])
    assert "opd/kendall_tau_topk" not in undefined
    assert undefined["opd/kendall_tau_valid_fraction"] == 0
    assert all(math.isfinite(value) for value in undefined.values())


def test_full_vocab_top1_is_not_candidate_argmax_and_overlap_is_set_based():
    student = torch.tensor([[8., 1., 5., 0., 10., 7., 2., 3.]])
    teacher = torch.tensor([[10., 9., 8., 7., 0., 1., 2., 3.]])
    teacher_ids = teacher.topk(4).indices
    student_ids = student.topk(4).indices
    on_teacher = student.gather(-1, teacher_ids)
    assert on_teacher.argmax(-1).item() == 0  # Old candidate-only agreement would be 1.
    result = metrics(on_teacher, teacher.gather(-1, teacher_ids), student_ids=student_ids, teacher_ids=teacher_ids)
    assert result["opd/top1_agreement"] == 0
    assert result["opd/topk_overlap"] == 0.5
    assert result["opd/diagnostic_top_k"] == 4


def test_jsd_and_global_position_mean_ignore_padding_and_do_not_affect_gradients():
    torch.manual_seed(42)
    student = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    teacher = torch.randn_like(student).sort(dim=-1, descending=True).values.requires_grad_()
    mask = torch.tensor([[True, False, False], [True, True, True]])
    # Invalid entries are deliberately nonfinite; filter them before computing diagnostics.
    with torch.no_grad():
        student[~mask] = float("nan")
        teacher[~mask] = float("nan")
    loss_before = direct_topk_loss(student[mask], "adjacent_logistic").mean()
    gradient_before = torch.autograd.grad(loss_before, student)[0]
    result = metrics(student, teacher, mask)
    p, q = student[mask].softmax(-1), teacher[mask].softmax(-1)
    m = (p + q) / 2
    expected = ((p * (p.log() - m.log())).sum(-1) + (q * (q.log() - m.log())).sum(-1)).mean() / 2
    assert result["opd/jsd_topk"] == pytest.approx(expected.item())
    assert result["opd/diagnostic_tokens"] == 4
    assert student.grad is None and teacher.grad is None
    assert all(isinstance(v, float) for v in result.values())
    gradient_after = torch.autograd.grad(direct_topk_loss(student[mask], "adjacent_logistic").mean(), student)[0]
    torch.testing.assert_close(gradient_before, gradient_after, rtol=0, atol=0)
    empty = metrics(student, teacher, torch.zeros_like(mask))
    assert empty["opd/diagnostic_tokens"] == 0
    assert "opd/jsd_topk" not in empty


def test_identical_distributions_have_zero_jsd():
    scores = torch.tensor([[3., 2., 1.]])
    result = metrics(scores, scores)
    assert result["opd/jsd_topk"] == pytest.approx(0, abs=1e-7)
    assert result["opd/topk_overlap"] == 1
    assert result["opd/top1_agreement"] == 1


@pytest.mark.parametrize("temperature", [0.6, 1., 2.])
@pytest.mark.parametrize("unscaled", [True, False])
def test_engine_separates_raw_diagnostics_from_sampling_log_probs(monkeypatch, temperature, unscaled):
    from verl.workers.engine.fsdp import transformer_impl as engine

    monkeypatch.setattr(
        engine, "logprobs_from_logits",
        lambda logits, labels, **kwargs: logits.float().log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1),
    )
    # The last output-head row is padding and must never enter either top-K list.
    logits = torch.tensor([[3., 1., 2., 50.], [0., 3., 1., 50.], [1., 2., 3., 50.]])
    ids = torch.tensor([[0, 2], [1, 2], [2, 1]])
    labels = torch.tensor([1, 2, 0])
    batch = TensorDict(
        {"input_ids": torch.nested.as_nested_tensor([labels], layout=torch.jagged)}, batch_size=1
    )
    tu.assign_non_tensor(batch, opd_enable=True, opd_top_k=2, opd_unscaled_scores=unscaled)
    stub = SimpleNamespace(
        opd_valid_token_ids=torch.arange(3), _gather_and_unpad_packed=lambda value, pad_size: value
    )
    output_args = {
        "input_ids_rmpad_rolled": labels, "temperature_rmpad": torch.full((3,), temperature),
        "opd_cand_ids_rmpad": ids, "pad_size": 0,
    }
    output = engine.FSDPEngineWithLMHead.prepare_model_outputs(
        stub, SimpleNamespace(logits=logits[None]), output_args, batch, None
    )
    scored = logits if unscaled else logits / temperature
    torch.testing.assert_close(output["opd_cand_log_probs"].values(), scored.log_softmax(-1).gather(-1, ids))
    torch.testing.assert_close(output["opd_topk_ids"].values(), logits[:, :3].topk(2).indices)
    torch.testing.assert_close(
        output["log_probs"].values(), (logits / temperature).log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1)
    )


@pytest.mark.parametrize("objective", ["adjacent_logistic", "topk_jsd"])
def test_driver_records_all_metrics_from_existing_prepasses(monkeypatch, objective):
    from verl.trainer.ppo.v1 import trainer_base
    from verl.workers.config.opd import OPDConfig

    def nested(x):
        return torch.nested.as_nested_tensor([x[:1], x[1:]], layout=torch.jagged)

    s = torch.tensor([[1., 2., 0.], [4., 3., 2.], [4., 2., 3.]])
    t = torch.tensor([[4., 3., 2.], [4., 3., 2.], [4., 3., 2.]])
    tids = torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
    sids = torch.tensor([[1, 0, 2], [3, 0, 1], [0, 2, 1]])
    mask = torch.ones(3)
    frame = TensorDict({
        "input_ids": torch.nested.as_nested_tensor([torch.arange(3), torch.arange(5)], layout=torch.jagged),
        "response_mask": nested(mask),
    }, batch_size=2)
    prepass = {"opd_topk_ids": nested(tids), "opd_topk_log_probs": nested(t)}
    stub = SimpleNamespace(
        opd_config=OPDConfig(
            enable=True, teacher_model_path="unused", objective=objective, top_k=3, strategy="only_tch",
            teacher_temperature=2.,
        ),
        _opd_student_extras={"opd_cand_log_probs": nested(s), "opd_topk_ids": nested(sids)},
    )

    class Batch:
        keys = ["a", "b"]
        partition_id = "test"

        def __len__(self):
            return 2

    batch = Batch()
    written = {}
    monkeypatch.setattr(trainer_base.tq, "kv_batch_get", lambda **kwargs: frame)

    def put(**kwargs):
        written.update(kwargs["fields"])
        return batch

    monkeypatch.setattr(trainer_base.tq, "kv_batch_put", put)
    result = {}
    trainer_base.PPOTrainer._compute_opd(stub, batch, result, prepass)
    expected = metrics(s, t, mask, sids, tids)
    assert all(result[key] == pytest.approx(value) for key, value in expected.items())
    assert stub._opd_student_extras is None
    assert written["opd_token_rewards"].values().eq(0).all()
    if objective == "adjacent_logistic":
        assert "opd_teacher_topk_log_probs" not in written
    else:
        torch.testing.assert_close(written["opd_teacher_topk_log_probs"].values(), (t / 2).log_softmax(-1))


@pytest.mark.parametrize("strategy", ["only_tch", "only_stu"])
def test_old_log_prob_annotation_survives_candidate_put(monkeypatch, strategy):
    """kv_batch_put returns a fresh meta with empty extra_info; the student pass must still
    compute its own top-K instead of reading back the teacher's prepass ids."""
    from verl.trainer.ppo.v1 import trainer_base
    from verl.workers.config.opd import OPDConfig

    class Batch:
        def __init__(self, extra_info):
            self.keys = ["a"]
            self.partition_id = "test"
            self.extra_info = extra_info

        def __len__(self):
            return 1

    frame = TensorDict({
        "input_ids": torch.nested.as_nested_tensor([torch.arange(4)], layout=torch.jagged),
        "response_mask": torch.nested.as_nested_tensor([torch.ones(2)], layout=torch.jagged),
    }, batch_size=1)
    monkeypatch.setattr(trainer_base.tq, "kv_batch_get", lambda **kwargs: frame)
    monkeypatch.setattr(trainer_base.tq, "kv_batch_put", lambda **kwargs: Batch({}))
    objective = "adjacent_logistic" if strategy == "only_tch" else "legacy"
    stub = SimpleNamespace(opd_config=OPDConfig(
        enable=True, teacher_model_path="unused", objective=objective, top_k=3, strategy=strategy, chunk_size=128,
    ))
    teacher_ids = torch.zeros(2, 3, dtype=torch.long)
    prepass = {"opd_topk_ids": torch.nested.as_nested_tensor([teacher_ids], layout=torch.jagged)}
    out = trainer_base.PPOTrainer._prepare_opd_old_log_prob(stub, Batch({"temperature": 0.7}), prepass)
    assert out.extra_info["opd_enable"] is True
    assert out.extra_info["opd_top_k"] == 3
    assert out.extra_info["opd_chunk_size"] == 128
    assert out.extra_info["temperature"] == 0.7
