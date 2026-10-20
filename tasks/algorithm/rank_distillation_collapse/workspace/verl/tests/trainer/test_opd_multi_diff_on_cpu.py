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
"""CPU tests for Multi-Diff OPD: several post-RL teachers against one shared reference.

The load-bearing properties are that hard routing hands every sample exactly its own
teacher's shift, that a single teacher reproduces plain Direct-OPD bit for bit, and
that ``sum`` adds the (scaled) shifts of every teacher.
"""

import pytest
import torch

from verl.trainer.opd.advantage import (
    build_opd_candidates,
    multi_diff_metrics,
    multi_diff_teacher_log_probs,
    opd_advantage,
    route_teacher_log_probs,
    token_set_shift_metrics,
)
from verl.workers.config.opd import OPDConfig

REF = "/dev/null/ref"
TWO_TEACHERS = [
    {"name": "math", "path": "/dev/null/math", "tags": ["math", "MATH"]},
    {"name": "code", "path": "/dev/null/code", "tags": ["code"]},
]


def _config(**kwargs) -> OPDConfig:
    kwargs.setdefault("enable", True)
    kwargs.setdefault("top_k", 0)
    kwargs.setdefault("teacher_ref_model_path", REF)
    kwargs.setdefault("teachers", TWO_TEACHERS)
    return OPDConfig(**kwargs)


def _log_probs(shape, seed):
    return -torch.rand(*shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64) * 5


def _advantage(cfg, teacher, teacher_ref, response_mask, old_log_probs=None, responses=None):
    b, t = teacher.shape
    responses = torch.zeros(b, t, dtype=torch.long) if responses is None else responses
    old_log_probs = _log_probs((b, t), 99) if old_log_probs is None else old_log_probs
    cands = build_opd_candidates(
        cfg,
        responses=responses,
        old_log_probs=old_log_probs,
        t_on_s_log_probs=teacher,
        t_ref_on_s_log_probs=teacher_ref,
    )
    return opd_advantage(cfg, response_mask=response_mask, **cands)


def test_route_picks_each_samples_teacher():
    stack = _log_probs((3, 4, 5), 0)
    route_idx = torch.tensor([2, 0, 1, 2])
    routed = route_teacher_log_probs(stack, route_idx)
    for b, k in enumerate(route_idx.tolist()):
        assert torch.equal(routed[b], stack[k, b])
    with pytest.raises(ValueError, match="route index"):
        route_teacher_log_probs(stack, torch.tensor([0, 1]))


def test_hard_routing_gives_each_sample_its_own_shift():
    teachers = _log_probs((2, 3, 4), 1)
    teacher_ref = _log_probs((3, 4), 2)
    route_idx = torch.tensor([1, 0, 1])
    response_mask = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    cfg = _config()

    effective = multi_diff_teacher_log_probs(teachers, teacher_ref, route_idx, route_mode="hard", adv_scales=(1, 1))
    # With unit scales the effective teacher is the routed one, not a reconstruction of it.
    assert torch.equal(effective, route_teacher_log_probs(teachers, route_idx))
    out = _advantage(cfg, effective, teacher_ref, response_mask)
    for b, k in enumerate(route_idx.tolist()):
        torch.testing.assert_close(out["opd_token_rewards"][b], (teachers[k, b] - teacher_ref[b]) * response_mask[b])


def test_single_teacher_is_bitwise_direct_opd():
    """One untagged teacher under hard routing must reproduce plain Direct-OPD exactly."""
    teacher = _log_probs((3, 6), 3)
    teacher_ref = _log_probs((3, 6), 4)
    old_log_probs = _log_probs((3, 6), 5)
    response_mask = torch.tensor([[1.0] * 6, [1.0] * 3 + [0.0] * 3, [0.0] * 6])
    single = _config(teachers=[{"name": "math", "path": "/dev/null/math"}])
    legacy = OPDConfig(enable=True, top_k=0, teacher_model_path="/dev/null/math", teacher_ref_model_path=REF)
    route_idx = torch.tensor(single.route_indices([None] * 3))

    effective = multi_diff_teacher_log_probs(
        teacher.unsqueeze(0), teacher_ref, route_idx, route_mode="hard", adv_scales=single.teacher_adv_scales
    )
    md_out = _advantage(single, effective, teacher_ref, response_mask, old_log_probs)
    ref_out = _advantage(legacy, teacher, teacher_ref, response_mask, old_log_probs)
    assert md_out.keys() == ref_out.keys()
    for key in ref_out:
        assert torch.equal(md_out[key], ref_out[key]), key


def test_hard_routing_scales_the_routed_shift():
    teachers = _log_probs((2, 2, 3), 6)
    teacher_ref = _log_probs((2, 3), 7)
    route_idx = torch.tensor([0, 1])
    effective = multi_diff_teacher_log_probs(
        teachers, teacher_ref, route_idx, route_mode="hard", adv_scales=(1.0, 0.25)
    )
    torch.testing.assert_close(effective[0], teachers[0, 0])
    torch.testing.assert_close(effective[1] - teacher_ref[1], 0.25 * (teachers[1, 1] - teacher_ref[1]))


def test_sum_mode_adds_every_scaled_shift():
    teachers = _log_probs((2, 3, 4), 8)
    teacher_ref = _log_probs((3, 4), 9)
    route_idx = torch.tensor([0, 1, 0])
    response_mask = torch.ones(3, 4)
    cfg = _config(route_mode="sum", domain_adv_scale={"code": 0.5})
    effective = multi_diff_teacher_log_probs(
        teachers, teacher_ref, route_idx, route_mode="sum", adv_scales=cfg.teacher_adv_scales
    )
    out = _advantage(cfg, effective, teacher_ref, response_mask)
    expected = (teachers[0] - teacher_ref) + 0.5 * (teachers[1] - teacher_ref)
    torch.testing.assert_close(out["opd_token_rewards"], expected)


def test_config_specs_routing_and_scales():
    cfg = _config(domain_adv_scale={"code": 2.0})
    assert cfg.multi_diff
    assert cfg.teacher_names == ("math", "code")
    assert cfg.teacher_specs[0].tags == ("math", "MATH")
    assert cfg.teacher_adv_scales == (1.0, 2.0)
    assert _config().teacher_adv_scales == (1.0, 1.0)
    assert cfg.route_indices(["code", "MATH", "math"]) == [1, 0, 0]
    with pytest.raises(ValueError, match="matches no teacher tags"):
        cfg.route_indices(["chemistry"])

    legacy = OPDConfig(enable=True, teacher_model_path="/dev/null/t")
    assert not legacy.multi_diff
    assert legacy.teacher_specs[0].name == "default"
    assert not legacy.needs_route_field
    assert legacy.route_indices(["anything", None]) == [0, 0]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"teacher_ref_model_path": None}, "requires the shared opd.teacher_ref_model_path"),
        ({"teacher_model_path": "/dev/null/t"}, "set only one of them"),
        ({"top_k": 4}, "requires opd.top_k=0"),
        ({"route_mode": "soft"}, "opd.route_mode must be one of"),
        ({"domain_adv_scale": {"chem": 1.0}}, "unknown teachers"),
        ({"domain_adv_scale": {"code": -1.0}}, "must be >= 0"),
        (
            {"teachers": [{"name": "a", "path": "/a", "tags": ["x"]}, {"name": "b", "path": "/b", "tags": ["x"]}]},
            "claimed by both",
        ),
        ({"teachers": [{"name": "a", "path": "/a"}, {"name": "b", "path": "/b", "tags": ["y"]}]}, "has no tags"),
        ({"teachers": [{"name": "a", "path": "/a", "tags": ["x"]}, {"name": "a", "path": "/b"}]}, "unique"),
        ({"teachers": [{"name": "a", "tags": ["x"]}]}, "non-empty name and path"),
        ({"teachers": [{"name": "a", "path": "/a", "tag": ["x"]}]}, "unknown keys"),
    ],
)
def test_config_rejects_invalid_multi_diff(kwargs, message):
    with pytest.raises((ValueError, NotImplementedError), match=message):
        _config(**kwargs)


def test_mask_token_ids_requires_sampled_token_mode():
    with pytest.raises(ValueError, match="mask_token_ids only applies"):
        OPDConfig(enable=True, teacher_model_path="/dev/null/t", top_k=4, mask_token_ids=[1])


def test_multi_diff_metrics_split_by_domain_and_skip_padding():
    teachers = _log_probs((2, 4, 3), 10)
    teacher_ref = _log_probs((4, 3), 11)
    old_log_probs = _log_probs((4, 3), 12)
    # Sample 3 is batch padding: an empty response that copies sample 0's route.
    route_idx = torch.tensor([0, 1, 0, 0])
    sample_valid = torch.tensor([True, True, True, False])
    response_mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    cfg = _config()
    effective = multi_diff_teacher_log_probs(teachers, teacher_ref, route_idx, route_mode="hard", adv_scales=(1, 1))
    out = _advantage(cfg, effective, teacher_ref, response_mask, old_log_probs)
    entropy = torch.stack([torch.full((4, 3), 2.0), torch.full((4, 3), 5.0)])
    metrics = multi_diff_metrics(
        out,
        response_mask,
        route_idx,
        cfg.teacher_names,
        teacher_log_probs=teachers,
        teacher_ref_log_probs=teacher_ref,
        sample_valid=sample_valid,
        teacher_entropy=entropy,
        teacher_ref_entropy=torch.ones(4, 3),
    )

    assert metrics["opd/math/frac"] == pytest.approx(2 / 3)
    assert metrics["opd/code/frac"] == pytest.approx(1 / 3)
    assert metrics["opd/math/response_len_mean"] == pytest.approx(2.0)
    assert metrics["opd/code/response_len_mean"] == pytest.approx(2.0)
    math_mask = response_mask.bool() & (route_idx == 0).unsqueeze(-1)
    math_shift = (teachers[0] - teacher_ref)[math_mask]
    assert metrics["opd/math/teacher_shift_mean"] == pytest.approx(math_shift.mean().item())
    assert metrics["opd/math/raw_shift_abs_mean"] == pytest.approx(math_shift.abs().mean().item())
    assert metrics["opd/math/kl_mean"] == pytest.approx((old_log_probs - teachers[0])[math_mask].mean().item())
    assert metrics["opd/math/teacher_entropy"] == pytest.approx(2.0)
    assert metrics["opd/code/teacher_entropy"] == pytest.approx(5.0)
    assert metrics["opd/code/teacher_entropy_minus_ref"] == pytest.approx(4.0)
    assert not any("by_position" in key for key in metrics)


def test_token_set_shift_metrics_cover_only_listed_ids():
    shift = torch.tensor([[1.0, -2.0, 3.0, 4.0]])
    responses = torch.tensor([[7, 151667, 5, 151668]])
    response_mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    metrics = token_set_shift_metrics(shift, responses, response_mask, [151667, 151668])
    assert metrics["opd/special_token_frac"] == pytest.approx(1 / 3)
    assert metrics["opd/special_token_shift_mean"] == pytest.approx(-2.0)
    assert metrics["opd/special_token_shift_abs_mean"] == pytest.approx(2.0)
    assert token_set_shift_metrics(shift, responses, response_mask, []) == {}
