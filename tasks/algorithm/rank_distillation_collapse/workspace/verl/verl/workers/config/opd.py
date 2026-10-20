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

"""Config for colocated on-policy distillation (KL-in-reward)."""

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Optional

from verl.base_config import BaseConfig

__all__ = [
    "OPDConfig",
    "OPDFullVocabConfig",
    "OPDTeacherSpec",
    "OPD_STRATEGIES",
    "OPD_REWARD_WEIGHT_MODES",
    "OPD_FULL_VOCAB_TARGETS",
    "OPD_ROUTE_MODES",
]

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Token set used to build the token-level reward at each position:
#   only_stu      student top-k ids
#   only_tch      teacher top-k ids
#   intersection  ids in both top-k sets
#   union         ids in either top-k set
OPD_STRATEGIES = ("only_stu", "only_tch", "intersection", "union")

# How the per-(position, k) KL terms are weighted before summing over k.
OPD_REWARD_WEIGHT_MODES = ("student_p", "teacher_p", "none")

# Target distribution of full-vocabulary OPD:
#   rkl   KL(p_stu || p_tch)
#   diff  full-vocabulary Direct-OPD, KL(p_stu || q) with q ∝ p_S0 * (p_tch / p_tch_ref)^(1/beta),
#         where S0 is the frozen initial student (its hidden comes from the ref engine)
OPD_FULL_VOCAB_TARGETS = ("rkl", "diff")

# How Multi-Diff OPD combines the post-RL teachers' shifts on a sample:
#   hard  each sample takes the shift of the one teacher whose tags contain its route field
#   sum   every sample takes the sum of all teachers' shifts (ablation, no routing)
OPD_ROUTE_MODES = ("hard", "sum")


class OPDTeacherSpec(NamedTuple):
    """One post-RL teacher of Multi-Diff OPD: its name, HF path and the route-field values it owns."""

    name: str
    path: str
    tags: tuple[str, ...]


@dataclass
class OPDFullVocabConfig(BaseConfig):
    """Full-vocabulary OPD (``verl/trainer/opd/full_vocab.py``).

    The teacher exports only the hidden state feeding its LM head; the student
    projects both its own and the teacher's hidden onto the vocabulary chunk by
    chunk inside its fused forward and minimizes the exact per-token reverse KL
    with an analytic gradient. No ``[N, vocab]`` tensor is ever materialized.

    Args:
        enable (bool): Whether full-vocabulary OPD replaces the sampled/top-k reward.
        target (str): One of :data:`OPD_FULL_VOCAB_TARGETS`. ``diff`` (full-vocabulary
            Direct-OPD) additionally requires ``opd.teacher_ref_model_path`` and the
            reference policy (``actor.use_kl_loss=True``) so the frozen initial student
            is available as a scorer.
        chunk_size (int): Rows per ``[chunk, vocab]`` fp32 buffer in the student's
            fused LM head. Bounds peak memory only; it does not change the result.
        loss_coef (float): Weight of the aggregated reverse KL in the actor loss.
        beta (float): ``diff`` target only. Weight of the student's own shift in the
            residual ``beta * log(p_theta / p_S0) - log(p_T+ / p_T0)``, whose squared
            loss has the gradient of ``2 * beta^2 * KL(p_theta || q)`` with
            ``q ∝ p_S0 * (p_T+ / p_T0)^(1/beta)``. Larger ``beta`` keeps the student
            closer to S0. Only ``q`` changes; ``loss_coef=2*beta^2`` recovers the
            squared-residual scale.
        teacher_weights (Optional[dict[str, float]]): ``diff`` target with several
            ``opd.teachers`` only. Weight of each teacher's shift in the consensus
            ``sum_k w_k * log p_T_k - log p_T0``, keyed by teacher name; missing names keep
            ``1.0`` and the weights are normalized to sum to 1. ``None`` weights every
            teacher equally.
    """

    enable: bool = False
    target: str = "rkl"
    chunk_size: int = 256
    loss_coef: float = 1.0
    beta: float = 1.0
    teacher_weights: Optional[dict[str, float]] = None

    def __post_init__(self):
        if not self.enable:
            return
        if self.target not in OPD_FULL_VOCAB_TARGETS:
            raise ValueError(f"opd.full_vocab.target must be one of {OPD_FULL_VOCAB_TARGETS}, got {self.target!r}.")
        if self.chunk_size <= 0:
            raise ValueError(f"opd.full_vocab.chunk_size must be > 0, got {self.chunk_size}.")
        if self.loss_coef <= 0:
            raise ValueError(f"opd.full_vocab.loss_coef must be > 0, got {self.loss_coef}.")
        if self.beta <= 0:
            raise ValueError(f"opd.full_vocab.beta must be > 0, got {self.beta}.")
        if self.target != "diff" and self.beta != 1.0:
            raise ValueError(f"opd.full_vocab.beta only applies to target='diff', got beta={self.beta}.")
        if self.teacher_weights is not None:
            if self.target != "diff":
                raise ValueError("opd.full_vocab.teacher_weights only applies to target='diff'.")
            negative = {name: w for name, w in self.teacher_weights.items() if w < 0}
            if negative:
                raise ValueError(f"opd.full_vocab.teacher_weights must be >= 0, got {negative}.")


@dataclass
class OPDConfig(BaseConfig):
    """Colocated on-policy distillation, via advantages or direct top-K losses.

    The teacher is a forward-only inner engine worker living in the *actor's*
    resource pool, so it needs no dedicated GPUs (unlike ``distillation.*``,
    which requires a separate ``teacher_pool``). Teacher log-probs are turned
    into a token-level reward; the advantage is the negative student/teacher
    log-prob difference, applied through the policy gradient.

    Args:
        enable (bool): Whether colocated OPD is enabled.
        teacher_model_path (Optional[str]): HF path of the teacher. Must share the
            student's tokenizer. Direct top-K objectives permit different padded
            LM-head sizes, selecting candidates from valid tokenizer IDs only.
        objective (str): ``legacy`` preserves advantage-based OPD;
            ``adjacent_logistic`` is O3PD; ``topk_jsd`` matches the conditional
            candidate distributions. Direct objectives bypass PPO entirely.
        loss_coef (float): Positive coefficient for direct top-K objectives.
        mask_teacher_ties (bool): ``adjacent_logistic`` only. Drop adjacent pairs whose
            teacher scores are exactly equal: the teacher states no order there, and
            ``ordered_topk`` breaks the tie by token ID. The per-position mean still
            divides by ``K - 1``.
        teacher_fp32_head (bool): Direct objectives only. Compute the teacher's LM head
            as an fp32 product of its (upcast) hidden states and weights, so teacher
            logits are not rounded to the bf16 grid before top-K selection.
        top_k (int): Size of the per-position token set. ``0`` means sampled-token
            OPD: the reward uses only the rolled-out token (equivalent to K=1).
        strategy (str): One of :data:`OPD_STRATEGIES`. Ignored when ``top_k == 0``.
        reward_weight_mode (str): One of :data:`OPD_REWARD_WEIGHT_MODES`. With
            ``student_p`` or ``teacher_p``, use the corresponding token
            probabilities as weights. With ``is_norm=True``, those
            probabilities are renormalized over the candidate set; with
            ``is_norm=False``, they remain full-vocabulary probabilities.
            ``none`` uses uniform weights over valid candidates.
        is_norm (bool): For ``top_k > 0``, whether to renormalize the student and
            teacher log-probabilities over the selected candidate set before
            taking their difference, and use the same conditional student
            distribution for the actor PPO ratio and score gradient. With
            ``reward_weight_mode=student_p``, this is the exact conditional
            reverse-KL policy gradient for the fixed candidate set at the old
            policy. It is ignored for sampled-token OPD (``top_k == 0``).
        outside_bucket (bool): For ``top_k > 0`` with ``is_norm=False`` and
            ``reward_weight_mode='student_p'``, add the complement of the
            candidate set as one aggregated candidate. This retains the
            student/teacher probability-mass mismatch outside the selected
            tokens.
        teacher_temperature (float): Divides the teacher logits before
            ``log_softmax``. ``1.0`` disables scaling.
        chunk_size (int): Rows of the packed ``(total_nnz, vocab)`` logits handled
            per gather step. Bounds the peak vocab-sized buffer to
            ``chunk_size * vocab``; it does not change the result.
        kl_estimator (str): ``k1`` uses ``log p_stu - log p_tch`` directly.
        grpo_outcome_weight (float): Weight of the GRPO outcome term when
            ``algorithm.adv_estimator='opd_token_kl_plus_grpo'``.
        sampled_adv_clip_min (Optional[float]): For sampled-token OPD
            (``top_k == 0``), lower bound applied to the per-token advantage
            ``log p_tch - log p_stu``. ``None`` disables the lower clip.
        sampled_adv_clip_max (Optional[float]): Upper bound of the same
            advantage. ``None`` disables the upper clip.
        teacher_ref_model_path (Optional[str]): HF path of the teacher *before* RL.
            When set, sampled-token OPD becomes Direct-OPD: the per-token advantage
            is the teacher's RL-induced shift ``log p_tch - log p_tch_ref`` at the
            sampled token instead of ``log p_tch - log p_stu``. The sampled
            advantage clip then applies to this shift. Requires ``top_k == 0``.
            With ``full_vocab.target='diff'`` it is the ``T0`` scorer of the
            full-vocabulary Direct-OPD target.
        full_vocab (OPDFullVocabConfig): Full-vocabulary OPD. When enabled, the
            distillation gradient is the exact reverse KL over the whole vocabulary
            and ``top_k`` / ``strategy`` / the sampled-token knobs do not apply.
        teachers (list[dict]): Multi-Diff OPD. Post-RL teachers that share the
            reference ``teacher_ref_model_path``, each a mapping with ``name``, ``path``
            and ``tags``. A sample is routed to the teacher whose ``tags`` contain the
            sample's ``route_field`` value. Tags must not overlap; a single teacher may
            leave ``tags`` empty to take every sample. Replaces ``teacher_model_path``
            and requires ``teacher_ref_model_path`` and ``top_k == 0``. With
            ``full_vocab.target='diff'`` the teachers are not routed: every sample takes
            their consensus shift (``full_vocab.teacher_weights``) and tags must be empty.
        route_field (str): Per-sample dataset field read for routing (ms-swift's
            ``teacher_tag_key``).
        route_mode (str): One of :data:`OPD_ROUTE_MODES`.
        domain_adv_scale (dict[str, float]): Per-teacher multiplier of its shift;
            teachers left out keep ``1.0``.
        mask_token_ids (list[int]): Token ids whose sampled-token advantage is zeroed,
            e.g. chat-template tokens a teacher never saw. Requires ``top_k == 0``.
    """

    enable: bool = False
    teacher_model_path: Optional[str] = None
    objective: str = "legacy"
    loss_coef: float = 1.0
    mask_teacher_ties: bool = False
    teacher_fp32_head: bool = False
    top_k: int = 0
    strategy: str = "only_stu"
    reward_weight_mode: str = "student_p"
    is_norm: bool = False
    outside_bucket: bool = False
    teacher_temperature: float = 1.0
    chunk_size: int = 4096
    kl_estimator: str = "k1"
    grpo_outcome_weight: float = 1.0
    sampled_adv_clip_min: Optional[float] = None
    sampled_adv_clip_max: Optional[float] = None
    teacher_ref_model_path: Optional[str] = None
    full_vocab: OPDFullVocabConfig = field(default_factory=OPDFullVocabConfig)
    teachers: list[Any] = field(default_factory=list)
    route_field: str = "ability"
    route_mode: str = "hard"
    domain_adv_scale: Optional[dict[str, float]] = None
    mask_token_ids: list[int] = field(default_factory=list)

    def __post_init__(self):
        """Validate the OPD configuration."""
        if not self.enable:
            return
        if self.objective not in ("legacy", "adjacent_logistic", "topk_jsd"):
            raise ValueError(f"Unknown opd.objective: {self.objective!r}")
        if self.direct_objective:
            if not (self.loss_coef > 0 and self.loss_coef < float("inf")):
                raise ValueError("opd.loss_coef must be finite and positive")
            if self.top_k < 2 or self.strategy != "only_tch":
                raise ValueError("Direct top-K objectives require top_k >= 2 and strategy='only_tch'")
            if self.full_vocab.enable or self.teacher_ref_model_path or self.teachers or self.outside_bucket:
                raise ValueError("Direct top-K objectives require one teacher, no full_vocab or outside_bucket")
        if self.mask_teacher_ties and self.objective != "adjacent_logistic":
            raise ValueError("opd.mask_teacher_ties requires opd.objective='adjacent_logistic'")
        if self.teacher_fp32_head and not self.direct_objective:
            raise ValueError("opd.teacher_fp32_head requires a direct top-K objective")
        if self.full_vocab.enable:
            if self.top_k != 0:
                raise ValueError(f"opd.full_vocab.enable requires opd.top_k=0, got {self.top_k}.")
            if self.multi_diff and self.full_vocab.target != "diff":
                raise NotImplementedError(
                    "opd.teachers with full-vocabulary OPD is only implemented for target='diff', "
                    "where the teachers form a consensus shift."
                )
            if self.full_vocab.teacher_weights is not None and not self.multi_diff:
                raise ValueError("opd.full_vocab.teacher_weights weights the consensus of several opd.teachers.")
            if self.direct_opd != (self.full_vocab.target == "diff"):
                raise ValueError(
                    "opd.full_vocab.target='diff' (full-vocabulary Direct-OPD) must be selected together with "
                    f"opd.teacher_ref_model_path, got target={self.full_vocab.target!r} and "
                    f"teacher_ref_model_path={self.teacher_ref_model_path!r}."
                )
            if self.sampled_adv_clip_enabled:
                logger.warning("opd.sampled_adv_clip_min/max do not apply to full-vocabulary OPD; ignored.")
        if self.multi_diff:
            self._validate_multi_diff()
        elif self.teacher_model_path is None:
            raise ValueError("opd.teacher_model_path (or opd.teachers) must be set when opd.enable=True.")
        if self.mask_token_ids and self.top_k != 0:
            raise ValueError("opd.mask_token_ids only applies to sampled-token OPD (top_k=0).")
        if self.top_k < 0:
            raise ValueError(f"opd.top_k must be >= 0, got {self.top_k}.")
        if self.strategy not in OPD_STRATEGIES:
            raise ValueError(f"opd.strategy must be one of {OPD_STRATEGIES}, got {self.strategy!r}.")
        if self.reward_weight_mode not in OPD_REWARD_WEIGHT_MODES:
            raise ValueError(
                f"opd.reward_weight_mode must be one of {OPD_REWARD_WEIGHT_MODES}, got {self.reward_weight_mode!r}."
            )
        if self.teacher_temperature <= 0:
            raise ValueError(f"opd.teacher_temperature must be > 0, got {self.teacher_temperature}.")
        if self.chunk_size <= 0:
            raise ValueError(f"opd.chunk_size must be > 0, got {self.chunk_size}.")
        if self.kl_estimator != "k1":
            raise NotImplementedError(f"Only kl_estimator='k1' is supported, got {self.kl_estimator!r}.")
        if self.top_k == 0 and self.strategy != "only_stu":
            logger.warning(
                "opd.top_k=0 selects sampled-token OPD; opd.strategy=%r is ignored.",
                self.strategy,
            )
        if (
            self.sampled_adv_clip_min is not None
            and self.sampled_adv_clip_max is not None
            and self.sampled_adv_clip_min >= self.sampled_adv_clip_max
        ):
            raise ValueError(
                "opd.sampled_adv_clip_min must be < opd.sampled_adv_clip_max, got "
                f"{self.sampled_adv_clip_min} and {self.sampled_adv_clip_max}."
            )
        if self.top_k > 0 and self.sampled_adv_clip_enabled:
            logger.warning("opd.sampled_adv_clip_min/max only apply to sampled-token OPD (top_k=0); ignored.")
        if self.outside_bucket:
            if self.top_k == 0:
                raise ValueError("opd.outside_bucket requires opd.top_k > 0.")
            if self.is_norm:
                raise ValueError("opd.outside_bucket requires opd.is_norm=False.")
            if self.reward_weight_mode != "student_p":
                raise ValueError("opd.outside_bucket currently requires opd.reward_weight_mode='student_p'.")
        if self.direct_opd and self.top_k != 0:
            raise ValueError(
                f"opd.teacher_ref_model_path (Direct-OPD) is only implemented for opd.top_k=0, got {self.top_k}."
            )

    def _validate_multi_diff(self) -> None:
        if self.teacher_model_path is not None:
            raise ValueError("opd.teachers replaces opd.teacher_model_path; set only one of them.")
        if not self.direct_opd:
            raise ValueError("opd.teachers (Multi-Diff OPD) requires the shared opd.teacher_ref_model_path.")
        if self.top_k != 0:
            raise ValueError(f"opd.teachers (Multi-Diff OPD) requires opd.top_k=0, got {self.top_k}.")

        specs = self.teacher_specs
        names = [spec.name for spec in specs]
        if len(set(names)) != len(names):
            raise ValueError(f"opd.teachers names must be unique, got {names}.")
        if self.full_vocab_consensus:
            self._validate_consensus(names)
            return

        if self.route_mode not in OPD_ROUTE_MODES:
            raise ValueError(f"opd.route_mode must be one of {OPD_ROUTE_MODES}, got {self.route_mode!r}.")
        if not self.route_field:
            raise ValueError("opd.route_field must name a per-sample dataset field.")
        owner: dict[str, str] = {}
        for spec in specs:
            if not spec.tags and len(specs) > 1:
                raise ValueError(
                    f"opd.teachers[{spec.name!r}] has no tags; only a single teacher may take every sample."
                )
            for tag in spec.tags:
                if tag in owner:
                    raise ValueError(f"opd.teachers tag {tag!r} is claimed by both {owner[tag]!r} and {spec.name!r}.")
                owner[tag] = spec.name
        scales = self.domain_adv_scale or {}
        unknown = set(scales) - set(names)
        if unknown:
            raise ValueError(f"opd.domain_adv_scale names unknown teachers {sorted(unknown)}; teachers are {names}.")
        negative = {name: scale for name, scale in scales.items() if scale < 0}
        if negative:
            raise ValueError(f"opd.domain_adv_scale must be >= 0, got {negative}.")

    def _validate_consensus(self, names: list[str]) -> None:
        tagged = [spec.name for spec in self.teacher_specs if spec.tags]
        if tagged:
            raise ValueError(
                f"opd.teachers {tagged} have tags, but the full-vocabulary consensus applies every teacher "
                "to every sample; leave tags empty."
            )
        if self.domain_adv_scale:
            raise ValueError(
                "opd.domain_adv_scale scales routed Multi-Diff OPD shifts; weight the full-vocabulary "
                "consensus with opd.full_vocab.teacher_weights instead."
            )
        # Each name becomes the scorer role `teacher_<name>`, i.e. a TransferQueue field suffix.
        invalid = [name for name in names if not name.isidentifier()]
        if invalid:
            raise ValueError(
                f"opd.teachers names must be identifiers under the full-vocabulary consensus, got {invalid}."
            )
        weights = self.full_vocab.teacher_weights or {}
        unknown = set(weights) - set(names)
        if unknown:
            raise ValueError(
                f"opd.full_vocab.teacher_weights names unknown teachers {sorted(unknown)}; teachers are {names}."
            )
        if sum(float(weights.get(name, 1.0)) for name in names) <= 0:
            raise ValueError(f"opd.full_vocab.teacher_weights must not all be zero, got {weights}.")

    @property
    def direct_objective(self) -> bool:
        """A differentiable top-K objective, independent of PPO advantages."""
        return self.objective in ("adjacent_logistic", "topk_jsd")

    @property
    def multi_diff(self) -> bool:
        """Whether Multi-Diff OPD is configured: several post-RL teachers against one shared reference."""
        return bool(self.teachers)

    @property
    def teacher_specs(self) -> tuple[OPDTeacherSpec, ...]:
        """The post-RL teachers in order; the legacy ``teacher_model_path`` is a single untagged teacher."""
        if not self.teachers:
            if self.teacher_model_path is None:
                return ()
            return (OPDTeacherSpec("default", self.teacher_model_path, ()),)
        specs = []
        for i, entry in enumerate(self.teachers):
            if not isinstance(entry, Mapping):
                raise ValueError(f"opd.teachers[{i}] must be a mapping with name/path/tags, got {entry!r}.")
            extra = set(entry) - {"name", "path", "tags"}
            if extra:
                raise ValueError(f"opd.teachers[{i}] has unknown keys {sorted(extra)}.")
            name, path = entry.get("name"), entry.get("path")
            if not name or not path:
                raise ValueError(f"opd.teachers[{i}] needs a non-empty name and path, got {dict(entry)!r}.")
            tags = entry.get("tags") or ()
            if isinstance(tags, str):
                tags = (tags,)
            specs.append(OPDTeacherSpec(str(name), str(path), tuple(str(tag) for tag in tags)))
        return tuple(specs)

    @property
    def teacher_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.teacher_specs)

    @property
    def needs_route_field(self) -> bool:
        """Whether routing reads ``route_field``; a single untagged teacher takes every sample."""
        specs = self.teacher_specs
        return not (len(specs) == 1 and not specs[0].tags)

    def route_indices(self, route_values) -> list[int]:
        """Map each sample's route-field value to the index of its teacher in :attr:`teacher_specs`."""
        specs = self.teacher_specs
        if not self.needs_route_field:
            return [0] * len(route_values)
        owner = {tag: i for i, spec in enumerate(specs) for tag in spec.tags}
        indices = []
        for value in route_values:
            idx = owner.get(str(value))
            if idx is None:
                tags = {spec.name: list(spec.tags) for spec in specs}
                raise ValueError(
                    f"opd.route_field={self.route_field!r} value {value!r} matches no teacher tags {tags}."
                )
            indices.append(idx)
        return indices

    @property
    def teacher_adv_scales(self) -> tuple[float, ...]:
        """Per-teacher shift multipliers aligned with :attr:`teacher_specs`."""
        scales = self.domain_adv_scale or {}
        return tuple(float(scales.get(name, 1.0)) for name in self.teacher_names)

    @property
    def direct_opd(self) -> bool:
        """Whether the advantage is the teacher's RL shift (Direct-OPD) rather than the student/teacher gap."""
        return bool(self.teacher_ref_model_path)

    @property
    def full_vocab_diff(self) -> bool:
        """Whether full-vocabulary Direct-OPD is configured: KL(p_stu || q), q ∝ p_S0 * (p_T+ / p_T0)^(1/beta)."""
        return self.full_vocab.enable and self.full_vocab.target == "diff"

    @property
    def full_vocab_consensus(self) -> bool:
        """Whether full-vocabulary Direct-OPD takes the consensus shift of the ``opd.teachers``."""
        return self.full_vocab_diff and self.multi_diff

    @property
    def consensus_teacher_weights(self) -> tuple[float, ...]:
        """Consensus weights aligned with :attr:`teacher_specs`, normalized to sum to 1."""
        weights = self.full_vocab.teacher_weights or {}
        raw = [float(weights.get(name, 1.0)) for name in self.teacher_names]
        total = sum(raw)
        return tuple(w / total for w in raw)

    @property
    def sampled_adv_clip_enabled(self) -> bool:
        """Whether either sampled-token advantage clip bound is set."""
        return self.sampled_adv_clip_min is not None or self.sampled_adv_clip_max is not None

    @property
    def num_tokens(self) -> int:
        """Width of the K dimension carried through the reward and advantage."""
        return 1 if self.top_k == 0 else self.top_k

    @property
    def needs_teacher_topk(self) -> bool:
        """Whether the teacher's own top-k ids/log-probs are required."""
        return self.top_k > 0 and self.strategy in ("only_tch", "intersection", "union")

    @property
    def needs_teacher_on_student(self) -> bool:
        """Whether teacher log-probs gathered at the student's ids are required."""
        return self.top_k == 0 or self.strategy in ("only_stu", "intersection", "union")

    @property
    def needs_student_on_teacher(self) -> bool:
        """Whether a student forward pass at the teacher's top-k ids is required."""
        return self.top_k > 0 and self.strategy in ("only_tch", "union")

    @property
    def advantage_width(self) -> int:
        """K dimension of the advantage tensor (union concatenates both sets)."""
        return 2 * self.top_k if (self.top_k > 0 and self.strategy == "union") else self.num_tokens
