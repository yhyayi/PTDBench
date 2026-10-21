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
from dataclasses import dataclass
from typing import Optional

from verl.base_config import BaseConfig

__all__ = ["OPDConfig", "OPD_STRATEGIES", "OPD_REWARD_WEIGHT_MODES"]

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


@dataclass
class OPDConfig(BaseConfig):
    """Colocated on-policy distillation via KL-in-reward.

    The teacher is a forward-only inner engine worker living in the *actor's*
    resource pool, so it needs no dedicated GPUs (unlike ``distillation.*``,
    which requires a separate ``teacher_pool``). Teacher log-probs are turned
    into a token-level reward; the advantage is the negative student/teacher
    log-prob difference, applied through the policy gradient.

    Args:
        enable (bool): Whether colocated OPD is enabled.
        teacher_model_path (Optional[str]): HF path of the teacher. Must share the
            student's tokenizer and vocab size.
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
    """

    enable: bool = False
    teacher_model_path: Optional[str] = None
    top_k: int = 0
    strategy: str = "only_stu"
    reward_weight_mode: str = "student_p"
    is_norm: bool = False
    outside_bucket: bool = False
    teacher_temperature: float = 1.0
    chunk_size: int = 4096
    kl_estimator: str = "k1"
    grpo_outcome_weight: float = 1.0

    def __post_init__(self):
        """Validate the OPD configuration."""
        if not self.enable:
            return
        if self.teacher_model_path is None:
            raise ValueError("opd.teacher_model_path must be set when opd.enable=True.")
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
        if self.outside_bucket:
            if self.top_k == 0:
                raise ValueError("opd.outside_bucket requires opd.top_k > 0.")
            if self.is_norm:
                raise ValueError("opd.outside_bucket requires opd.is_norm=False.")
            if self.reward_weight_mode != "student_p":
                raise ValueError("opd.outside_bucket currently requires opd.reward_weight_mode='student_p'.")

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
