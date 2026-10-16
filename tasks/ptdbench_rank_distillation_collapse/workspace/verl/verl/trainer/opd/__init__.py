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

"""Colocated on-policy distillation (OPD) via KL-in-reward.

The teacher runs as a forward-only inner engine worker inside the *actor's*
resource pool, so student and teacher share the same GPUs. Only ``[B, T, K]``
tensors (candidate ids + their log-probs) ever leave the device.

Only the driver-side pieces are re-exported here. ``losses`` and ``kernels`` are
worker-side and pull in engine dependencies, so they stay submodule-only imports.
"""

from .advantage import (
    build_opd_candidates,
    direct_opd_metrics,
    multi_diff_metrics,
    multi_diff_teacher_log_probs,
    opd_advantage,
    opd_kl_bucket_metrics,
    opd_topk_overlap_ratio,
    resolve_bucket_means,
    route_teacher_log_probs,
    token_set_shift_metrics,
)

__all__ = [
    "build_opd_candidates",
    "direct_opd_metrics",
    "multi_diff_metrics",
    "multi_diff_teacher_log_probs",
    "opd_advantage",
    "opd_kl_bucket_metrics",
    "opd_topk_overlap_ratio",
    "resolve_bucket_means",
    "route_teacher_log_probs",
    "token_set_shift_metrics",
]
