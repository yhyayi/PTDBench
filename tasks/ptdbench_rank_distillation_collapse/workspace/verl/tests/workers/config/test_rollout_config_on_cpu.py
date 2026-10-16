# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import pytest
from omegaconf import OmegaConf

from verl.workers.config import RolloutConfig, SamplingConfig, get_rollout_response_length


def test_validation_response_length_defaults_to_training_length():
    config = RolloutConfig(name="vllm", response_length=5120)

    assert get_rollout_response_length(config, validate=False) == 5120
    assert get_rollout_response_length(config, validate=True) == 5120


def test_legacy_config_without_validation_kwargs_uses_training_length():
    config = OmegaConf.create({"response_length": 5120})

    assert get_rollout_response_length(config, validate=True) == 5120


def test_validation_response_length_can_be_configured_independently():
    config = RolloutConfig(
        name="vllm",
        response_length=5120,
        val_kwargs=SamplingConfig(response_length=32768),
    )

    assert get_rollout_response_length(config, validate=False) == 5120
    assert get_rollout_response_length(config, validate=True) == 32768


@pytest.mark.parametrize("response_length", [0, -1])
def test_validation_response_length_must_be_positive(response_length):
    with pytest.raises(ValueError, match="positive integer or null"):
        SamplingConfig(response_length=response_length)


def test_raw_validation_response_length_must_be_positive():
    config = OmegaConf.create({"response_length": 5120, "val_kwargs": {"response_length": 0}})

    with pytest.raises(ValueError, match="positive integer"):
        get_rollout_response_length(config, validate=True)
