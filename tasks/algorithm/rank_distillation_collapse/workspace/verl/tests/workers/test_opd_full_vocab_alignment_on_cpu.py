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
"""CPU tests for the full-vocabulary OPD frame round trip in the FSDP engine.

The teacher's packed hidden row ``p`` predicts token ``p + 1``. After it is cut to
the response frame on the teacher side and re-packed on the student side, every
student logits row must see the teacher hidden that predicts the same token.
"""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

H = 4
PROMPT_LENS = [3, 5, 2]
RESP_LENS = [4, 1, 6]


def _micro_batch():
    seq_lens = [p + r for p, r in zip(PROMPT_LENS, RESP_LENS, strict=True)]
    total = sum(seq_lens)
    input_ids = torch.arange(100, 100 + total)
    seq_offsets = torch.tensor([0, *torch.tensor(seq_lens).cumsum(0).tolist()])
    resp_offsets = torch.tensor([0, *torch.tensor(RESP_LENS).cumsum(0).tolist()])
    response_mask = torch.ones(sum(RESP_LENS), dtype=torch.int64)
    response_mask[resp_offsets[2] + 1] = 0  # a masked-out (e.g. tool) token
    return TensorDict(
        {
            "input_ids": torch.nested.nested_tensor_from_jagged(input_ids, seq_offsets),
            "response_mask": torch.nested.nested_tensor_from_jagged(response_mask, resp_offsets),
        },
        batch_size=[len(seq_lens)],
    )


def _stub(heads=None):
    return SimpleNamespace(
        use_ulysses_sp=False,
        opd_frozen_heads=heads or {},
        _gather_and_unpad_packed=lambda tensor, pad_size: tensor[:-pad_size] if pad_size else tensor,
    )


def _teacher_hidden(micro_batch, pad_size):
    # Row p encodes the id of the token it predicts (p + 1 in the packed stream).
    ids = micro_batch["input_ids"].values()
    predicted = torch.roll(ids, shifts=-1).float()
    hidden = predicted.unsqueeze(-1).expand(-1, H).clone()
    return torch.nn.functional.pad(hidden, (0, 0, 0, pad_size)).unsqueeze(0)


@pytest.mark.parametrize("static_pad_size", [0, 3])
def test_round_trip(static_pad_size):
    micro_batch = _micro_batch()
    output = SimpleNamespace(lm_head_hidden=_teacher_hidden(micro_batch, static_pad_size))
    teacher_out = FSDPEngineWithLMHead._opd_full_vocab_hidden_output(
        _stub(), output, {"pad_size": static_pad_size}, micro_batch
    )
    response_hidden = teacher_out["opd_hidden"]
    assert response_hidden.dtype == torch.bfloat16
    assert response_hidden.offsets().diff().tolist() == RESP_LENS

    # Teacher side: response row t holds the hidden that predicts response token t.
    ids = micro_batch["input_ids"]
    for b, (seq, hid) in enumerate(zip(ids.unbind(), response_hidden.unbind(), strict=True)):
        response_tokens = seq[PROMPT_LENS[b] :].float()
        torch.testing.assert_close(hid[:, 0].float(), response_tokens)

    # Student side: re-pack onto the logits rows and check against the original frame.
    micro_batch["opd_hidden_teacher"] = response_hidden
    tu.assign_non_tensor(micro_batch, opd_teacher_temperature=1.5, opd_full_vocab_chunk_size=7)
    head = torch.randn(11, H)
    target = FSDPEngineWithLMHead._build_opd_full_vocab_target(
        _stub({"teacher": head}), micro_batch, {"teacher": 1.0}, static_pad_size=static_pad_size, sp_pad_size=0
    )
    assert target.temperature == 1.5 and target.chunk_size == 7
    (scorer,) = target.scorers
    assert scorer.head is head
    total = ids.values().numel()
    assert scorer.hidden.shape == (total + static_pad_size, H)
    assert target.row_mask.shape == (total + static_pad_size,)

    full_frame = _teacher_hidden(micro_batch, static_pad_size).squeeze(0)
    expected_mask = torch.zeros(total + static_pad_size, dtype=torch.bool)
    seq_end = 0
    resp_cursor = 0
    response_mask = micro_batch["response_mask"].values().bool()
    for p_len, r_len in zip(PROMPT_LENS, RESP_LENS, strict=True):
        seq_end += p_len + r_len
        rows = slice(seq_end - r_len - 1, seq_end - 1)
        expected_mask[rows] = response_mask[resp_cursor : resp_cursor + r_len]
        torch.testing.assert_close(scorer.hidden[rows].float(), full_frame[rows])
        resp_cursor += r_len
    assert torch.equal(target.row_mask, expected_mask)
    # Prompt and pad rows carry no target and stay zero.
    outside = torch.ones_like(expected_mask)
    seq_end = 0
    for p_len, r_len in zip(PROMPT_LENS, RESP_LENS, strict=True):
        seq_end += p_len + r_len
        outside[seq_end - r_len - 1 : seq_end - 1] = False
    assert (scorer.hidden[outside] == 0).all()


def test_missing_head_raises():
    micro_batch = _micro_batch()
    output = SimpleNamespace(lm_head_hidden=_teacher_hidden(micro_batch, 0))
    micro_batch["opd_hidden_teacher"] = FSDPEngineWithLMHead._opd_full_vocab_hidden_output(
        _stub(), output, {"pad_size": 0}, micro_batch
    )["opd_hidden"]
    with pytest.raises(RuntimeError, match="no frozen LM head"):
        FSDPEngineWithLMHead._build_opd_full_vocab_target(
            _stub(), micro_batch, {"teacher": 1.0}, static_pad_size=0, sp_pad_size=0
        )
