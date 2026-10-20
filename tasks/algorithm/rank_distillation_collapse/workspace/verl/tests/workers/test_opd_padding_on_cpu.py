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
"""CPU tests for moving OPD's ``(B, T, W)`` candidate ids through the packed path.

Candidate ids are produced response-framed on the driver, packed for the engine,
shifted so each row lands on the logits row that *predicts* it, and read back by
``no_padding_2_padding``. A one-off error here silently trains on the wrong
tokens, so the round trip is pinned end to end.
"""

import torch
from tensordict import TensorDict

from verl.workers.utils.padding import (
    left_right_2_no_padding,
    no_padding_2_padding,
    response_from_nested,
    response_to_full_nested,
)


def _batch(prompt_lens, response_lens, width, vocab=97, seed=0):
    """Build a left-padded-prompt / right-padded-response batch, as rollout emits."""
    gen = torch.Generator().manual_seed(seed)
    b = len(prompt_lens)
    max_prompt, max_response = max(prompt_lens), max(response_lens)
    total = max_prompt + max_response
    input_ids = torch.randint(1, vocab, (b, total), generator=gen)
    attention_mask = torch.zeros(b, total, dtype=torch.int64)
    for i, (p_len, r_len) in enumerate(zip(prompt_lens, response_lens, strict=True)):
        attention_mask[i, max_prompt - p_len : max_prompt + r_len] = 1
    position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0) * attention_mask
    prompts = input_ids[:, :max_prompt].clone()
    responses = input_ids[:, max_prompt:].clone()
    response_mask = attention_mask[:, max_prompt:].clone()
    cand_ids = torch.randint(0, vocab, (b, max_response, width), generator=gen)
    cand_ids = cand_ids * response_mask.unsqueeze(-1)  # padding slots carry nothing
    return TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "prompts": prompts,
            "responses": responses,
            "response_mask": response_mask,
            "opd_cand_ids": cand_ids,
        },
        batch_size=[b],
    )


def _engine_roundtrip(batch_td):
    """Mimic what the FSDP engine does to ``opd_cand_ids``, then read it back.

    ``prepare_model_inputs`` rolls the packed rows by -1 so row ``i`` holds the ids
    predicted *from* position ``i``; the model's output at that row is what
    ``no_padding_2_padding`` shifts back into the response frame.
    """
    nested = batch_td["opd_cand_ids"]
    packed = nested.values().reshape(-1, nested.values().shape[-1])
    rolled = torch.roll(packed, shifts=-1, dims=0)
    cu_seqlens = batch_td["input_ids"].offsets()
    out = torch.nested.nested_tensor_from_jagged(rolled, cu_seqlens)
    return no_padding_2_padding(out, batch_td)


def test_candidate_ids_survive_the_packed_round_trip():
    batch_td = _batch([5, 2, 7], [4, 6, 3], width=3, seed=1)
    original = batch_td["opd_cand_ids"].clone()
    response_mask = batch_td["response_mask"]

    packed_td = left_right_2_no_padding(batch_td)
    assert packed_td["opd_cand_ids"].is_nested, "candidate ids must be packed alongside input_ids"
    recovered = _engine_roundtrip(packed_td)

    assert recovered.shape == original.shape
    valid = response_mask.bool().unsqueeze(-1).expand_as(original)
    torch.testing.assert_close(recovered[valid], original[valid])
    # Positions past each sequence's response length must stay empty.
    assert (recovered[~valid] == 0).all()


def test_round_trip_is_exact_for_ragged_and_single_token_responses():
    for seed, prompt_lens, response_lens in [
        (2, [1, 9], [1, 8]),  # a length-1 response next to a long one
        (3, [4, 4, 4], [5, 5, 5]),  # uniform lengths (no padding at all)
        (4, [8, 1, 3, 6], [2, 7, 1, 4]),
    ]:
        batch_td = _batch(prompt_lens, response_lens, width=2, seed=seed)
        original = batch_td["opd_cand_ids"].clone()
        valid = batch_td["response_mask"].bool().unsqueeze(-1).expand_as(original)
        recovered = _engine_roundtrip(left_right_2_no_padding(batch_td))
        torch.testing.assert_close(recovered[valid], original[valid])


def test_width_one_matches_the_token_level_path():
    """W=1 (top_k=0) must behave exactly like the existing 2D log-prob round trip."""
    batch_td = _batch([3, 6], [5, 2], width=1, seed=5)
    original = batch_td["opd_cand_ids"].clone()
    valid = batch_td["response_mask"].bool().unsqueeze(-1).expand_as(original)
    recovered = _engine_roundtrip(left_right_2_no_padding(batch_td))
    torch.testing.assert_close(recovered[valid], original[valid])
    assert recovered.dtype == torch.int64


def test_int_dtype_and_extra_keys_are_preserved():
    batch_td = _batch([4, 4], [3, 6], width=4, seed=6)
    packed_td = left_right_2_no_padding(batch_td)
    assert packed_td["opd_cand_ids"].values().dtype == torch.int64
    # The packed candidate rows line up with the packed token rows.
    assert packed_td["opd_cand_ids"].values().shape[0] == packed_td["input_ids"].values().shape[0]


def test_v1_response_candidates_expand_to_full_sequence_and_roundtrip():
    input_ids = torch.nested.as_nested_tensor(
        [torch.tensor([10, 11, 12, 13, 14]), torch.tensor([20, 21, 22])], layout=torch.jagged
    )
    response_mask = torch.nested.as_nested_tensor(
        [torch.ones(2, dtype=torch.int64), torch.ones(1, dtype=torch.int64)], layout=torch.jagged
    )
    candidates = torch.tensor(
        [
            [[31, 32], [33, 34]],
            [[41, 42], [0, 0]],
        ]
    )

    full = response_to_full_nested(candidates, input_ids=input_ids, response_mask=response_mask)
    torch.testing.assert_close(full.offsets(), input_ids.offsets())
    torch.testing.assert_close(
        full.values(),
        torch.tensor(
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [31, 32],
                [33, 34],
                [0, 0],
                [0, 0],
                [41, 42],
            ]
        ),
    )

    # The engine rolls candidate rows by -1; response_from_nested then reads the
    # logits rows that predict the response and must recover the original frame.
    rolled = torch.roll(full.values(), shifts=-1, dims=0)
    model_output = torch.nested.nested_tensor_from_jagged(rolled, offsets=input_ids.offsets())
    recovered = response_from_nested(model_output, response_mask).to_padded_tensor(0)
    torch.testing.assert_close(recovered, candidates)


def test_v1_response_candidates_accept_nested_input():
    input_ids = torch.nested.as_nested_tensor(
        [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6, 7])], layout=torch.jagged
    )
    response_mask = torch.nested.as_nested_tensor(
        [torch.ones(1, dtype=torch.int64), torch.ones(2, dtype=torch.int64)], layout=torch.jagged
    )
    candidates = torch.nested.as_nested_tensor(
        [torch.tensor([[8, 9]]), torch.tensor([[10, 11], [12, 13]])], layout=torch.jagged
    )
    full = response_to_full_nested(candidates, input_ids=input_ids, response_mask=response_mask)
    assert full.values().shape == (7, 2)
    torch.testing.assert_close(full.values()[-2:], torch.tensor([[10, 11], [12, 13]]))
