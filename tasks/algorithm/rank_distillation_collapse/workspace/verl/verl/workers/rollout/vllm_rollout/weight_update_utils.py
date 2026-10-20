# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import torch

try:
    from vllm.model_executor.models.utils import _get_tied_embedding_params
except ImportError:
    _get_tied_embedding_params = None

WeightUpdate = tuple[str, torch.Tensor]


def resolve_tied_aliases(model: torch.nn.Module, weights: list[WeightUpdate]) -> list[WeightUpdate]:
    """Rename tied-embedding aliases (e.g. ``lm_head.weight``) to their canonical name.

    vLLM >= 0.29 raises when one ``load_weights`` call carries a tied alias without
    its canonical name, which bucketed transfers routinely do. The alias holds the
    same tensor, so loading it under the canonical name is equivalent.

    vLLM keys the aliases by its own qualnames, while ``load_weights`` receives
    checkpoint names and maps them through ``hf_to_vllm_mapper`` first (VL wrappers
    such as Qwen3.5 map ``lm_head.`` to ``language_model.lm_head.``), so the lookup
    happens on mapped names and the rename is mapped back to a checkpoint name.
    """
    aliases = _get_tied_embedding_params(model) if _get_tied_embedding_params is not None else {}
    if not aliases:
        return weights
    mapper = getattr(model, "hf_to_vllm_mapper", None)
    mapped = [mapper._map_name(name) if mapper is not None else name for name, _ in weights]
    present = set(mapped)
    resolved = []
    for (name, tensor), vllm_name in zip(weights, mapped, strict=True):
        canonical = aliases.get(vllm_name)
        if canonical is None:
            resolved.append((name, tensor))
        elif canonical not in present:
            resolved.append((_to_checkpoint_name(mapper, canonical), tensor))
    return resolved


def _to_checkpoint_name(mapper, vllm_name: str) -> str:
    """Invert ``mapper``'s prefix renames so that ``load_weights`` maps the result back to ``vllm_name``."""
    if mapper is None:
        return vllm_name
    for orig, new in mapper.orig_to_new_prefix.items():
        if new and vllm_name.startswith(new):
            candidate = orig + vllm_name[len(new) :]
            if mapper._map_name(candidate) == vllm_name:
                return candidate
    return vllm_name


def split_buffer_updates(
    model: torch.nn.Module, weights: list[WeightUpdate]
) -> tuple[list[WeightUpdate], list[WeightUpdate], dict[str, torch.Tensor]]:
    """Split incoming weight updates into parameter and buffer updates.

    Returns the parameter updates, the buffer updates, and the model's
    ``named_buffers`` map so callers can reuse it without re-iterating.
    """
    named_buffers = dict(model.named_buffers())
    param_updates, buffer_updates = [], []
    for name, tensor in weights:
        if name in named_buffers:
            buffer_updates.append((name, tensor))
        else:
            param_updates.append((name, tensor))
    return param_updates, buffer_updates, named_buffers


@torch.no_grad()
def apply_buffer_updates(
    model: torch.nn.Module,
    buffer_updates: list[WeightUpdate],
    named_buffers: dict[str, torch.Tensor] | None = None,
) -> int:
    """Copy updated buffer tensors into the target model in-place."""
    if not buffer_updates:
        return 0

    if named_buffers is None:
        named_buffers = dict(model.named_buffers())
    loaded = 0
    for name, tensor in buffer_updates:
        if name not in named_buffers:
            continue

        target = named_buffers[name]
        if target.shape != tensor.shape:
            raise ValueError(
                f"Buffer shape mismatch for {name}: expected {tuple(target.shape)}, got {tuple(tensor.shape)}"
            )

        source = tensor.to(device=target.device, dtype=target.dtype, non_blocking=False)
        target.copy_(source, non_blocking=False)
        loaded += 1

    return loaded
