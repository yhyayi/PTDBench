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

"""
Apply monkey-patch function to models
"""
import logging
import os
import sys
from types import MethodType, SimpleNamespace
from typing import Optional

import torch
import torch.distributed as dist
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel

from verl.utils.import_utils import is_trl_available
from verl.utils.transformers_compat import is_transformers_version_in_range
from verl.utils.ulysses import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_world_size,
    slice_input_tensor,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_PREFIX_GROUPER_PATCHED = False
_PREFIX_GROUPER_SUPPORTED_ATTENTIONS = {
    "flash_attention_2",
    "flash_attention_3",
    "sdpa",
    "flex_attention",
    "eager",
}


def _create_prefix_grouper_wrapper(original_fn):
    """Wrap attention function to support prefix_grouper in kwargs.

    The prefix_grouper parameter is passed through kwargs from the model's
    forward to the attention layers. This wrapper strips it before calling
    the original attention function so that it doesn't cause unexpected
    keyword argument errors.
    """

    def wrapper(*args, **kwargs):
        # Remove prefix_grouper from kwargs if present; it is consumed
        # by the attention mask / position_ids logic before reaching
        # the attention function.
        kwargs.pop("prefix_grouper", None)
        return original_fn(*args, **kwargs)

    return wrapper


def apply_prefix_grouper_patch():
    """Patch ALL_ATTENTION_FUNCTIONS to support prefix_grouper parameter.

    The prefix_grouper is a data structure passed through the model forward
    to enable grouped prefix processing in GRPO/PPO training. Since the
    HuggingFace attention functions don't accept this parameter, we wrap
    them to silently drop it.
    """
    global _PREFIX_GROUPER_PATCHED
    if _PREFIX_GROUPER_PATCHED:
        return

    for name in _PREFIX_GROUPER_SUPPORTED_ATTENTIONS:
        if name in ALL_ATTENTION_FUNCTIONS:
            ALL_ATTENTION_FUNCTIONS[name] = _create_prefix_grouper_wrapper(
                ALL_ATTENTION_FUNCTIONS[name]
            )

    _PREFIX_GROUPER_PATCHED = True


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to
    (batch, num_attention_heads, seqlen, head_dim).

    This is copied from transformers.models.llama.modeling_llama.repeat_kv.
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _ulysses_flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    query_length: int,
    *args,
    position_ids: Optional[torch.Tensor] = None,
    **kwargs,
):
    """Insert all-to-all before and after flash attention.

    DeepSpeed-Ulysses: https://arxiv.org/pdf/2309.14509

    For transformers>=4.55, the flash attention api has changed,
    we need to pass the query_length after doing ulysses all2all.
    See https://github.com/huggingface/transformers/issues/40399

    Args:
        query_states (torch.Tensor): (batch_size, seqlen/sp_size, nheads, head_dim)
        key_states (torch.Tensor): (batch_size, seqlen/sp_size, nheads_k, head_dim)
        value_states (torch.Tensor): (batch_size, seqlen/sp_size, nheads_k, head_dim)
        position_ids (torch.Tensor, optional): (batch_size, seqlen/sp_size)

    Returns:
        torch.Tensor: (batch_size, seqlen/sp_size, nheads, head_dim)
    """
    sp_size = get_ulysses_sequence_parallel_world_size()

    if sp_size > 1:
        # All-to-all: gather sequence, scatter heads
        # (bsz, seq/sp, nheads, head_dim) -> (bsz, seq, nheads/sp, head_dim)
        query_states = gather_seq_scatter_heads(query_states, seq_dim=1, head_dim=2)
        key_states = gather_seq_scatter_heads(key_states, seq_dim=1, head_dim=2)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=1, head_dim=2)

        # Also gather position_ids if present
        if position_ids is not None:
            sp_group = get_ulysses_sequence_parallel_group()
            position_ids_lst = [
                torch.empty_like(position_ids) for _ in range(sp_size)
            ]
            dist.all_gather(position_ids_lst, position_ids, group=sp_group)
            position_ids = torch.cat(position_ids_lst, dim=-1)

        # Recompute query_length after all-to-all
        query_length = query_states.shape[1]

    # Call the original flash attention forward
    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        *args,
        position_ids=position_ids,
        **kwargs,
    )

    if sp_size > 1:
        # Reverse all-to-all: gather heads, scatter sequence
        # (bsz, seq, nheads/sp, head_dim) -> (bsz, seq/sp, nheads, head_dim)
        attn_output = gather_heads_scatter_seq(attn_output, head_dim=2, seq_dim=1)

    return attn_output


def patch_vlm_for_ulysses_input_slicing(model_class: type):
    """Applies a monkey patch to the forward method of a given model class
    to enable Ulysses sequence parallelism input slicing.

    When using Ulysses SP, the input embeddings need to be sliced along the
    sequence dimension so each SP rank processes a different chunk. This
    function patches the model's forward to call slice_input_tensor on the
    input embeddings before passing them to the language model.
    """
    original_forward = model_class.forward

    def patched_forward(self, *args, **kwargs):
        sp_size = get_ulysses_sequence_parallel_world_size()
        if sp_size <= 1:
            return original_forward(self, *args, **kwargs)

        # For VLM models, the inputs_embeds or input_ids need slicing
        # along the sequence dimension (dim=1) for Ulysses SP.
        if "inputs_embeds" in kwargs and kwargs["inputs_embeds"] is not None:
            kwargs["inputs_embeds"] = slice_input_tensor(
                kwargs["inputs_embeds"], dim=1, padding=True
            )
        elif "input_ids" in kwargs and kwargs["input_ids"] is not None:
            kwargs["input_ids"] = slice_input_tensor(
                kwargs["input_ids"], dim=1, padding=True
            )
        elif len(args) > 0 and isinstance(args[0], torch.Tensor):
            # First positional arg is likely input_ids
            args_list = list(args)
            args_list[0] = slice_input_tensor(args_list[0], dim=1, padding=True)
            args = tuple(args_list)

        return original_forward(self, *args, **kwargs)

    model_class.forward = patched_forward


def patch_forward_with_backends(
    model: PreTrainedModel,
    use_fused_kernels: bool = False,
    fused_kernels_backend: str = None,
):
    """Choose the forward function based on the model and backend.

    When use_fused_kernels is True, this patches the model's forward method
    to use a fused kernel implementation (torch or triton backend) that
    computes log_probs and entropy directly without materializing the full
    logits tensor, saving memory.

    Args:
        model (PreTrainedModel): The model to apply the monkey patch.
        use_fused_kernels (bool): Whether to use fused kernels.
        fused_kernels_backend (str): The backend to use for fused kernels.
            One of "torch", "triton", or None (defaults to "torch").
    """
    if not use_fused_kernels:
        return

    if fused_kernels_backend is None:
        fused_kernels_backend = "torch"

    # Determine model type and import appropriate forward function
    model_type = model.config.model_type

    # Map model architectures to their forward modules
    _MODEL_FORWARD_MAP = {
        "llama": "verl.models.transformers.dense_common",
        "qwen2": "verl.models.transformers.dense_common",
        "qwen3": "verl.models.transformers.dense_common",
        "qwen3_5": "verl.models.transformers.qwen3_5",
        "glm4v": "verl.models.transformers.glm4v",
        "qwen2_vl": "verl.models.transformers.qwen2_vl",
        "qwen3_vl": "verl.models.transformers.qwen3_vl",
        "kimi_vl": "verl.models.transformers.kimi_vl",
    }

    forward_fn = None

    try:
        import importlib

        module_name = _MODEL_FORWARD_MAP.get(
            model_type, "verl.models.transformers.dense_common"
        )
        mod = importlib.import_module(module_name)

        if fused_kernels_backend == "triton":
            forward_fn = getattr(mod, "forward_with_triton_backend", None)
        elif fused_kernels_backend == "torch":
            forward_fn = getattr(mod, "forward_with_torch_backend", None)
        else:
            forward_fn = getattr(mod, "forward_with_normal_backend", None)
    except (ImportError, AttributeError):
        logger.warning(
            f"Could not load fused kernel forward for model_type={model_type}, "
            f"falling back to normal forward."
        )
        forward_fn = None

    if forward_fn is not None:
        model.forward = MethodType(forward_fn, model)
        logger.info(
            f"Patched model forward with {fused_kernels_backend} backend "
            f"for model_type={model_type}"
        )


def apply_monkey_patch(
    model: PreTrainedModel,
    ulysses_sp_size: int = 1,
    use_remove_padding: bool = True,
    use_fused_kernels: bool = False,
    fused_kernels_backend: str = None,
    use_prefix_grouper: bool = False,
    use_tiled_mlp: bool = False,
    tiled_mlp_shards: int = 4,
):
    """Apply monkey patch to the models for ulysses sequence parallel,
    fused kernel, tiled MLP and prefix grouper.

    In the end of this function forward function of the model is patched
    for fused kernel. If the model is not supported with fused kernel,
    please return after patch.

    Args:
        model: The model to apply the monkey patch.
        ulysses_sp_size: The size of ulysses sequence parallel.
        use_remove_padding: Whether to use remove padding.
        use_fused_kernels: Whether to use fused kernels.
        fused_kernels_backend: The backend to use for fused kernels.
        use_prefix_grouper: Whether to patch attention functions for prefix grouper.
        use_tiled_mlp: Whether to use TiledMLP for memory-efficient MLP computation.
        tiled_mlp_shards: Number of shards for TiledMLP (higher = lower memory, slightly slower).
    """
    # 1. Patch for prefix grouper
    if use_prefix_grouper:
        apply_prefix_grouper_patch()

    # 2. Patch attention layers for Ulysses sequence parallelism
    if ulysses_sp_size > 1:
        _patch_attention_for_ulysses(model)

    # 3. Patch for TiledMLP (memory-efficient MLP)
    if use_tiled_mlp:
        from verl.models.transformers.tiled_mlp import apply_tiled_mlp_monkey_patch

        model_type = model.config.model_type
        try:
            apply_tiled_mlp_monkey_patch(
                num_shards=tiled_mlp_shards, model_type=model_type
            )
        except ValueError:
            logger.warning(
                f"TiledMLP not supported for model_type={model_type}, skipping."
            )

    # 4. Patch forward for fused kernels (must be done last as it replaces
    #    the top-level forward)
    patch_forward_with_backends(
        model=model,
        use_fused_kernels=use_fused_kernels,
        fused_kernels_backend=fused_kernels_backend,
    )


def _patch_attention_for_ulysses(model: PreTrainedModel):
    """Replace the _flash_attention_forward in the model's attention modules
    with our Ulysses-aware version.

    This ensures that flash attention calls inside the model go through
    the all-to-all communication pattern required for sequence parallelism.
    """
    sp_size = get_ulysses_sequence_parallel_world_size()
    if sp_size <= 1:
        return

    # We need to patch the attention forward functions used by the model.
    # The approach is to monkey-patch _flash_attention_forward at the module
    # level so that any attention layer that imports it gets the Ulysses version.
    import transformers.modeling_flash_attention_utils as fa_utils

    fa_utils._flash_attention_forward = _ulysses_flash_attention_forward

    # Also patch ALL_ATTENTION_FUNCTIONS entries if they reference the old
    # function
    for name in list(ALL_ATTENTION_FUNCTIONS.keys()):
        fn = ALL_ATTENTION_FUNCTIONS[name]
        if fn is _flash_attention_forward or (
            hasattr(fn, "__name__")
            and fn.__name__ == "_flash_attention_forward"
        ):
            ALL_ATTENTION_FUNCTIONS[name] = _ulysses_flash_attention_forward

    logger.info(
        f"Patched _flash_attention_forward for Ulysses SP (sp_size={sp_size})"
    )
