# Keep signatures / non-target code; implement every NotImplementedError body.
# Do not delete or rename interfaces.


"""
Agent framework for multi-turn rollout and agentic reinforcement learning.
- AgentLoopBase: coroutine based abstract base class for agent loop.
  - SingleTurnAgentLoop: single turn agent loop.
  - ToolAgentLoop: ReAct agent loop with tool calling, with user defined tools.
- AgentLoopWorker: worker class for running agent loop coroutines in parallel.
- AgentLoopManager: manager class for running agent loop workers in parallel.

AgentLoopManager is one specific agent-framework implementation in verl,
and is designed to be fully replaceable by other agent frameworks such as:
- NVIDIA Nemo-Gym
- AWS Bedrock AgentCore
- SWE-agent
- ...
"""
import asyncio
import logging
import os
import random
from abc import ABC, abstractmethod
from typing import Any, Optional
from uuid import uuid4
import hydra
import numpy as np
import ray
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from pydantic import BaseModel, ConfigDict
from tensordict import TensorDict
from transformers import AutoProcessor, AutoTokenizer
from verl.experimental.agent_loop.utils import resolve_config_path
from verl.protocol import DataProto
from verl.tools.tool_registry import load_all_tools
from verl.trainer.distillation import is_distillation_enabled
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.dataset.rl_dataset import RLHFDataset, get_dataset_class
from verl.utils.model import compute_position_id_with_mask
from verl.utils.profiler import simple_timer
from verl.utils.ray_utils import auto_await, get_event_loop
from verl.utils.rollout_trace import RolloutTraceConfig, rollout_trace_attr
from verl.utils.skip import SkipManager
from verl.utils.tokenizer import build_multimodal_processor_inputs, get_processor_token_id, normalize_token_ids
from verl.utils.tokenizer.chat_template import apply_chat_template, initialize_system_prompt, initialize_turn_separator
from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.llm_server import LLMServerClient
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_LOGGING_LEVEL', 'WARN'))
DEFAULT_ROUTING_CACHE_SIZE = 10000

class AgentLoopMetrics(BaseModel):
    """Agent loop performance metrics."""
    generate_sequences: float = 0.0
    tool_calls: float = 0.0
    compute_score: float = 0.0
    num_preempted: int = -1

class AgentLoopOutput(BaseModel):
    """Agent loop output."""
    prompt_ids: list[int]
    'Prompt token ids.'
    response_ids: list[int]
    'Response token ids including LLM generated token, tool response token.'
    response_mask: list[int]
    'Response mask, 1 for LLM generated token, 0 for tool response token.'
    response_logprobs: Optional[list[float]] = None
    'Log probabilities for the response tokens.'
    routed_experts: Optional[Any] = None
    'Routed experts for the total tokens.'
    multi_modal_data: Optional[dict[str, Any]] = None
    'Multi-modal data for multi-modal tools.'
    reward_score: Optional[float] = None
    'Reward score for the trajectory.'
    num_turns: int = 0
    'Number of chat turns, including user, assistant, tool.'
    metrics: AgentLoopMetrics
    'Auxiliary performance metrics'
    extra_fields: dict[str, Any] = {}
    'Extra fields for dynamic addition.'
    mm_processor_kwargs: Optional[dict[str, Any]] = None
    'Processor/backend kwargs that must stay aligned across rollout and training paths.'

    def as_dict(self) -> dict[str, Any]:
        """Convert agent loop output to a dictionary."""
        output = self.model_dump(exclude_unset=True)
        output['prompts'] = torch.tensor(output.pop('prompt_ids'), dtype=torch.int64)
        output['responses'] = torch.tensor(output.pop('response_ids'), dtype=torch.int64)
        output['response_mask'] = torch.tensor(output.pop('response_mask'), dtype=torch.int64)
        response_logprobs = output.pop('response_logprobs', None)
        if response_logprobs is not None:
            output['rollout_log_probs'] = torch.tensor(response_logprobs, dtype=torch.float32)
        routed_experts = output.pop('routed_experts', None)
        if routed_experts is not None:
            output['routed_experts'] = torch.tensor(routed_experts, dtype=torch.int64)
        reward_score = output.pop('reward_score', None)
        if reward_score is not None:
            rm_scores = torch.zeros_like(output['response_mask'], dtype=torch.float32)
            rm_scores[-1] = reward_score
            output['rm_scores'] = rm_scores
        teacher_ids, teacher_logprobs = (output['extra_fields'].pop('teacher_ids', None), output['extra_fields'].pop('teacher_logprobs', None))
        if teacher_ids is not None:
            output['teacher_ids'] = teacher_ids
        if teacher_logprobs is not None:
            output['teacher_logprobs'] = teacher_logprobs
        return output

class _InternalAgentLoopOutput(AgentLoopOutput):
    """Internal agent loop output with padded sequences."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    prompt_ids: torch.Tensor
    'Padded prompt token ids.'
    response_ids: torch.Tensor
    'Padded response token ids.'
    input_ids: torch.Tensor
    'Padded input ids(prompt_ids + response_ids).'
    position_ids: torch.Tensor
    'Padded position ids.'
    response_mask: torch.Tensor
    'Padded response mask.'
    attention_mask: torch.Tensor
    'Padded attention mask.'
    response_logprobs: Optional[torch.Tensor] = None
    'Padded log probabilities for the response tokens.'
    teacher_logprobs: Optional[torch.Tensor] = None
    'Padded log probabilities from teacher model for prompt/response tokens.'
    teacher_ids: Optional[torch.Tensor] = None
    'Padded token ids corresponding to the teacher log probabilities.'
    routed_experts: Optional[torch.Tensor] = None
    'Padded routed experts for the total tokens.'
    multi_modal_inputs: Optional[dict[str, torch.Tensor]] = None
    'Multi-modal inputs for processors (e.g. pixel_values, image_grid_thw, video_grid_thw).'
    extra_fields: dict[str, Any] = {}
    'Extra fields for dynamic addition.'

class DictConfigWrap:
    """Wrapper for DictConfig to avoid hydra.utils.instantiate recursive resolve."""

    def __init__(self, config: DictConfig):
        self.config = config

class ToolListWrap:
    """Wraps a tool list so ``hydra.utils.instantiate`` doesn't recursively
    resolve its elements (which would demote them to ``DictConfig``)."""

    def __init__(self, tools: list):
        self.tools = tools

class AgentLoopBase(ABC):
    """An agent loop takes an input message, chat with OpenAI compatible LLM server and interact with various
    environments.

    Args:
        trainer_config (DictConfig): whole config for main entrypoint.
        server_manager (LLMServerClient): OpenAI compatible LLM server manager.
        tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
        processor (AutoProcessor): Processor for process messages.
        dataset_cls (type[Dataset]): Dataset class for creating dataset, Defaults to RLHFDataset.
        data_config (DictConfigWrap): Dataset config.
    """

    def __init__(self, trainer_config: DictConfigWrap, server_manager: LLMServerClient, tokenizer: AutoTokenizer, processor: AutoProcessor, dataset_cls: type[RLHFDataset], data_config: DictConfigWrap, **kwargs):
        self.config = trainer_config.config
        self.rollout_config = self.config.actor_rollout_ref.rollout
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.processor = processor
        self.dataset_cls = dataset_cls
        self.data_config = data_config.config
        self.apply_chat_template_kwargs = self.data_config.get('apply_chat_template_kwargs', {})
        self.mm_processor_kwargs = self.data_config.get('mm_processor_kwargs', {})
        self.continuous_token_builder = None
        self.enable_continuous_token = False
        continuous_token_config = self.data_config.continuous_token
        if continuous_token_config.enable and self.processor is None:
            model_config = self.config.actor_rollout_ref.model
            self.continuous_token_builder = create_continuous_token_builder(self.tokenizer, model_family=continuous_token_config.model_family, model_path=model_config.path, tokenizer_name_or_path=model_config.tokenizer_path, chat_template_kwargs=self.apply_chat_template_kwargs)
            self.enable_continuous_token = True
            self.system_prompt = None
            self.turn_separator = []
        else:
            if continuous_token_config.enable and self.processor is not None:
                logger.warning('Continuous Token is enabled but processor is set; falling back to legacy multimodal path.')
            processing_class = self.processor if self.processor is not None else self.tokenizer
            self.system_prompt = initialize_system_prompt(processing_class, **self.apply_chat_template_kwargs)
            self.turn_separator = initialize_turn_separator(processing_class, **self.apply_chat_template_kwargs)
        self.loop = get_event_loop()

    def _get_mm_processor_kwargs(self, audio_data: Optional[list[Any]]=None) -> dict[str, Any]:
        mm_processor_kwargs = dict(self.mm_processor_kwargs or {})
        if audio_data is not None and 'sampling_rate' not in mm_processor_kwargs:
            sampling_rate = getattr(getattr(self.processor, 'feature_extractor', None), 'sampling_rate', None)
            if sampling_rate is not None:
                mm_processor_kwargs['sampling_rate'] = int(sampling_rate)
        return mm_processor_kwargs

    async def process_vision_info(self, messages: list[dict]) -> dict:
        """Backward-compatible wrapper for multi-modal extraction."""
        return await self.process_multi_modal_info(messages)

    async def process_multi_modal_info(self, messages: list[dict]) -> dict:
        """Extract images, videos and audios from messages.

        Args:
            messages (list[dict]): Input messages.

        Returns:
            dict: Multi-modal data with keys like "images", "videos" and "audios".
        """
        multi_modal_data = {}
        if self.processor is not None:
            image_patch_size = getattr(getattr(self.processor, 'image_processor', None), 'patch_size', 14)
            if hasattr(self.dataset_cls, 'process_multi_modal_info'):
                images, videos, audios = await self.dataset_cls.process_multi_modal_info(messages, image_patch_size=image_patch_size, config=self.data_config)
            else:
                images, videos = await self.dataset_cls.process_vision_info(messages, image_patch_size=image_patch_size, config=self.data_config)
                audios = None
            if images is not None:
                multi_modal_data['images'] = images
            if videos is not None:
                multi_modal_data['videos'] = videos
            if audios is not None:
                multi_modal_data['audios'] = audios
        return multi_modal_data

    async def ct_build_initial_tokens(self, messages: list[dict], tools: list[dict]=None) -> list[int]:
        """Build initial tokens for continuous token agent loop."""
        raise NotImplementedError

    async def ct_merge_non_assistant_msg(self, previous_messages: list[dict], updated_messages: list[dict], runtime_token_ids: list[int], response_mask: list[int], response_logprobs: Optional[list[float]]=None, tools: list[dict]=None):
        """Merge non-assistant messages for continuous token agent loop."""
        raise NotImplementedError

    async def ct_merge_assistant_token(self, runtime_token_ids: list[int], assistant_token_ids: list[int], response_mask: list[int], response_logprobs: Optional[list[float]]=None, assistant_logprobs: Optional[list[float]]=None):
        """Merge assistant tokens for continuous token agent loop."""
        raise NotImplementedError

    def _cap_text_prompt_length(self, prompt_ids: list[int]) -> list[int]:
        """Cap prompt length to prompt_length from config."""
        prompt_length = self.rollout_config.prompt_length
        return prompt_ids[:prompt_length] if len(prompt_ids) > prompt_length else prompt_ids

    async def apply_chat_template(self, messages: list[dict], tools: list[dict]=None, images: list[Image.Image]=None, videos: list[tuple[torch.Tensor, dict]]=None, audios: list[Any]=None, mm_processor_kwargs: Optional[dict[str, Any]]=None, remove_system_prompt: bool=False):
        """Apply chat template to messages with optional tools, images, and videos.

Args:
    messages (list[dict]): Input messages.
    tools (list[dict], optional): Tools schemas. Defaults to None.
    images (list[Image.Image], optional): Input images. Defaults to None.
    videos (list[tuple[torch.Tensor, dict]], optional): Input videos. Defaults to None.
    remove_system_prompt (bool, optional): Whether to remove system prompt. Defaults to False.

Returns:
    list[int]: Prompt token ids."""
        processing_class = self.processor if self.processor is not None else self.tokenizer

        if mm_processor_kwargs is None:
            mm_processor_kwargs = self._get_mm_processor_kwargs(audios)

        # Add generation prompt and apply chat template
        if self.processor is not None and (images is not None or videos is not None or audios is not None):
            # Multimodal path with processor
            prompt_dict = build_multimodal_processor_inputs(
                self.processor,
                messages,
                images=images,
                videos=videos,
                audio=audios,
                tools=tools,
                add_generation_prompt=True,
                tokenize=True,
                **mm_processor_kwargs,
            )
            prompt_ids = prompt_dict.get('input_ids', [])
            if isinstance(prompt_ids, torch.Tensor):
                prompt_ids = prompt_ids.squeeze(0).tolist()
        elif self.enable_continuous_token:
            prompt_ids = await self.ct_build_initial_tokens(messages, tools=tools)
        else:
            prompt_ids = apply_chat_template(
                processing_class,
                messages,
                tools=tools,
                add_generation_prompt=True,
                tokenize=True,
                **self.apply_chat_template_kwargs,
            )

        # Normalize token ids
        prompt_ids = normalize_token_ids(prompt_ids)

        # Remove system prompt if requested
        if remove_system_prompt and self.system_prompt:
            if len(prompt_ids) >= len(self.system_prompt):
                if prompt_ids[:len(self.system_prompt)] == self.system_prompt:
                    prompt_ids = prompt_ids[len(self.system_prompt):]

        # Cap to prompt length
        prompt_ids = self._cap_text_prompt_length(prompt_ids)

        return prompt_ids

    @abstractmethod
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run agent loop to interact with LLM server and environment.

Args:
    sampling_params (Dict[str, Any]): LLM sampling params.
    **kwargs: dataset fields from `verl.utils.dataset.RLHFDataset`.

Returns:
    AgentLoopOutput: Agent loop output."""
        pass  # abstract method: subclasses (SingleTurnAgentLoop, ToolAgentLoop) provide implementation


'Agent loop registry: key is agent_name, value is a dict of agent loop config\nused by hydra.utils.instantiate to initialize agent loop instance.\n\nhttps://hydra.cc/docs/advanced/instantiate_objects/overview/\n'
_agent_loop_registry: dict[str, dict] = {}

def register(agent_name: str):
    """Register agent loop class."""
    def decorator(subclass: type[AgentLoopBase]) -> type[AgentLoopBase]:
        fqdn = f'{subclass.__module__}.{subclass.__qualname__}'
        _agent_loop_registry[agent_name] = {'_target_': fqdn}
        return subclass
    return decorator

class AgentLoopWorker:
    """Agent loop worker takes a batch of messages and run each message in an agent loop.

    Args:
        config (DictConfig): whole config for main entrypoint.
        llm_client (LLMServerClient): Client for the LLM server.
        teacher_client (dict[str, LLMServerClient]): Client for multiple teacher servers.
        reward_loop_worker_handles (List[ray.actor.ActorHandle]): Actor handles for streaming reward computation.
    """

    def __init__(self, config: DictConfig, llm_client: LLMServerClient, teacher_client: dict[str, LLMServerClient]=None, reward_loop_worker_handles: list[ray.actor.ActorHandle]=None):
        self.config = config
        self.llm_client = llm_client
        self.teacher_client = teacher_client
        self.reward_loop_worker_handles = reward_loop_worker_handles
        rollout_config, model_config = (config.actor_rollout_ref.rollout, config.actor_rollout_ref.model)
        self.rollout_config: RolloutConfig = omega_conf_to_dataclass(rollout_config)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config)
        self.dataset_cls = get_dataset_class(config.data)
        self.tokenizer = self.model_config.tokenizer
        self.processor = self.model_config.processor
        self.mm_processor_kwargs = config.data.get('mm_processor_kwargs', {})
        self.distillation_enabled = is_distillation_enabled(config.distillation)
        if self.distillation_enabled:
            from verl.experimental.teacher_loop.teacher_manager import AsyncTeacherLLMServerManager
            self.teacher_key: str = config.distillation.teacher_key
            self.teacher_server_manager = AsyncTeacherLLMServerManager(config=config, teacher_client=teacher_client)
        tool_config_path = self.rollout_config.multi_turn.tool_config_path
        function_tool_path = self.rollout_config.multi_turn.function_tool_path
        self.tools = load_all_tools(tool_config_path=resolve_config_path(tool_config_path) if tool_config_path else None, function_tool_path=resolve_config_path(function_tool_path) if function_tool_path else None)
        agent_loop_config_path = self.rollout_config.agent.agent_loop_config_path
        if agent_loop_config_path:
            resolved_path = resolve_config_path(agent_loop_config_path)
            agent_loop_configs = OmegaConf.load(resolved_path)
            for agent_loop_config in agent_loop_configs:
                _agent_loop_registry[agent_loop_config.name] = agent_loop_config
        if self.model_config.get('custom_chat_template', None) is not None:
            if self.model_config.processor is not None:
                self.model_config.processor.chat_template = self.model_config.custom_chat_template
            self.model_config.tokenizer.chat_template = self.model_config.custom_chat_template
        trace_config = self.rollout_config.trace
        RolloutTraceConfig.init(self.rollout_config.trace.project_name, self.rollout_config.trace.experiment_name, trace_config.get('backend'), trace_config.get('token2text', False), trace_config.get('max_samples_per_step_per_worker', None))

    def _get_mm_processor_kwargs(self, audio_data: Optional[list[Any]]=None) -> dict[str, Any]:
        """Return multimodal processor kwargs with audio sampling-rate defaults."""
        mm_processor_kwargs = dict(self.mm_processor_kwargs or {})
        if audio_data is not None and 'sampling_rate' not in mm_processor_kwargs:
            sampling_rate = getattr(getattr(self.processor, 'feature_extractor', None), 'sampling_rate', None)
            if sampling_rate is not None:
                mm_processor_kwargs['sampling_rate'] = int(sampling_rate)
        return mm_processor_kwargs

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate sequences from agent loop.

Args:
    batch (DataProto): Input batch.

Returns:
    DataProto: Output batch.
    - prompts: [bsz, prompt_length], prompt token ids from dataset.
    - responses: [bsz, response_length], output token ids include response tokens
      from LLM generation and observation tokens from tool_calls.
    - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
    - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
      and response tokens.
    - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
    - position_ids: [bsz, prompt_length + response_length], incremental position ids.

    For multi-turn conversations:
    responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
    response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|"""
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        # Determine validate mode from meta_info
        validate = batch.meta_info.get('validate', False)
        if validate:
            sampling_params['top_p'] = config.val_kwargs.top_p
            sampling_params['top_k'] = config.val_kwargs.top_k
            sampling_params['temperature'] = config.val_kwargs.temperature

        # Default agent name
        if 'agent_name' not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
        else:
            default_agent_loop = None

        # Get trajectory info
        if 'global_steps' in batch.non_tensor_batch and 'index' in batch.non_tensor_batch:
            trajectory_info = await get_trajectory_info(
                batch.non_tensor_batch['global_steps'],
                batch.non_tensor_batch['index'],
                validate,
            )
        else:
            batch_size = len(batch)
            trajectory_info = [{'step': 0, 'sample_index': i, 'rollout_n': 0, 'validate': validate}
                               for i in range(batch_size)]

        # Process each sample
        tasks = []
        for i in range(len(batch)):
            sample = batch[i]
            kwargs = {}
            if sample.non_tensor_batch:
                for k, v in sample.non_tensor_batch.items():
                    kwargs[k] = v

            agent_name = kwargs.pop('agent_name', default_agent_loop) if default_agent_loop is not None else kwargs.get('agent_name', config.agent.default_agent_loop)
            if isinstance(agent_name, np.ndarray):
                agent_name = str(agent_name)

            task = asyncio.create_task(
                self._run_agent_loop(
                    sampling_params,
                    trajectory=trajectory_info[i],
                    agent_name=agent_name,
                    trace=False,
                    **kwargs,
                )
            )
            tasks.append(task)

        outputs: list[_InternalAgentLoopOutput] = list(await asyncio.gather(*tasks))

        # Postprocess into DataProto
        return self._postprocess(outputs, batch.non_tensor_batch, validate=validate)

    async def _run_agent_loop(self, sampling_params: dict[str, Any], trajectory: dict[str, Any], *, agent_name: str, trace: bool=True, **kwargs) -> _InternalAgentLoopOutput:
        config = self.config
        agent_loop_config = _agent_loop_registry.get(agent_name)
        if agent_loop_config is None:
            raise ValueError(
                f"Agent loop '{agent_name}' not found in registry. "
                f"Available: {list(_agent_loop_registry.keys())}"
            )

        # Create agent loop instance via hydra
        agent_loop: AgentLoopBase = hydra.utils.instantiate(
            config=agent_loop_config,
            trainer_config=DictConfigWrap(config=config),
            server_manager=self.llm_client,
            tokenizer=self.tokenizer,
            processor=self.processor,
            dataset_cls=self.dataset_cls,
            data_config=DictConfigWrap(config=config.data),
            tools=ToolListWrap(self.tools) if self.tools else None,
        )

        # Run agent loop with trace context
        with rollout_trace_attr(
            sample_index=trajectory.get('sample_index'),
            step=trajectory.get('step'),
            rollout_n=trajectory.get('rollout_n'),
            validate=trajectory.get('validate', False),
            trace=trace,
        ):
            output: AgentLoopOutput = await agent_loop.run(sampling_params, **kwargs)

        # Postprocess output
        result = await self._agent_loop_postprocess(
            output,
            validate=trajectory.get('validate', False),
            **kwargs,
        )
        return result

    def _pad_token_ids(self, tokens: list[int], *, max_length: int, padding_side: str, return_attention_mask: bool) -> dict[str, torch.Tensor]:
        """Pad token ids to max_length.

        Args:
            tokens (list[int]): Input token ids.
            max_length (int): Maximum length.
            padding_side (str): 'left' or 'right'.
            return_attention_mask (bool): Whether to return attention mask.

        Returns:
            dict[str, torch.Tensor]: Padded token ids and optional attention mask.
        """
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        pad_size = max_length - len(tokens)
        if padding_side == 'left':
            padded = [pad_id] * pad_size + list(tokens)
        else:
            padded = list(tokens) + [pad_id] * pad_size
        input_ids = torch.tensor(padded, dtype=torch.int64).unsqueeze(0)
        result = {'input_ids': input_ids}
        if return_attention_mask:
            if padding_side == 'left':
                mask = [0] * pad_size + [1] * len(tokens)
            else:
                mask = [1] * len(tokens) + [0] * pad_size
            result['attention_mask'] = torch.tensor(mask, dtype=torch.int64).unsqueeze(0)
        return result

    async def _agent_loop_postprocess(self, output, validate, **kwargs) -> _InternalAgentLoopOutput:
        """Perform post-processing operations on the output of each individual agent loop."""
        # Handle both single output and list of outputs
        if isinstance(output, list):
            output_list = output
            final_output = output_list[-1] if output_list else output
        else:
            output_list = [output]
            final_output = output

        # Compute reward score
        await self._compute_score(output_list, kwargs=kwargs)

        # Compute teacher logprobs (for distillation)
        await self._compute_teacher_logprobs(
            final_output,
            prompt_ids=final_output.prompt_ids,
            response_ids=final_output.response_ids,
            validate=validate,
            sample_kwargs=kwargs,
        )

        # Pad sequences
        prompt_length = self.rollout_config.prompt_length
        response_length = self.rollout_config.response_length

        prompt_ids = final_output.prompt_ids if isinstance(final_output.prompt_ids, list) else list(final_output.prompt_ids)
        response_ids = final_output.response_ids[:response_length] if len(final_output.response_ids) > response_length else final_output.response_ids

        # Pad prompt_ids (left-padded)
        prompt_padded = self._pad_token_ids(
            prompt_ids,
            max_length=prompt_length,
            padding_side='left',
            return_attention_mask=False,
        )
        prompt_ids_tensor = prompt_padded['input_ids']  # (1, prompt_length)

        # Pad response_ids (right-padded)
        response_padded = self._pad_token_ids(
            response_ids,
            max_length=response_length,
            padding_side='right',
            return_attention_mask=False,
        )
        response_ids_tensor = response_padded['input_ids']  # (1, response_length)

        # Build input_ids = concat(prompt, response)
        input_ids = torch.cat([prompt_ids_tensor, response_ids_tensor], dim=1)  # (1, total_length)

        # Build attention_mask
        prompt_attn = self._pad_token_ids(
            prompt_ids,
            max_length=prompt_length,
            padding_side='left',
            return_attention_mask=True,
        )
        response_attn = self._pad_token_ids(
            response_ids,
            max_length=response_length,
            padding_side='right',
            return_attention_mask=True,
        )
        attention_mask = torch.cat(
            [prompt_attn['attention_mask'], response_attn['attention_mask']], dim=1
        )

        # Pad response_mask
        response_mask_data = final_output.response_mask[:response_length] if len(final_output.response_mask) > response_length else final_output.response_mask
        response_mask_padded = self._pad_token_ids(
            response_mask_data,
            max_length=response_length,
            padding_side='right',
            return_attention_mask=False,
        )
        response_mask = response_mask_padded['input_ids']  # (1, response_length)

        # Pad response_logprobs
        response_logprobs = None
        if final_output.response_logprobs is not None:
            logprobs_data = final_output.response_logprobs[:response_length]
            pad_size = response_length - len(logprobs_data)
            logprobs_padded = list(logprobs_data) + [0.0] * pad_size
            response_logprobs = torch.tensor(logprobs_padded, dtype=torch.float32).unsqueeze(0)

        # Multi-modal inputs
        multi_modal_inputs = self._compute_multi_modal_inputs(final_output, input_ids)

        # Position ids
        position_ids = self._compute_position_ids(
            input_ids, attention_mask, multi_modal_inputs,
            final_output.mm_processor_kwargs
        )

        # Teacher logprobs/ids from extra_fields
        teacher_logprobs = final_output.extra_fields.pop('teacher_logprobs', None)
        teacher_ids = final_output.extra_fields.pop('teacher_ids', None)

        # Routed experts
        routed_experts = None
        if final_output.routed_experts is not None:
            total_length = prompt_length + response_length
            re = final_output.routed_experts
            if hasattr(re, '__len__') and len(re) > total_length:
                re = re[:total_length]
            if isinstance(re, torch.Tensor):
                routed_experts = re
            else:
                routed_experts = torch.tensor(re, dtype=torch.int64)

        return _InternalAgentLoopOutput(
            prompt_ids=prompt_ids_tensor,
            response_ids=response_ids_tensor,
            input_ids=input_ids,
            position_ids=position_ids,
            response_mask=response_mask,
            attention_mask=attention_mask,
            response_logprobs=response_logprobs,
            teacher_logprobs=teacher_logprobs,
            teacher_ids=teacher_ids,
            routed_experts=routed_experts,
            multi_modal_inputs=multi_modal_inputs,
            reward_score=final_output.reward_score,
            num_turns=final_output.num_turns,
            metrics=final_output.metrics,
            extra_fields=final_output.extra_fields,
            multi_modal_data=final_output.multi_modal_data,
            mm_processor_kwargs=final_output.mm_processor_kwargs,
        )

    def _compute_multi_modal_inputs(self, output, input_ids) -> dict[str, torch.Tensor]:
        """Compute multi-modal inputs with image, video and audio."""
        multi_modal_inputs = {}
        if self.processor is None:
            return multi_modal_inputs
        multi_modal_data = output.multi_modal_data or {}
        images = multi_modal_data.get('images')
        videos = multi_modal_data.get('videos')
        audios = multi_modal_data.get('audios')
        current_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)
        multi_modal_inputs = build_multimodal_processor_inputs(
            self.processor,
            current_text,
            images=images,
            videos=videos,
            audio=audios,
            tokenize=False,
            return_tensors='pt',
        )
        return multi_modal_inputs

    def _compute_position_ids(self, input_ids, attention_mask, multi_modal_inputs, mm_processor_kwargs: Optional[dict[str, Any]]=None) -> torch.Tensor:
        """Compute position ids for multi-modal inputs."""
        if self.processor is None or not hasattr(self.processor, 'get_rope_index'):
            return compute_position_id_with_mask(attention_mask)
        multi_modal_kwargs = {'image_grid_thw': multi_modal_inputs.get('image_grid_thw'), 'video_grid_thw': multi_modal_inputs.get('video_grid_thw')}
        if multi_modal_inputs.pop('mm_token_type_ids', None) is not None:
            mm_token_type_ids = torch.zeros_like(input_ids)
            image_token_id = get_processor_token_id(self.processor, 'image')
            video_token_id = get_processor_token_id(self.processor, 'video')
            if image_token_id is not None:
                mm_token_type_ids[0][input_ids[0] == image_token_id] = 1
            if video_token_id is not None:
                mm_token_type_ids[0][input_ids[0] == video_token_id] = 2
            multi_modal_kwargs['mm_token_type_ids'] = mm_token_type_ids
        vision_position_ids, _ = self.processor.get_rope_index(input_ids=input_ids, attention_mask=attention_mask, **multi_modal_kwargs)
        vision_position_ids = vision_position_ids.transpose(0, 1)
        valid_mask = attention_mask[0].bool()
        text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
        text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
        text_position_ids = text_position_ids.unsqueeze(0)
        position_ids = torch.cat((text_position_ids, vision_position_ids), dim=1)
        return position_ids

    async def _compute_score(self, outputs: list[AgentLoopOutput], kwargs: dict) -> None:
        """Compute reward score for all outputs in a trajectory; assigns result to outputs[-1]."""
        if not outputs:
            return

        final_output = outputs[-1]
        reward_score = None
        reward_extra_info = {}

        # Try to use reward_loop_worker_handles if available
        if self.reward_loop_worker_handles is not None and len(self.reward_loop_worker_handles) > 0:
            try:
                # Build a minimal DataProto for reward computation
                prompts_tensor = torch.tensor(final_output.prompt_ids, dtype=torch.int64).unsqueeze(0)
                responses_tensor = torch.tensor(final_output.response_ids, dtype=torch.int64).unsqueeze(0)
                response_mask_tensor = torch.tensor(final_output.response_mask, dtype=torch.int64).unsqueeze(0)
                attn = torch.ones(1, prompts_tensor.size(1) + responses_tensor.size(1), dtype=torch.int64)
                batch = TensorDict({
                    'prompts': prompts_tensor,
                    'responses': responses_tensor,
                    'response_mask': response_mask_tensor,
                    'attention_mask': attn,
                }, batch_size=1)
                non_tensor_batch = {
                    'data_source': np.array([kwargs.get('data_source', 'unknown')]),
                    'reward_model': np.array([kwargs.get('reward_model', {'ground_truth': ''})]),
                    'extra_info': np.array([kwargs.get('extra_info', {})]),
                }
                data = DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

                # Round-robin dispatch to reward workers
                worker_idx = hash(str(kwargs.get('uid', ''))) % len(self.reward_loop_worker_handles)
                worker = self.reward_loop_worker_handles[worker_idx]
                result = await worker.compute_score.remote(data)
                reward_score = result.get('reward_score', 0.0)
                reward_extra_info = result.get('reward_extra_info', {})
            except Exception as e:
                logger.warning(f'Reward loop worker failed: {e}, falling back to inline computation.')

        # Fallback: compute reward inline
        if reward_score is None:
            try:
                # Decode the response
                response_ids = final_output.response_ids
                response_str = self.tokenizer.decode(response_ids, skip_special_tokens=True)

                data_source = kwargs.get('data_source', 'unknown')
                # Handle reward_model which may be a dict or array of dicts
                reward_model = kwargs.get('reward_model', {})
                if isinstance(reward_model, np.ndarray):
                    reward_model = reward_model.item() if reward_model.size == 1 else reward_model
                if isinstance(reward_model, dict):
                    ground_truth = reward_model.get('ground_truth', '')
                else:
                    ground_truth = str(reward_model) if reward_model is not None else ''

                extra_info = kwargs.get('extra_info', {})
                if isinstance(extra_info, np.ndarray):
                    extra_info = extra_info.item() if extra_info.size == 1 else extra_info

                # Load reward function from config
                from verl.trainer.ppo.reward import get_custom_reward_fn
                reward_fn = get_custom_reward_fn(self.config)

                if reward_fn is not None:
                    result = reward_fn(
                        data_source=data_source,
                        solution_str=response_str,
                        ground_truth=ground_truth,
                        extra_info=extra_info,
                    )
                    if isinstance(result, dict):
                        reward_score = result.get('score', 0.0)
                        reward_extra_info = {k: v for k, v in result.items() if k != 'score'}
                    else:
                        reward_score = float(result)
                        reward_extra_info = {'acc': reward_score}
                else:
                    # No reward function configured, default to 0
                    reward_score = 0.0
                    reward_extra_info = {}
            except Exception as e:
                logger.warning(f'Inline reward computation failed: {e}')
                reward_score = 0.0
                reward_extra_info = {'error': str(e)}

        final_output.reward_score = float(reward_score) if reward_score is not None else 0.0
        final_output.extra_fields['reward_extra_info'] = reward_extra_info

    async def _compute_teacher_logprobs(self, output: AgentLoopOutput, prompt_ids: list[int], response_ids: list[int], validate: bool, sample_kwargs: Optional[dict[str, Any]]=None) -> None:
        """Compute teacher logprobs for single sample."""
        if self.distillation_enabled and (not validate):
            routing_key = None
            if sample_kwargs is not None:
                routing_value = sample_kwargs.get(self.teacher_key)
                if routing_value is not None:
                    routing_key = routing_value.item() if hasattr(routing_value, 'item') else routing_value
            teacher_ids, teacher_logprobs = await self.teacher_server_manager.compute_teacher_logprobs_single(sequence_ids=prompt_ids + response_ids, multi_modal_data=output.multi_modal_data, mm_processor_kwargs=output.mm_processor_kwargs, routing_key=routing_key)
            output.extra_fields['teacher_ids'] = teacher_ids
            output.extra_fields['teacher_logprobs'] = teacher_logprobs

    def _postprocess(self, inputs: list[_InternalAgentLoopOutput], input_non_tensor_batch: dict | None=None, validate: bool=False) -> DataProto:
        """Process the padded outputs from _run_agent_loop and combine them into a batch."""
        prompt_length = self.rollout_config.prompt_length
        response_length = self.rollout_config.response_length
        total_length = prompt_length + response_length
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        if not inputs:
            batch_size = 0
            if input_non_tensor_batch is not None:
                first_key = next(iter(input_non_tensor_batch))
                batch_size = len(input_non_tensor_batch[first_key])
            batch = TensorDict({
                'prompts': torch.full((batch_size, prompt_length), pad_id, dtype=torch.int64),
                'responses': torch.full((batch_size, response_length), pad_id, dtype=torch.int64),
                'response_mask': torch.zeros((batch_size, response_length), dtype=torch.int64),
                'input_ids': torch.full((batch_size, total_length), pad_id, dtype=torch.int64),
                'attention_mask': torch.zeros((batch_size, total_length), dtype=torch.int64),
                'position_ids': torch.zeros((batch_size, total_length), dtype=torch.int64),
            }, batch_size=batch_size)
            non_tensor_batch = dict(input_non_tensor_batch) if input_non_tensor_batch else {}
            if 'multi_modal_inputs' not in non_tensor_batch:
                non_tensor_batch['multi_modal_inputs'] = np.array([{} for _ in range(batch_size)], dtype=object)
            if 'metrics' not in non_tensor_batch:
                non_tensor_batch['metrics'] = np.array([{} for _ in range(batch_size)], dtype=object)
            return DataProto(batch=batch, non_tensor_batch=non_tensor_batch,
                             meta_info={'validate': validate})

        # Stack tensors from all outputs
        batch_size = len(inputs)
        prompt_ids_list = [out.prompt_ids for out in inputs]
        response_ids_list = [out.response_ids for out in inputs]
        response_mask_list = [out.response_mask for out in inputs]
        input_ids_list = [out.input_ids for out in inputs]
        attention_mask_list = [out.attention_mask for out in inputs]
        position_ids_list = [out.position_ids for out in inputs]

        prompts = torch.cat(prompt_ids_list, dim=0)
        responses = torch.cat(response_ids_list, dim=0)
        response_mask = torch.cat(response_mask_list, dim=0)
        input_ids = torch.cat(input_ids_list, dim=0)
        attention_mask = torch.cat(attention_mask_list, dim=0)
        position_ids = torch.cat(position_ids_list, dim=0)

        batch_dict = {
            'prompts': prompts,
            'responses': responses,
            'response_mask': response_mask,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
        }

        # Add logprobs if available
        if inputs[0].response_logprobs is not None:
            response_logprobs = torch.cat(
                [out.response_logprobs for out in inputs if out.response_logprobs is not None], dim=0
            )
            if response_logprobs.size(0) == len(inputs):
                batch_dict['rollout_log_probs'] = response_logprobs

        # Add routed experts if available
        if inputs[0].routed_experts is not None:
            routed_list = [out.routed_experts for out in inputs if out.routed_experts is not None]
            if len(routed_list) == len(inputs):
                batch_dict['routed_experts'] = torch.stack(routed_list, dim=0)

        # Build rm_scores from reward_score
        rm_scores = torch.zeros(batch_size, response_length, dtype=torch.float32)
        for i, out in enumerate(inputs):
            if out.reward_score is not None:
                # Find the last valid (non-padding) response position
                resp_attn = attention_mask[i, prompt_length:]
                valid_len = resp_attn.sum().item()
                if valid_len > 0:
                    rm_scores[i, int(valid_len) - 1] = float(out.reward_score)
        batch_dict['rm_scores'] = rm_scores

        # Teacher logprobs/ids
        if inputs[0].teacher_logprobs is not None:
            teacher_logprobs_list = [out.teacher_logprobs for out in inputs if out.teacher_logprobs is not None]
            if len(teacher_logprobs_list) == len(inputs):
                batch_dict['teacher_logprobs'] = torch.cat(teacher_logprobs_list, dim=0)
        if inputs[0].teacher_ids is not None:
            teacher_ids_list = [out.teacher_ids for out in inputs if out.teacher_ids is not None]
            if len(teacher_ids_list) == len(inputs):
                batch_dict['teacher_ids'] = torch.cat(teacher_ids_list, dim=0)

        batch = TensorDict(batch_dict, batch_size=batch_size)

        # Build non_tensor_batch
        metrics_list = []
        for out in inputs:
            m = out.metrics
            if hasattr(m, 'model_dump'):
                m = m.model_dump()
            elif hasattr(m, '__dict__'):
                m = dict(m.__dict__)
            else:
                m = dict(m) if m else {}
            metrics_list.append(m)

        non_tensor_batch = dict(input_non_tensor_batch) if input_non_tensor_batch else {}
        non_tensor_batch['metrics'] = np.array(metrics_list, dtype=object)

        # Collect multi_modal_inputs from outputs
        multi_modal_inputs_list = []
        for out in inputs:
            mm_inputs = out.multi_modal_inputs
            if mm_inputs is None:
                mm_inputs = {}
            multi_modal_inputs_list.append(mm_inputs)
        non_tensor_batch['multi_modal_inputs'] = np.array(multi_modal_inputs_list, dtype=object)

        meta_info = {'validate': validate}

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)


async def get_trajectory_info(step, index, validate):
    """Get trajectory info.

    Args:
        step (int): global steps in the trainer.
        index (list): form datastore extra_info.index column.
        validate (bool): whether is a validate step.

    Returns:
        list: trajectory.
    """
    trajectory_info = []
    rollout_n = 0
    for i in range(len(index)):
        if i > 0 and index[i - 1] == index[i]:
            rollout_n += 1
        else:
            rollout_n = 0
        trajectory_info.append({'step': step, 'sample_index': index[i], 'rollout_n': rollout_n, 'validate': validate})
    return trajectory_info


class AgentLoopManager:
    """Agent loop manager that manages a group of agent loop workers.

    Args:
        config (DictConfig): whole config for main entrypoint.
        llm_client (LLMServerClient): Client for the LLM server.
        teacher_client (dict[str, LLMServerClient]): Client for multiple teacher servers.
        reward_loop_worker_handles (List[ray.actor.ActorHandle]): Actor handles for streaming reward computation.
    """

    def __init__(self, config: DictConfig, llm_client: LLMServerClient, teacher_client: dict[str, LLMServerClient]=None, reward_loop_worker_handles: list[ray.actor.ActorHandle]=None):
        self.config = config
        self.rollout_config = config.actor_rollout_ref.rollout
        self.model_config = config.actor_rollout_ref.model
        self.llm_client = llm_client
        self.teacher_client = teacher_client
        self.reward_loop_worker_handles = reward_loop_worker_handles
        if not hasattr(self, 'agent_loop_workers_class'):
            self.agent_loop_workers_class = ray.remote(AgentLoopWorker)

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        """Create agent loop manager."""
        instance = cls(*args, **kwargs)
        await instance._init_agent_loop_workers()
        return instance

    async def _init_agent_loop_workers(self):
        self.agent_loop_workers = []
        num_workers = self.rollout_config.agent.num_workers
        node_ids = [node['NodeID'] for node in ray.nodes() if node['Alive'] and node['Resources'].get('CPU', 0) > 0]
        for i in range(num_workers):
            node_id = node_ids[i % len(node_ids)]
            self.agent_loop_workers.append(self.agent_loop_workers_class.options(name=f'agent_loop_worker_{i}' + f'_{uuid4().hex[:8]}', scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(node_id=node_id, soft=True)).remote(self.config, self.llm_client, self.teacher_client, self.reward_loop_worker_handles))

    @auto_await
    @SkipManager.annotate(role='rollout')
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to agent loop workers.

Args:
    prompts (DataProto): Input batch.

Returns:
    DataProto: Output batch."""
        num_workers = len(self.agent_loop_workers)
        if num_workers == 0:
            raise RuntimeError("No agent loop workers available")

        # Split batch into chunks for each worker
        chunk_size = max(1, len(prompts) // num_workers)
        chunks = []
        for i in range(num_workers):
            start = i * chunk_size
            if i < num_workers - 1:
                end = start + chunk_size
            else:
                end = len(prompts)
            if start < len(prompts):
                chunks.append(prompts[start:end])

        if not chunks:
            # Empty input, return as-is
            return prompts

        # Dispatch to workers
        worker_futures = []
        for worker, chunk in zip(self.agent_loop_workers, chunks):
            future = worker.generate_sequences.remote(chunk)
            worker_futures.append(future)

        # Gather results
        results: list[DataProto] = ray.get(worker_futures)

        # Normalize non_tensor_batch keys across results before concatenation
        if len(results) > 1:
            all_keys: set[str] = set()
            for r in results:
                if r.non_tensor_batch:
                    all_keys.update(r.non_tensor_batch.keys())
            for r in results:
                batch_size = len(r.batch) if r.batch is not None else 0
                for key in all_keys:
                    if key not in r.non_tensor_batch:
                        # Fill missing keys with None-valued object arrays
                        r.non_tensor_batch[key] = np.array([None] * batch_size, dtype=object)

        # Concatenate results
        if len(results) == 1:
            output = results[0]
        else:
            output = DataProto.concat(results)

        # Compute performance metrics from non_tensor_batch
        metrics_list = []
        if output.non_tensor_batch and 'metrics' in output.non_tensor_batch:
            metrics_arr = output.non_tensor_batch['metrics']
            if isinstance(metrics_arr, np.ndarray):
                metrics_list = [metrics_arr.tolist()]

        timing = self._performance_metrics(metrics_list, output)
        output.meta_info['timing'] = timing

        return output

    def _performance_metrics(self, metrics: list[list[dict[str, str]]], output: DataProto) -> dict[str, float]:
        timing = {}
        # Flatten metrics, filtering out any empty/invalid entries
        flat_metrics = [metric for chunk in metrics for metric in chunk if isinstance(metric, dict) and 'generate_sequences' in metric]
        if not flat_metrics:
            return timing
        t_generate_sequences = np.array([metric['generate_sequences'] for metric in flat_metrics])
        t_tool_calls = np.array([metric['tool_calls'] for metric in flat_metrics])
        t_compute_score = np.array([metric['compute_score'] for metric in flat_metrics])
        num_preempted = np.array([metric['num_preempted'] for metric in flat_metrics])
        timing['agent_loop/num_preempted/min'] = num_preempted.min()
        timing['agent_loop/num_preempted/max'] = num_preempted.max()
        timing['agent_loop/num_preempted/mean'] = num_preempted.mean()
        timing['agent_loop/generate_sequences/min'] = t_generate_sequences.min()
        timing['agent_loop/generate_sequences/max'] = t_generate_sequences.max()
        timing['agent_loop/generate_sequences/mean'] = t_generate_sequences.mean()
        timing['agent_loop/tool_calls/min'] = t_tool_calls.min()
        timing['agent_loop/tool_calls/max'] = t_tool_calls.max()
        timing['agent_loop/tool_calls/mean'] = t_tool_calls.mean()
        timing['agent_loop/compute_score/min'] = t_compute_score.min()
        timing['agent_loop/compute_score/max'] = t_compute_score.max()
        timing['agent_loop/compute_score/mean'] = t_compute_score.mean()
        slowest = np.argmax(t_generate_sequences + t_tool_calls + t_compute_score)
        prompt_length = output.batch['prompts'].shape[1]
        timing['agent_loop/slowest/generate_sequences'] = t_generate_sequences[slowest]
        timing['agent_loop/slowest/tool_calls'] = t_tool_calls[slowest]
        timing['agent_loop/slowest/compute_score'] = t_compute_score[slowest]
        timing['agent_loop/slowest/num_preempted'] = num_preempted[slowest]
        if 'attention_mask' in output.batch:
            attention_mask = output.batch['attention_mask'][slowest]
            timing['agent_loop/slowest/prompt_length'] = attention_mask[:prompt_length].sum().item()
            timing['agent_loop/slowest/response_length'] = attention_mask[prompt_length:].sum().item()
        return timing
