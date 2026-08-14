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
        """Build the initial prompt token ids with Continuous Token."""
        prompt_ids = await self.loop.run_in_executor(None, lambda: self.continuous_token_builder.build_initial_tokens(messages, tools=tools))
        return self._cap_text_prompt_length(prompt_ids)

    async def ct_merge_non_assistant_msg(self, previous_messages: list[dict], updated_messages: list[dict], runtime_token_ids: list[int], response_mask: list[int], response_logprobs: Optional[list[float]]=None, tools: list[dict]=None):
        """Merge appended non-assistant messages into runtime tokens and metadata."""
        merge_result = await self.loop.run_in_executor(None, lambda: self.continuous_token_builder.merge_non_assistant_tokens(previous_messages, updated_messages, runtime_token_ids, tools=tools))
        aligned_response_mask, aligned_response_logprobs = self.continuous_token_builder.align_response_metadata(merge_result, response_mask, response_logprobs)
        return (merge_result, aligned_response_mask, aligned_response_logprobs)

    async def ct_merge_assistant_token(self, runtime_token_ids: list[int], assistant_token_ids: list[int], response_mask: list[int], response_logprobs: Optional[list[float]]=None, assistant_logprobs: Optional[list[float]]=None):
        """Merge assistant-generated tokens and align response metadata."""
        merge_result = await self.loop.run_in_executor(None, lambda: self.continuous_token_builder.merge_assistant_tokens(runtime_token_ids, assistant_token_ids))
        aligned_response_mask, aligned_response_logprobs = self.continuous_token_builder.align_response_metadata(merge_result, response_mask, response_logprobs, assistant_logprobs=assistant_logprobs)
        return (merge_result, aligned_response_mask, aligned_response_logprobs)

    def _cap_text_prompt_length(self, prompt_ids: list[int]) -> list[int]:
        prompt_length = self.rollout_config.prompt_length
        if len(prompt_ids) > prompt_length:
            logger.warning('Prompt of %d tokens exceeds rollout.prompt_length=%d; left-truncating.', len(prompt_ids), prompt_length)
            return prompt_ids[-prompt_length:]
        return prompt_ids

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
        apply_kwargs = dict(self.apply_chat_template_kwargs)
        if images is not None:
            apply_kwargs['images'] = images
        if videos is not None:
            apply_kwargs['videos'] = videos
        if audios is not None:
            apply_kwargs['audios'] = audios

        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: apply_chat_template(
                processing_class,
                messages,
                tools=tools,
                add_generation_prompt=True,
                tokenize=True,
                **apply_kwargs,
            )
        )
        prompt_ids = normalize_token_ids(prompt_ids)

        if remove_system_prompt and self.system_prompt:
            if prompt_ids[:len(self.system_prompt)] == self.system_prompt:
                prompt_ids = prompt_ids[len(self.system_prompt):]

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
        messages = list(kwargs["raw_prompt"])

        # 1. extract multimodal inputs from messages
        multi_modal_data = await self.process_multi_modal_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)

        # 2. apply chat template and tokenize
        use_continuous_token = self.enable_continuous_token and not multi_modal_data
        if use_continuous_token:
            prompt_ids = await self.ct_build_initial_tokens(messages)
        else:
            prompt_ids = await self.apply_chat_template(
                messages,
                images=images,
                videos=videos,
                audios=audios,
                mm_processor_kwargs=mm_processor_kwargs,
            )

        # 3. generate sequences
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            request_id = uuid4().hex
            output = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                audio_data=audios,
                video_data=videos,
                mm_processor_kwargs=mm_processor_kwargs,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1

        if use_continuous_token:
            merge_result, response_mask, response_logprobs = await self.ct_merge_assistant_token(
                prompt_ids,
                output.token_ids,
                [],
                [] if output.log_probs else None,
                assistant_logprobs=output.log_probs if output.log_probs else None,
            )
            response_ids = merge_result.token_ids[-len(response_mask):] if response_mask else []
            prompt_ids = merge_result.token_ids[:len(merge_result.token_ids) - len(response_mask)]
        else:
            response_ids = output.token_ids
            response_mask = [1] * len(output.token_ids)
            response_logprobs = output.log_probs

        response_length = self.rollout_config.response_length

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[:response_length],
            response_mask=response_mask[:response_length],
            response_logprobs=response_logprobs[:response_length] if response_logprobs else None,
            routed_experts=(
                output.routed_experts[:len(prompt_ids) + response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=mm_processor_kwargs,
            num_turns=2,
            metrics=metrics,
            extra_fields=output.extra_fields,
        )
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
        # Build sampling params from rollout config
        sampling_params = dict(
            temperature=self.rollout_config.temperature,
            top_k=self.rollout_config.top_k,
            top_p=self.rollout_config.top_p,
            n=self.rollout_config.n,
            repetition_penalty=self.rollout_config.repetition_penalty,
        )

        # Determine agent name
        agent_name = self.rollout_config.agent.default_agent_loop

        # Build kwargs for each sample
        batch_size = len(batch)
        non_tensor_batch = batch.non_tensor_batch

        # Extract trajectory info
        step = batch.meta_info.get("step", 0)
        index = non_tensor_batch.get("index", list(range(batch_size)))
        validate = batch.meta_info.get("validate", False)
        trajectory_info = await get_trajectory_info(step, index, validate)

        # Run agent loops in parallel
        tasks = []
        for i in range(batch_size):
            sample_kwargs = {}
            for key in non_tensor_batch:
                val = non_tensor_batch[key]
                if isinstance(val, np.ndarray):
                    item = val[i]
                    if isinstance(item, np.ndarray) and item.ndim == 0:
                        sample_kwargs[key] = item.item()
                    else:
                        sample_kwargs[key] = item
                else:
                    sample_kwargs[key] = val[i]

            tasks.append(
                self._run_agent_loop(
                    sampling_params,
                    trajectory_info[i],
                    agent_name=agent_name,
                    trace=True,
                    **sample_kwargs,
                )
            )

        outputs: list[_InternalAgentLoopOutput] = await asyncio.gather(*tasks)

        # Post-process into DataProto
        return self._postprocess(outputs, non_tensor_batch, validate=validate)

    async def _run_agent_loop(self, sampling_params: dict[str, Any], trajectory: dict[str, Any], *, agent_name: str, trace: bool=True, **kwargs) -> _InternalAgentLoopOutput:
        # Look up agent loop class from registry
        if agent_name not in _agent_loop_registry:
            raise ValueError(
                f"Agent loop '{agent_name}' not found in registry. "
                f"Available: {list(_agent_loop_registry.keys())}"
            )

        agent_loop_config = _agent_loop_registry[agent_name]

        # Instantiate the agent loop
        trainer_config = DictConfigWrap(self.config)
        data_config = DictConfigWrap(self.config.data)

        agent_loop: AgentLoopBase = hydra.utils.instantiate(
            agent_loop_config,
            trainer_config=trainer_config,
            server_manager=self.llm_client,
            tokenizer=self.tokenizer,
            processor=self.processor,
            dataset_cls=self.dataset_cls,
            data_config=data_config,
            tools=ToolListWrap(self.tools) if self.tools else None,
        )

        # Run the agent loop
        output: AgentLoopOutput = await agent_loop.run(
            sampling_params,
            **kwargs,
        )

        # Compute score (synchronous reward via reward_loop_worker_handles)
        kwargs_with_validate = dict(kwargs, validate=trajectory.get("validate", False))
        await self._compute_score([output], kwargs_with_validate)

        # Compute teacher logprobs if distillation enabled
        await self._compute_teacher_logprobs(
            output,
            output.prompt_ids,
            output.response_ids,
            validate=trajectory.get("validate", False),
            sample_kwargs=kwargs.get("extra_info", {}),
        )

        # Post-process
        return await self._agent_loop_postprocess(
            output,
            validate=trajectory.get("validate", False),
            **kwargs,
        )

    def _pad_token_ids(self, tokens: list[int], *, max_length: int, padding_side: str, return_attention_mask: bool) -> dict[str, torch.Tensor]:
        """Right/left pad a flat list of token ids to a ``(1, max_length)`` tensor."""
        if not tokens:
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            result = {'input_ids': torch.full((1, max_length), pad_id, dtype=torch.long)}
            if return_attention_mask:
                result['attention_mask'] = torch.zeros((1, max_length), dtype=torch.long)
            return result
        self.tokenizer.padding_side = padding_side
        padded = self.tokenizer.pad({'input_ids': [tokens]}, padding='max_length', max_length=max_length, return_tensors='pt', return_attention_mask=return_attention_mask)
        if padded['input_ids'].dim() == 1:
            padded['input_ids'] = padded['input_ids'].unsqueeze(0)
            if return_attention_mask:
                padded['attention_mask'] = padded['attention_mask'].unsqueeze(0)
        return padded

    async def _agent_loop_postprocess(self, output, validate, **kwargs) -> _InternalAgentLoopOutput:
        """Perform post-processing operations on the output of each individual agent loop."""
        prompt_ids = output.prompt_ids
        response_ids = output.response_ids
        response_mask = output.response_mask
        response_logprobs = output.response_logprobs

        prompt_length = self.rollout_config.prompt_length
        response_length = self.rollout_config.response_length

        # Pad prompt (left-pad)
        prompt_padded = self._pad_token_ids(
            prompt_ids, max_length=prompt_length, padding_side="left", return_attention_mask=False
        )
        prompt_tensor = prompt_padded["input_ids"]  # (1, prompt_length)

        # Pad response (right-pad)
        response_padded = self._pad_token_ids(
            response_ids, max_length=response_length, padding_side="right", return_attention_mask=False
        )
        response_tensor = response_padded["input_ids"]  # (1, response_length)

        # Build input_ids = prompt + response
        input_ids = torch.cat([prompt_tensor, response_tensor], dim=1)  # (1, total_length)

        # Build attention_mask
        prompt_attn = self._pad_token_ids(
            prompt_ids, max_length=prompt_length, padding_side="left", return_attention_mask=True
        )
        response_attn = self._pad_token_ids(
            response_ids, max_length=response_length, padding_side="right", return_attention_mask=True
        )
        attention_mask = torch.cat(
            [prompt_attn["attention_mask"], response_attn["attention_mask"]], dim=1
        )

        # Build response_mask (pad right with 0)
        padded_response_mask = list(response_mask) + [0] * (response_length - len(response_mask))
        response_mask_tensor = torch.tensor(
            [padded_response_mask[:response_length]], dtype=torch.int64
        )

        # Build position_ids
        multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
        position_ids = self._compute_position_ids(
            input_ids, attention_mask, multi_modal_inputs, output.mm_processor_kwargs
        )

        # Handle response_logprobs
        if response_logprobs is not None:
            padded_logprobs = list(response_logprobs) + [0.0] * (response_length - len(response_logprobs))
            response_logprobs_tensor = torch.tensor(
                [padded_logprobs[:response_length]], dtype=torch.float32
            )
        else:
            response_logprobs_tensor = None

        return _InternalAgentLoopOutput(
            prompt_ids=prompt_tensor,
            response_ids=response_tensor,
            input_ids=input_ids,
            position_ids=position_ids,
            response_mask=response_mask_tensor,
            attention_mask=attention_mask,
            response_logprobs=response_logprobs_tensor,
            routed_experts=output.routed_experts,
            multi_modal_inputs=multi_modal_inputs,
            multi_modal_data=output.multi_modal_data,
            mm_processor_kwargs=output.mm_processor_kwargs,
            reward_score=output.reward_score,
            num_turns=output.num_turns,
            metrics=output.metrics,
            extra_fields=output.extra_fields,
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
        multi_modal_inputs = build_multimodal_processor_inputs(self.processor, text=[current_text], images=images, videos=videos, audio=audios, mm_processor_kwargs=output.mm_processor_kwargs if output.mm_processor_kwargs is not None else self._get_mm_processor_kwargs(audios))
        multi_modal_inputs.pop('input_ids', None)
        multi_modal_inputs.pop('attention_mask', None)
        multi_modal_inputs = dict(multi_modal_inputs.convert_to_tensors('pt'))
        image_grid_thw = multi_modal_inputs.get('image_grid_thw')
        if image_grid_thw is not None:
            images_seqlens = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0])
            multi_modal_inputs['images_seqlens'] = images_seqlens
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
        # Compute reward score using reward loop workers if available
        if self.reward_loop_worker_handles is not None and len(self.reward_loop_worker_handles) > 0:
            last_output = outputs[-1]
            # Build a minimal DataProto for the reward worker
            reward_data = DataProto.from_single_dict(
                {
                    "prompts": torch.tensor(last_output.prompt_ids, dtype=torch.int64).unsqueeze(0),
                    "responses": torch.tensor(last_output.response_ids, dtype=torch.int64).unsqueeze(0),
                    "attention_mask": torch.ones(
                        1, len(last_output.prompt_ids) + len(last_output.response_ids), dtype=torch.int64
                    ),
                },
                meta_info={"validate": kwargs.get("validate", False)},
            )
            if "raw_prompt" in kwargs:
                reward_data.non_tensor_batch["raw_prompt"] = np.array(
                    [kwargs["raw_prompt"]], dtype=object
                )
            if "data_source" in kwargs:
                reward_data.non_tensor_batch["data_source"] = np.array(
                    [kwargs["data_source"]], dtype=object
                )
            if "reward_model" in kwargs:
                reward_data.non_tensor_batch["reward_model"] = np.array(
                    [kwargs["reward_model"]], dtype=object
                )
            if "extra_info" in kwargs and kwargs["extra_info"] is not None:
                reward_data.non_tensor_batch["extra_info"] = np.array(
                    [kwargs["extra_info"]], dtype=object
                )

            # Pick a random reward worker and compute score
            worker = random.choice(self.reward_loop_worker_handles)
            result = await worker.compute_score.remote(reward_data)
            reward_score = result.get("reward_score", 0.0) if isinstance(result, dict) else 0.0
        else:
            reward_score = 0.0

        outputs[-1].reward_score = reward_score

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
        # Combine all outputs into a batch
        prompt_ids = torch.cat([o.prompt_ids for o in inputs], dim=0)
        response_ids = torch.cat([o.response_ids for o in inputs], dim=0)
        input_ids = torch.cat([o.input_ids for o in inputs], dim=0)
        position_ids = torch.cat([o.position_ids for o in inputs], dim=0)
        response_mask = torch.cat([o.response_mask for o in inputs], dim=0)
        attention_mask = torch.cat([o.attention_mask for o in inputs], dim=0)

        batch_dict = {
            "prompts": prompt_ids,
            "responses": response_ids,
            "input_ids": input_ids,
            "position_ids": position_ids,
            "response_mask": response_mask,
            "attention_mask": attention_mask,
        }

        # Add optional fields
        if all(o.response_logprobs is not None for o in inputs):
            batch_dict["rollout_log_probs"] = torch.cat(
                [o.response_logprobs for o in inputs], dim=0
            )

        # Build reward scores (rm_scores): place reward at last valid response token
        rm_scores_list = []
        for o in inputs:
            rm_score = torch.zeros_like(o.response_mask, dtype=torch.float32)
            if o.reward_score is not None and o.reward_score != 0.0:
                # Find last position where response_mask == 1
                valid_len = int(o.response_mask.sum().item())
                if valid_len > 0:
                    rm_score[0, valid_len - 1] = o.reward_score
            rm_scores_list.append(rm_score)
        batch_dict["rm_scores"] = torch.cat(rm_scores_list, dim=0)

        # Collect metrics
        metrics_list = []
        for o in inputs:
            if hasattr(o.metrics, "model_dump"):
                metrics_list.append(o.metrics.model_dump())
            elif hasattr(o.metrics, "generate_sequences"):
                metrics_list.append({
                    "generate_sequences": o.metrics.generate_sequences,
                    "tool_calls": o.metrics.tool_calls,
                    "compute_score": o.metrics.compute_score,
                    "num_preempted": o.metrics.num_preempted,
                })
            else:
                metrics_list.append({})

        # Build non_tensors dict with multi_modal_inputs
        non_tensors = dict(input_non_tensor_batch) if input_non_tensor_batch else {}
        multi_modal_inputs_list = []
        for o in inputs:
            if o.multi_modal_inputs is not None and len(o.multi_modal_inputs) > 0:
                multi_modal_inputs_list.append(o.multi_modal_inputs)
            else:
                multi_modal_inputs_list.append({})
        non_tensors["multi_modal_inputs"] = np.array(multi_modal_inputs_list, dtype=object)

        return DataProto.from_dict(
            tensors=batch_dict,
            non_tensors=non_tensors,
            meta_info={"metrics": metrics_list, "validate": validate},
        )

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
        batch_size = len(prompts)

        if batch_size == 0:
            return prompts

        # Pad batch to be divisible by num_workers
        if batch_size % num_workers != 0:
            pad_size = num_workers - (batch_size % num_workers)
            # Duplicate last samples for padding
            pad_indices = list(range(batch_size - pad_size, batch_size))
            pad_batch = prompts.select_idxs(pad_indices)
            prompts = DataProto.concat([prompts, pad_batch])
        else:
            pad_size = 0

        chunks = prompts.chunk(num_workers)

        # Dispatch to workers and await results
        refs = [
            worker.generate_sequences.remote(chunk)
            for worker, chunk in zip(self.agent_loop_workers, chunks)
        ]
        outputs = [await ref for ref in refs]

        # Concatenate results
        result = DataProto.concat(outputs)

        # Remove padding if any was added
        if pad_size > 0:
            result = result[:batch_size]

        # Compute performance metrics
        metrics = [chunk.meta_info.get("metrics", []) for chunk in outputs]
        timing = self._performance_metrics(metrics, result)
        result.meta_info["timing"] = timing

        return result

    def _performance_metrics(self, metrics: list[list[dict[str, str]]], output: DataProto) -> dict[str, float]:
        timing = {}
        t_generate_sequences = np.array([metric['generate_sequences'] for chunk in metrics for metric in chunk])
        t_tool_calls = np.array([metric['tool_calls'] for chunk in metrics for metric in chunk])
        t_compute_score = np.array([metric['compute_score'] for chunk in metrics for metric in chunk])
        num_preempted = np.array([metric['num_preempted'] for chunk in metrics for metric in chunk])
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
