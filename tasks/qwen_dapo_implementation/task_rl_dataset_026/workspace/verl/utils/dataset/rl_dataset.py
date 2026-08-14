

import asyncio
import copy
import logging
import os
import re
import traceback
from collections import defaultdict
from io import BytesIO
from typing import Any, Optional
import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
from verl.utils.import_utils import load_extern_object
from verl.utils.tokenizer import build_multimodal_processor_inputs, normalize_token_ids
logger = logging.getLogger(__name__)

def collate_fn(data_list: list[dict]) -> dict:
    """Collate a batch of sample dicts into batched tensors and arrays.

Args:
    data_list: List of dicts mapping feature names to torch.Tensor or other values.

Returns:
    Dict where tensor entries are stacked into a torch.Tensor of shape
    (batch_size, \\*dims) and non-tensor entries are converted to
    np.ndarray of dtype object with shape (batch_size,)."""
    if not data_list:
        return {}
    batch = {}
    keys = data_list[0].keys()
    has_tensor = False
    for key in keys:
        values = [item[key] for item in data_list]
        if isinstance(values[0], torch.Tensor):
            batch[key] = torch.stack(values, dim=0)
            has_tensor = True
        else:
            arr = np.empty(len(values), dtype=object)
            arr[:] = values
            batch[key] = arr
    # The V0 trainer path (trainer.use_v1=False) uses len(batch.batch) on the
    # DataProto to determine batch size.  If no tensor fields are present the
    # DataProto batch TensorDict is None, which crashes with TypeError.
    # Insert a minimal sentinel tensor so that DataProto.from_single_dict
    # always produces a non-None TensorDict.
    if not has_tensor:
        batch["_ds_batch_tensor"] = torch.zeros(len(data_list), dtype=torch.long)
    return batch

class RLHFDataset(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(self, data_files: str | list[str], tokenizer: PreTrainedTokenizer, config: DictConfig, processor: Optional[ProcessorMixin]=None, max_samples: int=-1):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]
        self.data_files = [str(f) for f in data_files]
        self.original_data_files = list(self.data_files)
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = config.get("cache_dir", None)
        self.use_shm = config.get("use_shm", False)
        self.prompt_key = config.get("prompt_key", "prompt")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed", None)
        self.serialize_dataset = False
        self.max_samples = max_samples

        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local
        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        import pandas as pd
        dataframes = []
        for parquet_file in self.data_files:
            df = pd.read_parquet(parquet_file)
            dataframes.append(df)
        self.dataframe = pd.concat(dataframes, ignore_index=True)

        total = len(self.dataframe)
        logger.info(f"dataset len: {total}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rng_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rng_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.iloc[indices.tolist()]
            logger.info(f"selected {self.max_samples} random samples out of {total}")

        # Convert to HuggingFace Dataset for compatibility
        self.dataframe = datasets.Dataset.from_pandas(self.dataframe)

        if self.filter_overlong_prompts:
            self.maybe_filter_out_long_prompts()

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset=None):
        if dataframe is None:
            dataframe = self.dataframe

        max_length = self.max_prompt_length
        filtered = []
        for i, row in enumerate(dataframe):
            messages = self._build_messages(dict(row), self.prompt_key)
            # Tokenize to check length
            try:
                tokenized = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                )
                if len(tokenized) <= max_length:
                    filtered.append(i)
            except Exception:
                # If tokenization fails, keep the sample (conservative)
                filtered.append(i)

        if len(filtered) < len(dataframe):
            logger.info(
                f"filtered out {len(dataframe) - len(filtered)} long prompts "
                f"({len(filtered)} remaining)"
            )
            self.dataframe = dataframe.select(filtered)

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, 'original_data_files')
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)
            self._read_files_and_tokenize()
        else:
            print('old dataloader ckpt file is used, please train from scratch for better ckpt performance')

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()
            if 'dataframe' in state:
                del state['dataframe']
            return state
        return self.__dict__.copy()

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict, key: str):
        """Replace multimodal placeholders in messages with structured content.

This is required by processor.apply_chat_template.
- <image>: {"type": "image", "image": image} or {"type": "image", **image}
- <video>: {"type": "video", "video": video} or {"type": "video", **video}
- <audio>: {"type": "audio", "audio": audio} or {"type": "audio", **audio}

Args:
    example: Row dictionary from dataframe.
    key: The key in the example dict that contains the messages.

Returns:
    messages: List of messages with replaced placeholder."""
        messages = example.get(key, [])
        if isinstance(messages, np.ndarray):
            messages = messages.tolist()
        # Make a deep copy to avoid mutating the original
        messages = copy.deepcopy(messages)

        if self.processor is None:
            return messages

        images = example.get("images", []) or []
        videos = example.get("videos", []) or []
        audios = example.get("audios", []) or []

        if isinstance(images, np.ndarray):
            images = images.tolist()
        if isinstance(videos, np.ndarray):
            videos = videos.tolist()
        if isinstance(audios, np.ndarray):
            audios = audios.tolist()

        image_offset = 0
        video_offset = 0
        audio_offset = 0

        for message in messages:
            content = message.get("content")
            if not isinstance(content, str):
                continue

            # Check if there are multimodal placeholders
            has_placeholder = bool(re.search(r"<(image|video|audio)>", content))
            if not has_placeholder:
                if len(images) == 0 and len(videos) == 0 and len(audios) == 0:
                    # No multimodal data at all — keep as plain text
                    continue
                # Wrap plain text in content list for processor
                message["content"] = [{"type": "text", "text": content}]
                continue

            content_list = []
            segments = re.split(r"(<(?:image|video|audio)>)", content)
            segments = [s for s in segments if s != ""]
            for segment in segments:
                if segment == "<image>":
                    if image_offset < len(images):
                        img = images[image_offset]
                        if isinstance(img, str):
                            content_list.append({"type": "image", "image": img})
                        elif isinstance(img, Image.Image):
                            content_list.append({"type": "image", "image": img})
                        else:
                            content_list.append({"type": "image", **img} if isinstance(img, dict) else {"type": "image", "image": img})
                        image_offset += 1
                elif segment == "<video>":
                    if video_offset < len(videos):
                        vid = videos[video_offset]
                        if isinstance(vid, dict):
                            content_list.append({"type": "video", **vid})
                        else:
                            content_list.append({"type": "video", "video": vid})
                        video_offset += 1
                elif segment == "<audio>":
                    if audio_offset < len(audios):
                        aud = audios[audio_offset]
                        if isinstance(aud, dict):
                            content_list.append({"type": "audio", **aud})
                        else:
                            content_list.append({"type": "audio", "audio": aud})
                        audio_offset += 1
                else:
                    content_list.append({"type": "text", "text": segment})
            message["content"] = content_list

        return messages

    def __getitem__(self, item):
        """For rollout, apply_chat_template has been moved to AgentLoop, so we only return raw_prompt here."""
        row_dict = self.dataframe[item]
        messages = self._build_messages(row_dict, self.prompt_key)

        result = {
            "raw_prompt": messages,
            "data_source": row_dict.get("data_source", ""),
            "reward_model": row_dict.get("reward_model", {}),
            "extra_info": row_dict.get("extra_info", {}),
            "index": row_dict.get("extra_info", {}).get("index", item),
        }
        return result

    @classmethod
    async def process_vision_info(cls, messages: list[dict], image_patch_size, config: DictConfig) -> tuple[list[Image.Image], list[tuple[torch.Tensor, dict]]]:
        """Extract images and videos from messages.

        This method is called by AgentLoop (e.g SingleTurnAgentLoop) before apply_chat_template to
        the `raw_prompt` from dataset. User may customize RLHFDataset and override this method to
        support custom vision extraction.

        >>> messages = kwargs["raw_prompt"]
        >>> images, videos = RLHFDataset.process_vision_info(messages, image_patch_size)
        >>> videos, video_metadatas = zip(*videos)
        >>> raw_prompt = processor.apply_chat_template(messages, tokenize=False)
        >>> inputs = processor(text=[raw_prompt], images=images, videos=videos,
        ...                    video_metadata=video_metadatas, do_sample_frames=False)

        Args:
            messages: List of messages from dataset `raw_prompt`.
            image_patch_size: Image patch size for processor.
            config: Config for dataset.

        Returns:
            images: List of images.
            videos: List of videos, each video is a tuple of (video_tensor, video_metadata).
        """
        from qwen_vl_utils import process_vision_info
        loop = asyncio.get_running_loop()
        images, videos = await loop.run_in_executor(None, lambda: process_vision_info(messages, image_patch_size=image_patch_size, return_video_metadata=True))
        return (images, videos)

    @classmethod
    def _extract_audio_info(cls, messages: list[dict]) -> list[Any]:
        audios = []
        for message in messages:
            content = message.get('content')
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict) or item.get('type') != 'audio':
                    continue
                if 'audio' in item:
                    audios.append(item['audio'])
                elif 'audio_url' in item:
                    audios.append(item['audio_url'])
                else:
                    audios.append({k: v for k, v in item.items() if k != 'type'})
        return audios or None

    @classmethod
    def _process_multi_modal_info(cls, messages: list[dict], image_patch_size, config: DictConfig) -> tuple[list[Image.Image], list[Any], list[Any]]:
        has_visual = any((isinstance(message.get('content'), list) and any((isinstance(item, dict) and item.get('type') in {'image', 'video'} for item in message['content'])) for message in messages))
        if has_visual:
            from qwen_vl_utils import process_vision_info
            images, videos = process_vision_info(messages, image_patch_size=image_patch_size, return_video_metadata=True)
        else:
            images, videos = (None, None)
        audios = cls._extract_audio_info(messages)
        return (images, videos, audios)

    @classmethod
    async def process_multi_modal_info(cls, messages: list[dict], image_patch_size, config: DictConfig) -> tuple[list[Image.Image], list[Any], list[Any]]:
        return cls._process_multi_modal_info(messages, image_patch_size=image_patch_size, config=config)

    def split(self, num_splits: int):
        """split the dataset into num_splits sub-datasets
Args:
    num_splits: specified number of splits
Returns:
    List[RLHFDataset]: list of RLHFDataset splits
Raises:
    ValueError: if num_splits is not a positive integer"""
        if not isinstance(num_splits, int) or num_splits <= 0:
            raise ValueError(f"num_splits must be a positive integer, got {num_splits}")

        total = len(self.dataframe)
        split_size = (total + num_splits - 1) // num_splits

        sub_datasets = []
        for i in range(num_splits):
            start = i * split_size
            end = min((i + 1) * split_size, total)
            if start >= total:
                break
            # Create a shallow copy and assign a sliced dataframe
            sub_dataset = copy.copy(self)
            sub_dataset.dataframe = self.dataframe.select(range(start, end))
            sub_datasets.append(sub_dataset)

        return sub_datasets

def get_dataset_class(data_config: DictConfig):
    """Get RLHF dataset class.

Args:
    data_config: The data config.

Returns:
    dataset_cls: The dataset class."""
    # Check if a custom dataset class is specified in config
    custom_cls = data_config.get("custom_cls", None)
    if custom_cls is not None and custom_cls.get("path") and custom_cls.get("name"):
        return load_extern_object(custom_cls.path, custom_cls.name)
    return RLHFDataset
