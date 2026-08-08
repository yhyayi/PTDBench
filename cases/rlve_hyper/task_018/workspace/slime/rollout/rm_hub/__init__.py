import asyncio
from typing import Union

import re
import aiohttp

from slime.utils.misc import load_function
from slime.utils.types import Sample

from .math_utils import extract_answer as extract_boxed_answer
from .math_utils import grade_answer_verl

from .bbeh import compute_score as bbeh_compute_score

from .livecodebench import evaluate_single_example as livecodebench_compute_score

from .rlve_rm import rlve_rm

import json


async def remote_rm(args, sample: Sample):
    payload = {
        "prompt": sample.prompt,
        "response": sample.response,
        "label": sample.label,
    }
    session_kwargs = {}
    async with aiohttp.ClientSession(**session_kwargs) as session:
        async with session.post(args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


async def async_rm(args, sample: Sample, **kwargs):
    if args.custom_rm_path is not None:
        rm_function = load_function(args.custom_rm_path)
        return await rm_function(args, sample, **kwargs)

    rm_type = args.rm_type
    response = sample.response
    label = sample.label
    if rm_type.startswith("boxed_"):
        response = extract_boxed_answer(response) or ""
        rm_type = rm_type[len("boxed_") :]

    if rm_type in ["math", "bbeh", "livecodebench", "rlve"]:
        if rm_type == "math":
            answer_matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL)
            if answer_matches:
                solution = answer_matches[-1].strip()
                if r"\boxed{" not in solution:
                    solution = r"\boxed{" + solution + r"}"
            else:
                solution = response
            return 1 if grade_answer_verl(solution_str=solution, ground_truth=label) else 0
        elif rm_type == "bbeh":
            return bbeh_compute_score(predict_str=response, ground_truth=label)  # Direct call, no timeout needed
        elif rm_type == "livecodebench":
            return 1 if await livecodebench_compute_score(example=sample.metadata, response=response) else 0
        elif rm_type == "rlve":
            metadata = sample.metadata
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            return rlve_rm(args=args, environment=metadata["environment"], config=metadata["config"], response=response)
        else:
            raise NotImplementedError(f"Custom RM for {rm_type} is not implemented.")
    
    # Simple reward functions without timeout issues
    if rm_type == "remote_rm":
        raise NotImplementedError("Remote RM is not implemented.")
        return await remote_rm(args, sample)
    else:
        raise NotImplementedError(f"Rule-based RM for {rm_type} is not implemented.")


async def batched_async_rm(
    args,
    samples: list[Sample],
    **kwargs,
) -> list[Union[int, float]]:
    if args.custom_rm_path is not None:
        # Ensure the custom reward function is implemented in batch mode
        rm_function = load_function(args.custom_rm_path)
        return await rm_function(args, samples, **kwargs)
    tasks = [async_rm(args, sample, **kwargs) for sample in samples]
    rewards = await asyncio.gather(*tasks)
    return rewards
