import asyncio
import copy
import os
import json

import argparse
from tqdm import tqdm
from sglang.utils import wait_for_server, terminate_process
from slime.utils.data import Dataset
from transformers import AutoTokenizer

from slime.utils.types import Sample
from typing import List
from slime.rollout.rm_hub import async_rm

from slime.utils.http_utils import post

import weakref
from sglang.utils import execute_shell_command, reserve_port

from asyncio import Semaphore


# Fix uvloop compatibility issue with nest_asyncio
# Set the event loop policy to default before applying nest_asyncio
import asyncio
import nest_asyncio
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
nest_asyncio.apply()


def launch_server_cmd(command: str, host: str = "0.0.0.0", port: int = None):
    """
    Launch the server using the given command.
    If no port is specified, a free port is reserved.
    """
    if port is None:
        port, lock_socket = reserve_port(host)
    else:
        lock_socket = None

    full_command = f"{command} --port {port}"
    process = execute_shell_command(full_command)
    
    if lock_socket is not None:
        process_socket_map = weakref.WeakKeyDictionary()
        process_socket_map[process] = lock_socket

    return process, port


parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type = str, required = True)
parser.add_argument("--num-gpus", type = int, required = True)
parser.add_argument("--sglang-server-concurrency", type = int, default = 320)

parser.add_argument(
    "--eval-data",
    type=str,
    default=None,
    nargs="+",
    help=(
        "Path to the evaluation data, "
        "should first input the name of the eval dataset, then the path, and then the config, e.g. "
        "AIME25 data/BENCHMARKS/AIME/AIME25.json data/BENCHMARKS/AIME/evaluation_config.json"
    ),
)

parser.add_argument("--eval-temperature", type=float, default=0.6)
parser.add_argument("--eval-top-p", type=float, default=0.95)
parser.add_argument("--eval-top-k", type=int, default=-1)
parser.add_argument("--eval-max-response-len", type=int, default=32768)
parser.add_argument("--eval-max-context-len", type=int, default=None)

parser.add_argument(
    "--rollout-stop",
    type=str,
    nargs="+",
    default=None,
    help=(
        "The stop words for the inference engine during rollout. "
        "It can be a list of strings or a single string. "
        "It may be hard to pass special tokens in command line, in that case rollout_stop_token_ids can be used."
    ),
)
parser.add_argument(
    "--rollout-stop-token-ids",
    type=int,
    nargs="+",
    default=None,
    help=(
        "The stop token ids for the inference engine during rollout. "
        "It can be a list of integers or a single integer."
    ),
)
parser.add_argument(
    "--rollout-skip-special-tokens",
    action="store_true",
    default=False,
    help=(
        "Whether to skip special tokens in the response during rollout. "
        "This is useful when you want to use the response as a prompt for the next rollout."
    ),
)

parser.add_argument("--apply-chat-template", action = "store_true", default = False)
parser.add_argument(
    "--answer-marker-type",
    type = str,
    default = r"\boxed{}",
    help = "The type of answer marker to use.",
)
parser.add_argument(
    "--custom-prompt-preprocessor",
    type = str,
    required = True,
    choices = ("TinyZero", "ChatTemplate_NoSystemPrompt", ),
    help = "Choose a custom prompt preprocessor.",
)

args = parser.parse_args()


MAX_CONCURRENT_REQUESTS = args.sglang_server_concurrency
semaphore = Semaphore(MAX_CONCURRENT_REQUESTS * args.num_gpus)


async def generate_and_rm(args, sample, sampling_params, sglang_port) :
    sampling_params = copy.deepcopy(sampling_params)
    if args.eval_max_context_len is not None :
        assert "input_length" in sample.metadata, "input_length must be in sample.metadata when eval_max_context_len is set"
        sampling_params["max_new_tokens"] = min(sampling_params["max_new_tokens"], args.eval_max_context_len - sample.metadata["input_length"] - 32)
    response = await post(
        url = "http://localhost:{sglang_port}/generate".format(sglang_port = sglang_port),
        payload = {
            "text": sample.prompt,
            "sampling_params": sampling_params,
        },
    )
    sample.response = response["text"]

    sample.reward = await async_rm(args = args, sample = sample)

    return sample


async def eval_rollout_single_dataset(args, name, path, config, sglang_port) :
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code = True)

    dataset = Dataset(
        path,
        tokenizer = tokenizer,
        max_length = None,
        prompt_key = "user_prompt",
        label_key = config["label_key"],
        metadata_key = "metadata",
        tool_key = None,
        apply_chat_template = args.apply_chat_template,
        args = args,
    )

    sampling_params = dict(
        temperature = args.eval_temperature,
        top_p = args.eval_top_p,
        top_k = args.eval_top_k,
        max_new_tokens = args.eval_max_response_len,
        stop = args.rollout_stop,
        stop_token_ids = args.rollout_stop_token_ids,
        skip_special_tokens = args.rollout_skip_special_tokens,
        no_stop_trim = True,
        spaces_between_special_tokens = False,
    )

    args = copy.deepcopy(args)
    args.group_rm = False
    args.custom_rm_path = None
    args.rm_type = config["rm_type"]

    async def generate_and_rm_with_semaphore(args, sample, sampling_params, sglang_port):
        async with semaphore:
            return await generate_and_rm(args, sample, sampling_params, sglang_port)

    tasks = []
    sample_index = 0
    for i, prompt_sample in enumerate(dataset.samples):
        for j in range(config["n_samples_per_eval_prompt"]):
            sample = copy.deepcopy(prompt_sample)
            sample.index = sample_index
            sample_index += 1
            tasks.append(
                generate_and_rm_with_semaphore(
                    args = args,
                    sample = sample,
                    sampling_params = sampling_params,
                    sglang_port = sglang_port,
                )
            )

    data : List[Sample] = []
    do_print = True
    pbar = tqdm(total=len(tasks), desc="Rollout generation", disable=not do_print)
    for coro in asyncio.as_completed(tasks):
        sample = await coro
        if do_print:
            print([sample.prompt + sample.response], sample.reward, flush=True)
            do_print = False
        data.append(sample)
        pbar.update(1)
    pbar.close()

    data.sort(key=lambda sample: sample.index)

    accuracy_key = config["accuracy_key"]
    accuracy = [sample.reward if not accuracy_key else sample.reward[accuracy_key] for sample in data]
    return {
        name : {
            "responses" : [sample.response for sample in data],
            "accuracy" : accuracy,
            "mean_accuracy" : sum(accuracy) / len(accuracy),
            "raw_rewards" : [sample.reward for sample in data],
        }
    }


async def main(sglang_port):
    assert len(args.eval_data) > 0 and len(args.eval_data) % 3 == 0
    tasks = []
    for i in range(0, len(args.eval_data), 3):
        name, path, config = args.eval_data[i : i + 3]
        with open(config) as fin:
            config = json.load(fin)
        tasks.append(eval_rollout_single_dataset(args = args, name = name, path = path, config = config, sglang_port = sglang_port))
    
    dataset_results = await asyncio.gather(*tasks)
    
    results = {}
    for dataset_result in dataset_results:
        results.update(dataset_result)
    
    os.makedirs(os.path.join(args.model_path, "evaluation_result"), exist_ok = True)
    for dataset_name, result in results.items() :
        with open(os.path.join(args.model_path, "evaluation_result/{}.json".format(dataset_name)), "w") as fout :
            json.dump(result, fout, indent = 2)

if __name__ == "__main__":
    server_process, sglang_port = launch_server_cmd(
        """
python -m sglang_router.launch_server \
    --model-path {model_path} \
    --dp-size {dp} \
    --router-request-timeout-secs 10000000
    """.format(model_path = args.model_path, dp = args.num_gpus)
)
    wait_for_server(f"http://localhost:{sglang_port}")
    try:
        asyncio.run(main(sglang_port = sglang_port))
    finally:
        terminate_process(server_process)