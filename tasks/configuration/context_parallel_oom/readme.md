# verl_implementation diagnosis: context_parallel_oom

## Case overview

- Evaluation key: `verl_implementation/context_parallel_oom`
- Case family: `verl_implementation`
- Schema version: `2`
- Evaluation levels: `issue`, `solution`
- Model: `Qwen2.5-1.5B-Instruct`
- Training framework: `verl`
- Hardware the failure is specified for: 8× H800 80GB

This document is for maintainers and reviewers. It is not exposed to the
agent during evaluation.

## Canonical failure reason

The launch script is a healthy FSDP/GRPO stack with one bad parallel setting.
Context parallel is `actor_rollout_ref.actor.ulysses_sequence_parallel_size`
(and the same field on the ref). This FSDP script has no separate `cp` flag;
that field is cp. It defaults to **1**. Combined with:

- `data.max_response_length=30720` (prompt 2048, so a full sample is 32768,
  equal to `max_position_embeddings`)
- `actor_rollout_ref.model.enable_gradient_checkpointing=False`
- EOS is honored (`ignore_eos` stays at its default false). A repetitive
  generation can still grow a response to 30720; that is when the long
  sequence appears, not necessarily on step 0
- `use_remove_padding=True`, so activation memory follows the real length
- `rollout.gpu_memory_utilization=0.4` (unchanged; rollout is not the OOM)

each H800 holds an entire 32768-token sequence during the actor
forward/backward once a response actually reaches the cap. Eight cards only
shard weights. The activation for one sequence without checkpointing is on
the order of 60–80GB, so that update OOMs.
cp=8 is illegal: Qwen2.5-1.5B has 12 attention heads, and
`verl/models/transformers/monkey_patch.py` requires the head count to be
divisible by the Ulysses size, which must also divide 8 GPUs. The legal sizes
are 1, 2, and 4. The repair is **cp=4** on both actor and ref.

## Workspace and hidden evaluations

| Path | Visibility and purpose |
| --- | --- |
| `workspace/` | Agent-visible snapshot. The defect is the launch script, not the model code. |
| `evaluations/issue/` | Diagnosis instruction, oracle, and LLM judge. |
| `evaluations/solution/` | Repair-and-run instruction and a log grader. |

## Evaluation design

`issue` grades `$PTDBENCH_RESULTS/agent_answer.md` against hidden `oracle.md`
(weights 0.2 / 0.6 / 0.2, pass at 0.8). The judge may accept any causally
reasonable account of the OOM. A diagnosis whose cause or fix is changing
`max_response_length`, `max_prompt_length`, `ignore_eos`,
`gpu_memory_utilization`, or `use_remove_padding` scores 0 on that dimension.

`solution` does not grade TACO accuracy. The grader reads only
`$PTDBENCH_RESULTS/experiment_logs`. Pass requires all of the following:

- every resolved `max_response_length` is `30720`
- the log has no CUDA OOM
- an actor update is present (`step:` with an actor or reward metric)

## Frozen launch settings

| Setting | Broken value | Required repair |
| --- | --- | --- |
| `data.max_prompt_length` | 2048 | unchanged |
| `data.max_response_length` | 30720 | unchanged; changing it scores 0 |
| `enable_gradient_checkpointing` | False | not graded |
| `actor` / `ref` `ulysses_sequence_parallel_size` | 1 | not graded |
| `rollout.gpu_memory_utilization` | 0.4 | unchanged |
| `rollout.ignore_eos` | false (default) | stays off; repetition can still fill 30720 |
| `data.train_batch_size` | 16 | unchanged |
| `ppo_micro_batch_size_per_gpu` | 1 | unchanged |
| `trainer.total_training_steps` | 100 | the launch budget |

## External dataset asset

Same bytes and revision as `torch_functional`. See `environment/prepare.sh`.

- Repository: `LIF1014/ptdbench-verl-implementation-torch-functional-dataset`
- Pinned revision: `83da7ccf4a74543f5aa3e46191c4916d7029564b`

| Runtime path | Size | SHA-256 |
| --- | --- | --- |
| `workspace/taco_verl/test.parquet` | 61,082,676 bytes | `e364fa69d973f8e8644b2fa4d7dc6ec405d5b4adff64002c6ad52e7c787c6eaa` |
| `workspace/taco_verl/train.parquet` | 606,016,648 bytes | `7f25abdfa9e0f5baa173d153a32973fbe2cc8736503a35408b8ebb8ae2bc5706` |

## External model asset

- Repository: `LIF1014/ptdbench-verl-implementation-torch-functional`
- Pinned revision: `36a4c226ef70aef2c38df173a7cf86a47ad3ad00`
- `model.safetensors` SHA-256 `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`

## Environment requirement

- Base image: `verlai/verl:vllm017.latest` (`environment.image: verl`)
- 8 GPUs, `shm_size` 16GB via `environment/docker-compose.yaml`
- Launch script: `examples/grpo_trainer/run_qwen2_5_1.5b_instruct_taco_fsdp.sh`
