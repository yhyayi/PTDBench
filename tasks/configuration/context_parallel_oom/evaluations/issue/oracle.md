# Canonical issue oracle

Expected verdict: **unhealthy**.

## Root cause

The training code is healthy. The launch script's context parallel is too
small for the sequence it actually trains.

- Launch script: `examples/grpo_trainer/run_qwen2_5_1.5b_instruct_taco_fsdp.sh`
  - kind: `config`
  - observed: `actor_rollout_ref.actor.ulysses_sequence_parallel_size=1` and
    `actor_rollout_ref.ref.ulysses_sequence_parallel_size=1`. On this FSDP
    script that field is context parallel (`cp`). There is no other `cp` flag.
  - expected: both set to `4`. `8` is not a legal value: Qwen2.5-1.5B has 12
    attention heads, and Ulysses requires the head count to be divisible by
    the parallel size, which must also divide the 8 GPUs. Legal sizes are
    1, 2, and 4.
  - accompanying settings that are **not** the bug and must stay:
    `data.max_response_length=30720`, `data.max_prompt_length=2048` (a full
    sample is 32768 tokens, the model's `max_position_embeddings`), `rollout.ignore_eos=False`,
    `rollout.gpu_memory_utilization=0.4`, `use_remove_padding=True`.

## Observable impact

- On 8× H800 80GB, an actor forward/backward CUDA-OOMs once a sample actually
  reaches the 32768-token cap. Early steps can be short and fit. 
- `cp=1` does not split the sequence. Each GPU then stores the activations of
  that full sample because checkpointing is off. FSDP shards weights only.
- vLLM at `gpu_memory_utilization=0.4` is not the failing stage.
- Shortening `max_response_length` would hide the OOM and is not a fix.

## What the judge should accept

The preferred mechanism is cp=1 (`ulysses_sequence_parallel_size=1`) on the
actor and the ref. The judge may also accept another causally reasonable
account of the same OOM. Changing any accompanying setting listed above
(`max_response_length`, `max_prompt_length`, `ignore_eos`,
`gpu_memory_utilization`, `use_remove_padding`) is not a reasonable cause
and is not a fix.

## Trigger conditions

- `ULYSSES_SEQUENCE_PARALLEL_SIZE` defaults to 1 and is passed to both the
  actor and the ref.
- EOS is honored (`ignore_eos` stays false). Packed length follows the real
  response, which can still reach the cap when generation repeats.
- `ENABLE_GRADIENT_CHECKPOINTING` defaults to False.
