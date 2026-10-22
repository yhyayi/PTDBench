# RL training run diagnosis

Review the staged post-training experiment and determine whether it is healthy
or unhealthy. Treat the workspace as a historical snapshot: inspect its source,
effective configuration, launch scripts, data processing, reward logic, and
frozen evaluation evidence. Do not assume that an unusual setting or
implementation is causal without connecting it to observed behavior.

The designated hardware is 8× H800 (80GB). The launch script is:

```text
examples/grpo_trainer/run_qwen2_5_1.5b_instruct_taco_fsdp.sh
```

These settings are not the defect. Do not change them, and do not treat
changing them as the cause or the fix:

- `data.max_response_length` is **30720**. A diagnosis that says the run
  should use any other value is wrong.
- `data.max_prompt_length` stays **2048**. A full sample is 32768 tokens,
  this model's context limit.
- `actor_rollout_ref.rollout.ignore_eos` stays **False**. Generation stops
  at EOS. A response can still reach 30720 when the model repeats.
- `actor_rollout_ref.rollout.gpu_memory_utilization` stays **0.4**.
- `actor_rollout_ref.model.use_remove_padding` stays **True**.

The launch script trains for 100 steps.

Do not launch a full training run, use the network, or modify the historical
snapshot. The dataset, held-out set, model, and this response length are fixed.

## Deliverable

Write one concise UTF-8 diagnosis to:

```text
$PTDBENCH_RESULTS/agent_answer.md
```

State the verdict, identify the concrete root cause or causes, and explain the
causal chain using specific workspace evidence. Distinguish root causes from
symptoms. Do not propose or implement a fix, and do not write other deliverable
files.
