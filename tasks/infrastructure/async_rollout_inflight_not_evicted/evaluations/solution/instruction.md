# Improve the staged experiment

Review the staged post-training experiment and determine whether it is healthy
or unhealthy. Treat the workspace as a snapshot you may repair: inspect its
source, effective configuration, launch scripts, data processing, reward logic,
logs, tracked metrics, and frozen evaluation evidence. Do not assume that an
unusual setting or implementation is causal without connecting it to observed
behavior.

Use the rollout, throughput, recycling, truncation, runtime, optimization, and
training signals together. This is a fully-async run trained against a fixed
wall-clock budget rather than a fixed step count, so how much useful work the
engine completes per unit time is itself part of health.

If you find a defect, fix it in the staged workspace at `/workspace`. Then run
the experiment as a foreground process and wait until it returns. Do not use
the network. Do not use `nohup`, `&`, `tmux`, `screen`, or any detach
mechanism. The dataset, model, rollout configuration, and wall-clock training
budget are fixed.

The launch script is:

```text
scripts/train_grpo.sh
```

The effective run must use:

- model `/workspace/Qwen3-1.7B`
- Megatron torch_dist reference `/workspace/Qwen3-1.7B_torch_dist`
- train data `/workspace/data/dapo_math_17k_dedup.jsonl`
- `TRAIN_SECONDS=28800` (the fixed 8-hour wall-clock budget; do not shorten it)
- `LOG_DIR=$PTDBENCH_RESULTS/experiment_logs`

Create `$PTDBENCH_RESULTS/experiment_logs` first; the launch script writes the
driver log and the per-step rollout dumps (`dump/rollout_data/*.pt`) under
`LOG_DIR`. The run is scored only from files under
`$PTDBENCH_RESULTS/experiment_logs`; logs written anywhere else are ignored. Do
not copy, edit, or fabricate log metrics or dumps. Otherwise the run is scored 0.

Scoring reads two signals from those logs: the number of training steps
completed in the fixed wall clock (primary), and the last-window mean solve rate
over the dumped rollouts (secondary). Both must clear their thresholds.
