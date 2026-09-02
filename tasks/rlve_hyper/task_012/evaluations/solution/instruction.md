# Improve the staged experiment

Review the staged post-training experiment and determine whether it is healthy
or unhealthy. Treat the workspace as a snapshot you may repair: inspect its
source, effective configuration, launch scripts, data processing, reward logic,
logs, tracked metrics, and frozen evaluation evidence. Do not assume that an
unusual setting or implementation is causal without connecting it to observed
behavior.

Use held-out evaluation evidence together with rollout, truncation, runtime,
optimization, and training signals. Training reward alone does not establish
held-out health.

If you find a defect, fix it in the staged workspace at `/workspace`. Then run
the experiment as a foreground process and wait until it returns. Do not use
the network. Do not use `nohup`, `&`, `tmux`, `screen`, or any detach
mechanism. The dataset, held-out set, and model are fixed.

The launch script is:

```text
scripts/training/Qwen2.5-0.5B-Instruct/rlve.sh
```

It requires three positional arguments (`WANDB_PROJECT`, `RUN_NAME`, and a
dummy environment-list token). The locked environment list inside the script
is used; do not replace it. Example:

```text
bash scripts/training/Qwen2.5-0.5B-Instruct/rlve.sh ptdbench solution_rerun _
```

The effective run must use:

- model `/workspace/Qwen2.5-0.5B-Instruct`
- Megatron dist checkpoint `/workspace/Qwen2.5-0.5B-Instruct_torch_dist`
- held-out eval `data/HELD-OUT_ENVIRONMENTS/test_128.json`
- `LOG_DIR=$PTDBENCH_RESULTS/experiment_logs`
- `WANDB_MODE=offline`
- `WANDB_DIR=$PTDBENCH_RESULTS/experiment_logs`

Create `$PTDBENCH_RESULTS/experiment_logs` first. Offline W&B data must be
created below `$PTDBENCH_RESULTS/experiment_logs/wandb`. If the launch script
requires a W&B API key even in offline mode, provide a harmless local
placeholder. Do not copy, edit, or fabricate log metrics. Otherwise the run is
scored 0.
