# Improve the staged experiment

Review the staged post-training experiment and determine whether it is healthy
or unhealthy. Treat the workspace as a snapshot you may repair: inspect its
source, effective configuration, launch scripts, data processing, reward logic,
logs, tracked metrics, and frozen evaluation evidence. Do not assume that an
unusual setting or implementation is causal without connecting it to observed
behavior.

Use held-out evaluation evidence together with rollout, truncation, runtime,
optimization, and training signals. Training reward alone does not establish
held-out health — a reward that rises while held-out accuracy falls is a warning,
not a success.

If you find a defect, fix it in the staged workspace at `/workspace`. Then run
the experiment as a foreground process and wait until it returns. Do not use
the network. Do not use `nohup`, `&`, `tmux`, `screen`, or any detach
mechanism. The dataset, held-out set, and model are fixed.

The launch script is:

```text
run_buggy.sh
```

The effective run must use:

```text
MODEL_PATH=/workspace/Qwen3-4B-Base
CUDA_VISIBLE_DEVICES=0,1,2,3
N_GPUS_PER_NODE=4
LOG_DIR=$PTDBENCH_RESULTS/experiment_logs
LOGGER=console
WANDB_MODE=offline
WANDB_DIR=$PTDBENCH_RESULTS/experiment_logs
```

Create `$PTDBENCH_RESULTS/experiment_logs` first. Write the training and
evaluation log into that directory. Offline W&B data must be created below
`$PTDBENCH_RESULTS/experiment_logs/wandb`. The run is scored only from files
under `$PTDBENCH_RESULTS/experiment_logs`; logs written anywhere else are
ignored. Do not copy, edit, or fabricate log metrics. Do not modify the
preserved evidence under `workspace/ptdbench_case/` or any evaluation file.
Otherwise the run is scored 0.
