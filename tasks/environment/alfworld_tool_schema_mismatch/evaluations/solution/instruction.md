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
mechanism. The dataset, held-out set, model, training-step budget, and
evaluation interval are fixed.

The launch script is:

```text
tool_schema/alfworld_case/train.sh
```

It takes no positional arguments and reads four authorized, idle GPUs from
`N_GPUS=4` / `CUDA_VISIBLE_DEVICES`. The effective run must use:

- model `/workspace/tool_schema/Qwen3-4B`
- train data `/workspace/tool_schema/datasets/alfworld/difficulty-v1-s42/train.parquet`
- validation data `/workspace/tool_schema/datasets/alfworld/balanced1008-s42/validation.parquet`
- the fixed budget baked into `train.sh`: `trainer.total_training_steps=63`
  and `trainer.test_freq=20` (eval at steps 0, 20, 40, 60, 63)
- `RUN_DIR=$PTDBENCH_RESULTS/experiment_logs`
- `WANDB_MODE=offline`
- `WANDB_DIR=$PTDBENCH_RESULTS/experiment_logs`

Create `$PTDBENCH_RESULTS/experiment_logs` first. `train.sh` writes `train.log`,
`validation/`, `rollouts/`, and offline W&B under `RUN_DIR`, so point `RUN_DIR`
at `$PTDBENCH_RESULTS/experiment_logs`. The run is scored only from files under
`$PTDBENCH_RESULTS/experiment_logs`; logs written anywhere else are ignored. Do
not copy, edit, or fabricate log metrics. Otherwise the run is scored 0.
