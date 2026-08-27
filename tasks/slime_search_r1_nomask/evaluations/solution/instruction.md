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
examples/search-r1/run_qwen2.5_3B.sh
```

The effective run must use:

- HF checkpoint `/workspace/Qwen2.5-3B`
- Megatron torch_dist reference `/workspace/Qwen2.5-3B_torch_dist`
- train data `/workspace/nq_hotpotqa_train/train.parquet`
- validation data `/workspace/nq_hotpotqa_train/test.parquet`
- `--num-rollout 100`
- `--eval-interval 10`
- local retrieval server `http://127.0.0.1:8000/retrieve`
- `LOG_DIR=$PTDBENCH_RESULTS/experiment_logs`
- `WANDB_MODE=offline`
- `WANDB_DIR=$PTDBENCH_RESULTS/experiment_logs`

Create `$PTDBENCH_RESULTS/experiment_logs` first. Write the training and
evaluation log into that directory. Offline W&B data must be created below
`$PTDBENCH_RESULTS/experiment_logs/wandb`. If the launch script requires a W&B
API key even in offline mode, provide a harmless local placeholder. The run is
scored only from files under `$PTDBENCH_RESULTS/experiment_logs`; logs written
anywhere else are ignored. Do not copy, edit, or fabricate log metrics.
Otherwise the run is scored 0.
