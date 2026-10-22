# Improve the staged experiment

Review the staged post-training experiment and determine whether it is healthy
or unhealthy. Treat the workspace as a snapshot you may repair: inspect its
source, effective configuration, and launch script. Do not assume that an
unusual setting is causal without connecting it to memory behavior on the
designated hardware, 8× H800 (80GB).

If you find a defect, fix it in the staged workspace at `/workspace`. Then run
the experiment as a foreground process and wait until it returns. Do not use
the network. Do not use `nohup`, `&`, `tmux`, `screen`, or any detach
mechanism. The dataset, held-out set, and model are fixed.

The launch script is:

```text
examples/grpo_trainer/run_qwen2_5_1.5b_instruct_taco_fsdp.sh
```

The effective run must use:

```text
MODEL_PATH=/workspace/Qwen2.5-1.5B-Instruct
TRAIN_FILE=/workspace/taco_verl/train.parquet
TEST_FILE=/workspace/taco_verl/test.parquet
LOG_DIR=$PTDBENCH_RESULTS/experiment_logs
WANDB_MODE=offline
WANDB_DIR=$PTDBENCH_RESULTS/experiment_logs
WANDB_API_KEY=offline
```

These settings are mandatory and are not the repair:

- `data.max_response_length` must stay **30720**. You must not change it.
  Changing it, including via the environment or a Hydra override, scores 0.
  `data.max_prompt_length` stays 2048, so a full sample is 32768 tokens.
- `actor_rollout_ref.rollout.gpu_memory_utilization` stays **0.4**.
- Leave EOS handling on. Do not set `ignore_eos=True`.

The script trains for **100** steps. Run that budget. Do not shorten
`max_response_length` to make the 100 steps fit.

Create `$PTDBENCH_RESULTS/experiment_logs` first. Write the training log into
that directory. The run is scored only from files under
`$PTDBENCH_RESULTS/experiment_logs`; logs written anywhere else are ignored.
Do not copy, edit, or fabricate log metrics. Otherwise the run is scored 0.
