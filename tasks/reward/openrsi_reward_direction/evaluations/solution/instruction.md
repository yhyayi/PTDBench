# Improve the staged experiment

Review the staged post-training experiment and determine whether it is healthy
or unhealthy. Treat the workspace as a snapshot you may repair: inspect its
source, effective configuration, launch scripts, data processing, reward logic,
logs, tracked metrics, and frozen evaluation evidence. Do not assume that an
unusual setting or implementation is causal without connecting it to observed
behavior.

The run trains one policy over a pool of machine-learning-engineering tasks in
a single rollout. Use held-out evaluation evidence together with rollout,
truncation, runtime, optimization, and training signals. Training reward alone
does not establish held-out health.

## Your job

If you find a defect, fix it in the staged workspace at `/workspace`. **Do not
launch training.** Do not start the experiment, in the foreground or
background, and do not use `nohup`, `&`, `tmux`, `screen`, or any detach
mechanism. The system runs the experiment itself after you finish (see below);
mechanism. The system runs the experiment itself after you finish (see below);
anything you launch is wasted and scores nothing. Cheap, non-training checks
are fine.

When you are done, submit a report at `/workspace/report.md`. It must
summarize:

1. **Root cause** — the defect you found, and the evidence in the workspace
   (logs, metrics, code) that connects it to the observed behavior.
2. **Changes** — every file you changed and what you changed in it.
3. **Expected effect** — which metric the fix should move and in which
   direction.

A missing or empty `/workspace/report.md` scores 0.

## What the system will run

After you submit the report, the system deletes everything under
`$PTDBENCH_RESULTS/experiment_logs`, then re-runs the experiment in your
repaired workspace as a foreground process, using the launch script:

```text
examples/openmle_rl/scripts/run_openrsi_reward_direction.sh
```

with this environment:

```text
LOG_DIR=$PTDBENCH_RESULTS/experiment_logs
WANDB_MODE=offline
WANDB_DIR=$PTDBENCH_RESULTS/experiment_logs
```

The effective run uses the fixed budget baked into the task config
`examples/openmle_rl/configs/reward_direction.env`: `NUM_ROLLOUT=40` and
`EVAL_INTERVAL=5`. Offline W&B data will land below
`$PTDBENCH_RESULTS/experiment_logs/wandb`; if the launch script requires a
W&B API key even in offline mode, it gets a harmless local placeholder from the
environment. A task-specific grading script then reads **only this fresh
run's** files under `$PTDBENCH_RESULTS/experiment_logs` and scores the key
metrics; logs written anywhere else are ignored. Make sure your repaired
workspace still writes its training and evaluation logs into
`$PTDBENCH_RESULTS/experiment_logs`. Do not copy, edit, or fabricate log
metrics. Otherwise the run is scored 0.
