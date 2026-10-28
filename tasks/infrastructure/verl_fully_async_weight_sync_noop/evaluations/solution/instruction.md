# Repair the staged experiment

Inspect the source, launch script, effective configuration, and frozen
evidence in `/workspace`. Identify and repair the implementation defect that
connects the observed behavior to its cause. This is a debugging task, not a
training challenge: the defect is in the code, not in what the model learned.

The pinned model and VERL-formatted GSM8K files are prepared under
`/workspace/assets/`. Keep the dataset, held-out set, rollout count, response
limit, learning rate, and the other recorded hyperparameters as staged. In
particular, keep the trainer and the rollout engine in **separate** Ray
resource pools (`actor_rollout_ref.hybrid_engine=False`): colocating them
changes the topology the run is recorded under, and a run that does so is not a
repair of this defect.

**Do not run the experiment yourself — do not run `/workspace/run.sh` in the
foreground or background, and do not use `nohup`, `&`, `tmux`, `screen`, or
any detach mechanism.** The system runs exactly one evaluation itself after
you finish; an experiment you start is wasted and scores nothing. Cheap,
non-training checks are fine.

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
/workspace/run.sh
```

The run must complete 1632 rollout steps — 101 trainer steps — with held-out
evaluation before training and then every 20 parameter versions, writing fresh
logs below `$PTDBENCH_RESULTS/experiment_logs`. A task-specific grading script
then reads **only this fresh run's** files under
`$PTDBENCH_RESULTS/experiment_logs` and scores
`val-core/openai/gsm8k/acc/mean@4`; logs written anywhere else are ignored. Do
not copy, edit, or fabricate metric values. Otherwise the run scores 0.
