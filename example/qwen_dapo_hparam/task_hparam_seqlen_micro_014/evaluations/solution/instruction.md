# RL training run repair

Repair the staged post-training experiment and produce a new run that is
healthy under the fixed protocol. Inspect the source, effective configuration,
launch scripts, data processing, reward logic, logs, tracked metrics, and frozen
historical evidence. The historical run is unhealthy; use it only to identify
the defect and the strongest failure signals.

The most visible bad-run signals are the held-out validation collapse and the
short response budget: the frozen run ends near `0.015` on
`val-core/math_dapo/acc/mean@1`, while the same protocol's healthy 200-step
baseline is about `0.06`; rollout responses are clipped at length `512` with a
high `response_length/clip_ratio`. A correct solution fixes the underlying
configuration and demonstrates that the repaired run no longer exhibits those
signals.

Run the repaired post-training job using the task's fixed dataset, held-out set,
model, evaluation interval, and 200-step training budget. Do not use the
network. Do not modify the historical snapshot, oracle, verifier, datasets,
sealed baseline files, reward stack, evaluation set, model, or training-step
budget. The solution-level deliverable is the repaired run and its artifacts,
not a prose diagnosis.

## Deliverable

Write exactly one UTF-8 JSON file to:

```text
$PTDBENCH_RESULTS/submission.json
```

It must satisfy this shape. Replace every angle-bracket placeholder; for a
successful repair, each path should point to the artifacts from the repaired run
you executed. Paths may be absolute or relative to the workspace.

```json
{
  "run_dir": "<directory containing the repaired run artifacts>",
  "metrics_path": "<JSON metrics file from the repaired run>",
  "receipt_path": "<run receipt JSON showing the repaired job completed>",
  "log_path": "<training log from the repaired run>",
  "command_path": "<shell command used to launch the repaired run>",
  "changed_files": ["<workspace file changed to repair the run>"]
}
```

The verifier checks that the run completed, the effective sequence-length
configuration is repaired, and the repaired run's validation/truncation signals
are consistent with a healthy post-training run. Do not write prose outside
`submission.json`.
