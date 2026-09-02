# Metric-only solution verifier

`test.sh` is the single entry point. It does not use `submission.json`, an LLM,
rules, or rubrics. It strips ANSI escapes and reconstructs metrics from
`$PTDBENCH_RESULTS/experiment_logs`, including offline W&B files under
`$PTDBENCH_RESULTS/experiment_logs/wandb`.

It reads every log under `experiment_logs` and assigns a weighted score:

- validation: the highest finite `eval/HELD-OUT_ENVIRONMENTS_128` at any step
  must be at least `-0.24` to score `1`, otherwise `0`. A later lower step
  does not fail this component if an earlier peak already meets the threshold.
- learning rate: the mean `train/lr-pg_0` over training steps must be at
  least `8e-7` to score `1`, otherwise `0`
- final `score = 0.8 * validation + 0.2 * learning rate`

Pass requires both components (`score == 1.0`). The framework still records
the weighted float when only one component passes.

The verifier writes `experiment_metrics.json`, `metrics.json`, and
`result.json`. `test.sh` runs `tests/grade_submission.py`, which reads
`resources/grader_spec.json`.
