# Solution verifier

This self-contained verifier grades `$PTDBENCH_RESULTS/experiment_logs`
against `resources/grader_spec.json`. It does not use `submission.json`, an
LLM, rules, or rubrics.

The output is written to `$PTDBENCH_RESULTS/result.json`,
`$PTDBENCH_RESULTS/metrics.json`, and
`$PTDBENCH_RESULTS/experiment_metrics.json`.

Scoring is weighted:

- validation: highest finite `eval/HELD-OUT_ENVIRONMENTS_128` at least `-0.24`
- learning rate: mean `train/lr-pg_0` at least `8e-7`
- final `score = 0.8 * validation + 0.2 * learning rate`

Pass requires both components (`score == 1.0`).
