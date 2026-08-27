# Solution verifier

This self-contained verifier grades `$PTDBENCH_RESULTS/experiment_logs`
against `resources/grader_spec.json`. It does not use `submission.json`, an
LLM, rules, or rubrics.

The output is written to `$PTDBENCH_RESULTS/result.json`,
`$PTDBENCH_RESULTS/metrics.json`, and
`$PTDBENCH_RESULTS/experiment_metrics.json`. Pass is score `1` when the
highest finite `eval/HELD-OUT_ENVIRONMENTS_128` is at least `-0.24`;
otherwise score `0`.
