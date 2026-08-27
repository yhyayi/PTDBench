# Metric-only solution verifier

`test.sh` is the single entry point. It does not use `submission.json`, an LLM,
rules, or rubrics. It strips ANSI escapes and reconstructs `eval/aime` from
`$PTDBENCH_RESULTS/experiment_logs`, including offline W&B files under
`$PTDBENCH_RESULTS/experiment_logs/wandb`.

The launch script logs `eval/aime` as exact-match accuracy via
`--eval-reward-key acc`. It scores the finite `eval/aime` at step 99. That
value must be at least `0.18` (18%) and at most `1.0`. Pass is score `1`;
all failures are score `0`. The historical decontaminated accuracy is about
8.4%; a healthy protocol reaches about 25% at the same horizon.

The verifier writes `experiment_metrics.json`, `metrics.json`, and
`result.json`. `test.sh` runs `tests/grade_submission.py`, which reads
`resources/grader_spec.json`.
