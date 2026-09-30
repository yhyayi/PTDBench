# Metric-only solution verifier

`test.sh` is the single entry point. It does not use `submission.json`, an LLM,
rules, or rubrics. It strips ANSI escapes and reconstructs `eval/aime` from
`$PTDBENCH_RESULTS/experiment_logs`, including offline W&B files under
`$PTDBENCH_RESULTS/experiment_logs/wandb`.

The launch script logs `eval/aime` as exact-match accuracy via
`--eval-reward-key acc`. `score_mode` is `best`: the verifier takes the
highest finite `eval/aime` over the run rather than the value at step 69,
because a single eval point over 30 AIME problems is noisy. That value must
be at least `0.18` (18%) and at most `1.0`. Pass is score `1`; all failures
are score `0`. The historical decontaminated accuracy is about 8.4%; the
fixed recipe reaches about 28.8% at its best point (step 59) and about 25.7%
at the last eval (step 69), so it clears the bar comfortably.

The verifier writes `experiment_metrics.json`, `metrics.json`, and
`result.json`. `test.sh` runs `tests/grade_submission.py`, which reads
`resources/grader_spec.json`.
