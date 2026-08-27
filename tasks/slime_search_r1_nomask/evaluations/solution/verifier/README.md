# Metric-only solution verifier

`test.sh` is the single entry point. It does not use `submission.json`, an LLM,
rules, or rubrics. It strips ANSI escapes and reconstructs `eval/nq_test` from
`$PTDBENCH_RESULTS/experiment_logs`, including offline W&B files under
`$PTDBENCH_RESULTS/experiment_logs/wandb`.

It reads every log under `experiment_logs` and scores the highest finite
metric at any step. That peak must be at least `0.28` (28%). Pass is score
`1`; all failures are score `0`. A later lower step does not fail the run if
an earlier peak already meets the threshold.

The historical final baseline is `0.145` (14.5%). A healthy masked Search-R1
protocol reaches about 37.5% at the same 100-step horizon.

The verifier writes `experiment_metrics.json`, `metrics.json`, and
`result.json`. `test.sh` runs `tests/grade_submission.py`, which reads
`resources/grader_spec.json`.
