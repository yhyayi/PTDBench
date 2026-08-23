# Metric-only solution verifier

`test.sh` extracts the task metric only from fresh `$PTDBENCH_RESULTS/experiment_logs`, including offline W&B files under `$PTDBENCH_RESULTS/experiment_logs/wandb`. It does not grade `submission.json`, code rules, rubrics, or an LLM judgment.

It reads every log under `experiment_logs` and scores the highest finite `val-core/openai/gsm8k/acc/mean@1` at any step. That peak must be at least `0.4` (40%). Pass is score `1`; all failures are score `0`. A later lower step does not fail the run if an earlier peak already meets the threshold.

The verifier writes `experiment_metrics.json`, `metrics.json`, and `result.json`. `test.sh` runs `tests/grade_submission.py`, which reads `resources/grader_spec.json`.
