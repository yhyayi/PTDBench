# Metric-only solution verifier

`test.sh` runs `tests/grade_submission.py`, which reads `resources/grader_spec.json`.
It grades the fresh run's medal rate from `$PTDBENCH_RESULTS/experiment_logs`;
the graded value must be finite at the final step.

## System rerun (`run_spec.json`)

The solution protocol is debug-then-report: the agent fixes the workspace, submits `/workspace/report.md`, and does **not** launch training. This directory's `run_spec.json` drives the harness rerun stage — the command, environment (with `$PTDBENCH_*` references expanded per backend), timeout, and log roots — executed in the agent's workspace before this verifier runs. The harness deletes the declared log roots (`experiment_logs`) before the rerun, so grading only ever sees this fresh run's logs.
