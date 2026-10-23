# First-update solution verifier

`test.sh` reads fresh logs under `$PTDBENCH_RESULTS/experiment_logs` only.
It does not grade TACO accuracy.

Pass (`score=1`) requires all of the following in the resolved run. Anything
else is `score=0`:

- every `max_response_length` is `30720` (any other value is an immediate 0)
- the log does not contain a CUDA out-of-memory error
- an actor update appears (`step:` with an actor or reward metric)

The verifier writes `metrics.json` and `result.json`. `test.sh` runs
`tests/grade_submission.py`, which reads `resources/grader_spec.json`.

## System rerun (`run_spec.json`)

The solution protocol is debug-then-report: the agent fixes the workspace, submits `/workspace/report.md`, and does **not** launch training. This directory's `run_spec.json` drives the harness rerun stage — the command, environment (with `$PTDBENCH_*` references expanded per backend), timeout, and log roots — executed in the agent's workspace before this verifier runs. The harness deletes the declared log roots (`experiment_logs`) before the rerun, so grading only ever sees this fresh run's logs.
