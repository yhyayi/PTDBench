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
