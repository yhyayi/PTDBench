# Solution verifier

This self-contained verifier grades `$PTDBENCH_RESULTS/submission.json` against
`gold.yaml`. It applies the same issue gate, rule-first diagnosis, LLM fallback
prompts, and combined-score formula as the shared PTDBench grading primitives.

The gold verdict for this task is `healthy`. When the issue stage is `correct`
(submission `verdict` is `healthy`), the solution stage is `n/a` and the
combined score is `1.0`: no remediation is required for a healthy run. The
`fix.must` list is empty and `judge_solution.md` is not exercised for this
task.

The output is written to `$PTDBENCH_RESULTS/result.json` and
`$PTDBENCH_RESULTS/metrics.json`. Statuses are `correct`, `partial`, `wrong`,
or `unavailable`.

The default issue gate requires `correct`. `test.sh --solution-on-partial`
matches the evaluation grader's optional relaxed gate.

The verifier does not rerun training. Frozen historical evidence through
train step 100 lives in `workspace/outputs/` and
`evaluations/solution/oracle/historical_evidence_manifest.json`.

LLM fallback uses `PTDBENCH_JUDGE_PROVIDER`, `PTDBENCH_JUDGE_MODEL`,
`PTDBENCH_JUDGE_BASE_URL`, and the environment-variable name in
`PTDBENCH_JUDGE_API_KEY_ENV`. Set `PTDBENCH_NO_JUDGE=1` for deterministic
parity tests.
