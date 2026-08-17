# Issue verifier

This self-contained verifier grades
`$PTDBENCH_RESULTS/submission.json` against `gold.yaml`. Its rule-first
diagnosis logic and LLM fallback prompt mirror
`evaluation/script/grade_cases.py` and `evaluation/judge_issue.md`.

The output is written to `$PTDBENCH_RESULTS/result.json` and
`$PTDBENCH_RESULTS/metrics.json`. Statuses are `correct`, `partial`, `wrong`,
or `unavailable`; the issue score is respectively `1`, `0.5`, `0`, or `null`.

LLM fallback uses `PTDBENCH_JUDGE_PROVIDER`, `PTDBENCH_JUDGE_MODEL`,
`PTDBENCH_JUDGE_BASE_URL`, and the environment-variable name in
`PTDBENCH_JUDGE_API_KEY_ENV`. Set `PTDBENCH_NO_JUDGE=1` only for deterministic
parity tests.
