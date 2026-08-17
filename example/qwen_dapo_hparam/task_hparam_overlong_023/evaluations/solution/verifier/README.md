# Solution verifier

This self-contained verifier grades
`$PTDBENCH_RESULTS/submission.json` against `gold.yaml`. It applies the same
issue gate, rule-first diagnosis/fix matching, LLM fallback prompts, and
combined score formula as `evaluation/script/grade_cases.py`.

The output is written to `$PTDBENCH_RESULTS/result.json` and
`$PTDBENCH_RESULTS/metrics.json`. A correct issue opens solution grading; the
combined score is `issue_score * (0.5 + 0.5 * solution_score)`. Statuses are
`correct`, `partial`, `wrong`, or `unavailable`.

The default issue gate requires `correct`. `test.sh --solution-on-partial`
matches the evaluation grader's optional relaxed gate.

LLM fallback uses `PTDBENCH_JUDGE_PROVIDER`, `PTDBENCH_JUDGE_MODEL`,
`PTDBENCH_JUDGE_BASE_URL`, and the environment-variable name in
`PTDBENCH_JUDGE_API_KEY_ENV`. Set `PTDBENCH_NO_JUDGE=1` only for deterministic
parity tests.
