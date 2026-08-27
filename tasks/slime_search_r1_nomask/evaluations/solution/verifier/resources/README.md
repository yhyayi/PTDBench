`test.sh` is the single entry point. It does not use `submission.json`, an LLM,
rules, or rubrics. It strips ANSI escapes and reconstructs `eval/nq_test` from
`$PTDBENCH_RESULTS/experiment_logs`.
