# Issue verification

The standard `test.sh` invokes the unchanged shared LLM judge. It reads
`$PTDBENCH_RESULTS/agent_answer.md` and the hidden issue oracle, with
verdict/root-cause/evidence weights 0.2/0.6/0.2; pass requires verdict_score 1.0 with root_cause_score and evidence_score each above 0.8.
The judge uses `PTDBENCH_JUDGE_MODEL`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY`,
and writes `$PTDBENCH_RESULTS/result.json`.
