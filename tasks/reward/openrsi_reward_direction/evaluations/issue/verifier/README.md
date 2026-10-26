# Issue verifier

This verifier grades `$PTDBENCH_RESULTS/agent_answer.md` against hidden
`evaluations/issue/oracle.md` with an LLM judge (`judge.py` +
`judge_prompt.md`).

The judge scores verdict, root-cause, and evidence agreement (weights
0.2 / 0.6 / 0.2). Pass requires `verdict_score` == 1.0, and `root_cause_score` and `evidence_score` each > 0.8; the weighted total alone is not a pass condition. The gold verdict
for this task is `unhealthy`.

`judge.py` and `judge_prompt.md` are reused byte-for-byte from
`tasks/slime_search_r1` (judge.py sha256
`25592980596dc314e2f8aa20ea1e85a3e11aa7e1a0c2628d12556127470463c3`), so the
grading behaviour is identical across the family and only `oracle.md` carries
the task's answer.

`gold.yaml` is retained as a reviewer-facing reference. Machine grading reads
`oracle.md`, not `gold.yaml`.

Requires `PTDBENCH_JUDGE_MODEL`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY`.
The output is written to `$PTDBENCH_RESULTS/result.json`.
