# Issue verifier

`test.sh` grades `$PTDBENCH_RESULTS/agent_answer.md` against hidden
`oracle.md` with an LLM judge. It does not read `submission.json`.

The judge scores verdict, root-cause, and evidence agreement (weights
0.2 / 0.6 / 0.2). Pass is a weighted score of at least `0.8`. The gold
verdict for this task is `unhealthy`; the root cause is the observation
loss mask in `generate_with_search.py::generate`.

`gold.yaml` is retained as a reviewer-facing reference. Machine grading
reads `oracle.md`, not `gold.yaml`.

Requires `PTDBENCH_JUDGE_MODEL`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY`.
