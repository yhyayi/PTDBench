# Issue verifier

This verifier grades `$PTDBENCH_RESULTS/agent_answer.md` against hidden
`evaluations/issue/oracle.md` with an LLM judge (`judge.py` +
`judge_prompt.md`).

The judge scores verdict, root-cause, and evidence agreement (weights
0.2 / 0.6 / 0.2). Pass is a weighted score of at least `0.8`. The gold
verdict for this task is `unhealthy`; the essential root cause is the
tool-schema name mismatch in `tool_schema/alfworld_case/tools.py`, where the
advertised schema renames `clean`/`heat`/`cool` to
`clean_object`/`heat_object`/`cool_object` while `REGISTRY` and `command_for`
still dispatch only the un-suffixed names, so those calls return `unknown_tool`.

`gold.yaml` is retained as a reviewer-facing reference. Machine grading
reads `oracle.md`, not `gold.yaml`.

Requires `PTDBENCH_JUDGE_MODEL`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY`.
The output is written to `$PTDBENCH_RESULTS/result.json`.
