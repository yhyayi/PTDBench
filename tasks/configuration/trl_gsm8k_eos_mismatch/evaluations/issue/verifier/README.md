# Issue verifier

The unmodified official judge compares
`$PTDBENCH_RESULTS/agent_answer.md` with the hidden `oracle.md`.
It grades verdict, root cause and evidence using the official rubric.
The gold verdict is unhealthy; the essential defect is rollout stopping that
does not recognize the chat template end-of-turn marker.

Requires `PTDBENCH_JUDGE_MODEL`, `OPENAI_BASE_URL` and `OPENAI_API_KEY`.
Results are written to `$PTDBENCH_RESULTS/result.json`.
