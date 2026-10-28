# Issue verifier

This verifier grades `$PTDBENCH_RESULTS/agent_answer.md` against hidden
`evaluations/issue/oracle.md` with an LLM judge (`judge.py` +
`judge_prompt.md`).

The judge scores verdict, root-cause, and evidence agreement (weights
0.2 / 0.6 / 0.2). Pass requires `verdict_score` == 1.0, and `root_cause_score`
and `evidence_score` each > 0.8; the weighted total alone is not a pass
condition. The gold verdict for this task is `unhealthy`.

Two essential root causes: the missing `checkpoint_engine.backend` pin in
`verl/experimental/fully_async_policy/config/fully_async_ppo_trainer.yaml`,
which leaves the run on the colocated `"naive"` engine, and the
`if self.rollout is None: ... return` guard in
`ActorRolloutRefWorker.update_weights`
(`verl/workers/engine_workers.py`), which turns the dropped push into a no-op
instead of an error. Naming only one earns partial root-cause credit and must
not pass.

`judge.py` and `test.sh` are byte-identical to
`tasks/reward/retool_length_hack/evaluations/issue/verifier/`, as are
`instruction.md` and `judge_prompt.md`; `oracle.md` is the only per-task file.

Requires `PTDBENCH_JUDGE_MODEL`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY`.
The output is written to `$PTDBENCH_RESULTS/result.json`.
