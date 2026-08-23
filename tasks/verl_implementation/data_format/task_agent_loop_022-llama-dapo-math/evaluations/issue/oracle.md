# Canonical issue oracle

Reconciled from the task's frozen evidence and canonical verifier record.

Expected verdict: **unhealthy**.

## Root cause

Either listed cause is independently sufficient. Naming one with the
correct causal role is a complete attribution; naming both is also
correct. Do not require both.

- `verl/experimental/agent_loop/agent_loop.py`
  - kind: `symbol`
  - observed: rollout, reward, and non-tensor batch contracts were rewritten inconsistently
  - expected: baseline agent-loop assembly semantics
- `verl/utils/reward_score/math_dapo.py::is_correct_minerva`
  - kind: `format_contract`
  - observed: extracts Answer: only
  - expected: retain Answer: extraction and the healthy format stack's boxed-answer fallback

## Observable impact

- Primary source evaluation metric: `val-core/math_dapo/reward/mean@1`.
- six validation points rise from about 0.61% to 1.23% by step 100
- a healthy Llama-3.2-1B-Instruct + DAPO-Math-17k run reaches about 8.4% at 200 steps and 11.8% at 1000 steps; this case only ran 100 steps and is not reproduced at that level in the frozen workspace
- training critic/score is zero for most steps, and an earlier attempt fails with KeyError: multi_modal_inputs
- all 17,917 pinned parquet prompts explicitly request a final Answer: line; the workspace extractor honors that nominal contract but lacks the healthy format stack's boxed-answer fallback

## Trigger conditions

- `agent_loop.py` rollout, reward, and non-tensor batch contracts were rewritten inconsistently.
- `is_correct_minerva` extracts Answer: only and lacks the boxed-answer fallback.

## Documentation reconciliation

- Pinned parquet inspection disproves a literal prompt/extractor mismatch: every prompt requests Answer:, matching the workspace regex.
- The format defect is a robustness gap between actual model-output conventions and the healthy format stack; no frozen rollout sample records the boxed-output frequency.
- Either listed root cause is a correct attribution. A diagnosis that names only one should receive full `root_cause_score` if the causal role is right.
- The earlier KeyError: multi_modal_inputs is a symptom of a prior attempt, not a third scored cause.
