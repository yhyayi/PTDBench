# Canonical issue oracle

Reconciled from the task's frozen evidence and canonical verifier record.

Expected verdict: **unhealthy**.

## Root cause

Either listed cause is independently sufficient. Naming one with the
correct causal role is a complete attribution; naming both is also
correct. Do not require both.

- `verl/workers/engine/base.py`
  - kind: `symbol`
  - observed: 26 methods across BaseEngine / BaseEngineCtx / EngineRegistry were rewritten
  - expected: engine abstraction/registry logic consistent with baseline semantics
- `verl/utils/reward_score/gsm8k.py::compute_score`
  - kind: `format_contract`
  - observed: default GSM8K scoring extracts a literal `#### N` only
  - expected: keep `####` extraction and also accept Qwen2.5-0.5B-Instruct's preferred `\boxed{N}` / last-number format

## Observable impact

- Primary source evaluation metric: `val-core/openai/gsm8k/acc/mean@1`.
- val curve 0.0008 → 0.0; baseline under same setup reaches 47.8%
- no reward rise within 100 steps
- pinned GSM8K prompts request a final `####` answer and the default scorer honors that nominal contract, but Qwen2.5-0.5B-Instruct preferentially emits `\boxed{N}`, so almost every rollout is scored 0

## Trigger conditions

- `base.py` rewrote 26 methods across BaseEngine / BaseEngineCtx / EngineRegistry.
- `compute_score` defaults to extracting a literal `#### N` only.

## Documentation reconciliation

- Pinned parquet inspection shows a nominal prompt/extractor match: train and test prompts ask to output the final answer after `####`, and `compute_score` defaults to `method="strict"`.
- The format defect is a data-format / model-convention gap: this Qwen checkpoint prefers `\boxed{N}` (and other non-`####` numeric endings), so the `####`-only contract yields sparse or zero GRPO signal and held-out strict accuracy collapses.
- Either listed root cause is a correct attribution. A diagnosis that names only one should receive full `root_cause_score` if the causal role is right.
