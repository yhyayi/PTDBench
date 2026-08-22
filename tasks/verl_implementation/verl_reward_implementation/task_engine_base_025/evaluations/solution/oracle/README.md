# Canonical solution oracle

## Root cause

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

- `verl/workers/engine/base.py` is observed as 26 methods across BaseEngine / BaseEngineCtx / EngineRegistry were rewritten; the canonical expectation is engine abstraction/registry logic consistent with baseline semantics.
- `verl/utils/reward_score/gsm8k.py::compute_score` is observed as default GSM8K scoring extracts a literal `#### N` only; the canonical expectation is keep `####` extraction and also accept Qwen2.5-0.5B-Instruct's preferred `\boxed{N}` / last-number format.

## Required remediation

- `verl/workers/engine/base.py`: `restore`; target state/value: baseline BaseEngine, BaseEngineCtx, and EngineRegistry semantics.
- `verl/utils/reward_score/gsm8k.py::compute_score`: `align_format_contract`; target state/value: accept `#### N` and fall back to boxed / last numeric answer.

Machine grading reads only the new experiment logs. The required held-out
metric is `val-core/openai/gsm8k/acc/mean@1` at step 100. The historical final
baseline is `0.0`. The absolute pass line is `0.4` (40%).
Pass is score `1`; otherwise `0`.
