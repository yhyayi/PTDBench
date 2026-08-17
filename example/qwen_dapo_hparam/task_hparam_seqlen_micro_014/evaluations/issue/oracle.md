# Canonical issue oracle

Reconciled from the task's frozen evidence and canonical verifier record.

Expected verdict: **unhealthy**.

## Root cause

- `MAX_RESPONSE_LENGTH`
  - kind: `hparam`
  - observed: 512
  - expected: 1024
- `MAX_PROMPT_LENGTH`
  - kind: `hparam`
  - observed: 512
  - expected: 1024

## Observable impact

- Primary source evaluation metric: `val-core/math_dapo/acc/mean@1`.
- response_length/clip_ratio ≈ 0.766
- prompt_length/max drops from control 745 to 469 while prompt_length/clip_ratio stays 0 (filtering, not truncation)
- val-core 0.017 → 0.015; about 1/2.3 of the same-cluster control
- healthy protocol baseline (Qwen2.5-0.5B-Instruct + DAPO-Math-17k): about 6.0% at 200 steps, 7.7% at 1000 steps

## Trigger conditions

- `MAX_RESPONSE_LENGTH` is observed as 512; the canonical expectation is 1024.
- `MAX_PROMPT_LENGTH` is observed as 512; the canonical expectation is 1024.

## Documentation reconciliation

- The three micro-batch fields preserved in hollow_hparams are distractors; canonical evidence identifies only the two sequence-length limits.
