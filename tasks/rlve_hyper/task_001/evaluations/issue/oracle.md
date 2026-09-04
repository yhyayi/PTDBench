# Canonical issue oracle

Reconciled from the task's frozen evidence and canonical verifier record.

Expected verdict: **unhealthy**.

## Root cause

- `--attention-dropout`
  - kind: `hparam`
  - observed: 0.5
  - expected: 0

## Observable impact

- rollout/raw_reward start→end -0.650 → -0.594 (+0.056); healthy runs worst-case still +0.36
- train/entropy_loss 4.04 → 3.52; healthy upper bound 1.40
- train/grad_norm 2258 → 19760; healthy runs stay ≤ 9.3
- eval -0.611 → -0.639 (moves the wrong way)

## Trigger conditions

- The launch overrides `--attention-dropout` to `0.5`; the baseline value is `0`.

## Documentation reconciliation

- High entropy and exploding grad_norm are symptoms of the dropout setting, not separate root causes.
