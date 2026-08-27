# Canonical issue oracle

Reconciled from the task's frozen evidence and canonical verifier record.

Expected verdict: **unhealthy**.

## Root cause

- `--attention-dropout`
  - kind: `hparam`
  - observed: 0.7
  - expected: 0

## Observable impact

- raw_reward -0.595 → -0.592; barely moves
- entropy 5.50 → 5.63; out of healthy range
- grad_norm 30094 → 1672
- eval −0.605 → −0.610

## Trigger conditions

- The launch sets `--attention-dropout` to `0.7`; the baseline value is `0`.

## Documentation reconciliation

- High entropy and large grad_norm are symptoms of the dropout setting, not separate root causes.
