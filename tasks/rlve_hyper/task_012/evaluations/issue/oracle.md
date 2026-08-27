# Canonical issue oracle

Reconciled from the task's frozen evidence and canonical verifier record.

Expected verdict: **unhealthy**.

## Root cause

- `--hidden-dropout`
  - kind: `hparam`
  - observed: 0.5
  - expected: 0

## Observable impact

- raw_reward -0.604 → -0.604; completely stuck
- train/entropy_loss 6.00 → 6.26; far above healthy upper bound 1.40
- train/ppo_kl 12.3 (healthy runs ≈ 0)
- train/grad_norm 264452 → 727

## Trigger conditions

- The launch sets `--hidden-dropout` to `0.5`; the baseline value is `0`.

## Documentation reconciliation

- Extreme KL and grad_norm are symptoms of hidden dropout 0.5, not separate root causes.
