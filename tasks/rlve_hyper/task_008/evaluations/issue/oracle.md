# Canonical issue oracle

Reconciled from the task's frozen evidence and canonical verifier record.

Expected verdict: **unhealthy**.

## Root cause

- `--lr`
  - kind: `hparam`
  - observed: 1e-09
  - expected: 2e-06, or any larger learning rate the judge considers reasonable

## Observable impact

- within 329 steps raw_reward -0.640 → -0.618 (+0.023)
- eval -0.563 → -0.613; no improvement
- entropy does not fall (0.76 → 0.81); grad_norm stays single-digit (normal)

## Trigger conditions

- The launch sets `--lr` to `1e-09`; the baseline learning rate is `2e-06`.

## Documentation reconciliation

- Single-digit grad_norm is normal here and is not a root cause. The unhealthy curve is from a learning rate three orders of magnitude too small.
- The reference expected value is `2e-06`. The judge may accept any larger learning rate it considers reasonable for recovering a healthy schedule.
