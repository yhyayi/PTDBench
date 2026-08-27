# Canonical issue oracle

Reconciled from the task's frozen evidence and canonical verifier record.

Expected verdict: **unhealthy**.

## Root cause

- `--lr`
  - kind: `hparam`
  - observed: 1e-08
  - expected: 2e-06, or any larger learning rate the judge considers reasonable

## Observable impact

- within 245 steps raw_reward -0.617 → -0.670
- eval -0.611 → -0.573; improvement far below healthy p10 of +0.21
- entropy is flat

## Trigger conditions

- The launch sets `--lr` to `1e-08`; the baseline learning rate is `2e-06`.

## Documentation reconciliation

- Flat entropy and weak eval movement are symptoms of an undersized learning rate, not a second cause.
- The reference expected value is `2e-06`. The judge may accept any larger learning rate it considers reasonable for recovering a healthy schedule.
