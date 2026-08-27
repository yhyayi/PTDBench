# Canonical issue oracle

Reconciled from the task's frozen evidence and canonical verifier record.

Expected verdict: **unhealthy**.

## Root cause

- `--eval-temperature`
  - kind: `hparam`
  - observed: 2
  - expected: unset (baseline does not override this)

## Observable impact

- train rollout/raw_reward -0.554 → +0.408 (+0.96); training side looks healthy
- eval starts at -1.000 (healthy runs -0.62 ± 0.03), ends at -0.791
- train vs eval metrics diverge; signature of an eval-side fault

## Trigger conditions

- The launch overrides `--eval-temperature` to `2`; baseline does not set this flag.

## Documentation reconciliation

- Training-side improvement is not evidence of held-out health. The defect is the eval-temperature override, not the train recipe.
