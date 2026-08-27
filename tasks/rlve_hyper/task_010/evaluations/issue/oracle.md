# Canonical issue oracle

Reconciled from the task's frozen evidence and canonical verifier record.

Expected verdict: **unhealthy**.

## Root cause

- `--eval-max-response-len`
  - kind: `hparam`
  - observed: 20
  - expected: '>= 1024'

## Observable impact

- train raw_reward -0.552 → +0.388; training side looks healthy
- eval stays at -1.000 throughout (no movement)
- train and eval diverge

## Trigger conditions

- The launch overrides `--eval-max-response-len` to `20`; a healthy eval length is at least `1024`.

## Documentation reconciliation

- Training improvement is expected. The held-out collapse is from clipping eval responses at length 20.
- Naming a target of `1024` or any larger eval response length is a correct expected value.
