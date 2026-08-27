# Canonical issue oracle

Reconciled from the task's frozen evidence and canonical verifier record.

Expected verdict: **unhealthy**.

## Root cause

Both listed causes are essential. Identifying only one must receive
partial `root_cause_score`.

- `--lr-decay-iters`
  - kind: `hparam`
  - observed: 10000
  - expected: unset (baseline does not override this), or any value the judge considers reasonable
- `--lr-warmup-iters`
  - kind: `hparam`
  - observed: 2000
  - expected: 0, or any small warmup the judge considers reasonable

## Observable impact

- train/lr-pg_0 ends at only 1e-7; never reaches baseline 2e-6
- raw_reward -0.599 → -0.531 (+0.068)
- eval -0.620 → -0.522; improvement below healthy p10

## Trigger conditions

- `--lr-decay-iters` is overridden to `10000`; baseline does not set this flag.
- `--lr-warmup-iters` is set to `2000`; the baseline value is `0`.

## Documentation reconciliation

- The two flags jointly keep the schedule from reaching the baseline `2e-6`.
- Weak reward/eval movement is a symptom of the truncated learning-rate schedule.
- The judge may accept any expected values it considers reasonable for restoring a schedule that can reach the baseline learning rate; the listed targets are references, not the only correct numbers.
