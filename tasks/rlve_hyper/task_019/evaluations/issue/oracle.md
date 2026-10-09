# Canonical issue oracle

Reconciled from the task's frozen evidence and canonical verifier record.

Expected verdict: **unhealthy**.

## Root cause

The single essential root cause is `--lr-warmup-iters`. With only 100
training steps and `--lr-decay-style constant`, the schedule stays inside
the warmup ramp for the entire run, so `train/lr-pg_0` ends at 1e-7 instead
of the baseline 2e-6. Naming `--lr-warmup-iters` with its causal role earns
full `root_cause_score`.

- `--lr-warmup-iters` (essential)
  - kind: `hparam`
  - observed: 2000
  - expected: 0, or any small warmup the judge considers reasonable

`--lr-decay-iters` is a non-causal decoy. Under `--lr-decay-style constant`
the Megatron scheduler returns `max_lr` immediately after warmup and never
consults `lr_decay_iters`, so this override has no effect at any step count.
It is a legitimate cleanup (the baseline does not set it) but must NOT be
required: omitting it does not reduce `root_cause_score`, and an answer that
explicitly argues it is inert should be credited, not penalized.

- `--lr-decay-iters` (non-essential / inert under constant decay)
  - kind: `hparam`
  - observed: 10000
  - expected: unset (baseline does not override this); optional cleanup only

## Observable impact

- train/lr-pg_0 ends at only 1e-7; never reaches baseline 2e-6
- raw_reward -0.599 → -0.531 (+0.068)
- eval -0.620 → -0.522; improvement below healthy p10

## Trigger conditions

- `--lr-warmup-iters` is set to `2000`; the baseline value is `0`. Over a
  100-step budget this keeps the learning rate pinned inside the warmup ramp.
- `--lr-decay-iters` is overridden to `10000`; baseline does not set this flag,
  but under `--lr-decay-style constant` it is inert and does not affect the run.

## Documentation reconciliation

- `--lr-warmup-iters=2000` alone keeps the schedule from reaching the baseline `2e-6`.
- `--lr-decay-iters` is inert under constant decay and is not part of the causal chain.
- Weak reward/eval movement is a symptom of the truncated learning-rate schedule.
- The judge may accept any expected values it considers reasonable for restoring a schedule that can reach the baseline learning rate; the listed targets are references, not the only correct numbers.
