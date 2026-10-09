# Canonical issue oracle

Reconciled from the task's frozen evidence and canonical verifier record.

Expected verdict: **the single-run AIME25 result is an evaluation artifact
(under-sampling), not sufficient evidence of a PPO training failure.**

## Root cause

- Evaluation protocol (`config/eval_config.yaml`, consumed by `run_eval.py`)
  - kind: `config`
  - observed: `evaluation.samples_per_question: 1` — a single stochastic
    generation per question on a 30-question benchmark, with temperature `1.0`
    and top-p `1.0`
  - expected: many samples per question (the controlled reference uses `32`),
    aggregated with the mean reducer, so the estimate is stable
  - note: AIME25 has only 30 questions; one stochastic sample per question makes
    a single 30-question realization a high-variance estimator.

## Observable impact

- Controlled repeated evaluation of the *same* checkpoint produced per-epoch
  AIME25 accuracy ranging from **6.67% to 20%**.
- Exactly **10%** appeared in **7 of 32** repeated epochs.
- A single low score (e.g. 10%) is therefore within the normal sampling spread
  of a healthy checkpoint, not a distinct failure signal.

## Trigger conditions

- `config/eval_config.yaml` sets `samples_per_question: 1`; generation is
  stochastic (`temperature: 1.0`, `top_p: 1.0`). One realization of 30
  questions is reported as if it were a stable estimate.

## Documentation reconciliation

- A single low AIME25 number is not proof that PPO training failed. This is
  evaluation under-sampling / high-variance estimation, not a reward defect, an
  actor-learning-rate collapse, a parser mismatch, or a startup failure. The
  remedy is to increase samples-per-question (not to change training).
