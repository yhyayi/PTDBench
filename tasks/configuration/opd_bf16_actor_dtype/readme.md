# OPD actor stalls with bf16 master weights

Training DeepSeek-R1-Distill-Qwen-1.5B with sampled-token on-policy distillation on DAPO-Math-17k. The
run is finite and never crashes, but the actor barely learns because `MODEL_DTYPE=bf16` puts the actor's
FSDP master weights and AdamW update in bf16, so a `lr=1e-6` step rounds away for ~99% of the weights.

| | |
| --- | --- |
| **Author** | Yaxuan Li |
| **Type** | configuration |
| **Model** | DeepSeek-R1-Distill-Qwen-1.5B (student) · JustRL-DeepSeek-1.5B (teacher) |
| **Dataset** | DAPO-Math-17k (17,917 prompts) · AIME24 / AIME25 / AMC23 validation (loaded only) |
| **Framework** | verl, on-policy distillation (sampled-token, `token_reward_direct`), `on_policy_distillation.sh` |
| **Environment** | `opd-verl-uv` (not yet built; CUDA 12.8 + uv base, stack built into a uv venv by `prepare.sh`) · 2 × A800 80GB |
| **Evaluations** | issue · solution |
| **Recorded runs** | 1 defective reproduction (270 steps, ~18.5 h); threshold calibrated on 2 defective + 2 reference verifier runs (1 reference held out) |

## Difficulty

`on_policy_distillation.sh` passes `MODEL_DTYPE=${MODEL_DTYPE:-bf16}` as
`actor_rollout_ref.actor.fsdp_config.model_dtype=bf16`. In
`verl/verl/workers/fsdp_workers.py::ActorRolloutRefWorker._build_model_optimizer` the actor is loaded
and cast to that dtype, wrapped in FSDP, and the optimizer is built on its parameters. FSDP's
`MixedPrecision(param_dtype=bf16, reduce_dtype=fp32)` only sets compute and reduce dtypes, so the flat
master parameters, the AdamW states, and the gradients handed to it are all bf16. With `lr=1e-6`, Adam's
per-element step (≈1e-6) is below half a bf16 ulp for any weight with `|w| >= 2^-11 ≈ 4.9e-4`, so the
update rounds back to the unchanged value; only ~0.8% of the weights (always the same tiny-magnitude
ones) ever move. The in-code comment above this block warns about exactly this.

Nothing fails loudly: forward/backward run in bf16 either way, gradients are valid, and there is no
NaN or crash. The single strongest tell that this is a master-weight/optimizer precision failure, not
"the model can't learn", is that **every logged `actor/grad_norm` is exactly representable in bf16** —
a fingerprint of bf16 params and gradients. An fp32-actor run of the same recipe reaches a reverse KL
of ≈0.02 and a grad norm of ≈0.1 by step ~200.

| steps | `-critic/advantages/mean` (reverse KL) | `actor/grad_norm` | train accuracy |
|---|---:|---:|---:|
| 1–10 | 0.285 | 1.95 | 0.13 |
| 101–150 | 0.268 | 1.92 | 0.23 |
| 191–230 | 0.257 | 1.40 | 0.16 |

## Reference solution

- **issue**: name the **actor's bf16 master weights / AdamW update** (not merely "bf16") as the mechanism
  that freezes the weights at `lr=1e-6`. The teacher/reward-model dtype, gradient underflow, the rollout
  dtype, the hardware, the `pg_loss` scale, and vague "bf16 is imprecise" / "lr too small" reports are
  listed as wrong attributions and must not pass.
- **solution**: keep the actor master weights (and so the AdamW update) in fp32 — e.g.
  `actor.fsdp_config.model_dtype=fp32`, an fp32 master copy in the optimizer, or stochastic rounding —
  without touching the recipe (lr, data, models, objective). `oracle/reference.patch` sets the
  `MODEL_DTYPE` default back to `fp32`.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on verdict / root cause /
  evidence, weighted 0.2 / 0.6 / 0.2. Passing requires ≥ 0.8.
- **solution**: a deterministic `verifier/tests/check_run.py` reruns 15 steps (batch 16 × 4, 3072
  response tokens) through the repaired workspace with the recipe pinned, then requires: (1) all 15 steps
  logged; (2) at least 2.5% of the actor's bf16-view weights changed vs the initial checkpoint (SHA-256
  checked); (3) the mean advantage over steps 8–15 at least halfway from the buggy run to the reference
  repair, both recorded in `oracle/metrics.json`. Calibrated on 2 defective + 2 reference runs (one
  reference held out, plus a teacher-only-fix negative). Margins: bf16-view weights changed 0.77%
  (defective) vs 5.05% (reference) against the 2.5% gate; mean advantage steps 8–15 ≈ −0.267 (defective)
  vs ≈ −0.250 (reference). Score is the fraction of checks passed.
