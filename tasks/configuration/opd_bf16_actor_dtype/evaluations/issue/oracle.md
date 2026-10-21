# Canonical issue oracle

Reconciled from the staged workspace, the run's logs, controlled reproductions and the reference repair in
`evaluations/solution/oracle/reference.patch`.

Expected verdict: **unhealthy** (silent training failure: the run is finite and completes, but the actor
does not learn).

## Root cause

- `on_policy_distillation.sh`: `MODEL_DTYPE=${MODEL_DTYPE:-bf16}` is passed as
  `actor_rollout_ref.actor.fsdp_config.model_dtype=bf16`, so the **actor's master weights are bf16**.
- `verl/verl/workers/fsdp_workers.py::ActorRolloutRefWorker._build_model_optimizer` loads the actor with
  `torch_dtype=model_dtype`, casts it with `actor_module.to(torch_dtype)`, wraps it in FSDP and builds
  the optimizer on `actor_module_fsdp.parameters()`. FSDP's `MixedPrecision(param_dtype=bf16,
  reduce_dtype=fp32)` only sets the compute and gradient-reduction dtypes; the sharded flat parameters
  that the optimizer updates stay in `model_dtype`, gradients are cast back to it, and
  `torch.optim.AdamW` keeps its states and applies the update in bf16. The code comment above this block
  ("we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect") warns
  about exactly this.
- With `lr=1e-6`, Adam's normalized per-element step is about `lr` (~1e-6). bf16 has 8 significant
  bits, so for any weight with `|w| >= 2^-11 ≈ 4.9e-4` half a bf16 ulp exceeds 1e-6 and
  `p.addcdiv_(...)` rounds back to the unchanged value (weight decay `lr*wd*w` is far smaller still).
  Only the ~0.8% of weights below that magnitude change, and always the same ones; about 99% of the
  actor never moves. The gradient direction is correct; the update is lost to rounding.

## Observable impact

- The per-token reverse KL (`-critic/advantages/mean`) only drifts from ≈0.285 to ≈0.255 over ~240
  steps, while an fp32-actor run of the same recipe reaches ≈0.02 by step ~200.
- `actor/grad_norm` stays around 1.4–2.0 (an fp32-actor run falls to ≈0.1), and every logged
  `actor/grad_norm` value is exactly representable in bf16, a fingerprint of bf16 gradients and params.
- Training rule accuracy (`critic/true_reward/mean`) stays ≈0.13–0.23 (fp32 actor: ≈0.55), and offline
  AIME24/AIME25/AMC23 avg@16 of checkpoints 20–120 is flat.
- No NaN/Inf, no crash: forward/backward run in bf16 either way and the gradients are valid.

## Trigger conditions

- Actor master weights in bf16 (`actor.fsdp_config.model_dtype=bf16`) with a plain PyTorch AdamW
  (no fp32 master copy, no stochastic rounding) and a learning rate small enough (here 1e-6) that the
  per-step update falls below bf16 resolution for typical weight magnitudes.
- Not hardware specific: replaying the same AdamW step on CPU gives the same frozen weights.

## Documentation reconciliation

- The teacher/reward-model dtype (`+reward_model.model.dtype=bf16`) is not the cause. The teacher is
  forward-only and runs under bf16 autocast anyway; switching its storage dtype changes its per-token
  log-probs by ~0.007 nats on average and the initial reverse-KL estimate by <0.5%, and an fp32 actor
  with a bf16 teacher learns normally. The same `MODEL_DTYPE` variable feeds both, so blaming "bf16"
  without identifying the actor's master weights/optimizer is incomplete.
- bf16 gradients do not underflow (bf16 has fp32's exponent range) and no loss scaling is needed;
  "vanishing" or "underflowing gradients" is the wrong mechanism. The large, flat `grad_norm` is a
  consequence of the frozen model, not a gradient-scale problem.
- The vLLM rollout dtype (bf16 by design), the unused reference model (`use_kl_loss=False`), the
  FlashInfer sampler setting and the GPU model are not responsible.
- `actor/pg_loss` in this verl version is averaged over micro-batches, so its absolute value depends
  on the number of GPUs and micro-batches; it is not evidence of a bug by itself.
- Casting the actor to bf16 also rounds the RoPE `inv_freq` buffer, a secondary accuracy loss of
  ~0.1% on the loss; it does not explain the stall.

## Repair direction

Keep the actor's master weights (and hence the AdamW update) in fp32, e.g.
`actor_rollout_ref.actor.fsdp_config.model_dtype=fp32`; FSDP mixed precision still computes in bf16.
Equivalent repairs keep an fp32 master copy for the optimizer or use stochastic rounding (for example
torchao's AdamW with `bf16_stochastic_round`). The forward-only teacher may stay in bf16.
