# Colocated On-Policy Distillation via KL-in-Reward (verl 0.9.0)

## Goal
OPD where teacher and student share the same GPU pool. Teacher log-probs become a
token-level reward; advantage = `-(logπ_stu - logπ_tch)`, fed through the policy
gradient. `top_k=0` → sampled-token OPD. `top_k>0` → four strategies
(`only_stu` / `only_tch` / `union` / `intersection`).

## Why not the reward-model vLLM server
`only_stu` / `union` need teacher log-probs at the **student's per-position top-k
ids**, which vLLM's sampling API cannot express:
- `prompt_logprobs=-1` (full vocab): V=151936 × T=7168 ≈ **4.36 GB fp32 per
  sequence**, materialized as `list[dict[int, Logprob]]` ≈ 1.1e9 Python objects.
  A 256-sequence batch is ~1 TB. Not viable over HTTP.
- `logprob_token_ids` is one flat list per request, not per-position.

The teacher only does a fixed-length forward pass — no sampling, no KV cache — so
vLLM's scheduler buys nothing. We host it as a **colocated forward-only engine
worker** and gather on-GPU, so only `[B,T,K]` leaves the device. This is also
what the reference impl does (FSDP RM worker, not vLLM).

## Architecture
The ref policy is already a colocated forward-only inner `TrainingWorker` in the
actor's pool (`engine_workers.py:537-582`). The teacher is structurally the same
thing with a different `model_path`. We add a third inner worker: `self.teacher`.

```
generate → student top-k pass → teacher pass → build rm_scores [B,T,K]
        → advantage [B,T,K] → actor update (3D PPO loss)
```

## Gradient
Per position t, over the strategy's token set S_t:

    adv[t,k] = -w[t,k] · (logπ_stu(k) - logπ_tch(k))          # detached
    loss     = -Σ_t Σ_k adv[t,k] · ratio[t,k]

When `is_norm=True`, every probability/log-prob above is conditioned on the
fixed candidate set and `ratio = π_θ(k|C) / π_old(k|C)`. With
`reward_weight_mode=student_p`, ratio ≡ 1 recovers the exact gradient of
`KL(π_stu(.|C) || π_tch(.|C))`.

With the full vocabulary and `w = softmax(logπ_stu)`, ratio ≡ 1 recovers the exact
reverse-KL gradient. For strict top-k sets with `is_norm=False`, this is a sparse
approximation to the full-vocabulary gradient. `advantages` is `[B,T,K]`; the actor recomputes
top-k log-probs at the stored ids during update, so ratio ≠ 1 (mini-batching,
`ppo_epochs>1`) stays correct and clipping still applies. `top_k=0` degenerates
to K=1 on the sampled token, i.e. naive OPD.

### Token sets and what each strategy needs
`S_ids/S_logp` = student top-k; `T_ids/T_logp` = teacher top-k;
`T_on_S` = teacher log-prob gathered at `S_ids`; `S_on_T` = student log-prob at `T_ids`.

| strategy | set | needs | teacher-side cost |
|---|---|---|---|
| `top_k=0` | sampled token | `T_on_S` (K=1) | gather 1/pos |
| `only_stu` | `S_ids` | `S_logp`, `T_on_S` | gather K/pos |
| `only_tch` | `T_ids` | `T_logp`, `S_on_T` | topk + student 2nd pass |
| `intersection` | `S_ids ∩ T_ids` | above + `overlap_mask` | topk + gather |
| `union` | `S_ids ∪ T_ids` | all four + `T_in_S` mask | topk + gather + student 2nd pass |

All of it comes from **one teacher forward**: chunked `log_softmax` over the vocab,
then `topk` + `gather` per chunk. Peak extra memory is `chunk × V`, never `B×T×V`.
`only_tch` / `union` additionally need a student forward at `T_ids` (`S_on_T`).

## Files to change

1. **`verl/workers/config/opd.py`** (new) — `OPDConfig`: `enable`, `teacher_model_path`,
   `top_k`, `strategy`, `reward_weight_mode`, `teacher_temperature`, `chunk_size`,
   `kl_estimator`, `grpo_outcome_weight`.
2. **`verl/trainer/config/opd/opd.yaml`** + `dp_opd.yaml` (new) — teacher engine config,
   `forward_only: True`, param offload on by default. Mirrors `ref/dp_ref.yaml`.
   Registered in `ppo_trainer.yaml` defaults as `opd@opd`.
3. **`verl/workers/engine_workers.py`** — build `self.teacher` inner `TrainingWorker`
   when OPD is on (clone of the ref branch, different `model_path`); add
   `compute_teacher_opd(data)` and extend `compute_log_prob` to emit student top-k.
4. **`verl/workers/engine/fsdp/transformer_impl.py`** — in `prepare_model_outputs`,
   add an `opd_mode` branch producing `[total_nnz,K]` nested outputs
   (`topk_ids`, `topk_log_probs`, `gathered_log_probs`, `overlap_mask`, `t_in_s_mask`).
   Chunked over `total_nnz` so the vocab-sized buffer stays bounded.
   The existing `logits_processor_func` hook is unusable here — it asserts
   per-token scalars `(total_nnz,)` (line 1376).
5. **`verl/workers/utils/padding.py`** — extend `left_right_2_no_padding` to convert
   the OPD `[B,S,K]` tensors to nested, following the existing `teacher_logprobs`
   precedent (lines 84-96). `no_padding_2_padding` already handles trailing dims.
6. **`verl/workers/utils/losses.py`** — in `ppo_loss`, when `advantages.dim()==3`
   use `model_output["topk_log_probs"]`, expand `response_mask` to `[B,T,1]`, and
   normalize by valid `(t,k)` count.
7. **`verl/trainer/ppo/core_algos.py`** — register `opd_token_kl` and
   `opd_token_kl_plus_grpo` advantage estimators (3D-aware; GRPO term stays 2D and
   broadcasts over K).
8. **`verl/trainer/ppo/ray_trainer.py`** — orchestrate the teacher pass in `fit()`
   between reward and advantage; write `rm_scores`; guard `use_kl_in_reward` and
   `use_critic` against 3D; pass OPD tensors into `_update_actor`.
9. **`verl/trainer/ppo/utils.py`** — `need_teacher_opd(config)` helper.
10. **`verl/trainer/ppo/metric_utils.py`** — make `compute_data_metrics` 3D-safe
    (`masked_select` at line 490 breaks on a 2D mask vs 3D advantages); add
    `opd/overlap_ratio`, `opd/kl_mean`, `opd/student_mass`, `opd/teacher_mass`.
11. **`examples/opd_colocate/run_opd_colocate.sh`** (new) — runnable script for the
    local 1.5B student + 1.5B teacher pair.

## Tests (CPU, no GPU needed)
- `tests/trainer/test_opd_advantage_on_cpu.py` — the four strategies against a
  hand-computed reference on a tiny vocab; assert `only_stu` with
  `reward_weight_mode=student_p` matches the analytic reverse-KL gradient over the
  top-k set; assert `top_k=0` equals the sampled-token form.
- `tests/workers/test_opd_padding_on_cpu.py` — `[B,S,K]` → nested → `[B,T,K]`
  roundtrip and left-shift alignment.
- `tests/workers/test_opd_gather_on_cpu.py` — chunked gather/topk/overlap-mask
  equals the unchunked computation.

## Verification
`ruff` + `ruff format`, then the three CPU tests, then a short end-to-end smoke run
(`DeepSeek-R1-Distill-Qwen-1.5B` student, `JustRL-DeepSeek-1.5B` teacher,
`top_k=8`, 2 steps) per strategy, checking loss is finite and
`opd/overlap_ratio` is sane. I'll report which of these actually ran.

## Deliberately out of scope
- Megatron / veomni OPD paths (FSDP only; the branch will raise a clear error).
- `union-intersection` (the reference's 5th strategy) — not in your list.
- Multi-teacher routing.
- Teacher tokenizer ≠ student tokenizer (asserted equal at startup).

## Known costs
- `advantages` and `old_log_probs` become `[B,T,K]`: at B=256, T=7168, K=16 that is
  ~1.2 GB each in fp32 on CPU between phases. Stored as bf16 where safe; ids as int32.
- Teacher weights are resident in the same pool. With `param_offload=True` they sit
  on CPU between passes, costing a H2D copy per step.
- `only_tch` / `union` need a second student forward for `S_on_T`.
