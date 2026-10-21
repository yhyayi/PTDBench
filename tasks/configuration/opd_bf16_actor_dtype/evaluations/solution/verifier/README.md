# Solution verifier

`test.sh` runs a shortened OPD job through the repaired workspace's `on_policy_distillation.sh`
(15 steps, batch 16 prompts × 4 samples, 3072 response tokens, GPUs from `CUDA_VISIBLE_DEVICES`,
15–20 minutes on 2× A800) and saves the actor in HF format at step 15. Trailing Hydra overrides pin
the recipe (models, data, `token_reward_direct`, `log_prob_top_k=0`, AdamW lr 1e-6 / weight decay 0.01 /
betas / grad clip 1.0); `MODEL_DTYPE` and the other launcher variables are unset so only the workspace's
own launcher and code decide the precision setup. A failure before the first training step is retried
once.

`tests/check_run.py` writes `$PTDBENCH_RESULTS/result.json` and passes only if:

| check | rule |
| --- | --- |
| `completed_steps` | steps 1–15 all logged `critic/advantages/mean` |
| `weights_updated` | ≥ 2.5% of the actor's weights changed in bf16 value vs. the initial student (SHA-256 checked) |
| `kl_progress` | `(late − buggy) / (reference − buggy) ≥ 0.5`, where `late` is the mean advantage over steps 8–15 and `buggy`/`reference` come from `oracle/metrics.json` |

`score` is the fraction of checks passed. The share of bf16-representable grad norms is reported as a
diagnostic only, so repairs with an fp32 master copy or stochastic rounding are judged by behavior.

`oracle/metrics.json` records two buggy and two reference runs of this exact configuration on
2× A800-80GB.
