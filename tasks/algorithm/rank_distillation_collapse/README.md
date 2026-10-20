# Ranking distillation collapse

Training a 1.5B student with on-policy ranking distillation on DAPO-Math-17k. The ranking loss improves while generation quality collapses.

| | |
| --- | --- |
| **Author** | AutoRLBench |
| **Domain** | algorithm |
| **Tags** | `verl` `distillation` `ranking-loss` `degeneration` |
| **Model** | DeepSeek-R1-Distill-Qwen-1.5B (student) · DeepSeek-R1-Distill-Qwen-7B (teacher) |
| **Dataset** | DAPO-Math-17k · AIME25 / AMC23 / AIME26 validation |
| **Framework** | verl, full-parameter AdamW, `opd_train/run_recorded.sh` |
| **Environment** | `rank-distillation-cpu` (CPU issue profile) · 8 × A800 80GB for the training profile |
| **Evaluations** | issue |
| **Recorded runs** | 2 real runs (A: BF16 teacher head, B: FP32 head + tie masking), ~11 h each |

## Difficulty

The objective is a softplus over adjacent pairs of the teacher's top-K candidates. It constrains only their *relative* order and is invariant to a shared shift of their logits. The loss can keep falling while the student's total probability on the candidate set goes to zero.

| | A | B |
| --- | ---: | ---: |
| teacher top-8 mass, step 1 → 200 | 0.932 → 0.0007 | 0.933 → 0.0004 |
| output entropy, step 1 → 200 | 1.02 → 11.75 | 0.99 → 11.80 |
| all three validation sets, step 140 | 0% | 0% |

Run B removes teacher ties and still collapses, which rules out teacher precision. No single-factor ablation exists. A correct answer explains the mechanism and does not overclaim causal proof.

## Reference solution

- **issue**: state that the ranking objective is invariant to a common shift of the candidate logits, that this direction is unpenalized, and that with shared parameters the off-candidate outputs are not independent. Generic "missing regularization", teacher precision, instrumentation-only explanations, and overclaimed causality do not pass.

## Verification

- **issue**: an LLM judge scores verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. Passing requires ≥ 0.8. If no judge model is configured, the verifier writes `status=unavailable` instead of a pass.

The default prepare profile installs only CPU PyTorch, so the agent can run tensor probes offline. `AUTORLBENCH_PREPARE_PROFILE=training` installs the full GPU stack and fetches the ~18.8 GB of weights.
