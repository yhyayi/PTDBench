# Offline evaluation of the run's checkpoints

Actor checkpoints saved by `on_policy_distillation.sh` (every 20 steps) were merged to Hugging Face format
and evaluated offline with vLLM, one GPU per checkpoint:

- benchmarks: AIME24 (30 problems), AIME25 (30), AMC23 (40)
- sampling: 16 samples per problem, temperature 0.7, top_p 0.95, max 31744 new tokens
- grading: rule-based answer matching (no model verifier)
- `mean_score` is avg@16, `best_score` is pass@16

`global_step_*/grading_results.json` holds the per-benchmark results. AMC23 was not generated for step 120.

| step | AIME24 avg@16 | AIME25 avg@16 | AMC23 avg@16 |
|---:|---:|---:|---:|
| 20 | 27.7 | 24.2 | 73.4 |
| 40 | 27.3 | 24.0 | 70.3 |
| 60 | 26.2 | 23.3 | 69.4 |
| 80 | 29.4 | 24.4 | 71.6 |
| 100 | 30.0 | 23.3 | 72.8 |
| 120 | 31.9 | 24.4 | - |
