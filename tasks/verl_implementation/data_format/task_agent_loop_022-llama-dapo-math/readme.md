# Data-format run assessment: task_agent_loop_022-llama-dapo-math

## Case overview

- Evaluation key: `verl_implementation/data_format/task_agent_loop_022-llama-dapo-math`
- Case family: `data_format`
- Schema version: `2`
- Evaluation levels: `issue`, `solution`
- Model: `Llama-3.2-1B-Instruct`
- Dataset: `DAPO-Math-17k`
- Training framework: `verl`
- Primary source evaluation metric: `val-core/math_dapo/reward/mean@1`

This document is for maintainers and reviewers. It is not exposed to the
agent during evaluation. Metric names preserve the source experiment;
tasks using the same held-out protocol may intentionally share a name.

## Canonical failure reason

Rewritten agent-loop methods break rollout and reward assembly, while the Answer-only scorer lacks the healthy format stack's boxed-answer fallback; validation reaches only 1.23% versus a healthy protocol baseline of 8.4% at 200 steps (11.8% at 1000 steps)

## Workspace and hidden evaluations

| Path | Visibility and purpose |
| --- | --- |
| `workspace/` | Agent-visible historical experiment snapshot; evidence rather than canonical ground truth. |
| `evaluations/issue/` | Neutral diagnosis instruction, issue oracle, and task-local issue verifier. |
| `evaluations/solution/` | Repair-and-run instruction, hidden reference material, and metric-only verifier. |

The agent cannot read oracle or verifier paths. `issue` grading compares
`$PTDBENCH_RESULTS/agent_answer.md` to hidden `oracle.md` with an LLM
judge. `solution` grading reads only the new experiment logs.

## Source and preserved context

The following source and historical packaging fields were moved from
`metadata.yaml` so the machine metadata remains aligned with the core
schema. They are preserved here without changing their meaning.

```yaml
bench_category: data_format
source_task: Benchmark_Evaluation/tasks/task_data_format/task_agent_loop_022-llama-dapo-math
canonical_task: task_llama_dapo_math_17k/task_agent_loop_022
data_format_variant: true
evidence_kind: dirty_agent_workspace
packaging_tier: B0
packaging_ready: true
b1_scoring_ready: false
verify:
  num_gpus: 2
  timeout_minutes: 90
  max_train_steps: 100
  eval_metric_key: val-core/math_dapo/reward/mean@1
historical_run:
  metric_status: log_reconstructed
  eval_first: 0.006148686416992733
  eval_last: 0.012297372833985467
  eval_max: 0.012297372833985467
  n_eval_points: 6
  attempt: 6
```

## Original failure evidence

- six validation points rise from about 0.61% to 1.23% by step 100
- a healthy Llama-3.2-1B-Instruct + DAPO-Math-17k run reaches about 8.4% at 200 steps and 11.8% at 1000 steps; this case only ran 100 steps and is not reproduced at that level in the frozen workspace
- training critic/score is zero for most steps, and an earlier attempt fails with KeyError: multi_modal_inputs
- all 17,917 pinned parquet prompts explicitly request a final Answer: line; the workspace extractor honors that nominal contract but lacks the healthy format stack's boxed-answer fallback

The preserved source metric is `val-core/math_dapo/reward/mean@1`. The frozen workspace remains
the evidence package; a full training rerun is outside verifier scope.

## Healthy protocol baseline

A healthy run of Llama-3.2-1B-Instruct on DAPO-Math-17k under the same
verl/GRPO protocol reaches about **8.4% at 200 steps** and **11.8% at
1000 steps**. This case's budget is 100 steps; the frozen `eval_last`
1.23% is well below that protocol baseline.

## Canonical expected outcome

Verdict: **unhealthy**.

Root cause targets:

- `verl/experimental/agent_loop/agent_loop.py`
- `verl/utils/reward_score/math_dapo.py::is_correct_minerva`

Required fix targets:

- `verl/experimental/agent_loop/agent_loop.py`: `restore`; target state/value: baseline rollout, reward, mask, and batch-field semantics.
- `verl/utils/reward_score/math_dapo.py::is_correct_minerva`: `align_format_contract`; target state/value: accept Answer: and fall back to the last boxed answer.

Documentation reconciliation:
- Pinned parquet inspection disproves a literal prompt/extractor mismatch: every prompt requests Answer:, matching the workspace regex.
- The format defect is a robustness gap between actual model-output conventions and the healthy format stack; no frozen rollout sample records the boxed-output frequency.
- Either listed root cause is a correct attribution.

## Evaluation design

`issue` grades `$PTDBENCH_RESULTS/agent_answer.md` against hidden
`oracle.md`. The judge scores verdict, root-cause, and evidence agreement
(weights 0.2 / 0.6 / 0.2). Pass is a weighted score of at least `0.8`.
Either listed root cause is independently sufficient; naming one with the
correct causal role should receive full `root_cause_score`.

`solution` does not grade a structured diagnosis. The agent must repair the
staged workspace, run the designated launch script, and produce a finite
`val-core/math_dapo/reward/mean@1` at step 100 of at least `0.06` (6.0%).
Pass is score `1`; otherwise `0`.

## External dataset asset

- Repository: `LIF1014/ptdbench-data-format-task-agent-loop-022-llama-dapo-math-dataset`
- Publication status: `published`
- Pinned revision: `87e9793734359014fcbc22fa62d866930180b606`
- Dataset license: `Apache-2.0`
- Provenance: Processed from BytedTsinghua-SIA/DAPO-Math-17k; task-specific bytes are pinned.

| Runtime path | Size | SHA-256 |
| --- | ---: | --- |
| `workspace/DAPO-Math-17k_verl/test.parquet` | 757,051 bytes | `bc2c8d0ddd217dd419884c34b3ce886a8bb7f5db996ce579524dc317503ef704` |
| `workspace/DAPO-Math-17k_verl/train.parquet` | 6,667,434 bytes | `9656de825e924e851abedb8d069a66d894d71abc78d0cd3df04b69796cb3e1c3` |

The Docker image downloads the immutable revision at build time, verifies
the artifact, and hydrates the bind-mounted workspace only for the agent stage.

## Environment requirement

- Base image: `verlai/verl:vllm017.latest`
- Use the Docker agent backend after external asset publication.
- The local backend does not build the task image or hydrate external assets.
- Model weights and historical training checkpoints are not dataset assets.
