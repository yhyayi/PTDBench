# rlve_hyper hyperparameter diagnosis: task_016

## Case overview

- Evaluation key: `rlve_hyper/task_016`
- Case family: `rlve_hyper`
- Schema version: `2`
- Evaluation levels: `issue`, `solution`
- Model: `Qwen2.5-0.5B-Instruct`
- Dataset: `RLVE held-out environments`
- Training framework: `RLVE / slime + GRPO`
- Primary source evaluation metric: `eval/HELD-OUT_ENVIRONMENTS_128`

This document is for maintainers and reviewers. It is not exposed to the
agent during evaluation. Metric names preserve the source experiment;
tasks using the same held-out protocol may intentionally share a name.

## Canonical failure reason

`--attention-dropout=0.7` leaves held-out eval around -0.61 with entropy
and grad_norm well outside the healthy range

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
bench_category: rlve_hyper
bench_subcategory: collapse
source_task: Benchmark_Evaluation/tasks/task_RLVE_hyper/task_016
source_run: run_42_attention_dropout_0p7
evidence_kind: historical_failure
packaging_tier: B0
packaging_ready: true
b1_scoring_ready: false
label_review: false
bug_type: hyperparameter/dropout
buggy_keys: attention_dropout
verify:
  num_gpus: 4
  timeout_minutes: 90
  max_train_steps: 100
  eval_interval: 20
  eval_metric_key: eval/HELD-OUT_ENVIRONMENTS_128
historical_baseline:
  actual_metric_key: eval/HELD-OUT_ENVIRONMENTS_128
  eval_last: -0.6099539726536959
  eval_min: -0.6128061883700294
  truncate_ratio_max: 0.109375
  n_eval_points: 3
failure_signature: metric_failure
required_final_score: "-0.24"
```

## Original failure evidence

- raw_reward -0.595 to -0.592; barely moves
- entropy 5.50 to 5.63; out of healthy range
- grad_norm 30094 to 1672
- eval -0.605 to -0.610

The preserved source metric is `eval/HELD-OUT_ENVIRONMENTS_128`. The frozen workspace remains
the evidence package; a full training rerun is outside verifier scope.

## Healthy protocol baseline

A healthy run of Qwen2.5-0.5B-Instruct on the same RLVE held-out
environments under slime/GRPO keeps training reward moving (healthy
worst-case raw_reward gain still about +0.36) and held-out
`eval/HELD-OUT_ENVIRONMENTS_128` around `-0.62 ± 0.03` at the start of
eval, with eval improvement at least the healthy p10 of about +0.21.
This case's budget is 100 rollouts; the frozen `eval_last` is well below
the solution gate of `-0.24`.

## Canonical expected outcome

Verdict: **unhealthy**.

Root cause targets:

- `--attention-dropout`

Required fix targets:

- `scripts/training/Qwen2.5-0.5B-Instruct/rlve.sh::--attention-dropout`: `decrease`; target state/value: `0`.

Documentation reconciliation:
- High entropy and large grad_norm are symptoms of the dropout setting, not separate root causes.

## Evaluation design

`issue` grades `$PTDBENCH_RESULTS/agent_answer.md` against hidden
`oracle.md`. The judge scores verdict, root-cause, and evidence agreement
(weights 0.2 / 0.6 / 0.2). Pass is a weighted score of at least `0.8`.

`solution` does not grade a structured diagnosis. The agent must repair the
staged workspace, run the designated launch script, and produce a finite
`eval/HELD-OUT_ENVIRONMENTS_128` of at least `-0.24`. Pass is score `1`;
otherwise `0`.

## External dataset asset

- Repository: `LIF1014/ptdbench-rlve-hyper-task-016-dataset`
- Publication status: `published`
- Pinned revision: `bc39fe4708a7fad8e7a1649dbc0b2d56aa915f62`
- Dataset license: `MIT`
- Provenance: Packaged from Benchmark_Evaluation RLVE held-out environments; task-specific workspace assets are pinned.

| Runtime path | Size | SHA-256 |
| --- | ---: | --- |
| `workspace/data/HELD-OUT_ENVIRONMENTS/test_128.json` | 110,359 bytes | `3ebd3497305f75cb0712c4a7dd5105b3af0c32b1cdc85f2bdf1a6913e6db4018` |
| `artifacts/workspace-assets.tar.gz` | 9,207,578 bytes | `d7164e8c27586efae4daeb972f858092390c475938fcb848e213612bb4190a9e` |

The Docker image downloads the immutable revision at build time, verifies
the artifact, and hydrates the bind-mounted workspace only for the agent stage.

## Environment requirement

- Base image: `slimerl/slime:v0.5.0rc0-cu126`
- Use the Docker agent backend after external asset publication.
- The local backend does not build the task image or hydrate external assets.
- Model weights and historical training checkpoints are not dataset assets.
