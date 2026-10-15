# rlve_hyper hyperparameter diagnosis: task_001

## Case overview

- Evaluation key: `rlve_hyper/task_001`
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

`--attention-dropout=0.5` keeps held-out eval moving the wrong way (-0.611
to -0.639) with exploding grad_norm, versus healthy runs that still gain
about +0.36 raw reward

## Workspace and hidden evaluations

| Path | Visibility and purpose |
| --- | --- |
| `workspace/` | Agent-visible historical experiment snapshot; evidence rather than canonical ground truth. |
| `evaluations/issue/` | Neutral diagnosis instruction, issue oracle, and task-local issue verifier. |
| `evaluations/solution/` | Repair-and-run instruction, hidden reference material, and metric-only verifier. |

The agent cannot read oracle or verifier paths. `issue` grading compares
`$PTDBENCH_RESULTS/agent_answer.md` to hidden `oracle.md` with an LLM
judge. `solution` grading reads only the new experiment logs.

`environment/` is a third agent-visible surface: Harbor bind-mounts it read-only
at `/ptdbench/environment`, so `environment/prepare.sh` must not name the task
or its category either. `readme.md`, `metadata.yaml` and
`evaluations/*/oracle/` are never copied into the Harbor task directory, so the
defect names they carry (`hyperparameter`, `attention_dropout`,
`run_11_attention_dropout_0.5`) stay hidden.

### Name-leak audit

The category name `rlve_hyper` points straight at the defect class, so it must
not reach the agent. It does not arrive through the harness:
`TaskPackage.task_id` is `paths.root.name`, i.e. the leaf `task_001`, so
`PTDBENCH_TASK_ID`, the Harbor task name (`ptdbench/task_001-solution`), the
trial name and the compose project name all carry only the numeric leaf. The
recorded `agent_trace.json` contains zero occurrences of `hyper`, including in
the `env | grep PTDBENCH` the agent runs on startup.

Two leaks were found and fixed:

| Leak | Where | Fix |
| --- | --- | --- |
| `rlve_hyper/task_001` in a header comment | `environment/prepare.sh:2` | comment no longer names the task or category |
| `"source_run": "run_11_attention_dropout_0.5"` | `workspace/historical_evidence/manifest.json` | field dropped from the agent-visible copy; `oracle/historical_evidence_manifest.json` keeps it |

The second was the serious one: the run label names both the root cause and its
value, and the recorded agent run did read that file. The converter wrote one
manifest dict to both the visible and the oracle path, so
`convert/convert_rlve_hyper.py` now routes the visible copy through
`visible_manifest()`. `attention_dropout: 0.5` still appears in
`historical_evidence/hparams_effective.json` and in the launch script — that is
the intended evidence, not a leak. The remaining `task_001` strings
(`rlve.sh`'s `RUN_NAME` default, the manifest's `task_id`) are numeric and name
nothing.

## Source and preserved context

The following source and historical packaging fields were moved from
`metadata.yaml` so the machine metadata remains aligned with the core
schema. They are preserved here without changing their meaning.

```yaml
bench_category: rlve_hyper
bench_subcategory: collapse
source_task: Benchmark_Evaluation/tasks/task_RLVE_hyper/task_001
source_run: run_11_attention_dropout_0.5
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
  eval_last: -0.6393607996231563
  eval_min: -0.647408764043118
  truncate_ratio_max: 0.1171875
  n_eval_points: 4
failure_signature: metric_failure
required_final_score: "-0.24"
```

Two of those preserved fields no longer describe the live protocol:

- `timeout_minutes: 90` is inherited from the source benchmark and is wrong by
  about 3.4x. The calibration run took 5h04m on 4x H100 (02:29:03 to 07:33:36).
  Nothing enforces it: `_write_task_config()` in
  `ptdbench/evaluation/harbor_evaluator.py` never emits `[agent] timeout_sec`,
  so Harbor's `AgentConfig.timeout_sec` stays `None` and the agent phase is
  untimed (`--timeout-multiplier` has no base to scale). Treat the number as a
  provisioning hint only, and provision for ~5h.
- `required_final_score: "-0.24"` was a family-wide constant shared verbatim by
  `task_001` / `task_012` / `task_019` despite their different baselines, and it
  was never derived from a run. The live bar is `-0.50` on the final step; see
  `evaluations/solution/verifier/resources/README.md` for the calibration.

## Original failure evidence

- rollout/raw_reward start to end -0.650 to -0.594 (+0.056); healthy runs worst-case still +0.36
- train/entropy_loss 4.04 to 3.52; healthy upper bound 1.40
- train/grad_norm 2258 to 19760; healthy runs stay <= 9.3
- eval -0.611 to -0.639 (moves the wrong way)

The preserved source metric is `eval/HELD-OUT_ENVIRONMENTS_128`. The frozen workspace remains
the evidence package; a full training rerun is outside verifier scope.

## Healthy protocol baseline

A healthy run of Qwen2.5-0.5B-Instruct on the same RLVE held-out
environments under slime/GRPO keeps training reward moving (healthy
worst-case raw_reward gain still about +0.36) and held-out
`eval/HELD-OUT_ENVIRONMENTS_128` around `-0.62 ± 0.03` at the start of
eval. This case's budget is 100 rollouts.

That cross-run p10 figure is no longer what sets the gate. Both arms of this
task have since been run end to end under the live protocol, and the bar is
calibrated against them directly: repaired step 99 `-0.170`, defect step 99
`-0.634`, gate `-0.50`. The calibration table and the reasoning for placing the
bar off the midpoint are in
`evaluations/solution/verifier/resources/README.md`.

## Canonical expected outcome

Verdict: **unhealthy**.

Root cause targets:

- `--attention-dropout`

Required fix targets:

- `scripts/training/Qwen2.5-0.5B-Instruct/rlve.sh::--attention-dropout`: `decrease`; target state/value: `0`.

Documentation reconciliation:
- High entropy and exploding grad_norm are symptoms of the dropout setting, not separate root causes.

## Evaluation design

`issue` grades `$PTDBENCH_RESULTS/agent_answer.md` against hidden
`oracle.md`. The judge scores verdict, root-cause, and evidence agreement
(weights 0.2 / 0.6 / 0.2). Pass is a weighted score of at least `0.8`.

`solution` does not grade a structured diagnosis. The agent must repair the
staged workspace, run the designated launch script to completion, and produce a
finite `eval/HELD-OUT_ENVIRONMENTS_128` of at least `-0.50` **at step 99**, the
last eval point of the 100-rollout schedule. Pass is score `1`; otherwise `0`.
A missing step 99 fails, so the run has to finish; the curve peak is reported
but not graded.

## External dataset asset

- Repository: `LIF1014/ptdbench-rlve-hyper-task-001-dataset`
- Publication status: `published`
- Pinned revision: `1c33b31d0632b6a5508b8f1b88d3c98f1cf4e2e7`
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
