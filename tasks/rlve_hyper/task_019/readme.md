# rlve_hyper hyperparameter diagnosis: task_019

## Case overview

- Evaluation key: `rlve_hyper/task_019`
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

`--lr-warmup-iters=2000` keeps the schedule from reaching the 2e-6 baseline
(over the 100-step budget the lr stays inside the warmup ramp and ends at
1e-7), so eval improvement stays below healthy p10. `--lr-decay-iters=10000`
is a non-causal decoy: under `--lr-decay-style constant` the scheduler
returns `max_lr` immediately after warmup and never consults
`lr_decay_iters`, so it has no effect at any step count.

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
source_task: Benchmark_Evaluation/tasks/task_RLVE_hyper/task_019
source_run: run_48_warmup_iters_2000_lr_warmup_decay=10000
evidence_kind: historical_failure
packaging_tier: B0
packaging_ready: true
b1_scoring_ready: false
label_review: false
bug_type: hyperparameter/warmup
buggy_keys: lr_warmup_iters
decoy_keys: lr_decay_iters  # present in the script but inert under constant decay
verify:
  num_gpus: 4
  # The source field said 180 (~160 min); the repaired rerun of this staged
  # workspace measured 5h28m on 4xH100 (see "Measured reruns" below). This field
  # is documentation only -- harbor_evaluator._write_task_config() emits no
  # [agent] timeout_sec, so the agent phase is untimed in practice.
  timeout_minutes: 330
  max_train_steps: 100
  eval_interval: 20
  eval_metric_key: eval/HELD-OUT_ENVIRONMENTS_128
historical_baseline:
  actual_metric_key: eval/HELD-OUT_ENVIRONMENTS_128
  eval_last: -0.5221301909316525
  eval_min: -0.644444577723881
  truncate_ratio_max: 0.1484375
  n_eval_points: 6
failure_signature: metric_failure
required_final_score: "-0.40"  # recalibrated 2026-10-15; the source value was "-0.24"
```

## Original failure evidence

- train/lr-pg_0 ends at only 1e-7; never reaches baseline 2e-6 (warmup ramp over 100 of 2000 iters)
- raw_reward -0.599 to -0.531 (+0.068)
- eval -0.620 to -0.522; improvement below healthy p10

The preserved source metric is `eval/HELD-OUT_ENVIRONMENTS_128`. The frozen workspace remains
the evidence package; a full training rerun is outside verifier scope.

## Healthy protocol baseline

A healthy run of Qwen2.5-0.5B-Instruct on the same RLVE held-out
environments under slime/GRPO keeps training reward moving (healthy
worst-case raw_reward gain still about +0.36) and held-out
`eval/HELD-OUT_ENVIRONMENTS_128` around `-0.62 ± 0.03` at the start of
eval, with eval improvement at least the healthy p10 of about +0.21.
This case's budget is 100 rollouts; the frozen `eval_last` is well below
the solution gate of `-0.40`.

## Measured reruns

Both arms of this exact staged workspace were rerun on 4xH100 (devspace
`1177503`, `/user/chenyinghao/harbor-runs/task_019__stage__env`). The two trees
differ only by the oracle repair in `rlve.sh`; every other tracked file matches
by md5.

| step | defect arm (`--lr-warmup-iters 2000`) | repaired arm (`--lr-warmup-iters 0`) |
| ---: | ---: | ---: |
| 0 | -0.6195 | -0.5767 |
| 19 | -0.6149 | -0.2533 |
| 39 | -0.5635 | -0.2384 |
| 59 | -0.6305 | -0.1355 |
| 79 | -0.6525 | -0.1419 |
| **99** | **-0.5321** | **-0.1021** |
| mean `train/lr-pg_0` | 5.05e-8 | 2.0e-6 |

The defect arm's step-99 value (-0.5321) confirms the frozen full-scale
`eval_last` (-0.5221) despite the staged downscaling, so the historical baseline
remains usable. Both arms evaluate the same base checkpoint at step 0 and differ
by 0.042, which puts one eval point's sigma at about 0.03. The repaired arm took
5h28m wall-clock at a steady ~65 min per 20 steps.

## Canonical expected outcome

Verdict: **unhealthy**.

Root cause targets:

- `--lr-warmup-iters` (essential)
- `--lr-decay-iters` (non-essential; inert under constant decay)

Required fix targets:

- `scripts/training/Qwen2.5-0.5B-Instruct/rlve.sh::--lr-warmup-iters`: `decrease`; target state/value: `0`, or any small warmup the judge considers reasonable. This is the only fix that restores the schedule.
- `scripts/training/Qwen2.5-0.5B-Instruct/rlve.sh::--lr-decay-iters`: `remove` (optional cleanup); target state/value: unset. Removing it is harmless but not required — it does not affect the run under `--lr-decay-style constant`.

Documentation reconciliation:
- `--lr-warmup-iters=2000` alone keeps the schedule from reaching the baseline `2e-6`.
- `--lr-decay-iters` is inert under constant decay and is not part of the causal chain; omitting it from a diagnosis must not reduce the score.
- Weak reward/eval movement is a symptom of the truncated learning-rate schedule.
- The judge may accept any expected values it considers reasonable for restoring a schedule that can reach the baseline learning rate; the listed targets are references, not the only correct numbers.

## Evaluation design

`issue` grades `$PTDBENCH_RESULTS/agent_answer.md` against hidden
`oracle.md`. The judge scores verdict, root-cause, and evidence agreement
(weights 0.2 / 0.6 / 0.2). Pass is a weighted score of at least `0.8`.
`--lr-warmup-iters` is the sole essential root cause; naming it with its
causal role earns full `root_cause_score`. `--lr-decay-iters` is a
non-causal decoy (inert under `--lr-decay-style constant`): omitting it must
not reduce the score, and an answer that correctly argues it is inert should
be credited. Listed expected values are references; the judge may accept any
values it considers reasonable for restoring a schedule that can reach the
baseline learning rate.

`solution` does not grade a structured diagnosis. The agent must repair the
staged workspace, run the designated launch script, and produce fresh
experiment logs. Scoring is weighted:

- validation: `eval/HELD-OUT_ENVIRONMENTS_128` **at step 99** (the last step of
  the locked 100-rollout budget) must exist and fall in `[-0.40, 0.30]` to score
  `1`, otherwise `0`
- learning rate: `train/lr-pg_0` must be logged at step 99 across at least 95
  steps, and its mean must be at least `8e-7` to score `1`, otherwise `0`
- final `score = 0.8 * validation + 0.2 * learning rate`

Pass requires both components (`score == 1.0`). The framework still records
the weighted float when only one component passes.

Recalibrated 2026-10-15 against the measured reruns above. What changed and why:

- Grading moved from the peak at any step to the value at `final_step = 99`. The
  old peak rule passed a repaired run truncated after `eval 39` with `1.0` (its
  -0.2384 cleared the old `-0.24` bar by 0.0016), so 60% of the compute budget
  was optional; peak also grew with `--num-rollout`, while a fixed step index
  does not.
- `final_step` went from `100` to `99`. Step 100 never appears in any log
  (`--num-rollout 100` gives steps 0..99, `--eval-interval 20` gives
  0/19/39/59/79/99), so the old value would have scored a perfect run as 0 the
  moment anything keyed on it.
- `required_final_score` -0.24 -> -0.40, between the defect arm's step-99 value
  (-0.5321) and the repaired arm's worst post-training point (-0.2533 at step
  19), whose midpoint is -0.393 -- ~0.13 of margin each way. The repaired arm's
  post-training points swing by 0.15, so the bar is set against its weakest
  checkpoint, not against its step-99 value. `-0.24` was a family constant
  shared verbatim by `task_{001,012,019}` despite step-99 baselines of -0.639 /
  -0.614 / -0.522, and it predates both reruns.
- Added `required_final_score_max = 0.30`. RLVE reward is bounded to `[-1, 1]`
  (`Gym/environment.py:183`), the repaired arm ended at -0.102, and the grader
  otherwise accepts any line in any log file under `experiment_logs` -- a
  fabricated `eval 99: {... 0.9}` used to pass.
- Added budget coverage: the step-99 eval point and at least 95 logged
  `train/lr-pg_0` steps are required.
- `required_learning_rate` stays at `8e-7`; the measured means sit 16x below and
  2.5x above it.

Residual, accepted: eval and lr points are still merged per step across every
log under `experiment_logs` with a per-step max, so repeated runs give
best-of-N at step 99, and `oracle/task.yaml`'s `max_runs: 1` is not enforced in
the container. The lr component can also be satisfied by raising `--lr` while
leaving the warmup intact (mean lr is restored, the constant schedule is not);
such a run still has to clear the step-99 eval bar. And a partial repair such as
`--lr-warmup-iters 100` keeps mean lr at ~1e-6, clearing the lr component, and
may land its step-99 eval between the two arms -- no partial arm has been
measured, so where it falls relative to `-0.40` is unknown.

## Task-name sanitization

Audited 2026-10-15 for leaks of the task's own name into agent-visible surfaces.
What the agent can actually read is `workspace/` plus `instruction.md` plus the
`environment/` directory (mounted at `/ptdbench/environment` by the docker
backend and at `/workspace/.ptdbench-environment` by the cybertron backend);
`metadata.yaml`, this readme, `evaluations/*/oracle/` and
`evaluations/*/verifier/` are not mounted for the agent stage, and
`PTDBENCH_TASK_ID` is the leaf id only (`ptdbench/core/types.py:77`), not
`rlve_hyper/task_019`.

Fixed:

- `environment/prepare.sh:2` opened with `# rlve_hyper/task_019`, which named
  the case family -- i.e. "the defect is a hyperparameter" -- in a file the agent
  can read. Now `__BENCH_TASK_ID__`, with a note that the file is agent-visible.
  A cross-reference to a sibling task by name was also neutralized.
- `workspace/historical_evidence/manifest.json` carried
  `source_run: run_48_warmup_iters_2000_lr_warmup_decay=10000`, which spells out
  both buggy keys and both values. The field is dropped from the agent-visible
  copy (`convert_rlve_hyper.visible_manifest`) and kept in
  `evaluations/solution/oracle/historical_evidence_manifest.json` for
  provenance. The committed workspace copy now equals the generator's output.
- `workspace/scripts/training/Qwen2.5-0.5B-Instruct/rlve.sh:14` defaulted
  `RUN_NAME` to `rlve_task_019`, which labels the agent's own wandb group, run
  directory and log paths. Now
  `${2:-${BENCH_TASK_LEAF_ID:-__BENCH_TASK_ID__}}`, matching the convention used
  by the verl tasks; no harness code sets `BENCH_TASK_LEAF_ID`, so a fresh run
  emits the placeholder.

Not a leak, deliberately kept: `lr_warmup_iters: 2000` in
`historical_evidence/{manifest,hparams_effective}.json`. `rlve.sh` itself passes
`--lr-warmup-iters 2000`, so the frozen hparams only restate a visible config
value -- that is the evidence the issue level expects the agent to find.

Open: `convert_rlve_hyper.py` re-copies `rlve.sh` and `environment/` from the
upstream `Benchmark_Evaluation` tree, which is not in this repo. A full
re-conversion will reintroduce the `RUN_NAME` default and the `prepare.sh`
header; only the `source_run` drop is handled by the converter itself.
Low-risk but worth knowing: `ptdbench/evaluation/harbor_evaluator.py` writes the
`metadata.yaml` title (`rlve_hyper hyperparameter diagnosis: task_019`) into
`task.toml`'s `[task] description`. Harbor builds the agent prompt from
`instruction.md`, not from the description, so this was not treated as an
in-container leak -- but it is a harness-level surface shared by every task.

## External dataset asset

- Repository: `LIF1014/ptdbench-rlve-hyper-task-019-dataset`
- Publication status: `published`
- Pinned revision: `235103f0bdca99ca0c86da0b79ac1de92acb52c5`
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
