# llama_dapo_implementation implementation diagnosis: task_monkey_patch_011

## Case overview

- Evaluation key: `verl_implementation/llama_dapo_implementation/task_monkey_patch_011`
- Case family: `llama_dapo_implementation`
- Schema version: `2`
- Evaluation levels: `issue`, `solution`
- Model: `Llama-3.2-1B-Instruct`
- Dataset: `DAPO-Math-17k`
- Training framework: `verl`
- Primary source evaluation metric: `val-core/math_dapo/acc/mean@1`

This document is for maintainers and reviewers. It is not exposed to the
agent during evaluation. Metric names preserve the source experiment;
tasks using the same held-out protocol may intentionally share a name.

## Canonical failure reason

A rewritten apply_monkey_patch no longer installs the baseline flash-attention hook, cutting val-core 43% below control with 70.5% zero-advantage steps, while `TRAIN_BATCH_SIZE=4` with sparse ~3–5% 0/1 rewards yields mostly all-incorrect GRPO groups and zero advantage

## Workspace and hidden evaluations

| Path | Visibility and purpose |
| --- | --- |
| `workspace/` | Agent-visible historical experiment snapshot; evidence rather than canonical ground truth. |
| `evaluations/issue/` | Neutral diagnosis instruction, issue oracle, and task-local issue verifier. |
| `evaluations/solution/` | Repair-and-run instruction, hidden reference material, and metric-only verifier. |

The agent cannot read oracle or verifier paths. `issue` grading compares
`$PTDBENCH_RESULTS/agent_answer.md` to hidden `oracle.md` with an LLM
judge. `solution` grading reads only the new experiment logs.

### Leaf task id must not reach the agent (fixed 2026-10-15)

The leaf id `task_monkey_patch_011` names the defect file
(`verl/models/transformers/monkey_patch.py`), so any occurrence on an
agent-visible surface hands the agent the answer. Two surfaces carried it.

**`workspace/` → `/workspace`.** 16 files / 131 occurrences: the frozen
training log, every `wandb/` `config.yaml` / `output.log` / `debug.log`, the
hydra configs under `outputs/2026-07-20/`, and five `outputs/*.json` receipts
— all carrying `trainer.experiment_name=task_monkey_patch_011` or a source
path containing it. All rewritten to `__BENCH_TASK_ID__`.

`__BENCH_TASK_ID__` is the right token, not an arbitrary placeholder: the
launch script defaults to
`EXPERIMENT_NAME=${EXPERIMENT_NAME:-${BENCH_TASK_LEAF_ID:-__BENCH_TASK_ID__}}`
and no harness code sets `BENCH_TASK_LEAF_ID`, so a fresh agent run emits
exactly that string. The snapshot now matches what the agent's own run
produces.

**`environment/` → `/ptdbench/environment`.** Mounted read-only for the agent
stage by design (`PTDBENCH_ENVIRONMENT_DIR` in `ptdbench/core/types.py`;
`ptdbench/agents/base.py` states the agent seeing `prepare.sh` is intentional,
because it describes the environment and "is not part of the answer"). This
task's `prepare.sh` broke that contract in its comments: the leaf task path in
the header, the legacy HF dataset repo name
`…-task-monkey-patch-011-dataset`, and — worse than the name — a block quoting
`evaluations/solution/verifier/resources/grader_spec.json`, its
`historical_final_baseline`, and the `required_final_score "0.07"` pass bar.
All of it removed; the script's executable content is byte-identical (verified
by diffing non-comment lines). The header now carries the rule for future
edits.

Audited surfaces and their container paths: `workspace/` → `/workspace`,
`environment/` → `/ptdbench/environment`, `evaluations/<level>/instruction.md`
→ `/task/instruction.md` (the only file copied into `visible_task`). The
`/ptdbench/{prepared,models,datasets}` mounts are keyed by HF repo and dataset
recipe name, neither of which encodes the task id, and the container is named
`ptdbench-<stage>-<uuid>`.

Re-check after any regeneration:

```bash
grep -rIilE "monkey_patch_011|monkey-patch-011|required_final_score|grader_spec" \
  workspace environment evaluations/*/instruction.md
```

must return nothing. Two near-misses are expected and must stay: the symbol
`apply_monkey_patch` is real verl source, and `0.04022346368715084` is the
historical run's own logged metric, i.e. the evidence the agent is meant to
read.

Not changed, harness-level: `PTDBenchEvaluator._run_directory_prefix` puts the
task id in the **host** run directory name. Invisible under the Docker backend
(the agent sees `/workspace`, `/task`, `/results`), but it would be visible
under the local backend — which this task does not use.

### Archived from `prepare.sh`

Moved here because it is maintainer-only. The dataset recipe is shared with
sibling DAPO-Math tasks (`docker/ptdbench_datasets/dapo_math_17k.py`); upstream
parquet is 17917 problems replicated 100×.

Superseded legacy artifact, kept for provenance: the legacy Dockerfile pulled
`artifacts/workspace-assets.tar.gz` from the HF dataset repo named in the
"External dataset asset" section below, revision
`c3a84bacaadea32cc7a2b9d81162678ac2cb684c`, sha256
`e6ff5139b9fd4a4c9cca8d29d1d50faaea2220098e9d97829a737fb50fd314db` (5676512
bytes), unpacked to `/opt/ptdbench-assets` and hydrated to `/workspace` by
`hydrate-assets.sh`. That tarball was never downloaded locally and its contents
were never verified.

`prepare.sh` also carried a 2026-09-13 note that the grader had to be
recomputed, because `historical_final_baseline` was 72/**1790** from a test
split the current recipe no longer reproduces. That is now done — see the
calibration section below.

## Source and preserved context

The following source and historical packaging fields were moved from
`metadata.yaml` so the machine metadata remains aligned with the core
schema. They are preserved here without changing their meaning.

```yaml
bench_category: llama_dapo_implementation
source_task: Benchmark_Evaluation/tasks/task_llama_dapo_math_17k/task_monkey_patch_011
evidence_kind: dirty_agent_failure
packaging_tier: B0
packaging_ready: true
b1_scoring_ready: false
target_files: verl/models/transformers/monkey_patch.py
validation_stack: DAPO_V4_LLAMA
already_at_baseline: false
metric_status: log_reconstructed
verify:
  num_gpus: 2
  timeout_minutes: 90
  max_train_steps: 200
  eval_metric_key: val-core/math_dapo/acc/mean@1
historical_baseline:
  actual_metric_key: val-core/math_dapo/acc/mean@1
  eval_last: 0.04022346368715084
  eval_max: 0.0446927374301676
  n_eval_points: 11
  attempt: 1
```

## Original failure evidence

- val-core last-3 mean 0.0389; 43% below same-cluster zero-diff control 0.0687
- actor/entropy late 2.98 (control 1.36); policy barely converges
- response_length/mean 376 (control 563)
- zero-advantage steps 70.5% (control 56%); rewarded steps 29.5% (control 44%)
- `TRAIN_BATCH_SIZE=4`, `ROLLOUT_N=8`, 200 steps; held-out `val-core/math_dapo/acc/mean@1` stays around 3–4% (final 0.0402) while most training steps have zero gradient (59/200 nonzero `actor/grad_norm`)

The preserved source metric is `val-core/math_dapo/acc/mean@1`. The frozen workspace remains
the evidence package; a full training rerun is outside verifier scope.

## Healthy protocol baseline

A healthy run of Llama-3.2-1B-Instruct on DAPO-Math-17k under the same
verl/GRPO protocol was reported to reach about **8.4% at 200 steps** and
**11.8% at 1000 steps** on the held-out metric. This case's budget is 200
steps; the frozen last-3 mean 0.0389 is well below the 8.4% figure. The
0.0687 figure above is a same-cluster zero-diff control, not that protocol
baseline.

Both figures are inherited from the source experiment and are **not**
reproduced on the current stack: the 2026-10-03 oracle repair on devspace
`1177503` reaches 7.36% at step 200 (peak 7.47%). The 2026-10-15 recalibration
below uses that measured run, not the 8.4% claim. Do not derive thresholds
from this section.

## Canonical expected outcome

Verdict: **unhealthy**.

Root cause targets:

- `verl/models/transformers/monkey_patch.py::apply_monkey_patch`
- `examples/grpo_trainer/run_llama3_2_1b_instruct_dapo_math_fsdp.sh::TRAIN_BATCH_SIZE`

Required fix targets:

- `verl/models/transformers/monkey_patch.py::apply_monkey_patch`: `restore`; target state/value: the complete baseline apply_monkey_patch, including the flash-attention hook used when use_remove_padding or SP>1. **This one is sufficient for the `solution` level on its own** — the calibration run that clears the bar applies only this fix.
- `examples/grpo_trainer/run_llama3_2_1b_instruct_dapo_math_fsdp.sh::TRAIN_BATCH_SIZE`: `increase`; target state/value: a larger per-update prompt batch (for example 8) so mixed-reward GRPO groups are less rare. Optional: a valid `issue`-level attribution, never exercised by a calibration run, and not required to pass `solution`.

`repeat_kv` layout is not a scored cause. This run uses SP=1, so that helper is not on the executed path.

Documentation reconciliation:
- Frozen evidence: `TRAIN_BATCH_SIZE=4`, `ROLLOUT_N=8`, 200 steps; held-out `val-core/math_dapo/acc/mean@1` stays around 3–4% (final 0.0402) while most training steps have zero gradient.
- This batch-size defect is independently sufficient to explain the unhealthy held-out curve: GRPO needs within-group reward diversity, and 4 prompts at ~5% base accuracy often produce all-incorrect groups.
- Either listed root cause is a correct attribution.

## Evaluation design

`issue` grades `$PTDBENCH_RESULTS/agent_answer.md` against hidden
`oracle.md`. The judge scores verdict, root-cause, and evidence agreement
(weights 0.2 / 0.6 / 0.2). Pass is a weighted score of at least `0.8`.
Either listed root cause is independently sufficient; naming one with the
correct causal role should receive full `root_cause_score`.

`solution` does not grade a structured diagnosis. The agent must repair the
staged workspace, run the designated launch script, and produce a finite
`val-core/math_dapo/acc/mean@1` at step 200 of at least `0.060` (6.0%).
Only step 200 is graded. Pass is score `1`; otherwise `0`.

### Solution threshold calibration (2026-10-15)

Calibrated from a matched defect/repair pair on devspace `1177503`
(`/user/chenyinghao/harbor-runs/task_monkey_patch_011__stage__env`), same node,
same 2 GPUs, same 200 steps, same `TRAIN_BATCH_SIZE=4`, dataset digests as
pinned by `environment/prepare.sh` (test split 1807 rows). The two runs differ
only in `verl/models/transformers/monkey_patch.py`, edited between them:

| arm | step 200 | peak |
| --- | ---: | ---: |
| staged defect (`..._20261003_030020.log`, no flash-attn hook installed) | 0.04759 | 0.06862 @100 |
| oracle repair (`..._20261003_041643.log`, hook installed) | 0.07360 | 0.07471 @180 |

`0.060` is the midpoint, ≈2.2 SE above the defect arm and ≈2.4 SE below the
repaired arm. The superseded protocol graded the **peak over all steps**
against `0.07`, where the unrepaired workspace reached 0.06862 = 124/1807 —
three correct answers short of passing — and where lengthening the run or
raising `TEST_FREQ` inflated the graded number without any repair.
`historical_final_baseline` moved from `0.04022346368715084` (72/1790, an
obsolete test split) to the defect arm's measured 0.04759 (86/1807).

Open: the repaired arm has one run, so step-200 run-to-run variance is
unmeasured; a repeat of both arms (~29 min each) would firm up the bar.

The repaired run above cleared the bar **without** changing
`TRAIN_BATCH_SIZE`. Restoring `apply_monkey_patch` is therefore the only
remediation the `solution` level requires. `TRAIN_BATCH_SIZE` remains a valid
`issue`-level attribution, and raising it is not penalized, but it is not
calibrated and not required.

## External dataset asset

- Repository: `LIF1014/ptdbench-llama-dapo-implementation-task-monkey-patch-011-dataset`
- Publication status: `published`
- Pinned revision: `c3a84bacaadea32cc7a2b9d81162678ac2cb684c`
- Dataset license: `Apache-2.0`
- Provenance: Processed from BytedTsinghua-SIA/DAPO-Math-17k; task-specific bytes are pinned.

| Runtime path | Size | SHA-256 |
| --- | ---: | --- |
| `workspace/DAPO-Math-17k_verl/test.parquet` | 675,271 bytes | `f860d1c3bfb0ce4a937b5e91bb43549a0a7ded51b1725c58c9c10f9f075d2aeb` |
| `workspace/DAPO-Math-17k_verl/train.parquet` | 5,885,886 bytes | `2897263b99e9b4af7523edeb5ec1ab9cac6e1bba63f0bfa78d4cd9356426ca5b` |

The Docker image downloads the immutable revision at build time, verifies
the artifact, and hydrates the bind-mounted workspace only for the agent stage.

## Environment requirement

- Base image: `verlai/verl:vllm017.latest`
- Use the Docker agent backend after external asset publication.
- The local backend does not build the task image or hydrate external assets.
- Model weights and historical training checkpoints are not dataset assets.
