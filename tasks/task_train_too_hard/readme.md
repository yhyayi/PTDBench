# data diagnosis: task_train_too_hard

## Case overview

- Evaluation key: `task_train_too_hard`
- Case family: `data`
- Schema version: `2`
- Evaluation levels: `issue`
- Model: `Qwen2.5-1.5B-Instruct`
- Training framework: `verl`
- Primary source evaluation metric: `val-core/taco/acc/mean@1`

This document is for maintainers and reviewers. It is not exposed to the
agent during evaluation.

There is no solution level. Issue grading compares
`$PTDBENCH_RESULTS/agent_answer.md` to hidden `oracle.md`.

## Canonical failure reason

The training set is too hard, and its difficulty does not match the validation
set. `workspace/taco_verl/train.parquet` contains only TACO `VERY_HARD` rows.
`workspace/taco_verl/test.parquet` is the existing TACO `EASY` validation set.

## Workspace and hidden evaluations

| Path                 | Visibility and purpose                                              |
| -------------------- | ------------------------------------------------------------------- |
| `workspace/`         | Agent-visible experiment snapshot. Code is the healthy baseline.   |
| `environment/`       | Agent-visible, read-only at `/ptdbench/environment`; `prepare.sh` fetches the model and dataset. |
| `evaluations/issue/` | Neutral diagnosis instruction, issue oracle, and issue verifier.   |

The agent cannot read oracle or verifier paths.

## Canonical expected outcome

Verdict: **unhealthy**.

Essential root cause, either formulation:

- the training set is too hard
- the training set and the validation set do not match in difficulty

Code, launch hyperparameters, and the model are not the root cause.

## Historical training run

Log: `workspace/logs/qwen2_5_1.5b_instruct_taco_fsdp_20261010_073442.log`.
Wandb: `workspace/logs/wandb/offline-run-20261010_073718-kovw4pvq`.

The run finished `400/400` steps. The progress bar records `8:40:39`
(about 8 hours 41 minutes, 78.10 seconds per step). The train dataloader has
145 steps per epoch and `total_training_steps` is 400, so the step cap stops
the job during the third epoch. Epoch index 0 covers steps 1–145, epoch index
1 starts at step 146, and epoch index 2 starts at step 291.

Held-out `val-core/taco/acc/mean@1` on 184 TACO `EASY` rows. Counts are
`round(acc * 184)`.

| Step | Fully correct | acc | pass_rate |
| ---- | ------------- | --- | --------- |
| 0 | 9/184 | 0.0489 | 0.1967 |
| 50 | 18/184 | 0.0978 | 0.2887 |
| 100 | 17/184 | 0.0924 | 0.2918 |
| 150 | 16/184 | 0.0870 | 0.3050 |
| 200 | 6/184 | 0.0326 | 0.1167 |
| 250 | 1/184 | 0.0054 | 0.0649 |
| 300 | 4/184 | 0.0217 | 0.1149 |
| 350 | 5/184 | 0.0272 | 0.1433 |
| 400 | 7/184 | 0.0380 | 0.1351 |

The last validation accuracy, 0.0380 (7/184), is below the step-0 value,
0.0489 (9/184). The peak is step 50 at 0.0978 (18/184). After step 200 the
curve stays below the step-0 baseline.

`critic/rewards/mean` over training steps 1–400 averages 0.0938. Steps 1–100
average about 0.0082. Of the 400 steps, 96 are 0. The maximum is 0.4375 at
step 299. Step 400 is 0.1250. Fifty-step means:

| Steps | Mean training reward |
| ----- | -------------------- |
| 1–50 | 0.0081 |
| 51–100 | 0.0083 |
| 101–150 | 0.0188 |
| 151–200 | 0.0934 |
| 201–250 | 0.1789 |
| 251–300 | 0.1467 |
| 301–350 | 0.1470 |
| 351–400 | 0.1495 |

The 101–150 mean is exactly 0.01875. From step 201 onward the fifty-step
means stay near 0.15 while held-out accuracy does not recover.

## Evaluation design

`issue` grades `$PTDBENCH_RESULTS/agent_answer.md` against hidden `oracle.md`.
The judge scores verdict, root-cause, and evidence agreement
(weights 0.2 / 0.6 / 0.2). Pass is a weighted score of at least `0.8`.

Root-cause credit is full when the diagnosis states that the training set is
too hard, or that the training and validation sets do not match in difficulty.
Either formulation is enough. A code defect is not required.

## External dataset asset

- Repository: `LIF1014/ptdbench-data-task-train-too-hard-dataset`
- Publication status: `published`
- Pinned revision: `22a1fee0ea2527f514c4115f6c792216a99d7e6f`
- Dataset license: `Apache-2.0`
- Provenance: TACO `VERY_HARD` training rows and the existing TACO `EASY` validation snapshot.


| Runtime path                            | Size              | SHA-256                                                            |
| --------------------------------------- | ----------------- | ------------------------------------------------------------------ |
| `workspace/taco_verl/test.parquet`      | 61,082,426 bytes  | `0593078d24134d204b305dd0373acfc33daed9a2c73fdad7304a2fde191daea3` |
| `workspace/taco_verl/train.parquet`     | 168,834,942 bytes | `8640b3e6c7a5ff3d5b51aa46e74f74147f3c565cd89b73676d5113f8e546d573` |


The archive `artifacts/workspace-assets.tar.gz` is 167,612,568 bytes, SHA-256
`eddeece8c2828e21b581d5168dd4604470165cb061a53e68d50b26d49ce6d916`.
It contains 2329 `VERY_HARD` training rows, all `stdin`, and the unchanged
184-row `EASY` validation set. `VERY_HARD` has no call-based problems. The
converter was checked against the existing EASY `train.parquet`: 5996 `stdin`
rows and 2372 `call_based` rows matched. Rows above 2048 prompt tokens are
dropped; 4 `VERY_HARD` rows were dropped that way.

The agent-visible parquet files and this archive revision do not store
`extra_info.difficulty`. Prompts, `reward_model`, `index`, `split`,
`code_type`, and `fn_name` are unchanged. `environment/prepare.sh` downloads
this revision in the prepare stage, verifies the archive, unpacks
`workspace/taco_verl/` into `$PTDBENCH_WORKSPACE/taco_verl`, and re-verifies the
two parquet files after unpacking.

## External model asset

- Repository: `LIF1014/ptdbench-verl-implementation-torch-functional`
- Publication status: `published`
- Pinned revision: `36a4c226ef70aef2c38df173a7cf86a47ad3ad00`
- Model: `Qwen2.5-1.5B-Instruct` (`model.safetensors` SHA-256 `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`)

This snapshot repository is shared with `verl_implementation/torch_functional`
and is not specific to this task. `environment/prepare.sh` fetches it to
`$PTDBENCH_WORKSPACE/Qwen2.5-1.5B-Instruct`, the `MODEL_PATH` default of the
launch script.

## Environment requirement

- `metadata.environment.image: verl` resolves to the shared image
  `docker/images/verl.Dockerfile` (`verlai/verl:vllm017.latest` pinned by
  digest), the same image the other verl + TACO tasks use.
- `environment/prepare.sh` is the only networked stage. It fetches the model
  and the dataset archive recorded above; nothing is baked into an image.
- Launch script: `examples/grpo_trainer/run_qwen2_5_1.5b_instruct_taco_fsdp.sh`
- The designated experiment uses 8 GPUs and a 400-step budget.
- Issue evaluation does not launch training.

**No GPU declaration, on purpose.** `metadata.environment.env` is empty, unlike
the sibling verl tasks. This task has only an `issue` level: the agent reads a
historical snapshot and the verifier judges `agent_answer.md`, so neither stage
runs the training stack. Declaring `NVIDIA_VISIBLE_DEVICES` would make the
docker runtime pass `--gpus` to the agent and verifier containers
(`ptdbench/runtimes/providers/docker.py::_gpu_args`), which fails outright on a
GPU-less host. To re-run the recorded 8-GPU experiment, set
`PTDBENCH_DOCKER_GPUS` on the host; a Harbor-side rerun additionally needs the
GPU/shm compose override that the sibling tasks ship as
`environment/docker-compose.yaml`. Add both here if a `solution` level is ever
added.

**`pyext` is not installed.** `verl/utils/reward_score/taco.py` tries
`from pyext import RuntimeModule` and falls back to its own complete
`_FallbackRuntimeModule` on `ImportError`, and the stdin path every row takes
goes through it, so the choice reaches the reward. The frozen dependency list of
the recorded run (`workspace/logs/wandb/*/files/requirements.txt`) does not list
`pyext`, and that list — not the legacy Dockerfile — is the reference for
reproducing this run. `reward_design/task_evaluator` installs it because its own
frozen list has `pyext==0.7`; to do the same here, copy the vendored-install
block from that task's `prepare.sh`. Do not edit `docker/images/verl.Dockerfile`
— that image is shared by every verl task.

### Notes moved out of `environment/prepare.sh`

`prepare.sh` is mounted read-only into the agent container at
`/ptdbench/environment`, so it must not carry maintainer background that names
the defect or the grading thresholds. Those notes live here. Keep it that way.

**Migration.** `prepare.sh` replaced `environment/{Dockerfile,hydrate-assets.sh}`
on 2026-10-16, matching the house scheme the other tasks moved to on 2026-09-23.
The legacy Dockerfile built a task-specific image from `verlai/verl:vllm017.latest`
that installed a patched `pyext`, downloaded the model and the dataset archive at
build time, and hydrated `/workspace` from `/opt/ptdbench-assets` in the agent
stage. Repositories, revisions, byte counts and SHA-256 values carried over
verbatim; the hydration target paths are unchanged. The archive layout was
re-checked against the published revision: it holds exactly
`workspace/taco_verl/train.parquet` and `workspace/taco_verl/test.parquet`.

**Task id leak.** Both the task directory name and the published dataset
repository name (`...-task-train-too-hard-dataset`) state the answer, and the
agent sees both: `PTDBENCH_TASK_ID` is exported into the agent stage
(`ptdbench/runtimes/base.py::standard_env`) and `prepare.sh` is readable at
`/ptdbench/environment`. Closing this needs a task rename plus a republished
dataset repository under a neutral name, so it is recorded here rather than
worked around in `prepare.sh`.
