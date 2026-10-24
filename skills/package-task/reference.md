# Package task reference

## Required inputs

If any item is missing, stop.

### Task location

- Source workspace path
- Target directory: `tasks/<type>/<task_id>/`, where `<type>` is one of the six canonical categories
- `task_id`, title
- `type` (the category, = the directory) and `domain` (free-form label)
- Model name, dataset name, framework, algorithm

`failure_reason` is one sentence written from the filtered diff, including the metric symptoms the user gave.

### Two commits

- git repo path
- buggy commit (the workspace matches this one)
- baseline commit (the expected correct state)
- Unrelated hunks to exclude (leave empty if none)

Attribution command:

```bash
git diff <buggy> <baseline>
```

Keep only code and launch scripts, plus the training settings, hyperparameters, rewards, and data processing they directly reference. Exclude:

- `logs/`, `wandb/`, `outputs/`, `*.log`
- checkpoint, `*.safetensors`, `*.bin`, `*.pt`, `*.ckpt`
- Train and validation data files (parquet, jsonl data, etc.)

### How this training run is launched

- Launch script path relative to `workspace/` (this is `training.script` in `metadata.yaml`)
- Positional args (this is `training.script_args`) and required env vars
- In-container model path
- In-container train set path and validation set path
- Step count and eval interval (when fixed, write them into `solution/instruction.md`)
- Path of the failed run's logs in the source workspace
- Whether the source includes wandb; keep it if so

### Environment and assets

- Environment image name: the `<name>` of `<name>.Dockerfile` under `docker/images/` (run `ls docker/images/` for the current set — at time of writing `verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12`, `slime-v0.3.2-sglang0.5.15-torch2.11.0-flashattn2.8.3-cu129-py3.12`, `slime-v0.5.0rc0-sglang0.5.0rc0-torch2.8.0-flashattn2.7.4-cu126-py3.12`, `search-r1-verl`, `vllm0.11.0-torch2.8.0-flashattn2.8.3-cu128-py3.12`, `rank-distillation-cpu`)
- GPUs to use: `CUDA_VISIBLE_DEVICES` / `NVIDIA_VISIBLE_DEVICES`
- Each HF asset: repo, 40-char commit sha revision, `--type` (`model` or `dataset`), `--allow` list, SHA-256 and byte size of files to verify, `--dest`
- Datasets using a built-in recipe: recipe name (`dapo_math_17k`, `gsm8k`, `search_r1_nq_hotpotqa`, `wiki18_e5`), SHA-256 and byte size of output files, `--dest`
- Single-URL assets: URL, SHA-256, byte size, whether to `--extract`
- Whether `docker-compose.yaml` is needed (multi-GPU or large `/dev/shm`, on the harbor backend)

Do not collect a base image: environment images are defined by `docker/images/*.Dockerfile`, and tasks do not ship a Dockerfile.

### Scoring keys (user input for this task)

The eval-set key is always required:

- Eval-set key name
- Where the key appears: text log, offline wandb, or both
- One real log sample containing the key and a number
- `final_step`
- `historical_final_baseline`
- `required_final_score`
- Optional: `score_mode` (`best` or `final`, default `best`), `required_final_score_max`

The scoring method is optional. If not given, the score depends entirely on the eval-set key.

When the user specifies a scoring method, each reference score needs a key name, threshold, weight, where it appears, and one log sample, plus a pass rule. A reference score can be a training score or another logged value such as learning rate. See [examples.md](examples.md) for a filled-in example.

### Historical evidence

- The failed run's eval point series (step, value), used to write `historical_evidence_manifest.json` and calibrate the threshold

## The shared issue protocol

Start the issue level from the standard set in `tasks/reward/retool_length_hack/evaluations/issue/`. Keep LF line
endings. The judge is task-agnostic (only `oracle.md` is per-task), so:

| Target | Source | Rule |
| --- | --- | --- |
| `evaluations/issue/verifier/judge.py` | `verifier/judge.py` in the reference task | **byte-identical** (`cmp`) |
| `evaluations/issue/verifier/test.sh` | `verifier/test.sh` in the reference task | **byte-identical** (`cmp`) |
| `evaluations/solution/verifier/test.sh` | `tasks/reward/retool_length_hack/evaluations/solution/verifier/test.sh` | **byte-identical** (`cmp`) |
| `evaluations/issue/instruction.md` | `tasks/reward/retool_length_hack/evaluations/issue/instruction.md` | standard text, or a **minor** task-specific adjustment |
| `evaluations/issue/verifier/judge_prompt.md` | `verifier/judge_prompt.md` in the reference task | standard rubric, or a **minor** task-specific adjustment |

`judge.py` + `test.sh` are byte-identical across the LLM-judge tasks (e.g. `infrastructure/torch_functional`,
`environment/search_r1_zero_query_vector`). `instruction.md` is the same generic brief for most tasks; adjust it
only when the task shape differs (e.g. `configuration/aime25_single_sample_variance` is an evaluation-result
diagnosis, not a training run). Keep task specifics in `oracle.md` where possible rather than editing the rubric,
and never fork `judge.py`'s scoring to hardcode one task's answer. A pure metric-decided symptom may instead use a
deterministic `test.sh` (no `judge.py`) — an allowed alternative, not the default.

When needed, copy `environment/docker-compose.yaml` verbatim from `tasks/reward/retool_length_hack/environment/docker-compose.yaml`.

## issue/oracle.md

```markdown
# Canonical issue oracle

Reconciled from the task's frozen evidence and the diff between `<buggy>` and `<baseline>`.

Expected verdict: **unhealthy**.

## Root cause

- `<path-or-symbol>`
  - kind: `<symbol|hparam|script>`
  - observed: `<buggy side>`
  - expected: `<baseline side>`

## Observable impact

- Primary source evaluation metric: `<metric_key>`.
- <symptoms from logs, not listed as separate root causes>

## Trigger conditions

- <trigger conditions from the diff>

## Documentation reconciliation

- <which log observations are symptoms>
```

With more than one Root cause, add at the end of that section:

```text
Listed causes are all essential. Identifying only part of them must receive partial root_cause_score.
```

## issue/verifier/README.md

A fixed description of how this tier is graded, written after `tasks/reward/retool_length_hack/evaluations/issue/verifier/README.md`: the judge reads `$PTDBENCH_RESULTS/agent_answer.md` and compares it with the hidden `oracle.md`, weights 0.2 / 0.6 / 0.2, pass mark 0.8, requires `PTDBENCH_JUDGE_MODEL`, `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and writes the result to `$PTDBENCH_RESULTS/result.json`.

## issue/verifier/gold.yaml (optional)

For reviewers only; machine grading does not read it. If written, it must match `oracle.md`:

```yaml
category: <implementation|hyperparameter|reward>
framework: <framework>
task: <task_id>
verdict: <healthy|unhealthy>
root_cause:
  - <one root cause, matching Root cause in oracle.md>
evidence_metrics:
  - <one piece of observable evidence>
fix:
  must:
    - <required fix action>
```

## solution/instruction.md

From the start through "The dataset, held-out set, ... are fixed", reuse `tasks/reward/retool_length_hack/evaluations/solution/instruction.md`. After that, replace with this task's user input:

```markdown
The launch script is:

```text
<launch-script-relative-path>
```

<user-supplied args or example invocation>

The effective run must use:

- model `<in-container model path>`
- train data `<train set path>`
- validation data `<validation set path>`
- <fixed step count and eval interval, e.g. `--num-rollout 100`, `--eval-interval 10`>
- `LOG_DIR=$PTDBENCH_RESULTS/experiment_logs`
- `WANDB_MODE=offline`
- `WANDB_DIR=$PTDBENCH_RESULTS/experiment_logs`

Create `$PTDBENCH_RESULTS/experiment_logs` first. Offline W&B data must be
created below `$PTDBENCH_RESULTS/experiment_logs/wandb`. The run is scored only
from files under `$PTDBENCH_RESULTS/experiment_logs`; logs written anywhere
else are ignored. Do not copy, edit, or fabricate log metrics. Otherwise the
run is scored 0.
```

## solution/oracle/README.md

Fill in from the same filtered diff. The machine grading paragraph only repeats this task's user-given keys, thresholds, and weights.

```markdown
# Canonical solution oracle

## Root cause

- `<path-or-symbol>`
  - kind: `<kind>`
  - observed: `<buggy>`
  - expected: `<baseline>`

## Observable impact

- Primary source evaluation metric: `<metric_key>`.

## Trigger conditions

- <same as issue oracle>

## Required remediation

- `<path>`: revert to the baseline's `<expected>`.

Machine grading reads only the new experiment logs.
<state keys, thresholds, weights, and pass rule per grader_spec>
```

## solution/oracle/task.yaml

```yaml
task_id: <task_id>
category: <family or recipe category>
framework: <framework>
launch_script: <launch script relative to workspace/>
model: <model name>
dataset: <dataset name>
algo: <algo>
expected_verdict: <healthy|unhealthy>
evidence_kind: <historical_run|dirty_agent_failure>
buggy_commit: <sha>
baseline_commit: <sha>
eval_metric_key: <metric_key>
last_eval_step: <final_step>
causes:
  - path: <repo-relative path>
    kind: <symbol|hparam|script>
    observed: <buggy>
    expected: <baseline>
key_files:
  - <files touched by the diff>
external_assets_not_in_git:
  - <asset delivered by prepare.sh, one-line description>
```

When the same framework already has a more complete `task.yaml` (e.g. `infrastructure/torch_functional`), reuse that skeleton, but still write in every root cause from this diff.

## solution/oracle/standard_answer.txt

The judge's hard criteria, plain text, with a fixed header:

```text
STANDARD_ANSWER_V1
CASE_ID: <task_id>
BENCHMARK_GROUP: <B-...>
EXPECTED_VERDICT: <healthy|unhealthy>
EXPECTED_ATTRIBUTION: <one sentence, the root cause the diff points to; for healthy, write none and the reason>
PRIMARY_METRIC: <metric_key>

REQUIRED_ALL:
- <each item that must be named; list each root cause separately>

GLOBAL_JUDGE_POLICY:
- The score must be exactly 0 or 1.
- Agent A must explicitly classify the run as healthy or unhealthy.
- Every REQUIRED_ALL item must be satisfied.
- A verdict mismatch forces score 0.
- Equivalent technical wording in any language is acceptable.
- task.yaml may clarify targets and aliases but may not relax REQUIRED_ALL.
```

`healthy` tasks add one more rule: reporting a benign but unusual setting as a defect scores 0.

## solution/oracle/historical_evidence_manifest.json

The failed run's metric series, which reviewers use to recheck the threshold:

```json
{
  "schema_version": 1,
  "evidence_kind": "dirty_agent_failure",
  "task_id": "<task_id>",
  "family": "<family>",
  "metrics": {
    "metric_status": "log_reconstructed",
    "expected_metric_key": "<metric_key>",
    "actual_metric_key": "<metric_key>",
    "eval_points": [
      { "step": 0, "eval_score": 0.0307, "eval_key": "<metric_key>", "truncate_ratio": null }
    ]
  }
}
```

## grader_spec.json

```json
{
  "schema_version": 2,
  "task_id": "<task_id>",
  "metric_key": "<user-supplied eval key>",
  "final_step": 0,
  "historical_final_baseline": "<user-supplied value>",
  "required_final_score": "<user-supplied value>",
  "log_roots": ["experiment_logs"]
}
```

Optional keys: `score_mode` (`best` or `final`; `grade_submission.py` defaults to `best`), `required_final_score_max` (an upper bound that rejects absurd fabricated values). Write all numbers as strings; they are read with `Decimal`.

If no scoring method is specified, do not add other score keys. When the user specifies reference scores, add them to the same `grader_spec.json` as a **flat triple**; do not use `components` / `pass_rule`:

```json
{
  "training_reward_key": "<reference score key>",
  "required_training_reward": "<threshold>",
  "validation_weight": "0.8",
  "training_weight": "0.2"
}
```

Common naming sets: `training_reward_key` / `required_training_reward` / `training_weight`, and `learning_rate_key` / `required_learning_rate` / `learning_rate_weight`. The eval-set side is always `validation_weight`. Name keys after what the reference score is, and list them all in the required-keys list in `grade_submission.py`.

`grade_submission.py` reads the sibling `../resources/grader_spec.json`. Self-test with the sample string: it must extract the user-given key and the number in the sample. Outputs:

- `experiment_metrics.json`
- `metrics.json`
- `result.json`: `{"passed": <bool>, "score": <number>, "metrics": {...}}`

`resources/README.md` states that the entry point is `test.sh`, that scores come only from `$PTDBENCH_RESULTS/experiment_logs`, and lists the same keys and thresholds as the spec. `verifier/README.md` adds one more layer: which space the key reads (score or accuracy), why `score_mode` was chosen, which points on the historical curve the threshold was calibrated against, and the thinnest margin.

## Environment image

`environment.image` in `metadata.yaml` is the `<name>` of some `<name>.Dockerfile` under `docker/images/`. The image tag comes from `# ptdbench-image:` at the top of the Dockerfile; if it is not available locally, it is built first (or prebuild with `python3 -m ptdbench build-images <name>`).

Images are shared across tasks, and their contents are those tasks' common ground-truth environment. Do not modify an existing Dockerfile for a new task; that would also change the other tasks' environments. If no existing image fits, create a new `docker/images/<name>.Dockerfile`: pin `FROM` to a digest, and at the end copy verbatim the whole block after `# ---- PTDBench tooling ...` from another Dockerfile (tests compare it character by character). See `docker/README.md`.

The framework itself (verl / slime) does not count as part of the image: it comes from the checkout in the task workspace.

## environment/prepare.sh

```bash
#!/usr/bin/env bash
# <task_id>: what to prepare before training.
#
# Runs in a separate prepare-stage container: root, the `<image-name>` image, **the only
# networked stage**, and $PTDBENCH_WORKSPACE, $PTDBENCH_MODELS, and $PTDBENCH_DATASETS are all writable.
set -euo pipefail

# --- Base model ---------------------------------------------------------------
# <why dest must be exactly this path: which guard on which line of the launch script looks for it>
ptdbench_fetch_hf \
    --repo <org/repo> \
    --revision <40-char commit sha> \
    --allow config.json \
    --allow tokenizer.json \
    --allow model.safetensors \
    --expect model.safetensors=<sha256> \
    --expect-size model.safetensors=<bytes> \
    --dest "$PTDBENCH_WORKSPACE/<relative model path>"

# --- Dataset ------------------------------------------------------------------
ptdbench_dataset \
    --recipe <dapo_math_17k|gsm8k|search_r1_nq_hotpotqa|wiki18_e5> \
    --expect train.parquet=<sha256> \
    --expect-size train.parquet=<bytes> \
    --expect test.parquet=<sha256> \
    --expect-size test.parquet=<bytes> \
    --dest "$PTDBENCH_WORKSPACE/<relative dataset path>"
```

Three commands:

| Command | Purpose | Required |
| --- | --- | --- |
| `ptdbench_fetch_hf` | Fetch an HF repo by allow list | `--repo`, `--revision` (40-char sha), `--dest`; `--type dataset` for dataset repos |
| `ptdbench_fetch_url` | Single URL | `--url`, `--sha256`, `--dest`; add `--extract` for archives |
| `ptdbench_dataset` | Recipe built into the image | `--recipe`, `--dest` |

All three accept repeatable `--expect PATH=SHA256` and `--expect-size PATH=BYTES`. Verify at least one main file per asset. A failed check is a **hard failure**, with no fallback retry.

Available variables:

| Name | What it is |
| --- | --- |
| `$PTDBENCH_WORKSPACE` | The workspace the agent receives, writable during prepare; assets go here |
| `$PTDBENCH_PREPARED` | Output directory; writing `KEY=VALUE` lines to `$PTDBENCH_PREPARED/env` passes env vars to the agent/verifier (read line by line as data, never sourced) |
| `$PTDBENCH_MODELS` / `$PTDBENCH_DATASETS` | Download caches shared across runs, read-only in the agent/verifier stages; **task scripts must not reference them directly**, since the harbor backend deletes them after prepare |
| `$PTDBENCH_ENVIRONMENT` | The task's own `environment/` (read-only) |

When prepare needs a format conversion (e.g. HF → Megatron torch_dist), probe for CUDA first and, if unavailable, **skip instead of erroring**. The harbor backend runs prepare as a build-time layer with no GPU, so the training script's own guard does it in the agent stage.

## metadata.yaml

The task card is **`metadata.yaml`** (YAML, not TOML — there is no `task.toml`).
`ptdbench/tasks/validator.py::validate_metadata` requires every top-level key below, so none may be omitted.

```yaml
title: <title on the website task card>
submitter:
  name: <name>
license: MIT
created_date: "<YYYY-MM-DD>"      # quote it; a bare date is parsed as a YAML date, not a string
type: <configuration|reward|data|environment|algorithm|infrastructure>
domain: <free-form label, e.g. post_train / math_tool_use / the type again>
model:
  name: <model name>
dataset:
  name: <dataset name>
training:
  framework: <framework>
  algo: <algo>
  script: <launch script path relative to workspace/>
  script_args: ["<arg>"]          # only when the launch script takes positional args
environment:
  image: <environment name under docker/images/>
  env:                            # optional; values must be strings
    CUDA_VISIBLE_DEVICES: "0,1"
    NVIDIA_VISIBLE_DEVICES: "0,1"
outcome:
  failure_type: <see below>
  failure_reason: <one sentence from the filtered diff, with the user-given metric symptoms>
```

`type` vs `domain`: **`type`** is one of the six canonical categories and MUST equal the directory the task
lives under (`tasks/<type>/<id>`) — the validator rejects a mismatch. **`domain`** is a free-form descriptive
label (existing tasks use `post_train`, `math_tool_use`, …); set it to whatever best describes the task, or
repeat the `type` when nothing more specific fits. There is no `tags` field.

`failure_type` only needs to be non-empty. Values used by existing tasks: `reward_hacking`, `evaluation_under_sampling`,
`data_difficulty_mismatch`, `retriever_query_device_copy`, `training_collapse`, `implementation_bug`.

The validator rejects:

- Any missing required top-level key (`title`, `submitter`, `license`, `created_date`, `type`, `domain`,
  `model`, `dataset`, `training`, `environment`, `outcome`)
- `type` that is not one of the six categories, or that does not match the task's `tasks/<type>/<id>` directory
- Missing `image`, or `image` with no matching `<name>.Dockerfile` under `docker/images/`
- Legacy fields `conda` / `requirements` / `dockerfile`
- `script` that is absolute, contains `..`, or points to a file not in `workspace/`
- `script_args` without `script`; `script_args` that is not a list of strings
- `env` values that are not strings (`CUDA_VISIBLE_DEVICES: 0` is rejected), or that set `PTDBENCH_*`

Declaring `framework` without `script` only gives a warning, but the task card relies on it to record how the task runs, so do not omit it.

## readme.md

Write it per [readme-template.md](readme-template.md). The file is `readme.md` (lowercase); the uppercase
`README.md` files under `evaluations/*/verifier/`, `evaluations/solution/oracle/`, and `docker/` are different
files and stay uppercase.
