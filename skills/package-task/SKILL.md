---
name: package-task
description: >-
  Packages a buggy RL training workspace (a raw task, a git commit pair, or a
  historical snapshot) into a PTDBench task under tasks/<type>/<task_id>/:
  metadata.yaml, readme.md, workspace/, environment/prepare.sh, and issue/solution
  evaluations. Use when the user asks to package, build, or convert a run into
  the task format, for any model, framework, or dataset.
---

# package-task

Packages one faulty RL training run into `tasks/<type>/<task_id>/`. `<type>` is one of
`configuration`, `reward`, `data`, `environment`, `algorithm`, `infrastructure` — the category, which is both
the directory and the required `type` field in `metadata.yaml`. (The separate `domain` field is a free-form
label and is not constrained to these six.)

| File | Contents |
| --- | --- |
| [SKILL.md](SKILL.md) | Workflow and rules (this file) |
| [reference.md](reference.md) | Template for every output file |
| [examples.md](examples.md) | A filled-in user input |
| [readme-template.md](readme-template.md) | Template and style guide for the task `readme.md` |

Authoritative sources in the repo: `ptdbench/tasks/validator.py` (validation rules), `docker/README.md` (environment images).
For finished tasks, see `tasks/reward/retool_length_hack/` (issue + solution) and
`tasks/data/task_train_too_hard_new/` (issue only).

## Outputs

```text
tasks/<type>/<task_id>/
├── readme.md                     # for reviewers, not visible to the agent
├── metadata.yaml                 # task card: metadata + environment
├── workspace/                    # defective side's code, launch script, and frozen evidence
├── environment/
│   ├── prepare.sh                # REQUIRED (may be empty): the only networked stage: fetch model, fetch data
│   ├── docker-compose.yaml       # optional: multi-GPU / large shm on Harbor
│   └── data_manifest.json        # optional: when data ships with the task
└── evaluations/
    ├── issue/                    # diagnosis
    │   ├── instruction.md
    │   ├── oracle.md
    │   └── verifier/{test.sh,judge.py,judge_prompt.md}
    └── solution/                 # optional: fix and rerun
        ├── instruction.md
        ├── oracle/
        └── verifier/{test.sh,tests/grade_submission.py,resources/grader_spec.json}
```

Do not produce `agent_trace.json`, `environment/Dockerfile`, `requirements.txt`, or `validation/`.
Tasks do not ship their own image: `environment.image` in `metadata.yaml` points to `docker/images/<name>.Dockerfile`.
`environment/prepare.sh` must always exist (the harness looks for it at that path); when the task needs no
networked fetch it is a minimal no-op (`#!/usr/bin/env bash` + `set -euo pipefail`), not absent.

## Required inputs

If any item is missing, stop and ask. Field list in [reference.md](reference.md#required-inputs), example in [examples.md](examples.md).

- Source (one of three forms, see next section)
- `type` (the category = the `tasks/<type>/` directory), `domain` (free-form label), `task_id`, title; model, dataset, framework, algorithm
- Launch script (relative to `workspace/`), positional args, env vars; in-container paths for model, train set, and validation set
- Logs of the failed run (keep W&B too if present)
- Environment image name, GPUs to use
- For each asset: source, 40-char revision, SHA-256 and byte size, destination
- Solution tier: metric key, one log sample, `final_step`, `required_final_score`, and **how many runs per side** the threshold was calibrated on

## Three source forms

| Form | Where the root cause comes from | Typical |
| --- | --- | --- |
| 1. Two git commits | `git diff <buggy> <baseline>` | Injected defect |
| 2. raw task: `buggy_workspace/` + `fixed_workspace/` | `diff -rq`, then `diff -u` per file | See [CONTRIBUTING.md](../../CONTRIBUTING.md) for format |
| 3. Historical snapshot, no control | The user-supplied `bug_root_cause`; if missing, ask. Do not infer it from logs yourself | `data/task_train_too_hard_new` |

The root cause comes only from code, launch scripts, and the hyperparameters, rewards, and data processing they directly reference.
Differences in logs, W&B, checkpoints, weights, and data files never count as root cause.

Form 2 **publishes only the defective side**. Putting both sides in `workspace/` hands over the answer with a single `diff -r`:

| Item in the raw task | Destination |
| --- | --- |
| Defective-side code + launch script + run artifacts | `workspace/` |
| Changed files from the control side | `evaluations/solution/oracle/` |
| Control-side run results and curves | `evaluations/solution/oracle/` (used to calibrate the threshold) |
| The raw `Dockerfile` | Map to an existing image under `docker/images/`, or add a new one |
| The raw `readme.md` / `README.md` | Rewrite as the task `readme.md` |

## Workflow

1. **Identify the root cause.** Take the diff for the source form and filter out unrelated parts. If nothing substantive remains after filtering, stop.
2. **Copy the evaluation scaffolding.** Start the issue level from the shared standard protocol most tasks use
   (`instruction.md` + `verifier/{test.sh,judge.py,judge_prompt.md}`), copying from
   `tasks/reward/retool_length_hack/`; keep `judge.py` and `test.sh` byte-identical (`cmp`), and adjust only
   `instruction.md` / `judge_prompt.md`, and only minimally, when the task genuinely needs it — see
   [Issue evaluation protocol](#issue-evaluation-protocol). Also copy the solution `verifier/test.sh` verbatim.
3. **Write the oracle.** Use the diff from step 1 to write `evaluations/issue/oracle.md` and `solution/oracle/`.
   When there are multiple root causes, `oracle.md` must state that naming only some of them earns partial credit.
4. **Write the solution tier** (only when there is a control side or a post-fix run): `instruction.md`, `grader_spec.json`,
   `grade_submission.py`. If the log format matches, reuse an existing `grade_submission.py` and change only the spec.
5. **Assemble the workspace.** Add the defective-side code and frozen evidence, slim and sanitize it (see below), then write `environment/prepare.sh`,
   `metadata.yaml`, and `readme.md`.
6. **Offline self-check.** Replay the grader on the log samples. With a control side, replay once per side: the defective side must fail and the control side must pass.
7. **Validate.** `python3 -m ptdbench validate tasks/<type>/<task_id>` must report no errors.
   On a machine with Docker and network, also run `./docker/warm-cache.sh <task_id>` to confirm `prepare.sh` runs end to end.

## Rules

### Issue evaluation protocol

Most tasks share one standard issue protocol, and new tasks should follow it rather than invent their own. It is
an LLM judge over a written diagnosis:

- `instruction.md` — the generic "diagnose this snapshot as healthy/unhealthy, write `agent_answer.md`" brief.
- `verifier/test.sh` — a thin wrapper that execs `judge.py`.
- `verifier/judge.py` — reads `oracle.md` + the agent's `agent_answer.md`, scores verdict / root cause /
  evidence (0.2 / 0.6 / 0.2) with the model named by `$PTDBENCH_JUDGE_MODEL`, pass mark 0.8, writes `result.json`.
- `verifier/judge_prompt.md` — the rubric.

These four files are **task-agnostic**: the only per-task file is `oracle.md`. So copy them from the reference
task (`tasks/reward/retool_length_hack/evaluations/issue/`) and keep `judge.py` and `test.sh` **byte-identical**
(`cmp`). `instruction.md` and `judge_prompt.md` may take **minor** task-specific adjustments when the task really
calls for it — e.g. an evaluation-shaped run rather than a training run (`configuration/aime25_single_sample_variance`
rewrites the brief), or a one-line accept-criterion hint. Prefer putting task specifics in `oracle.md`, which the
judge already reads; do **not** fork `judge.py`'s scoring logic to hardcode one task's answer.

A task whose symptom is a measurable number may instead ship a **deterministic** verifier (a `test.sh` that
computes the metric and exits non-zero on failure, no `judge.py`) — `infrastructure/critic_value_head_random_init`
does this at the solution level. That is an allowed alternative, not the default; use it only when a metric
decides the outcome more reliably than a judge, and still write the standard LLM-judge protocol for the issue
level unless the issue itself is a pure metric check.

### Issue attribution

- For each root cause, state the path or symbol, observed (defective side), and expected (control side).
- Logs and curves go only in Observable impact, never as root cause.
- Judge weights are fixed at verdict / root cause / evidence = 0.2 / 0.6 / 0.2, pass mark 0.8.
- **The defective arm also improved** is a common trap: the root cause is "lower than control", not "no learning". List
  "training did not improve" as a wrong attribution in the oracle.

### Threshold

- The threshold sits between the defective side's upper bound and the control side's lower bound. Report the margin to both.
- The threshold must be above the control side's value at step 0; otherwise fixing the code without finishing the run would pass.
- `required_final_score_max` is only for rejecting fabricated values (accuracy caps at `1.0`).
- State **how many runs per side** it was calibrated on. With one run per side, report margins only, not variance, and say so in the readme.
- `final_step` is the last step of the **rerun**, not the last step of the frozen evidence.

### Frozen evidence: slimming and sanitizing

| Keep | Drop |
| --- | --- |
| Training logs, launch logs, server logs, `experiment.json` | Most per-step rollout dumps (keep three samples: first, middle, last) |
| All per-step metric files (a few KB each) | W&B `*.wandb` binaries, `latest-run/` |
| Offline W&B `files/`, `logs/` | Checkpoints, weights, rebuildable derived assets |

Sanitizing changes only labels and paths, **never any metric value**. Remove: the words `broken` / `fixed`, absolute paths,
usernames, hostnames, IPs, and any port or GPU index that reveals which side it is. After sanitizing, scan again:

```bash
grep -rIl 'broken\|fixed\|<original path fragment>\|<original username>' workspace/
```

Any remaining hit must be an upstream word. Do not leave a record of what was sanitized anywhere under
`environment/` (or anywhere else mounted into the agent stage): that directory is mounted read-only into the
agent, so a note spelling out which side is defective or where the fix is would hand over the answer.

### Derived assets

Anything computable from shipped content plus a pinned model (FAISS indexes, torch_dist weights) is generated by
`environment/<script>.py` during prepare, not shipped. The generator must be self-contained, fall back to CPU,
and assert only **the side unrelated to the defect**: `environment/` is readable by the agent, so asserting query-side quality here
points at the defect. Guard outputs with a `.complete` marker, hard-link them into the workspace, and do not write `--expect`.

### Environment

- Pick an image from `docker/images/` that matches the recorded stack. Do not modify existing Dockerfiles. If none fits, add one
  as described in `docker/README.md`; if it is not built yet, write `not yet built` in the readme.
- `environment/prepare.sh` **must always exist** — the harness resolves it at that fixed path. When the task needs
  no networked fetch (assets already in the workspace, nothing to convert), ship a minimal no-op
  (`#!/usr/bin/env bash` + `set -euo pipefail`), not an absent file. `environment/` itself is otherwise optional.
- `prepare.sh` fetches assets with `ptdbench_fetch_hf` / `ptdbench_fetch_url` / `ptdbench_dataset`,
  each with `--expect` and `--expect-size`. HF revisions are 40-char commit shas.
- Set `--dest` to `$PTDBENCH_WORKSPACE/<path the launch script actually reads>`. `$PTDBENCH_MODELS` /
  `$PTDBENCH_DATASETS` are only caches; task scripts must not reference them directly.
- **The launch script must resolve every model / dataset input to a path inside the workspace.** Nothing in the
  harness mounts `/models`, `/datasets`, or any host path, so a script whose *only* default is an absolute mount
  (e.g. `MODEL_DIR=/models/Qwen3-1.7B`) fails. If the raw script hardcodes a mount, repoint its default to a
  workspace-relative path (`${WS}/Qwen3-1.7B`) that `prepare.sh` populates. An absolute path is tolerable only as
  the first choice in a chain that falls back to a workspace path. (Writable scratch/output dirs like `/ckpt` are
  not inputs, but keep them under the workspace or `/tmp` so they are writable.)
- GPUs go in `environment.env` of `metadata.yaml`, and values must be strings. Do not set `PTDBENCH_*`
  or `HF_HUB_OFFLINE` here: the framework sets online and offline mode per stage.

### Do not invent what the raw task does not contain

Packaging **restructures and conforms**; it does not author task substance. Do not fabricate anything the raw
task did not establish — oracle root causes, evaluation thresholds, measured numbers, instructions, diagnostic
claims, or workspace code. Every such piece must be derived from the raw task's own code, diff, logs, readme,
or metadata. If a required piece is genuinely missing from the raw task (e.g. no root cause given for a form-3
snapshot, no asset coordinates), **stop and ask** — do not make one up.

This rule is about *content*. It does not forbid the structural/packaging files the layout requires and that
carry no invented claims: `metadata.yaml` (converted from the raw card), `environment/prepare.sh` (required;
may be a no-op), and `environment/docker-compose.yaml`. Renaming, moving,
slimming, and sanitizing existing files is fine; manufacturing new task substance is not.

### metadata.yaml and readme.md

- `metadata.yaml` fields are in [reference.md](reference.md#metadatayaml), including the required `type`
  (the category = the `tasks/<type>/` directory) and the free-form `domain`. The file `training.script` points
  to must exist in `workspace/`.
- Write `readme.md` (lowercase) per [readme-template.md](readme-template.md): a one-sentence summary, an identity
  table, and the Difficulty / Reference solution / Verification sections. The agent sees `instruction.md`;
  the readme is for reviewers only and must not be mounted into the agent stage.

## Pre-completion checklist

- [ ] Skeleton complete; no `agent_trace.json`, `environment/Dockerfile`, `validation/`
- [ ] `environment/prepare.sh` exists (a no-op is fine when nothing is fetched)
- [ ] `metadata.yaml` has `type` set to the category, and it equals the `tasks/<type>/` directory the task is in
- [ ] Nothing in the package is invented: every oracle/instruction/metric is derived from the raw task, not authored
- [ ] The launch script reads its model/dataset inputs from inside `workspace/`, never only from an absolute mount
- [ ] Issue `judge.py`, `test.sh`, and solution `test.sh` match their source under `cmp`; `instruction.md` / `judge_prompt.md` are the standard text or a documented minimal adjustment
- [ ] Oracle root causes match the filtered diff; partial-credit sentence present for multiple causes
- [ ] Grader replay: defective side exit 1, control side exit 0
- [ ] Workspace has the launch script and failure logs; sanitize scan shows only upstream hits
- [ ] All `env` values in `metadata.yaml` are strings, no `PTDBENCH_*`
- [ ] Every asset in `prepare.sh` has `--expect` / `--expect-size`, and every revision is a 40-char sha
- [ ] `python3 -m ptdbench validate <task_dir>` reports no errors
- [ ] `readme.md` (lowercase) states how many runs per side calibrated the threshold, and which steps were not verified
