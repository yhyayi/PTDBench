# Slime ReTool GRPO run assessment

## Case overview

- Evaluation key: `slime_retool`
- Case family: `slime_implementation`
- Schema version: `2`
- Evaluation levels: `issue`, `solution`
- Model: `Qwen3-4B` (Qwen3-4B-Instruct-2507 SFT → RL)
- Dataset: `DAPO-Math-17k` (RL prompts) + `ReTool-SFT` (SFT warm-up)
- Training framework: `slime` (Megatron train + SGLang rollout)
- Recipe: `examples/retool/` — multi-turn code-interpreter (tool-use) GRPO
- Primary source evaluation metric: `eval/aime`

This document is for maintainers and reviewers. It is not exposed to the
agent during evaluation.

## Canonical verdict

**healthy.**

This is a correct task: the staged workspace is a faithful reproduction of the
canonical ReTool recipe (SFT → RL with a Python code-interpreter tool) on top of
an unmodified upstream `slime` snapshot. No defect has been injected. Frozen
historical evidence through train step 100 shows modest held-out gain, not a
collapse.

## Workspace and hidden evaluations

| Path | Visibility and purpose |
| --- | --- |
| `workspace/` | Agent-visible historical experiment snapshot; evidence rather than canonical ground truth. |
| `evaluations/issue/` | Neutral diagnosis instruction, issue oracle, and task-local issue verifier. |
| `evaluations/solution/` | Diagnosis/remediation instruction, hidden reference material, and combined verifier. |

The agent cannot read oracle or verifier paths. `issue` grading compares
`$PTDBENCH_RESULTS/agent_answer.md` to hidden `oracle.md` with an LLM
judge. `solution` grading reconstructs `eval/aime` from the logs the agent's
own run wrote under `$PTDBENCH_RESULTS/experiment_logs`.

## Source and preserved context

```yaml
evidence_kind: historical_run
packaging_ready: true
verify:
  num_gpus: 8
  max_train_steps: 100
  eval_interval: 10
  last_eval_step: 99
  eval_metric_key: eval/aime
historical_run:
  metric_status: wandb_extracted
  wandb_run_id: pc70p4t4
  wandb_group: qwen3-4B-retool-multi-turn
  eval_first: -0.59125
  eval_last: -0.4965625
  held_out_accuracy_last: 0.25171875
  n_eval_points: 11
  attempt: 1
```

The workspace preserves the SFT and RL launch scripts, the custom multi-turn
tool-calling rollout, the sandboxed Python code interpreter, the math_dapo
exact-match reward, a compact frozen metrics package under
`workspace/outputs/`, and a slimmed offline W&B run (console `output.log`
through train step 100, config, summary; no `.wandb` binary history).

## Original run evidence

- eleven AIME-2024 eval points from step 0 to step 99
- `eval/aime` moves from about −0.591 (20.4% acc) to −0.497 (25.2% acc)
- eval response length grows from about 4.2k to 10.1k; eval truncation rises
  from about 1% to 22%
- train response length at step 100 is about 6394 tokens with a 5.1% truncate
  ratio

The preserved source metric is `eval/aime`. The frozen workspace is the
evidence package the agent diagnoses from; it is also the calibration source
for the solution-level threshold (see below).

## Why the recipe is healthy

The canonical ReTool recipe is reproduced correctly. Key correctness signals:

- **Multi-turn tool rollout** (`examples/retool/generate_with_retool.py`):
  interpreter observations (`<interpreter>…</interpreter>`) injected between
  turns are appended with `trainable=False` (`loss_mask=0`), so tool output is
  never trained on. Only model-generated tokens are appended with
  `trainable=True` (`loss_mask=1`).
- **Token/log-prob alignment**: with `return_logprob=True`, token ids and log
  probs are read directly from `output_token_logprobs`; if sglang returns no
  `output_token_logprobs` the sample is aborted (not silently desynced), and
  `len(response_token_ids) == len(rollout_log_probs)` is asserted per turn. On a
  retried/partial sample, stale rollout state is reset at the top of `generate`
  so concatenation cannot produce a length mismatch.
- **Context-budget safety**: per-turn `max_new_tokens` is clamped to the
  remaining context budget, and verbatim tool output that would overflow the
  budget is trimmed with the text field re-synced from the trimmed tokens — so
  a sample can never exceed `max_tokens_per_gpu * cp_size` and crash the
  trainer.
- **Reward** (`generate_with_retool.py::reward_func`): math_dapo exact-match
  with `strict_box_verify=True` on `Answer: \boxed{...}`, plus a bounded
  tool-call shaping term that only nudges already-failing samples. Length does
  not add reward.
- **Algorithm/optimizer** (`examples/retool/retool_qwen3_4b_rl.sh`): standard
  GRPO (`--advantage-estimator grpo`, `--use-kl-loss --kl-loss-coef 0.00
  --kl-loss-type low_var_kl`, `--eps-clip 0.2 --eps-clip-high 0.28`,
  `--entropy-coef 0.00`), Adam `--lr 1e-6` constant schedule.
- **Wiring**: the recipe plugs into `slime` only through the documented hooks
  `--custom-generate-function-path generate_with_retool.generate` and
  `--custom-rm-path generate_with_retool.reward_func`; the core framework is
  unmodified.

None of these constitute a defect. Unusual-looking but correct choices
(`kl-loss-coef 0.00`, `--rotary-base 5000000` for Qwen3-4B, `max_turns=16`,
`tool_concurrency=32`, W&B debug logging, runtime asset paths under `/root/…`)
are expected properties of the recipe, not causes of unhealthy training.

## Canonical expected outcome

Verdict: **healthy**.

- Root cause targets: none (empty `root_cause`).
- Required fix targets: none (empty `fix`).

An agent that invents a root cause and flips the verdict to unhealthy is
wrong. Correctly assessed unusual-but-benign settings must not be reported as
defects.

## Evaluation design

`issue` grades `$PTDBENCH_RESULTS/agent_answer.md` against hidden
`oracle.md`. The judge scores verdict, root-cause, and evidence agreement
(weights 0.2 / 0.6 / 0.2). Pass is a weighted score of at least `0.8`.
Diagnosis is from the frozen snapshot only — the instruction forbids launching
training, exactly as in the bug-injected tasks.

`solution` is deterministic and metric-only, in the same shape as
`rlve_hyper/*` and `slime_retool_length_hack`: fix what you found, run the
experiment for real, and the verifier reconstructs `eval/aime` from the logs
under `$PTDBENCH_RESULTS/experiment_logs`. Nothing about the protocol tells
the agent whether this run is healthy — that is the point. The bar is
`-0.55` on the best eval point, calibrated so an untrained step 0 (`-0.5913`)
fails and every trained point in the historical trace passes; see
`evaluations/solution/verifier/README.md`.

**Scheduling note for maintainers.** This task has no injected defect, so its
solution level asks for a fix that does not exist. That is resolved outside
the task package: tasks with no injected defect are scheduled at the `issue`
level only. The solution level is kept in the bug-injected shape so that the
task package itself carries no tell.

## Environment requirement

The task runs on the unified PTDBench image, in the `slime-v0.3.1` conda
environment (`environment: conda: slime-v0.3.1`) — torch 2.11 / CUDA 13.0 /
sglang 0.5.15, the stack `slimerl/slime:latest` shipped at slime 0.3.1. It is
*not* the `slime-rc0` environment the `rlve_hyper` tasks use: the ReTool recipe
imports `sglang.srt.constants.GPU_MEMORY_TYPE_CUDA_GRAPH`, which sglang 0.5.0rc0
does not define. See `docker/envs/README.md`.

`environment/prepare.sh` delivers the four assets the ReTool recipe needs:
DAPO-Math-17k jsonl, AIME-2024 jsonl, the SFT'd HF checkpoint (all three
pinned by commit sha and checksummed), and the torch_dist conversion of that
checkpoint. The conversion has no upstream repo, so prepare builds it on the
spot with `WORLD_SIZE=1` and `--rotary-base 5000000`; if no CUDA device is
visible — which is the case on the harbor backend, where prepare is a
build-time layer — it is skipped and the launch script's own
`if [ ! -d "$DIST_MODEL_DIR" ]` guard does it during the agent stage instead.

Two things to know before attempting a real run:

- The recipe needs **8 GPUs** (`ray start --num-gpus 8`,
  `--actor-num-gpus-per-node 8`). `metadata.yaml` declares them through
  `NVIDIA_VISIBLE_DEVICES`, which the docker runtime turns into `--gpus`.
  On the harbor backend that translation does not exist — the generated
  `task.toml` has no `gpus` field and harbor's local docker environment
  reports no GPU capability — so `environment/docker-compose.yaml` carries the
  device reservation and `shm_size` instead. Harbor merges it onto its own
  `main` service; the adapter copies `environment/` verbatim, so no adapter
  change is involved. That file is inert on the native docker path.
- A full 100-rollout run on Qwen3-4B at 16k context is a long job. The
  harness has one `--timeout-s` shared by all three stages, so it has to
  cover prepare's ~16GB download plus the training run.

The workspace code and the ReTool recipe are distributed under Apache-2.0.
