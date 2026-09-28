# Slime Search-R1 GRPO run assessment

## Case overview

- Evaluation key: `slime_search_r1`
- Case family: `slime_implementation`
- Schema version: `2`
- Evaluation levels: `issue`, `solution`
- Model: `Qwen2.5-3B`
- Dataset: `Search-R1 nq_hotpotqa_train` (NQ + HotpotQA merged)
- Training framework: `slime` (Megatron train + SGLang rollout)
- Recipe: `examples/search-r1/` — multi-turn retrieval-augmented GRPO
- Primary source evaluation metric: `eval/nq_test`

This document is for maintainers and reviewers. It is not exposed to the
agent during evaluation.

## Canonical verdict

**healthy.**

This is a correct task: the staged workspace is a faithful reproduction of the
canonical Search-R1 recipe on top of an unmodified upstream `slime` snapshot.
No defect has been injected. Frozen historical evidence through train step 100
shows a clear held-out gain, not a collapse.

## Workspace and hidden evaluations

| Path | Visibility and purpose |
| --- | --- |
| `workspace/` | Agent-visible historical experiment snapshot; evidence rather than canonical ground truth. |
| `evaluations/issue/` | Neutral diagnosis instruction, issue oracle, and task-local issue verifier. |
| `evaluations/solution/` | Repair-and-run instruction, hidden reference material, and metric-only verifier. |

The agent cannot read oracle or verifier paths. `issue` grading compares
`$PTDBENCH_RESULTS/agent_answer.md` to hidden `oracle.md` with an LLM
judge. `solution` grading reconstructs `eval/nq_test` from the logs the agent's
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
  eval_metric_key: eval/nq_test
historical_run:
  metric_status: wandb_extracted
  wandb_run_id: var4dpbc
  wandb_group: search-r1_qwen2.5-3B
  eval_first: 0.046
  eval_last: 0.37533333333333335
  held_out_accuracy_last: 0.37533333333333335
  n_eval_points: 11
  attempt: 1
```

The workspace preserves the launch script, the custom multi-turn search
rollout, the exact-match reward, retrieval-server clients, a compact frozen
metrics package under `workspace/outputs/`, and a slimmed offline W&B run
(console `output.log` through train step 100, config, summary; no `.wandb`
binary history).

## Original run evidence

- eleven NQ-test eval points from step 0 to step 99
- `eval/nq_test` moves from about 4.6% to 37.5%
- eval response length falls from about 859 to 31; eval truncation falls
  from about 38% to 0.07%
- train response length at step 100 is about 39 tokens with a 0% truncate
  ratio

The preserved source metric is `eval/nq_test`. The frozen workspace is the
evidence package the agent diagnoses from; it is also the calibration source
for the solution-level threshold (see below).

## Why the recipe is healthy

The canonical Search-R1 recipe is reproduced correctly. Key correctness
signals a reviewer can confirm from the code and frozen metrics:

- **Multi-turn rollout** (`examples/search-r1/generate_with_search.py`):
  observation (`<information>…</information>`) tokens injected between turns
  are appended with `loss_mask=0` (`trainable=False`), so retrieved context is
  not trained on. Only model-generated tokens carry `loss_mask=1`.
- **Token/log-prob alignment**: with `return_logprob=True` (TIS), the code
  takes token ids and log probs directly from `output_token_logprobs` and
  skips string post-processing, asserting `len(response_token_ids) ==
  len(rollout_log_probs)`. A documented bugfix adds `</search>` / `</answer>`
  as stop strings so generation halts at the tool/answer boundary.
- **Reward** (`examples/search-r1/qa_em_format.py`): exact-match with a format
  state machine (`<think>→<search>→<information>→…→<answer>`) and graded
  scoring matching the reference implementation.
- **Algorithm/optimizer** (`examples/search-r1/run_qwen2.5_3B.sh`): standard
  GRPO (`--advantage-estimator grpo`, `--use-kl-loss --kl-loss-coef 0.001
  --kl-loss-type low_var_kl`, `--eps-clip 0.2 --eps-clip-high 0.28`,
  `--entropy-coef 0.00`), Adam `--lr 1e-6` constant schedule.
- **Wiring**: the recipe plugs into `slime` only through the documented hooks
  `--custom-generate-function-path generate_with_search.generate` and
  `--custom-rm-path generate_with_search.reward_func`.

None of these constitute a defect. Unusual-looking but correct choices
(`max_turns=2`, commented-out eval/W&B blocks, retrieval server at
`127.0.0.1:8000`, `format_score=0.2`) are expected properties of the recipe,
not causes of unhealthy training.

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
`slime_search_r1_nomask` and `slime_retool`: fix what you found, run the
experiment for real, and the verifier reconstructs `eval/nq_test` from the logs
under `$PTDBENCH_RESULTS/experiment_logs`. Nothing about the protocol tells the
agent whether this run is healthy — that is the point. The bar is `0.28` on the
best eval point, the same bar the injected twin uses, so the threshold itself
is not a tell either; it fails an untrained step 0 (`0.046`) and the twin's
failure mode (peak `0.1527`), and an unmodified run clears it by `0.096`
(historical peak `0.3763`). See
`evaluations/solution/verifier/README.md`.

**Scheduling note for maintainers.** This task has no injected defect, so its
solution level asks for a fix that does not exist. That is resolved outside
the task package: tasks with no injected defect are scheduled at the `issue`
level only. The solution level is kept in the bug-injected shape so that the
task package itself carries no tell.

## Environment requirement

The task runs on the `slime-v0.3.2` PTDBench image
(`environment: image: slime-v0.3.2`, built from
`docker/images/slime-v0.3.2.Dockerfile`). There is no per-task Dockerfile;
`environment/prepare.sh` does that work instead.

`metadata.yaml`'s `environment.env` block is byte-for-byte the one
`slime_search_r1_nomask` uses — the eight GPUs and the offline W&B settings.
The two tasks are the same recipe, so a difference in container environment
between them would be both a tell and a reason their reruns would not be
comparable. `WANDB_API_KEY` has to carry a placeholder value rather than be
unset: `run_qwen2.5_3B.sh` builds `--wandb-key ${WANDB_API_KEY}` unquoted, and
an empty expansion silently shifts the argument list.

It deliberately does **not** set `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE`.
The framework sets those per stage (`_network_policy_env` in `ptdbench/runtimes/base.py`), so
the agent and verifier stages are off the hub either way, while `prepare` —
the one stage that is supposed to have network — keeps it. A task-level copy
applies to all three stages and would silently disable every download in
`prepare.sh`; that is exactly the failure this block used to have.

`prepare.sh` installs one thing: `faiss-gpu-cuvs` (plus the four cuVS runtime
wheels it dlopens), which only the two search-r1 tasks need. It goes into
`$PTDBENCH_PREPARED` rather than into the image, because the
`slime-v0.3.2` image is shared by all four slime tasks — adding a package there
would change `slime_retool`'s runtime.
The GPU build is required, not the CPU one:
`local_dense_retriever/retrieval_server.py:211-214` calls
`faiss.index_cpu_to_all_gpus`, which `faiss-cpu` does not define.
It is the `-cuvs` build rather than plain `faiss-gpu` because this environment
is CUDA 13 (the former `slime-v0.3.1` environment had `torch==2.11.0+cu130`
and unsuffixed `nvidia-*` wheels under `nvidia/cu13/`); `faiss-gpu` is the
CUDA 12 build and looks for `nvidia-*-cu12` modules that are not here. The cuVS
wheels are pinned to 26.6 rather than 26.8 because 26.8 wants
`nvidia-nvjitlink>=13.3` and the former environment pinned 13.0.88.

`prepare.sh` also stages every asset a real run needs, into
`$PTDBENCH_WORKSPACE` (which is `$PWD` for the launch script, so the recipe's
own `DATA_DIR`/`MODEL_DIR` defaults find them):

| Asset | How |
| --- | --- |
| `Qwen2.5-3B/` | `ptdbench_fetch_hf`, revision-pinned, both shards checksummed |
| `Qwen2.5-3B_torch_dist/` | converted on the spot by `tools/convert_hf_to_torch_dist.py`, when the prepare stage finds a GPU with at least 12 GiB free; otherwise left to the launch script's own guard |
| `e5-base-v2/` | `ptdbench_fetch_hf`; the retrieval server loads the query encoder by name and would otherwise want the hub |
| `nq_hotpotqa_train/{train,test}.parquet` | `ptdbench_dataset --recipe search_r1_nq_hotpotqa` |
| `wiki-18/{e5_Flat.index,wiki-18.jsonl}` | `ptdbench_dataset --recipe wiki18_e5` |

Issue-level diagnosis needs none of them; a solution-level rerun needs all of
them, which is why they are fetched rather than assumed. About 90 GB is cached
under `$PTDBENCH_DATASETS` / `$PTDBENCH_MODELS` and hardlinked into each run's
workspace, so the cost is paid once per host, not once per evaluation. Both
recipes are deterministic and revision-pinned for the same reason the image's
`FROM` is pinned by digest: the solution threshold was calibrated against these
exact bytes.

The two `ptdbench_dataset` recipes are the pinned equivalent of the recipe's
own `examples/search-r1/prepare_data.sh`, whose `-f` guards then make the
`prepare_data.sh` call at `run_qwen2.5_3B.sh:62` a no-op. What prepare does
*not* do is start the retrieval server — it cannot: the prepare stage is a
container created and removed around `prepare.sh`
(`ptdbench/runtimes/providers/docker.py::close_sandbox`), so a server
backgrounded there does not outlive the stage. `run_qwen2.5_3B.sh` therefore
brings it up itself (`examples/search-r1/start_retriver.sh`) whenever `:8000`
has no listener, and waits for readiness. What prepare *can* do, and does, is
warm the corpus's `datasets` arrow cache so that start pays only the index
load: the persistent copy lives under `$PTDBENCH_DATASETS`, hardlinked into
`$PTDBENCH_WORKSPACE/.hf_datasets_cache` because the agent stage needs a
writable cache directory and only the workspace is one.
`slime_search_r1_nomask` stages the same five assets with a byte-identical
`prepare.sh` body.

Two more things to know before attempting a real run:

- The recipe needs **8 GPUs** (`ray start --num-gpus 8`,
  `--actor-num-gpus-per-node 8`). `metadata.yaml` declares them through
  `NVIDIA_VISIBLE_DEVICES`, which the docker runtime turns into `--gpus`.
  On the harbor backend that translation does not exist, so
  `environment/docker-compose.yaml` carries the device reservation and
  `shm_size` instead. That file is inert on the native docker path.
- The retrieval server holds the FAISS index in GPU memory alongside the
  training job and takes minutes to load. `run_qwen2.5_3B.sh` refuses to start
  without a listener on `:8000`, and its rerun preamble deliberately spares the
  retriever process so it survives across attempts.

The workspace code and the Search-R1 recipe are distributed under Apache-2.0.
