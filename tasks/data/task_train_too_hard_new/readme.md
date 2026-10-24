# data diagnosis: TACO GRPO run (training set too hard)

## Case overview

- Evaluation key: `data/task_train_too_hard_new`
- Case family: `data`
- Evaluation levels: `issue`
- Model: `Qwen2.5-1.5B-Instruct`
- Training framework: `verl` (GRPO)
- Environment image: `verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12`
- Primary source evaluation metric: `val-core/taco/acc/mean@1`

This document is for maintainers and reviewers. It is not exposed to the
agent during evaluation. There is no solution level. Issue grading compares
`$PTDBENCH_RESULTS/agent_answer.md` to hidden `oracle.md`.

## Canonical failure reason

`workspace/taco_verl/train.parquet` contains only TACO `VERY_HARD` rows;
`workspace/taco_verl/test.parquet` is the existing TACO `EASY` validation set.
The training problems are too hard for the 1.5B model under the binary
all-tests-pass reward, so almost every GRPO group is all-zero, advantages are
zero on most steps, and training never learns. Held-out accuracy stays flat.

The train/validation difficulty mismatch is evidence for "too hard", not the
mechanism: the run would stall the same way with a `VERY_HARD` validation set.
Code, launch hyperparameters, and the model are the healthy baseline.

## Workspace and hidden evaluations

| Path                 | Visibility and purpose                                            |
| -------------------- | ----------------------------------------------------------------- |
| `workspace/`         | Agent-visible snapshot: code, one training log, wandb, dumps.    |
| `environment/`       | Agent-visible (read-only): `prepare.sh`.                          |
| `evaluations/issue/` | Neutral instruction, hidden oracle, and issue verifier.           |

`workspace/rollout_data/` holds the per-step training rollout dumps
(400 files x 128 rollouts) and `workspace/validation_data/` the nine
validation dumps (184 rollouts each).

## Historical training run

- Log: `workspace/logs/qwen2_5_1.5b_instruct_taco_fsdp_20261021_080557.log`
- Wandb: `workspace/logs/wandb/offline-run-20261021_080907-4pye3rsp`
- 4 GPUs, 400/400 steps in 15:18:01 (137.70 s/step); 145 steps per epoch
  (2329 training rows), so the step cap stops training in the third epoch.
- Config: `train_batch_size=16`, `rollout.n=8`, `lr=1e-6`, `use_kl_loss=True`,
  `kl_loss_coef=0.02` (low_var_kl), `entropy_coeff=0.001`, rollout temperature
  1.0, validation temperature 0.7.

Held-out results on 184 TACO `EASY` rows (counts are `round(acc * 184)`):

| Step | Fully correct | acc    | pass_rate |
| ---- | ------------- | ------ | --------- |
| 0    | 13/184        | 0.0707 | 0.2246    |
| 50   | 17/184        | 0.0924 | 0.2335    |
| 100  | 16/184        | 0.0870 | 0.2212    |
| 150  | 16/184        | 0.0870 | 0.2416    |
| 200  | 13/184        | 0.0707 | 0.2224    |
| 250  | 15/184        | 0.0815 | 0.2343    |
| 300  | 13/184        | 0.0707 | 0.2386    |
| 350  | 12/184        | 0.0652 | 0.2473    |
| 400  | 14/184        | 0.0761 | 0.2250    |

Training signal:

- `critic/rewards/mean` averages 0.0078; 254/400 steps are 0; max 0.1016.
- 256/400 steps have all-zero advantages and `pg_loss` = 0.
- Rollout dumps: 0.78% of 51,200 training rollouts succeed; 175/6,400 groups
  (2.7%) are mixed, 2 all-correct, the rest all-zero; 104 of 2,151 distinct
  training prompts are ever solved.
- Entropy rises 0.69 -> ~1.0-1.1 (peak 2.04); response length 300-750 tokens,
  clip ratio 0.2%. Hard-coded `print(<constant>)` outputs: 9/51,200 training,
  0/1,656 validation.

An earlier version of this task shipped a run with `use_kl_loss=False`,
`entropy_coeff=0`. That run failed differently (reward hacking with hard-coded
outputs, entropy and length collapse, validation below step 0), which made
"no KL / no entropy regularization" an independently sufficient explanation.
It has been removed from the workspace; only the run above is shipped.

## Evaluation design

The judge scores verdict, root cause, and evidence (weights 0.2 / 0.6 / 0.2);
pass is a weighted score of at least `0.8`. Full root-cause credit requires
"training data too hard" connected to the missing learning signal. Stating only
"too hard" or only a train/validation difficulty mismatch, without the causal
chain, gets partial root-cause credit (0.3) and cannot reach the pass line.
See `evaluations/issue/verifier/judge_prompt.md`.

## External dataset asset

- Repository: `LIF1014/ptdbench-data-task-train-too-hard-dataset`
- Pinned revision: `22a1fee0ea2527f514c4115f6c792216a99d7e6f`
- Dataset license: `Apache-2.0`
- Archive `artifacts/workspace-assets.tar.gz`: 167,612,568 bytes, SHA-256
  `eddeece8c2828e21b581d5168dd4604470165cb061a53e68d50b26d49ce6d916`.

| Runtime path                        | Size              | SHA-256                                                            |
| ----------------------------------- | ----------------- | ------------------------------------------------------------------ |
| `workspace/taco_verl/test.parquet`  | 61,082,426 bytes  | `0593078d24134d204b305dd0373acfc33daed9a2c73fdad7304a2fde191daea3` |
| `workspace/taco_verl/train.parquet` | 168,834,942 bytes | `8640b3e6c7a5ff3d5b51aa46e74f74147f3c565cd89b73676d5113f8e546d573` |

2329 `VERY_HARD` training rows (all `stdin`; 4 rows above 2048 prompt tokens
dropped) and the unchanged 184-row `EASY` validation set. The parquet files do
not store `extra_info.difficulty`.

Known leak: the repository name above contains the answer and appears in the
agent-visible `environment/prepare.sh`. Republishing the archive under a
neutral repository name is still open.

## External model asset

- Repository: `Qwen/Qwen2.5-1.5B-Instruct` (official)
- Pinned revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Byte-identical, on every allowed file, to the former task snapshot
  `LIF1014/ptdbench-verl-implementation-torch-functional@36a4c22`; switched
  because that repo name is visible to the agent through the shared download cache.
- `model.safetensors` SHA-256 `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`

## Environment requirement

- Image: `verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12` (`docker/images/verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12.Dockerfile`); assets come from
  `environment/prepare.sh`.
- The recorded run had `pyext==0.7` installed (see the wandb requirements);
  issue evaluation does not run the scorer, so `prepare.sh` does not install it.
- Issue evaluation does not launch training and needs no GPU.
