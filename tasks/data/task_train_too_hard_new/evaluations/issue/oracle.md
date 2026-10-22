# Canonical issue oracle

Expected verdict: **unhealthy**.

## Root cause

The training set is too hard for the held-out set. The training distribution and
the validation distribution do not match in difficulty.

Either statement is the essential root cause:

- the training set is too hard
- the training set and the validation set do not match in difficulty

## Observable impact

- Primary source evaluation metric: `val-core/taco/acc/mean@1`.
- Training optimizes on TACO `VERY_HARD` rows in `workspace/taco_verl/train.parquet`.
- Validation scores TACO `EASY` rows in `workspace/taco_verl/test.parquet`.
- Held-out accuracy is therefore not a measurement of the training distribution.
- The bundled run in `workspace/logs/qwen2_5_1.5b_instruct_taco_fsdp_20261010_073442.log` finished 400/400 steps in 8:40:39. The train dataloader is 145 steps per epoch, so the step cap stops training during the third epoch.
- Held-out accuracy on 184 rows starts at 0.0489 (9/184), peaks at step 50 at 0.0978 (18/184), falls to 0.0054 (1/184) at step 250, and ends at 0.0380 (7/184). The final value is below the step-0 baseline.
- `critic/rewards/mean` over steps 1–400 averages 0.0938. Steps 1–100 stay near 0.008. From step 201 the fifty-step means sit around 0.15–0.18. A higher training reward does not lift the held-out curve. These figures illustrate the mismatch. Quoting them is not an additional required claim.

## Evidence

- The rows do not carry an `extra_info.difficulty` label. The mismatch is in the problems: every training row is a TACO `VERY_HARD` problem, and every validation row is a TACO `EASY` problem.
- Citing a `difficulty` metadata field is not required.
- The launch script, model, `verl/utils/reward_score/taco.py`, and
  `verl/utils/torch_functional.py` are the healthy baseline. They are not the
  root cause.
