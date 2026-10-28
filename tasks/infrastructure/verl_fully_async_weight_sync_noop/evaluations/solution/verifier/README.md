# Solution verifier notes

The verifier reads the held-out **accuracy** metric
`val-core/openai/gsm8k/acc/mean@4` and requires at least `0.746`
(`score_mode: best`, `final_step: 100`).

## Which space the key reads

`acc`, not `score` and not `format`. The run's reward is
`score = 0.2 * format + 0.8 * acc` (`workspace/tools/reward_gsm8k.py`), where
`acc` is a format-independent correctness rate: the graded answer is the number
after `####` when the completion has one and the completion's last number
otherwise. The repaired arm's advantage over the defective arm is much larger in
the `format` component than in `acc`, so gating on `score` would report a far
wider margin than the accuracy evidence supports. Gating on `acc` keeps the
threshold honest about what a working weight sync actually buys on GSM8K.

`val-core` comes from the rollouter's validation pass, not from the trainer:
`async_training.use_trainer_do_validate` is asserted `False` in this recipe
(`fully_async_main.py`), and enabling it would move the trainer pool to
`Role.ActorRollout` and remove the defect. Validation is therefore driven by
`trainer.test_freq` alone, which leaves the topology — and the defect — intact.
It covers all 1,319 test rows at `n=4`, `temperature=1.0`; the recipe default
(`n=1`, greedy) is one sample per prompt and a noisier estimate of the same
quantity.

## Why `score_mode: best`

Both modes separate the four recorded runs at this threshold, so the choice is
about which failure is worse. The repaired arm rises monotonically, so its
maximum *is* its endpoint and `best` costs it nothing; what `best` buys is
tolerance for a repaired run whose step-100 evaluation happens to dip. The
defective arm's exposure is bounded by measurement: across 12 evaluation points
in two passes its highest reading is 0.7314, still 0.0146 below the threshold.

`final_step` stays at 100 either way: `grade_submission.py` requires a finite
value at `final_step` regardless of `score_mode`, so a run that stops early
cannot pass on a lucky mid-training peak.

## Calibration

Two runs per side, same two workspaces, same host, same two GPUs, in sequence
on 2026-10-28. All numbers are `val-core/openai/gsm8k/acc/mean@4`.

| | pass 1 | pass 2 |
| --- | --- | --- |
| defective, step 100 | 0.7248 | 0.7199 |
| defective, max any step | 0.7314 | 0.7286 |
| baseline, step 100 | 0.7743 | 0.7604 |
| step 0 (untrained) | 0.7259 / 0.7263 | 0.7221 / 0.7235 |

Threshold `0.746`:

- **+0.0146** above the defective arm's highest reading at any step in either
  pass (0.7314)
- **-0.0144** below the lower of the two baseline judged values (0.7604)
- **+0.0197** above the higher of the two arms' step-0 values (0.7263)

The step-0 margin is what the skill requires: repairing the code without
completing the run cannot pass. Step 0 comes from
`trainer.val_before_train=True` in `run.sh`.

Replayed through the shipped `grade_submission.py` at this threshold: 2/2
defective runs fail, 2/2 baseline runs pass, under both `best` and `final`.
Two runs per side establish these margins only; no variance or significance is
claimed.
