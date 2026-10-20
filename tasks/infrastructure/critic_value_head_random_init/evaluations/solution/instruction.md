# Task: fix the inflated critic warmup loss

`/workspace` is a Multi-Turn PPO training workspace for `slime` v0.3.2
(Qwen3-1.7B on Endless Terminals). It has a defect in how the critic is built.

The reward is binary — 1.0 if an episode's pytest verifier passes, else 0.0 —
and GAE runs with `--gamma 1.0 --lambd 1.0`, so the critic's regression target
for every response token is the episode return in `{0, 1}`. A critic that has
not yet seen a single reward should therefore report a value loss bounded by
the reward scale: at most 1.0.

It does not. `./run.sh` trains a checkpoint (Stage A), reloads it, and runs
critic-only warmup steps (Stage B); Stage B's first `train/critic-value_loss`
comes out around **5-6**, and the whole warmup window is spent working it back
down. The diagnosis is in `evaluations/issue/oracle.md` if you want it, but the
symptom is reproducible directly: run `./run.sh` and read
`${OUT_DIR}/stage_b_value_loss.json`.

## What to change

Make a freshly constructed critic predict exactly 0 everywhere, so that its
first reported value loss is `mean(R²) = pass_rate ≤ 1`.

The critic's value head is installed by
`slime/backends/megatron_utils/model_provider.py`. The workspace overrides that
one file: edit `slime_patch/model_provider.py`, which `run.sh` installs into
`/root/slime` at startup (Step 0). Editing `/root/slime` directly instead will
be overwritten on the next run.

Keep the change minimal and keep it correct for the actor: the same module
builds both roles, and the actor's `[vocab_size, hidden_size]` LM head must
keep loading from the checkpoint exactly as it does now.

## How you will be checked

1. **Offline** — `python3 tools/check_value_head.py` must pass, i.e. a freshly
   built value head must have an all-zero weight *and* bias. This needs only
   torch; run it as often as you like, it takes seconds.
2. **End to end** — `./run.sh` must complete and Stage B must report
   `first_value_loss < 1.0` and `warmup_value_loss_mean < 1.0` over at least
   one critic optimizer step, on rollouts that really generated episodes.

A full run needs 2 GPUs and takes roughly 15 minutes, plus a one-off pass to
build and validate the Endless Terminals prompt set on the first run. Do not
weaken the check by suppressing the metric, shortening the warmup to zero
steps, or changing the reward scale, the advantage estimator or
`--value-clip`: the verifier asserts that critic optimizer steps, generated
rollouts and non-zero response lengths are all present. (It does not look for
a reward metric — Stage B is critic-only, and slime logs `rollout/rewards`
from inside `train_actor`, so a correct Stage B has none.)
