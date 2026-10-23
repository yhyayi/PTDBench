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
down. The diagnosis is in `evaluations/issue/oracle.md` if you want it.

## Your job

**Do not launch training — do not run `./run.sh` yourself, in the foreground
or background, and do not use `nohup`, `&`, `tmux`, `screen`, or any detach
mechanism.** The system runs the experiment itself after you finish (see
below); a run you start is wasted and scores nothing.

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

When you are done, submit a report at `/workspace/report.md`. It must
summarize:

1. **Root cause** — why a freshly constructed critic reports a value loss of
   5-6 instead of ≤ 1.
2. **Changes** — every file you changed and what you changed in it.
3. **Expected effect** — the metric the fix should move and its expected
   value.
4. **How you verified the fix without training** — the checks you ran.

A missing or empty `/workspace/report.md` scores 0.

## How you will be checked

After you submit the report, the system runs `./run.sh` end-to-end in your
repaired workspace (a full run needs 2 GPUs and takes roughly 15 minutes, plus
a one-off pass to build and validate the Endless Terminals prompt set on the
first run) and a task-specific verifier grades the fresh results:

1. **Offline** — `python3 tools/check_value_head.py` must pass, i.e. a freshly
   built value head must have an all-zero weight *and* bias. This needs only
   torch; run it as often as you like, it takes seconds. It is the cheap way
   to confirm the fix without training.
2. **End to end** — the system's `./run.sh` run must complete and Stage B must
   report `first_value_loss < 1.0` and `warmup_value_loss_mean < 1.0` over at
   least one critic optimizer step, on rollouts that really generated
   episodes.

Do not weaken the check by suppressing the metric, shortening the warmup to
zero steps, or changing the reward scale, the advantage estimator or
`--value-clip`: the verifier asserts that critic optimizer steps, generated
rollouts and non-zero response lengths are all present. (It does not look for
a reward metric — Stage B is critic-only, and slime logs `rollout/rewards`
from inside `train_actor`, so a correct Stage B has none.)
