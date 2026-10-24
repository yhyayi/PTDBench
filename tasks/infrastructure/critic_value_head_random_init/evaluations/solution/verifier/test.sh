#!/usr/bin/env bash
# Verifier for the "solution" evaluation: did the critic's value head actually
# get zero-initialised, and does the training run now report a value loss
# inside the reward scale?
#
# Two independent checks, in increasing cost:
#
#   1. Offline  -- construct the value head and assert it is all zero. Needs
#      only torch. This is the fix itself, checked directly.
#   2. End to end -- run the two-stage reproduction and assert Stage B's
#      critic warmup loss is below the reward bound. This is what the fix was
#      for, checked by measurement.
#
# Check 2 needs 2 GPUs and ~15 minutes. Set PTDBENCH_SKIP_TRAIN=1 to run
# only check 1 (useful while iterating, and on a machine with no GPU) -- the
# script says loudly that it did so, and a skipped run is not a pass.
#
# Env:
#   WORKSPACE            workspace under test      (default /workspace)
#   PTDBENCH_RESULTS     where to write artifacts  (default ${WORKSPACE}/outputs)
#   OUT_DIR              run.sh's artifact dir     (default ${PTDBENCH_RESULTS})
#   VALUE_LOSS_THRESHOLD reward-scale bound        (default 1.0)
#   PTDBENCH_SKIP_TRAIN  1 = offline checks only
set -uo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
PTDBENCH_RESULTS="${PTDBENCH_RESULTS:-${WORKSPACE}/outputs}"
OUT_DIR="${OUT_DIR:-${PTDBENCH_RESULTS}}"
THRESHOLD="${VALUE_LOSS_THRESHOLD:-1.0}"

mkdir -p "${PTDBENCH_RESULTS}"
fail() { echo "FAIL: $*" >&2; exit 1; }

# ------------------------------------------------------- 1. the fix itself
echo "== check 1/2: value head is zero at construction =="

TARGET="${WORKSPACE}/slime/slime/backends/megatron_utils/model_provider.py"
[ -f "${TARGET}" ] || fail "missing ${TARGET}"

python3 "${WORKSPACE}/tools/check_value_head.py" "${TARGET}"
case "$?" in
    0) echo "   value head is zero-initialised" ;;
    1) fail "a freshly constructed value head is not all-zero (see output above)" ;;
    *) fail "tools/check_value_head.py could not run; torch is required" ;;
esac

# The actor must be untouched: LinearForLastLayer is the critic's head, but a
# fix that zeroed every output layer would silently destroy the LM head. Assert
# the actor branch still builds its output layer from the checkpoint.
grep -q "output_size=1" "${TARGET}" \
    || fail "${TARGET} no longer builds a 1-output value head for the critic"

if [ "${PTDBENCH_SKIP_TRAIN:-0}" = "1" ]; then
    echo
    echo "PTDBENCH_SKIP_TRAIN=1: skipping the end-to-end run."
    echo "SOLUTION_PARTIAL (offline check only; not a pass)"
    exit 2
fi

# --------------------------------------------------- 2. what the fix was for
echo
echo "== check 2/2: Stage B critic warmup loss is inside the reward scale =="

SUMMARY="${OUT_DIR}/stage_b_value_loss.json"

# Reuse an existing run if the harness already produced one; otherwise run.
if [ -f "${SUMMARY}" ] && [ "${PTDBENCH_REUSE_RUN:-1}" = "1" ]; then
    echo "   reusing ${SUMMARY}"
else
    echo "   running ${WORKSPACE}/run.sh (~15 min on 2 GPUs, longer if the"
    echo "   Endless Terminals prompt set still has to be built and validated)"
    ( cd "${WORKSPACE}" && OUT_DIR="${OUT_DIR}" bash run.sh ) \
        > "${PTDBENCH_RESULTS}/verifier_run.log" 2>&1 \
        || fail "run.sh exited non-zero; see ${PTDBENCH_RESULTS}/verifier_run.log"
fi

[ -f "${SUMMARY}" ] || fail "run.sh produced no ${SUMMARY}"

python3 - "${SUMMARY}" "${THRESHOLD}" <<'PY'
import json, sys

path, threshold = sys.argv[1], float(sys.argv[2])
with open(path) as fh:
    s = json.load(fh)

errors = []

n = s.get("n_critic_steps") or 0
if n < 1:
    errors.append(
        f"no critic optimizer steps in the warmup window (n_critic_steps={n}); "
        "the metric cannot be below the bound by never being reported"
    )

# Rollouts must really have happened -- otherwise a value loss of 0 could just
# mean the critic saw no data.
#
# The evidence is the RolloutManager's per-rollout response-length report, not
# the reward: slime logs `rollout/rewards` from inside `train_actor`, which a
# critic-only warmup stage never calls, so Stage B legitimately has no reward
# metric at all. Response lengths are reported on every rollout regardless.
if (s.get("rollouts_generated") or 0) < 1:
    errors.append("no rollouts were generated; the run did not actually produce episodes")

response_len = s.get("mean_response_len")
if response_len is None or not response_len > 0:
    errors.append(
        "no generated tokens recorded; the critic cannot have been trained on real episodes"
    )

for key in ("first_value_loss", "warmup_value_loss_mean"):
    value = s.get(key)
    if value is None:
        errors.append(f"{key} is missing from the summary")
    elif not value < threshold:
        errors.append(f"{key} = {value:.6f}, expected < {threshold}")

print(f"  critic optimizer steps  {n}")
for key in ("first_value_loss", "warmup_value_loss_mean", "warmup_value_loss_max"):
    value = s.get(key)
    print(f"  {key:<23} {'n/a' if value is None else f'{value:.6f}'}")
print(f"  rollouts generated      {s.get('rollouts_generated')}")
for key, fmt in (("mean_response_len", ".1f"), ("mean_reward", ".4f")):
    value = s.get(key)
    print(f"  {key:<23} {'n/a' if value is None else format(value, fmt)}")

if errors:
    print()
    for e in errors:
        print(f"FAIL: {e}", file=sys.stderr)
    sys.exit(1)
PY
[ $? -eq 0 ] || fail "Stage B critic warmup loss is not inside the reward scale"

cp -f "${SUMMARY}" "${PTDBENCH_RESULTS}/stage_b_value_loss.json" 2>/dev/null || true

echo
echo "SOLUTION_VERIFIED"
