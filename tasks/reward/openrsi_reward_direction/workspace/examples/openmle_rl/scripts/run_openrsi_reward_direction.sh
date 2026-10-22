#!/usr/bin/env bash
# OpenRSI execution-grounded RL — task entry point.
#
# Thin wrapper over the released single-node asynchronous launcher. The launcher
# takes an absolute path to an env file and does the real work (sourcing config,
# deriving batch sizes, prechecking assets, starting Ray/SGLang, invoking
# train_async.py). Everything this wrapper adds is the PTDBench stage contract:
# a foreground run whose stdout/stderr is captured under $LOG_DIR.
#
# Direct launcher invocation (equivalent, for environments that do not need the
# log-capture contract):
#   PRECHECK_ONLY=1 bash examples/openmle_rl/scripts/run_openmle_rl_async_single_node.sh \
#     examples/openmle_rl/configs/reward_direction.env
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RL_DIR="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
WORKSPACE="$(CDPATH= cd -- "$RL_DIR/../.." && pwd)"

CONFIG_FILE="${1:-${OPENMLE_RL_CONFIG:-$RL_DIR/configs/reward_direction.env}}"
LOG_DIR="${LOG_DIR:-$PWD/experiment_logs}"
mkdir -p "$LOG_DIR"

export OPENMLE_RL_CONFIG="$CONFIG_FILE"
# Offline W&B: the graded run must not depend on a reachable W&B endpoint.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="$LOG_DIR"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cd "$WORKSPACE"

# Foreground only. No nohup/&/tmux/screen: the run must return before the
# verifier reads experiment_logs.
bash examples/openmle_rl/scripts/run_openmle_rl_async_single_node.sh "$CONFIG_FILE" 2>&1 \
    | tee "$LOG_DIR/train.log"
