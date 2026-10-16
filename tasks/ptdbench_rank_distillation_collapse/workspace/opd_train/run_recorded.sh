#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
variant=${1:-A}
if (( $# )); then shift; fi
export LOGGER='[console]'
export WANDB_MODE=disabled SWANLAB_MODE=disabled
export O3PD_RAY_CPUS=${O3PD_RAY_CPUS:-64}
export EXPERIMENT_NAME="ranking_distillation_${variant}"
case "$variant" in
  A)
    export VAL_BEFORE_TRAIN=True
    flags=(opd.teacher_fp32_head=False opd.mask_teacher_ties=False)
    ;;
  B)
    export VAL_BEFORE_TRAIN=False
    flags=(opd.teacher_fp32_head=True opd.mask_teacher_ties=True)
    ;;
  *) echo 'Usage: run_recorded.sh A|B [Hydra overrides...]' >&2; exit 2 ;;
esac
exec bash "$ROOT/opd_train/run_o3pd_ds7b_1p5b.sh" "${flags[@]}" "$@"
