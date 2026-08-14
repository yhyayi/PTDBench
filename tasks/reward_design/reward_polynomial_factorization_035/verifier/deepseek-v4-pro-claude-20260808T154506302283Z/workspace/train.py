import os
import sys

import ray
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from slime.ray.placement_group import create_actor_group, create_placement_groups, create_rollout_manager
from slime.utils.arguments import parse_args
from slime.utils.wandb_utils import init_wandb_primary


def _run_scan_analyze_and_exit() -> None:
    """Eval dump 已寫入時，離線分析並寫 pre.json，然後結束（不進入訓練）。"""
    dump = os.environ.get("RLVE_SCAN_EVAL_DUMP", "").strip()
    out = os.environ.get(
        "RLVE_SCAN_PRE_JSON",
        "/workspace/outputs/pre.json",
    ).strip()
    report = os.environ.get(
        "RLVE_SCAN_REPORT_JSON",
        "/workspace/outputs/pre_report.json",
    ).strip()
    n_samples = int(os.environ.get("RLVE_SCAN_N_SAMPLES", "8"))
    if not dump or not os.path.isfile(dump):
        raise FileNotFoundError(f"RLVE_SCAN_EVAL_DUMP missing: {dump}")

    # scripts/analyze_scan_eval.py lives in workspace
    root = os.getcwd()
    script = os.path.join(root, "scripts", "analyze_scan_eval.py")
    cmd = [
        sys.executable,
        script,
        "--raw",
        dump,
        "--out",
        out,
        "--report",
        report,
        "--n-samples",
        str(n_samples),
    ]
    print(f"[scan_grpo] analyzing: {' '.join(cmd)}", flush=True)
    import subprocess

    subprocess.check_call(cmd)
    print(f"[scan_grpo] done -> {out}", flush=True)


def train(args):
    # allocate the GPUs
    pgs = create_placement_groups(args)
    wandb_run_id = init_wandb_primary(args)

    actor_model = create_actor_group(args, pgs["actor"], wandb_run_id=wandb_run_id)

    # create the rollout manager, with sglang engines inside.
    rollout_manager = create_rollout_manager(args, pgs["rollout"], wandb_run_id=wandb_run_id)

    # calculate num_rollout from num_epoch
    num_rollout_per_epoch = None
    if args.num_rollout is None:
        num_rollout_per_epoch = ray.get(rollout_manager.controller.get_num_rollout_per_epoch.remote())
        args.num_rollout = num_rollout_per_epoch * args.num_epoch
    assert args.num_rollout > 0

    # sync the initialization (model initalization, load checkpoint, etc.)
    start_rollout_ids = ray.get(
        actor_model.async_init(args, role="actor", with_ref=args.kl_coef != 0 or args.use_kl_loss)
    )
    assert len(set(start_rollout_ids)) == 1
    if args.start_rollout_id is None:
        args.start_rollout_id = start_rollout_ids[0]

    if args.rollout_global_dataset:
        ray.get(rollout_manager.controller.load.remote(args.start_rollout_id - 1))
    elif args.rlve:
        ray.get(rollout_manager.controller.load.remote(args.start_rollout_id - 1))
    else:
        assert False, "None of args.rollout_global_dataset or args.rlve is set - there is no data source."

    # initialize the connection for weight update during training
    ray.get(actor_model.async_init_weight_update_connections(rollout_manager))

    if args.offload:
        ray.get(rollout_manager.async_onload(tags=[GPU_MEMORY_TYPE_WEIGHTS]))

    # always update weight first so that sglang has the loaded weights from training.
    ray.get(actor_model.async_update_weights())

    if args.offload:
        ray.get(rollout_manager.async_onload(tags=[GPU_MEMORY_TYPE_KV_CACHE]))

    # ---- SCAN mode: only run eval (Ray + SGLang)，不訓練 ----
    if os.environ.get("RLVE_SCAN_GRPO", "").strip() in ("1", "true", "yes", "on"):
        print("[scan_grpo] eval-only mode: running async_eval then exit", flush=True)
        ray.get(rollout_manager.async_eval(args.start_rollout_id))
        _run_scan_analyze_and_exit()
        return

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # TODO extract the duplicated eval logic
        if args.eval_interval is not None and rollout_id == 0:
            ray.get(rollout_manager.async_eval(rollout_id))

        rollout_data_ref = ray.get(rollout_manager.async_generate(rollout_id))

        if args.offload:
            ray.get(rollout_manager.async_offload())

        ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        if args.save_interval is not None and (
            (rollout_id + 1) % args.save_interval == 0
            or (num_rollout_per_epoch is not None and (rollout_id + 1) % num_rollout_per_epoch == 0)
        ):
            ray.get(actor_model.async_save_model(rollout_id))
            if args.rollout_global_dataset:
                ray.get(rollout_manager.controller.save.remote(rollout_id))
            elif args.rlve:
                ray.get(rollout_manager.controller.save.remote(rollout_id))
            else:
                assert False, "None of args.rollout_global_dataset or args.rlve is set - there is no data source."

        if args.offload:
            ray.get(actor_model.async_offload())
            ray.get(rollout_manager.async_onload(tags=[GPU_MEMORY_TYPE_WEIGHTS]))

        ray.get(actor_model.async_update_weights())

        if args.offload:
            ray.get(rollout_manager.async_onload(tags=[GPU_MEMORY_TYPE_KV_CACHE]))

        if args.eval_interval is not None and (
            (rollout_id + 1) % args.eval_interval == 0
            or (num_rollout_per_epoch is not None and (rollout_id + 1) % num_rollout_per_epoch == 0)
        ):
            ray.get(rollout_manager.async_eval(rollout_id))


if __name__ == "__main__":
    args = parse_args()
    train(args)
