#!/usr/bin/env python3
"""Build an isolated clean control, measure both, and collect real RL evidence."""
import argparse
from collections import deque
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "evaluations/solution/verifier"
sys.path.insert(0, str(VERIFIER / "tests"))
from metrics import paired_verdict
from parity import read, write, SPEC_PATH
from training_evidence import inspect_training
from create_control import create_control


def run(command, log_path, env):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("Running:", " ".join(map(str, command)), flush=True)
    try:
        with log_path.open("w") as log:
            subprocess.run(list(map(str, command)), env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError:
        print(f"Stage failed; last lines from {log_path}:", file=sys.stderr, flush=True)
        with log_path.open() as log:
            print("".join(deque(log, maxlen=80)), file=sys.stderr, flush=True)
        raise


def publish_manifest(manifest):
    write(ROOT / "evaluations/solution/oracle/historical_evidence_manifest.json", manifest)
    write(ROOT / "evaluations/issue/verifier/resources/qualification.json", manifest)
    write(ROOT / "authoring/qualification.json", manifest)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--score-gpus", type=int, default=1)
    parser.add_argument("--training-gpus", type=int, default=8)
    parser.add_argument("--workspace", type=Path, default=Path(os.environ.get("PTDBENCH_WORKSPACE", ROOT / "workspace")))
    parser.add_argument("--run-dir", type=Path, help="Fresh output directory for this qualification attempt")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    repo = workspace / "verl"
    staging = args.run_dir.resolve() if args.run_dir else ROOT / "authoring/runs"
    candidate_logs = workspace / "logs" / staging.name if args.run_dir else workspace / "logs"
    healthy = staging / "healthy_workspace"
    healthy_repo = healthy / "verl"
    model = workspace / "models/Qwen3-VL-8B-Instruct"
    script = VERIFIER / "tests/parity.py"
    spec = read(SPEC_PATH)
    manifest = read(ROOT / "evaluations/solution/oracle/historical_evidence_manifest.json")
    manifest.update(release_ready=False, historical_run_executed=False, paired_8b_parity_executed=False,
                    metric_status="qualification_running", measured_8b_metrics=None, source_logs=[],
                    threshold_spec_sha256=hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest(),
                    thresholds={key: spec[key] for key in (
                        "baseline_max_nat", "severe_max_nat", "min_separation_ratio")},
                    run_directory=str(staging),
                    training_logs={"candidate": str(candidate_logs), "healthy": str(staging / "healthy_rl")})
    publish_manifest(manifest)
    try:
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() < args.training_gpus:
            raise RuntimeError(f"Qualification requires {args.training_gpus} visible CUDA GPUs")
        if not (workspace / "data/asset_manifest.json").is_file():
            raise RuntimeError("Run environment/prepare.sh first")
        sha = create_control(workspace, healthy, ROOT / "evaluations/solution/oracle")
        if sha != manifest["upstream_commit"]:
            raise RuntimeError("Unexpected upstream source identity")
        manifest["source_diff_sha256"] = hashlib.sha256(
            (ROOT / "evaluations/solution/oracle/source.diff").read_bytes()).hexdigest()
        write(staging / "launch_plan.json", {
            "candidate_source": str(repo), "fixed_source": str(healthy_repo),
            "fixed_source_commit": sha, "initial_checkpoint": str(model),
            "candidate_launcher": str(workspace / "run_qwen3vl_geo3k_grpo.sh"),
            "fixed_launcher": str(healthy / "run_qwen3vl_geo3k_grpo.sh"),
            "launcher_sha256": hashlib.sha256((workspace / "run_qwen3vl_geo3k_grpo.sh").read_bytes()).hexdigest(),
            "execution_order": ["capture", "candidate_score", "healthy_score", "candidate_rl", "healthy_rl"],
            "training_gpus_each": args.training_gpus,
            "training_steps_each": spec["required_training_steps"]})
        print(f"Buggy source: {repo}; fixed source: {healthy_repo}; common initial checkpoint: {model}", flush=True)
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
               "VLLM_WORKER_MULTIPROC_METHOD": "spawn"}
        run([sys.executable, script, "capture", "--workspace", workspace, "--out", staging / "capture.json"],
            staging / "capture.log", env)
        for name, ws in (("candidate", workspace), ("healthy", healthy)):
            run([sys.executable, "-m", "torch.distributed.run", "--standalone",
                 f"--nproc_per_node={args.score_gpus}", script, "score", "--workspace", ws,
                 "--model", model, "--assets", workspace, "--capture", staging / "capture.json",
                 "--out", staging / f"{name}_scores.json"], staging / f"{name}_score.log", env)
        paired = paired_verdict(read(staging / "candidate_scores.json"), read(staging / "healthy_scores.json"), spec)
        write(staging / "paired_verdict.json", paired)
        print("Paired result:", json.dumps(paired, ensure_ascii=False), flush=True)
        manifest.update(paired_8b_parity_executed=True, measured_8b_metrics=paired)
        if not paired["qualified"]:
            raise RuntimeError("Paired 8B experiment did not meet the active severity/repair gates")
        training = {}
        for name, ws, logs in (("candidate", workspace, candidate_logs),
                                ("healthy", healthy, staging / "healthy_rl")):
            train_env = {**env, "MODEL_PATH": str(model), "LOG_DIR": str(logs), "WANDB_DIR": str(logs),
                         "NGPUS_PER_NODE": str(args.training_gpus),
                         "TOTAL_TRAINING_STEPS": str(spec["required_training_steps"])}
            run(["bash", ws / "run_qwen3vl_geo3k_grpo.sh"], staging / f"{name}_launch.log", train_env)
            training[name] = inspect_training(logs, spec["required_training_steps"])
            if not training[name]["passed"]:
                raise RuntimeError(f"{name} RL run lacked complete finite-gradient/validation evidence")
        manifest.update(release_ready=True, historical_run_executed=True, metric_status="measured",
                        training=training, reason=None,
                        qualified_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        asset_manifest_sha256=hashlib.sha256((workspace / "data/asset_manifest.json").read_bytes()).hexdigest(),
                        fixture_pins={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in (workspace / "data/parity").iterdir() if p.is_file()},
                        source_logs=[{"path": str(Path("workspace/logs") / staging.name / "train.log")
                                             if args.run_dir else "workspace/logs/train.log",
                                      "sha256": hashlib.sha256((candidate_logs / "train.log").read_bytes()).hexdigest()}])
        publish_manifest(manifest)
        print("Paired severe mismatch, repaired parity, and both RL runs passed. release_ready=true", flush=True)
        return 0
    except Exception as exc:
        manifest.update(release_ready=False, metric_status="qualification_failed", reason=str(exc))
        publish_manifest(manifest)
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
