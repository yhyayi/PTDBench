import json
from pathlib import Path

RAW = Path("/Users/chenyinghao/workspace/PTDBench/raw_tasks/hsz_search-r1-delivery")
T = Path("/Users/chenyinghao/workspace/PTDBench/tasks/search_r1_zero_query_vector")
ORACLE = T / "evaluations/solution/oracle"
comparison = json.loads((RAW / "shared/baseline_evidence/comparison.json").read_text())

def rollout_metrics(side, step):
    """Per-step training metrics from the run's own rollouts/metrics_<step>.json."""
    path = RAW / f"workspace_{side}/results/baseline32/rollouts/metrics_{step}.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    return {
        "response_len_mean": data.get("response_length/mean"),
        "response_len_clip_ratio": data.get("response_length/clip_ratio"),
        "train_reward_mean": data.get("critic/score/mean"),
        "valid_search_rate": data.get("env/number_of_valid_search"),
    }

def points(side):
    out = []
    for entry in comparison[side]["curve"]:
        step = entry["step"]
        metrics = entry["metrics"]
        point = {
            "step": step,
            "eval_score": metrics["val/em"],
            "eval_key": "val/em",
            "eval_f1": metrics["val/f1"],
            "answer_format_rate": metrics["val/format"],
            "searched_rate": metrics["val/searched"],
            "eval_count": metrics["val/count"],
            "held_out_accuracy": metrics["val/em"],
        }
        point.update(rollout_metrics(side, step))
        out.append(point)
    return out

defective, control = points("broken"), points("fixed")
common = {
    "schema_version": 1,
    "evidence_kind": "historical_run",
    "metric_status": "trainer_log_extracted",
    "expected_metric_key": "val/em",
    "historical_cutoff_train_step": 32,
    "last_eval_step": 32,
    "eval_interval": 10,
    "eval_steps": [p["step"] for p in defective],
    "note": (
        "Both runs completed all 32 steps and exited 0, on one host, same seed (42), "
        "same 1024/128 HotpotQA split, same e5 corpus and index, same 128-question "
        "greedy held-out protocol. They differ only in retrieval_server.py lines "
        "117-120. One seed per side: this is a paired pilot, not a variance estimate."
    ),
}

manifest = dict(common)
manifest["side"] = "defective (the staged workspace)"
manifest["eval_points"] = defective
manifest["eval_first"] = defective[0]["eval_score"]
manifest["eval_last"] = defective[-1]["eval_score"]
manifest["eval_max"] = max(p["eval_score"] for p in defective)
manifest["control_at_horizon"] = {
    "val/em": control[-1]["eval_score"],
    "val/f1": control[-1]["eval_f1"],
    "val/em_at_step_0": control[0]["eval_score"],
}
(ORACLE / "historical_evidence_manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
)

dirty = dict(common)
dirty["sides"] = {
    "defective": {
        "source": "workspace/results/baseline32 (staged, slimmed and path-sanitized)",
        "eval_points": defective,
        "eval_best": max(p["eval_score"] for p in defective),
    },
    "control": {
        "source": "the delivery's healthy twin; its retrieval_server.py is sealed under "
                  "oracle/code_target/baseline/ and oracle/format_gold/",
        "eval_points": control,
        "eval_best": max(p["eval_score"] for p in control),
    },
}
dirty["graded_threshold"] = {
    "metric_key": "val/em",
    "score_mode": "best",
    "required_final_score": 0.24,
    "defective_best": max(p["eval_score"] for p in defective),
    "control_best": max(p["eval_score"] for p in control),
    "margin_below_bar": round(0.24 - max(p["eval_score"] for p in defective), 5),
    "margin_above_bar": round(max(p["eval_score"] for p in control) - 0.24, 5),
}
(ORACLE / "dirty_reference/historical_metrics.json").write_text(
    json.dumps(dirty, indent=2, ensure_ascii=False) + "\n"
)
print("defective:", [(p["step"], p["eval_score"], p.get("response_len_mean")) for p in defective])
print("control:  ", [(p["step"], p["eval_score"], p.get("response_len_mean")) for p in control])
print("margins:", dirty["graded_threshold"]["margin_below_bar"], dirty["graded_threshold"]["margin_above_bar"])
