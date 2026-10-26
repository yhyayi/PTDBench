from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


VERIFIER = Path(__file__).resolve().parents[1]


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_config(path):
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    if not isinstance(value, dict):
        raise ValueError("Generation configuration must be a JSON object")
    return value


def first_stop(sequence, eos_ids):
    return next((index for index, token in enumerate(sequence) if token in eos_ids), None)


def grade_config(workspace, spec):
    """Gate 1: the repair itself, checked directly on the config file."""
    relative = Path(spec["candidate"])
    candidate = workspace / relative
    current = workspace
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Configuration path must not contain symlinks")
    config = read_config(candidate)
    reference = read_config(VERIFIER.parent / "oracle/generation_config.json")
    eos_ids = config.get("eos_token_id")
    if not isinstance(eos_ids, list) or not eos_ids:
        raise ValueError("eos_token_id must be a nonempty list of integer identifiers")
    if any(type(token) is not int for token in eos_ids):
        raise ValueError("EOS identifiers must be integers, not strings or booleans")
    expected = set(spec["expected_eos_token_ids"])
    return {
        "eos_set_matches": set(eos_ids) == expected and len(eos_ids) == len(expected),
        "other_fields_unchanged": {
            key: value for key, value in config.items() if key != "eos_token_id"
        } == {key: value for key, value in reference.items() if key != "eos_token_id"},
        "assistant_turn_stops": first_stop([42, 151645, 43, 151643], eos_ids) == 1,
        "end_of_text_still_stops": first_stop([42, 151643, 43], eos_ids) == 1,
        "ordinary_tokens_continue": first_stop([42, 43, 44], eos_ids) is None,
    }


def grade_eval(results, spec):
    """Gate 2: the evaluation produced by the harness rerun.

    The rerun (run_spec.json) replayed the retained rollout records with the
    stopping tokens from the repaired model/generation_config.json and wrote
    experiment_logs/eval_metrics.json. The repair passes only if that run's
    accuracy clears the calibrated threshold; a config that passes the
    structural checks but degrades inference is not a fix.
    """
    path = results / spec["eval_metrics_relpath"]
    if not path.is_file():
        raise ValueError(
            f"no {spec['eval_metrics_relpath']} under the results root; the "
            "system rerun produced no evaluation metrics"
        )
    metrics = json.loads(path.read_text(encoding="utf-8"))
    accuracy = metrics.get("accuracy")
    if not isinstance(accuracy, (int, float)):
        raise ValueError("eval_metrics.json has no numeric accuracy")
    required = float(spec["required_eval_accuracy"])
    checks = {
        "eval_metrics_found": True,
        "eval_accuracy_clears_threshold": accuracy >= required,
        "eval_n_examples_complete": metrics.get("n_examples") == spec["eval_n_examples"],
        "eval_stop_tokens_are_candidate_eos": (
            metrics.get("eos_token_id") is not None
        ),
    }
    info = {
        "eval_accuracy": accuracy,
        "required_eval_accuracy": required,
        "eval_n_examples": metrics.get("n_examples"),
        "eval_eos_token_id": metrics.get("eos_token_id"),
        "eval_frac_finished_eos": metrics.get("frac_finished_eos"),
    }
    return checks, info, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Verify the EOS configuration repair against the replay evaluation."
    )
    parser.add_argument("--workspace", type=Path, default=Path(os.environ.get("PTDBENCH_WORKSPACE", "/workspace")))
    parser.add_argument("--results", type=Path, default=Path(os.environ.get("PTDBENCH_RESULTS", "/results")))
    args = parser.parse_args()
    spec = json.loads((VERIFIER / "resources/grader_spec.json").read_text())
    eval_metrics = None
    eval_info = {}
    try:
        checks = grade_config(args.workspace, spec)
        eval_checks, eval_info, eval_metrics = grade_eval(args.results, spec)
        checks.update(eval_checks)
        passed = all(checks.values())
        reason = "All configuration and evaluation checks passed" if passed else "Failed checks"
    except (OSError, ValueError, TypeError) as error:
        checks = {}
        passed = False
        reason = str(error)
    metrics = {
        "evaluation_mode": spec["evaluation_mode"],
        "checks": checks,
        "reason": reason,
        "training_rerun_verified": False,
        "model_inference_verified": eval_metrics is not None,
        **eval_info,
        "required_eval_accuracy": spec["required_eval_accuracy"],
        "historical_defective_reference_accuracy": spec["historical_defective_reference_accuracy"],
        "historical_repaired_reference_accuracy": spec["historical_repaired_reference_accuracy"],
        "absolute_improvement": (
            eval_info["eval_accuracy"] - float(spec["historical_defective_reference_accuracy"])
            if eval_metrics is not None else None
        ),
        "historical_final_baseline": float(spec["historical_defective_reference_accuracy"]),
    }
    result = {"passed": passed, "score": float(passed), "metrics": metrics}
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (args.results / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
