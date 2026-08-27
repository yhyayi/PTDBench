#!/usr/bin/env python3
"""Grade one schema-v2 PTDBench submission with evaluation-v2 semantics."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

from runtime_utils import (
    GradingError,
    STAGE_SCORE,
    atomic_write_json,
    grade_issue,
    grade_solution,
    load_gold,
    load_submission,
)


def _issue_detail(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: result[key]
        for key in (
            "verdict_match",
            "decided_by",
            "matched_gold_names",
            "root_cause_verdict",
            "why",
        )
        if key in result
    }


def _solution_detail(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: result[key]
        for key in (
            "decided_by",
            "covered_must",
            "missing_must",
            "fix_verdict",
            "why",
        )
        if key in result
    }


def grade(
    level: str,
    gold: Dict[str, Any],
    submission: Dict[str, Any],
    verifier_dir: Path,
    *,
    no_judge: bool = False,
    solution_on_partial: bool = False,
) -> Dict[str, Any]:
    output: Dict[str, Any] = {
        "task_key": (
            f"{gold.get('category')}/{gold.get('framework')}/{gold.get('task')}"
        ),
        "category": gold.get("category"),
        "framework": gold.get("framework"),
        "task": gold.get("task"),
        "gold_verdict": gold.get("verdict"),
        "agent_verdict": submission.get("verdict"),
        "evaluation_level": level,
    }
    issue_result = grade_issue(
        gold,
        submission,
        verifier_dir / "judge_issue.md",
        no_judge=no_judge,
    )
    issue = issue_result["root_cause_verdict"]
    output["issue"] = issue
    output["issue_detail"] = _issue_detail(issue_result)

    if level == "issue":
        output["solution"] = None
        output["case_score"] = STAGE_SCORE.get(issue)
        if issue == "unavailable":
            output["note"] = "issue judge unavailable"
        return output

    if issue == "unavailable":
        output.update(
            {
                "solution": None,
                "case_score": None,
                "note": "issue judge unavailable",
            }
        )
        return output

    gate = {"correct", "partial"} if solution_on_partial else {"correct"}
    if issue not in gate:
        output.update(
            {
                "solution": None,
                "case_score": STAGE_SCORE.get(issue, 0.0) * 0.5,
            }
        )
        return output

    if gold.get("verdict") == "healthy":
        output.update(
            {
                "solution": "n/a",
                "case_score": STAGE_SCORE[issue],
            }
        )
        return output

    solution_result = grade_solution(
        gold,
        submission,
        issue_result,
        verifier_dir / "judge_solution.md",
        no_judge=no_judge,
    )
    solution = solution_result["fix_verdict"]
    output["solution"] = solution
    output["solution_detail"] = _solution_detail(solution_result)
    if solution == "unavailable":
        output["case_score"] = STAGE_SCORE[issue] * 0.5
        output["note"] = "solution judge unavailable; diagnosis-only half credit"
    else:
        output["case_score"] = STAGE_SCORE[issue] * (
            0.5 + 0.5 * STAGE_SCORE[solution]
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--level",
        choices=("issue", "solution"),
        default=os.environ.get("PTDBENCH_EVALUATION_LEVEL", "issue"),
    )
    parser.add_argument("--submission")
    parser.add_argument("--gold")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--solution-on-partial", action="store_true")
    args = parser.parse_args()

    verifier_dir = Path(
        os.environ.get("PTDBENCH_VERIFIER", Path(__file__).resolve().parent)
    ).resolve()
    results_dir = Path(os.environ.get("PTDBENCH_RESULTS", "/results")).resolve()
    submission_path = Path(args.submission).resolve() if args.submission else (
        results_dir / "submission.json"
    )
    gold_path = Path(args.gold).resolve() if args.gold else verifier_dir / "gold.yaml"

    try:
        gold = load_gold(gold_path)
        submission = load_submission(submission_path)
        grading = grade(
            args.level,
            gold,
            submission,
            verifier_dir,
            no_judge=args.no_judge,
            solution_on_partial=args.solution_on_partial,
        )
        score = grading.get("case_score")
        passed = score == 1.0
        payload = {
            "passed": passed,
            "score": score,
            "evaluation_level": args.level,
            "metrics": grading,
        }
        atomic_write_json(results_dir / "metrics.json", grading)
        atomic_write_json(results_dir / "result.json", payload)
        return 0 if passed else 1
    except (GradingError, OSError, ValueError) as exc:
        grading = {
            "evaluation_level": args.level,
            "issue": "wrong",
            "solution": None,
            "case_score": 0.0,
            "input_error": str(exc),
        }
        payload = {
            "passed": False,
            "score": 0.0,
            "evaluation_level": args.level,
            "metrics": grading,
        }
        try:
            atomic_write_json(results_dir / "metrics.json", grading)
            atomic_write_json(results_dir / "result.json", payload)
        except OSError:
            pass
        print(f"verifier input error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
