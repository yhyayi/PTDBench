from __future__ import annotations

import json
from typing import Any, Dict

from ..core.types import TaskPaths, EvaluationPaths, ValidationReport


def validate_task_paths(paths: TaskPaths, strict: bool = True) -> ValidationReport:
    """检查 task 的目录和入口文件。"""
    report = ValidationReport()
    _expect_file(report, paths.metadata_yaml, "metadata.yaml")
    _expect_dir(report, paths.workspace, "workspace")
    _expect_dir(report, paths.environment, "environment")
    if not paths.evaluations:
        report.add("error", "evaluations", "at least one evaluation is required")
    for evaluation in paths.evaluations.values():
        _validate_evaluation_paths(
            report,
            evaluation,
            strict=strict,
            legacy=evaluation.root == paths.root,
        )
    return report


def _validate_evaluation_paths(
    report: ValidationReport,
    evaluation: EvaluationPaths,
    *,
    strict: bool,
    legacy: bool,
) -> None:
    prefix = "" if legacy else f"evaluations/{evaluation.level}/"
    if not legacy:
        _expect_dir(report, evaluation.root, prefix.rstrip("/"))
    _expect_file(report, evaluation.instruction_md, f"{prefix}instruction.md")
    if evaluation.level == "issue":
        _expect_file(report, evaluation.oracle, f"{prefix}oracle.md")
    else:
        _expect_dir(report, evaluation.oracle, f"{prefix}oracle")
    _expect_dir(report, evaluation.verifier, f"{prefix}verifier")
    _expect_file(report, evaluation.verifier_test, f"{prefix}verifier/test.sh")
    if strict:
        _expect_optional_dir(
            report,
            evaluation.verifier_tests,
            f"{prefix}verifier/tests",
        )
        _expect_optional_dir(
            report,
            evaluation.verifier_resources,
            f"{prefix}verifier/resources",
        )


def validate_metadata(metadata: Dict[str, Any]) -> ValidationReport:
    """检查 metadata 中当前 schema 强制要求的字段。"""
    report = ValidationReport()
    required_fields = [
        "title",
        "submitter",
        "license",
        "created_date",
        "domain",
        "model",
        "dataset",
        "training",
        "environment",
        "outcome",
    ]
    for key in required_fields:
        if key not in metadata or metadata[key] in (None, ""):
            report.add("error", f"metadata.{key}", "missing required field")

    for key in ("submitter", "model", "dataset", "training", "environment", "outcome"):
        _expect_mapping(report, metadata.get(key), f"metadata.{key}")

    submitter = metadata.get("submitter") or {}
    if isinstance(submitter, dict):
        _expect_nonempty(report, submitter.get("name"), "metadata.submitter.name")

    model = metadata.get("model") or {}
    if isinstance(model, dict):
        _expect_nonempty(report, model.get("name"), "metadata.model.name")

    dataset = metadata.get("dataset") or {}
    if isinstance(dataset, dict):
        _expect_nonempty(report, dataset.get("name"), "metadata.dataset.name")

    training = metadata.get("training") or {}
    if isinstance(training, dict):
        for key in ("framework", "algo"):
            _expect_nonempty(report, training.get(key), f"metadata.training.{key}")

    environment = metadata.get("environment") or {}
    if isinstance(environment, dict):
        _expect_nonempty(
            report,
            environment.get("dockerfile", "environment/Dockerfile"),
            "metadata.environment.dockerfile",
        )

    outcome = metadata.get("outcome") or {}
    if isinstance(outcome, dict):
        for key in ("failure_type", "failure_reason"):
            _expect_nonempty(report, outcome.get(key), f"metadata.outcome.{key}")
    return report


def dump_report(report: ValidationReport) -> str:
    """把校验结果转为 CLI 可直接打印的 JSON。"""
    return json.dumps(report.to_dict(), indent=2, ensure_ascii=False)


def _expect_file(
    report: ValidationReport, path, label: str, required: bool = True
) -> None:
    if not path.exists():
        report.add("error" if required else "warning", label, "missing file")
    elif not path.is_file():
        report.add("error", label, "expected a file")


def _expect_dir(
    report: ValidationReport, path, label: str, required: bool = True
) -> None:
    if not path.exists():
        report.add("error" if required else "warning", label, "missing directory")
    elif not path.is_dir():
        report.add("error", label, "expected a directory")


def _expect_optional_dir(report: ValidationReport, path, label: str) -> None:
    if path.exists() and not path.is_dir():
        report.add("error", label, "expected a directory")


def _expect_mapping(report: ValidationReport, value: Any, label: str) -> None:
    if not isinstance(value, dict):
        report.add("error", label, "expected a mapping")


def _expect_nonempty(report: ValidationReport, value: Any, label: str) -> None:
    if value in (None, "", [], {}):
        report.add("error", label, "missing required value")
