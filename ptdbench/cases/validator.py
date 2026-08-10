from __future__ import annotations

import json
from typing import Any, Dict

from ..core.types import CasePaths, ValidationReport


def validate_case_paths(paths: CasePaths, strict: bool = True) -> ValidationReport:
    """检查 case 的目录和入口文件。"""
    report = ValidationReport()
    _expect_file(report, paths.instruction_md, "instruction.md")
    _expect_file(report, paths.metadata_yaml, "metadata.yaml")
    _expect_dir(report, paths.workspace, "workspace")
    _expect_dir(report, paths.oracle, "oracle")
    _expect_dir(report, paths.environment, "environment")
    _expect_dir(report, paths.verifier, "verifier")
    _expect_file(report, paths.verifier_test, "verifier/test.sh")
    if strict:
        _expect_dir(report, paths.verifier_tests, "verifier/tests", required=False)
        _expect_dir(report, paths.verifier_resources, "verifier/resources", required=False)
    return report


def validate_metadata(metadata: Dict[str, Any]) -> ValidationReport:
    """检查 metadata 中当前 schema 强制要求的字段。"""
    report = ValidationReport()
    required_fields = [
        "case_id",
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
        for key in ("framework", "algo", "run_command"):
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


def _expect_mapping(report: ValidationReport, value: Any, label: str) -> None:
    if not isinstance(value, dict):
        report.add("error", label, "expected a mapping")


def _expect_nonempty(report: ValidationReport, value: Any, label: str) -> None:
    if value in (None, "", [], {}):
        report.add("error", label, "missing required value")
