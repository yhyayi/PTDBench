from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .filesystem import pick_existing
from .simpleyaml import SimpleYamlError, load_simple_yaml
from .types import CasePackage, CasePaths, ValidationReport


class SchemaError(ValueError):
    pass


def load_metadata(path: Path) -> Dict[str, Any]:
    # metadata 用仓库内置的小 YAML 解析器读取，避免第一版引入额外依赖。
    if not path.exists():
        raise SchemaError(f"missing metadata file: {path}")
    try:
        data = load_simple_yaml(str(path))
    except SimpleYamlError as exc:
        raise SchemaError(f"failed to parse {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SchemaError(f"{path} must contain a mapping at the top level")
    return data


def load_case_paths(root: Path) -> CasePaths:
    # 文件名允许少量兼容写法，但核心目录必须固定，方便 evaluator 维护
    # agent-visible 和 verifier-only 的边界。
    root = root.resolve()
    instruction_md = pick_existing(root, ["instruction.md"])
    if instruction_md is None:
        raise SchemaError(f"missing instruction.md under {root}")
    readme_md = pick_existing(root, ["readme.md", "README.md"])
    metadata_yaml = pick_existing(root, ["metadata.yaml", "metadata.yml"])
    if metadata_yaml is None:
        raise SchemaError(f"missing metadata.yaml under {root}")
    workspace = root / "workspace"
    oracle = root / "oracle"
    environment = root / "environment"
    verifier = root / "verifier"
    verifier_test = verifier / "test.sh"
    verifier_tests = verifier / "tests"
    verifier_resources = verifier / "resources"
    return CasePaths(
        root=root,
        instruction_md=instruction_md,
        readme_md=readme_md,
        metadata_yaml=metadata_yaml,
        workspace=workspace,
        oracle=oracle,
        environment=environment,
        verifier=verifier,
        verifier_test=verifier_test,
        verifier_tests=verifier_tests,
        verifier_resources=verifier_resources,
    )


def validate_case_paths(paths: CasePaths, strict: bool = True) -> ValidationReport:
    report = ValidationReport()
    # case package 必须包含公开 instruction、可写 workspace，以及隐藏 verifier 边界。
    _expect_file(report, paths.instruction_md, "instruction.md")
    _expect_file(report, paths.metadata_yaml, "metadata.yaml")
    _expect_dir(report, paths.workspace, "workspace")
    _expect_dir(report, paths.oracle, "oracle")
    _expect_dir(report, paths.environment, "environment")
    _expect_dir(report, paths.verifier, "verifier")
    _expect_file(report, paths.verifier_test, "verifier/test.sh")
    if strict:
        _expect_dir(report, paths.verifier / "tests", "verifier/tests", required=False)
        _expect_dir(report, paths.verifier / "resources", "verifier/resources", required=False)
    return report


def validate_metadata(metadata: Dict[str, Any]) -> ValidationReport:
    report = ValidationReport()
    # 第一版只强制真正需要的字段；之后可以渐进扩展，尽量不打破旧模板。
    required_top_level = [
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
    for key in required_top_level:
        if key not in metadata or metadata[key] in (None, ""):
            report.add("error", f"metadata.{key}", "missing required field")

    _expect_mapping(report, metadata.get("submitter"), "metadata.submitter")
    _expect_mapping(report, metadata.get("model"), "metadata.model")
    _expect_mapping(report, metadata.get("dataset"), "metadata.dataset")
    _expect_mapping(report, metadata.get("training"), "metadata.training")
    _expect_mapping(report, metadata.get("environment"), "metadata.environment")
    _expect_mapping(report, metadata.get("outcome"), "metadata.outcome")

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
        _expect_nonempty(report, environment.get("dockerfile", "environment/Dockerfile"), "metadata.environment.dockerfile")

    outcome = metadata.get("outcome") or {}
    if isinstance(outcome, dict):
        _expect_nonempty(report, outcome.get("failure_type"), "metadata.outcome.failure_type")
        _expect_nonempty(report, outcome.get("failure_reason"), "metadata.outcome.failure_reason")
    return report


def load_case_package(root: Path, strict: bool = True) -> CasePackage:
    # 先解析路径，再汇总路径错误和 metadata 错误，避免遇到第一个缺失文件
    # 就直接中断。
    paths = load_case_paths(root)
    path_report = validate_case_paths(paths, strict=strict)
    metadata = load_metadata(paths.metadata_yaml)
    meta_report = validate_metadata(metadata)
    report = ValidationReport()
    report.issues.extend(path_report.issues)
    report.issues.extend(meta_report.issues)
    return CasePackage(paths=paths, metadata=metadata, report=report)


def _expect_file(report: ValidationReport, path: Path, label: str, required: bool = True) -> None:
    if not path.exists():
        level = "error" if required else "warning"
        report.add(level, label, "missing file")
    elif not path.is_file():
        report.add("error", label, "expected a file")


def _expect_dir(report: ValidationReport, path: Path, label: str, required: bool = True) -> None:
    if not path.exists():
        level = "error" if required else "warning"
        report.add(level, label, "missing directory")
    elif not path.is_dir():
        report.add("error", label, "expected a directory")


def _expect_mapping(report: ValidationReport, value: Any, label: str) -> None:
    if not isinstance(value, dict):
        report.add("error", label, "expected a mapping")


def _expect_nonempty(report: ValidationReport, value: Any, label: str) -> None:
    if value in (None, "", [], {}):
        report.add("error", label, "missing required value")


def dump_report(report: ValidationReport) -> str:
    # 方便 CLI 输出和 debug 日志复用。
    return json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
