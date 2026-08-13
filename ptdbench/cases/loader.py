from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ..core.types import CasePackage, CasePaths, EvaluationPaths, ValidationReport
from ..utils.filesystem import pick_existing
from ..utils.yaml import SimpleYamlError, load_simple_yaml
from .validator import validate_case_paths, validate_metadata, validate_schema_version


class SchemaError(ValueError):
    """case package 无法加载或不满足基本结构。"""


def load_metadata(path: Path) -> Dict[str, Any]:
    """读取 metadata.yaml，并确保顶层是 mapping。"""
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
    """把 case 根目录解析成固定的路径对象。"""
    root = root.resolve()
    readme_md = pick_existing(root, ["readme.md", "README.md"])
    metadata_yaml = pick_existing(root, ["metadata.yaml", "metadata.yml"])
    if metadata_yaml is None:
        raise SchemaError(f"missing metadata.yaml under {root}")

    evaluations = _load_evaluation_paths(root)
    return CasePaths(
        root=root,
        readme_md=readme_md,
        metadata_yaml=metadata_yaml,
        workspace=root / "workspace",
        environment=root / "environment",
        evaluations=evaluations,
    )


def _load_evaluation_paths(root: Path) -> Dict[str, EvaluationPaths]:
    """Load schema v2 evaluations, or synthesize solution for a v1 case."""
    evaluations_root = root / "evaluations"
    evaluations: Dict[str, EvaluationPaths] = {}
    if evaluations_root.is_dir():
        for level in ("issue", "solution"):
            evaluation_root = evaluations_root / level
            if not evaluation_root.exists():
                continue
            oracle = evaluation_root / ("oracle.md" if level == "issue" else "oracle")
            evaluations[level] = _evaluation_paths(level, evaluation_root, oracle)
        if not evaluations:
            raise SchemaError(
                f"{evaluations_root} must contain an issue/ or solution/ evaluation"
            )
        return evaluations

    instruction_md = root / "instruction.md"
    if not instruction_md.exists():
        raise SchemaError(
            f"missing evaluations/ and legacy instruction.md under {root}"
        )
    evaluations["solution"] = _evaluation_paths(
        "solution",
        root,
        root / "oracle",
    )
    return evaluations


def _evaluation_paths(level: str, root: Path, oracle: Path) -> EvaluationPaths:
    verifier = root / "verifier"
    return EvaluationPaths(
        level=level,
        root=root,
        instruction_md=root / "instruction.md",
        oracle=oracle,
        verifier=verifier,
        verifier_test=verifier / "test.sh",
        verifier_tests=verifier / "tests",
        verifier_resources=verifier / "resources",
    )


def load_case_package(root: str | Path, strict: bool = True) -> CasePackage:
    """加载 case，并同时返回路径校验和 metadata 校验结果。"""
    paths = load_case_paths(Path(root))
    path_report = validate_case_paths(paths, strict=strict)
    metadata = load_metadata(paths.metadata_yaml)
    metadata_report = validate_metadata(metadata)
    version_report = validate_schema_version(paths, metadata)
    report = ValidationReport()
    report.issues.extend(path_report.issues)
    report.issues.extend(metadata_report.issues)
    report.issues.extend(version_report.issues)
    return CasePackage(paths=paths, metadata=metadata, report=report)
