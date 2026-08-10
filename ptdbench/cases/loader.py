from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ..core.types import CasePackage, CasePaths, ValidationReport
from ..utils.filesystem import pick_existing
from ..utils.yaml import SimpleYamlError, load_simple_yaml
from .validator import validate_case_paths, validate_metadata


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
    instruction_md = pick_existing(root, ["instruction.md"])
    if instruction_md is None:
        raise SchemaError(f"missing instruction.md under {root}")
    readme_md = pick_existing(root, ["readme.md", "README.md"])
    metadata_yaml = pick_existing(root, ["metadata.yaml", "metadata.yml"])
    if metadata_yaml is None:
        raise SchemaError(f"missing metadata.yaml under {root}")

    verifier = root / "verifier"
    return CasePaths(
        root=root,
        instruction_md=instruction_md,
        readme_md=readme_md,
        metadata_yaml=metadata_yaml,
        workspace=root / "workspace",
        oracle=root / "oracle",
        environment=root / "environment",
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
    report = ValidationReport()
    report.issues.extend(path_report.issues)
    report.issues.extend(metadata_report.issues)
    return CasePackage(paths=paths, metadata=metadata, report=report)
