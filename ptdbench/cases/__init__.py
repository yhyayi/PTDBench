"""case package 的加载和格式校验。"""

from .loader import SchemaError, load_case_package, load_case_paths, load_metadata
from .validator import dump_report, validate_case_paths, validate_metadata

__all__ = [
    "SchemaError",
    "load_case_package",
    "load_case_paths",
    "load_metadata",
    "validate_case_paths",
    "validate_metadata",
    "dump_report",
]
