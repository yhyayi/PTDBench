from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class CasePaths:
    """一个 case package 解析后的文件系统布局。"""

    root: Path
    instruction_md: Path
    readme_md: Optional[Path]
    metadata_yaml: Path
    workspace: Path
    oracle: Path
    environment: Path
    verifier: Path
    verifier_test: Path
    verifier_tests: Optional[Path]
    verifier_resources: Optional[Path]


@dataclass(frozen=True)
class ValidationIssue:
    """validator 产出的一条目录或 schema 问题。"""

    severity: str
    path: str
    message: str


@dataclass
class ValidationReport:
    """汇总所有校验问题，方便贡献者一次性修复。"""

    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def add(self, severity: str, path: str, message: str) -> None:
        self.issues.append(ValidationIssue(severity=severity, path=path, message=message))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "issues": [
                {"severity": item.severity, "path": item.path, "message": item.message}
                for item in self.issues
            ],
        }


@dataclass(frozen=True)
class CasePackage:
    """加载后的 case：路径、metadata 和校验报告。"""

    paths: CasePaths
    metadata: Dict[str, Any]
    report: ValidationReport

    @property
    def case_id(self) -> str:
        for key in ("case_id", "id", "task_id"):
            value = self.metadata.get(key)
            if isinstance(value, str) and value:
                return value
        return self.paths.root.name


@dataclass(frozen=True)
class AgentSpec:
    """统一后的 agent 配置。"""

    kind: str = "codex"
    command: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    extra_args: Sequence[str] = field(default_factory=tuple)
    backend: Optional[str] = None
    network: str = "bridge"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "command": self.command,
            "model": self.model,
            "base_url": self.base_url,
            "api_key": None if self.api_key is None else "<redacted>",
            "extra_args": list(self.extra_args),
            "backend": self.backend,
            "network": self.network,
        }


@dataclass(frozen=True)
class CommandResult:
    """一次命令执行的统一结果，agent 和 verifier 共用。"""

    argv: List[str]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_s: float
    cwd: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "argv": self.argv,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timed_out": self.timed_out,
            "duration_s": self.duration_s,
            "cwd": self.cwd,
        }


@dataclass(frozen=True)
class StageArtifact:
    """给后续评测阶段预留的 artifact 记录结构。"""

    name: str
    path: Path
    command: Optional[CommandResult] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationConfig:
    """调用方运行时配置，与 case package 自身数据分开。"""

    agent: Optional[AgentSpec] = None
    agent_cmd: Optional[str] = None
    verifier_cmd: Optional[str] = None
    backend: str = "docker"
    verifier_network: str = "none"
    timeout_s: Optional[int] = None
    keep_artifacts: bool = False
    image_tag_prefix: str = "ptdbench"
    work_dir: Optional[Path] = None


@dataclass(frozen=True)
class EvaluationResult:
    """PTDBench 输出的最终评测摘要。"""

    case_id: str
    passed: bool
    score: Optional[float]
    metrics: Dict[str, Any]
    report_dir: Path
    agent_spec: Optional[AgentSpec]
    agent: Optional[CommandResult]
    verifier: Optional[CommandResult]
    workspace_before_hash: str
    workspace_after_hash: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "passed": self.passed,
            "score": self.score,
            "metrics": self.metrics,
            "report_dir": str(self.report_dir),
            "agent_spec": None if self.agent_spec is None else self.agent_spec.to_dict(),
            "agent": None if self.agent is None else self.agent.to_dict(),
            "verifier": None if self.verifier is None else self.verifier.to_dict(),
            "workspace_before_hash": self.workspace_before_hash,
            "workspace_after_hash": self.workspace_after_hash,
            "metadata": self.metadata,
        }
