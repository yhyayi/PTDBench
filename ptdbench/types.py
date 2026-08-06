from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class CasePaths:
    # 一个提交包解析后的文件系统布局。
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
    # validator 产出的一条 schema 或目录结构问题。
    severity: str
    path: str
    message: str


@dataclass
class ValidationReport:
    # 汇总所有校验问题，方便贡献者一次性修完。
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
                {"severity": i.severity, "path": i.path, "message": i.message}
                for i in self.issues
            ],
        }


@dataclass(frozen=True)
class CasePackage:
    # 加载后的 package：包含解析好的路径、metadata，以及加载时生成的校验报告。
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
    # 统一后的 agent 描述：先支持 codex 和裸 shell 两种形态，后续可以继续扩展。
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
    # 一次进程执行的轻量记录，agent 和 verifier 共用。
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
    # 给后续阶段预留的 artifact 记录结构。
    name: str
    path: Path
    command: Optional[CommandResult] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationConfig:
    # 调用方选择的运行时配置，和 case package 自身数据分开。
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
    # PTDBench 输出的最终评测摘要。
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
