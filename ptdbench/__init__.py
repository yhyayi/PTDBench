"""PTDBench 评测框架。"""

from .agents import AgentExecutionContext, AgentRunner, create_agent_runner, register_agent_runner
from .core.types import (
    AgentSpec,
    CasePackage,
    EvaluationConfig,
    EvaluationResult,
    ValidationIssue,
    ValidationReport,
)
from .cases import load_case_package
from .evaluation.evaluator import CaseEvaluator
from .runtimes import (
    CommandSpec,
    DockerRuntime,
    EnvironmentArtifact,
    LocalRuntime,
    SandboxRequest,
    SandboxRuntime,
    create_runtime,
)

__all__ = [
    "AgentRunner",
    "AgentExecutionContext",
    "CaseEvaluator",
    "EvaluationConfig",
    "EvaluationResult",
    "CommandSpec",
    "SandboxRuntime",
    "SandboxRequest",
    "EnvironmentArtifact",
    "LocalRuntime",
    "DockerRuntime",
    "create_runtime",
    "AgentSpec",
    "CasePackage",
    "ValidationIssue",
    "ValidationReport",
    "create_agent_runner",
    "load_case_package",
    "register_agent_runner",
]
