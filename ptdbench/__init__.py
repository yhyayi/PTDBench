"""PTDBench 评测框架。"""

from .agents import AgentExecutionContext, AgentRunner, create_agent_runner, register_agent_runner
from .evaluator import CaseEvaluator, EvaluationConfig, EvaluationResult
from .schema import load_case_package
from .types import AgentSpec, CasePackage, ValidationIssue, ValidationReport

__all__ = [
    "AgentRunner",
    "AgentExecutionContext",
    "CaseEvaluator",
    "EvaluationConfig",
    "EvaluationResult",
    "AgentSpec",
    "CasePackage",
    "ValidationIssue",
    "ValidationReport",
    "create_agent_runner",
    "load_case_package",
    "register_agent_runner",
]
