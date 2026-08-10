from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from ..agents import AgentExecutionContext, AgentRunner, create_agent_runner, resolve_agent_spec
from ..core.types import AgentSpec, CasePackage, EvaluationConfig, EvaluationResult
from ..cases import SchemaError, load_case_package
from ..runtimes import CommandSpec, EnvironmentArtifact, SandboxRequest, SandboxRuntime, create_runtime
from ..utils.filesystem import copy_tree, ensure_dir, tree_digest, write_text


@dataclass(frozen=True)
class PreparedRun:
    """一次评测的隔离 staging 目录。"""

    root: Path
    visible_case: Path
    workspace: Path
    results: Path


class CaseEvaluator:
    """负责安排 case、agent、runtime 和 verifier 的执行顺序。"""

    def __init__(self, backend: str = "docker"):
        self.backend = backend
        # evaluator 只选择 runtime，不直接处理 Docker 或本机进程细节。
        self.runtime = create_runtime(backend)

    def load(self, case_root: str | Path, strict: bool = True) -> CasePackage:
        return load_case_package(Path(case_root), strict=strict)

    def validate(self, case_root: str | Path, strict: bool = True) -> Dict[str, Any]:
        return self.load(case_root, strict=strict).report.to_dict()

    def evaluate(
        self,
        case_root: str | Path,
        config: EvaluationConfig | None = None,
        strict: bool = True,
    ) -> EvaluationResult:
        """运行一次 agent + verifier 评测。"""
        config = config or EvaluationConfig(backend=self.backend)
        if config.backend != self.backend:
            return CaseEvaluator(config.backend).evaluate(case_root, config=config, strict=strict)

        package = self.load(case_root, strict=strict)
        if not package.report.ok:
            raise SchemaError("case package failed validation")

        agent_spec = resolve_agent_spec(config.agent, config.agent_cmd)
        agent_runner = (
            create_agent_runner(agent_spec, default_backend=self.backend)
            if agent_spec
            else None
        )
        run = self._prepare_run(package, config)
        agent_runtime = create_runtime(agent_runner.backend) if agent_runner else None

        # agent 和 verifier 若使用同一种需要镜像的 runtime，只构建一次镜像。
        artifacts: Dict[str, EnvironmentArtifact] = {}
        for runtime in (agent_runtime, self.runtime):
            if runtime is None or not runtime.needs_environment_image:
                continue
            if runtime.name not in artifacts:
                artifacts[runtime.name] = runtime.build_environment(
                    package.paths.environment,
                    dockerfile="Dockerfile",
                    tag_prefix=config.image_tag_prefix,
                )

        workspace_before_hash = tree_digest(package.paths.workspace)
        agent_result = None
        try:
            if agent_runner is not None:
                context = self._build_agent_context(package, run, config)
                agent_result = self._run_agent(
                    agent_runner,
                    context,
                    runtime=agent_runtime,
                    artifact=artifacts.get(agent_runtime.name) if agent_runtime else None,
                )
                self._ensure_agent_trace(run.results, agent_runner, agent_result)

            verifier_result = self._run_verifier(
                package,
                run,
                config,
                artifact=artifacts.get(self.runtime.name),
                agent_spec=agent_spec,
            )
            workspace_after_hash = tree_digest(run.workspace)
            metrics = self._extract_metrics(run.results, verifier_result)
            result = EvaluationResult(
                case_id=package.case_id,
                passed=bool(metrics.get("passed", verifier_result.returncode == 0)),
                score=self._extract_score(metrics),
                metrics=metrics,
                report_dir=run.results,
                agent_spec=agent_spec,
                agent=agent_result,
                verifier=verifier_result,
                workspace_before_hash=workspace_before_hash,
                workspace_after_hash=workspace_after_hash,
                metadata=package.metadata,
            )
            self._write_report(run.results / "report.json", result)
            return result
        finally:
            if not config.keep_artifacts:
                shutil.rmtree(run.root, ignore_errors=True)

    def _prepare_run(self, package: CasePackage, config: EvaluationConfig) -> PreparedRun:
        """创建运行目录，只向 agent 暴露 instruction 和 workspace 副本。"""
        parent = config.work_dir
        if parent is not None:
            parent = parent.resolve()
            parent.mkdir(parents=True, exist_ok=True)
        root = Path(
            tempfile.mkdtemp(
                prefix="ptdbench-run-",
                dir=None if parent is None else str(parent),
            )
        ).resolve()
        visible_case = ensure_dir(root / "visible_case")
        workspace = ensure_dir(root / "workspace")
        results = ensure_dir(root / "results")
        copy_tree(package.paths.workspace, workspace)
        write_text(
            visible_case / "instruction.md",
            package.paths.instruction_md.read_text(encoding="utf-8"),
        )
        return PreparedRun(root, visible_case, workspace, results)

    def _build_agent_context(
        self,
        package: CasePackage,
        run: PreparedRun,
        config: EvaluationConfig,
    ) -> AgentExecutionContext:
        return AgentExecutionContext(
            case_id=package.case_id,
            instruction=package.paths.instruction_md.read_text(encoding="utf-8"),
            metadata=dict(package.metadata),
            timeout_s=config.timeout_s,
            case_root=package.paths.root,
            workspace=run.workspace,
            visible_case=run.visible_case,
            results=run.results,
        )

    def _run_agent(
        self,
        runner: AgentRunner,
        context: AgentExecutionContext,
        runtime: Optional[SandboxRuntime],
        artifact: Optional[EnvironmentArtifact],
    ) -> Any:
        if runtime is None:
            raise ValueError("agent runtime is required when an agent is configured")
        return runner.run(context, runtime=runtime, artifact=artifact)

    def _ensure_agent_trace(self, results_dir: Path, runner: AgentRunner, result: Any) -> None:
        trace_path = results_dir / "agent_trace.json"
        if trace_path.exists():
            return
        payload = {
            "kind": runner.kind,
            "backend": runner.backend,
            "result": result.to_dict() if hasattr(result, "to_dict") else result,
        }
        write_text(trace_path, json.dumps(payload, indent=2, ensure_ascii=False))

    def _run_verifier(
        self,
        package: CasePackage,
        run: PreparedRun,
        config: EvaluationConfig,
        artifact: Optional[EnvironmentArtifact],
        agent_spec: Optional[AgentSpec],
    ) -> Any:
        # 新 verifier 只依赖标准环境变量。这里暂时保留旧 Judge-B case 的
        # 配置 handoff，以免迁移历史 case 时突然失效；新 case 不应使用它。
        request = SandboxRequest(
            stage="verifier",
            case_id=package.case_id,
            host_case_root=package.paths.root,
            host_visible_case=run.visible_case,
            host_workspace=run.workspace,
            host_results=run.results,
            env=self._legacy_verifier_handoff_env(agent_spec),
            network=config.verifier_network,
            timeout_s=config.timeout_s,
            workdir="case_root",
        )
        command = CommandSpec(
            shell=config.verifier_cmd or "bash verifier/test.sh",
            timeout_s=config.timeout_s,
        )
        return self.runtime.run(request, command, artifact=artifact)

    def _legacy_verifier_handoff_env(self, agent_spec: Optional[AgentSpec]) -> Dict[str, str]:
        """兼容旧 verifier 的 Judge-B 配置；它不是新的 agent workflow 接口。"""
        if agent_spec is None:
            return {}
        env = {"PTDBENCH_CENTRAL_AGENT_KIND": agent_spec.kind}
        if agent_spec.kind == "codex":
            env["PTDBENCH_JUDGE_PROVIDER"] = "openai-compatible"
        if agent_spec.model:
            env["PTDBENCH_JUDGE_MODEL"] = agent_spec.model
        if agent_spec.base_url:
            env["PTDBENCH_JUDGE_BASE_URL"] = agent_spec.base_url
        if agent_spec.api_key:
            env["PTDBENCH_JUDGE_API_KEY_ENV"] = "PTDBENCH_JUDGE_API_KEY"
            env["PTDBENCH_JUDGE_API_KEY"] = agent_spec.api_key
        return env

    def _write_report(self, path: Path, result: EvaluationResult) -> None:
        write_text(path, json.dumps(result.to_dict(), indent=2, ensure_ascii=False))

    def _extract_metrics(self, results_dir: Path, verifier_result: Any) -> Dict[str, Any]:
        """优先读取 verifier 的 JSON 结果，否则退回退出码。"""
        metrics: Dict[str, Any] = {}
        for filename in ("result.json", "metrics.json"):
            path = results_dir / filename
            if not path.exists():
                continue
            try:
                with open(path, "r", encoding="utf-8") as file:
                    payload = json.load(file)
            except Exception as exc:
                metrics["result_parse_error"] = str(exc)
                continue
            if not isinstance(payload, dict):
                continue
            nested = payload.get("metrics")
            if isinstance(nested, dict):
                metrics.update(nested)
            for key in ("passed", "score"):
                if key in payload:
                    metrics[key] = payload[key]
            for key, value in payload.items():
                if key not in {"metrics", "passed", "score"} and key not in metrics:
                    metrics[key] = value
        if verifier_result is None:
            metrics["passed"] = False
        else:
            metrics.setdefault("passed", verifier_result.returncode == 0)
        return metrics

    @staticmethod
    def _extract_score(metrics: Dict[str, Any]) -> Optional[float]:
        if isinstance(metrics.get("score"), (int, float)):
            return float(metrics["score"])
        if metrics.get("passed") is True:
            return 1.0
        if metrics.get("passed") is False:
            return 0.0
        return None
