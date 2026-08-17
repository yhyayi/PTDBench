from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from ..agents import AgentExecutionContext, AgentRunner, create_agent_runner, resolve_agent_spec
from ..core.types import (
    AgentSpec,
    TaskPackage,
    CommandResult,
    EvaluationConfig,
    EvaluationPaths,
    EvaluationResult,
)
from ..tasks import SchemaError, load_task_package
from ..runtimes import (
    CommandSpec,
    EnvironmentArtifact,
    SandboxRequest,
    SandboxRuntime,
    create_runtime,
)
from ..utils.filesystem import copy_tree, ensure_dir, tree_digest, write_text


@dataclass(frozen=True)
class PreparedRun:
    """一次评测的隔离 staging 目录。"""

    root: Path
    visible_task: Path
    workspace: Path
    results: Path


@dataclass(frozen=True)
class AgentRunOutcome:
    setup: CommandResult
    execution: Optional[CommandResult]


class TaskEvaluator:
    """负责安排 task、agent、runtime 和 verifier 的执行顺序。"""

    def __init__(self, backend: str = "docker"):
        self.backend = backend
        self.log_level = "quiet"
        # evaluator 只选择 runtime，不直接处理 Docker 或本机进程细节。
        self.runtime = create_runtime(backend)

    def load(self, task_root: str | Path, strict: bool = True) -> TaskPackage:
        return load_task_package(Path(task_root), strict=strict)

    def validate(self, task_root: str | Path, strict: bool = True) -> Dict[str, Any]:
        return self.load(task_root, strict=strict).report.to_dict()

    def evaluate(
        self,
        task_root: str | Path,
        config: EvaluationConfig | None = None,
        strict: bool = True,
    ) -> EvaluationResult:
        """运行一次 agent + verifier 评测。"""
        config = config or EvaluationConfig(backend=self.backend)
        self.log_level = config.log_level
        if self.log_level not in {"quiet", "debug"}:
            raise ValueError("log_level must be 'quiet' or 'debug'")
        if config.backend != self.backend:
            return TaskEvaluator(config.backend).evaluate(task_root, config=config, strict=strict)

        package = self.load(task_root, strict=strict)
        if not package.report.ok:
            raise SchemaError("task package failed validation")
        evaluation = self._select_evaluation(package, config.evaluation_level)

        agent_spec = resolve_agent_spec(config.agent, config.agent_cmd)
        agent_runner = create_agent_runner(agent_spec) if agent_spec else None
        run = self._prepare_run(package, evaluation, config)
        self._append_run_log(
            run.results,
            f"PTDBench task={package.task_id} backend={self.runtime.name} "
            f"evaluation_level={evaluation.level} "
            f"agent={agent_runner.kind if agent_runner else 'none'}",
        )
        try:
            artifact = None
            if self.runtime.needs_environment_image:
                self._append_run_log(
                    run.results,
                    f"[START] environment_prepare backend={self.runtime.name}",
                )
                artifact = self.runtime.build_environment(
                    package.paths.environment,
                    dockerfile=self._environment_dockerfile(package),
                    tag_prefix=config.image_tag_prefix,
                )
                self._append_run_log(
                    run.results,
                    f"[OK] environment_prepare backend={self.runtime.name} "
                    f"artifact={artifact.reference or '-'}",
                )

            workspace_before_hash = tree_digest(package.paths.workspace)
            agent_setup_result = None
            agent_result = None
            if agent_runner is not None:
                context = self._build_agent_context(package, evaluation, run, config)
                outcome = self._run_agent(
                    agent_runner,
                    context,
                    runtime=self.runtime,
                    artifact=artifact,
                )
                agent_setup_result = outcome.setup
                agent_result = outcome.execution
                self._ensure_agent_trace(
                    run.results,
                    agent_runner,
                    agent_result or agent_setup_result,
                    stage="agent" if agent_result is not None else "agent_setup",
                    backend=self.runtime.name,
                )

            verifier_result = self._run_verifier(
                package,
                evaluation,
                run,
                config,
                artifact=artifact,
                agent_spec=agent_spec,
            )
            workspace_after_hash = tree_digest(run.workspace)
            metrics = self._extract_metrics(run.results, verifier_result)
            stages_ok = (
                (agent_setup_result is None or agent_setup_result.returncode == 0)
                and (agent_result is None or agent_result.returncode == 0)
            )
            result = EvaluationResult(
                task_id=package.task_id,
                evaluation_level=evaluation.level,
                passed=stages_ok
                and bool(metrics.get("passed", verifier_result.returncode == 0)),
                score=self._extract_score(metrics) if stages_ok else 0.0,
                metrics=metrics,
                report_dir=run.results,
                agent_spec=agent_spec,
                agent_setup=agent_setup_result,
                agent=agent_result,
                verifier=verifier_result,
                workspace_before_hash=workspace_before_hash,
                workspace_after_hash=workspace_after_hash,
                metadata=package.metadata,
            )
            self._write_report(run.results / "report.json", result)
            self._append_run_log(
                run.results,
                f"[RESULT] passed={str(result.passed).lower()} score={result.score}",
            )
            return result
        except Exception as exc:
            self._append_run_log(
                run.results,
                f"[ERROR] orchestration {type(exc).__name__}: {exc}",
            )
            raise
        finally:
            if not config.keep_artifacts:
                shutil.rmtree(run.root, ignore_errors=True)

    @staticmethod
    def _select_evaluation(
        package: TaskPackage,
        level: str,
    ) -> EvaluationPaths:
        try:
            return package.paths.evaluations[level]
        except KeyError as exc:
            available = ", ".join(package.evaluation_levels) or "none"
            raise SchemaError(
                f"evaluation level {level!r} is unavailable; available levels: {available}"
            ) from exc

    def _prepare_run(
        self,
        package: TaskPackage,
        evaluation: EvaluationPaths,
        config: EvaluationConfig,
    ) -> PreparedRun:
        """创建运行目录，只向 agent 暴露 instruction 和 workspace 副本。"""
        parent = config.work_dir
        if parent is not None:
            parent = parent.resolve()
            parent.mkdir(parents=True, exist_ok=True)
        root = Path(
            tempfile.mkdtemp(
                prefix=self._run_directory_prefix(package.task_id, evaluation.level),
                dir=None if parent is None else str(parent),
            )
        ).resolve()
        visible_task = ensure_dir(root / "visible_task")
        workspace = ensure_dir(root / "workspace")
        results = ensure_dir(root / "results")
        copy_tree(package.paths.workspace, workspace)
        write_text(
            visible_task / "instruction.md",
            evaluation.instruction_md.read_text(encoding="utf-8"),
        )
        return PreparedRun(root, visible_task, workspace, results)

    @staticmethod
    def _run_directory_prefix(task_id: str, evaluation_level: str) -> str:
        """Build a readable, sortable prefix while retaining mkdtemp uniqueness."""
        safe_task_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", task_id).strip("-._")
        safe_task_id = safe_task_id or "task"
        timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
        return f"ptdbench-run-{safe_task_id}-{evaluation_level}-{timestamp}-"

    def _build_agent_context(
        self,
        package: TaskPackage,
        evaluation: EvaluationPaths,
        run: PreparedRun,
        config: EvaluationConfig,
    ) -> AgentExecutionContext:
        return AgentExecutionContext(
            task_id=package.task_id,
            instruction=evaluation.instruction_md.read_text(encoding="utf-8"),
            metadata=dict(package.metadata),
            timeout_s=config.timeout_s,
            task_root=package.paths.root,
            workspace=run.workspace,
            visible_task=run.visible_task,
            results=run.results,
            evaluation_level=evaluation.level,
            log_level=config.log_level,
        )

    def _run_agent(
        self,
        runner: AgentRunner,
        context: AgentExecutionContext,
        runtime: Optional[SandboxRuntime],
        artifact: Optional[EnvironmentArtifact],
    ) -> AgentRunOutcome:
        if runtime is None:
            raise ValueError("agent runtime is required when an agent is configured")
        request = context.sandbox_request(
            runtime,
            network=runner.spec.network,
        )
        if artifact is None:
            if runtime.needs_environment_image:
                raise ValueError("agent runtime needs a built environment artifact")
            artifact = EnvironmentArtifact(backend=runtime.name)
        sandbox = runtime.create_sandbox(request, artifact=artifact)
        try:
            self._append_run_log(context.results, "[START] agent_setup")
            setup_result = runner.setup(context, runtime=runtime, sandbox=sandbox)
            self._write_stage_logs(context.results, "agent_setup", setup_result)
            if setup_result.returncode != 0:
                self._append_command_summary(
                    context.results,
                    "agent_setup",
                    setup_result,
                )
                self._append_run_log(
                    context.results,
                    "[SKIP] agent reason=agent_setup_failed",
                )
                return AgentRunOutcome(setup=setup_result, execution=None)
            self._append_command_summary(context.results, "agent_setup", setup_result)
            self._append_run_log(context.results, "[START] agent")
            execution = runner.run(
                context,
                runtime=runtime,
                artifact=artifact,
                sandbox=sandbox,
            )
            self._write_stage_logs(context.results, "agent", execution)
            self._append_command_summary(context.results, "agent", execution)
            return AgentRunOutcome(setup=setup_result, execution=execution)
        finally:
            runtime.close_sandbox(sandbox)

    def _ensure_agent_trace(
        self,
        results_dir: Path,
        runner: AgentRunner,
        result: Any,
        *,
        stage: str,
        backend: str,
    ) -> None:
        trace_path = results_dir / "agent_trace.json"
        if trace_path.exists():
            return
        payload = {
            "kind": runner.kind,
            "backend": backend,
            "stage": stage,
            "result": result.to_dict() if hasattr(result, "to_dict") else result,
        }
        write_text(trace_path, json.dumps(payload, indent=2, ensure_ascii=False))

    def _run_verifier(
        self,
        package: TaskPackage,
        evaluation: EvaluationPaths,
        run: PreparedRun,
        config: EvaluationConfig,
        artifact: Optional[EnvironmentArtifact],
        agent_spec: Optional[AgentSpec],
    ) -> Any:
        request = SandboxRequest(
            stage="verifier",
            task_id=package.task_id,
            host_task_root=evaluation.root,
            host_visible_task=run.visible_task,
            host_workspace=run.workspace,
            host_results=run.results,
            evaluation_level=evaluation.level,
            host_oracle=evaluation.oracle,
            host_verifier=evaluation.verifier,
            env=self._verifier_judge_env(agent_spec),
            network=config.verifier_network,
            timeout_s=config.timeout_s,
            workdir="task_root",
        )
        command = CommandSpec(
            shell=config.verifier_cmd or 'bash "$PTDBENCH_VERIFIER/test.sh"',
            timeout_s=config.timeout_s,
        )
        if artifact is None:
            if self.runtime.needs_environment_image:
                raise ValueError("verifier runtime needs a built environment artifact")
            artifact = EnvironmentArtifact(backend=self.runtime.name)
        sandbox = self.runtime.create_sandbox(request, artifact=artifact)
        try:
            self._append_run_log(run.results, "[START] verifier")
            result = self.runtime.exec(sandbox, command)
            self._write_stage_logs(run.results, "verifier", result)
            self._append_command_summary(run.results, "verifier", result)
            return result
        finally:
            self.runtime.close_sandbox(sandbox)

    def _append_run_log(self, results_dir: Path, line: str) -> None:
        path = results_dir / "run.log"
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        write_text(path, existing + line.rstrip() + "\n")
        if self.log_level == "debug":
            print(line.rstrip(), file=sys.stderr, flush=True)

    @staticmethod
    def _write_stage_logs(
        results_dir: Path,
        stage: str,
        result: CommandResult,
    ) -> None:
        write_text(results_dir / f"{stage}.stdout.log", result.stdout or "")
        write_text(results_dir / f"{stage}.stderr.log", result.stderr or "")

    def _append_command_summary(
        self,
        results_dir: Path,
        stage: str,
        result: CommandResult,
    ) -> None:
        status = "OK" if result.returncode == 0 and not result.timed_out else "FAIL"
        self._append_run_log(
            results_dir,
            f"[{status}] {stage} exit={result.returncode} "
            f"timed_out={str(result.timed_out).lower()} "
            f"duration={result.duration_s:.2f}s",
        )
        error = (result.stderr or "").strip()
        if not error and result.returncode != 0:
            error = (result.stdout or "").strip()
        if error:
            tail = " | ".join(error.splitlines()[-5:])
            self._append_run_log(results_dir, f"  error_tail: {tail}")
        if status == "FAIL":
            self._append_run_log(
                results_dir,
                f"  logs: {stage}.stdout.log, {stage}.stderr.log",
            )

    def _verifier_judge_env(self, agent_spec: Optional[AgentSpec]) -> Dict[str, str]:
        """Expose the configured model endpoint to API-backed verifiers."""
        if agent_spec is None:
            return {}
        env = {
            "PTDBENCH_CENTRAL_AGENT_KIND": agent_spec.kind,
            "PTDBENCH_JUDGE_PROVIDER": "openai-compatible",
        }
        judge_model = agent_spec.judge_model or agent_spec.model
        if judge_model:
            env["PTDBENCH_JUDGE_MODEL"] = judge_model
        env["OPENAI_BASE_URL"] = agent_spec.base_url or "https://api.openai.com/v1"
        if agent_spec.api_key:
            env["OPENAI_API_KEY"] = agent_spec.api_key
        return env

    @staticmethod
    def _environment_dockerfile(package: TaskPackage) -> str:
        """Resolve metadata's task-relative Dockerfile to a context-relative path."""
        environment = package.metadata.get("environment")
        declared = environment.get("dockerfile") if isinstance(environment, dict) else None
        value = str(declared or "environment/Dockerfile")
        prefix = "environment/"
        if value == "environment":
            value = "Dockerfile"
        elif value.startswith(prefix):
            value = value[len(prefix) :]
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise SchemaError(f"metadata.environment.dockerfile must stay under environment/: {value}")
        if not (package.paths.environment / path).is_file():
            raise SchemaError(f"Dockerfile not found: {package.paths.environment / path}")
        return value

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
