from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .agents import AgentExecutionContext, AgentRunner, create_agent_runner, resolve_agent_spec
from .executor import CommandSpec, DockerExecutor, DockerImageSpec, DockerMount, LocalExecutor
from .filesystem import copy_tree, ensure_dir, tree_digest, write_text
from .schema import SchemaError, load_case_package
from .types import CasePackage, EvaluationConfig, EvaluationResult


@dataclass(frozen=True)
class PreparedRun:
    # 每次评测都有独立的 staging 目录，避免 workspace 副本、可见题面、
    # 以及结果文件和其他评测互相污染。
    root: Path
    visible_case: Path
    workspace: Path
    results: Path


class CaseEvaluator:
    def __init__(self, backend: str = "docker"):
        if backend not in {"docker", "local"}:
            raise ValueError("backend must be 'docker' or 'local'")
        self.backend = backend
        self.local = LocalExecutor()
        self.docker = DockerExecutor()

    def load(self, case_root: str | Path, strict: bool = True) -> CasePackage:
        return load_case_package(Path(case_root), strict=strict)

    def validate(self, case_root: str | Path, strict: bool = True) -> Dict[str, Any]:
        package = self.load(case_root, strict=strict)
        return package.report.to_dict()

    def evaluate(
        self,
        case_root: str | Path,
        config: EvaluationConfig | None = None,
        strict: bool = True,
    ) -> EvaluationResult:
        # 对外保持同一套 API；如果调用时切换 backend，就转给对应的
        # evaluator 实例处理。
        config = config or EvaluationConfig(backend=self.backend)
        if config.backend != self.backend:
            return CaseEvaluator(backend=config.backend).evaluate(case_root, config=config, strict=strict)
        package = self.load(case_root, strict=strict)
        if not package.report.ok:
            raise SchemaError("case package failed validation")
        agent_spec = resolve_agent_spec(config.agent, config.agent_cmd)
        agent_runner = create_agent_runner(agent_spec, default_backend=self.backend) if agent_spec else None
        # 所有可变状态都放进临时 staging 目录，agent 的修改可以被检查，
        # 但不会碰到原始 case package。
        run = self._prepare_run(package, config)
        needs_image = self.backend == "docker" or (
            agent_runner is not None and agent_runner.needs_environment_image
        )
        image_tag = self._build_image(package, prefix=config.image_tag_prefix) if needs_image else None
        # 在 agent 运行前先记录原始 workspace 的哈希。
        workspace_before_hash = tree_digest(package.paths.workspace)
        agent_result = None
        verifier_result = None
        result: EvaluationResult | None = None
        try:
            if agent_runner is not None:
                agent_context = self._build_agent_context(package, run, config)
                agent_result = self._run_agent(
                    agent_runner,
                    agent_context,
                    image_tag=image_tag,
                )
                self._ensure_agent_trace(run.results, agent_runner, agent_result)
            verifier_result = self._run_verifier(package, run, config, image_tag=image_tag)
            # agent 修改后重新计算 staged workspace 的哈希，用稳定方式记录
            # 这次修复是否真的改动了文件。
            workspace_after_hash = tree_digest(run.workspace)
            metrics = self._extract_metrics(run.results, verifier_result)
            score = self._extract_score(metrics)
            passed = bool(metrics.get("passed", verifier_result is not None and verifier_result.returncode == 0))
            result = EvaluationResult(
                case_id=package.case_id,
                passed=passed,
                score=score,
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
        # 公开题面和可写 workspace 分开放。agent 只拿到 workspace 副本
        # 以及 instruction.md；readme.md 是纯 human-review 文档，不进运行时。
        # 默认用系统临时目录；如果用户传 work_dir，就把本次 run 放到那里，
        # 方便断点调试或长期保留中间产物。
        root_parent = config.work_dir
        if root_parent is not None:
            root_parent = root_parent.resolve()
            root_parent.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="ptdbench-run-", dir=None if root_parent is None else str(root_parent))).resolve()
        visible_case = ensure_dir(root / "visible_case")
        workspace = ensure_dir(root / "workspace")
        results = ensure_dir(root / "results")

        copy_tree(package.paths.workspace, workspace)
        write_text(visible_case / "instruction.md", package.paths.instruction_md.read_text(encoding="utf-8"))
        return PreparedRun(root=root, visible_case=visible_case, workspace=workspace, results=results)

    def _build_image(self, package: CasePackage, prefix: str = "ptdbench") -> str:
        # 环境镜像只来自 environment/Dockerfile，任务作者可以放依赖，
        # 但不应该把 verifier 逻辑烘进 agent 可见环境。
        env = package.paths.environment
        image = DockerImageSpec(
            context_dir=env,
            dockerfile="Dockerfile",
            tag_prefix=prefix,
        )
        return self.docker.build(image)

    def _build_agent_context(
        self,
        package: CasePackage,
        run: PreparedRun,
        config: EvaluationConfig,
    ) -> AgentExecutionContext:
        # evaluator 只负责把 staging 后的路径整理成通用上下文；
        # 具体 agent 如何消费这些信息，交给 runner 自己决定。
        return AgentExecutionContext(
            case_id=package.case_id,
            instruction=package.paths.instruction_md.read_text(encoding="utf-8"),
            metadata=dict(package.metadata),
            timeout_s=config.timeout_s,
            local_workspace=run.workspace,
            local_visible_case=run.visible_case,
            local_results=run.results,
            local_env=self._common_env(package, run, stage="agent", container=False),
            container_env=self._common_env(package, run, stage="agent", container=True),
        )

    def _run_agent(
        self,
        runner: AgentRunner,
        context: AgentExecutionContext,
        image_tag: Optional[str],
    ) -> Any:
        return runner.run(context, local=self.local, docker=self.docker, image_tag=image_tag)

    def _ensure_agent_trace(self, results_dir: Path, runner: AgentRunner, agent_result: Any) -> None:
        trace_path = results_dir / "agent_trace.json"
        if trace_path.exists():
            return
        payload = {
            "kind": runner.kind,
            "backend": runner.backend,
            "result": agent_result.to_dict() if hasattr(agent_result, "to_dict") else agent_result,
        }
        write_text(trace_path, json.dumps(payload, indent=2, ensure_ascii=False))

    def _run_verifier(self, package: CasePackage, run: PreparedRun, config: EvaluationConfig, image_tag: Optional[str]) -> Any:
        if self.backend == "local":
            # local verifier 直接在源 case 目录运行，方便快速迭代。
            spec = CommandSpec(
                shell=config.verifier_cmd or "bash verifier/test.sh",
                cwd=package.paths.root,
                env=self._common_env(package, run, stage="verifier", container=False),
                timeout_s=config.timeout_s,
            )
            return self.local.run(spec)
        tag = image_tag or self._build_image(package, prefix=config.image_tag_prefix)
        spec = CommandSpec(
            shell=config.verifier_cmd or "bash /case/verifier/test.sh",
            cwd=Path("/workspace"),
            env=self._common_env(package, run, stage="verifier", container=True),
            timeout_s=config.timeout_s,
        )
        mounts = [
            # verifier 能看到修复后的 workspace，也能看到完整 case 树，
            # 包括 oracle/ 和 verifier-only 资源。
            DockerMount(run.workspace, "/workspace", read_only=True),
            DockerMount(package.paths.root, "/case", read_only=True),
            DockerMount(run.results, "/results", read_only=False),
        ]
        result = self.docker.run(tag, spec, mounts=mounts, workdir="/workspace", network=config.verifier_network)
        return result

    def _common_env(self, package: CasePackage, run: PreparedRun, stage: str, container: bool) -> Dict[str, str]:
        if container:
            # 容器模式使用固定容器内路径，让 verifier 不依赖宿主机目录结构。
            return {
                "PTDBENCH_CASE_ID": package.case_id,
                "PTDBENCH_CASE_ROOT": "/case",
                "PTDBENCH_VISIBLE_CASE": "/case" if stage == "agent" else "",
                "PTDBENCH_WORKSPACE": "/workspace",
                "PTDBENCH_ORACLE": "/case/oracle" if stage == "verifier" else "",
                "PTDBENCH_VERIFIER": "/case/verifier" if stage == "verifier" else "",
                "PTDBENCH_RESULTS": "/results",
                "PTDBENCH_STAGE": stage,
            }
        return {
            "PTDBENCH_CASE_ID": package.case_id,
            "PTDBENCH_CASE_ROOT": str((run.visible_case if stage == "agent" else package.paths.root).resolve()),
            "PTDBENCH_VISIBLE_CASE": str(run.visible_case.resolve()),
            "PTDBENCH_WORKSPACE": str(run.workspace.resolve()),
            "PTDBENCH_ORACLE": str((package.paths.oracle if stage == "verifier" else Path()).resolve()) if stage == "verifier" else "",
            "PTDBENCH_VERIFIER": str((package.paths.verifier if stage == "verifier" else Path()).resolve()) if stage == "verifier" else "",
            "PTDBENCH_RESULTS": str(run.results.resolve()),
            "PTDBENCH_STAGE": stage,
        }

    def _write_report(self, path: Path, result: EvaluationResult) -> None:
        # 下游工具应优先消费这个顶层 report。
        write_text(path, json.dumps(result.to_dict(), indent=2, ensure_ascii=False))

    def _extract_metrics(self, results_dir: Path, verifier_result: Any) -> Dict[str, Any]:
        # 优先读取 verifier 写出的结构化结果；如果没有，就退回到退出码，
        # 这样最小模板也能工作。
        metrics: Dict[str, Any] = {}
        for name in ("result.json", "metrics.json"):
            path = results_dir / name
            if not path.exists():
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception as exc:
                metrics["result_parse_error"] = str(exc)
                continue
            if isinstance(payload, dict):
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
            return metrics
        metrics.setdefault("passed", verifier_result.returncode == 0)
        return metrics

    def _extract_score(self, metrics: Dict[str, Any]) -> Optional[float]:
        # score 解析保持通用：显式 score 优先；只有 pass/fail 时转成二值分；
        # 其他情况保留为未打分。
        if "score" in metrics and isinstance(metrics["score"], (int, float)):
            return float(metrics["score"])
        if metrics.get("passed") is True:
            return 1.0
        if metrics.get("passed") is False:
            return 0.0
        return None
