from __future__ import annotations

import json
import re
import shlex
import shutil
import sys
import tempfile
from datetime import datetime
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional

from ..agents import AgentExecutionContext, AgentRunner, create_agent_runner, resolve_agent_spec
from ..core.types import (
    DOCKER_DIR,
    PTDBENCH_ENVIRONMENT_DIR,
    TOOLING_INSTALLER,
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
from .run_spec import REPORT_FILENAME, load_run_spec
from ..utils.filesystem import copy_tree, ensure_dir, tree_digest, write_text


@dataclass(frozen=True)
class PreparedRun:
    """一次评测的隔离 staging 目录。"""

    root: Path
    visible_task: Path
    workspace: Path
    results: Path
    # prepare 阶段的产物落点。prepare 可写，agent / verifier 只读挂同一个目录。
    prepared: Path
    # 跨 run、跨任务共享的下载缓存。prepare 可写，agent / verifier 只读。
    models: Path
    datasets: Path


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
        """运行一次 agent + (rerun) + verifier 评测。

        solution 层的顺序是 agent → rerun → verifier：agent 修代码、提交
        report.md，但**不跑训练**；rerun 由 harness 用 agent 留下的 workspace
        按任务的 run_spec.json 重跑实验；verifier 照旧只认
        $PTDBENCH_RESULTS 下这一遍的新日志。issue 层没有 rerun。
        """
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
                image = package.environment.environment_image
                if image is None:
                    raise SchemaError(
                        f"{package.paths.metadata_yaml}: metadata.environment.image "
                        f"is required by the {self.runtime.name} backend"
                    )
                # 镜像按环境名共享：已存在就直接用，不存在才照
                # docker/images/<image>.Dockerfile 构建。
                artifact = self.runtime.ensure_environment(image)
                self._append_run_log(
                    run.results,
                    f"[OK] environment_prepare backend={self.runtime.name} "
                    f"artifact={artifact.reference or '-'}",
                )

            workspace_before_hash = tree_digest(package.paths.workspace)
            prepare_result = self._run_prepare(
                package, evaluation, run, config, artifact
            )
            if prepare_result is not None and prepare_result.returncode != 0:
                # prepare 失败了还往下跑，agent 会在一个没有模型的 workspace 里作业，
                # 最后的失败会被记成 agent 的失败——那是一条查半天的假线索。
                self._append_run_log(
                    run.results, "[SKIP] agent reason=prepare_failed"
                )
                raise RuntimeError(
                    f"task {package.task_id!r}: prepare failed with exit "
                    f"{prepare_result.returncode}; see prepare.stderr.log"
                )

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

            # solution 层的提交门槛：agent 修完必须交 /workspace/report.md。
            # 没交等于没完成协议，rerun 和 verifier 都不跑——尤其对
            # verifier 自己驱动训练的任务（critic_value_head），不拦这一道
            # 就会出现"什么都没交也有分"。
            report_file = run.workspace / REPORT_FILENAME
            has_report = report_file.is_file() and report_file.stat().st_size > 0
            rerun_result = None
            verifier_result = None
            if evaluation.level == "solution" and not has_report:
                self._append_run_log(
                    run.results, f"[SKIP] rerun reason=no_{REPORT_FILENAME}"
                )
                self._append_run_log(
                    run.results, f"[SKIP] verifier reason=no_{REPORT_FILENAME}"
                )
            else:
                if has_report:
                    # report 归档进 results：workspace 跑完会被清理，判分侧
                    # 复盘"agent 到底改了什么"要有据可查。
                    shutil.copy2(report_file, run.results / REPORT_FILENAME)
                rerun_result = self._run_rerun(
                    package, evaluation, run, config, artifact=artifact
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
                and (rerun_result is None or rerun_result.returncode == 0)
            )
            if not has_report and evaluation.level == "solution":
                metrics["passed"] = False
                metrics["failure"] = (
                    f"agent did not submit {REPORT_FILENAME} in the workspace"
                )
            result = EvaluationResult(
                task_id=package.task_id,
                evaluation_level=evaluation.level,
                passed=stages_ok and bool(metrics.get("passed", False)),
                score=self._extract_score(metrics) if stages_ok else 0.0,
                metrics=metrics,
                report_dir=run.results,
                agent_spec=agent_spec,
                agent_setup=agent_setup_result,
                agent=agent_result,
                rerun=rerun_result,
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
        prepared = ensure_dir(root / "prepared")
        # 缓存留在 run 目录之外才有意义：run 目录跑完就删，模型下一次还得重下。
        # EvaluationConfig 保证这两个值一定有（默认是仓库根的 models/ 和 datasets/），
        # 所以这里没有"这次没有缓存"的分支。
        models = ensure_dir(config.models_dir)
        datasets = ensure_dir(config.datasets_dir)
        # Agent 改的是这份副本，不是仓库里的任务目录。
        copy_tree(package.paths.workspace, workspace)
        write_text(
            visible_task / "instruction.md",
            evaluation.instruction_md.read_text(encoding="utf-8"),
        )
        return PreparedRun(
            root, visible_task, workspace, results, prepared, models, datasets
        )

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
            debian_mirror=config.debian_mirror,
            # agent 阶段不挂 environment/：见 _environment_mount 的注释。
            environment_dir=None,
            prepared_dir=run.prepared,
            models_dir=run.models,
            datasets_dir=run.datasets,
            task_env=dict(package.environment.env),
        )

    @staticmethod
    def _environment_mount(package: TaskPackage) -> Optional[Path]:
        """任务的 environment/ 目录，没有就不挂。

        只给 prepare 和 verifier 用。**agent 阶段不挂**：这个目录里的
        prepare.sh 写着任务是怎么搭起来的——下哪个 HF 仓库、解哪个归档、
        校验哪些 sha256——而其中任何一个名字都可能直接点出要诊断的缺陷
        （数据集仓库名有时就是照着它承载的那个缺陷起的）。

        agent 阶段没有任何东西读它：activate hook 在镜像的 /etc/profile.d 下，
        不在这里；仓库现有任务的 agent 阶段也没有引用 $PTDBENCH_ENVIRONMENT。
        以前挂着只是因为三个阶段共用一张挂载表。

        在此之前，"别把答案写进 prepare.sh" 是一条写在 prepare.sh 文件头里的
        约定，靠每个任务作者自觉；约定覆盖不到"仓库名本身就是答案"这种情况。
        """
        return package.paths.environment if package.paths.environment.is_dir() else None

    @staticmethod
    def _prepare_script(package: TaskPackage) -> Optional[str]:
        """prepare.sh 在容器里的路径，任务没有这个脚本就返回 None。

        脚本必须待在任务的 `environment/` 目录里，因为那是唯一被挂进容器的那一份。
        声明一个在外面的路径会在容器里变成"文件不存在"，报错点离原因很远，所以在
        这里就拦掉。
        """
        prepare = package.environment.prepare
        if prepare is None:
            return None
        if not prepare.is_file():
            raise SchemaError(
                f"task {package.task_id!r}: metadata.environment.prepare points to "
                f"{prepare}, which does not exist"
            )
        try:
            relative = prepare.resolve().relative_to(package.paths.environment.resolve())
        except ValueError as exc:
            raise SchemaError(
                f"task {package.task_id!r}: metadata.environment.prepare must stay under "
                f"{package.paths.environment}; only that directory is mounted into the "
                f"prepare container"
            ) from exc
        return f"{PTDBENCH_ENVIRONMENT_DIR}/{relative.as_posix()}"

    def _run_prepare(
        self,
        package: TaskPackage,
        evaluation: EvaluationPaths,
        run: PreparedRun,
        config: EvaluationConfig,
        artifact: Optional[EnvironmentArtifact],
    ) -> Optional[CommandResult]:
        """跑任务的 prepare.sh：下模型、下数据集、写 prepared/env。

        这是三个阶段里**唯一**联网的一个。放在 agent 之前而不是烘进镜像，是因为
        下载的东西是逐任务的，烘进去就等于每个任务一个几十 GB 的镜像层——那正是
        这次改造要去掉的东西。
        """
        script = self._prepare_script(package)
        if script is None:
            return None
        request = SandboxRequest(
            stage="prepare",
            task_id=package.task_id,
            host_task_root=run.visible_task,
            host_visible_task=run.visible_task,
            host_workspace=run.workspace,
            host_results=run.results,
            evaluation_level=evaluation.level,
            host_environment=self._environment_mount(package),
            host_prepared=run.prepared,
            host_models=run.models,
            host_datasets=run.datasets,
            # 仓库 docker/ 只读挂进 prepare 容器；跑脚本前用它装最新 fetch/dataset 工具。
            # 目录不在（装成 wheel 时 docker/ 可能不跟着走）就不挂，下面的调用也兜底跳过。
            host_tooling=DOCKER_DIR if DOCKER_DIR.is_dir() else None,
            env=self._prepare_mirror_env(config),
            # prepare 是唯一该联网的阶段；把它也断网，模型就没人下了。
            network="bridge",
            timeout_s=config.timeout_s,
            workdir="workspace",
            task_env=dict(package.environment.env),
        )
        if artifact is None:
            if self.runtime.needs_environment_image:
                raise ValueError("prepare runtime needs a built environment artifact")
            artifact = EnvironmentArtifact(backend=self.runtime.name)
        self._append_run_log(
            run.results, f"[START] prepare script={script}"
        )
        sandbox = self.runtime.create_sandbox(request, artifact=artifact)
        try:
            # prepare.sh 调的 ptdbench_fetch_* / ptdbench_dataset 是 shim，exec 的是
            # /usr/local/libexec 下的工具。装成镜像烤进去的那份可能是旧的，所以挂了
            # 仓库 docker/（host_tooling）就先跑安装脚本，把当前版本装进去再跑脚本。
            # 安装脚本不在（没挂）时跳过，行为与旧版一致。
            shell = f"bash {shlex.quote(script)}"
            if request.host_tooling is not None:
                shell = (
                    f"if [ -f {shlex.quote(TOOLING_INSTALLER)} ]; then "
                    f"bash {shlex.quote(TOOLING_INSTALLER)}; fi; " + shell
                )
            result = self.runtime.exec(
                sandbox,
                CommandSpec(
                    shell=shell,
                    timeout_s=config.timeout_s,
                    live_log_name="prepare",
                    live_log_to_stderr=config.log_level == "debug",
                ),
            )
        finally:
            self.runtime.close_sandbox(sandbox)
        self._write_stage_logs(run.results, "prepare", result)
        self._append_command_summary(run.results, "prepare", result)
        return result

    @staticmethod
    def _prepare_mirror_env(config: EvaluationConfig) -> Dict[str, str]:
        """把镜像源交给 prepare 阶段。

        以前这些是构建参数，因为下载发生在 Dockerfile 里；下载搬到 prepare 之后，
        它们就得跟着搬到这个阶段的环境变量里，否则在墙内的机器上一个模型都下不来。
        """
        env: Dict[str, str] = {}
        if config.huggingface_mirror:
            env["HF_ENDPOINT"] = config.huggingface_mirror
        return env

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

    def _run_rerun(
        self,
        package: TaskPackage,
        evaluation: EvaluationPaths,
        run: PreparedRun,
        config: EvaluationConfig,
        artifact: Optional[EnvironmentArtifact],
    ) -> Optional[CommandResult]:
        """按任务的 run_spec.json 用 agent 留下的 workspace 重跑实验。

        没有 run_spec.json 就返回 None（纯协议修复的
        aime25_single_sample_variance：没有要重跑的实验）。这些任务
        提交 report 后直接进 verifier，agent 侧的约束由 instruction 承担。
        run_spec.json 存在但不合法是任务定义的错，load_run_spec 会当场抛
        RunSpecError——不是在训练跑了一半的时候才冒出来。
        """
        if evaluation.level != "solution":
            return None
        spec = load_run_spec(evaluation.root, package.task_id)
        if spec is None:
            return None
        # 判分只认这一遍的新日志：先把 spec 声明的 log roots 清掉。agent 阶段
        # 就算违反协议自己跑过训练，留下的旧日志也不算数。
        for log_root in spec.log_roots:
            shutil.rmtree(run.results / log_root, ignore_errors=True)
        # spec.env 里的 $PTDBENCH_RESULTS 等引用要换成当前后端的真实路径
        # （docker 是 /results，local 是宿主机目录）。standard_env 只读
        # request 的路径字段、不读 env，所以先建请求再展开不会绕圈。
        request = SandboxRequest(
            stage="rerun",
            task_id=package.task_id,
            host_task_root=run.visible_task,
            host_visible_task=run.visible_task,
            host_workspace=run.workspace,
            host_results=run.results,
            evaluation_level=evaluation.level,
            host_environment=self._environment_mount(package),
            # rerun 也要读 prepare 交付的权重：只读挂，agent 改不了它，
            # 实验跑在 prepare 交付的那份资产上这条才有保证。
            host_prepared=run.prepared,
            env={},
            # 实验里 ray 集群、本地检索服务都要 loopback；离线由
            # _network_policy_env 的 HF_HUB_OFFLINE 等变量保证，不走断网。
            network="bridge",
            timeout_s=spec.timeout_s if spec.timeout_s is not None else config.timeout_s,
            workdir="workspace",
            task_env=dict(package.environment.env),
        )
        resolved = {
            key: value
            for key, value in self.runtime.standard_env(request).items()
            if key.startswith("PTDBENCH_") and value
        }
        request = replace(request, env=spec.expand_env(resolved))
        if artifact is None:
            if self.runtime.needs_environment_image:
                raise ValueError("rerun runtime needs a built environment artifact")
            artifact = EnvironmentArtifact(backend=self.runtime.name)
        timeout_s = spec.timeout_s if spec.timeout_s is not None else config.timeout_s
        self._append_run_log(run.results, "[START] rerun")
        sandbox = self.runtime.create_sandbox(request, artifact=artifact)
        try:
            result = self.runtime.exec(
                sandbox,
                CommandSpec(
                    shell=spec.command,
                    timeout_s=timeout_s,
                    # 训练一跑几个小时：像 prepare 一样流式落盘，出事时
                    # rerun.stdout.log 里能看到它死在哪一步。
                    live_log_name="rerun",
                    live_log_to_stderr=config.log_level == "debug",
                ),
            )
        finally:
            self.runtime.close_sandbox(sandbox)
        self._write_stage_logs(run.results, "rerun", result)
        self._append_command_summary(run.results, "rerun", result)
        return result

    def _run_verifier(
        self,
        package: TaskPackage,
        evaluation: EvaluationPaths,
        run: PreparedRun,
        config: EvaluationConfig,
        artifact: Optional[EnvironmentArtifact],
        agent_spec: Optional[AgentSpec],
    ) -> Any:
        verifier_env = dict(config.verifier_env)
        verifier_env.update(self._verifier_judge_env(agent_spec))
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
            env=verifier_env,
            network=config.verifier_network,
            timeout_s=config.timeout_s,
            workdir="task_root",
            host_environment=self._environment_mount(package),
            # verifier 也要看得到 prepare 交付的东西：判分脚本经常要读 checkpoint
            # 或对齐数据集里的样本。只读挂，所以读到的一定是 prepare 写的那一份。
            host_prepared=run.prepared,
            # 下载缓存只读挂：判分脚本要对着模型或数据集的原始副本核对时用得上，
            # 而只读保证它核对的不是一份 agent 能改的东西。
            host_models=run.models,
            host_datasets=run.datasets,
            # 任务声明的 env 两个阶段都给：verifier 判分时往往要按同样的卡数
            # 加载 checkpoint，少了 CUDA_VISIBLE_DEVICES 会跟 agent 看到的不是一台机器。
            task_env=dict(package.environment.env),
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
