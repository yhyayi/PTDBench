from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from ...executor import CommandSpec, DockerExecutor, DockerMount, LocalExecutor
from ...types import CommandResult
from ..base import AgentExecutionContext, AgentRunner
from ..factory import register_agent_runner


@register_agent_runner("codex")
class CodexAgentRunner(AgentRunner):
    # 第一版 Codex 适配器：把任务说明拼成 prompt，然后按 CLI 调用。

    def run(
        self,
        context: AgentExecutionContext,
        *,
        local: LocalExecutor,
        docker: DockerExecutor,
        image_tag: Optional[str],
    ) -> CommandResult:
        prompt = self._build_prompt(context)
        env = self._build_env(context)
        runtime_last_message_path = self._runtime_last_message_path(context)
        last_message_host_path = self._last_message_host_path(context)
        argv = self._build_argv(prompt, last_message_path=runtime_last_message_path)
        if self.backend == "local":
            spec = CommandSpec(
                argv=argv,
                cwd=context.local_workspace,
                env=env,
                timeout_s=context.timeout_s,
            )
            result = local.run(spec)
            self._write_trajectory(context.local_results, prompt, env, result, last_message_host_path)
            return result

        if image_tag is None:
            raise ValueError("docker codex agent needs an environment image")

        spec = CommandSpec(
            argv=argv,
            cwd=Path(context.container_workspace),
            env=env,
            timeout_s=context.timeout_s,
        )
        mounts = [
            DockerMount(context.local_workspace, context.container_workspace, read_only=False),
            DockerMount(context.local_visible_case, context.container_visible_case, read_only=True),
            DockerMount(context.local_results, context.container_results, read_only=False),
        ]
        result = docker.run(
            image_tag,
            spec,
            mounts=mounts,
            workdir=context.container_workspace,
            network=self.spec.network,
        )
        self._write_trajectory(context.local_results, prompt, env, result, last_message_host_path)
        return result

    def _build_env(self, context: AgentExecutionContext) -> Dict[str, str]:
        env = dict(context.container_env if self.backend == "docker" else context.local_env)
        if self.spec.base_url:
            env["OPENAI_BASE_URL"] = self.spec.base_url
        if self.spec.api_key:
            env["OPENAI_API_KEY"] = self.spec.api_key
        return env

    def _build_argv(self, prompt: str, *, last_message_path: str) -> list[str]:
        argv = [self.spec.command or "codex", "exec"]
        argv.append("--skip-git-repo-check")
        argv.extend(["--sandbox", "workspace-write"])
        if self.spec.model:
            argv.extend(["--model", self.spec.model])
        argv.append("--json")
        argv.extend(["--output-last-message", last_message_path])
        argv.extend(list(self.spec.extra_args))
        argv.append(prompt)
        return argv

    def _build_prompt(self, context: AgentExecutionContext) -> str:
        return "\n".join(
            [
                "You are solving a PTDBench task.",
                "Only modify files under the workspace.",
                "Do not touch oracle/ or verifier/.",
                "Task instruction:",
                context.instruction.strip(),
                "",
                "Workspace path: use $PTDBENCH_WORKSPACE.",
                "Public task text path: use $PTDBENCH_VISIBLE_CASE/instruction.md.",
            ]
        )

    def _runtime_last_message_path(self, context: AgentExecutionContext) -> str:
        base = Path(context.container_results) if self.backend == "docker" else context.local_results.resolve()
        return str(base / ".codex_last_message.tmp")

    def _last_message_host_path(self, context: AgentExecutionContext) -> Path:
        return context.local_results / ".codex_last_message.tmp"

    def _write_trajectory(
        self,
        results_dir: Path,
        prompt: str,
        env: Dict[str, str],
        result: CommandResult,
        last_message_path: Path,
    ) -> None:
        # 轨迹尽量压成一份，保留重放需要的关键字段就够了。
        last_message = ""
        if last_message_path.exists():
            last_message = last_message_path.read_text(encoding="utf-8")
            try:
                last_message_path.unlink()
            except OSError:
                pass
        self._write_agent_trace(
            results_dir,
            {
                "kind": self.kind,
                "backend": self.backend,
                "prompt": prompt,
                "env": self._redact_env(env),
                "last_message": last_message,
                "result": result.to_dict(),
            },
        )
