from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Dict, Optional

from ...core.types import CommandResult
from ...runtimes import CommandSpec, EnvironmentArtifact, SandboxInstance, SandboxRuntime
from ..base import AgentExecutionContext, AgentRunner
from ..factory import register_agent_runner


@register_agent_runner("codex")
class CodexAgentRunner(AgentRunner):
    # 第一版 Codex 适配器：把任务说明拼成 prompt，然后按 CLI 调用。
    provider_name = "ptdbench"

    def setup(
        self,
        context: AgentExecutionContext,
        *,
        runtime: SandboxRuntime,
        sandbox: SandboxInstance,
    ) -> CommandResult:
        if runtime.name != "docker":
            return CommandResult([], 0, "", "", False, 0.0, "")
        config_lines = [
            f"model = {self._toml_string(self.spec.model)}" if self.spec.model else None,
            (
                f"model_reasoning_effort = {self._toml_string(self.spec.reasoning_effort)}"
                if self.spec.reasoning_effort
                else None
            ),
            'disable_response_storage = true',
            'preferred_auth_method = "apikey"',
        ]
        if self.spec.base_url:
            config_lines.extend(
                [
                    f'model_provider = "{self.provider_name}"',
                    "",
                    f"[model_providers.{self.provider_name}]",
                    f'name = "{self.provider_name}"',
                    f"base_url = {self._toml_string(self.spec.base_url)}",
                    'wire_api = "responses"',
                    "requires_openai_auth = true",
                ]
            )
        config_lines.extend(
            [
                "",
                '[projects."/workspace"]',
                'trust_level = "trusted"',
            ]
        )
        config_lines = [line for line in config_lines if line is not None]
        config_command = "printf '%s\\n' " + " ".join(
            shlex.quote(line) for line in config_lines
        ) + " > /root/.codex/config.toml;"
        auth_command = (
            "test -n \"${OPENAI_API_KEY:-}\"; "
            "printf '%s' \"$OPENAI_API_KEY\" | /usr/local/bin/codex login --with-api-key; "
            "/usr/local/bin/codex login status;"
        )
        debian_mirror_command = (
            'if [ -n "${PTDBENCH_DEBIAN_MIRROR:-}" ]; then '
            'mirror="${PTDBENCH_DEBIAN_MIRROR%/}"; '
            "for sources in /etc/apt/sources.list /etc/apt/sources.list.d/debian.sources; do "
            '[ -f "$sources" ] || continue; '
            'sed -i '
            '-e "s|http://deb.debian.org/debian-security|$mirror/debian-security|g" '
            '-e "s|https://deb.debian.org/debian-security|$mirror/debian-security|g" '
            '-e "s|http://security.debian.org/debian-security|$mirror/debian-security|g" '
            '-e "s|https://security.debian.org/debian-security|$mirror/debian-security|g" '
            '-e "s|http://deb.debian.org/debian|$mirror/debian|g" '
            '-e "s|https://deb.debian.org/debian|$mirror/debian|g" '
            '"$sources"; '
            "done; "
            "fi; "
        )
        bash_install = " ".join(
            [
                "set -euo pipefail;",
                "echo '[setup] downloading nvm' >&2;",
                "export NVM_DIR=/root/.nvm;",
                'nvm_install_url="${PTDBENCH_NVM_INSTALL_URL:-https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh}";',
                "curl --fail --location --show-error --silent --retry 5 --retry-all-errors --connect-timeout 15 --max-time 120 ",
                '"$nvm_install_url" -o /tmp/install-nvm.sh;',
                "test -s /tmp/install-nvm.sh;",
                "echo '[setup] installing nvm' >&2;",
                "bash /tmp/install-nvm.sh;",
                "test -s \"$NVM_DIR/nvm.sh\";",
                'source "$NVM_DIR/nvm.sh";',
                "echo '[setup] installing node 22' >&2;",
                "nvm install 22;",
                "nvm alias default 22;",
                "echo '[setup] installing @openai/codex@latest' >&2;",
                "npm install -g @openai/codex@latest;",
                'for bin in node npm codex; do path=$(command -v "$bin");',
                "target=/usr/local/bin/$bin;",
                '[ "$path" = "$target" ] || ln -sf "$path" "$target";',
                "done;",
                "mkdir -p /root/.codex;",
                "echo '[setup] writing codex config' >&2;",
                config_command,
                "echo '[setup] configuring API-key auth' >&2;",
                auth_command,
                "echo '[setup] complete' >&2;",
                "/usr/local/bin/codex --version",
            ]
        )
        install = CommandSpec(
            shell=(
                "set -e; "
                f"{debian_mirror_command}"
                "if command -v apk >/dev/null 2>&1; then "
                "apk add --no-cache ca-certificates curl bash nodejs npm ripgrep; "
                "else "
                "if command -v apt-get >/dev/null 2>&1; then "
                "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y "
                "ca-certificates curl bash ripgrep; "
                "elif command -v dnf >/dev/null 2>&1; then "
                "dnf install -y ca-certificates curl bash ripgrep; "
                "elif command -v yum >/dev/null 2>&1; then "
                "yum install -y ca-certificates curl bash ripgrep; "
                "fi; "
                "fi; "
                f"exec bash -lc {shlex.quote(bash_install)}"
            ),
            env=self._build_env(
                {"PTDBENCH_DEBIAN_MIRROR": context.debian_mirror}
                if context.debian_mirror
                else None
            ),
            timeout_s=context.timeout_s,
            user="0",
            live_log_name="agent_setup",
            live_log_to_stderr=context.log_level == "debug",
        )

        return runtime.exec(sandbox, install)

    def run(
        self,
        context: AgentExecutionContext,
        *,
        runtime: SandboxRuntime,
        artifact: Optional[EnvironmentArtifact],
        sandbox: Optional[SandboxInstance] = None,
    ) -> CommandResult:
        prompt = self._build_prompt(context)
        env = self._build_env()
        request = context.sandbox_request(
            runtime,
            env=env,
            network=self.spec.network,
        )
        runtime_last_message_path = self._runtime_last_message_path(runtime, request)
        last_message_host_path = context.results / ".codex_last_message.tmp"
        argv = self._build_argv(
            prompt,
            last_message_path=runtime_last_message_path,
            externally_sandboxed=runtime.name == "docker",
        )
        command = CommandSpec(argv=argv, env=env)
        result = (
            runtime.exec(sandbox, command)
            if sandbox is not None
            else runtime.run(request, command, artifact=artifact)
        )
        self._write_trajectory(
            context.results,
            prompt,
            env,
            result,
            last_message_host_path,
            backend=runtime.name,
        )
        return result

    @staticmethod
    def _toml_string(value: str) -> str:
        """Encode a value as a TOML basic string without shell interpolation."""
        return json.dumps(value, ensure_ascii=False)

    def _build_env(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        env: Dict[str, str] = dict(self.spec.env)
        if extra:
            env.update(extra)
        if self.spec.base_url:
            env["OPENAI_BASE_URL"] = self.spec.base_url
        if self.spec.api_key:
            env["OPENAI_API_KEY"] = self.spec.api_key
        return env

    def _build_argv(
        self,
        prompt: str,
        *,
        last_message_path: str,
        externally_sandboxed: bool,
    ) -> list[str]:
        argv = [self.spec.command or "codex", "exec"]
        argv.append("--skip-git-repo-check")
        if externally_sandboxed:
            # Docker is the security boundary. Codex's Linux bubblewrap sandbox
            # cannot create a nested user namespace in a default container.
            argv.append("--dangerously-bypass-approvals-and-sandbox")
        else:
            argv.extend(["--sandbox", "workspace-write"])
        if self.spec.model:
            argv.extend(["--model", self.spec.model])
        if self.spec.reasoning_effort:
            argv.extend(
                [
                    "-c",
                    f"model_reasoning_effort={self._toml_string(self.spec.reasoning_effort)}",
                ]
            )
        argv.append("--json")
        argv.extend(["--output-last-message", last_message_path])
        argv.extend(list(self.spec.extra_args))
        argv.append(prompt)
        return argv

    def _build_prompt(self, context: AgentExecutionContext) -> str:
        return "\n".join(
            [
                "You are solving a PTDBench task.",
                (
                    "Only modify files under the workspace, except for deliverables "
                    "that the task explicitly requires under $PTDBENCH_RESULTS."
                ),
                "Do not touch oracle/ or verifier/.",
                "Task instruction:",
                context.instruction.strip(),
                "",
                "Workspace path: use $PTDBENCH_WORKSPACE.",
                "Results path: use $PTDBENCH_RESULTS only for required deliverables.",
                "Public task text path: use $PTDBENCH_VISIBLE_TASK/instruction.md.",
            ]
        )

    def _runtime_last_message_path(self, runtime: SandboxRuntime, request) -> str:
        return runtime.path(request, "results") + "/.codex_last_message.tmp"

    def _write_trajectory(
        self,
        results_dir: Path,
        prompt: str,
        env: Dict[str, str],
        result: CommandResult,
        last_message_path: Path,
        *,
        backend: str,
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
                "backend": backend,
                "prompt": prompt,
                "env": self._redact_env(env),
                "last_message": last_message,
                "result": result.to_dict(),
            },
        )
