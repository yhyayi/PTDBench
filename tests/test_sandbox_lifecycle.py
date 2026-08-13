from __future__ import annotations

import tempfile
import unittest
import subprocess
from unittest.mock import patch
from pathlib import Path

from ptdbench.agents.base import AgentExecutionContext, AgentRunner
from ptdbench.agents.installed.codex import CodexAgentRunner
from ptdbench.core.types import AgentSpec, CommandResult
from ptdbench.evaluation.evaluator import CaseEvaluator
from ptdbench.runtimes import (
    CommandSpec,
    EnvironmentArtifact,
    SandboxInstance,
    SandboxRequest,
    SandboxRuntime,
)
from ptdbench.runtimes.providers.docker import DockerRuntime


def result(label: str, returncode: int = 0) -> CommandResult:
    return CommandResult([label], returncode, label, "", False, 0.0, "/workspace")


class RecordingRuntime(SandboxRuntime):
    name = "recording"

    def __init__(self) -> None:
        self.events: list[str] = []

    @property
    def needs_environment_image(self) -> bool:
        return False

    def build_environment(self, environment_dir, *, dockerfile="Dockerfile", tag_prefix="ptdbench"):
        return EnvironmentArtifact(self.name)

    def path(self, request, name):
        return {
            "case_root": "/case",
            "visible_case": "/case",
            "workspace": "/workspace",
            "results": "/results",
            "oracle": "/case/oracle",
            "verifier": "/case/verifier",
        }[name]

    def run(self, request, spec, *, artifact=None):
        raise AssertionError("persistent lifecycle should be used")

    def create_sandbox(self, request, *, artifact):
        self.events.append(f"create:{request.stage}")
        return SandboxInstance(self.name, request.stage, request)

    def exec(self, sandbox, spec):
        self.events.append(f"exec:{sandbox.request.stage}")
        return result("exec")

    def close_sandbox(self, sandbox):
        self.events.append(f"close:{sandbox.request.stage}")


class RecordingAgent(AgentRunner):
    def __init__(self, setup_returncode: int = 0) -> None:
        super().__init__(AgentSpec(kind="recording", backend="recording"))
        self.setup_returncode = setup_returncode

    def setup(self, context, *, runtime, sandbox):
        runtime.events.append("agent:setup")
        return result("setup", self.setup_returncode)

    def run(self, context, *, runtime, artifact, sandbox=None):
        runtime.events.append("agent:run")
        return result("run")


class SandboxLifecycleTests(unittest.TestCase):
    def context(self, root: Path) -> AgentExecutionContext:
        for name in ("workspace", "visible", "results"):
            (root / name).mkdir()
        return AgentExecutionContext(
            case_id="case",
            instruction="do work",
            metadata={},
            timeout_s=10,
            case_root=root,
            workspace=root / "workspace",
            visible_case=root / "visible",
            results=root / "results",
        )

    def test_setup_and_run_share_one_sandbox(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = RecordingRuntime()
            outcome = CaseEvaluator()._run_agent(
                RecordingAgent(),
                self.context(Path(temp_dir)),
                runtime,
                None,
            )
        self.assertEqual(outcome.setup.returncode, 0)
        self.assertIsNotNone(outcome.execution)
        self.assertEqual(
            runtime.events,
            ["create:agent", "agent:setup", "agent:run", "close:agent"],
        )

    def test_setup_failure_skips_run_and_closes_sandbox(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = RecordingRuntime()
            outcome = CaseEvaluator()._run_agent(
                RecordingAgent(setup_returncode=1),
                self.context(Path(temp_dir)),
                runtime,
                None,
            )
        self.assertIsNone(outcome.execution)
        self.assertEqual(
            runtime.events,
            ["create:agent", "agent:setup", "close:agent"],
        )

    def test_sensitive_docker_env_uses_name_only(self):
        argv, process_env = DockerRuntime._docker_env(
            {"OPENAI_API_KEY": "secret", "PTDBENCH_STAGE": "agent"}
        )
        self.assertEqual(
            argv,
            ["-e", "OPENAI_API_KEY", "-e", "PTDBENCH_STAGE=agent"],
        )
        self.assertEqual(process_env["OPENAI_API_KEY"], "secret")
        self.assertNotIn("secret", " ".join(argv))

    def test_failed_stage_summary_points_to_debug_logs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            results = Path(temp_dir)
            CaseEvaluator("local")._append_command_summary(
                results,
                "agent_setup",
                CommandResult(
                    ["false"],
                    2,
                    "setup output\n",
                    "installation failed\n",
                    False,
                    1.25,
                    "/workspace",
                ),
            )
            run_log = (results / "run.log").read_text(encoding="utf-8")

        self.assertIn("[FAIL] agent_setup exit=2", run_log)
        self.assertIn("error_tail: installation failed", run_log)
        self.assertIn(
            "logs: agent_setup.stdout.log, agent_setup.stderr.log",
            run_log,
        )

    def test_docker_proxy_is_opt_in_for_agent_only(self):
        request = SandboxRequest(
            stage="agent",
            case_id="case",
            host_case_root=Path("/case"),
            host_visible_case=Path("/visible"),
            host_workspace=Path("/workspace"),
            host_results=Path("/results"),
        )
        with patch.dict(
            "os.environ",
            {"PTDBENCH_DOCKER_PROXY": "http://host.docker.internal:7890"},
            clear=False,
        ):
            env = DockerRuntime._proxy_env(request, {"PTDBENCH_STAGE": "agent"})
        self.assertEqual(env["HTTP_PROXY"], "http://host.docker.internal:7890")
        self.assertEqual(env["HTTPS_PROXY"], "http://host.docker.internal:7890")
        self.assertEqual(env["ALL_PROXY"], "http://host.docker.internal:7890")

    def test_codex_setup_configures_provider_and_logs_in_via_stdin(self):
        class SetupRuntime(RecordingRuntime):
            name = "docker"

            def exec(self, sandbox, spec):
                self.setup_spec = spec
                return result("setup")

        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = SetupRuntime()
            context = self.context(Path(temp_dir))
            runner = CodexAgentRunner(
                AgentSpec(
                    kind="codex",
                    backend="recording",
                    model="gpt-5.5",
                    reasoning_effort="medium",
                    base_url="https://yeysai.com/v1",
                    api_key="secret-key",
                )
            )
            runner.setup(
                context,
                runtime=runtime,
                sandbox=SandboxInstance("recording", "agent", context.sandbox_request(runtime)),
            )

        shell = runtime.setup_spec.shell
        self.assertIn('model_provider = "ptdbench"', shell)
        self.assertIn('model_reasoning_effort = "medium"', shell)
        self.assertIn('base_url = "https://yeysai.com/v1"', shell)
        self.assertIn("codex login --with-api-key", shell)
        self.assertIn('"$OPENAI_API_KEY" | /usr/local/bin/codex login', shell)
        self.assertNotIn("secret-key", shell)
        self.assertEqual(runtime.setup_spec.env["OPENAI_API_KEY"], "secret-key")
        self.assertEqual(runtime.setup_spec.live_log_name, "agent_setup")
        self.assertFalse(runtime.setup_spec.live_log_to_stderr)
        self.assertIn("[setup] downloading nvm", shell)
        self.assertIn("[setup] installing @openai/codex@latest", shell)
        self.assertIn("[setup] complete", shell)

    def test_docker_stream_process_writes_live_stage_logs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            process = subprocess.Popen(
                ["sh", "-lc", "echo setup-out; echo setup-error >&2"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr, timed_out = DockerRuntime._stream_process(
                process,
                timeout_s=5,
                results_dir=Path(temp_dir),
                log_name="agent_setup",
            )
            self.assertFalse(timed_out)
            self.assertEqual(stdout, "setup-out\n")
            self.assertEqual(stderr, "setup-error\n")
            self.assertEqual(
                (Path(temp_dir) / "agent_setup.stdout.log").read_text(),
                stdout,
            )
            self.assertEqual(
                (Path(temp_dir) / "agent_setup.stderr.log").read_text(),
                stderr,
            )

    def test_codex_setup_debug_enables_terminal_streaming(self):
        class SetupRuntime(RecordingRuntime):
            name = "docker"

            def exec(self, sandbox, spec):
                self.setup_spec = spec
                return result("setup")

        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = SetupRuntime()
            context = self.context(Path(temp_dir))
            context = AgentExecutionContext(
                **{**context.__dict__, "log_level": "debug"}
            )
            runner = CodexAgentRunner(AgentSpec(kind="codex", api_key="secret"))
            runner.setup(
                context,
                runtime=runtime,
                sandbox=SandboxInstance("recording", "agent", context.sandbox_request(runtime)),
            )

        self.assertTrue(runtime.setup_spec.live_log_to_stderr)

    def test_run_directory_prefix_is_readable_and_safe(self):
        prefix = CaseEvaluator._run_directory_prefix(
            "swe_slugfix/001: trial",
            "issue",
        )
        self.assertRegex(
            prefix,
            r"^ptdbench-run-swe_slugfix-001-trial-issue-\d{8}-\d{6}-\d{6}-$",
        )

    def test_codex_uses_outer_docker_sandbox(self):
        runner = CodexAgentRunner(
            AgentSpec(kind="codex", reasoning_effort="high")
        )
        argv = runner._build_argv(
            "prompt",
            last_message_path="/results/last-message",
            externally_sandboxed=True,
        )
        self.assertIn("--dangerously-bypass-approvals-and-sandbox", argv)
        self.assertNotIn("--sandbox", argv)
        self.assertIn('model_reasoning_effort="high"', argv)

    def test_codex_keeps_workspace_sandbox_for_local_runtime(self):
        runner = CodexAgentRunner(AgentSpec(kind="codex"))
        argv = runner._build_argv(
            "prompt",
            last_message_path="/results/last-message",
            externally_sandboxed=False,
        )
        self.assertIn("--sandbox", argv)
        self.assertIn("workspace-write", argv)
        self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", argv)


if __name__ == "__main__":
    unittest.main()
