from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from ptdbench.cases import SchemaError, load_case_package
from ptdbench.core.types import AgentSpec, EvaluationConfig
from ptdbench.evaluation.evaluator import CaseEvaluator
from ptdbench.runtimes import SandboxRequest
from ptdbench.runtimes.providers.docker import DockerRuntime


ROOT = Path(__file__).resolve().parents[1]
SWE_CASE = ROOT / "example" / "swe_slugfix"


def create_legacy_case(root: Path) -> Path:
    case = root / "legacy"
    for directory in ("workspace", "oracle", "environment", "verifier"):
        (case / directory).mkdir(parents=True, exist_ok=True)
    (case / "instruction.md").write_text("Fix the workspace.\n", encoding="utf-8")
    (case / "environment" / "Dockerfile").write_text(
        "FROM python:3.11-slim\n",
        encoding="utf-8",
    )
    (case / "verifier" / "test.sh").write_text(
        "#!/usr/bin/env bash\nexit 0\n",
        encoding="utf-8",
    )
    (case / "metadata.yaml").write_text(
        """case_id: legacy_case
title: Legacy case
submitter:
  name: test
license: MIT
created_date: 2026-08-14
domain: code
model:
  name: test
dataset:
  name: test
training:
  framework: none
  algo: none
  run_command: n/a
environment:
  dockerfile: environment/Dockerfile
outcome:
  failure_type: test
  failure_reason: test
""",
        encoding="utf-8",
    )
    return case


class EvaluationSchemaTests(unittest.TestCase):
    def test_v2_case_loads_both_evaluations(self):
        package = load_case_package(SWE_CASE)

        self.assertTrue(package.report.ok, package.report.to_dict())
        self.assertEqual(package.evaluation_levels, ("issue", "solution"))
        self.assertTrue(package.paths.evaluations["issue"].oracle.is_file())
        self.assertTrue(package.paths.evaluations["solution"].oracle.is_dir())
        self.assertFalse(package.paths.evaluations["issue"].verifier_resources.exists())

    def test_legacy_case_is_mapped_to_solution(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            legacy_case = create_legacy_case(Path(temp_dir))
            package = load_case_package(legacy_case)

            self.assertTrue(package.report.ok, package.report.to_dict())
            self.assertEqual(package.evaluation_levels, ("solution",))
            self.assertEqual(
                package.paths.evaluations["solution"].instruction_md,
                (legacy_case / "instruction.md").resolve(),
            )

    def test_unavailable_evaluation_fails_before_running(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            legacy_case = create_legacy_case(Path(temp_dir))
            with self.assertRaisesRegex(
                SchemaError,
                "available levels: solution",
            ):
                CaseEvaluator("local").evaluate(
                    legacy_case,
                    EvaluationConfig(backend="local", evaluation_level="issue"),
                )

    def test_issue_evaluation_reads_instruction_defined_answer(self):
        command = (
            "printf '%s\\n' "
            "'The implementation replaces only spaces. Punctuation and underscores "
            "are not handled, and repeated separators are not collapsed.' "
            '> "$PTDBENCH_RESULTS/agent_answer.md"'
        )
        result = CaseEvaluator("local").evaluate(
            SWE_CASE,
            EvaluationConfig(
                backend="local",
                evaluation_level="issue",
                agent=AgentSpec(
                    kind="shell",
                    command=command,
                    backend="local",
                ),
            ),
        )

        self.assertTrue(result.passed, result.to_dict())
        self.assertEqual(result.evaluation_level, "issue")
        self.assertTrue(result.metrics["answer_present"])

    def test_solution_evaluation_uses_hidden_tests(self):
        result = CaseEvaluator("local").evaluate(
            SWE_CASE,
            EvaluationConfig(backend="local", evaluation_level="solution"),
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.evaluation_level, "solution")
        self.assertEqual(result.metrics["tests_run"], 5)

    def test_solution_agent_can_fix_workspace_without_oracle_access(self):
        command = (
            "test -z \"$PTDBENCH_ORACLE\" && "
            "printf \"%s\\n\" "
            "\"from __future__ import annotations\" \"\" \"import re\" \"\" \"\" "
            "\"def slugify(text: str) -> str:\" "
            "\"    return re.sub(r\\\"[^a-z0-9]+\\\", \\\"-\\\", "
            "text.strip().lower()).strip(\\\"-\\\")\" "
            '> "$PTDBENCH_WORKSPACE/toy_swe/slugify.py"'
        )
        result = CaseEvaluator("local").evaluate(
            SWE_CASE,
            EvaluationConfig(
                backend="local",
                evaluation_level="solution",
                agent=AgentSpec(
                    kind="shell",
                    command=command,
                    backend="local",
                ),
            ),
        )

        self.assertTrue(result.passed, result.to_dict())
        self.assertEqual(result.metrics["tests_run"], 5)
        self.assertNotEqual(result.workspace_before_hash, result.workspace_after_hash)

    def test_docker_verifier_mounts_selected_hidden_paths(self):
        evaluation = load_case_package(SWE_CASE).paths.evaluations["issue"]
        request = SandboxRequest(
            stage="verifier",
            case_id="case",
            host_case_root=evaluation.root,
            host_visible_case=Path("/visible"),
            host_workspace=Path("/workspace"),
            host_results=Path("/results"),
            evaluation_level="issue",
            host_oracle=evaluation.oracle,
            host_verifier=evaluation.verifier,
        )

        mounts = DockerRuntime()._mounts(request)
        by_container_path = {mount.container_path: mount for mount in mounts}
        self.assertEqual(by_container_path["/case"].host_path, evaluation.root)
        self.assertTrue(by_container_path["/case"].read_only)
        runtime = DockerRuntime()
        self.assertEqual(runtime.path(request, "oracle"), "/case/oracle.md")
        self.assertEqual(runtime.path(request, "verifier"), "/case/verifier")


if __name__ == "__main__":
    unittest.main()
