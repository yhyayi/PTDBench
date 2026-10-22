from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

from ptdbench.cli import build_parser, expand_task_roots, main as cli_main
from ptdbench.core.harbor_types import (
    HarborEvaluationConfig,
    HarborEvaluationReport,
    HarborRun,
    HarborTaskUnit,
    PreparedHarborTask,
)
from ptdbench.core.types import (
    DATASETS_DIR,
    DOCKER_DIR,
    MODELS_DIR,
    PREPARED_DIR,
    PTDBENCH_ENVIRONMENT_DIR,
    CommandResult,
    EnvironmentImage,
)
from ptdbench.evaluation.harbor_evaluator import (
    ACTIVATE_HOOK,
    HarborAdapterError,
    HarborEvaluator,
    HarborPreparer,
    HarborResultMapper,
    HarborRunner,
    HarborTaskMaterializer,
    _is_remote_backend,
)
from ptdbench.tasks import load_task_package


METADATA = """title: Harbor fixture
submitter:
  name: test
license: MIT
created_date: 2026-09-13
domain: rl
model:
  name: Qwen2.5-1.5B
dataset:
  name: taco
training:
  framework: verl
  algo: grpo
  script: train.sh
environment:
  image: verl
  env:
    CUDA_VISIBLE_DEVICES: "0"
outcome:
  failure_type: test
  failure_reason: test
"""


def create_task(root: Path, name: str = "harbor_fixture") -> Path:
    """一个带 issue / solution 两套评测、prepare.sh 和 verl 镜像的最小任务。"""
    task = root / name
    (task / "workspace" / "pkg").mkdir(parents=True)
    (task / "workspace" / "train.sh").write_text(
        "#!/usr/bin/env bash\nexit 0\n", encoding="utf-8"
    )
    (task / "workspace" / "pkg" / "code.py").write_text("x = 1\n", encoding="utf-8")
    (task / "environment").mkdir()
    (task / "environment" / "prepare.sh").write_text(
        "#!/usr/bin/env bash\necho KEY=1 >> \"$PTDBENCH_PREPARED/env\"\n",
        encoding="utf-8",
    )
    for level in ("issue", "solution"):
        evaluation = task / "evaluations" / level
        (evaluation / "verifier").mkdir(parents=True)
        (evaluation / "instruction.md").write_text(f"{level} instruction\n", encoding="utf-8")
        (evaluation / "verifier" / "test.sh").write_text(
            "#!/usr/bin/env bash\nexit 0\n", encoding="utf-8"
        )
        if level == "issue":
            (evaluation / "oracle.md").write_text("answer\n", encoding="utf-8")
        else:
            (evaluation / "oracle").mkdir()
            (evaluation / "oracle" / "fix.diff").write_text("diff\n", encoding="utf-8")
    (task / "metadata.yaml").write_text(METADATA, encoding="utf-8")
    return task


class FixtureTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.task_root = create_task(self.root)
        self.package = load_task_package(self.task_root)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def other_package(self, name: str = "harbor_fixture_copy"):
        return load_task_package(create_task(self.root, name))


def make_unit(package, level: str, workspace_before_hash: str = "before", **kwargs) -> HarborTaskUnit:
    return HarborTaskUnit(
        package=package,
        evaluation=package.paths.evaluations[level],
        workspace_before_hash=workspace_before_hash,
        **kwargs,
    )


class HarborTaskMaterializerTests(FixtureTestCase):
    def materialize(self, level: str = "solution", config=None, **unit_kwargs) -> Path:
        unit = make_unit(self.package, level, **unit_kwargs)
        prepared = HarborTaskMaterializer().materialize(
            [unit],
            self.root / f"out-{level}-{len(list(self.root.iterdir()))}",
            config or HarborEvaluationConfig(evaluation_level=level),
        )
        return prepared.task_dir(unit)

    def toml(self, task_dir: Path) -> str:
        return (task_dir / "task.toml").read_text(encoding="utf-8")

    def test_only_selected_evaluation_is_materialized(self):
        task_dir = self.materialize("issue")

        self.assertTrue((task_dir / "tests" / "oracle.md").is_file())
        self.assertFalse((task_dir / "tests" / "oracle").exists())
        self.assertEqual(
            (task_dir / "instruction.md").read_text(encoding="utf-8"),
            "issue instruction\n",
        )
        self.assertIn('PTDBENCH_ORACLE = "/tests/oracle.md"', self.toml(task_dir))

    def test_task_uses_the_prebuilt_image_and_ships_no_dockerfile(self):
        task_dir = self.materialize()

        self.assertIn(
            f'docker_image = "{EnvironmentImage("verl").tag}"', self.toml(task_dir)
        )
        self.assertIn('workdir = "/workspace"', self.toml(task_dir))
        self.assertFalse((task_dir / "environment" / "Dockerfile").exists())
        self.assertFalse((task_dir / "tests" / "Dockerfile").exists())

    def test_environment_directory_is_the_workspace_to_upload(self):
        task_dir = self.materialize()
        environment = task_dir / "environment"

        self.assertTrue((environment / "pkg" / "code.py").is_file())
        self.assertTrue((environment / "train.sh").is_file())
        # 隐藏文件和 prepare.sh 都不进 agent 的 workspace。
        self.assertFalse((environment / "oracle").exists())
        self.assertFalse((environment / "verifier").exists())
        self.assertFalse((environment / "prepare.sh").exists())
        self.assertTrue((task_dir / "tests" / "oracle" / "fix.diff").is_file())
        self.assertTrue((task_dir / "tests" / "verifier" / "test.sh").is_file())

    def test_staged_workspace_wins_over_the_task_workspace(self):
        staged = self.root / "staged"
        (staged / "weights").mkdir(parents=True)
        (staged / "weights" / "model.bin").write_text("w", encoding="utf-8")

        task_dir = self.materialize(workspace=staged, prepared=self.root / "prep")

        uploaded = task_dir / "environment" / "weights" / "model.bin"
        self.assertTrue(uploaded.is_file())
        # 硬链接：几十 GB 的权重不在磁盘上多占一份。
        self.assertEqual(
            uploaded.stat().st_ino, (staged / "weights" / "model.bin").stat().st_ino
        )
        self.assertFalse((task_dir / "environment" / "pkg").exists())

    def test_empty_workspace_still_uploads(self):
        staged = self.root / "empty"
        staged.mkdir()

        task_dir = self.materialize(workspace=staged)

        self.assertTrue(any((task_dir / "environment").iterdir()))

    def test_workspace_with_a_dockerfile_is_rejected(self):
        staged = self.root / "with-dockerfile"
        staged.mkdir()
        (staged / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")

        with self.assertRaisesRegex(HarborAdapterError, "prebuilt image"):
            self.materialize(workspace=staged)

    def test_verifier_shares_the_agent_container(self):
        task_toml = self.toml(self.materialize())

        self.assertIn('environment_mode = "shared"', task_toml)
        self.assertIn('artifacts = [{ source = "/workspace", exclude = [', task_toml)
        self.assertNotIn("PTDBENCH_ORACLE", task_toml.split("[verifier.env]")[0])

    def test_workspace_artifact_skips_weights_and_checkpoints(self):
        artifacts = next(
            line for line in self.toml(self.materialize()).splitlines()
            if line.startswith("artifacts = ")
        )

        for pattern in ("*.safetensors", "*.distcp", "*.pt"):
            self.assertIn(json.dumps(pattern), artifacts)

    def test_evaluation_solution_becomes_the_harbor_oracle_solution(self):
        solution = self.task_root / "evaluations" / "solution" / "solution"
        solution.mkdir()
        (solution / "solve.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        (solution / "helper.py").write_text("", encoding="utf-8")

        task_dir = self.materialize()

        self.assertTrue((task_dir / "solution" / "helper.py").is_file())
        self.assertTrue(os.access(task_dir / "solution" / "solve.sh", os.X_OK))
        # 不进 agent 上传的 workspace，也不进 verifier 的 tests/。
        self.assertFalse((task_dir / "environment" / "solution").exists())
        self.assertFalse((task_dir / "tests" / "solution").exists())

    def test_environment_start_timeout_grows_with_the_uploaded_workspace(self):
        staged = self.root / "big"
        staged.mkdir()
        with open(staged / "model.safetensors", "wb") as handle:
            handle.truncate(3 * (1 << 30))  # 稀疏文件，不真的占 3GB

        small = self.toml(self.materialize())
        big = self.toml(self.materialize(workspace=staged))

        self.assertIn("build_timeout_sec = 660", small)
        self.assertIn("build_timeout_sec = 780", big)

    def test_no_solution_without_solve_script(self):
        self.assertFalse((self.materialize() / "solution").exists())

    def test_both_stages_get_mount_paths_offline_switches_and_task_env(self):
        task_toml = self.toml(self.materialize())
        agent_env, verifier_env = task_toml.split("[environment.env]")[1].split(
            "[verifier.env]"
        )

        for section in (agent_env, verifier_env):
            self.assertIn(f'PTDBENCH_PREPARED = "{PREPARED_DIR}"', section)
            self.assertIn(f'PTDBENCH_MODELS = "{MODELS_DIR}"', section)
            self.assertIn(f'PTDBENCH_DATASETS = "{DATASETS_DIR}"', section)
            self.assertIn(f'PTDBENCH_ENVIRONMENT = "{PTDBENCH_ENVIRONMENT_DIR}"', section)
            self.assertIn(f'BASH_ENV = "{ACTIVATE_HOOK}"', section)
            self.assertIn('HF_HUB_OFFLINE = "1"', section)
            self.assertIn('WANDB_MODE = "offline"', section)
            self.assertIn('CUDA_VISIBLE_DEVICES = "0"', section)
        self.assertIn('PTDBENCH_STAGE = "agent"', agent_env)
        self.assertIn('PTDBENCH_STAGE = "verifier"', verifier_env)

    def test_only_the_verifier_learns_the_task_id(self):
        # 任务 id 往往直接点出缺陷，agent 跑一下 `env` 就能看到。
        task_toml = self.toml(self.materialize())
        agent_env, verifier_env = task_toml.split("[environment.env]")[1].split(
            "[verifier.env]"
        )
        self.assertNotIn("PTDBENCH_TASK_ID", agent_env)
        self.assertIn("PTDBENCH_TASK_ID = ", verifier_env)

    def test_no_gpu_option_is_written(self):
        # Harbor 的 docker 环境不支持 GPU；写了 gpus 会让 trial 初始化失败。
        self.assertNotIn("gpus", self.toml(self.materialize()))

    def test_wrapper_activates_the_hook_and_writes_harbor_reward(self):
        task_dir = self.materialize()
        wrapper = (task_dir / "tests" / "test.sh").read_text(encoding="utf-8")

        self.assertIn(f". {ACTIVATE_HOOK}", wrapper)
        self.assertLess(
            wrapper.index(ACTIVATE_HOOK), wrapper.index('bash "$PTDBENCH_VERIFIER/test.sh"')
        )
        self.assertIn("ptdbench-result.json", wrapper)
        self.assertIn("/logs/verifier/reward.txt", wrapper)

    def test_each_task_gets_its_own_directory_under_one_batch(self):
        units = [
            make_unit(self.package, "solution"),
            make_unit(self.other_package(), "solution"),
        ]

        prepared = HarborTaskMaterializer().materialize(
            units, self.root / "batch", HarborEvaluationConfig()
        )

        self.assertEqual(prepared.tasks_dir, (self.root / "batch" / "tasks").resolve())
        self.assertEqual(
            {path.name for path in prepared.tasks_dir.iterdir()},
            {unit.dir_name for unit in units},
        )

    def test_duplicate_task_names_are_rejected(self):
        units = [make_unit(self.package, "solution")] * 2

        with self.assertRaisesRegex(HarborAdapterError, "duplicate Harbor task name"):
            HarborTaskMaterializer().materialize(
                units, self.root / "duplicate", HarborEvaluationConfig()
            )

    def test_common_agent_and_verifier_settings_enter_task_config(self):
        task_toml = self.toml(
            self.materialize(
                config=HarborEvaluationConfig(
                    judge_model="judge-model",
                    agent_network="no-network",
                    verifier_network="public",
                )
            )
        )

        self.assertIn('PTDBENCH_JUDGE_MODEL = "judge-model"', task_toml)
        self.assertIn("${OPENAI_BASE_URL:-https://api.openai.com/v1}", task_toml)
        self.assertIn("${OPENAI_API_KEY:-}", task_toml)
        self.assertIn('network_mode = "no-network"', task_toml)
        self.assertIn('network_mode = "public"', task_toml)

    def test_manifest_records_the_image(self):
        task_dir = self.materialize()
        manifest = json.loads(
            (task_dir / "adapter-manifest.json").read_text(encoding="utf-8")
        )

        self.assertEqual(manifest["docker_image"], EnvironmentImage("verl").tag)
        self.assertEqual(manifest["verifier_mode"], "shared")
        self.assertNotIn("debian_mirror", manifest)


class HarborRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._fixture = tempfile.TemporaryDirectory()
        self.addCleanup(self._fixture.cleanup)
        self.package = load_task_package(create_task(Path(self._fixture.name)))

    def _prepared(self, root: Path, create: bool = True, **unit_kwargs) -> PreparedHarborTask:
        prepared = PreparedHarborTask(
            root=root,
            tasks_dir=root / "tasks",
            units=[make_unit(self.package, "solution", **unit_kwargs)],
        )
        if create:
            prepared.tasks_dir.mkdir(parents=True)
        return prepared

    @patch.object(HarborRunner, "_run_streaming")
    def test_runner_builds_a_secret_free_harbor_command(self, run_mock):
        run_mock.return_value.returncode = 0
        run_mock.return_value.stdout = "done"
        run_mock.return_value.stderr = ""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prepared = self._prepared(root)
            config = HarborEvaluationConfig(
                harbor_command=("uv", "run", "harbor"),
                harbor_args=(
                    "--job-name",
                    "native-job",
                    "--jobs-dir",
                    str(root / "jobs"),
                    "--agent",
                    "codex",
                    "--model",
                    "gpt-test",
                    "--env",
                    "docker",
                    "--agent-kwarg",
                    "reasoning_effort=high",
                    "--n-concurrent",
                    "2",
                    "--agent-env",
                    "CUSTOM_TOKEN=raw-secret",
                ),
            )

            harbor_run = HarborRunner().run(prepared, config)

        argv = run_mock.call_args.args[0]
        self.assertEqual(argv[:4], ["uv", "run", "harbor", "run"])
        self.assertIn("--path", argv)
        self.assertEqual(argv[argv.index("--path") + 1], str(prepared.tasks_dir))
        self.assertIn("--jobs-dir", argv)
        self.assertIn("--model", argv)
        self.assertIn("reasoning_effort=high", argv)
        self.assertIn("--n-concurrent", argv)
        self.assertIn("CUSTOM_TOKEN=raw-secret", argv)
        self.assertNotIn("raw-secret", " ".join(harbor_run.command.argv))
        self.assertEqual(run_mock.call_args.args[1], Path.cwd())
        self.assertEqual(
            harbor_run.job_dir,
            (root / "jobs" / "native-job").resolve(),
        )

    def test_only_materialized_task_path_is_reserved(self):
        config = HarborEvaluationConfig(harbor_args=("--path", "other-task"))
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared = self._prepared(Path(temp_dir), create=False)
            with self.assertRaisesRegex(ValueError, "task materializer"):
                HarborRunner().run(prepared, config)

    @patch.object(HarborRunner, "_run_streaming")
    def test_job_config_is_forwarded_without_overriding_its_output(self, run_mock):
        run_mock.return_value.returncode = 0
        run_mock.return_value.stdout = "Results written to configured/job/result.json\n"
        run_mock.return_value.stderr = ""
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared = self._prepared(Path(temp_dir))

            harbor_run = HarborRunner().run(
                prepared,
                HarborEvaluationConfig(harbor_args=("--config", "job.yaml")),
            )

        argv = run_mock.call_args.args[0]
        self.assertEqual(argv[-2:], ["--config", "job.yaml"])
        self.assertIn("--mounts-json", argv)
        self.assertNotIn("--jobs-dir", argv)
        self.assertNotIn("--job-name", argv)
        self.assertEqual(
            harbor_run.job_dir,
            (Path.cwd() / "configured" / "job").resolve(),
        )

    @patch.object(HarborRunner, "_run_streaming")
    def test_prepare_output_and_caches_are_mounted_read_only(self, run_mock):
        run_mock.return_value.returncode = 0
        run_mock.return_value.stdout = ""
        run_mock.return_value.stderr = ""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prepared = self._prepared(root, prepared=root / "prep")
            config = HarborEvaluationConfig(
                models_dir=root / "models", datasets_dir=root / "datasets"
            )

            HarborRunner().run(prepared, config)

        argv = run_mock.call_args.args[0]
        mounts = json.loads(argv[argv.index("--mounts-json") + 1])
        by_target = {mount["target"]: mount for mount in mounts}
        self.assertEqual(
            set(by_target),
            {PREPARED_DIR, PTDBENCH_ENVIRONMENT_DIR, MODELS_DIR, DATASETS_DIR},
        )
        self.assertEqual(by_target[PREPARED_DIR]["source"], str((root / "prep").resolve()))
        self.assertEqual(
            by_target[PTDBENCH_ENVIRONMENT_DIR]["source"],
            str(self.package.paths.environment.resolve()),
        )
        self.assertEqual(by_target[MODELS_DIR]["source"], str((root / "models").resolve()))
        for mount in mounts:
            self.assertEqual(mount["type"], "bind")
            self.assertTrue(mount["read_only"])

    @patch.object(HarborRunner, "_run_streaming")
    def test_task_compose_file_is_passed_as_an_overlay(self, run_mock):
        run_mock.return_value.returncode = 0
        run_mock.return_value.stdout = ""
        run_mock.return_value.stderr = ""
        compose = self.package.paths.environment / "docker-compose.yaml"
        compose.write_text("services:\n  main:\n    shm_size: 16gb\n", encoding="utf-8")
        with tempfile.TemporaryDirectory() as temp_dir:
            HarborRunner().run(self._prepared(Path(temp_dir)), HarborEvaluationConfig())

        argv = run_mock.call_args.args[0]
        self.assertEqual(
            argv[argv.index("--extra-docker-compose") + 1], str(compose.resolve())
        )

    @patch.object(HarborRunner, "_run_streaming")
    def test_no_overlay_without_task_compose_file(self, run_mock):
        run_mock.return_value.returncode = 0
        run_mock.return_value.stdout = ""
        run_mock.return_value.stderr = ""
        with tempfile.TemporaryDirectory() as temp_dir:
            HarborRunner().run(self._prepared(Path(temp_dir)), HarborEvaluationConfig())

        self.assertNotIn("--extra-docker-compose", run_mock.call_args.args[0])

    def test_user_mounts_are_rejected(self):
        for flag in ("--mounts", "--mounts-json", "--mounts-json=[]"):
            config = HarborEvaluationConfig(harbor_args=(flag, "[]"))
            with tempfile.TemporaryDirectory() as temp_dir:
                prepared = self._prepared(Path(temp_dir), create=False)
                with self.assertRaisesRegex(ValueError, "managed by PTDBench"):
                    HarborRunner().run(prepared, config)

    def test_a_job_holds_exactly_one_task(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prepared = PreparedHarborTask(
                root=root,
                tasks_dir=root / "tasks",
                units=[make_unit(self.package, "solution"), make_unit(self.package, "issue")],
            )
            with self.assertRaisesRegex(HarborAdapterError, "one PTDBench task per job"):
                HarborRunner().run(prepared, HarborEvaluationConfig())

    @patch.object(HarborRunner, "_run_streaming")
    def test_batch_suffix_keeps_user_job_names_apart(self, run_mock):
        run_mock.return_value.returncode = 0
        run_mock.return_value.stdout = ""
        run_mock.return_value.stderr = ""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prepared = self._prepared(root)
            config = HarborEvaluationConfig(
                harbor_args=("--job-name", "night", "--jobs-dir", str(root / "jobs"))
            )

            harbor_run = HarborRunner().run(prepared, config, job_name_suffix="t1")

        argv = run_mock.call_args.args[0]
        self.assertEqual(argv[argv.index("--job-name") + 1], "night-t1")
        self.assertEqual(argv.count("--job-name"), 1)
        self.assertEqual(harbor_run.job_dir, (root / "jobs" / "night-t1").resolve())

    def test_cli_accepts_several_task_roots_and_forwards_harbor_arguments(self):
        args, harbor_args = build_parser().parse_known_args(
            [
                "harbor-evaluate",
                "task-a",
                "task-b",
                "--evaluation-level",
                "issue",
                "--judge-model",
                "judge-model",
                "--harbor-temp-dir",
                "custom-temp",
                "--huggingface-mirror",
                "https://hf-mirror.com",
                "--models-dir",
                "m",
                "--agent",
                "codex",
                "--model",
                "gpt-test",
                "--env",
                "daytona",
                "--n-concurrent",
                "8",
            ]
        )

        self.assertEqual(args.task_root, ["task-a", "task-b"])
        self.assertEqual(args.evaluation_level, "issue")
        self.assertEqual(args.judge_model, "judge-model")
        self.assertEqual(args.harbor_temp_dir, Path("custom-temp"))
        self.assertEqual(args.huggingface_mirror, "https://hf-mirror.com")
        self.assertEqual(args.models_dir, "m")
        self.assertEqual(
            harbor_args,
            [
                "--agent",
                "codex",
                "--model",
                "gpt-test",
                "--env",
                "daytona",
                "--n-concurrent",
                "8",
            ],
        )

    def test_temp_run_directory_uses_configured_parent(self):
        package = self.package
        with tempfile.TemporaryDirectory() as temp_dir:
            parent = Path(temp_dir) / "harbor-tmp"
            run_dir = HarborEvaluator._make_run_dir(
                [make_unit(package, "solution")],
                parent,
            )
            try:
                self.assertEqual(run_dir.parent, parent.resolve())
                self.assertTrue(
                    run_dir.name.startswith("ptdbench-harbor-harbor_fixture-solution-")
                )
            finally:
                shutil.rmtree(run_dir)

    def test_temp_run_directory_is_named_for_the_batch(self):
        package = self.package
        with tempfile.TemporaryDirectory() as temp_dir:
            parent = Path(temp_dir) / "harbor-tmp"
            run_dir = HarborEvaluator._make_run_dir(
                [make_unit(package, "solution"), make_unit(package, "issue")],
                parent,
            )
            try:
                self.assertTrue(run_dir.name.startswith("ptdbench-harbor-batch-2-"))
            finally:
                shutil.rmtree(run_dir)


class ExpandTaskRootsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _task(self, path: Path) -> Path:
        path.mkdir(parents=True)
        (path / "metadata.yaml").write_text("id: x\n", encoding="utf-8")
        return path

    def test_task_directory_is_used_as_is(self):
        task = self._task(self.root / "task-a")

        self.assertEqual(expand_task_roots([str(task)]), [task])

    def test_parent_directory_expands_to_its_tasks(self):
        family = self.root / "family"
        second = self._task(family / "task-b")
        first = self._task(family / "task-a")
        (family / "not-a-task").mkdir()

        self.assertEqual(expand_task_roots([str(family)]), [first, second])

    def test_repeated_paths_are_submitted_once(self):
        family = self.root / "family"
        task = self._task(family / "task-a")

        roots = expand_task_roots([str(family), str(task)])

        self.assertEqual(roots, [task])

    def test_directory_without_tasks_is_rejected(self):
        empty = self.root / "empty"
        empty.mkdir()

        with self.assertRaises(SystemExit):
            expand_task_roots([str(empty)])

    def test_missing_path_is_rejected(self):
        with self.assertRaises(SystemExit):
            expand_task_roots([str(self.root / "absent")])


class HarborResultMapperTests(FixtureTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.unit = make_unit(self.package, "solution")
        self.prepared = PreparedHarborTask(
            root=self.root / "prepared",
            tasks_dir=self.root / "prepared" / "tasks",
            units=[self.unit],
        )
        self.job_dir = self.root / "job"
        self.trial_dir = self.job_dir / "trial-1"
        (self.trial_dir / "artifacts").mkdir(parents=True)
        (self.trial_dir / "verifier").mkdir()
        shutil.copytree(
            self.package.paths.workspace,
            self.trial_dir / "artifacts" / "workspace",
        )
        self._write_manifest("ok")
        (self.trial_dir / "verifier" / "ptdbench-result.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "score": 0.75,
                    "metrics": {"tests_run": 4},
                }
            ),
            encoding="utf-8",
        )
        (self.trial_dir / "result.json").write_text(
            json.dumps(
                {
                    "trial_name": "trial-1",
                    "task_name": self.unit.harbor_task_name,
                    "verifier_environment_mode": "shared",
                    "verifier_result": {"rewards": {"reward": 0.75}},
                    "exception_info": None,
                }
            ),
            encoding="utf-8",
        )
        (self.job_dir / "result.json").write_text(
            json.dumps({"n_total_trials": 1, "stats": {"n_completed_trials": 1}}),
            encoding="utf-8",
        )

    def _write_manifest(self, status: str) -> None:
        (self.trial_dir / "artifacts" / "manifest.json").write_text(
            json.dumps(
                [
                    {
                        "source": "/workspace",
                        "destination": "artifacts/workspace",
                        "type": "directory",
                        "status": status,
                    }
                ]
            ),
            encoding="utf-8",
        )

    def _command(self, returncode: int = 0, stderr: str = "") -> CommandResult:
        return CommandResult(
            argv=["harbor", "run"],
            returncode=returncode,
            stdout="",
            stderr=stderr,
            timed_out=False,
            duration_s=1.0,
            cwd=str(self.root),
        )

    def _report(self, prepared: PreparedHarborTask | None = None):
        return HarborResultMapper().map(
            prepared or self.prepared,
            HarborRun(command=self._command(), job_dir=self.job_dir),
            HarborEvaluationConfig(),
        )

    def _map(self):
        return self._report().trials[0]

    def test_maps_harbor_trial_and_ptdbench_metrics(self):
        result = self._map()

        self.assertTrue(result.passed)
        self.assertEqual(result.score, 0.75)
        self.assertEqual(result.task_id, self.package.task_id)
        self.assertEqual(result.workspace_before_hash, "before")
        self.assertEqual(result.metrics["tests_run"], 4)
        self.assertEqual(result.metrics["harbor_rewards"], {"reward": 0.75})
        self.assertTrue((self.trial_dir / "report.json").is_file())
        self.assertTrue(
            (self.trial_dir / "ptdbench-harbor-command.json").is_file()
        )

    def test_workspace_artifact_failure_fails_closed(self):
        self._write_manifest("failed")

        result = self._map()

        self.assertFalse(result.passed)
        self.assertEqual(result.score, 0.0)
        self.assertIn("/workspace", result.metrics["infrastructure_errors"][0])

    def test_maps_all_trials_into_a_job_report(self):
        second_trial = self.job_dir / "trial-2"
        shutil.copytree(self.trial_dir, second_trial)
        second_result = json.loads(
            (second_trial / "result.json").read_text(encoding="utf-8")
        )
        second_result["trial_name"] = "trial-2"
        (second_trial / "result.json").write_text(
            json.dumps(second_result), encoding="utf-8"
        )

        report = self._report()

        self.assertEqual(len(report.trials), 2)
        self.assertTrue(report.passed)
        self.assertEqual(report.score, 0.75)
        self.assertEqual(report.to_dict()["task_ids"], [self.package.task_id])
        self.assertTrue((self.job_dir / "ptdbench-report.json").is_file())

    def test_each_trial_is_mapped_back_to_its_own_task(self):
        other_unit = make_unit(self.other_package(), "solution", "other-before")
        second_trial = self.job_dir / "trial-2"
        shutil.copytree(self.trial_dir, second_trial)
        (second_trial / "result.json").write_text(
            json.dumps(
                {
                    "trial_name": "trial-2",
                    "task_name": other_unit.harbor_task_name,
                    "verifier_environment_mode": "shared",
                    "verifier_result": {"rewards": {"reward": 0.75}},
                    "exception_info": None,
                }
            ),
            encoding="utf-8",
        )
        prepared = PreparedHarborTask(
            root=self.prepared.root,
            tasks_dir=self.prepared.tasks_dir,
            units=[self.unit, other_unit],
        )

        report = self._report(prepared)

        by_task = {trial.task_id: trial for trial in report.trials}
        self.assertEqual(
            set(by_task),
            {self.package.task_id, other_unit.package.task_id},
        )
        self.assertEqual(by_task[self.package.task_id].workspace_before_hash, "before")
        self.assertEqual(
            by_task[other_unit.package.task_id].workspace_before_hash,
            "other-before",
        )
        self.assertEqual(
            report.to_dict()["task_ids"],
            [self.package.task_id, other_unit.package.task_id],
        )
        self.assertEqual(report.to_dict()["n_completed_tasks"], 2)
        self.assertEqual(report.to_dict()["missing_task_ids"], [])

    def test_report_fails_when_a_materialized_task_has_no_trial(self):
        other_unit = make_unit(self.other_package(), "solution")
        prepared = PreparedHarborTask(
            root=self.prepared.root,
            tasks_dir=self.prepared.tasks_dir,
            units=[self.unit, other_unit],
        )

        report = self._report(prepared)

        self.assertFalse(report.passed)
        self.assertEqual(report.completed_task_ids, (self.package.task_id,))
        self.assertEqual(report.missing_task_ids, (other_unit.package.task_id,))
        self.assertEqual(report.to_dict()["n_completed_tasks"], 1)
        self.assertEqual(
            report.to_dict()["missing_task_ids"],
            [other_unit.package.task_id],
        )

    @patch("ptdbench.cli.HarborEvaluator.evaluate")
    def test_cli_keeps_batch_report_when_only_one_task_has_a_trial(self, evaluate):
        other_unit = make_unit(self.other_package(), "solution")
        prepared = PreparedHarborTask(
            root=self.prepared.root,
            tasks_dir=self.prepared.tasks_dir,
            units=[self.unit, other_unit],
        )
        evaluate.return_value = self._report(prepared)
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            returncode = cli_main(
                [
                    "harbor-evaluate",
                    str(self.task_root),
                    str(other_unit.package.paths.root),
                ]
            )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(returncode, 1)
        self.assertEqual(
            payload["task_ids"],
            [self.package.task_id, other_unit.package.task_id],
        )
        self.assertEqual(payload["missing_task_ids"], [other_unit.package.task_id])

    def test_unknown_task_name_in_a_batch_is_an_error(self):
        prepared = PreparedHarborTask(
            root=self.prepared.root,
            tasks_dir=self.prepared.tasks_dir,
            units=[self.unit, make_unit(self.other_package(), "solution")],
        )
        trial = json.loads((self.trial_dir / "result.json").read_text(encoding="utf-8"))
        trial["task_name"] = "ptdbench/unrelated-solution"
        (self.trial_dir / "result.json").write_text(
            json.dumps(trial), encoding="utf-8"
        )

        with self.assertRaisesRegex(HarborAdapterError, "does not match any"):
            self._report(prepared)

    def test_reads_legacy_embedded_trial_results(self):
        (self.trial_dir / "result.json").unlink()
        (self.job_dir / "result.json").write_text(
            json.dumps(
                {
                    "trial_results": [
                        {
                            "trial_name": "trial-1",
                            "task_name": self.unit.harbor_task_name,
                            "verifier_environment_mode": "shared",
                            "verifier_result": {"rewards": {"reward": 0.75}},
                            "exception_info": None,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        result = self._map()

        self.assertTrue(result.passed)
        self.assertEqual(result.score, 0.75)

    def test_single_task_trial_without_task_name_still_maps(self):
        trial = json.loads((self.trial_dir / "result.json").read_text(encoding="utf-8"))
        del trial["task_name"]
        (self.trial_dir / "result.json").write_text(
            json.dumps(trial), encoding="utf-8"
        )

        result = self._map()

        self.assertEqual(result.task_id, self.package.task_id)

    def test_unknown_task_name_in_a_single_task_job_is_an_error(self):
        trial = json.loads((self.trial_dir / "result.json").read_text(encoding="utf-8"))
        trial["task_name"] = "ptdbench/unrelated-solution"
        (self.trial_dir / "result.json").write_text(
            json.dumps(trial), encoding="utf-8"
        )

        with self.assertRaisesRegex(HarborAdapterError, "does not match any"):
            self._report()

    def test_failed_harbor_command_without_result_has_a_clear_error(self):
        (self.job_dir / "result.json").unlink()

        with self.assertRaisesRegex(HarborAdapterError, "exit code 2"):
            HarborResultMapper().map(
                self.prepared,
                HarborRun(
                    command=self._command(returncode=2, stderr="invalid option"),
                    job_dir=self.job_dir,
                ),
                HarborEvaluationConfig(),
            )


class HarborPreparerTests(FixtureTestCase):
    def _preparer(self, prepare_result):
        evaluator = MagicMock()
        staged = MagicMock(workspace=self.root / "ws", prepared=self.root / "prep")
        evaluator._prepare_run.return_value = staged
        evaluator.runtime.ensure_environment.return_value = MagicMock(reference="img:tag")
        evaluator._run_prepare.return_value = prepare_result
        return HarborPreparer(evaluator), evaluator

    def _result(self, returncode: int) -> CommandResult:
        return CommandResult(
            argv=["prepare"], returncode=returncode, stdout="", stderr="boom\n",
            timed_out=False, duration_s=0.1, cwd="/",
        )

    def test_prepare_runs_in_the_task_image_and_stages_the_unit(self):
        preparer, evaluator = self._preparer(self._result(0))
        config = HarborEvaluationConfig(
            huggingface_mirror="https://hf-mirror.com/",
            models_dir=self.root / "models",
            datasets_dir=self.root / "datasets",
        )

        unit = preparer.prepare(make_unit(self.package, "solution"), self.root / "w", config)

        self.assertEqual(unit.workspace, self.root / "ws")
        self.assertEqual(unit.prepared, self.root / "prep")
        self.assertEqual(
            evaluator.runtime.ensure_environment.call_args.args[0], EnvironmentImage("verl")
        )
        eval_config = evaluator._prepare_run.call_args.args[2]
        self.assertEqual(eval_config.huggingface_mirror, "https://hf-mirror.com")
        self.assertEqual(eval_config.models_dir, (self.root / "models").resolve())
        self.assertEqual(eval_config.work_dir, self.root / "w")

    def test_prepare_failure_stops_the_task(self):
        preparer, _ = self._preparer(self._result(3))

        with self.assertRaisesRegex(HarborAdapterError, "prepare failed with exit 3"):
            preparer.prepare(
                make_unit(self.package, "solution"), self.root / "w", HarborEvaluationConfig()
            )

    def test_task_without_prepare_still_stages(self):
        preparer, _ = self._preparer(None)

        unit = preparer.prepare(
            make_unit(self.package, "solution"), self.root / "w", HarborEvaluationConfig()
        )

        self.assertEqual(unit.prepared, self.root / "prep")


class HarborEvaluatorFlowTests(FixtureTestCase):
    """prepare → 物化 → 每个 task 一个 Harbor job → 合并报告。"""

    def _evaluator(self, fail_prepare_for=()):
        preparer = MagicMock()

        def prepare(unit, work_dir, config):
            if unit.package.task_id in fail_prepare_for:
                raise HarborAdapterError("prepare failed with exit 1")
            workspace = work_dir / "workspace"
            shutil.copytree(unit.package.paths.workspace, workspace)
            return HarborTaskUnit(
                package=unit.package,
                evaluation=unit.evaluation,
                workspace_before_hash=unit.workspace_before_hash,
                workspace=workspace,
                prepared=work_dir,
            )

        preparer.prepare.side_effect = prepare
        runner = MagicMock()
        seen = []

        def run(prepared, config, job_name_suffix=None):
            task_dir = prepared.task_dir(prepared.units[0])
            seen.append(
                (
                    [unit.package.task_id for unit in prepared.units],
                    job_name_suffix,
                    (task_dir / "task.toml").read_text(encoding="utf-8"),
                    sorted(p.name for p in (task_dir / "environment").iterdir()),
                )
            )
            return HarborRun(command=_command(), job_dir=self.root / f"job{len(seen)}")

        runner.run.side_effect = run
        mapper = MagicMock()

        def map_(prepared, harbor_run, config):
            unit = prepared.units[0]
            return HarborEvaluationReport(
                task_ids=[unit.package.task_id],
                evaluation_level=config.evaluation_level,
                trials=[_trial(unit.package.task_id, harbor_run.job_dir)],
                report_dir=harbor_run.job_dir,
                command=harbor_run.command,
            )

        mapper.map.side_effect = map_
        evaluator = HarborEvaluator(
            runner=runner, result_mapper=mapper, preparer=preparer
        )
        return evaluator, seen

    def test_single_task_is_prepared_then_run_on_its_prebuilt_image(self):
        evaluator, seen = self._evaluator()

        report = evaluator.evaluate(
            self.task_root, HarborEvaluationConfig(temp_dir=self.root / "tmp")
        )

        self.assertEqual(report.task_ids, (self.package.task_id,))
        (task_ids, suffix, task_toml, uploaded), = seen
        self.assertIsNone(suffix)
        self.assertIn("docker_image", task_toml)
        self.assertIn("pkg", uploaded)
        # 临时目录（prepare staging + 物化 task）跑完即删。
        self.assertEqual(list((self.root / "tmp").iterdir()), [])

    def test_batch_runs_one_job_per_task_and_merges_reports(self):
        other = self.other_package()
        evaluator, seen = self._evaluator()

        report = evaluator.evaluate([self.task_root, other.paths.root])

        self.assertEqual([entry[0] for entry in seen], [[self.package.task_id], [other.task_id]])
        self.assertTrue(all(entry[1] for entry in seen))
        self.assertEqual(report.task_ids, (self.package.task_id, other.task_id))
        self.assertEqual(len(report.trials), 2)
        self.assertEqual(report.report_dir, self.root / "job2")
        self.assertTrue(report.passed)

    def test_prepare_failure_in_a_batch_leaves_the_task_missing(self):
        other = self.other_package()
        evaluator, seen = self._evaluator(fail_prepare_for={other.task_id})

        report = evaluator.evaluate([self.task_root, other.paths.root])

        self.assertEqual(len(seen), 1)
        self.assertEqual(report.missing_task_ids, (other.task_id,))
        self.assertFalse(report.passed)

    def test_prepare_failure_of_a_single_task_propagates(self):
        evaluator, _ = self._evaluator(fail_prepare_for={self.package.task_id})

        with self.assertRaisesRegex(HarborAdapterError, "prepare failed"):
            evaluator.evaluate(self.task_root)


def _command() -> CommandResult:
    return CommandResult(
        argv=["harbor", "run"], returncode=0, stdout="", stderr="",
        timed_out=False, duration_s=1.0, cwd="/",
    )


def _trial(task_id: str, report_dir: Path):
    from ptdbench.core.types import EvaluationResult

    return EvaluationResult(
        task_id=task_id, evaluation_level="solution", passed=True, score=1.0,
        metrics={}, report_dir=report_dir, agent_spec=None, agent_setup=None,
        agent=None, verifier=None, workspace_before_hash="b",
        workspace_after_hash="a", metadata={},
    )


class RemoteBackendTests(FixtureTestCase):
    """`--env cybertron`：prepare 在 devspace 上跑，本地既不 prepare 也不挂载。

    这条路以前从没端到端通过：本地 prepare 要 x86-64 CUDA 镜像（arm64 机器上根本
    起不来），而 _mounts / compose overlay 发的都是本机路径、后端又声明
    mounted=False，GPU 数更是没人写进 task.toml，devspace 会开成 0 卡。
    """

    def _prepared(self, package=None) -> PreparedHarborTask:
        unit = make_unit(package or self.package, "solution")
        return PreparedHarborTask(
            root=self.root / "harbor",
            tasks_dir=self.root / "harbor" / "tasks",
            units=[unit],
        )

    @staticmethod
    def _config(*harbor_args: str, **kwargs) -> HarborEvaluationConfig:
        return HarborEvaluationConfig(harbor_args=harbor_args, **kwargs)

    def test_detects_the_backend_from_either_option_spelling(self) -> None:
        for args in (("-e", "cybertron"), ("--env", "cybertron"), ("--env=cybertron",)):
            with self.subTest(args=args):
                self.assertTrue(_is_remote_backend(args))
        for args in ((), ("-e", "docker"), ("--env", "daytona")):
            with self.subTest(args=args):
                self.assertFalse(_is_remote_backend(args))

    def test_no_mounts_are_sent_to_a_remote_backend(self) -> None:
        prepared = self._prepared()
        self.assertEqual(HarborRunner._mounts(prepared, self._config("-e", "cybertron")), [])
        # docker 这条路一个挂载都不能少。
        self.assertEqual(
            {mount["target"] for mount in HarborRunner._mounts(prepared, self._config("-e", "docker"))},
            {PTDBENCH_ENVIRONMENT_DIR, MODELS_DIR, DATASETS_DIR},
        )

    def test_no_compose_overlay_is_sent_to_a_remote_backend(self) -> None:
        compose = self.package.paths.environment / "docker-compose.yaml"
        compose.write_text("services: {}\n", encoding="utf-8")
        prepared = self._prepared(load_task_package(self.task_root))
        self.assertEqual(
            HarborRunner._compose_overlay_args(prepared, ("-e", "cybertron")), []
        )
        self.assertEqual(
            HarborRunner._compose_overlay_args(prepared, ("-e", "docker")),
            ["--extra-docker-compose", str(compose.resolve())],
        )

    def test_remote_prepare_paths_are_injected_as_environment_kwargs(self) -> None:
        args = HarborRunner._remote_prepare_args(self._prepared(), ("-e", "cybertron"))
        self.assertEqual(
            args,
            [
                "--environment-kwarg",
                f"prepare_dir={self.package.paths.environment.resolve()}",
                "--environment-kwarg",
                f"tooling_dir={DOCKER_DIR.resolve()}",
            ],
        )
        self.assertEqual(HarborRunner._remote_prepare_args(self._prepared(), ("-e", "docker")), [])

    def test_an_explicit_prepare_dir_is_not_overridden(self) -> None:
        """重复选项要看全部取值，只看最后一个会把用户给的覆盖掉。"""
        args = HarborRunner._remote_prepare_args(
            self._prepared(),
            ("-e", "cybertron", "--environment-kwarg", "prepare_dir=/custom"),
        )
        self.assertNotIn("prepare_dir=/custom", args)
        self.assertEqual(
            args, ["--environment-kwarg", f"tooling_dir={DOCKER_DIR.resolve()}"]
        )

    def test_gpus_reaches_the_task_toml_for_a_remote_backend(self) -> None:
        """后端只认 task.toml 的 gpus；不写就会开出 0 卡的 devspace。"""
        # fixture 的 CUDA_VISIBLE_DEVICES 是 "0" -> 1 张。
        self.assertEqual(
            HarborTaskMaterializer._toml_gpus(self.package, self._config("-e", "cybertron")),
            "gpus = 1\n",
        )
        self.assertEqual(
            HarborTaskMaterializer._toml_gpus(
                self.package, self._config("-e", "cybertron", gpus=4)
            ),
            "gpus = 4\n",
        )
        # docker 后端拒收 task.toml 里的 gpus，一行都不能写。
        self.assertEqual(
            HarborTaskMaterializer._toml_gpus(self.package, self._config("-e", "docker")), ""
        )

    def test_materialized_task_toml_declares_gpus(self) -> None:
        destination = self.root / "materialized"
        config = self._config("-e", "cybertron")
        prepared = HarborTaskMaterializer().materialize(
            [make_unit(self.package, "solution")], destination, config
        )
        task_toml = (prepared.task_dir(prepared.units[0]) / "task.toml").read_text(
            encoding="utf-8"
        )
        self.assertIn("gpus = 1", task_toml)

    def test_local_prepare_is_skipped_for_a_remote_backend(self) -> None:
        """本地 prepare 要 x86-64 CUDA 镜像，远端后端上不该被调用。"""
        preparer = MagicMock(spec=HarborPreparer)
        runner = MagicMock(spec=HarborRunner)
        runner.run.return_value = HarborRun(command=_command(), job_dir=self.root / "jobs")
        mapper = MagicMock(spec=HarborResultMapper)
        mapper.map.return_value = HarborEvaluationReport(
            task_ids=["harbor_fixture"], evaluation_level="solution",
            trials=[], report_dir=self.root / "jobs", command=_command(),
        )
        evaluator = HarborEvaluator(
            materializer=HarborTaskMaterializer(), runner=runner,
            result_mapper=mapper, preparer=preparer,
        )
        with redirect_stdout(io.StringIO()):
            evaluator.evaluate([self.task_root], self._config("-e", "cybertron"))
        preparer.prepare.assert_not_called()
        # 没有本地 prepare，上传的 workspace 就是任务自带那棵树。
        unit = runner.run.call_args.args[0].units[0]
        self.assertIsNone(unit.workspace)
        self.assertIsNone(unit.prepared)


if __name__ == "__main__":
    unittest.main()
