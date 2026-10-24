"""environment 段：`image` 指向 docker/images/<name>.Dockerfile，外加 prepare / env。"""

from __future__ import annotations

import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ptdbench.core.types import IMAGES_DIR, EnvironmentImage, known_images
from ptdbench.tasks import SchemaError, load_task_package
from ptdbench.tasks.validator import validate_metadata


METADATA_HEAD = """title: Image task
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
"""

METADATA_TAIL = """outcome:
  failure_type: test
  failure_reason: test
"""


def create_task(
    root: Path,
    environment: str,
    *,
    prepare: str | None = None,
) -> Path:
    """铺一个最小任务，environment 段由调用方给。"""
    task = root / "image-task"
    for directory in ("workspace", "oracle", "environment", "verifier"):
        (task / directory).mkdir(parents=True, exist_ok=True)
    (task / "instruction.md").write_text("Fix it.\n", encoding="utf-8")
    # metadata 里的 training.script 指向它；校验会检查这个文件真的在 workspace 下。
    (task / "workspace" / "train.sh").write_text(
        "#!/usr/bin/env bash\nexit 0\n", encoding="utf-8"
    )
    (task / "verifier" / "test.sh").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    (task / "metadata.yaml").write_text(
        METADATA_HEAD + environment + METADATA_TAIL, encoding="utf-8"
    )
    if prepare is not None:
        (task / "environment" / "prepare.sh").write_text(prepare, encoding="utf-8")
    return task


class EnvironmentPlanTests(unittest.TestCase):
    def _load(self, environment: str, **kwargs):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        return load_task_package(create_task(Path(temp.name), environment, **kwargs))

    # --- loader -------------------------------------------------------------

    def test_image_task_picks_up_the_conventional_prepare(self) -> None:
        # prepare.sh 几乎每个任务都有；逼着每份 metadata.yaml 再抄一遍路径只会
        # 增加抄错的机会。
        package = self._load(
            "environment:\n  image: verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12\n",
            prepare="#!/usr/bin/env bash\n",
        )
        plan = package.environment
        self.assertTrue(package.report.ok, package.report.to_dict())
        self.assertEqual(plan.image, "verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12")
        self.assertEqual(plan.environment_image, EnvironmentImage("verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12"))
        self.assertEqual(plan.prepare.name, "prepare.sh")

    def test_explicit_prepare_wins_over_the_convention(self) -> None:
        package = self._load(
            "environment:\n  image: verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12\n  prepare: environment/fetch.sh\n",
        )
        self.assertEqual(package.environment.prepare.name, "fetch.sh")

    def test_absent_prepare_leaves_the_plan_empty_rather_than_dangling(self) -> None:
        # plan 里带一个不存在的路径，会把"任务没写 prepare.sh"和"写了但路径错了"
        # 混成同一种状态，而后者必须报错。
        plan = self._load("environment:\n  image: slime-v0.3.2-sglang0.5.15-torch2.11.0-flashattn2.8.3-cu129-py3.12\n").environment
        self.assertIsNone(plan.prepare)

    def test_unknown_image_is_a_schema_error(self) -> None:
        with self.assertRaises(SchemaError) as ctx:
            self._load("environment:\n  image: pytorch\n")
        message = str(ctx.exception)
        self.assertIn("environment.image must be one of", message)
        # 报错要说清楚怎么修：新环境不是改个字符串就有的。
        self.assertIn("docker/images/pytorch.Dockerfile", message)

    def test_legacy_schema_is_reported_not_silently_ignored(self) -> None:
        package = self._load("environment:\n  dockerfile: environment/Dockerfile\n")
        self.assertFalse(package.report.ok)
        paths = {issue.path for issue in package.report.issues}
        self.assertIn("metadata.environment.dockerfile", paths)
        self.assertIn("metadata.environment.image", paths)

    # --- validator branches -------------------------------------------------

    def _environment_issues(self, environment: dict) -> list:
        report = validate_metadata({"environment": environment})
        return [i.path for i in report.issues if i.path.startswith("metadata.environment")]

    def test_validator_accepts_every_image_under_docker_images(self) -> None:
        # 唯一事实来源是 docker/images/ 下的 Dockerfile——镜像就是照它构建的。
        names = known_images()
        self.assertTrue(names, "docker/images/ should exist in a source checkout")
        for name in sorted(names):
            self.assertEqual(self._environment_issues({"image": name}), [], name)

    def test_validator_rejects_an_image_without_a_dockerfile(self) -> None:
        self.assertEqual(
            self._environment_issues({"image": "pytorch"}),
            ["metadata.environment.image"],
        )

    def test_validator_requires_an_image(self) -> None:
        self.assertEqual(self._environment_issues({}), ["metadata.environment.image"])
        self.assertEqual(
            self._environment_issues({"image": ""}), ["metadata.environment.image"]
        )

    def test_validator_rejects_every_legacy_key(self) -> None:
        # 忽略的话，任务作者会以为它们还在起作用。
        for key in ("conda", "dockerfile", "requirements"):
            self.assertEqual(
                self._environment_issues({"image": "verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12", key: "x"}),
                [f"metadata.environment.{key}"],
                key,
            )

    def test_validator_rejects_a_non_path_prepare(self) -> None:
        self.assertEqual(
            self._environment_issues({"image": "verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12", "prepare": 3}),
            ["metadata.environment.prepare"],
        )


class EnvironmentEnvTests(unittest.TestCase):
    """metadata.environment.env：逐任务不同的变量（镜像只带同一环境共用的那批）。"""

    def _load(self, environment: str):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        return load_task_package(create_task(Path(temp.name), environment))

    def test_env_is_parsed_into_the_plan(self):
        package = self._load(
            "environment:\n"
            "  image: verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12\n"
            "  env:\n"
            '    CUDA_VISIBLE_DEVICES: "0,1"\n'
            '    NVIDIA_VISIBLE_DEVICES: "0,1"\n'
        )
        self.assertEqual(
            package.environment.env,
            {"CUDA_VISIBLE_DEVICES": "0,1", "NVIDIA_VISIBLE_DEVICES": "0,1"},
        )

    def test_absent_env_is_an_empty_mapping(self):
        self.assertEqual(self._load("environment:\n  image: verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12\n").environment.env, {})

    def test_tasks_cannot_hijack_the_harness_channel(self):
        # PTDBENCH_* 是 harness 告诉容器"你是哪个 stage"的通道。
        with self.assertRaises(SchemaError) as ctx:
            self._load(
                "environment:\n  image: verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12\n  env:\n    PTDBENCH_STAGE: verifier\n"
            )
        self.assertIn("PTDBENCH_STAGE", str(ctx.exception))

    def test_unquoted_number_is_rejected_rather_than_coerced(self):
        # `CUDA_VISIBLE_DEVICES: 0` 在 YAML 里是 int。悄悄 str() 的话，
        # "只用 0 号卡" 和 "写错了" 就再也分不出来。
        with self.assertRaises(SchemaError) as ctx:
            self._load(
                "environment:\n  image: verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12\n  env:\n    CUDA_VISIBLE_DEVICES: 0\n"
            )
        self.assertIn("CUDA_VISIBLE_DEVICES", str(ctx.exception))

    def test_scalar_env_is_rejected(self):
        with self.assertRaises(SchemaError):
            self._load("environment:\n  image: verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12\n  env: nope\n")


class EnvironmentEnvRuntimeTests(unittest.TestCase):
    def _request(self, **kwargs):
        from ptdbench.runtimes import SandboxRequest

        root = Path("/host/run")
        defaults = dict(
            stage="agent",
            task_id="t",
            host_task_root=root / "task",
            host_visible_task=root / "visible",
            host_workspace=root / "workspace",
            host_results=root / "results",
        )
        defaults.update(kwargs)
        return SandboxRequest(**defaults)

    def test_task_env_reaches_the_command_but_cannot_shadow_the_harness(self):
        from ptdbench.runtimes.base import CommandSpec
        from ptdbench.runtimes.providers.docker import DockerRuntime

        request = self._request(task_env={"CUDA_VISIBLE_DEVICES": "0,1"})
        merged = DockerRuntime()._merged_spec(request, CommandSpec(shell="true"))

        self.assertEqual(merged.env["CUDA_VISIBLE_DEVICES"], "0,1")
        self.assertEqual(merged.env["PTDBENCH_STAGE"], "agent")

    def test_caller_env_still_wins_over_the_task_declaration(self):
        from ptdbench.runtimes.base import CommandSpec
        from ptdbench.runtimes.providers.docker import DockerRuntime

        request = self._request(
            task_env={"CUDA_VISIBLE_DEVICES": "0,1"},
            env={"CUDA_VISIBLE_DEVICES": "2,3"},
        )
        merged = DockerRuntime()._merged_spec(request, CommandSpec(shell="true"))
        self.assertEqual(merged.env["CUDA_VISIBLE_DEVICES"], "2,3")


class EnvironmentImageTests(unittest.TestCase):
    def test_image_name_maps_to_a_fixed_tag_and_its_dockerfile(self):
        image = EnvironmentImage("verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12")
        self.assertEqual(image.tag, "ptdbench:verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12")
        self.assertEqual(image.dockerfile, IMAGES_DIR / "verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12.Dockerfile")
        self.assertEqual(image.context, IMAGES_DIR.parent)

    def test_every_dockerfile_declares_a_unique_tag(self):
        # tag 是按栈命名的，一个环境一个；两个环境撞 tag，后构建的会覆盖先构建的。
        tags = {name: EnvironmentImage(name).tag for name in sorted(known_images())}
        for name, tag in tags.items():
            self.assertRegex(tag, r"^ptdbench(:[a-z0-9.-]+|/[a-z0-9.-]+:[A-Za-z0-9._-]+)$", name)
            self.assertNotIn(":latest", tag, name)
        self.assertEqual(len(set(tags.values())), len(tags), tags)

    def test_missing_tag_header_is_an_error_not_a_default(self):
        with tempfile.TemporaryDirectory() as temp:
            dockerfile = Path(temp) / "x.Dockerfile"
            dockerfile.write_text("FROM scratch\n", encoding="utf-8")
            with mock.patch.object(
                EnvironmentImage, "dockerfile", new_callable=mock.PropertyMock,
                return_value=dockerfile,
            ):
                with self.assertRaises(RuntimeError):
                    EnvironmentImage("x").tag

    def test_every_base_image_is_pinned_by_digest(self):
        # tag 会被上游改指向；不钉 digest，同一个 Dockerfile 在两台机器上就可能
        # 构建出两套栈。
        for name in sorted(known_images()):
            text = EnvironmentImage(name).dockerfile.read_text(encoding="utf-8")
            froms = re.findall(r"^FROM\s+(\S+)", text, flags=re.MULTILINE)
            self.assertTrue(froms, name)
            for ref in froms:
                self.assertRegex(ref, r"@sha256:[0-9a-f]{64}$", name)

    def test_tooling_block_is_identical_in_every_dockerfile(self):
        # 每个镜像都靠同一段工具层（activate hook、fetch shim 等）和 harness 对接；
        # 哪份漂了，那个环境的任务就会在运行时才以奇怪的方式失败。
        marker = "# ---- PTDBench tooling: keep identical in every docker/images/*.Dockerfile ----"
        blocks = {}
        for name in sorted(known_images()):
            text = EnvironmentImage(name).dockerfile.read_text(encoding="utf-8")
            self.assertIn(marker, text, name)
            blocks[name] = text[text.index(marker):]
        self.assertEqual(len(set(blocks.values())), 1, sorted(blocks))


class EnsureEnvironmentTests(unittest.TestCase):
    def _run(self, *, exists: bool, rebuild: bool = False):
        from ptdbench.runtimes.providers.docker import DockerRuntime

        runtime = DockerRuntime()
        image = EnvironmentImage("verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12")
        with mock.patch.object(DockerRuntime, "_image_exists", return_value=exists), \
                mock.patch("ptdbench.runtimes.providers.docker.subprocess.run") as run:
            artifact = runtime.ensure_environment(image, rebuild=rebuild)
        self.assertEqual(artifact.reference, "ptdbench:verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12")
        return run

    def test_existing_image_is_reused_without_building(self):
        self.assertFalse(self._run(exists=True).called)

    def test_missing_image_is_built_from_its_dockerfile(self):
        run = self._run(exists=False)
        argv = run.call_args.args[0]
        self.assertEqual(argv[:2], ["docker", "build"])
        self.assertEqual(argv[argv.index("-f") + 1], str(IMAGES_DIR / "verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12.Dockerfile"))
        self.assertEqual(argv[argv.index("-t") + 1], "ptdbench:verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12")
        self.assertEqual(argv[-1], str(IMAGES_DIR.parent))
        self.assertTrue(run.call_args.kwargs["check"])

    def test_rebuild_builds_even_when_the_image_exists(self):
        self.assertTrue(self._run(exists=True, rebuild=True).called)

    def test_build_failure_propagates(self):
        from ptdbench.runtimes.providers.docker import DockerRuntime

        with mock.patch.object(DockerRuntime, "_image_exists", return_value=False), \
                mock.patch(
                    "ptdbench.runtimes.providers.docker.subprocess.run",
                    side_effect=subprocess.CalledProcessError(1, ["docker", "build"]),
                ):
            with self.assertRaises(subprocess.CalledProcessError):
                DockerRuntime().ensure_environment(EnvironmentImage("verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12"))


class BuildImagesCommandTests(unittest.TestCase):
    def _main(self, *argv):
        from ptdbench import cli
        from ptdbench.runtimes.providers.docker import DockerRuntime

        with mock.patch.object(DockerRuntime, "ensure_environment") as ensure, \
                mock.patch("builtins.print"):
            code = cli.main(["build-images", *argv])
        return code, [(c.args[0].name, c.kwargs["rebuild"]) for c in ensure.call_args_list]

    def test_default_is_every_image(self):
        code, calls = self._main()
        self.assertEqual(code, 0)
        self.assertEqual(calls, [(name, False) for name in sorted(known_images())])

    def test_names_and_rebuild_are_passed_through(self):
        self.assertEqual(self._main("verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12", "--rebuild"), (0, [("verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12", True)]))

    def test_unknown_name_is_rejected_before_building(self):
        with self.assertRaises(SystemExit) as ctx:
            self._main("verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12", "pytorch")
        self.assertIn("pytorch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
