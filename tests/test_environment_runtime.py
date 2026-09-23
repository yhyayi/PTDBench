"""统一镜像的运行时接线：三个阶段各挂什么、各拿到哪些环境变量。

这个文件盯的是访问控制，不是功能。三张挂载表的差别（谁可写、谁看得到 cache、
谁看得到 oracle）就是"agent 不能给自己判分、verifier 不能读到被改过的数据"
这条性质的全部实现，所以每一条都值得有一个会失败的测试盯着。
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ptdbench.core.types import (
    DATASETS_DIR,
    MODELS_DIR,
    PREPARED_DIR,
    PTDBENCH_ENVIRONMENT_DIR,
    EnvironmentPlan,
)
from ptdbench.runtimes import SandboxRequest
from ptdbench.runtimes.providers.docker import DockerRuntime


ROOT = Path("/host/run")


def make_request(stage: str = "agent", **kwargs) -> SandboxRequest:
    """一个"什么都挂上了"的请求；单条测试再按需去掉某个目录。"""
    defaults = dict(
        stage=stage,
        task_id="task_001",
        host_task_root=ROOT / "task",
        host_visible_task=ROOT / "visible",
        host_workspace=ROOT / "workspace",
        host_results=ROOT / "results",
        host_environment=ROOT / "environment",
        host_prepared=ROOT / "prepared",
        host_models=ROOT / "models",
        host_datasets=ROOT / "datasets",
    )
    defaults.update(kwargs)
    return SandboxRequest(**defaults)


class MountTableTests(unittest.TestCase):
    def _mounts(self, request) -> dict:
        return {
            mount.container_path: mount
            for mount in DockerRuntime()._mounts(request)
        }

    # --- workspace：谁可写 ---------------------------------------------------

    def test_prepare_can_write_the_workspace(self) -> None:
        # 权重就下在任务脚本引用的那个相对路径上（./Qwen2.5-0.5B-Instruct），
        # workspace 只读的话 prepare.sh 第一条命令就失败。
        self.assertFalse(self._mounts(make_request("prepare"))["/workspace"].read_only)

    def test_agent_can_write_the_workspace(self) -> None:
        self.assertFalse(self._mounts(make_request("agent"))["/workspace"].read_only)

    def test_verifier_gets_a_read_only_workspace(self) -> None:
        # 判分不能改现场。
        self.assertTrue(self._mounts(make_request("verifier"))["/workspace"].read_only)

    # --- /task：别剧透 -------------------------------------------------------

    def test_prepare_and_agent_only_see_the_visible_task(self) -> None:
        # prepare 跑的是任务自带的脚本，没有理由让它读到 oracle 或判分脚本。
        for stage in ("prepare", "agent"):
            mount = self._mounts(make_request(stage))["/task"]
            self.assertEqual(mount.host_path, ROOT / "visible", stage)
            self.assertTrue(mount.read_only, stage)

    def test_verifier_sees_the_whole_evaluation_root(self) -> None:
        mount = self._mounts(make_request("verifier"))["/task"]
        self.assertEqual(mount.host_path, ROOT / "task")
        self.assertTrue(mount.read_only)

    # --- prepared：prepare 写，后两个阶段只读 --------------------------------

    def test_prepare_owns_the_prepared_directory(self) -> None:
        self.assertFalse(self._mounts(make_request("prepare"))[PREPARED_DIR].read_only)

    def test_agent_and_verifier_get_prepared_read_only(self) -> None:
        # verifier 要读 prepare 交付的模型/数据集，但不能读到一份 agent 改过的，
        # 否则判分依据是可以被应试者编辑的。
        for stage in ("agent", "verifier"):
            self.assertTrue(self._mounts(make_request(stage))[PREPARED_DIR].read_only, stage)

    def test_prepared_is_absent_when_the_run_has_none(self) -> None:
        self.assertNotIn(PREPARED_DIR, self._mounts(make_request("agent", host_prepared=None)))

    # --- models / datasets：prepare 可写，后两个阶段只读 ---------------------

    def test_prepare_can_write_the_download_cache(self) -> None:
        # 下载发生在 prepare，写不了就等于没有缓存。
        mounts = self._mounts(make_request("prepare"))
        for container_path in (MODELS_DIR, DATASETS_DIR):
            self.assertIn(container_path, mounts)
            self.assertFalse(mounts[container_path].read_only, container_path)

    def test_later_stages_see_the_download_cache_read_only(self) -> None:
        # 看得到是为了排查时对得上账；只读是因为判分依据不能是 agent 改得动的东西。
        for stage in ("agent", "verifier"):
            mounts = self._mounts(make_request(stage))
            for container_path in (MODELS_DIR, DATASETS_DIR):
                self.assertIn(container_path, mounts, (stage, container_path))
                self.assertTrue(mounts[container_path].read_only, (stage, container_path))

    def test_stages_without_a_download_cache_still_run(self) -> None:
        # 没有共享缓存只是"这次的下载不会被下一次复用"，不是错误。
        for stage in ("prepare", "agent", "verifier"):
            mounts = self._mounts(make_request(stage, host_models=None, host_datasets=None))
            self.assertNotIn(MODELS_DIR, mounts, stage)
            self.assertNotIn(DATASETS_DIR, mounts, stage)

    # --- environment：三个阶段都只读 ----------------------------------------

    def test_environment_is_read_only_in_every_stage(self) -> None:
        for stage in ("prepare", "agent", "verifier"):
            mount = self._mounts(make_request(stage))[PTDBENCH_ENVIRONMENT_DIR]
            self.assertEqual(mount.host_path, ROOT / "environment", stage)
            self.assertTrue(mount.read_only, stage)

    def test_legacy_task_mounts_no_environment(self) -> None:
        # raw 任务的 environment/ 里只有一个 Dockerfile，构建时已经用过了。
        request = make_request("agent", host_environment=None)
        self.assertNotIn(PTDBENCH_ENVIRONMENT_DIR, self._mounts(request))

    # --- results ------------------------------------------------------------

    def test_prepare_has_no_results_mount(self) -> None:
        # prepare 不产出给判分看的东西；它的日志由 harness 在宿主机侧落盘。
        self.assertNotIn("/results", self._mounts(make_request("prepare")))

    def test_no_mount_is_declared_twice(self) -> None:
        for stage in ("prepare", "agent", "verifier"):
            mounts = DockerRuntime()._mounts(make_request(stage))
            targets = [mount.container_path for mount in mounts]
            self.assertEqual(len(targets), len(set(targets)), f"{stage}: {targets}")


class GpuWiringTests(unittest.TestCase):
    """任务卡片声明的显卡怎么变成 `docker create` 的 flag。

    这条是"任务能不能真的跑训练"的全部实现。宿主机的 default runtime 是 runc，
    把 NVIDIA_VISIBLE_DEVICES `-e` 进容器不会让容器看见任何设备——只有 --gpus
    会触发 nvidia 的 hook。镜像侧早就是按这个写的（ptdbench-activate.sh 里
    "that is where --gpus puts libcuda.so.1"），这里补上缺的另一半。
    """

    def _args(self, **kwargs) -> list:
        return DockerRuntime._gpu_args(make_request("agent", **kwargs))

    def test_a_device_list_becomes_a_device_spec(self) -> None:
        # 任务卡片里写的是裸设备表（rlve_hyper/task_001 是 "0,1,2,3"），
        # docker 只认 device= 前缀。
        args = self._args(task_env={"NVIDIA_VISIBLE_DEVICES": "0,1,2,3"})
        self.assertEqual(args[:2], ["--gpus", "device=0,1,2,3"])

    def test_all_is_passed_through(self) -> None:
        self.assertEqual(self._args(task_env={"NVIDIA_VISIBLE_DEVICES": "all"})[:2],
                         ["--gpus", "all"])

    def test_gpu_stages_get_a_usable_shm(self) -> None:
        # docker 默认的 64MB /dev/shm 会让 ray/sglang 在第一次 rollout 爆掉，
        # 而报出来的是一个看不懂的 bus error。
        args = self._args(task_env={"NVIDIA_VISIBLE_DEVICES": "0"})
        self.assertIn("--shm-size=16g", args)

    def test_a_cpu_only_task_is_untouched(self) -> None:
        # 接线前后逐字节相同——纯 CPU 的任务和判分阶段没有理由多占内存。
        self.assertEqual(self._args(), [])

    def test_explicitly_asking_for_no_gpu_is_honoured(self) -> None:
        # `none` / `void` 是 NVIDIA_VISIBLE_DEVICES 自己的"不要 GPU"取值。
        for value in ("none", "void", ""):
            self.assertEqual(self._args(task_env={"NVIDIA_VISIBLE_DEVICES": value}), [], value)

    def test_the_host_can_override_the_task_card(self) -> None:
        # 一台只有 2 张卡的机器不该因为卡片写着 8 张就跑不起来。
        with mock.patch.dict(os.environ, {"PTDBENCH_DOCKER_GPUS": "device=0,1"}):
            args = self._args(task_env={"NVIDIA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"})
        self.assertEqual(args[:2], ["--gpus", "device=0,1"])

    def test_the_host_can_override_the_shm_size(self) -> None:
        with mock.patch.dict(os.environ, {"PTDBENCH_DOCKER_SHM_SIZE": "64g"}):
            args = self._args(task_env={"NVIDIA_VISIBLE_DEVICES": "0"})
        self.assertIn("--shm-size=64g", args)

    def test_every_stage_can_reach_the_gpu_it_declares(self) -> None:
        # prepare 要 GPU 把 HF 权重转成 torch_dist，agent 要跑训练。verifier 也拿得到
        # 是因为 task_env 对三个阶段一视同仁——它不声明就自然没有。
        for stage in ("prepare", "agent", "verifier"):
            request = make_request(stage, task_env={"NVIDIA_VISIBLE_DEVICES": "0"})
            self.assertIn("--gpus", DockerRuntime._gpu_args(request), stage)


class StageRequestTests(unittest.TestCase):
    def test_prepare_without_a_landing_zone_is_rejected(self) -> None:
        # 没有落点的 prepare 会"成功"却什么也没留下，直到 agent 找不到模型才暴露，
        # 而那时报的是任务的错。
        with self.assertRaises(ValueError) as ctx:
            SandboxRequest(
                stage="prepare",
                task_id="t",
                host_task_root=ROOT,
                host_visible_task=ROOT,
                host_workspace=ROOT,
                host_results=ROOT,
            )
        self.assertIn("host_prepared", str(ctx.exception))

    def test_unknown_stage_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            make_request("provision")


class StandardEnvTests(unittest.TestCase):
    def _env(self, stage: str = "agent", **kwargs) -> dict:
        return DockerRuntime().standard_env(make_request(stage, **kwargs))

    def test_conda_env_is_exported(self) -> None:
        # 用 slime-v0.3.1 而不是随便一个名字：注册表里唯一带点和连字符的环境名，
        # 任何把它当版本号拆开的代码都会在这里露出来。
        self.assertEqual(
            self._env(conda_env="slime-v0.3.1")["PTDBENCH_CONDA_ENV"], "slime-v0.3.1"
        )

    def test_raw_task_exports_an_empty_conda_env(self) -> None:
        # activate hook 靠空字符串判断"这个任务不激活任何环境"。
        self.assertEqual(self._env()["PTDBENCH_CONDA_ENV"], "")

    def test_every_stage_gets_the_environment_name(self) -> None:
        # 导出不等于激活：activate 脚本自己按 PTDBENCH_STAGE 决定。
        for stage in ("prepare", "agent", "verifier"):
            env = self._env(stage, conda_env="verl")
            self.assertEqual(env["PTDBENCH_CONDA_ENV"], "verl", stage)
            self.assertEqual(env["PTDBENCH_STAGE"], stage, stage)

    def test_prepare_and_agent_never_learn_where_the_oracle_is(self) -> None:
        for stage in ("prepare", "agent"):
            env = self._env(stage)
            self.assertEqual(env["PTDBENCH_ORACLE"], "", stage)
            self.assertEqual(env["PTDBENCH_VERIFIER"], "", stage)
            self.assertEqual(env["PTDBENCH_TASK_ROOT"], "/task", stage)

    def test_verifier_gets_the_grading_paths(self) -> None:
        env = self._env("verifier")
        self.assertEqual(env["PTDBENCH_ORACLE"], "/task/oracle")
        self.assertEqual(env["PTDBENCH_VERIFIER"], "/task/verifier")
        # visible_task 对 verifier 没有意义：它看得到整个 /task。
        self.assertEqual(env["PTDBENCH_VISIBLE_TASK"], "")

    def test_paths_match_what_is_actually_mounted(self) -> None:
        env = self._env("prepare")
        self.assertEqual(env["PTDBENCH_ENVIRONMENT"], PTDBENCH_ENVIRONMENT_DIR)
        self.assertEqual(env["PTDBENCH_PREPARED"], PREPARED_DIR)
        self.assertEqual(env["PTDBENCH_MODELS"], MODELS_DIR)
        self.assertEqual(env["PTDBENCH_DATASETS"], DATASETS_DIR)

    def test_the_download_cache_is_exported_to_every_stage(self) -> None:
        # 三个阶段都挂得到，所以三个阶段都该拿到真实路径而不是空串。
        for stage in ("prepare", "agent", "verifier"):
            env = self._env(stage)
            self.assertEqual(env["PTDBENCH_MODELS"], MODELS_DIR, stage)
            self.assertEqual(env["PTDBENCH_DATASETS"], DATASETS_DIR, stage)

    def test_unmounted_directories_are_the_empty_string(self) -> None:
        # "你看不到"，而不是"路径在那儿但目录是空的"——后者会让 prepare.sh 把模型
        # 下到一个跑完就消失的地方。
        env = self._env("agent", host_models=None, host_datasets=None)
        self.assertEqual(env["PTDBENCH_MODELS"], "")
        self.assertEqual(env["PTDBENCH_DATASETS"], "")
        env = self._env("agent", host_environment=None, host_prepared=None)
        self.assertEqual(env["PTDBENCH_ENVIRONMENT"], "")
        self.assertEqual(env["PTDBENCH_PREPARED"], "")


class NetworkPolicyEnvTests(unittest.TestCase):
    """离线开关按阶段给，不是烤在镜像的 ENV 里。"""

    def test_prepare_is_allowed_to_reach_the_network(self) -> None:
        # 镜像级 ENV 对三个阶段一视同仁，等于让 prepare 自己关掉自己要用的网络。
        env = DockerRuntime().standard_env(make_request("prepare"))
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            self.assertNotIn(key, env)

    def test_prepare_still_does_not_report_to_wandb(self) -> None:
        # prepare 只是在下载，没有什么可上报的。
        env = DockerRuntime().standard_env(make_request("prepare"))
        self.assertEqual(env["WANDB_MODE"], "offline")

    def test_agent_and_verifier_are_pinned_offline(self) -> None:
        # 一次 silent 的联网下载可能把评测跑在一份没人校验过的权重上，还看不出来。
        for stage in ("agent", "verifier"):
            env = DockerRuntime().standard_env(make_request(stage))
            self.assertEqual(env["HF_HUB_OFFLINE"], "1", stage)
            self.assertEqual(env["TRANSFORMERS_OFFLINE"], "1", stage)
            self.assertEqual(env["HF_DATASETS_OFFLINE"], "1", stage)
            self.assertEqual(env["WANDB_MODE"], "offline", stage)
            # 有些框架在 offline 下仍然会因为没有 key 而直接退出。
            self.assertTrue(env["WANDB_API_KEY"])


class PrepareScriptTests(unittest.TestCase):
    """evaluator 怎么把 prepare.sh 翻译成容器里的一条路径。"""

    def _package(self, root: Path, *, prepare: Path | None, conda: str | None = "verl"):
        from ptdbench.core.types import TaskPaths

        paths = TaskPaths(
            root=root,
            readme_md=None,
            metadata_yaml=root / "metadata.yaml",
            workspace=root / "workspace",
            environment=root / "environment",
            evaluations={},
        )
        from ptdbench.core.types import TaskPackage
        from ptdbench.core.types import ValidationReport

        return TaskPackage(
            paths=paths,
            metadata={},
            report=ValidationReport(),
            environment=EnvironmentPlan(conda=conda, prepare=prepare),
        )

    def test_script_resolves_under_the_mounted_environment(self) -> None:
        from ptdbench.evaluation.evaluator import TaskEvaluator

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "environment").mkdir()
            script = root / "environment" / "prepare.sh"
            script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            package = self._package(root, prepare=script)
            self.assertEqual(
                TaskEvaluator("local")._prepare_script(package),
                f"{PTDBENCH_ENVIRONMENT_DIR}/prepare.sh",
            )

    def test_no_prepare_script_means_no_prepare_stage(self) -> None:
        from ptdbench.evaluation.evaluator import TaskEvaluator

        with tempfile.TemporaryDirectory() as temp:
            package = self._package(Path(temp), prepare=None)
            self.assertIsNone(TaskEvaluator("local")._prepare_script(package))

    def test_script_outside_the_environment_directory_is_rejected(self) -> None:
        # 只有 environment/ 挂进容器；声明一个在外面的路径会在容器里变成
        # "文件不存在"，报错点离原因很远。
        from ptdbench.evaluation.evaluator import TaskEvaluator
        from ptdbench.tasks import SchemaError

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "environment").mkdir()
            outside = root / "prepare.sh"
            outside.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            package = self._package(root, prepare=outside)
            with self.assertRaises(SchemaError) as ctx:
                TaskEvaluator("local")._prepare_script(package)
            self.assertIn("must stay under", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
