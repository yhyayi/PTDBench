"""run_qwen2.5_3B.sh 的 retrieval 段落：起服务、等就绪、起不来就报错。

这段是两个 search-r1 任务里唯一"有控制流"的 shell：探端口、后台拉起、轮询、两条
失败分支。它的每条分支在真实运行里都要几分钟到几十分钟才看得见一次，而写错的代价
是一次 8 卡重跑白跑——所以在这里用一个假的 start_retriver.sh 把四条路径都走一遍。

段落本身只依赖 SLIME_DIR / DATA_DIR / MODEL_DIR 和几个 RETRIEVAL_* 变量，所以可以
从脚本里按注释边界切出来单独跑，不用 GPU、不用 conda、也不用碰上面那一百行
ray/megatron 的准备工作。切不出来（有人改了边界注释）本身就是一条失败。
"""

from __future__ import annotations

import os
import signal
import shutil
import socket
import subprocess
import tempfile
import textwrap
import time
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
RECIPE = "workspace/examples/search-r1/run_qwen2.5_3B.sh"
TASKS = ("tasks/slime_search_r1", "tasks/slime_search_r1_nomask")

BLOCK_START = "# --- the local dense retrieval server"
BLOCK_END = "ROLLOUT_ARGS=("


def extract_block(script: Path) -> str:
    """切出 retrieval 段落：从它自己的标题注释到下一个 *_ARGS 数组之前。"""
    lines = script.read_text().splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith(BLOCK_START)]
    ends = [i for i, line in enumerate(lines) if line.startswith(BLOCK_END)]
    if len(starts) != 1:
        raise AssertionError(f"{script}: expected exactly one {BLOCK_START!r}, got {len(starts)}")
    if not ends or ends[0] < starts[0]:
        raise AssertionError(f"{script}: no {BLOCK_END!r} after the retrieval block")
    return "\n".join(lines[starts[0] : ends[0]])


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def a_retriever_is_already_alive() -> bool:
    """段落的 retrieval_alive 谓词在这台机器上此刻是不是已经成立。

    段落判"服务还活着"用的是 `pgrep -f`，一个**全局**的、进程名级别的判断——它没法
    只看自己起的那个进程，因为它要认的恰恰是上一次重跑留下来的那个。生产里这是对
    的：一个容器一个检索服务。但这意味着"服务死了"这条分支在一台真的跑着检索服务的
    机器上（比如 g33，或者刚跑过手工 probe 的开发机）根本无法被观测——不是段落错
    了，是那个前提不成立。那种情况下跳过，比报一个假的失败有用。
    """
    for pattern in ("local_dense_retriever/retrieval_server.py", "search-r1/start_retriver.sh"):
        if subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0:
            return True
    return False


class RetrievalGuardTest(unittest.TestCase):
    """把段落放进一个只有 stub 的假 SLIME_DIR 里跑。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.block = extract_block(REPO_ROOT / TASKS[1] / RECIPE)

    def setUp(self) -> None:
        self.sandboxes: list[Path] = []

    def tearDown(self) -> None:
        """每个 case 结束时把自己的 stub 收掉。

        段落用 setsid 起服务，正是为了让它活过这个 shell——重跑时复用一个已经载好
        60GiB 索引的服务，是这整段的目的之一。代价是 stub 也会活过 case，而段落判
        "还活着"用的是 `pgrep -f .../start_retriver.sh`，一个进程名级别的、全局的
        判断（和脚本开头 kill_python_except_retriever 用的是同一个口径）。上一个
        case 漏下来的 stub 会让下一个 case 以为服务还在，然后一路等到超时。

        生产里这不是问题：一个容器一个检索服务。测试里必须自己扫干净。

        按**进程组**杀，不是按命令行匹配杀：stub 是 `cd $SLIME_DIR` 之后用相对路径
        调起来的，它和它的子进程（`sleep`、`python3 -c ...`）的命令行里都没有沙箱
        路径，`pkill -f <sandbox>` 一个也匹配不到。setsid 给了这棵树一个独立的
        session，组长的 pid 就是 pgid，所以 killpg 一次能收干净——这也正是段落用
        setsid 的原因（见 recipe 里那段注释）。
        """
        for sandbox in self.sandboxes:
            pidfile = sandbox / "stub.pid"
            try:
                pgid = int(pidfile.read_text().strip())
            except (OSError, ValueError):
                continue        # stub 没跑起来（比如复用那个 case），没什么要收的
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        for sandbox in self.sandboxes:
            shutil.rmtree(sandbox, ignore_errors=True)

    def run_block(self, stub: str, *, port: int, timeout_s: int = 60) -> subprocess.CompletedProcess:
        """stub 是假的 start_retriver.sh 的 body。返回段落自己的退出状态。"""
        tmp = Path(tempfile.mkdtemp(prefix="ptdbench-retrieval-guard-"))
        self.sandboxes.append(tmp)
        recipe_dir = tmp / "examples" / "search-r1"
        recipe_dir.mkdir(parents=True)
        # 真的那份没有 shebang，段落靠 `bash <file>` 调它；stub 保持一样的形状。
        # 开头那行是给 tearDown 用的：setsid 之后 $$ 就是新 session 组长，也就是
        # 这棵进程树的 pgid。
        (recipe_dir / "start_retriver.sh").write_text(f'echo $$ > "{tmp}/stub.pid"\n' + stub)

        script = tmp / "block.sh"
        script.write_text(
            textwrap.dedent(
                f"""\
                set -e
                SLIME_DIR={tmp}
                DATA_DIR="$SLIME_DIR"
                MODEL_DIR="$SLIME_DIR"
                RETRIEVAL_PORT={port}
                RETRIEVAL_STARTUP_TIMEOUT={timeout_s}
                """
            )
            + self.block
        )
        return subprocess.run(
            ["bash", str(script)],
            capture_output=True,
            text=True,
            timeout=timeout_s + 120,
        )

    def test_reuses_a_server_that_is_already_listening(self) -> None:
        """端口上有人监听就直接复用：60GiB 索引不该为了一次重跑再读一遍。"""
        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            port = server.getsockname()[1]
            # stub 故意是一条失败的命令：走到它就说明没有复用。
            result = self.run_block("exit 17\n", port=port)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("reusing the server already listening", result.stdout)
        self.assertIn(f"server up on :{port}", result.stdout)

    def test_fails_fast_when_the_server_dies(self) -> None:
        """服务起不来就立刻报错，而不是等满整个超时——等满是最贵的失败方式。

        这是唯一一条依赖"这台机器上没有别的检索服务"的 case：它要看的就是
        retrieval_alive 变成假。这个前提不成立时跳过，理由见
        a_retriever_is_already_alive。
        """
        if a_retriever_is_already_alive():
            self.skipTest("a retriever is already running here; the death branch is unobservable")
        port = free_port()
        result = self.run_block("echo 'stub: boom' >&2\nexit 1\n", port=port, timeout_s=60)
        self.assertEqual(result.returncode, 1)
        self.assertIn("exited before it started listening", result.stderr)
        # 服务自己的日志要被带出来：不带的话报错等于没说原因。
        self.assertIn("stub: boom", result.stderr)

    def test_times_out_when_the_server_never_listens(self) -> None:
        """活着但一直不监听，也必须有个尽头。"""
        port = free_port()
        result = self.run_block("sleep 600\n", port=port, timeout_s=5)
        self.assertEqual(result.returncode, 1)
        self.assertIn("still no listener", result.stderr)

    def test_starts_the_server_and_waits_for_the_port(self) -> None:
        """端口空着就自己拉起来，并且等到真的开始监听才往下走。"""
        port = free_port()
        stub = textwrap.dedent(
            f"""\
            python3 -c "
            import socket, time
            time.sleep(2)
            s = socket.socket()
            s.bind(('127.0.0.1', {port}))
            s.listen(1)
            time.sleep(30)
            "
            """
        )
        result = self.run_block(stub, port=port)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("starting the local dense retriever", result.stdout)
        self.assertIn(f"server up on :{port}", result.stdout)

    def test_the_started_server_does_not_hold_the_callers_stdout(self) -> None:
        """活着的检索服务不能攥着调用方的 stdout 管道。

        这是这段最容易写错、也最难看出来的地方。服务是**故意**要活过这个脚本的——
        下一次重跑复用它，省掉几分钟的索引载入。但如果后台那一层还持有脚本自己的
        stdout，管道的写端就一直不关：读端（手工调试时的
        `| tee .../train.log`，agent 阶段则是 harness 自己的 capture）永远等不到
        EOF，一次跑完的训练看起来就是卡住了。

        写法上的区别只是重定向挂在子 shell 上还是挂在里面那条 setsid 上，两种都能
        跑、都能把日志写对，只有这个测试能分开它们。

        所以这里要的是真实的复用情形：服务起来了、在监听、而且远远活过段落本身。
        段落一退出，capture 就必须读到 EOF。
        """
        port = free_port()
        stub = textwrap.dedent(
            f"""\
            python3 -c "
            import socket, time
            s = socket.socket()
            s.bind(('127.0.0.1', {port}))
            s.listen(1)
            time.sleep(600)
            "
            """
        )
        started = time.monotonic()
        # timeout_s 只是段落自己的就绪超时；capture 挂住的话，run_block 会在
        # timeout_s + 120 秒后抛 TimeoutExpired，那就是这个测试要抓的失败。
        result = self.run_block(stub, port=port, timeout_s=60)
        elapsed = time.monotonic() - started

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"server up on :{port}", result.stdout)
        # stub 还要再活 600 秒。段落在服务就绪后立刻返回，所以这里应该是个位数秒。
        self.assertLess(elapsed, 60, "capture did not see EOF until the stub exited")


class RetrievalGuardParityTest(unittest.TestCase):
    """两个任务跑同一个 recipe，这段必须一字不差。

    slime_search_r1 是 slime_search_r1_nomask 的健康对照：容器环境或启动方式上的任何
    差别都会让两边在同一个 horizon 上的 eval/nq_test 不可比，而这两个数就是
    nomask 这个任务的全部结论。
    """

    def test_retrieval_block_is_identical(self) -> None:
        blocks = {task: extract_block(REPO_ROOT / task / RECIPE) for task in TASKS}
        self.assertEqual(*blocks.values())

    def test_launch_scripts_are_identical(self) -> None:
        """一字不差，连 checkpoint 存哪、wandb group 叫什么都一样。

        这两处以前分别是 Qwen2.5-3B_slime_nomask 和 search-r1_qwen2.5-3B-nomask，
        在 agent 能读到的启动脚本里把任务名写成了「nomask」——缺陷的名字。注入的
        差异只能待在 generate_with_search.py 的 loss_mask 那一行；启动脚本里任何
        带 nomask 的字样都是在白送答案。两个任务跑在各自的容器和 workspace 里，
        名字相同不会撞车。
        """
        left, right = (
            (REPO_ROOT / task / RECIPE).read_text() for task in TASKS
        )
        self.assertEqual(left, right)

    def test_prepare_scripts_differ_only_in_their_own_names(self) -> None:
        """prepare.sh 的 body 是逐字节一致的，只有头两行注释提到各自的任务名。"""
        left, right = (
            (REPO_ROOT / task / "environment/prepare.sh").read_text().splitlines()
            for task in TASKS
        )
        self.assertEqual(len(left), len(right))
        differing = [i for i, (a, b) in enumerate(zip(left, right)) if a != b]
        # 0-based 的 1 和 3，也就是文件里的第 2 行和第 4 行。
        self.assertEqual(differing, [1, 3])

    def test_the_start_script_the_block_names_exists(self) -> None:
        """文件名是 start_retriver.sh，少一个 'e'。拼成对的那个是不存在的。"""
        for task in TASKS:
            recipe = REPO_ROOT / task / RECIPE
            self.assertIn("start_retriver.sh", recipe.read_text())
            self.assertNotIn("start_retriever.sh", recipe.read_text())
            self.assertTrue(
                (REPO_ROOT / task / "workspace/examples/search-r1/start_retriver.sh").is_file()
            )


if __name__ == "__main__":
    unittest.main()
