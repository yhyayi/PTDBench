"""共用的小工具：写 parquet + 打印可以直接抄进 prepare.sh 的校验参数。"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd


def hf_cache_root() -> Path:
    """HF 系库的缓存根，宿主机上就是 datasets/hf。

    布局刻意和 `HF_HOME=$PTDBENCH_DATASETS/hf` 会产出的一样（hub 在 hf/hub，
    datasets 在 hf/datasets），所以和 ptdbench_fetch.hf_cache_dir() 看的是同一棵树。

    **必须在 import huggingface_hub / datasets 之前调用。** 那两个库都在 import 时
    就把 HF_HOME 读进模块级常量（constants.HF_HUB_CACHE），之后再改 os.environ
    是个静默的 no-op——ptdbench_fetch.py 原来就是这么把 954MB 权重下到容器可写层
    里的。配方里的 import 都是函数内延迟 import，就是为了让这个调用能排在前面；
    谁要把 import 提到模块顶层，这个保证就没了。
    """
    root = Path(os.environ.get("PTDBENCH_DATASETS", "/ptdbench/datasets")) / "hf"
    root.mkdir(parents=True, exist_ok=True)
    # 除了下面显式传的 cache_dir，还有些路径（元数据、etag、load_dataset 内部
    # 调的 hf_hub_download）只认环境变量，所以两个都设。
    os.environ.setdefault("HF_HOME", str(root))
    return root


def hf_datasets_cache() -> Path:
    """`load_dataset(cache_dir=...)` 该传的那个目录。"""
    return hf_cache_root() / "datasets"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_split(rows: list[dict], out_dir: Path, name: str) -> Path:
    """写一个 split，返回路径。

    显式走 pandas -> parquet，不用 datasets 自己的 to_parquet：后者的分片和压缩
    参数在不同版本间会变，而我们要的是"同样的输入产出同样的字节"。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def report(paths: list[Path]) -> None:
    """打印可以直接贴进 prepare.sh 的校验参数，省得手抄。

    `ptdbench_dataset --expect/--expect-size` 和 `ptdbench_fetch_hf` 的同名参数是
    一套语义：只校验任务显式声明过的文件，对不上就硬失败。配方本身已经断言了行数和
    schema，所以这两个参数是**可选**的第二道闸——它挡的是"缓存里那份被动过"，
    而行数断言挡不住这个。
    """
    print()
    print("=" * 72)
    print("把下面这段接在对应任务 environment/prepare.sh 的 ptdbench_dataset 调用后面：")
    print("=" * 72)
    for p in sorted(paths):
        print(f"    --expect {p.name}={sha256_of(p)} \\")
        print(f"    --expect-size {p.name}={p.stat().st_size} \\")

