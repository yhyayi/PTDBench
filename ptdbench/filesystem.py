from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence


def pick_existing(root: Path, candidates: Sequence[str]) -> Optional[Path]:
    # 接受少量常见文件名，让模板灵活一点，不强迫贡献者只用一个精确名字。
    for rel in candidates:
        path = root / rel
        if path.exists():
            return path
    return None


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_tree(src: Path, dst: Path, ignore: Optional[Iterable[str]] = None) -> Path:
    # 每次都重建目标目录，保证 run 从干净、确定的 workspace 副本开始。
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*(ignore or [])))
    return dst


def safe_relpath(path: Path, root: Path) -> str:
    rel = path.resolve().relative_to(root.resolve())
    return str(rel)


def iter_files(root: Path, ignore_dirs: Optional[Iterable[str]] = None) -> Iterator[Path]:
    ignore = set(ignore_dirs or [])
    for dirpath, dirnames, filenames in os.walk(root):
        # 调用方要求时跳过易变或很大的生成目录。
        dirnames[:] = [d for d in dirnames if d not in ignore]
        base = Path(dirpath)
        for fn in filenames:
            yield base / fn


def tree_digest(root: Path, ignore_dirs: Optional[Iterable[str]] = None) -> str:
    # 同时哈希路径和内容；文件移动或内容修改都会改变 digest，
    # 可用于缓存键和变更检测。
    h = hashlib.sha256()
    for path in sorted(iter_files(root, ignore_dirs=ignore_dirs)):
        rel = path.resolve().relative_to(root.resolve())
        h.update(str(rel).encode("utf-8"))
        h.update(b"\0")
        h.update(str(path.stat().st_size).encode("utf-8"))
        h.update(b"\0")
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        h.update(b"\0")
    return h.hexdigest()


def write_text(path: Path, text: str) -> Path:
    # 写文件前确保父目录存在，减少调用方样板代码。
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path
