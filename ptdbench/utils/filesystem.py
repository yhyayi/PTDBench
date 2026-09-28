from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence


def pick_existing(root: Path, candidates: Sequence[str]) -> Optional[Path]:
    """从候选文件名中返回第一个存在的文件。"""
    for relative_path in candidates:
        path = root / relative_path
        if path.exists():
            return path
    return None


def ensure_dir(path: Path) -> Path:
    """创建目录并返回原路径，方便在表达式中使用。"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_tree(src: Path, dst: Path, ignore: Optional[Iterable[str]] = None) -> Path:
    """复制目录树；如果目标存在，先清理目标以保证结果确定。

    以 symlinks=True 复制，保留符号链接原样（包括悬空链接），与
    HarborTaskMaterializer._link_tree 的行为一致。
    """
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True, ignore=shutil.ignore_patterns(*(ignore or [])))
    return dst


def iter_files(root: Path, ignore_dirs: Optional[Iterable[str]] = None) -> Iterator[Path]:
    """遍历目录中的文件，可选跳过生成目录。"""
    ignored = set(ignore_dirs or [])
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in ignored]
        for filename in filenames:
            yield Path(dirpath) / filename


def tree_digest(root: Path, ignore_dirs: Optional[Iterable[str]] = None) -> str:
    """同时哈希相对路径和文件内容，用于变更检测与缓存键。"""
    digest = hashlib.sha256()
    for path in sorted(iter_files(root, ignore_dirs=ignore_dirs)):
        relative_path = path.resolve().relative_to(root.resolve())
        digest.update(str(relative_path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("utf-8"))
        digest.update(b"\0")
        with open(path, "rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def write_text(path: Path, text: str) -> Path:
    """写入 UTF-8 文本，并自动创建父目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        file.write(text)
    return path
