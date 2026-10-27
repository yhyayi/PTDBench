"""Utilities for discovering and executing task scripts across platforms.

Supports two script formats with a priority-based fallback:
  .sh  → direct execution (callers run chmod +x as root separately)
  .bat → Windows batch file (cmd /c)

Discovery can be filtered by the target OS declared in ``task.toml``'s
``[environment].os`` field — Linux tasks see only ``.sh``, Windows tasks see
only ``.bat``.  When no OS is provided the legacy priority order (all
extensions) is used for back-compat.

Users who need PowerShell or other interpreters can call them from within
a ``.bat`` file (e.g. ``powershell -File script.ps1``).
"""

import shlex
from pathlib import Path, PurePath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harbor.models.task.config import TaskOS

SUPPORTED_EXTENSIONS: list[str] = [".sh", ".bat"]
LINUX_EXTENSIONS: list[str] = [".sh"]
WINDOWS_EXTENSIONS: list[str] = [".bat"]

_WINDOWS_UNSAFE_CHARS: tuple[str, ...] = ('"', "%", "!", "\r", "\n", "\x00")
# Characters that require quoting in cmd.exe (special metacharacters)
_WINDOWS_CHARS_REQUIRING_QUOTES: tuple[str, ...] = (
    " ",
    "&",
    "|",
    "<",
    ">",
    "^",
    "(",
    ")",
)


def quote_windows_shell_arg(value: str | PurePath) -> str:
    """Quote a value for embedding inside a cmd.exe command string.

    Wraps in double quotes when the value contains spaces or cmd.exe
    metacharacters. cmd.exe treats ``&|<>^()`` as literal within a quoted
    region. Rejects characters that cannot be neutralized: ``"`` terminates
    the quoted region, ``%`` and ``!`` still trigger variable expansion
    inside quotes, and ``\\r\\n\\x00`` break line parsing.

    Forward slashes are converted to backslashes for cmd.exe compatibility
    because cmd.exe's redirection operators (``>``, ``>>``, ``<``) do not
    recognize forward-slash paths, and cmd /S /C with double-quoted paths
    containing backslashes can fail with "filename syntax incorrect" errors.

    Simple paths without special characters are returned unquoted to avoid
    issues with cmd.exe's quote handling in combination with /S flag.
    """
    s = str(value)
    # Convert forward slashes to backslashes for cmd.exe compatibility.
    s = s.replace("/", "\\")
    for char in _WINDOWS_UNSAFE_CHARS:
        if char in s:
            raise ValueError(
                f"Value {s!r} contains {char!r}, which cannot be safely "
                "embedded in a cmd.exe command string"
            )
    # Only quote if the value contains characters that require it
    needs_quotes = any(char in s for char in _WINDOWS_CHARS_REQUIRING_QUOTES)
    return f'"{s}"' if needs_quotes else s


def quote_shell_arg(value: str | PurePath, task_os: "TaskOS | None") -> str:
    """Quote a value for the shell of the given target OS.

    Windows tasks run under cmd.exe (double-quote wrapped); other tasks
    under a POSIX shell (``shlex.quote``). When *task_os* is ``None``,
    defaults to POSIX quoting.
    """
    from harbor.models.task.config import TaskOS

    if task_os == TaskOS.WINDOWS:
        return quote_windows_shell_arg(value)
    return shlex.quote(str(value))


def _extensions_for_os(task_os: "TaskOS | None") -> list[str]:
    """Return the script extensions to try for the given target OS."""
    if task_os is None:
        return SUPPORTED_EXTENSIONS
    # Local import to avoid a circular dependency at module load time.
    from harbor.models.task.config import TaskOS

    if task_os == TaskOS.WINDOWS:
        return WINDOWS_EXTENSIONS
    return LINUX_EXTENSIONS


def discover_script(
    directory: Path | str,
    base_name: str,
    *,
    task_os: "TaskOS | None" = None,
) -> Path | None:
    """Find the first matching script in *directory* named ``{base_name}{ext}``.

    Extensions are tried in priority order, filtered by *task_os* when given.
    Returns ``None`` when no candidate exists.
    """
    directory = Path(directory)
    for ext in _extensions_for_os(task_os):
        candidate = directory / f"{base_name}{ext}"
        if candidate.exists():
            return candidate
    return None


def needs_chmod(script_path: str | PurePath) -> bool:
    """Return ``True`` when the script requires ``chmod +x`` before execution."""
    return _extension(script_path) == ".sh"


def build_execution_command(
    script_path: str | PurePath,
    stdout_path: str | PurePath | None = None,
    *,
    task_os: "TaskOS | None" = None,
    heartbeat_sec: float | None = None,
) -> str:
    """Build the shell command to run *script_path* inside the environment.

    For ``.bat`` files, ``cmd /c`` is used.

    Callers are responsible for running ``chmod +x`` as root before calling
    this for ``.sh`` scripts (use :func:`needs_chmod` to check).

    If *stdout_path* is given, stdout and stderr are redirected to that file.

    When *task_os* is provided, *script_path* and *stdout_path* are quoted
    for the target OS's shell. When omitted, they are interpolated raw —
    callers must pre-quote in that case.

    If *heartbeat_sec* is given (POSIX targets only), the script runs in the
    background while a companion loop prints a timestamped heartbeat line to
    stdout every *heartbeat_sec* seconds, then ``wait`` reaps the script and
    propagates its exit status. Long verifier runs over connection-reaping
    transports (e.g. idle SSH proxies) otherwise sit silent on the channel
    for hours and get the connection killed mid-run.
    """
    script_str = str(script_path)
    ext = _extension(script_str)

    if heartbeat_sec is not None:
        if ext == ".bat":
            raise ValueError(
                f"heartbeat_sec is not supported for .bat scripts ({script_str})"
            )
        if heartbeat_sec <= 0:
            raise ValueError(f"heartbeat_sec must be positive, got {heartbeat_sec}")

    script_piece = (
        quote_shell_arg(script_str, task_os) if task_os is not None else script_str
    )

    if ext == ".bat":
        cmd = f"cmd /c {script_piece}"
    else:
        # .sh and unknown extensions: attempt direct execution.
        cmd = script_piece

    if stdout_path is not None:
        stdout_str = str(stdout_path)
        stdout_piece = (
            quote_shell_arg(stdout_str, task_os) if task_os is not None else stdout_str
        )
        cmd = f"({cmd}) > {stdout_piece} 2>&1"

    if heartbeat_sec is not None:
        # The script (already redirected) runs in the background while a
        # companion loop emits a timestamped line on the exec channel every
        # interval, then ``wait`` propagates the script's exit status as the
        # command's own. Without those bytes, transports that reap idle
        # connections (e.g. SSH proxies with an idle timeout) can kill the
        # channel hours into a silent verifier run.
        interval = f"{heartbeat_sec:g}"
        cmd = (
            f"{cmd} & __harbor_hb=$!; "
            f"while kill -0 $__harbor_hb 2>/dev/null; do "
            f'echo "[heartbeat] $(date -u +%Y-%m-%dT%H:%M:%SZ)"; '
            f"sleep {interval}; done; "
            f"wait $__harbor_hb"
        )

    return cmd


def _extension(path: str | PurePath) -> str:
    """Return the lowercase file extension (e.g. ``'.sh'``).

    ``PurePath`` is sufficient here because suffix detection works for both
    forward-slash paths (``C:/tests/test.bat``) and backslash paths used by
    Windows containers.
    """
    return PurePath(path).suffix.lower()
