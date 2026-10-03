"""One bot per phone, enforced with an OS file lock.

Two or three bots on one iPhone (stray scheduled tasks) fight over the same game and
the same screen stream all night: blind taps land on each other's screens, the phone
runs out of stream sockets. Each bot holds an exclusive lock on local/bot-<serial>.lock
for as long as it runs. The OS drops the lock when the process ends, however it ends,
so a killed bot never leaves a stale lock behind.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path


class AlreadyRunning(RuntimeError):
    """Another bot already holds this phone's lock."""


def lock_path(root: Path, serial: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", serial).strip("-") or "default"
    return root / "local" / f"bot-{safe}.lock"


class InstanceLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = None

    def acquire(self) -> InstanceLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        f = open(self.path, "a+b")  # held open while the bot runs
        try:
            _lock(f)
        except OSError:
            f.close()
            raise AlreadyRunning(
                "Another copy of the bot is already playing on this phone"
            ) from None
        self._file = f
        try:  # who holds it, for a person looking at the file (Windows: our own lock allows it)
            f.seek(0)
            f.truncate()
            f.write(f"{os.getpid()}\n".encode())
            f.flush()
        except OSError:
            pass
        return self

    def release(self) -> None:
        f, self._file = self._file, None
        if f is None:
            return
        try:
            _unlock(f)
        except OSError:
            pass
        f.close()

    @property
    def held(self) -> bool:
        return self._file is not None


if sys.platform == "win32":
    import msvcrt

    def _lock(f) -> None:
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock(f) -> None:
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _lock(f) -> None:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(f) -> None:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def acquire(root: Path, serial: str) -> InstanceLock:
    """Lock this phone for this process, or raise AlreadyRunning."""
    return InstanceLock(lock_path(root, serial)).acquire()
