"""Advisory file locks.

Guarantees that one task+repo is only touched by one process at a time, even
when launchd, a manual run and an experiment overlap.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path

from . import paths
from .errors import LockBusy


class FileLock:
    """Non-blocking exclusive lock; can optionally wait for a bounded time."""

    def __init__(self, name: str, *, directory: Path | None = None) -> None:
        safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in name)
        self.path = (directory or paths.locks_dir()) / f"{safe}.lock"
        self._fh = None
        self._meta: dict[str, object] = {}

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()

    def acquire(self, *, wait_sec: float = 0.0, poll: float = 0.5) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + max(0.0, wait_sec)
        while True:
            fh = open(self.path, "a+", encoding="utf-8")
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                fh.seek(0)
                holder = fh.read().strip()
                fh.close()
                if time.monotonic() >= deadline:
                    raise LockBusy(
                        f"lock {self.path.name} held by another process: {holder or 'unknown'}"
                    )
                time.sleep(poll)
                continue

            self._meta = {
                "pid": os.getpid(),
                "host": os.uname().nodename,
                "acquired_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "name": self.path.stem,
            }
            fh.seek(0)
            fh.truncate()
            fh.write(json.dumps(self._meta, ensure_ascii=False))
            fh.flush()
            self._fh = fh
            return

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            self._fh.seek(0)
            self._fh.truncate()
            self._fh.write('{"released": true}\n')
            self._fh.flush()
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()
            self._fh = None

    @property
    def holder(self) -> str:
        try:
            return self.path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""
