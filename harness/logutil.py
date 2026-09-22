"""Daily logs plus a per-run log file."""

from __future__ import annotations

import datetime as dt
import logging
import sys
from pathlib import Path

from . import paths

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_configured: set[str] = set()


def _handler(path: Path) -> logging.Handler:
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(_FORMAT))
    return handler


def logger(name: str, *, run_dir: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Return a logger writing to runs/logs/<date>.log and optionally runs/<id>/run.log."""
    log = logging.getLogger(name)
    log.setLevel(level)
    log.propagate = False

    daily = paths.logs_dir() / f"{dt.date.today().isoformat()}.log"
    key = f"{name}:{daily}"
    if key not in _configured:
        log.addHandler(_handler(daily))
        _configured.add(key)

    if run_dir is not None:
        run_log = run_dir / "run.log"
        run_key = f"{name}:{run_log}"
        if run_key not in _configured:
            log.addHandler(_handler(run_log))
            _configured.add(run_key)
    return log


def console(level: int = logging.INFO) -> logging.Logger:
    log = logging.getLogger("agent.cli")
    log.setLevel(level)
    log.propagate = False
    if not log.handlers:
        # Human-facing status goes to stdout: this is a CLI, and people pipe it.
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(stream)
    return log
