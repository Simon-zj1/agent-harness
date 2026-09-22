"""Filesystem layout.

`AGENT_HOME` relocates all mutable state (memory/, runs/, logs) so tests and
experiments can run in isolation without touching the real workspace.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def repo_root() -> Path:
    raw = os.environ.get("AGENT_REPO_ROOT")
    if raw:
        return Path(raw).expanduser().resolve()
    return _REPO_ROOT


def home() -> Path:
    raw = os.environ.get("AGENT_HOME")
    if raw:
        return Path(raw).expanduser().resolve()
    return repo_root()


def tasks_dir() -> Path:
    raw = os.environ.get("AGENT_TASKS_DIR")
    if raw:
        return Path(raw).expanduser().resolve()
    local = home() / "tasks"
    if local.is_dir():
        return local
    return repo_root() / "tasks"


def experiments_dir() -> Path:
    raw = os.environ.get("AGENT_EXPERIMENTS_DIR")
    if raw:
        return Path(raw).expanduser().resolve()
    local = home() / "experiments"
    if local.is_dir():
        return local
    return repo_root() / "experiments"


def config_path() -> Path:
    raw = os.environ.get("AGENT_CONFIG")
    if raw:
        return Path(raw).expanduser().resolve()
    local = home() / "config" / "agent.toml"
    if local.is_file():
        return local
    return repo_root() / "config" / "agent.toml"


def memory_dir() -> Path:
    return home() / "memory"


def runs_dir() -> Path:
    return home() / "runs"


def logs_dir() -> Path:
    return runs_dir() / "logs"


def locks_dir() -> Path:
    return runs_dir() / "locks"


def ledger_path() -> Path:
    return runs_dir() / "runs.db"


def memory_index_path() -> Path:
    return memory_dir() / "index.db"


def ensure_layout() -> None:
    for path in (
        memory_dir(),
        memory_dir() / "runs",
        memory_dir() / "notes",
        runs_dir(),
        logs_dir(),
        locks_dir(),
        runs_dir() / "experiments",
    ):
        path.mkdir(parents=True, exist_ok=True)


def display(path: Path | str) -> str:
    """Render a path relative to the agent home when possible."""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(home()))
    except (ValueError, OSError):
        return str(p)
