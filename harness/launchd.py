"""Render and install the user-level launchd agent for a task.

Nothing is loaded until you explicitly ask for it.
"""

from __future__ import annotations

import plistlib
import subprocess
from pathlib import Path

from . import paths
from .errors import TaskError
from .taskspec import TaskSpec

DEFAULT_LABEL = "com.simonzj.agent"
LAUNCH_AGENTS = Path("~/Library/LaunchAgents").expanduser()


def label_for(task: TaskSpec) -> str:
    trigger = task.trigger or {}
    return str(trigger.get("label") or f"{DEFAULT_LABEL}.{task.name}")


def render(task: TaskSpec, *, python: str | None = None) -> dict:
    trigger = task.trigger or {}
    if trigger.get("type", "launchd") != "launchd":
        raise TaskError(f"task {task.name} does not declare a launchd trigger")
    repo = paths.repo_root()
    launcher = repo / "agent"
    args = [str(launcher), "run", task.name, "--trigger", "launchd"]
    if trigger.get("publish"):
        args.append("--publish")

    payload: dict = {
        "Label": label_for(task),
        "ProgramArguments": ["/bin/zsh", "-lc", _shell_command(repo, args)],
        "WorkingDirectory": str(repo),
        "StandardOutPath": str(paths.logs_dir() / f"{task.name}.launchd.out.log"),
        "StandardErrorPath": str(paths.logs_dir() / f"{task.name}.launchd.err.log"),
        "RunAtLoad": bool(trigger.get("run_at_load", False)),
        "ProcessType": "Background",
        "EnvironmentVariables": {
            "AGENT_HOME": str(paths.home()),
            "AGENT_REPO_ROOT": str(repo),
            "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
            "AGENT_PYTHON": python or "python3",
        },
    }
    calendar = trigger.get("calendar") or [{"hour": 23, "minute": 0}]
    if len(calendar) == 1:
        payload["StartCalendarInterval"] = {
            "Hour": int(calendar[0].get("hour", 23)),
            "Minute": int(calendar[0].get("minute", 0)),
        }
    else:
        payload["StartCalendarInterval"] = [
            {"Hour": int(entry.get("hour", 0)), "Minute": int(entry.get("minute", 0))}
            for entry in calendar
        ]
    return payload


def to_xml(task: TaskSpec, *, python: str | None = None) -> str:
    return plistlib.dumps(render(task, python=python), fmt=plistlib.FMT_XML).decode("utf-8")


def install(task: TaskSpec, *, load: bool = False, python: str | None = None) -> Path:
    LAUNCH_AGENTS.mkdir(parents=True, exist_ok=True)
    target = LAUNCH_AGENTS / f"{label_for(task)}.plist"
    target.write_text(to_xml(task, python=python), encoding="utf-8")
    lint = subprocess.run(["plutil", "-lint", str(target)], capture_output=True, text=True)
    if lint.returncode != 0:
        raise TaskError(f"generated plist is invalid: {lint.stderr.strip()}")
    if load:
        subprocess.run(["launchctl", "unload", str(target)], capture_output=True, text=True)
        loaded = subprocess.run(["launchctl", "load", str(target)], capture_output=True, text=True)
        if loaded.returncode != 0:
            raise TaskError(f"launchctl load failed: {loaded.stderr.strip()}")
    return target


def status(task: TaskSpec) -> dict:
    label = label_for(task)
    target = LAUNCH_AGENTS / f"{label}.plist"
    listed = subprocess.run(["launchctl", "list"], capture_output=True, text=True)
    loaded = label in listed.stdout
    return {"label": label, "plist": str(target), "installed": target.is_file(), "loaded": loaded}


def _shell_command(repo: Path, args: list[str]) -> str:
    quoted = " ".join(_quote(part) for part in args)
    return f"cd {_quote(str(repo))} && {quoted}"


def _quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"
