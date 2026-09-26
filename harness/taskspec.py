"""Declarative task definitions (task.toml, or task.yaml when PyYAML exists).

TOML is the canonical format: the harness stays standard-library only, and
`tomllib` ships with Python 3.11+.
"""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import paths
from .errors import TaskError


@dataclass
class StepSpec:
    id: str
    command: list[str]
    when: str = "always"  # always | validation_passed | published
    skip_on_dry_run: bool = False
    retries: int | None = None
    timeout_sec: int | None = None
    allow_degrade: bool = False
    requires_publish: bool = False
    optional: bool = False
    note: str = ""


@dataclass
class ValidatorSpec:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    required: bool = True


@dataclass
class TaskSpec:
    name: str
    dir: Path
    title: str = ""
    description: str = ""
    date_mode: str = "today"  # today | yesterday
    timezone: str = "Asia/Shanghai"
    allowed_tools: list[str] = field(default_factory=list)
    readable_paths: list[str] = field(default_factory=list)
    writable_paths: list[str] = field(default_factory=list)
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    budget: dict[str, Any] = field(default_factory=dict)
    publish: dict[str, Any] = field(default_factory=dict)
    policy: dict[str, Any] = field(default_factory=dict)
    trigger: dict[str, Any] = field(default_factory=dict)
    paths_table: dict[str, str] = field(default_factory=dict)
    steps: list[StepSpec] = field(default_factory=list)
    validators: list[ValidatorSpec] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def vars(self, *, date: str, run_dir: Path | None = None) -> dict[str, str]:
        extra = {k: str(v) for k, v in self.paths_table.items()}
        extra["date"] = date
        if run_dir is not None:
            extra["run_dir"] = str(run_dir)
        return extra

    def path_for(self, template: str, *, date: str, run_dir: Path | None = None) -> Path:
        """Resolve a task-relative, `{date}`/`{run_dir}` templated path."""
        rendered = render(template, date=date, run_dir=run_dir, extra=self.paths_table)
        path = Path(rendered).expanduser()
        if not path.is_absolute():
            path = self.dir / path
        return path

    def allowed_tool_names(self) -> list[str]:
        return list(self.allowed_tools)


def render(
    template: str,
    *,
    date: str,
    run_dir: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    text = _expand_env(template)
    text = text.replace("{date}", date)
    if run_dir is not None:
        text = text.replace("{run_dir}", str(run_dir))
    for key, value in (extra or {}).items():
        # Values from a task's [paths] table may themselves be env-templated.
        text = text.replace("{" + str(key) + "}", _expand_env(str(value)))
    return text


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(text: str) -> str:
    """Expand `${VAR}` and `${VAR:-default}` so paths stay portable.

    Task files can then ship author defaults while any machine overrides them:
        tools_dir = "${DAILY_TRENDS_DIR:-/Users/me/daily-trends}"
    """

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        return os.environ.get(name, default if default is not None else "")

    return _ENV_PATTERN.sub(replace, text)


def load(name_or_dir: str | Path, *, root: Path | None = None) -> TaskSpec:
    base = Path(name_or_dir)
    if not base.is_absolute() and not base.is_dir():
        base = (root or paths.tasks_dir()) / str(name_or_dir)
    if not base.is_dir():
        raise TaskError(f"task directory not found: {base}")

    toml_path = base / "task.toml"
    yaml_path = base / "task.yaml"
    if toml_path.is_file():
        raw = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    elif yaml_path.is_file():
        raw = _load_yaml(yaml_path)
    else:
        raise TaskError(f"{base} has no task.toml or task.yaml")

    name = raw.get("name") or base.name
    steps = [_step(entry) for entry in raw.get("steps", [])]
    if not steps:
        raise TaskError(f"task {name} declares no steps")
    seen: set[str] = set()
    for step in steps:
        if step.id in seen:
            raise TaskError(f"task {name} has duplicate step id {step.id!r}")
        seen.add(step.id)

    validators = [
        ValidatorSpec(
            name=entry["name"],
            args=entry.get("args", {}),
            required=entry.get("required", True),
        )
        for entry in raw.get("validators", [])
    ]

    return TaskSpec(
        name=name,
        dir=base.resolve(),
        title=raw.get("title", ""),
        description=raw.get("description", ""),
        date_mode=raw.get("date_mode", "today"),
        timezone=raw.get("timezone", "Asia/Shanghai"),
        allowed_tools=raw.get("allowed_tools", []),
        readable_paths=raw.get("readable_paths", []),
        writable_paths=raw.get("writable_paths", []),
        inputs=raw.get("inputs", {}),
        outputs=raw.get("outputs", {}),
        context=raw.get("context", {}),
        budget=raw.get("budget", {}),
        publish=raw.get("publish", {}),
        policy=raw.get("policy", {}),
        trigger=raw.get("trigger", {}),
        paths_table=raw.get("paths", {}),
        steps=steps,
        validators=validators,
        raw=raw,
    )


def available(root: Path | None = None) -> list[str]:
    base = root or paths.tasks_dir()
    if not base.is_dir():
        return []
    names = []
    for child in sorted(base.iterdir()):
        if child.is_dir() and ((child / "task.toml").is_file() or (child / "task.yaml").is_file()):
            names.append(child.name)
    return names


def _step(entry: dict[str, Any]) -> StepSpec:
    if "id" not in entry or "command" not in entry:
        raise TaskError(f"step needs both id and command: {entry}")
    command = entry["command"]
    if isinstance(command, str):
        command = command.split()
    if not isinstance(command, list) or not command:
        raise TaskError(f"step {entry.get('id')}: command must be a non-empty list")
    return StepSpec(
        id=entry["id"],
        command=[str(part) for part in command],
        when=entry.get("when", "always"),
        skip_on_dry_run=bool(entry.get("skip_on_dry_run", False)),
        retries=entry.get("retries"),
        timeout_sec=entry.get("timeout_sec"),
        allow_degrade=bool(entry.get("allow_degrade", False)),
        requires_publish=bool(entry.get("requires_publish", False)),
        optional=bool(entry.get("optional", False)),
        note=entry.get("note", ""),
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise TaskError(
            f"{path} is YAML but PyYAML is not installed; use task.toml "
            "(stdlib `tomllib`) or install PyYAML in a venv"
        ) from exc
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TaskError(f"{path} must contain a mapping")
    return loaded
