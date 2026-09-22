"""Tool interface, permission model and call ledger.

Every side effect a task performs goes through a registered tool, so the
harness always knows who wrote what, under which permission, and whether it
succeeded.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import paths
from .errors import PermissionDenied, ToolError
from .ledger import Ledger


@dataclass
class ToolContext:
    run_id: str
    run_dir: Path
    task_name: str
    target_date: str
    dry_run: bool = False
    publish: bool = False
    readable_paths: list[Path] = field(default_factory=list)
    writable_paths: list[Path] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "task_name": self.task_name,
            "target_date": self.target_date,
            "dry_run": self.dry_run,
            "publish": self.publish,
            "readable_paths": [str(p) for p in self.readable_paths],
            "writable_paths": [str(p) for p in self.writable_paths],
            "data": self.data,
        }

    @classmethod
    def from_json(cls, blob: dict[str, Any]) -> "ToolContext":
        return cls(
            run_id=blob["run_id"],
            run_dir=Path(blob["run_dir"]),
            task_name=blob["task_name"],
            target_date=blob["target_date"],
            dry_run=bool(blob.get("dry_run")),
            publish=bool(blob.get("publish")),
            readable_paths=[Path(p) for p in blob.get("readable_paths", [])],
            writable_paths=[Path(p) for p in blob.get("writable_paths", [])],
            data=blob.get("data", {}),
        )

    def within_writable(self, path: Path) -> bool:
        target = path.expanduser().resolve()
        return any(_is_within(target, root.expanduser().resolve()) for root in self.writable_paths)

    def within_readable(self, path: Path) -> bool:
        target = path.expanduser().resolve()
        roots = list(self.readable_paths) + list(self.writable_paths)
        return any(_is_within(target, root.expanduser().resolve()) for root in roots)


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Any]
    permission_note: str = ""
    side_effects: bool = False

    def spec(self) -> dict[str, Any]:
        """OpenAI-style function-calling spec."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


class Registry:
    """Holds the tools a task is allowed to use and audits each call."""

    def __init__(
        self,
        ctx: ToolContext,
        *,
        tools: dict[str, Tool] | None = None,
        ledger: Ledger | None = None,
        logger: Any | None = None,
    ) -> None:
        self.ctx = ctx
        self.tools: dict[str, Tool] = dict(tools or {})
        self.ledger = ledger
        self.logger = logger
        self._seq = 0

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def allow(self, names: list[str]) -> None:
        missing = [n for n in names if n not in self.tools]
        if missing:
            raise ToolError(f"task allows unknown tools: {missing}")
        self.tools = {n: self.tools[n] for n in names}

    def specs(self) -> list[dict[str, Any]]:
        return [tool.spec() for tool in self.tools.values()]

    def call(self, name: str, args: dict[str, Any] | None = None) -> Any:
        args = dict(args or {})
        tool = self.tools.get(name)
        if tool is None:
            raise PermissionDenied(f"tool {name!r} is not allowed for this task")
        _validate_args(name, tool.input_schema, args)
        self._seq += 1
        started = time.monotonic()
        try:
            result = tool.handler(self.ctx, **args)
        except Exception as exc:  # audit then re-raise
            duration = int((time.monotonic() - started) * 1000)
            if self.ledger:
                self.ledger.record_tool_call(
                    self.ctx.run_id, self._seq, name, args, False, duration, str(exc)
                )
            raise
        duration = int((time.monotonic() - started) * 1000)
        if self.ledger:
            self.ledger.record_tool_call(self.ctx.run_id, self._seq, name, args, True, duration)
        if self.logger:
            self.logger.info("tool %s ok in %dms", name, duration)
        return result


def _is_within(target: Path, root: Path) -> bool:
    try:
        target.relative_to(root)
        return True
    except ValueError:
        return False


_TYPES = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _validate_args(name: str, schema: dict[str, Any], args: dict[str, Any]) -> None:
    """Validate the JSON-Schema subset the built-in tools actually use."""
    if not schema:
        return
    properties = schema.get("properties", {})
    for key in schema.get("required", []):
        if key not in args:
            raise ToolError(f"{name}: missing required argument {key!r}")
    extra = [k for k in args if k not in properties]
    if schema.get("additionalProperties") is False and extra:
        raise ToolError(f"{name}: unexpected arguments {extra}")
    for key, value in args.items():
        spec = properties.get(key)
        if not spec:
            continue
        expected = spec.get("type")
        if expected and expected in _TYPES and not isinstance(value, _TYPES[expected]):
            raise ToolError(
                f"{name}: argument {key!r} must be {expected}, got {type(value).__name__}"
            )
        if expected == "boolean" and not isinstance(value, bool):
            raise ToolError(f"{name}: argument {key!r} must be boolean")
        if expected == "integer" and isinstance(value, bool):
            raise ToolError(f"{name}: argument {key!r} must be integer")
        if "enum" in spec and value not in spec["enum"]:
            raise ToolError(f"{name}: argument {key!r} must be one of {spec['enum']}")
        if spec.get("type") == "array" and "items" in spec:
            item_type = spec["items"].get("type")
            if item_type in _TYPES:
                for item in value:
                    if not isinstance(item, _TYPES[item_type]):
                        raise ToolError(
                            f"{name}: argument {key!r} items must be {item_type}"
                        )


OBJ = "object"


def schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": OBJ,
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def dump_result(result: Any) -> str:
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False, default=str)


def resolve_path(ctx: ToolContext, raw: str) -> Path:
    """Resolve a task-relative path against the agent home."""
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return (ctx.data.get("task_dir") and Path(ctx.data["task_dir"]) / path) or paths.home() / path
