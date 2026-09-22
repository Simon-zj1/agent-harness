"""Filesystem tools confined to the paths a task declared."""

from __future__ import annotations

from pathlib import Path

from ..config import AgentConfig
from ..errors import PermissionDenied
from ..registry import Tool, ToolContext, schema


def _resolve(ctx: ToolContext, raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        base = ctx.data.get("task_dir")
        path = (Path(base) / path) if base else (Path.cwd() / path)
    return path


def build(ctx: ToolContext, *, config: AgentConfig, sandboxed: bool = True) -> list[Tool]:
    def fs_read(ctx_: ToolContext, path: str, max_bytes: int = 400_000) -> dict:
        target = _resolve(ctx_, path)
        if sandboxed and not ctx_.within_readable(target):
            raise PermissionDenied(f"read outside declared paths: {target}")
        if not target.is_file():
            raise FileNotFoundError(f"no such file: {target}")
        text = target.read_text(encoding="utf-8", errors="replace")
        truncated = len(text) > max_bytes
        return {
            "path": str(target),
            "bytes": target.stat().st_size,
            "truncated": truncated,
            "content": text[:max_bytes],
        }

    def fs_write(ctx_: ToolContext, path: str, content: str) -> dict:
        target = _resolve(ctx_, path)
        if sandboxed and not ctx_.within_writable(target):
            raise PermissionDenied(f"write outside declared paths: {target}")
        if ctx_.dry_run and ctx_.data.get("fs_write_dry_run_blocks"):
            return {"path": str(target), "dry_run": True, "written": False}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"path": str(target), "bytes": len(content.encode()), "written": True}

    def fs_list(ctx_: ToolContext, path: str, pattern: str = "*") -> dict:
        target = _resolve(ctx_, path)
        if sandboxed and not ctx_.within_readable(target):
            raise PermissionDenied(f"list outside declared paths: {target}")
        if not target.is_dir():
            raise NotADirectoryError(str(target))
        entries = sorted(p.name for p in target.glob(pattern))
        return {"path": str(target), "count": len(entries), "entries": entries[:500]}

    def fs_exists(ctx_: ToolContext, path: str) -> dict:
        target = _resolve(ctx_, path)
        return {"path": str(target), "exists": target.exists(), "is_file": target.is_file()}

    return [
        Tool(
            name="fs_read",
            description="Read a UTF-8 text file inside the task's declared read paths.",
            input_schema=schema(
                {"path": {"type": "string"}, "max_bytes": {"type": "integer"}},
                ["path"],
            ),
            handler=fs_read,
            permission_note="read-only, confined to declared paths",
        ),
        Tool(
            name="fs_write",
            description="Write a UTF-8 text file inside the task's declared writable paths.",
            input_schema=schema(
                {"path": {"type": "string"}, "content": {"type": "string"}},
                ["path", "content"],
            ),
            handler=fs_write,
            permission_note="write, confined to declared writable paths",
            side_effects=True,
        ),
        Tool(
            name="fs_list",
            description="List directory entries.",
            input_schema=schema({"path": {"type": "string"}, "pattern": {"type": "string"}}, ["path"]),
            handler=fs_list,
            permission_note="read-only",
        ),
        Tool(
            name="fs_exists",
            description="Check whether a path exists.",
            input_schema=schema({"path": {"type": "string"}}, ["path"]),
            handler=fs_exists,
            permission_note="read-only",
        ),
    ]
