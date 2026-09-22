"""Built-in tool set."""

from __future__ import annotations

from ..config import AgentConfig
from ..registry import Registry, ToolContext
from . import claude_exec, codex_exec, fs_tool, http_tool, notify_tool, shell_tool


def build_registry(
    ctx: ToolContext,
    *,
    config: AgentConfig,
    ledger=None,
    logger=None,
    sandboxed: bool = True,
) -> Registry:
    registry = Registry(ctx, ledger=ledger, logger=logger)
    for module in (fs_tool, shell_tool, http_tool, notify_tool, codex_exec, claude_exec):
        for tool in module.build(ctx, config=config, sandboxed=sandboxed):
            registry.register(tool)
    return registry


__all__ = ["build_registry"]
