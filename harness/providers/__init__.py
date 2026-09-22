"""Provider factory."""

from __future__ import annotations

from ..config import AgentConfig, ProviderConfig
from ..errors import ConfigError
from .base import ChatResponse, Provider, ToolCall
from .openai_compat import OpenAICompatProvider
from .replay import ReplayProvider


def build(config: ProviderConfig) -> Provider:
    if config.type in ("openai_compat", "openai", "deepseek", "local"):
        return OpenAICompatProvider(config)
    if config.type == "replay":
        return ReplayProvider(config)
    raise ConfigError(f"unsupported provider type {config.type!r} for {config.name!r}")


def get(agent_config: AgentConfig, name: str | None = None) -> Provider:
    return build(agent_config.provider(name))


__all__ = ["build", "get", "ChatResponse", "Provider", "ToolCall", "OpenAICompatProvider", "ReplayProvider"]
