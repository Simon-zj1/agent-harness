"""Deterministic provider for offline tests and replay arms."""

from __future__ import annotations

import hashlib
import json

from ..config import ProviderConfig
from .base import ChatResponse, ToolCall


class ReplayProvider:
    """Never calls the network: returns a stable, inspectable answer.

    When the last user message contains a `TOOL_CALL:` directive the provider
    emits that tool call once, which lets the loop be tested offline.
    """

    def __init__(self, config: ProviderConfig | None = None) -> None:
        self.config = config or ProviderConfig(name="replay", type="replay", model="replay")
        self.name = self.config.name
        self.calls = 0

    def available(self) -> tuple[bool, str]:
        return True, "offline replay"

    def chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        self.calls += 1
        last = next(
            (m for m in reversed(messages) if m.get("role") == "user"), {"content": ""}
        )
        content = str(last.get("content", ""))
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]

        directive = _directive(content)
        if directive and not any(m.get("role") == "tool" for m in messages):
            call = ToolCall(
                id=f"replay_{digest}",
                name=directive["name"],
                arguments=directive.get("arguments", {}),
            )
            return ChatResponse(tool_calls=[call], model="replay", finish_reason="tool_calls")

        return ChatResponse(
            text=json.dumps(
                {"replay": True, "sha": digest, "chars": len(content)}, ensure_ascii=False
            ),
            tokens_in=len(content) // 4,
            tokens_out=8,
            model="replay",
            finish_reason="stop",
        )


def _directive(content: str) -> dict | None:
    marker = "TOOL_CALL:"
    if marker not in content:
        return None
    tail = content.split(marker, 1)[1].strip().splitlines()[0]
    try:
        parsed = json.loads(tail)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict) and "name" in parsed:
        return parsed
    return None
