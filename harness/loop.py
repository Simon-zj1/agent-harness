"""A small, explicit agent loop (experimental module).

This is the part that is *ours*: the model is a component, the harness decides
when to stop, what the tool budget is, and how results are validated.
The production `daily-trends` compose path currently calls providers directly;
this loop is kept as a tested experiment rather than presented as the main loop.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .providers.base import ChatResponse
from .registry import Registry


@dataclass
class LoopResult:
    status: str
    final_text: str = ""
    steps: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    duration_ms: int = 0
    error: str | None = None

    def metrics(self) -> dict[str, Any]:
        return {
            "loop_steps": self.steps,
            "loop_tool_calls": len(self.tool_calls),
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "duration_ms": self.duration_ms,
        }


def run_loop(
    provider: Any,
    registry: Registry,
    *,
    system_prompt: str,
    user_prompt: str,
    max_steps: int = 12,
    deadline_sec: int | None = None,
    logger: Any | None = None,
    on_step: Callable[[int, ChatResponse], None] | None = None,
) -> LoopResult:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    result = LoopResult(status="max_steps")
    started = time.monotonic()
    tools = registry.specs()

    for step in range(1, max_steps + 1):
        if deadline_sec and time.monotonic() - started > deadline_sec:
            result.status = "timeout"
            result.error = f"loop exceeded {deadline_sec}s"
            break
        response = provider.chat(messages, tools=tools or None)
        result.steps = step
        result.tokens_in += response.tokens_in
        result.tokens_out += response.tokens_out
        if on_step:
            on_step(step, response)
        if logger:
            logger.info(
                "loop step %d: %d tool call(s), %d/%d tokens",
                step,
                len(response.tool_calls),
                response.tokens_in,
                response.tokens_out,
            )

        if not response.tool_calls:
            result.status = "done"
            result.final_text = response.text
            break

        messages.append(
            {
                "role": "assistant",
                "content": response.text or "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": call.raw_arguments or json.dumps(call.arguments),
                        },
                    }
                    for call in response.tool_calls
                ],
            }
        )
        for call in response.tool_calls:
            try:
                output = registry.call(call.name, call.arguments)
                payload = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False, default=str)
                ok = True
            except Exception as exc:
                payload, ok = f"ERROR: {exc}", False
            result.tool_calls.append({"name": call.name, "ok": ok, "args": call.arguments})
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": payload[:20_000]}
            )

    result.duration_ms = int((time.monotonic() - started) * 1000)
    return result
