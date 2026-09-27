"""Delivery: macOS notification now, webhook as an optional hook."""

from __future__ import annotations

import json
import subprocess
import urllib.parse
import urllib.request

from ..config import AgentConfig
from ..errors import PermissionDenied, ToolError
from ..registry import Tool, ToolContext, schema


def build(ctx: ToolContext, *, config: AgentConfig, sandboxed: bool = True) -> list[Tool]:
    def notify(
        ctx_: ToolContext,
        title: str,
        message: str,
        subtitle: str = "",
        webhook: str | None = None,
    ) -> dict:
        target = webhook if webhook is not None else config.notify.webhook
        planned = {"title": title, "message": message, "subtitle": subtitle, "webhook": bool(target)}
        if target:
            parsed = urllib.parse.urlparse(target)
            if parsed.scheme not in ("http", "https"):
                raise ToolError(f"notify webhook must be http(s): {target!r}")
            allowed = set(config.notify.webhook_allow_hosts or [])
            if allowed and parsed.hostname not in allowed:
                raise PermissionDenied(
                    f"notify webhook host {parsed.hostname!r} is not in "
                    f"webhook_allow_hosts={sorted(allowed)}"
                )

        if ctx_.dry_run or not config.notify.enabled:
            return {**planned, "delivered": False, "reason": "dry-run or notify disabled"}

        delivered = {"macos": False, "webhook": False}
        if config.notify.macos:
            script = (
                f'display notification {_as_applescript(message)} '
                f'with title {_as_applescript(title)}'
            )
            if subtitle:
                script += f" subtitle {_as_applescript(subtitle)}"
            proc = subprocess.run(
                ["osascript", "-e", script], capture_output=True, text=True, check=False
            )
            delivered["macos"] = proc.returncode == 0
        if target:
            payload = json.dumps(
                {"title": title, "message": message, "subtitle": subtitle}, ensure_ascii=False
            ).encode()
            request = urllib.request.Request(
                target, data=payload, headers={"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(request, timeout=15) as response:
                    delivered["webhook"] = 200 <= response.status < 300
            except Exception:
                delivered["webhook"] = False
        return {**planned, "delivered": delivered}

    return [
        Tool(
            name="notify",
            description="Deliver a short notification to the owner (macOS notification / webhook).",
            input_schema=schema(
                {
                    "title": {"type": "string"},
                    "message": {"type": "string"},
                    "subtitle": {"type": "string"},
                    "webhook": {"type": "string"},
                },
                ["title", "message"],
            ),
            handler=notify,
            permission_note="local notification + optional outbound webhook",
            side_effects=True,
        )
    ]


def _as_applescript(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
