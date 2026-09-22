"""HTTP tool built on curl so the local proxy (socks5h) keeps working."""

from __future__ import annotations

import os
import shutil
import subprocess
import urllib.parse

from ..config import AgentConfig
from ..errors import ToolError
from ..registry import Tool, ToolContext, schema

PROXY_CANDIDATES = (
    "socks5h://127.0.0.1:7890",
    "http://127.0.0.1:7890",
    "socks5h://127.0.0.1:1086",
)
PROBE_URL = "https://example.com"
_resolved: dict[str, str | None] = {}


def resolve_proxy(mode: str = "auto") -> str | None:
    if mode in _resolved:
        return _resolved[mode]
    if mode == "none":
        _resolved[mode] = None
        return None
    if mode != "auto":
        _resolved[mode] = mode
        return mode
    for env in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy", "HTTP_PROXY"):
        value = os.environ.get(env)
        if value:
            _resolved[mode] = value
            return value
    for candidate in PROXY_CANDIDATES:
        if _curl_ok(candidate):
            _resolved[mode] = candidate
            return candidate
    _resolved[mode] = None
    return None


def _curl_ok(proxy: str) -> bool:
    if shutil.which("curl") is None:
        return False
    cmd = ["curl", "-s", "-o", "/dev/null", "-m", "6", "-w", "%{http_code}", "-x", proxy, PROBE_URL]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (subprocess.SubprocessError, OSError):
        return False
    return proc.returncode == 0 and proc.stdout.strip().startswith(("2", "3"))


def build(ctx: ToolContext, *, config: AgentConfig, sandboxed: bool = True) -> list[Tool]:
    def http_get(
        ctx_: ToolContext,
        url: str,
        timeout_sec: int = 30,
        proxy: str = "auto",
        max_bytes: int = 500_000,
    ) -> dict:
        return _curl(ctx_, url, timeout_sec, proxy, max_bytes, method="GET")

    def http_probe(
        ctx_: ToolContext,
        url: str,
        timeout_sec: int = 15,
        proxy: str = "auto",
    ) -> dict:
        result = _curl(ctx_, url, timeout_sec, proxy, 2000, method="HEAD")
        return {
            "url": url,
            "reachable": bool(result.get("status_code", 0) and result["status_code"] < 400),
            "status_code": result.get("status_code"),
            "error": result.get("error"),
            "proxy": result.get("proxy"),
        }

    return [
        Tool(
            name="http_get",
            description="Fetch a URL with curl (honours the local proxy when auto).",
            input_schema=schema(
                {
                    "url": {"type": "string"},
                    "timeout_sec": {"type": "integer"},
                    "proxy": {"type": "string"},
                    "max_bytes": {"type": "integer"},
                },
                ["url"],
            ),
            handler=http_get,
            permission_note="outbound network",
            side_effects=True,
        ),
        Tool(
            name="http_probe",
            description="HEAD a URL to check it resolves (used by link validators).",
            input_schema=schema(
                {"url": {"type": "string"}, "timeout_sec": {"type": "integer"}, "proxy": {"type": "string"}},
                ["url"],
            ),
            handler=http_probe,
            permission_note="outbound network",
            side_effects=True,
        ),
    ]


def _curl(
    ctx: ToolContext,
    url: str,
    timeout_sec: int,
    proxy_mode: str,
    max_bytes: int,
    *,
    method: str,
) -> dict:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ToolError(f"refusing to fetch non-http URL: {url}")
    if shutil.which("curl") is None:
        raise ToolError("curl is not installed")

    proxy = resolve_proxy(proxy_mode)
    marker = "\n__STATUS__:"
    cmd = [
        "curl",
        "-sS",
        "-L",
        "--compressed",
        "--max-time",
        str(timeout_sec),
        "-A",
        "agent-harness/0.1 (+local)",
        "-w",
        f"{marker}%{{http_code}}",
    ]
    if method == "HEAD":
        cmd += ["-I", "-o", "/dev/null"]
    if proxy:
        cmd += ["-x", proxy]
    cmd.append(url)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec + 10)
    except subprocess.SubprocessError as exc:
        return {"url": url, "ok": False, "error": str(exc), "proxy": proxy}
    if proc.returncode != 0:
        return {
            "url": url,
            "ok": False,
            "error": (proc.stderr or "").strip()[:500],
            "proxy": proxy,
        }

    body, _, status = proc.stdout.rpartition(marker)
    try:
        code = int(status.strip() or 0)
    except ValueError:
        code = 0
    truncated = len(body) > max_bytes
    return {
        "url": url,
        "ok": 200 <= code < 400,
        "status_code": code,
        "proxy": proxy,
        "truncated": truncated,
        "body": body[:max_bytes],
    }
