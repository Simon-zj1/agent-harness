"""Claude Code CLI as a second, comparable executor."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from ..config import AgentConfig
from ..errors import ToolError
from ..registry import Tool, ToolContext, schema


def build(ctx: ToolContext, *, config: AgentConfig, sandboxed: bool = True) -> list[Tool]:
    def claude_exec(
        ctx_: ToolContext,
        prompt: str,
        workdir: str,
        model: str | None = None,
        permission_mode: str | None = None,
        timeout_sec: int | None = None,
    ) -> dict:
        executor = config.executors.get("claude")
        if executor is None:
            raise ToolError("no [executors.claude] section in the config")
        binary = shutil.which(executor.command)
        workdir_path = Path(workdir).expanduser()
        if not workdir_path.is_dir():
            raise ToolError(f"workdir does not exist: {workdir_path}")

        argv = [
            # Keep the planned argv readable in dry-run without the binary.
            binary or executor.command,
            "-p",
            prompt,
            "--output-format",
            "json",
            "--permission-mode",
            permission_mode or executor.permission_mode or "acceptEdits",
            "--add-dir",
            str(workdir_path),
        ]
        if model or executor.model:
            argv += ["--model", model or str(executor.model)]
        argv += list(executor.extra_args)

        if ctx_.dry_run and ctx_.data.get("executor_dry_run_blocks", True):
            return {"argv": argv, "dry_run": True, "executed": False}

        # Only a real invocation needs the binary; see codex_exec for why this
        # check cannot come first.
        if binary is None:
            raise ToolError(f"executor binary not found: {executor.command}")

        timeout = timeout_sec or executor.timeout_sec
        try:
            proc = subprocess.run(
                argv,
                cwd=str(workdir_path),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolError(f"claude exec timed out after {timeout}s") from exc

        payload: dict = {}
        text = proc.stdout.strip()
        if text.startswith("{"):
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = {}
        usage = payload.get("usage") or {}
        return {
            "executor": "claude",
            "argv": argv,
            "executed": True,
            "returncode": proc.returncode,
            "last_message": str(payload.get("result", ""))[-8000:] or text[-4000:],
            "tokens_in": int(usage.get("input_tokens") or 0),
            "tokens_out": int(usage.get("output_tokens") or 0),
            "cost_usd": payload.get("total_cost_usd"),
            "stderr": proc.stderr[-2000:],
        }

    return [
        Tool(
            name="claude_exec",
            description="Delegate a self-contained task to the local Claude Code CLI (print mode).",
            input_schema=schema(
                {
                    "prompt": {"type": "string"},
                    "workdir": {"type": "string"},
                    "model": {"type": "string"},
                    "permission_mode": {"type": "string"},
                    "timeout_sec": {"type": "integer"},
                },
                ["prompt", "workdir"],
            ),
            handler=claude_exec,
            permission_note="spawns Claude Code with its own permission mode",
            side_effects=True,
        )
    ]
