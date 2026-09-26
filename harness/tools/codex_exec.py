"""Codex CLI as a replaceable executor tool.

The harness keeps memory, scheduling and acceptance; Codex does the heavy
multi-step work when a task asks for it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from ..config import AgentConfig
from ..errors import ToolError
from ..registry import Tool, ToolContext, schema


def build(ctx: ToolContext, *, config: AgentConfig, sandboxed: bool = True) -> list[Tool]:
    def codex_exec(
        ctx_: ToolContext,
        prompt: str,
        workdir: str,
        sandbox: str | None = None,
        model: str | None = None,
        ephemeral: bool | None = None,
        timeout_sec: int | None = None,
    ) -> dict:
        executor = config.executors.get("codex")
        if executor is None:
            raise ToolError("no [executors.codex] section in the config")
        binary = shutil.which(executor.command)
        workdir_path = Path(workdir).expanduser()
        if not workdir_path.is_dir():
            raise ToolError(f"workdir does not exist: {workdir_path}")

        out_file = ctx_.run_dir / "codex-last-message.txt"
        argv = [
            # Falling back to the configured name keeps the planned argv readable
            # in dry-run on a machine where the executor is not installed.
            binary or executor.command,
            "exec",
            "-C",
            str(workdir_path),
            "-s",
            sandbox or executor.sandbox or "workspace-write",
            "--skip-git-repo-check",
            "--json",
            "-o",
            str(out_file),
        ]
        if model or executor.model:
            argv += ["-m", model or str(executor.model)]
        if ephemeral if ephemeral is not None else executor.ephemeral:
            argv.append("--ephemeral")
        argv += list(executor.extra_args)
        argv.append(prompt)

        if ctx_.dry_run and ctx_.data.get("executor_dry_run_blocks", True):
            return {"argv": argv, "dry_run": True, "executed": False}

        # Only a real invocation needs the binary. Requiring it earlier made
        # dry-run fail on any machine without the executor installed, which is
        # exactly the machine dry-run exists for.
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
            raise ToolError(f"codex exec timed out after {timeout}s") from exc

        usage = _parse_usage(proc.stdout)
        last_message = out_file.read_text(encoding="utf-8") if out_file.is_file() else ""
        return {
            "executor": "codex",
            "argv": argv,
            "executed": True,
            "returncode": proc.returncode,
            "duration_ok": proc.returncode == 0,
            "last_message": last_message[-8000:],
            "tokens_in": usage.get("input_tokens", 0),
            "tokens_out": usage.get("output_tokens", 0),
            "stderr": proc.stderr[-2000:],
        }

    return [
        Tool(
            name="codex_exec",
            description=(
                "Delegate a self-contained multi-step task to the local Codex CLI "
                "(non-interactive, sandboxed) and return its final message."
            ),
            input_schema=schema(
                {
                    "prompt": {"type": "string"},
                    "workdir": {"type": "string"},
                    "sandbox": {"type": "string", "enum": ["read-only", "workspace-write", "danger-full-access"]},
                    "model": {"type": "string"},
                    "ephemeral": {"type": "boolean"},
                    "timeout_sec": {"type": "integer"},
                },
                ["prompt", "workdir"],
            ),
            handler=codex_exec,
            permission_note="spawns the Codex CLI with its own sandbox policy",
            side_effects=True,
        )
    ]


def _parse_usage(stdout: str) -> dict[str, int]:
    """Best-effort extraction of token usage from `codex exec --json` events."""
    usage = {"input_tokens": 0, "output_tokens": 0}
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        payload = event.get("usage") or (event.get("msg") or {}).get("usage")
        if not isinstance(payload, dict):
            continue
        for key, target in (
            ("input_tokens", "input_tokens"),
            ("output_tokens", "output_tokens"),
            ("prompt_tokens", "input_tokens"),
            ("completion_tokens", "output_tokens"),
        ):
            if isinstance(payload.get(key), int):
                usage[target] = max(usage[target], int(payload[key]))
    return usage
