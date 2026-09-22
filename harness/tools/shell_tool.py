"""Shell tool: argv-only, allowlisted, audited."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from ..config import AgentConfig
from ..errors import PermissionDenied, ToolError
from ..registry import Tool, ToolContext, schema

MAX_CAPTURE = 20_000


def build(ctx: ToolContext, *, config: AgentConfig, sandboxed: bool = True) -> list[Tool]:
    allowlist = set(config.shell_allowlist)

    def shell_run(
        ctx_: ToolContext,
        argv: list[str],
        cwd: str | None = None,
        timeout_sec: int | None = None,
        allow_failure: bool = False,
    ) -> dict:
        if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
            raise ToolError("shell_run expects argv as a list of strings")
        program = Path(argv[0]).name
        if sandboxed and program not in allowlist:
            raise PermissionDenied(
                f"command {program!r} is not in the shell allowlist: {sorted(allowlist)}"
            )
        workdir = Path(cwd).expanduser() if cwd else Path(ctx_.data.get("task_dir") or os.getcwd())
        if sandboxed and not ctx_.within_readable(workdir):
            raise PermissionDenied(f"cwd outside declared paths: {workdir}")

        timeout = timeout_sec or int(config.runtime.get("step_timeout_sec", 1800))
        if ctx_.dry_run and program in config.runtime.get("mutating_commands", ["git", "rm", "rsync"]):
            return {
                "argv": argv,
                "cwd": str(workdir),
                "dry_run": True,
                "executed": False,
                "returncode": 0,
            }

        env = _clean_env()
        try:
            proc = subprocess.run(
                argv,
                cwd=str(workdir),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolError(f"command timed out after {timeout}s: {' '.join(argv)}") from exc

        result = {
            "argv": argv,
            "cwd": str(workdir),
            "executed": True,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-MAX_CAPTURE:],
            "stderr": proc.stderr[-MAX_CAPTURE:],
        }
        if proc.returncode != 0 and not allow_failure:
            raise ToolError(
                f"command failed ({proc.returncode}): {' '.join(argv)}\n{proc.stderr[-2000:]}"
            )
        return result

    return [
        Tool(
            name="shell_run",
            description=(
                "Run a command as an argv list (no shell string). The program must be in the "
                "allowlist; dry-run suppresses mutating commands."
            ),
            input_schema=schema(
                {
                    "argv": {"type": "array", "items": {"type": "string"}},
                    "cwd": {"type": "string"},
                    "timeout_sec": {"type": "integer"},
                    "allow_failure": {"type": "boolean"},
                },
                ["argv"],
            ),
            handler=shell_run,
            permission_note=f"allowlist={sorted(allowlist)}",
            side_effects=True,
        )
    ]


def _clean_env() -> dict[str, str]:
    """Keep secrets out of child processes unless they are explicitly needed."""
    keep = {
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "TMPDIR",
        "SHELL",
        "USER",
        "DEEPSEEK_API_KEY",
        "ANTHROPIC_API_KEY",
        "X_API_BEARER_TOKEN",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "AGENT_HOME",
        "AGENT_REPO_ROOT",
        "AGENT_TASKS_DIR",
        "AGENT_CONFIG",
        "AGENT_CONTEXT",
        "AGENT_RUN_ID",
        "AGENT_RUN_DIR",
        "AGENT_DATE",
        "AGENT_DRY_RUN",
        "AGENT_PUBLISH",
        "AGENT_STEP_ID",
        "AGENT_NOTIFY",
        "AGENT_COMPOSE",
        "AGENT_CONTEXT_STRATEGY",
        "AGENT_MEMORY",
    }
    return {k: v for k, v in os.environ.items() if k in keep}
