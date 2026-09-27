"""Shell tool: argv-only, allowlisted, audited.

Shell commands are the widest primitive in the task vocabulary, so they are
also the narrowest when it comes to write permission.  A command that *can*
write must declare the paths it is allowed to write before it runs; read-only
commands such as ``git status`` stay read-only.
"""

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
        writes: list[str] | None = None,
        env: dict[str, str] | None = None,
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
        may_write = _command_may_write(program, argv, config.runtime.get("mutating_commands", []))
        if sandboxed and may_write:
            if not writes:
                raise PermissionDenied(
                    f"command {program!r} may write but shell_run declares no writes; "
                    "add explicit writes=[...]"
                )
            declared_writes = [
                _resolve_write_target(raw, workdir) for raw in writes
            ]
            for raw in writes:
                target = _resolve_write_target(raw, workdir)
                if not ctx_.within_writable(target):
                    raise PermissionDenied(f"write outside declared paths: {target}")
            _validate_path_args(program, argv, workdir, ctx_, declared_writes)

        if ctx_.dry_run and may_write:
            return {
                "argv": argv,
                "cwd": str(workdir),
                "writes": [str(_resolve_write_target(raw, workdir)) for raw in (writes or [])],
                "dry_run": True,
                "executed": False,
                "returncode": 0,
            }

        child_env = _clean_env(env)
        try:
            proc = subprocess.run(
                argv,
                cwd=str(workdir),
                env=child_env,
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
                    "writes": {"type": "array", "items": {"type": "string"}},
                    "env": {"type": "object"},
                },
                ["argv"],
            ),
            handler=shell_run,
            permission_note=f"allowlist={sorted(allowlist)}",
            side_effects=True,
        )
    ]


def _clean_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Keep secrets out of child processes.

    Provider credentials are read by the provider layer, not inherited by
    arbitrary shell commands.  If a command genuinely needs a credential, the
    caller must pass it as an explicit argv/env argument and accept that it is
    a narrower contract than the parent process's environment.
    """
    keep = {
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "TMPDIR",
        "SHELL",
        "USER",
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
    cleaned = {k: v for k, v in os.environ.items() if k in keep}
    cleaned.update({str(k): str(v) for k, v in (extra or {}).items()})
    return cleaned


_READ_ONLY_GIT = {
    "status",
    "log",
    "show",
    "rev-parse",
    "diff",
    "grep",
    "ls-files",
}


def _resolve_write_target(raw: str, workdir: Path) -> Path:
    target = Path(raw).expanduser()
    return target if target.is_absolute() else (workdir / target)


def _command_may_write(
    program: str,
    argv: list[str],
    mutating_commands: list[str] | set[str],
) -> bool:
    """Return whether an argv-only command can mutate the filesystem.

    This is deliberately conservative for interpreters: ``python3`` is treated
    as a writer because a script can write anywhere.  For ``git`` and ``curl``
    the answer depends on the subcommand/flags so ``git status`` and plain GET
    requests do not need write declarations.
    """
    if program == "git":
        subcommand = _git_subcommand(argv)
        return subcommand not in _READ_ONLY_GIT
    if program == "curl":
        return _curl_may_write(argv)
    if program in {"python3", "python", "codex", "claude"}:
        return True
    if program in {"osascript", "launchctl", "plutil", "rm", "cp", "mkdir", "touch", "mv", "rsync"}:
        return True
    if program == "sed" and any(arg == "-i" or (arg.startswith("-i") and arg != "-i") for arg in argv):
        return True
    return program in mutating_commands


def _git_subcommand(argv: list[str]) -> str:
    """Find the first git subcommand after global options."""
    skip_value = {"-C", "-c", "--git-dir", "--work-tree"}
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg in skip_value:
            i += 2
            continue
        if arg.startswith("-"):
            i += 1
            continue
        return arg
    return ""


def _curl_may_write(argv: list[str]) -> bool:
    """Classify curl by its request method/output flags."""
    output_flags = {"-o", "--output", "-O", "--remote-name", "-T", "--upload-file"}
    data_flags = {
        "-d",
        "--data",
        "--data-raw",
        "--data-binary",
        "--data-urlencode",
        "-F",
        "--form",
    }
    method = "GET"
    for index, arg in enumerate(argv):
        if arg in output_flags or arg in data_flags:
            return True
        if arg in {"-X", "--request"}:
            if index + 1 < len(argv):
                method = argv[index + 1].upper()
    return method not in {"GET", "HEAD"}


def _validate_path_args(
    program: str,
    argv: list[str],
    workdir: Path,
    ctx: ToolContext,
    declared_writes: list[Path],
) -> None:
    """Check obvious filesystem arguments against the declared write scope.

    This is intentionally a defense-in-depth check for the common destructive
    commands, not an OS sandbox.  Interpreters and external executors still
    require a narrower execution boundary if they are exposed to untrusted
    content.
    """
    args = [arg for arg in argv[1:] if not arg.startswith("-")]
    if not args:
        return
    if program in {"rm", "mkdir", "touch", "mv"}:
        for arg in args:
            target = _resolve_write_target(arg, workdir)
            if not any(_is_within(target, root) for root in declared_writes):
                raise PermissionDenied(
                    f"{program} path is outside declared writes: {target}"
                )
        return
    if program == "cp" and len(args) >= 2:
        source = _resolve_write_target(args[0], workdir)
        destination = _resolve_write_target(args[-1], workdir)
        if not ctx.within_readable(source):
            raise PermissionDenied(f"cp source is outside declared reads: {source}")
        if not any(_is_within(destination, root) for root in declared_writes):
            raise PermissionDenied(
                f"cp destination is outside declared writes: {destination}"
            )
        return
    if program == "sed" and any(arg == "-i" or (arg.startswith("-i") and arg != "-i") for arg in argv):
        for arg in args[1:]:
            target = _resolve_write_target(arg, workdir)
            if not any(_is_within(target, root) for root in declared_writes):
                raise PermissionDenied(
                    f"sed -i path is outside declared writes: {target}"
                )


def _is_within(target: Path, root: Path) -> bool:
    try:
        target.expanduser().resolve().relative_to(root.expanduser().resolve())
        return True
    except ValueError:
        return False
