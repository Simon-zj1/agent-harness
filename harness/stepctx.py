"""Step-script API: steps are subprocesses, so they get a serialised context."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import AgentConfig, load as load_config
from .registry import Registry, ToolContext
from .taskspec import render
from .tools import build_registry


@dataclass
class StepContext:
    run_id: str
    run_dir: Path
    task_name: str
    task_dir: Path
    target_date: str
    step_id: str
    dry_run: bool
    publish: bool
    executor: str
    compose_mode: str
    context_strategy: str
    memory_enabled: bool
    trigger: str
    experiment: str | None
    arm: str | None
    config: AgentConfig
    tool_ctx: ToolContext
    registry: Registry
    logger: Any
    env: dict[str, str]
    task_raw: dict[str, Any]

    def path(self, template: str) -> Path:
        rendered = render(
            template,
            date=self.target_date,
            run_dir=self.run_dir,
            extra=self.tool_ctx.data.get("paths", {}),
        )
        candidate = Path(rendered).expanduser()
        return candidate if candidate.is_absolute() else (self.task_dir / rendered)

    def artifact(self, path: Path | str) -> str:
        return str(Path(path))


def load() -> StepContext:
    raw_context = os.environ.get("AGENT_CONTEXT")
    if not raw_context:
        raise SystemExit("AGENT_CONTEXT is not set: run this step through `./agent run`")
    payload = json.loads(Path(raw_context).read_text(encoding="utf-8"))
    config = load_config(Path(payload["agent_config_path"]) if payload.get("agent_config_path") else None)
    tool_ctx = ToolContext.from_json(payload["tool_ctx"])
    run = payload["run"]
    task = payload["task"]

    run_dir = Path(run["run_dir"])
    registry = build_registry(
        tool_ctx,
        config=config,
        ledger=_step_ledger(),
        logger=_step_logger(run_dir, task["name"]),
        sandboxed=True,
    )
    registry.allow(task.get("allowed_tools") or [])

    return StepContext(
        run_id=run["run_id"],
        run_dir=run_dir,
        task_name=task["name"],
        task_dir=Path(task["dir"]),
        target_date=run["target_date"],
        step_id=os.environ.get("AGENT_STEP_ID", ""),
        dry_run=bool(run["dry_run"]),
        publish=bool(run["publish"]),
        executor=run.get("executor", "builtin"),
        compose_mode=run.get("compose_mode", "auto"),
        context_strategy=run.get("context_strategy", "full"),
        memory_enabled=bool(run.get("memory_enabled", False)),
        trigger=run.get("trigger", "manual"),
        experiment=run.get("experiment"),
        arm=run.get("arm"),
        config=config,
        tool_ctx=tool_ctx,
        registry=registry,
        logger=_step_logger(run_dir, task["name"]),
        env=payload.get("env", {}),
        task_raw=task.get("raw", {}),
    )


def _step_logger(run_dir: Path, task: str) -> Any:
    from .logutil import logger

    return logger(f"agent.step.{task}", run_dir=run_dir)


def _step_ledger() -> Any:
    """Steps audit their tool calls into the same runs.db the kernel uses."""
    from .ledger import Ledger

    return Ledger()


def finish(
    ctx: StepContext,
    status: str,
    *,
    artifacts: list[str] | None = None,
    metrics: dict[str, Any] | None = None,
    notes: str = "",
    degradations: list[str] | None = None,
) -> int:
    """Write step result and exit with a status the runner understands."""
    payload = {
        "step_id": ctx.step_id,
        "status": status,
        "artifacts": artifacts or [],
        "metrics": metrics or {},
        "notes": notes,
        "degradations": degradations or [],
    }
    result_path = ctx.run_dir / "steps" / f"{ctx.step_id}.result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"step": ctx.step_id, "status": status, **payload.get("metrics", {})}, ensure_ascii=False))
    if status == "failed":
        return 1
    return 0


def fail(ctx: StepContext, message: str, *, metrics: dict[str, Any] | None = None) -> int:
    print(f"[error] {message}", file=sys.stderr)
    return finish(ctx, "failed", metrics=metrics, notes=message)


def degrade(
    ctx: StepContext,
    message: str,
    *,
    artifacts: list[str] | None = None,
    metrics: dict[str, Any] | None = None,
) -> int:
    print(f"[degraded] {message}", file=sys.stderr)
    return finish(
        ctx,
        "degraded",
        artifacts=artifacts,
        metrics=metrics,
        notes=message,
        degradations=[message],
    )


def default_date(ctx: StepContext) -> str:
    return ctx.target_date


def memory_block(ctx: StepContext, *, task: str | None = None) -> str:
    if not ctx.memory_enabled:
        return ""
    from . import memory

    return memory.context_for_prompt(task=task or ctx.task_name)
