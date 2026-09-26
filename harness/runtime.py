"""The deterministic kernel: one run = task + date, executed once, audited fully."""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from . import decisions, memory, paths, taskspec, validators as validators_mod
from .config import AgentConfig, load as load_config
from .errors import HarnessError, LockBusy, StepFailed, ValidationFailed
from .ledger import Ledger, RunRow, now_iso
from .locks import FileLock
from .logutil import logger as make_logger
from .registry import ToolContext
from .tools import build_registry


@dataclass
class RunOptions:
    task: str
    date: str | None = None
    dry_run: bool = False
    force: bool = False
    executor: str | None = None
    publish: bool | None = None
    trigger: str = "manual"
    experiment: str | None = None
    arm: str | None = None
    compose_mode: str | None = None
    context_strategy: str = "full"
    memory_enabled: bool = False
    notify: bool | None = None
    wait_lock: float = 0.0
    skip_steps: list[str] = field(default_factory=list)
    extra_env: dict[str, str] = field(default_factory=dict)
    tags: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunOutcome:
    run: RunRow
    skipped: bool = False
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.run.status in ("success", "degraded")


class Runner:
    def __init__(self, config: AgentConfig | None = None, ledger: Ledger | None = None) -> None:
        paths.ensure_layout()
        self.config = config or load_config()
        self.ledger = ledger or Ledger()
        self.log = make_logger("agent.runtime")

    # -- public API ---------------------------------------------------------
    def run(self, options: RunOptions) -> RunOutcome:
        task = taskspec.load(options.task)
        target_date = options.date or default_date(task)
        executor = options.executor or self.config.default_executor

        stale = self.ledger.mark_stale_running(task.name)
        for run_id in stale:
            self.log.warning("marked stale run %s as crashed", run_id)

        if not options.dry_run and not options.force:
            done = self.ledger.find_done(task.name, target_date)
            if done:
                return RunOutcome(
                    run=done,
                    skipped=True,
                    message=(
                        f"{task.name} {target_date} already finished as {done.status} "
                        f"({done.run_id}); use --force to run again"
                    ),
                )
        elif not options.dry_run and options.force:
            previous = self.ledger.find_done(task.name, target_date)
            if previous:
                self.ledger.supersede(
                    previous.run_id, f"forced re-run for {task.name} {target_date}"
                )
                self.log.info("superseded previous run %s", previous.run_id)

        publish_enabled = (
            bool(options.publish)
            if options.publish is not None
            else bool(task.publish.get("default_enabled", False))
        ) and not options.dry_run

        scopes = list(task.raw.get("lock_scope", ["task"]))
        locks = [FileLock(f"{task.name}.{scope}") for scope in scopes]
        acquired: list[FileLock] = []
        try:
            for lock in locks:
                lock.acquire(wait_sec=options.wait_lock)
                acquired.append(lock)
            return self._execute(
                task=task,
                options=options,
                target_date=target_date,
                executor=executor,
                publish_enabled=publish_enabled,
            )
        finally:
            for lock in reversed(acquired):
                lock.release()

    # -- internals ----------------------------------------------------------
    def _execute(
        self,
        *,
        task: taskspec.TaskSpec,
        options: RunOptions,
        target_date: str,
        executor: str,
        publish_enabled: bool,
    ) -> RunOutcome:
        run_id = self._run_id(task.name, target_date)
        run_dir = paths.runs_dir() / run_id
        (run_dir / "steps").mkdir(parents=True, exist_ok=True)
        log = make_logger(f"agent.run.{task.name}", run_dir=run_dir)

        tool_ctx = ToolContext(
            run_id=run_id,
            run_dir=run_dir,
            task_name=task.name,
            target_date=target_date,
            dry_run=options.dry_run,
            publish=publish_enabled,
            # A task can always read and write its own directory (scripts, prompts)
            # plus the directory this run writes into.
            readable_paths=self._resolve_paths(
                task.readable_paths, task, target_date, run_dir
            )
            + [task.dir],
            writable_paths=self._resolve_paths(
                task.writable_paths, task, target_date, run_dir
            )
            + [run_dir, task.dir],
            data={
                "task_dir": str(task.dir),
                "config_path": str(self.config.path),
                "paths": task.paths_table,
                "accept_content": options.trigger != "experiment",
            },
        )
        context_path = run_dir / "context.json"
        context_path.write_text(
            json.dumps(self._context_payload(task, options, target_date, executor, publish_enabled, tool_ctx, run_dir), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        row = RunRow(
            run_id=run_id,
            task=task.name,
            target_date=target_date,
            status="running",
            executor=executor,
            dry_run=options.dry_run,
            published=False,
            trigger=options.trigger,
            experiment=options.experiment,
            arm=options.arm,
            compose_mode=options.compose_mode or str(task.context.get("compose_default", "auto")),
            context_strategy=options.context_strategy,
            memory_enabled=options.memory_enabled,
            started_at=now_iso(),
            inputs={
                "date": target_date,
                "dry_run": options.dry_run,
                "executor": executor,
                "publish": publish_enabled,
                "compose_mode": options.compose_mode,
                "context_strategy": options.context_strategy,
                "memory": options.memory_enabled,
                **options.tags,
            },
            notes=None,
        )
        self.ledger.insert_run(row)
        log.info(
            "run %s started (task=%s date=%s executor=%s dry_run=%s publish=%s)",
            run_id,
            task.name,
            target_date,
            executor,
            options.dry_run,
            publish_enabled,
        )

        started = time.monotonic()
        steps_done: list[dict[str, Any]] = []
        artifacts: list[str] = []
        degradations: list[str] = []
        metrics: dict[str, Any] = {}
        failure: Exception | None = None
        validation_results: list[dict[str, Any]] = []
        published = False
        deadline = started + float(task.budget.get("run_timeout_sec", self.config.budget.run_timeout_sec))

        immediate = [s for s in task.steps if s.when == "always"]
        deferred = [s for s in task.steps if s.when != "always"]
        skipped_notes: list[str] = []

        try:
            for seq, step in enumerate(immediate, start=1):
                if step.id in options.skip_steps:
                    record = self._skipped_step(step.id, "skipped by --skip-steps")
                    steps_done.append(record)
                    self.ledger.record_step(run_id, record, seq)
                    skipped_notes.append(f"step {step.id} skipped by --skip-steps")
                    continue
                if time.monotonic() > deadline:
                    raise HarnessError(
                        f"run exceeded {task.budget.get('run_timeout_sec', self.config.budget.run_timeout_sec)}s budget"
                    )
                record = self._run_step(
                    step=step,
                    seq=seq,
                    task=task,
                    run_dir=run_dir,
                    target_date=target_date,
                    context_path=context_path,
                    options=options,
                    executor=executor,
                    publish_enabled=publish_enabled,
                    log=log,
                )
                steps_done.append(record)
                self.ledger.record_step(run_id, record, seq)
                artifacts += record.get("artifacts", [])
                degradations += record.get("degradations", [])
                self._merge_metrics(metrics, record.get("metrics", {}))
                if record["status"] == "failed":
                    raise StepFailed(f"step {step.id} failed: {record.get('detail', '')}")

            validation_results = validators_mod.run_all(
                task.validators,
                task_dir=task.dir,
                date=target_date,
                run_dir=run_dir,
                extra=task.vars(date=target_date, run_dir=run_dir),
                logger=log,
                policy=decisions.DecisionPolicy.from_table(task.policy),
            )
            for result in validation_results:
                self.ledger.record_validation(run_id, result)
            # A run-level verdict so the ledger can answer "how often was this
            # run blocked by something we could not verify?" without re-parsing
            # every validator payload.
            metrics["verification"] = decisions.combine(validation_results)
            # A blocked run should leave behind something a human or a repair
            # step can act on: which claim broke, the receipts, and the fix.
            brief = decisions.repair_brief(validation_results)
            if brief["count"]:
                (run_dir / "validation-failures.json").write_text(
                    json.dumps(brief, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                metrics["blocking_validators"] = brief["blocking"]
            required_failed = [
                result
                for result, spec in zip(validation_results, task.validators)
                if spec.required and not result.get("ok")
            ]
            if required_failed:
                degradations.append(
                    "validation failed: " + ", ".join(r["name"] for r in required_failed)
                )

            if not required_failed:
                for seq, step in enumerate(deferred, start=len(immediate) + 1):
                    if step.id in options.skip_steps:
                        record = self._skipped_step(step.id, "skipped by --skip-steps")
                        steps_done.append(record)
                        self.ledger.record_step(run_id, record, seq)
                        skipped_notes.append(f"step {step.id} skipped by --skip-steps")
                        continue
                    if step.requires_publish and not publish_enabled:
                        record = {
                            "id": step.id,
                            "status": "skipped",
                            "detail": "publish disabled (dry-run or --no-publish)",
                            "artifacts": [],
                            "degradations": [],
                            "metrics": {},
                            "duration_ms": 0,
                        }
                    else:
                        record = self._run_step(
                            step=step,
                            seq=seq,
                            task=task,
                            run_dir=run_dir,
                            target_date=target_date,
                            context_path=context_path,
                            options=options,
                            executor=executor,
                            publish_enabled=publish_enabled,
                            log=log,
                        )
                    steps_done.append(record)
                    self.ledger.record_step(run_id, record, seq)
                    artifacts += record.get("artifacts", [])
                    degradations += record.get("degradations", [])
                    self._merge_metrics(metrics, record.get("metrics", {}))
                    if record.get("published"):
                        published = True
                    if record["status"] == "failed":
                        raise StepFailed(f"step {step.id} failed: {record.get('detail', '')}")
        except (HarnessError, LockBusy) as exc:
            failure = exc
        except Exception as exc:  # noqa: BLE001 - must never lose the ledger row
            failure = exc

        if failure is None and any(r.get("ok") is False for r in validation_results):
            required_ok = all(
                result.get("ok")
                for result, spec in zip(validation_results, task.validators)
                if spec.required
            )
            if not required_ok:
                failure = ValidationFailed(
                    "; ".join(
                        f"{r['name']}: {r.get('detail', 'failed')}"
                        for r in validation_results
                        if not r.get("ok")
                    )
                )

        degradations += skipped_notes

        hard_failure = failure is not None
        row.status = (
            "failed" if hard_failure else ("degraded" if degradations else "success")
        )
        row.failure_class = getattr(failure, "failure_class", None) if failure else None
        row.error = str(failure) if failure else None
        row.finished_at = now_iso()
        row.duration_ms = int((time.monotonic() - started) * 1000)
        row.steps = steps_done
        row.validators = validation_results
        row.outputs = {**{f"artifact_{i}": a for i, a in enumerate(artifacts)}, "published": published}
        row.published = published
        row.tool_calls = self.ledger.tool_call_count(run_id)
        row.metrics = {**metrics, "duration_ms": row.duration_ms, "tool_calls": row.tool_calls}
        row.tokens_in = int(metrics.get("tokens_in", 0))
        row.tokens_out = int(metrics.get("tokens_out", 0))
        row.cost_usd = metrics.get("cost_usd")
        row.notes = " | ".join(degradations) if degradations else None
        self.ledger.finish_run(row)

        log.info(
            "run %s finished status=%s published=%s duration=%sms",
            run_id,
            row.status,
            published,
            row.duration_ms,
        )

        self._write_memory(row, task=task, artifacts=artifacts, degradations=degradations)
        if self._should_notify(options, row):
            self._notify(row, task=task, tool_ctx=tool_ctx, log=log)

        return RunOutcome(run=row, skipped=False, message=f"{row.status} ({run_id})")

    # -- step execution -----------------------------------------------------
    def _run_step(
        self,
        *,
        step: taskspec.StepSpec,
        seq: int,
        task: taskspec.TaskSpec,
        run_dir: Path,
        target_date: str,
        context_path: Path,
        options: RunOptions,
        executor: str,
        publish_enabled: bool,
        log: Any,
    ) -> dict[str, Any]:
        started_at = now_iso()
        started = time.monotonic()

        if options.dry_run and step.skip_on_dry_run:
            return {
                "id": step.id,
                "status": "skipped",
                "detail": "skipped on dry-run",
                "artifacts": [],
                "degradations": [],
                "metrics": {},
                "started_at": started_at,
                "finished_at": now_iso(),
                "duration_ms": 0,
            }

        argv = self._resolve_command(step, task=task, target_date=target_date, run_dir=run_dir)
        env = self._step_env(
            task=task,
            options=options,
            target_date=target_date,
            executor=executor,
            publish_enabled=publish_enabled,
            context_path=context_path,
            step_id=step.id,
            run_dir=run_dir,
        )
        log_path = run_dir / "steps" / f"{step.id}.log"
        result_path = run_dir / "steps" / f"{step.id}.result.json"
        if result_path.exists():
            result_path.unlink()

        retries = step.retries if step.retries is not None else self.config.budget.step_retries
        timeout = step.timeout_sec or self.config.budget.step_timeout_sec
        attempt = 0
        exit_code: int | None = None
        stderr_tail = ""
        while True:
            attempt += 1
            log.info("step %s attempt %d: %s", step.id, attempt, " ".join(argv))
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(
                    f"\n=== {now_iso()} attempt {attempt}: {' '.join(argv)} (cwd={task.dir})\n"
                )
                handle.flush()
                try:
                    proc = subprocess.run(
                        argv,
                        cwd=str(task.dir),
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                    )
                    exit_code = proc.returncode
                    stdout, stderr = proc.stdout, proc.stderr
                except subprocess.TimeoutExpired as exc:
                    exit_code = 124
                    stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
                    stderr = f"timeout after {timeout}s"
                handle.write(stdout or "")
                if stderr:
                    handle.write(f"\n--- stderr ---\n{stderr}")
                handle.flush()
            stderr_tail = (stderr or "").strip()[-600:]
            if exit_code == 0 or attempt > retries:
                break
            time.sleep(min(2 ** (attempt - 1), 5))

        payload: dict[str, Any] = {}
        if result_path.is_file():
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {}

        status = payload.get("status")
        if exit_code != 0:
            status = "failed" if not step.allow_degrade else "degraded"
        elif status is None:
            status = "ok"
        if status == "failed" and step.allow_degrade:
            status = "degraded"

        detail = payload.get("notes") or stderr_tail or ("ok" if status == "ok" else status)
        record = {
            "id": step.id,
            "status": status,
            "detail": detail[:1000],
            "artifacts": payload.get("artifacts", []),
            "degradations": payload.get("degradations", []),
            "metrics": payload.get("metrics", {}),
            "exit_code": exit_code,
            "started_at": started_at,
            "finished_at": now_iso(),
            "duration_ms": int((time.monotonic() - started) * 1000),
            "log": str(log_path),
            "published": bool(payload.get("published")),
        }
        if status == "skipped" and exit_code not in (0, None):
            record["status"] = "failed"
        return record

    def _resolve_command(
        self,
        step: taskspec.StepSpec,
        *,
        task: taskspec.TaskSpec,
        target_date: str,
        run_dir: Path,
    ) -> list[str]:
        resolved: list[str] = []
        extra = task.vars(date=target_date, run_dir=run_dir)
        for index, part in enumerate(step.command):
            rendered = taskspec.render(part, date=target_date, run_dir=run_dir, extra=extra)
            if index > 0 and not part.startswith("-") and rendered.endswith((".py", ".sh")):
                candidate = Path(rendered)
                if not candidate.is_absolute():
                    rendered = str(task.dir / rendered)
            resolved.append(rendered)
        return resolved

    @staticmethod
    def _skipped_step(step_id: str, detail: str) -> dict[str, Any]:
        return {
            "id": step_id,
            "status": "skipped",
            "detail": detail,
            "artifacts": [],
            "degradations": [],
            "metrics": {},
            "duration_ms": 0,
        }

    def _step_env(
        self,
        *,
        task: taskspec.TaskSpec,
        options: RunOptions,
        target_date: str,
        executor: str,
        publish_enabled: bool,
        context_path: Path,
        step_id: str,
        run_dir: Path,
    ) -> dict[str, str]:
        env = dict(os.environ)
        env.update(
            {
                "AGENT_CONTEXT": str(context_path),
                "AGENT_RUN_DIR": str(run_dir),
                "AGENT_TASK": task.name,
                "AGENT_DATE": target_date,
                "AGENT_DRY_RUN": "1" if options.dry_run else "0",
                "AGENT_PUBLISH": "1" if publish_enabled else "0",
                "AGENT_EXECUTOR": executor,
                "AGENT_STEP_ID": step_id,
                "AGENT_COMPOSE": options.compose_mode or "",
                "AGENT_CONTEXT_STRATEGY": options.context_strategy,
                "AGENT_MEMORY": "on" if options.memory_enabled else "off",
                "AGENT_HOME": str(paths.home()),
                "AGENT_REPO_ROOT": str(paths.repo_root()),
                "PYTHONPATH": _prepend_pythonpath(str(paths.repo_root())),
            }
        )
        env.update(options.extra_env)
        return env

    def _context_payload(
        self,
        task: taskspec.TaskSpec,
        options: RunOptions,
        target_date: str,
        executor: str,
        publish_enabled: bool,
        tool_ctx: ToolContext,
        run_dir: Path,
    ) -> dict[str, Any]:
        return {
            "run": {
                "run_id": tool_ctx.run_id,
                "run_dir": str(run_dir),
                "target_date": target_date,
                "dry_run": options.dry_run,
                "publish": publish_enabled,
                "executor": executor,
                "compose_mode": options.compose_mode or str(task.context.get("compose_default", "auto")),
                "context_strategy": options.context_strategy,
                "memory_enabled": options.memory_enabled,
                "trigger": options.trigger,
                "experiment": options.experiment,
                "arm": options.arm,
            },
            "task": {
                "name": task.name,
                "dir": str(task.dir),
                "allowed_tools": task.allowed_tool_names(),
                "raw": task.raw,
            },
            "tool_ctx": tool_ctx.to_json(),
            "agent_config_path": str(self.config.path),
            "env": options.extra_env,
        }

    def _resolve_paths(
        self,
        entries: list[str],
        task: taskspec.TaskSpec,
        target_date: str,
        run_dir: Path,
    ) -> list[Path]:
        resolved = []
        extra = task.vars(date=target_date, run_dir=run_dir)
        for entry in entries:
            rendered = taskspec.render(entry, date=target_date, run_dir=run_dir, extra=extra)
            candidate = Path(rendered).expanduser()
            resolved.append(candidate if candidate.is_absolute() else (task.dir / rendered))
        return resolved

    def _merge_metrics(self, metrics: dict[str, Any], new: dict[str, Any]) -> None:
        for key, value in (new or {}).items():
            if key in ("tokens_in", "tokens_out") and isinstance(value, (int, float)):
                metrics[key] = int(metrics.get(key, 0)) + int(value)
            elif key == "cost_usd" and isinstance(value, (int, float)):
                metrics[key] = round(float(metrics.get(key, 0.0)) + float(value), 6)
            else:
                metrics[key] = value

    def _run_id(self, task: str, target_date: str) -> str:
        stamp = dt.datetime.now().strftime("%H%M%S")
        return f"{target_date}-{task}-{stamp}-{uuid.uuid4().hex[:6]}"

    def _write_memory(
        self,
        row: RunRow,
        *,
        task: taskspec.TaskSpec,
        artifacts: list[str],
        degradations: list[str],
    ) -> None:
        if row.status == "failed":
            summary = f"运行失败：{row.error or '未提供原因'}"
        else:
            validator_line = ", ".join(
                f"{v['name']}={'ok' if v.get('ok') else 'fail'}" for v in row.validators
            )
            summary = (
                f"任务 {task.name} 在 {row.target_date} 以 {row.status} 结束，"
                f"执行器 {row.executor}，耗时 {row.duration_ms} ms。"
                + (f" 验证：{validator_line}。" if validator_line else "")
            )
        try:
            memory.write_run_summary(
                run_id=row.run_id,
                task=row.task,
                target_date=row.target_date,
                status=row.status,
                summary=summary,
                metrics={**row.metrics, "duration_ms": row.duration_ms, "tool_calls": row.tool_calls},
                artifacts=artifacts,
                degradations=degradations,
                extra_meta={
                    "executor": row.executor,
                    "experiment": row.experiment or "",
                    "arm": row.arm or "",
                },
            )
        except Exception as exc:  # memory must never break a run
            self.log.warning("could not write memory summary: %s", exc)

    def _should_notify(self, options: RunOptions, row: RunRow) -> bool:
        if options.notify is False or not self.config.notify.enabled:
            return False
        mode = self.config.notify.on
        if options.notify is True:
            return True
        if mode == "always":
            return True
        if mode == "failure":
            return row.status in ("failed", "crashed")
        return False

    def _notify(self, row: RunRow, *, task: taskspec.TaskSpec, tool_ctx: ToolContext, log: Any) -> None:
        if "notify" not in task.allowed_tool_names():
            log.warning("task %s does not allow the notify tool; skipping notification", task.name)
            return
        registry = build_registry(tool_ctx, config=self.config, ledger=self.ledger, logger=log)
        title = f"{task.name} · {row.status}"
        artifacts = [v for v in row.outputs.values() if isinstance(v, str)]
        message = row.error or f"产物 {len(artifacts)} 项；耗时 {row.duration_ms} ms"
        try:
            registry.call("notify", {"title": title, "message": message[:400]})
        except Exception as exc:  # notification is best-effort
            log.warning("notification failed: %s", exc)


def default_date(task: taskspec.TaskSpec, *, now: dt.datetime | None = None) -> str:
    tz = ZoneInfo(task.timezone)
    current = (now or dt.datetime.now(tz)).astimezone(tz)
    if task.date_mode == "yesterday":
        current -= dt.timedelta(days=1)
    return current.date().isoformat()


def _prepend_pythonpath(value: str) -> str:
    existing = os.environ.get("PYTHONPATH", "")
    return f"{value}:{existing}" if existing else value
