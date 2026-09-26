"""Run ledger (runs/runs.db) — the single source of truth for evals.

Every run, step and tool call is recorded here so experiments can compare
success rate, artifact validity, cost and wall-clock time.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
import atexit
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import paths

_OPEN: list["Ledger"] = []


@atexit.register
def _close_open_ledgers() -> None:
    for ledger in list(_OPEN):
        try:
            ledger.close()
        except Exception:  # noqa: BLE001 - shutdown must not raise
            pass

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    task TEXT NOT NULL,
    target_date TEXT NOT NULL,
    executor TEXT NOT NULL DEFAULT 'builtin',
    status TEXT NOT NULL,
    dry_run INTEGER NOT NULL DEFAULT 0,
    published INTEGER NOT NULL DEFAULT 0,
    trigger TEXT NOT NULL DEFAULT 'manual',
    experiment TEXT,
    arm TEXT,
    compose_mode TEXT,
    context_strategy TEXT,
    memory_enabled INTEGER,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    duration_ms INTEGER,
    pid INTEGER,
    host TEXT,
    inputs_json TEXT,
    outputs_json TEXT,
    validators_json TEXT,
    metrics_json TEXT,
    steps_json TEXT,
    tokens_in INTEGER DEFAULT 0,
    tokens_out INTEGER DEFAULT 0,
    cost_usd REAL,
    cost_known INTEGER NOT NULL DEFAULT 0,
    failure_class TEXT,
    error TEXT,
    tool_calls INTEGER NOT NULL DEFAULT 0,
    human_intervention INTEGER NOT NULL DEFAULT 0,
    notes TEXT
);

-- One successful non-dry run per task+date is what makes reruns idempotent.
CREATE UNIQUE INDEX IF NOT EXISTS idx_runs_done
    ON runs(task, target_date)
    WHERE status IN ('success', 'degraded') AND dry_run = 0;

CREATE INDEX IF NOT EXISTS idx_runs_task_date ON runs(task, target_date);
CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs(experiment, arm);

CREATE TABLE IF NOT EXISTS steps (
    run_id TEXT NOT NULL,
    step_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    duration_ms INTEGER,
    exit_code INTEGER,
    degraded INTEGER NOT NULL DEFAULT 0,
    detail TEXT,
    PRIMARY KEY (run_id, step_id)
);

CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    tool TEXT NOT NULL,
    args_json TEXT,
    ok INTEGER NOT NULL DEFAULT 0,
    duration_ms INTEGER,
    error TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS validations (
    run_id TEXT NOT NULL,
    name TEXT NOT NULL,
    ok INTEGER NOT NULL,
    detail TEXT,
    metrics_json TEXT,
    failures_json TEXT,
    -- Typed-decision columns. `ok` alone cannot distinguish "verified" from
    -- "could not verify", which is exactly the gap that let fabricated URLs
    -- through. decision/failure_class make the uncertainty countable.
    decision TEXT,
    failure_class TEXT,
    evidence_json TEXT,
    policy_action TEXT,
    PRIMARY KEY (run_id, name)
);
"""

# Columns added after the first release; existing ledgers are migrated in place.
_VALIDATION_MIGRATIONS: tuple[tuple[str, str], ...] = (
    ("decision", "TEXT"),
    ("failure_class", "TEXT"),
    ("evidence_json", "TEXT"),
    ("policy_action", "TEXT"),
)


@dataclass
class RunRow:
    run_id: str
    task: str
    target_date: str
    status: str
    executor: str = "builtin"
    dry_run: bool = False
    published: bool = False
    trigger: str = "manual"
    experiment: str | None = None
    arm: str | None = None
    compose_mode: str | None = None
    context_strategy: str | None = None
    memory_enabled: bool | None = None
    started_at: str = ""
    finished_at: str | None = None
    duration_ms: int | None = None
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    validators: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    steps: list[dict[str, Any]] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float | None = None
    failure_class: str | None = None
    error: str | None = None
    tool_calls: int = 0
    human_intervention: bool = False
    notes: str | None = None
    pid: int | None = None


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


class Ledger:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or paths.ledger_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, timeout=10)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=10000")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)
        self._migrate()
        self._conn.commit()
        _OPEN.append(self)

    def _migrate(self) -> None:
        """Add columns introduced after a ledger was first created."""
        existing = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(validations)")
        }
        for column, column_type in _VALIDATION_MIGRATIONS:
            if column not in existing:
                self._conn.execute(
                    f"ALTER TABLE validations ADD COLUMN {column} {column_type}"
                )

    # -- runs ---------------------------------------------------------------
    def insert_run(self, row: RunRow) -> None:
        self._conn.execute(
            """
            INSERT INTO runs (
                run_id, task, target_date, executor, status, dry_run, published,
                trigger, experiment, arm, compose_mode, context_strategy, memory_enabled,
                started_at, pid, host, inputs_json, steps_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                row.run_id,
                row.task,
                row.target_date,
                row.executor,
                row.status,
                int(row.dry_run),
                int(row.published),
                row.trigger,
                row.experiment,
                row.arm,
                row.compose_mode,
                row.context_strategy,
                None if row.memory_enabled is None else int(row.memory_enabled),
                row.started_at or now_iso(),
                row.pid if row.pid is not None else os.getpid(),
                os.uname().nodename,
                json.dumps(row.inputs, ensure_ascii=False),
                json.dumps(row.steps, ensure_ascii=False),
            ),
        )
        self._conn.commit()

    def finish_run(self, row: RunRow) -> None:
        self._conn.execute(
            """
            UPDATE runs SET
                status=?, published=?, finished_at=?, duration_ms=?, outputs_json=?,
                validators_json=?, metrics_json=?, steps_json=?, tokens_in=?, tokens_out=?,
                cost_usd=?, cost_known=?, failure_class=?, error=?, tool_calls=?,
                human_intervention=?, notes=?
            WHERE run_id=?
            """,
            (
                row.status,
                int(row.published),
                row.finished_at or now_iso(),
                row.duration_ms,
                json.dumps(row.outputs, ensure_ascii=False),
                json.dumps(row.validators, ensure_ascii=False),
                json.dumps(row.metrics, ensure_ascii=False),
                json.dumps(row.steps, ensure_ascii=False),
                row.tokens_in,
                row.tokens_out,
                row.cost_usd,
                int(row.cost_usd is not None),
                row.failure_class,
                row.error,
                row.tool_calls,
                int(row.human_intervention),
                row.notes,
                row.run_id,
            ),
        )
        self._conn.commit()

    def get_run(self, run_id: str) -> RunRow | None:
        cur = self._conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,))
        row = cur.fetchone()
        return _to_run_row(row) if row else None

    def find_done(self, task: str, target_date: str) -> RunRow | None:
        cur = self._conn.execute(
            """
            SELECT * FROM runs
            WHERE task=? AND target_date=? AND dry_run=0
              AND status IN ('success','degraded')
            ORDER BY finished_at DESC LIMIT 1
            """,
            (task, target_date),
        )
        row = cur.fetchone()
        return _to_run_row(row) if row else None

    def supersede(self, run_id: str, reason: str) -> None:
        """Retire an earlier result so a forced re-run can take its place."""
        self._conn.execute(
            "UPDATE runs SET status='superseded', "
            "notes=COALESCE(notes || '\n', '') || ? WHERE run_id=?",
            (f"superseded: {reason}", run_id),
        )
        self._conn.commit()

    def list_runs(
        self,
        *,
        task: str | None = None,
        limit: int = 20,
        experiment: str | None = None,
        experiment_prefix: str | None = None,
    ) -> list[RunRow]:
        clauses, params = [], []
        if task:
            clauses.append("task=?")
            params.append(task)
        if experiment:
            clauses.append("experiment=?")
            params.append(experiment)
        if experiment_prefix:
            clauses.append("experiment LIKE ?")
            params.append(f"{experiment_prefix}%")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        cur = self._conn.execute(
            f"SELECT * FROM runs {where} ORDER BY started_at DESC LIMIT ?",
            (*params, limit),
        )
        return [_to_run_row(r) for r in cur.fetchall()]

    def annotate(
        self,
        run_id: str,
        *,
        intervened: bool | None = None,
        notes: str | None = None,
    ) -> None:
        if intervened is not None:
            self._conn.execute(
                "UPDATE runs SET human_intervention=? WHERE run_id=?",
                (int(intervened), run_id),
            )
        if notes is not None:
            self._conn.execute(
                "UPDATE runs SET notes=COALESCE(notes || '\n', '') || ? WHERE run_id=?",
                (notes, run_id),
            )
        self._conn.commit()

    def tool_call_count(self, run_id: str) -> int:
        cur = self._conn.execute(
            "SELECT COUNT(*) AS n FROM tool_calls WHERE run_id=?", (run_id,)
        )
        row = cur.fetchone()
        return int(row["n"]) if row else 0

    def mark_stale_running(self, task: str | None = None) -> list[str]:
        """Mark runs whose process died as crashed so they stop blocking reruns."""
        clauses = ["status='running'"]
        params: list[Any] = []
        if task:
            clauses.append("task=?")
            params.append(task)
        cur = self._conn.execute(
            f"SELECT run_id, pid FROM runs WHERE {' AND '.join(clauses)}", params
        )
        stale: list[str] = []
        for row in cur.fetchall():
            pid = row["pid"]
            if pid and _pid_alive(int(pid)):
                continue
            stale.append(row["run_id"])
        for run_id in stale:
            self._conn.execute(
                "UPDATE runs SET status='crashed', failure_class='stale_process', "
                "finished_at=?, error=COALESCE(error,'process disappeared') WHERE run_id=?",
                (now_iso(), run_id),
            )
        self._conn.commit()
        return stale

    # -- steps / tools / validations ---------------------------------------
    def record_step(self, run_id: str, step: dict[str, Any], seq: int) -> None:
        self._conn.execute(
            """
            INSERT INTO steps (run_id, step_id, seq, status, started_at, finished_at,
                               duration_ms, exit_code, degraded, detail)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(run_id, step_id) DO UPDATE SET
                status=excluded.status, started_at=excluded.started_at,
                finished_at=excluded.finished_at, duration_ms=excluded.duration_ms,
                exit_code=excluded.exit_code, degraded=excluded.degraded,
                detail=excluded.detail
            """,
            (
                run_id,
                step.get("id", "?"),
                seq,
                step.get("status", "unknown"),
                step.get("started_at"),
                step.get("finished_at"),
                step.get("duration_ms"),
                step.get("exit_code"),
                int(bool(step.get("degraded"))),
                step.get("detail"),
            ),
        )
        self._conn.commit()

    def record_tool_call(
        self,
        run_id: str,
        seq: int,
        tool: str,
        args: dict[str, Any],
        ok: bool,
        duration_ms: int,
        error: str | None = None,
    ) -> None:
        safe_args = {k: v for k, v in args.items() if not _looks_secret(k)}
        self._conn.execute(
            """
            INSERT INTO tool_calls (run_id, seq, tool, args_json, ok, duration_ms, error, created_at)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                run_id,
                seq,
                tool,
                json.dumps(safe_args, ensure_ascii=False, default=str),
                int(ok),
                duration_ms,
                error,
                now_iso(),
            ),
        )
        self._conn.execute(
            "UPDATE runs SET tool_calls=tool_calls+1 WHERE run_id=?", (run_id,)
        )
        self._conn.commit()

    def record_validation(self, run_id: str, result: dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT INTO validations (
                run_id, name, ok, detail, metrics_json, failures_json,
                decision, failure_class, evidence_json, policy_action
            )
            VALUES (?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(run_id, name) DO UPDATE SET
                ok=excluded.ok, detail=excluded.detail,
                metrics_json=excluded.metrics_json, failures_json=excluded.failures_json,
                decision=excluded.decision, failure_class=excluded.failure_class,
                evidence_json=excluded.evidence_json, policy_action=excluded.policy_action
            """,
            (
                run_id,
                result.get("name", "?"),
                int(bool(result.get("ok"))),
                result.get("detail"),
                json.dumps(result.get("metrics", {}), ensure_ascii=False),
                json.dumps(result.get("failures", []), ensure_ascii=False),
                result.get("decision"),
                result.get("failure_class"),
                json.dumps(result.get("evidence", []), ensure_ascii=False),
                result.get("policy_action"),
            ),
        )
        self._conn.commit()

    # -- reporting ---------------------------------------------------------
    def totals(self, task: str | None = None) -> dict[str, Any]:
        clauses, params = [], []
        if task:
            clauses.append("task=?")
            params.append(task)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        cur = self._conn.execute(
            f"""
            SELECT COUNT(*) AS runs,
                   SUM(CASE WHEN status IN ('success','degraded') THEN 1 ELSE 0 END) AS ok,
                   SUM(CASE WHEN status IN ('failed','crashed') THEN 1 ELSE 0 END) AS failed,
                   SUM(tokens_in) AS tokens_in, SUM(tokens_out) AS tokens_out,
                   SUM(duration_ms) AS duration_ms,
                   SUM(human_intervention) AS interventions
            FROM runs {where}
            """,
            params,
        )
        row = cur.fetchone()
        return dict(row) if row else {}

    def close(self) -> None:
        if self in _OPEN:
            _OPEN.remove(self)
        self._conn.close()


def _looks_secret(key: str) -> bool:
    lowered = key.lower()
    return any(token in lowered for token in ("key", "token", "secret", "password", "authorization"))


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _to_run_row(row: sqlite3.Row) -> RunRow:
    def load(name: str, default: Any) -> Any:
        raw = row[name]
        if raw in (None, ""):
            return default
        return json.loads(raw)

    memory_enabled = row["memory_enabled"]
    return RunRow(
        run_id=row["run_id"],
        task=row["task"],
        target_date=row["target_date"],
        status=row["status"],
        executor=row["executor"],
        dry_run=bool(row["dry_run"]),
        published=bool(row["published"]),
        trigger=row["trigger"],
        experiment=row["experiment"],
        arm=row["arm"],
        compose_mode=row["compose_mode"],
        context_strategy=row["context_strategy"],
        memory_enabled=None if memory_enabled is None else bool(memory_enabled),
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        duration_ms=row["duration_ms"],
        inputs=load("inputs_json", {}),
        outputs=load("outputs_json", {}),
        validators=load("validators_json", []),
        metrics=load("metrics_json", {}),
        steps=load("steps_json", []),
        tokens_in=row["tokens_in"] or 0,
        tokens_out=row["tokens_out"] or 0,
        cost_usd=row["cost_usd"],
        failure_class=row["failure_class"],
        error=row["error"],
        tool_calls=row["tool_calls"] or 0,
        human_intervention=bool(row["human_intervention"]),
        notes=row["notes"],
        pid=row["pid"],
    )
