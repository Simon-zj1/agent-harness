"""File-first memory with a SQLite index.

Two kinds of entries, deliberately separated:
  * runs/   — factual, auto-written summaries of what actually happened
  * notes/  — long-term conclusions, only ever written on purpose (human-confirmed)
"""

from __future__ import annotations

import datetime as dt
import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from . import paths
from .errors import HarnessError


@dataclass
class MemoryEntry:
    path: Path
    kind: str
    title: str
    date: str = ""
    task: str = ""
    tags: list[str] = field(default_factory=list)
    source: str = ""
    body: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "kind": self.kind,
            "title": self.title,
            "date": self.date,
            "task": self.task,
            "tags": self.tags,
            "source": self.source,
        }


def slugify(text: str, *, max_len: int = 60) -> str:
    ascii_text = re.sub(r"[^\w\u4e00-\u9fff-]+", "-", text.strip(), flags=re.UNICODE)
    ascii_text = re.sub(r"-{2,}", "-", ascii_text).strip("-")
    return (ascii_text or "entry")[:max_len]


def _front_matter(meta: dict[str, Any]) -> str:
    lines = ["---"]
    for key, value in meta.items():
        if isinstance(value, list):
            rendered = ", ".join(str(v) for v in value)
        else:
            rendered = str(value)
        lines.append(f"{key}: {rendered}")
    lines.append("---")
    return "\n".join(lines) + "\n"


def parse_entry(path: Path) -> MemoryEntry | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    meta: dict[str, str] = {}
    body = text
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            header = text[4:end]
            body = text[end + 4 :].lstrip("\n")
            for line in header.splitlines():
                if ":" in line:
                    key, _, value = line.partition(":")
                    meta[key.strip()] = value.strip()
    kind = path.parent.name if path.parent.name in ("runs", "notes") else "other"
    tags = [t.strip() for t in meta.get("tags", "").split(",") if t.strip()]
    title = meta.get("title") or _first_heading(body) or path.stem
    return MemoryEntry(
        path=path,
        kind=kind,
        title=title,
        date=meta.get("date", ""),
        task=meta.get("task", ""),
        tags=tags,
        source=meta.get("source", ""),
        body=body,
    )


def _first_heading(body: str) -> str:
    for line in body.splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return ""


def iter_entries() -> Iterable[MemoryEntry]:
    for sub in ("runs", "notes"):
        base = paths.memory_dir() / sub
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.md")):
            entry = parse_entry(path)
            if entry:
                yield entry


def write_run_summary(
    *,
    run_id: str,
    task: str,
    target_date: str,
    status: str,
    summary: str,
    metrics: dict[str, Any] | None = None,
    artifacts: list[str] | None = None,
    degradations: list[str] | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    """Auto-written, factual record of a run (never conclusions)."""
    base = paths.memory_dir() / "runs"
    base.mkdir(parents=True, exist_ok=True)
    meta = {
        "date": target_date,
        "task": task,
        "run_id": run_id,
        "status": status,
        "tags": [task, status],
    }
    meta.update(extra_meta or {})
    lines = [
        _front_matter(meta),
        f"# {task} · {target_date} · {status}",
        "",
        summary.strip(),
        "",
    ]
    if artifacts:
        lines.append("## Artifacts")
        lines += [f"- {item}" for item in artifacts]
        lines.append("")
    if degradations:
        lines.append("## Degradations")
        lines += [f"- {item}" for item in degradations]
        lines.append("")
    if metrics:
        lines.append("## Metrics")
        for key in sorted(metrics):
            lines.append(f"- {key}: {metrics[key]}")
        lines.append("")
    # The run id already carries date + task, so keep the filename short.
    path = base / f"{run_id}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    reindex()
    return path


def add_note(
    title: str,
    body: str,
    *,
    tags: list[str] | None = None,
    source: str = "",
    date: str | None = None,
    task: str = "",
) -> Path:
    """Long-term memory. Only written when a human asks for it."""
    base = paths.memory_dir() / "notes"
    base.mkdir(parents=True, exist_ok=True)
    stamp = date or dt.date.today().isoformat()
    meta = {"date": stamp, "title": title, "tags": tags or [], "source": source, "task": task}
    content = "\n".join([_front_matter(meta), f"# {title}", "", body.strip(), ""])
    path = base / f"{stamp}-{slugify(title)}.md"
    if path.exists():
        path = base / f"{stamp}-{slugify(title)}-{dt.datetime.now():%H%M%S}.md"
    path.write_text(content, encoding="utf-8")
    reindex()
    return path


def promote_run(
    run_id: str,
    *,
    ledger: Any,
    title: str,
    body: str | None = None,
    tags: list[str] | None = None,
) -> Path:
    run = ledger.get_run(run_id)
    if run is None:
        raise HarnessError(f"unknown run {run_id}")
    text = body or "\n".join(
        [
            f"- 任务：{run.task}（{run.target_date}）",
            f"- 结果：{run.status}"
            + (f"（{run.failure_class}）" if run.failure_class else ""),
            f"- 产物：{', '.join(str(v) for v in run.outputs.values()) or '无'}",
            f"- 验证：{_validators_line(run.validators)}",
            f"- 工具调用：{run.tool_calls}，tokens：{run.tokens_in}/{run.tokens_out}",
            f"- 备注：{run.notes or '无'}",
        ]
    )
    return add_note(
        title,
        text,
        tags=tags or [run.task, "promoted"],
        source=f"run:{run_id}",
        date=run.target_date,
        task=run.task,
    )


def _validators_line(validators: list[dict[str, Any]]) -> str:
    if not validators:
        return "未运行"
    return "; ".join(
        f"{v.get('name')}={'ok' if v.get('ok') else 'fail'}" for v in validators
    )


# -- index -----------------------------------------------------------------
BUSY_TIMEOUT_MS = 5000
_fts_state: dict[str, bool] = {}


def _connect() -> sqlite3.Connection:
    path = paths.memory_index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=BUSY_TIMEOUT_MS / 1000)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS entries (
            path TEXT PRIMARY KEY,
            kind TEXT, task TEXT, date TEXT, tags TEXT, title TEXT, body TEXT
        );
        """
    )
    return conn


def _ensure_fts(conn: sqlite3.Connection) -> bool:
    """Create the FTS table if SQLite supports it. Uses the caller's connection."""
    key = str(paths.memory_index_path())
    if key in _fts_state:
        return _fts_state[key]
    try:
        existing = {
            row[1] for row in conn.execute("PRAGMA table_info(entries_fts)").fetchall()
        }
        if existing and "path" not in existing:
            conn.execute("DROP TABLE entries_fts")
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts USING fts5(path UNINDEXED, title, body)"
        )
        _fts_state[key] = True
    except sqlite3.OperationalError:
        _fts_state[key] = False
    return bool(_fts_state[key])


def fts_available() -> bool:
    """Report FTS support without opening a competing write transaction."""
    conn = _connect()
    try:
        return _ensure_fts(conn)
    finally:
        conn.close()


def reindex() -> int:
    conn = _connect()
    entries = list(iter_entries())
    try:
        with conn:  # one transaction, no nested connections
            conn.execute("DELETE FROM entries")
            conn.executemany(
                "INSERT OR REPLACE INTO entries (path, kind, task, date, tags, title, body) "
                "VALUES (?,?,?,?,?,?,?)",
                [
                    (
                        str(entry.path),
                        entry.kind,
                        entry.task,
                        entry.date,
                        ",".join(entry.tags),
                        entry.title,
                        entry.body,
                    )
                    for entry in entries
                ],
            )
            if _ensure_fts(conn):
                try:
                    conn.execute("DELETE FROM entries_fts")
                    conn.execute(
                        "INSERT INTO entries_fts (path, title, body) "
                        "SELECT path, title, body FROM entries"
                    )
                except sqlite3.OperationalError:
                    _fts_state[str(paths.memory_index_path())] = False
    finally:
        conn.close()
    return len(entries)


def search(query: str, *, limit: int = 10, kind: str | None = None) -> list[dict[str, Any]]:
    conn = _connect()
    results: list[dict[str, Any]] = []
    try:
        if _ensure_fts(conn):
            sql = (
                "SELECT e.path, e.kind, e.task, e.date, e.title, e.tags, "
                "snippet(entries_fts, 2, '[', ']', '…', 12) AS excerpt "
                "FROM entries_fts f JOIN entries e ON e.path = f.path "
                "WHERE entries_fts MATCH ?"
            )
            params: list[Any] = [_fts_query(query)]
            if kind:
                sql += " AND e.kind = ?"
                params.append(kind)
            sql += " ORDER BY bm25(entries_fts) LIMIT ?"
            params.append(limit)
            try:
                rows = conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError:
                _fts_state[str(paths.memory_index_path())] = False
                rows = []
        else:
            rows = []
        if not rows:
            like = f"%{query}%"
            sql = (
                "SELECT path, kind, task, date, title, tags, substr(body, 1, 400) AS excerpt "
                "FROM entries WHERE (title LIKE ? OR body LIKE ?)"
            )
            params = [like, like]
            if kind:
                sql += " AND kind = ?"
                params.append(kind)
            sql += " ORDER BY date DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
        results = [dict(row) for row in rows]
    finally:
        conn.close()
    return results


def _fts_query(query: str) -> str:
    tokens = [t for t in re.split(r"\s+", query.strip()) if t]
    return " OR ".join(f'"{t}"' for t in tokens) if tokens else '""'


def stats() -> dict[str, Any]:
    conn = _connect()
    try:
        rows = conn.execute("SELECT kind, COUNT(*) AS n FROM entries GROUP BY kind").fetchall()
    finally:
        conn.close()
    by_kind = {row["kind"]: row["n"] for row in rows}
    return {"entries": sum(by_kind.values()), "by_kind": by_kind, "fts": fts_available()}


def recent(kind: str = "runs", limit: int = 5) -> list[MemoryEntry]:
    entries = [e for e in iter_entries() if e.kind == kind]
    entries.sort(key=lambda e: (e.date, e.path.name), reverse=True)
    return entries[:limit]


def context_for_prompt(*, task: str, limit: int = 3) -> str:
    """A short memory block injected into prompts (opt-in per task)."""
    entries = recent("notes", limit=limit) + [
        e for e in recent("runs", limit=limit) if e.task == task
    ]
    if not entries:
        return ""
    lines = ["## 历史记忆（来自本机 memory/，可能有遗漏）"]
    for entry in entries:
        excerpt = " ".join(entry.body.split())[:200]
        lines.append(f"- [{entry.kind}] {entry.title}：{excerpt}")
    return "\n".join(lines)


def export_json() -> str:
    return json.dumps([entry.as_dict() for entry in iter_entries()], ensure_ascii=False, indent=2)
