"""Measure what the pipeline *chose*, not just whether it cited honestly.

`verification_eval` measures the provenance gate: did the article cite things
that exist in the capture. It says nothing about whether the 20 items chosen
were the right 20. This module closes that gap with the smallest honest version
of what AIHOT calls SelectBench:

1. build a balanced sheet of candidates - half of them what the pipeline
   actually published, half of them high-signal rows it skipped;
2. a human marks `gold` = select / reject / either;
3. score the pipeline's own choices against those labels (precision, recall).

Two rules keep this from becoming theatre:

* the sheet ships with `gold` **empty** and a `draft` column that is explicitly
  labelled a guess, so a machine suggestion can never be mistaken for a label;
* scoring refuses to print numbers when no human labels exist. An unlabelled
  sheet is not evidence, and reporting "100% agreement with itself" would be
  exactly the self-congratulation this module exists to prevent.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from . import content_schema, dedup, paths
from .errors import HarnessError

GOLD_SELECT = "select"
GOLD_REJECT = "reject"
GOLD_EITHER = "either"
GOLD_VALUES = (GOLD_SELECT, GOLD_REJECT, GOLD_EITHER)

DEFAULT_SEED = 20261006


def data_dir() -> Path:
    from . import verification_eval

    return verification_eval.data_dir()


def available_days() -> list[str]:
    from . import verification_eval

    return verification_eval.available_days()


@dataclass
class Row:
    case_id: str
    day: str
    bucket: str  # selected | unselected
    title: str
    url: str
    section: str = ""
    group: str = ""
    kind: str = ""
    signal: str = ""
    draft: str = ""
    draft_reason: str = ""
    gold: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _selected_rows(days: Iterable[str], *, root: Path | None = None) -> list[Row]:
    base = root or data_dir()
    rows: list[Row] = []
    for day in days:
        path = base / f"{day}.json"
        if not path.is_file():
            continue
        try:
            content = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        view = content_schema.normalize(content)
        urls = dedup.reference_url_map(view.references)
        for item in view.items:
            url = next((urls[s] for s in item.sources if s in urls), "")
            rows.append(
                Row(
                    case_id="",
                    day=day,
                    bucket="selected",
                    title=item.title_zh or item.title_en,
                    url=url,
                    section=item.section,
                    group=item.group or "",
                    kind=item.variant,
                    signal=f"{len(item.sources)} 个引用",
                    draft=GOLD_SELECT,
                    draft_reason="管线选了它；请确认或改成 reject",
                )
            )
    return rows


def build_sheet(
    days: Iterable[str],
    *,
    sample: int = 100,
    root: Path | None = None,
    seed: int = DEFAULT_SEED,
) -> list[Row]:
    """Balanced sheet: half published, half high-signal-but-skipped."""
    days = list(days)
    rng = random.Random(seed)
    selected = _selected_rows(days, root=root)
    unselected: list[Row] = []
    for day in days:
        unselected.extend(_raw_rows(day, root=root))

    rng.shuffle(selected)
    rng.shuffle(unselected)
    want_selected = min(len(selected), sample // 2)
    want_unselected = min(len(unselected), sample - want_selected)
    rows = _interleave(selected[:want_selected], unselected[:want_unselected])
    for index, row in enumerate(rows, start=1):
        row.case_id = f"sel-{row.day}-{index:03d}"
    return rows


def _interleave(selected: list[Row], unselected: list[Row]) -> list[Row]:
    out: list[Row] = []
    left, right = list(selected), list(unselected)
    while left or right:
        if left:
            out.append(left.pop(0))
        if right:
            out.append(right.pop(0))
    return out


def write_sheet(rows: list[Row], outdir: Path | None = None) -> dict[str, Path]:
    target = outdir or (paths.runs_dir() / "verification" / "selection")
    target.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    jsonl_path = target / f"sheet-{stamp}.jsonl"
    csv_path = target / f"sheet-{stamp}.csv"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Row.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())
    return {"jsonl": jsonl_path, "csv": csv_path}


def load_sheet(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.is_file():
        raise HarnessError(f"sheet not found: {target}")
    if target.suffix == ".csv":
        with target.open(encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    rows: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def score_sheet(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Score the pipeline's selection against human labels.

    `prediction` is what the pipeline did (bucket == selected). Rows without a
    `gold` value are ignored, and a sheet with no labels returns `labelled = 0`
    instead of a flattering zero-error report.
    """
    labelled = [
        row
        for row in rows
        if str(row.get("gold", "")).strip().lower() in (GOLD_SELECT, GOLD_REJECT)
    ]
    either = [
        row for row in rows if str(row.get("gold", "")).strip().lower() == GOLD_EITHER
    ]
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    per_day: dict[str, dict[str, int]] = {}
    for row in labelled:
        gold_select = str(row["gold"]).strip().lower() == GOLD_SELECT
        predicted_select = str(row.get("bucket", "")) == "selected"
        if gold_select and predicted_select:
            key = "tp"
        elif gold_select:
            key = "fn"
        elif predicted_select:
            key = "fp"
        else:
            key = "tn"
        counts[key] += 1
        day = str(row.get("day", "?"))
        per_day.setdefault(day, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})[key] += 1

    precision = (
        counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else None
    )
    recall = (
        counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else None
    )
    total = sum(counts.values())
    accuracy = (counts["tp"] + counts["tn"]) / total if total else None
    return {
        "sheet_rows": len(rows),
        "labelled": len(labelled),
        "either": len(either),
        "unlabelled": len(rows) - len(labelled) - len(either),
        "counts": counts,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "per_day": per_day,
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def markdown(report: dict[str, Any]) -> str:
    def pct(value: float | None) -> str:
        return "—" if value is None else f"{value * 100:.1f}%"

    lines = [
        "# 精选质量评测（人工标注）",
        "",
        f"- 生成时间：{report['generated_at']}",
        f"- 标注条数：{report['labelled']}（两可 {report['either']}，未标注 {report['unlabelled']}）",
        f"- 查准率 precision：{pct(report['precision'])}",
        f"- 查全率 recall：{pct(report['recall'])}",
        f"- 准确率 accuracy：{pct(report['accuracy'])}",
        "",
        "| | 管线选了 | 管线没选 |",
        "| --- | --- | --- |",
        f"| 人工认为该选 | TP={report['counts']['tp']} | FN={report['counts']['fn']} |",
        f"| 人工认为不该选 | FP={report['counts']['fp']} | TN={report['counts']['tn']} |",
        "",
        "## 说明",
        "",
        "- `prediction` 就是管线当前的选择（bucket=selected），这里衡量「选得准不准」，",
        "  不是「引用是否可核验」——后者由 `verify eval` 负责。",
        "- 标注少于 30 条时不要引用这些数字：置信区间宽到没有意义。",
        "",
    ]
    if report["per_day"]:
        lines += [
            "## 按天",
            "",
            "| 日期 | TP | FP | FN | TN |",
            "| --- | --- | --- | --- | --- |",
        ]
        for day in sorted(report["per_day"]):
            row = report["per_day"][day]
            lines.append(
                f"| {day} | {row['tp']} | {row['fp']} | {row['fn']} | {row['tn']} |"
            )
        lines.append("")
    return "\n".join(lines)


_RAW_KINDS = (
    ("hn", "title", "url", "points", "Hacker News"),
    ("github", "full_name", "html_url", "stars_per_day", "GitHub"),
    ("arxiv", "title", "url", "published", "arXiv"),
    ("techmeme", "title", "url", "hour", "Techmeme"),
)


def _raw_rows(day: str, *, root: Path | None = None) -> list[Row]:
    """High-signal rows from the capture, minus the ones the article used."""
    base = root or data_dir()
    raw_path = base / "raw" / f"{day}.json"
    content_path = base / f"{day}.json"
    if not raw_path.is_file():
        return []
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    cited: set[str] = set()
    if content_path.is_file():
        try:
            view = content_schema.normalize(json.loads(content_path.read_text(encoding="utf-8")))
            cited = set(dedup.reference_url_map(view.references).values())
        except json.JSONDecodeError:
            cited = set()

    rows: list[Row] = []
    for key, title_field, url_field, signal_field, label in _RAW_KINDS:
        entries = raw.get(key) or []
        if not isinstance(entries, list):
            continue

        def sort_key(entry: dict) -> float:
            value = entry.get(signal_field)
            return -float(value) if isinstance(value, (int, float)) else 0.0

        for entry in sorted((e for e in entries if isinstance(e, dict)), key=sort_key)[:25]:
            url = str(entry.get(url_field) or "").rstrip("/")
            if not url or url in cited:
                continue
            rows.append(
                Row(
                    case_id="",
                    day=day,
                    bucket="unselected",
                    title=str(entry.get(title_field) or url)[:200],
                    url=url,
                    kind=key,
                    signal=f"{label} {signal_field}={entry.get(signal_field)}",
                    draft=GOLD_REJECT,
                    draft_reason=f"管线没选它（{label} 高信号项）；请确认或改成 select",
                )
            )
    for name, entries in (raw.get("feeds") or {}).items():
        if not isinstance(entries, list):
            continue
        for entry in entries[:10]:
            if not isinstance(entry, dict):
                continue
            url = str(entry.get("url") or "").rstrip("/")
            if not url or url in cited:
                continue
            rows.append(
                Row(
                    case_id="",
                    day=day,
                    bucket="unselected",
                    title=str(entry.get("title") or url)[:200],
                    url=url,
                    kind=f"feed:{name}",
                    signal=str(entry.get("date") or entry.get("published") or ""),
                    draft=GOLD_REJECT,
                    draft_reason=f"管线没选它（{name} 源）；请确认或改成 select",
                )
            )
    return rows
