#!/usr/bin/env python3
"""Fetch the raw capture for the target date, degrading when sources are down."""

from __future__ import annotations

import json
from datetime import date as _date

from harness import stepctx


def main() -> int:
    ctx = stepctx.load()
    tools = ctx.path("{tools_dir}")
    script = tools / "tools" / "fetch_sources.py"
    raw_path = tools / "data" / "raw" / f"{ctx.target_date}.json"

    # A capture for a past date is evidence, not a cache. Re-fetching it replaces
    # the only record of what the day actually looked like, and every citation in
    # that day's article was verified against it. This has already cost one day
    # (2026-09-22, overwritten by a run that looked harmless), so a past date with
    # an existing capture is left alone. Today's capture may still be refreshed.
    if raw_path.is_file() and ctx.target_date != _date.today().isoformat():
        return stepctx.finish(
            ctx,
            "ok",
            artifacts=[str(raw_path)],
            metrics={**_counts(raw_path), "refetched": 0},
            notes=f"沿用 {ctx.target_date} 已有抓取（历史日期的 raw 视为证据，不重新抓取）",
        )

    if not script.is_file():
        return stepctx.fail(ctx, f"找不到抓取脚本：{script}")

    result = ctx.registry.call(
        "shell_run",
        {
            "argv": ["python3", str(script), "--date", ctx.target_date, "--proxy", "auto"],
            "cwd": str(tools),
            "timeout_sec": 1400,
            "allow_failure": True,
            "writes": [str(raw_path.parent)],
        },
    )

    if result["returncode"] != 0:
        if raw_path.is_file():
            return stepctx.degrade(
                ctx,
                f"抓取失败（exit {result['returncode']}），沿用已存在的 raw 数据："
                f"{(result.get('stderr') or '').strip()[-200:]}",
                artifacts=[str(raw_path)],
                metrics=_counts(raw_path),
            )
        return stepctx.fail(
            ctx,
            f"抓取失败且没有可用的 raw 数据：{(result.get('stderr') or '').strip()[-400:]}",
        )

    if not raw_path.is_file():
        return stepctx.fail(ctx, f"抓取脚本返回成功但没有产物：{raw_path}")

    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    metrics = _counts(raw_path)
    status_text = str(raw.get("x_search_status", ""))
    if status_text and not status_text.startswith("HTTP 2"):
        return stepctx.degrade(
            ctx,
            f"X 付费读取不可用（{status_text}），X 侧为热榜快照 + 抽样原帖",
            artifacts=[str(raw_path)],
            metrics=metrics,
        )
    return stepctx.finish(
        ctx, "ok", artifacts=[str(raw_path)], metrics=metrics, notes="抓取成功"
    )


def _counts(raw_path) -> dict:
    if not raw_path.is_file():
        return {"raw_present": 0}
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    feeds = raw.get("feeds") or {}
    return {
        "raw_present": 1,
        "hn": len(raw.get("hn") or []),
        "github": len(raw.get("github") or []),
        "arxiv": len(raw.get("arxiv") or []),
        "tweets_recent": len(raw.get("tweets_recent") or []),
        "tweets_evergreen": len(raw.get("tweets_evergreen") or []),
        "techmeme": len(raw.get("techmeme") or []),
        "feeds": sum(len(v) for v in feeds.values() if isinstance(v, list)),
    }


if __name__ == "__main__":
    raise SystemExit(main())
