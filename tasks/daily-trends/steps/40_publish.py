#!/usr/bin/env python3
"""Commit and push the rendered site - only after every validator passed."""

from __future__ import annotations

from harness import stepctx


def main() -> int:
    ctx = stepctx.load()
    if not ctx.publish:
        return stepctx.finish(ctx, "skipped", notes="publish 未启用（dry-run 或 --no-publish）")

    site = ctx.path("{site_repo}")
    config = ctx.task_raw.get("publish", {})
    remote = str(config.get("remote", "origin"))
    branch = str(config.get("branch", "main"))
    message = str(config.get("commit_message", "Daily trends: {date}")).replace(
        "{date}", ctx.target_date
    )
    add_paths = list(
        config.get("add", ["trends", "css/trends.css", "tech/index.html", "sitemap.xml"])
    )

    ctx.registry.call(
        "shell_run",
        {
            "argv": ["git", "-C", str(site), "add", "--", *add_paths],
            "cwd": str(site),
            "timeout_sec": 300,
            "writes": [str(site)],
        },
    )
    status = ctx.registry.call(
        "shell_run",
        {
            "argv": ["git", "-C", str(site), "status", "--porcelain", "--", *add_paths],
            "cwd": str(site),
            "timeout_sec": 120,
            "writes": [str(site)],
        },
    )
    if not (status.get("stdout") or "").strip():
        return stepctx.finish(
            ctx, "ok", notes="站点已是最新，无需提交", metrics={"published": 0}
        )

    ctx.registry.call(
        "shell_run",
        {
            "argv": ["git", "-C", str(site), "commit", "-m", message],
            "cwd": str(site),
            "timeout_sec": 300,
            "writes": [str(site)],
        },
    )
    push = ctx.registry.call(
        "shell_run",
        {
            "argv": ["git", "-C", str(site), "push", remote, branch],
            "cwd": str(site),
            "timeout_sec": 600,
            "allow_failure": True,
            "writes": [str(site)],
        },
    )
    if push["returncode"] != 0:
        return stepctx.fail(
            ctx,
            f"推送失败：{(push.get('stderr') or '').strip()[-400:]}",
            metrics={"published": 0},
        )
    head = ctx.registry.call(
        "shell_run",
        {
            "argv": ["git", "-C", str(site), "rev-parse", "--short", "HEAD"],
            "cwd": str(site),
            "timeout_sec": 60,
            "writes": [str(site)],
        },
    )
    return stepctx.finish(
        ctx,
        "ok",
        artifacts=[f"https://www.simon-zj.top/trends/{ctx.target_date}/"],
        metrics={"published": 1},
        notes=f"已推送 {(head.get('stdout') or '').strip()} 到 {remote}/{branch}",
    )


if __name__ == "__main__":
    raise SystemExit(main())
