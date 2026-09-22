#!/usr/bin/env python3
"""Sync the site repo with its remote before anything gets rendered."""

from __future__ import annotations

from harness import stepctx


def main() -> int:
    ctx = stepctx.load()
    if ctx.dry_run:
        return stepctx.finish(ctx, "skipped", notes="dry-run: 不动站点仓库")

    site = ctx.path("{site_repo}")
    publish = ctx.task_raw.get("publish", {})
    remote = str(publish.get("remote", "origin"))
    branch = str(publish.get("branch", "main"))

    fetch = ctx.registry.call(
        "shell_run",
        {
            "argv": ["git", "-C", str(site), "fetch", remote, branch],
            "cwd": str(site),
            "timeout_sec": 300,
            "allow_failure": True,
        },
    )
    if fetch["returncode"] != 0:
        return stepctx.degrade(
            ctx,
            f"无法从 {remote}/{branch} 同步站点（网络或代理不可用），在本地副本上继续："
            f"{(fetch.get('stderr') or '').strip()[-200:]}",
        )

    rebase = ctx.registry.call(
        "shell_run",
        {
            "argv": ["git", "-C", str(site), "rebase", "FETCH_HEAD"],
            "cwd": str(site),
            "timeout_sec": 300,
            "allow_failure": True,
        },
    )
    if rebase["returncode"] != 0:
        ctx.registry.call(
            "shell_run",
            {
                "argv": ["git", "-C", str(site), "rebase", "--abort"],
                "cwd": str(site),
                "allow_failure": True,
            },
        )
        return stepctx.degrade(
            ctx,
            f"git rebase 冲突，已回滚到同步前状态：{(rebase.get('stderr') or '').strip()[-200:]}",
        )
    return stepctx.finish(ctx, "ok", notes=f"synced {remote}/{branch}", metrics={"sync": "ok"})


if __name__ == "__main__":
    raise SystemExit(main())
