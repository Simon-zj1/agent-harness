#!/usr/bin/env python3
"""Normalise references and render pages; dry-run only produces a preview."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from harness import stepctx


def main() -> int:
    ctx = stepctx.load()
    tools = ctx.path("{tools_dir}")
    site = ctx.path("{site_repo}")
    content_path = ctx.run_dir / "content.json"
    if not content_path.is_file():
        return stepctx.fail(ctx, f"缺少撰写产物：{content_path}")
    content = json.loads(content_path.read_text(encoding="utf-8"))

    accept = bool(ctx.tool_ctx.data.get("accept_content", True))
    artifacts: list[str] = []
    metrics: dict = {}

    try:
        preview = _render_preview(ctx, tools=tools, content=content)
    except Exception as exc:  # noqa: BLE001 - preview must report, not crash
        return stepctx.fail(ctx, f"预览渲染失败：{exc}")
    artifacts.append(str(preview))
    metrics["preview"] = 1

    if not accept:
        return stepctx.finish(
            ctx,
            "ok",
            artifacts=artifacts,
            metrics=metrics,
            notes="实验模式：只生成预览，不写站点仓库",
        )

    canonical = tools / "data" / f"{ctx.target_date}.json"
    if not canonical.is_file():
        return stepctx.fail(ctx, f"内容未写入正式目录：{canonical}")

    dry_flag = ["--dry-run"] if ctx.dry_run else []
    normalize = ctx.registry.call(
        "shell_run",
        {
            "argv": [
                "python3",
                str(tools / "tools" / "normalize_refs.py"),
                "--date",
                ctx.target_date,
                *dry_flag,
            ],
            "cwd": str(tools),
            "allow_failure": True,
            "timeout_sec": 300,
            "writes": [str(tools / "data")],
        },
    )
    metrics["normalize_exit"] = normalize["returncode"]
    if normalize["returncode"] != 0:
        return stepctx.degrade(
            ctx,
            f"引用归一化失败：{(normalize.get('stderr') or '').strip()[-300:]}",
            artifacts=artifacts,
            metrics=metrics,
        )

    if ctx.dry_run:
        return stepctx.finish(
            ctx,
            "ok",
            artifacts=artifacts,
            metrics=metrics,
            notes="dry-run：已归一化引用并生成预览，未写站点仓库",
        )

    if not ctx.publish:
        # Publishing from this task is disabled (see [publish] in task.toml:
        # the 个人网站 checkout is a second writer on the same Pages branch and
        # the hexo build already ships trends/*). Writing into that checkout
        # anyway is what re-dirtied it and caused a 792-line deletion, so the
        # site writes now stay in the run directory.
        try:
            page = _render_sandboxed(ctx, tools=tools, content=content)
        except Exception as exc:  # noqa: BLE001 - report, do not crash
            return stepctx.fail(ctx, f"沙箱渲染失败：{exc}")
        artifacts.append(str(page))
        metrics["site_write"] = 0
        return stepctx.finish(
            ctx,
            "ok",
            artifacts=artifacts,
            metrics=metrics,
            notes="未启用发布：预览与渲染产物全部留在 run 目录，未写站点或博客仓库",
        )

    render = ctx.registry.call(
        "shell_run",
        {
            "argv": [
                "python3",
                str(tools / "tools" / "render_site.py"),
                "--date",
                ctx.target_date,
            ],
            "cwd": str(tools),
            "allow_failure": True,
            "timeout_sec": 600,
            "writes": [str(site)],
        },
    )
    metrics["render_exit"] = render["returncode"]
    if render["returncode"] != 0:
        return stepctx.degrade(
            ctx,
            f"页面渲染失败：{(render.get('stderr') or '').strip()[-300:]}",
            artifacts=artifacts,
            metrics=metrics,
        )

    for path in (
        site / "trends" / ctx.target_date / "index.html",
        site / "trends" / "index.html",
    ):
        if not path.is_file():
            return stepctx.degrade(
                ctx,
                f"渲染完成但缺少产物：{path}",
                artifacts=artifacts,
                metrics=metrics,
            )
        artifacts.append(str(path))
        (ctx.run_dir / f"rendered-{path.parent.name or 'index'}.html").write_text(
            path.read_text(encoding="utf-8"), encoding="utf-8"
        )
    return stepctx.finish(
        ctx, "ok", artifacts=artifacts, metrics=metrics, notes="已渲染到站点仓库"
    )


def _render_sandboxed(ctx, *, tools: Path, content: dict) -> Path:
    """Render entirely into the run directory by patching the renderer's roots.

    The preview root is used for both the site output and the hexo source data,
    so a dry-run never writes to the real site or blog checkout.
    """
    preview_root = ctx.run_dir / "site-preview"
    data_dir = ctx.run_dir / "preview-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / f"{ctx.target_date}.json").write_text(
        json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    module = _load_module(tools / "tools" / "render_site.py", "daily_trends_render_site")
    module.DATA_DIR = data_dir
    module.ARTIFACT_ROOT = preview_root / "artifacts"
    module.HEXO_SOURCE = preview_root / "hexo-source"
    # HEXO_DATA is derived from HEXO_SOURCE at import time, so patching the
    # source alone left it pointing at the real blog checkout: preview runs
    # silently rewrote <blog>/source/_data/ai_briefing.json, and the
    # relative_to(HEXO_SOURCE) in the renderer then crashed on the mismatched
    # path. Rebinding it keeps every write inside the preview root.
    module.HEXO_DATA = module.HEXO_SOURCE / "_data"
    module.SITE_ROOT = preview_root
    module.TRENDS_DIR = preview_root / "trends"
    module.TECH_INDEX = preview_root / "tech" / "index.html"
    # The official renderer uses an existing article page as its shell template.
    # Reading the real one is fine (read-only); everything it *writes* stays in
    # the preview root so a dry-run never touches the site repo.
    real_shell = ctx.path("{site_repo}") / "tech" / "tools" / "paper-reading-workflow" / "index.html"
    module.SHELL_PAGE = real_shell if real_shell.is_file() else _seed_shell(preview_root)

    argv = sys.argv
    sys.argv = ["render_site.py", "--date", ctx.target_date]
    try:
        code = module.main()
    finally:
        sys.argv = argv
    if code != 0:
        raise RuntimeError(f"render returned {code}")
    return preview_root / "trends" / ctx.target_date / "index.html"


def _render_preview(ctx, *, tools: Path, content: dict) -> Path:
    return _render_sandboxed(ctx, tools=tools, content=content)


def _seed_shell(preview_root: Path) -> Path:
    """Minimal fallback shell so previews still work on a fresh checkout."""
    path = preview_root / "shell" / "index.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "<!doctype html><html lang=\"zh\"><head><meta charset=\"utf-8\">"
        "<title>preview shell</title>"
        "<link rel=\"stylesheet\" href=\"/css/trends.css\"></head>"
        "<body><main class=\"article\"></main></body></html>",
        encoding="utf-8",
    )
    return path


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    raise SystemExit(main())
