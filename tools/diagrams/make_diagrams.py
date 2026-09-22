#!/usr/bin/env python3
"""Generate the bilingual SVG diagrams used by the project page.

The diagrams are generated, not hand-drawn, so they can be regenerated when the
architecture changes:

    python3 tools/diagrams/make_diagrams.py --out /tmp/diagrams

Palette matches www.simon-zj.top (site.css custom properties).
"""

from __future__ import annotations

import argparse
from pathlib import Path

INK = "#151515"
MUTED = "#666862"
LINE = "rgba(21,21,21,0.12)"
SURFACE = "#ffffff"
SOFT = "#eeeee8"
ACCENT = "#e05a36"
ACCENT_SOFT = "#f4d9cf"
TEAL = "#1f6f68"
BLUE = "#315f86"
AMBER = "#a86b18"

FONT = "-apple-system, BlinkMacSystemFont, 'PingFang SC', 'Helvetica Neue', Arial, sans-serif"
MONO = "'SF Mono', ui-monospace, Menlo, monospace"


def esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def text(x: float, y: float, value: str, *, size: int = 16, fill: str = INK,
         weight: str = "400", family: str = FONT, anchor: str = "start",
         opacity: float = 1.0) -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" '
        f'opacity="{opacity}">{esc(value)}</text>'
    )


def rect(x: float, y: float, w: float, h: float, *, fill: str = SURFACE,
         stroke: str = LINE, radius: int = 10, width: float = 1.0) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'
    )


def line(x1: float, y1: float, x2: float, y2: float, *, stroke: str = LINE,
         width: float = 1.0, dash: str | None = None,
         marker: bool = False) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    mark = ' marker-end="url(#arrow)"' if marker else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
        f'stroke-width="{width}"{dash_attr}{mark}/>'
    )


def path(points: list[tuple[float, float]], *, stroke: str = MUTED,
         width: float = 1.2, marker: bool = True) -> str:
    d = " ".join(
        ("M" if index == 0 else "L") + f" {x} {y}" for index, (x, y) in enumerate(points)
    )
    mark = ' marker-end="url(#arrow)"' if marker else ""
    return f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{width}"{mark}/>'


def pill(x: float, y: float, label: str, *, fill: str, color: str, size: int = 13,
         pad: int = 12) -> str:
    width = len(label) * (size * 0.62) + pad * 2
    return (
        rect(x, y, width, size + 14, fill=fill, stroke="none", radius=(size + 14) / 2)
        + text(x + pad, y + size + 3, label, size=size, fill=color, weight="600")
    )


def svg(width: int, height: int, body: list[str]) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img">'
        "<defs>"
        '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        'markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{MUTED}"/>'
        "</marker>"
        "</defs>"
        + rect(0, 0, width, height, fill=SOFT, stroke="none", radius=0)
        + "".join(body)
        + "</svg>\n"
    )


LAYERS = [
    {
        "zh": ("触发与验收", "定时/事件唤醒；四项校验不过就不发布；运行全部入台账"),
        "en": ("Triggers & acceptance", "Scheduled or event wake-ups; four gates must pass; every run is ledgered"),
        "accent": ACCENT,
    },
    {
        "zh": ("任务与编排", "任务声明式描述（步骤/工具白名单/可写路径/预算/降级策略）"),
        "en": ("Tasks & orchestration", "Declarative tasks: steps, tool allow-list, writable paths, budget, degradation"),
        "accent": AMBER,
    },
    {
        "zh": ("工具与权限", "文件/命令/网络/通知都走统一接口，越权直接拒绝，调用可审计"),
        "en": ("Tools & permissions", "Files, shell, HTTP and notifications behind one interface; out-of-scope writes are refused"),
        "accent": BLUE,
    },
    {
        "zh": ("记忆", "事实自动写入 runs/，结论由人确认后写入 notes/，本机可审计"),
        "en": ("Memory", "Facts land in runs/ automatically; conclusions need human confirmation in notes/"),
        "accent": TEAL,
    },
    {
        "zh": ("模型适配", "provider 接口统一，云端模型为主，本地模型留同一接口"),
        "en": ("Model adapters", "One provider interface: cloud models by default, a local endpoint slots in"),
        "accent": MUTED,
    },
]


def diagram_architecture(lang: str) -> str:
    zh = lang == "zh"
    width, height = 1240, 720
    body: list[str] = []
    body.append(
        text(60, 64, "本地 Agent Harness · 五层结构" if zh else "Local Agent Harness · Five layers",
             size=27, weight="650")
    )
    body.append(
        text(60, 94,
             "记忆、权限、触发、验收归自己；模型与重活外包给可替换的执行器。" if zh
             else "You own memory, permissions, triggers and acceptance; models and heavy lifting are swappable.",
             size=15, fill=MUTED)
    )

    top = 130
    row_h = 84
    gap = 12
    for index, layer in enumerate(LAYERS):
        title, detail = layer["zh"] if zh else layer["en"]
        y = top + index * (row_h + gap)
        body.append(rect(60, y, 780, row_h, fill=SURFACE, stroke=LINE))
        body.append(rect(60, y, 6, row_h, fill=layer["accent"], stroke="none", radius=3))
        body.append(text(88, y + 34, title, size=18, weight="600"))
        body.append(text(88, y + 60, detail, size=13.5, fill=MUTED))

    # right column: executors
    body.append(rect(880, top, 300, row_h * 2 + gap, fill=SURFACE, stroke=LINE))
    body.append(text(904, top + 34, "可替换执行器" if zh else "Swappable executors",
                     size=18, weight="600"))
    for i, name in enumerate(["Codex CLI", "Claude Code", "DeepSeek API", "本地模型 / Local"]):
        body.append(text(904, top + 66 + i * 26, "· " + name, size=14, fill=MUTED))

    body.append(rect(880, top + (row_h + gap) * 2, 300, 84, fill=ACCENT_SOFT, stroke="none"))
    body.append(text(904, top + (row_h + gap) * 2 + 34, "无人值守触发" if zh else "Unattended trigger",
                     size=16, weight="600", fill="#8c3a1f"))
    body.append(text(904, top + (row_h + gap) * 2 + 60, "launchd / 定时 / 事件" if zh else "launchd / schedule / events",
                     size=13, fill="#8c3a1f"))

    body.append(rect(880, top + (row_h + gap) * 3, 300, 84, fill=SURFACE, stroke=LINE))
    body.append(text(904, top + (row_h + gap) * 3 + 34, "全部留在本机" if zh else "Everything stays local",
                     size=16, weight="600"))
    body.append(text(904, top + (row_h + gap) * 3 + 60, "memory/ · runs/ · 配置" if zh else "memory/ · runs/ · config",
                     size=13, fill=MUTED))

    # footer band
    y = top + 5 * (row_h + gap) + 6
    body.append(rect(60, y, 1120, 62, fill=SURFACE, stroke=LINE))
    body.append(
        text(84, y + 38,
             "与直接用通用 Coding Agent 的差别：会话与记忆归自己、能被定时唤醒、带自己的凭据与产物契约、验收标准写在代码里。" if zh
             else "How it differs from driving a general coding agent: you own the sessions and memory, it can wake itself up, it holds your credentials and output contract, and acceptance lives in code.",
             size=13.5, fill=MUTED)
    )
    return svg(width, height, body)


LIFECYCLE = [
    ("launchd 23:00", "定时唤醒", "launchd 23:00", "scheduled wake-up"),
    ("任务锁", "同任务只跑一个", "task lock", "one run at a time"),
    ("抓取当日数据", "代理/额度不可用则降级", "fetch capture", "degrades if sources fail"),
    ("分片撰写", "6 次调用 + 合并", "chunked compose", "six calls, then merge"),
    ("渲染页面", "dry-run 只出预览", "render pages", "dry-run keeps a preview"),
    ("四项校验", "不过就不发布", "four gates", "no publish without passing"),
    ("发布 + 记忆 + 通知", "推送、写台账、写记忆", "publish, ledger, memory", "push, ledger, memory"),
]


def diagram_lifecycle(lang: str) -> str:
    zh = lang == "zh"
    width, height = 1240, 600
    body: list[str] = []
    body.append(
        text(60, 64, "一次无人值守运行" if zh else "One unattended run", size=27, weight="650")
    )
    body.append(
        text(60, 94,
             "从定时唤醒到产物上线，中途任何一步降级都会被记录，而不是悄悄糊过去。" if zh
             else "From scheduled wake-up to a published artifact: every degradation is recorded instead of hidden.",
             size=15, fill=MUTED)
    )

    columns, card_w, card_h, gap = 4, 250, 148, 26
    start_x, start_y = 60, 140
    positions: list[tuple[float, float]] = []
    for index, step in enumerate(LIFECYCLE):
        row, col = divmod(index, columns)
        x = start_x + col * (card_w + gap)
        y = start_y + row * (card_h + 74)
        positions.append((x, y))
        accent = ACCENT if index >= 6 else (TEAL if index == 5 else BLUE)
        body.append(rect(x, y, card_w, card_h, fill=SURFACE, stroke=LINE))
        body.append(rect(x, y, card_w, 5, fill=accent, stroke="none", radius=3))
        body.append(text(x + 22, y + 42, f"{index + 1:02d}", size=13, fill=MUTED, family=MONO))
        body.append(text(x + 22, y + 76, step[0] if zh else step[2], size=17, weight="600"))
        body.append(text(x + 22, y + 106, step[1] if zh else step[3], size=13, fill=MUTED))
        if col < columns - 1 and index + 1 < len(LIFECYCLE) and (index + 1) % columns != 0:
            body.append(line(x + card_w + 4, y + card_h / 2, x + card_w + gap - 4,
                             y + card_h / 2, marker=True))

    # snake connector: bottom of card 04 -> down -> left -> up into card 05
    last_x = positions[3][0] + card_w / 2
    first_x = positions[4][0] + card_w / 2
    mid_y = positions[3][1] + card_h + 34
    body.append(
        path([(last_x, positions[3][1] + card_h + 4), (last_x, mid_y), (first_x, mid_y),
              (first_x, positions[4][1] - 4)])
    )

    band_y = 520
    body.append(rect(60, band_y, 1120, 60, fill=SURFACE, stroke=LINE))
    body.append(text(84, band_y + 26, "降级路径" if zh else "Degradation path", size=14, weight="600"))
    body.append(
        text(84, band_y + 48,
             "代理不可用 / X 额度耗尽 → 沿用已有数据继续跑并标记 degraded；校验不过或推送失败 → 判失败并通知，绝不发半成品。" if zh
             else "Proxy or quota failures reuse existing data and mark the run degraded; failed gates or a failed push stop the release and raise a notification.",
             size=13, fill=MUTED)
    )
    return svg(width, height, body)


ROUTES = [
    {
        "zh": "一次写完 · 全量上下文",
        "en": "Single shot · full context",
        "tin": 133_892,
        "tout": 16_384,
        "ok": False,
        "zh_note": "输出撞上限，JSON 截断",
        "en_note": "hit the output ceiling, JSON truncated",
    },
    {
        "zh": "一次写完 · 预筛上下文",
        "en": "Single shot · prefiltered",
        "tin": 36_340,
        "tout": 16_384,
        "ok": False,
        "zh_note": "同样被截断",
        "en_note": "truncated as well",
    },
    {
        "zh": "分片撰写 · 预筛上下文",
        "en": "Chunked · prefiltered",
        "tin": 72_438,
        "tout": 13_945,
        "ok": True,
        "zh_note": "20 + 10 条，可核验率 1.00",
        "en_note": "20 + 10 items, traceability 1.00",
    },
]

OUTPUT_CAP = 16_384


def diagram_evidence(lang: str) -> str:
    zh = lang == "zh"
    width, height = 1240, 700
    body: list[str] = []
    body.append(
        text(60, 64, "实测：为什么必须分片" if zh else "Measured: why chunking is required",
             size=27, weight="650")
    )
    body.append(
        text(60, 94,
             "2026-09-22 的真实抓取数据，deepseek-chat，同一份输入跑三条路线。" if zh
             else "Real capture from 2026-09-22, deepseek-chat, three routes over the same input.",
             size=15, fill=MUTED)
    )

    left_x, right_x, panel_w = 60, 660, 520
    panel_y, panel_h = 130, 430
    for x, title_zh, title_en, unit in (
        (left_x, "输入 tokens", "Input tokens", ""),
        (right_x, "输出 tokens（上限 16,384）", "Output tokens (ceiling 16,384)", ""),
    ):
        body.append(rect(x, panel_y, panel_w, panel_h, fill=SURFACE, stroke=LINE))
        body.append(text(x + 24, panel_y + 36, title_zh if zh else title_en, size=16, weight="600"))

    max_in = max(route["tin"] for route in ROUTES)
    bar_area = panel_w - 200
    cap_x = right_x + 24 + max(6.0, bar_area * OUTPUT_CAP / max_in)
    # Drawn before the bars and labels so it stays a background reference line.
    body.append(
        f'<line x1="{cap_x}" y1="{panel_y + 56}" x2="{cap_x}" y2="{panel_y + panel_h - 40}" '
        f'stroke="{ACCENT}" stroke-width="1.4" stroke-dasharray="5 4" opacity="0.45"/>'
    )
    body.append(
        text(cap_x + 8, panel_y + 74, "上限" if zh else "ceiling", size=12, fill=ACCENT, opacity=0.8)
    )

    for index, route in enumerate(ROUTES):
        y = panel_y + 80 + index * 110
        label = route["zh"] if zh else route["en"]
        note = route["zh_note"] if zh else route["en_note"]

        # input panel
        body.append(text(left_x + 24, y + 8, label, size=14.5, weight="600"))
        body.append(text(left_x + 24, y + 30, note, size=12.5, fill=MUTED))
        width_in = max(6.0, bar_area * route["tin"] / max_in)
        color = TEAL if route["ok"] else ACCENT
        body.append(rect(left_x + 24, y + 42, width_in, 22, fill=color, stroke="none", radius=6))
        body.append(text(left_x + 32 + width_in, y + 58, f'{route["tin"]:,}',
                         size=13, family=MONO, fill=INK if route["ok"] else ACCENT))

        # output panel
        body.append(text(right_x + 24, y + 8, label, size=14.5, weight="600"))
        body.append(text(right_x + 24, y + 30, note, size=12.5, fill=MUTED))
        width_out = max(6.0, bar_area * route["tout"] / max_in)
        body.append(rect(right_x + 24, y + 42, width_out, 22, fill=color, stroke="none", radius=6))
        body.append(text(right_x + 32 + width_out, y + 58, f'{route["tout"]:,}',
                         size=13, family=MONO, fill=INK if route["ok"] else ACCENT))

    # legend
    legend_y = 596
    body.append(rect(60, legend_y - 26, 16, 16, fill=TEAL, stroke="none", radius=4))
    body.append(text(86, legend_y - 12, "通过校验" if zh else "passed the gates", size=14))
    body.append(rect(300, legend_y - 26, 16, 16, fill=ACCENT, stroke="none", radius=4))
    body.append(text(326, legend_y - 12, "失败（输出被截断）" if zh else "failed (truncated output)", size=14))
    body.append(
        text(60, legend_y + 34,
             "结论：20 条双语稿件塞不进一次响应；分片后 token 更省、可核验率 1.00。" if zh
             else "Conclusion: a 20-item bilingual article does not fit one response; chunking is cheaper and fully traceable.",
             size=13.5, fill=MUTED)
    )
    return svg(width, height, body)


DIAGRAMS = {
    "architecture": diagram_architecture,
    "run-lifecycle": diagram_lifecycle,
    "experiment-evidence": diagram_evidence,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the project's SVG diagrams")
    parser.add_argument("--out", required=True, help="output directory")
    args = parser.parse_args()

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, render in DIAGRAMS.items():
        for lang in ("zh", "en"):
            path = out / f"{name}-{lang}.svg"
            path.write_text(render(lang), encoding="utf-8")
            written.append(path)
    for path in written:
        print(f"{path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
