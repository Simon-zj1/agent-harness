#!/usr/bin/env python3
"""Write the bilingual content: replay (cheap), llm (own loop) or delegate (Codex/Claude)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from harness import providers, stepctx, validators

MAX_CHARS_FULL = 220_000
MAX_CHARS_PREFILTERED = 60_000
MAX_CHUNK_EVIDENCE = 40_000

# Fixed scaffolding: the model only fills items, so it can never invent sections
# that the renderer and validators do not know about.
INSIGHTS_GROUPS = [
    ("agent-engineering", "Agent 工程优化（上下文工程 / harness / 多 Agent 协同 / 编排）",
     "Agent engineering: context engineering, harnesses, multi-agent orchestration"),
    ("robotics", "机器人与具身智能（感知 / 预测 / 世界模型）",
     "Robotics and embodied AI: perception, prediction, world models"),
    ("ai-productivity", "AI 提效与工作方式", "AI productivity and ways of working"),
    ("industry-moves", "模型公司动向与人物 / 实验室观点",
     "Model-lab moves and people/lab viewpoints"),
]

CHUNK_MAX_TOKENS = 8000

# Hard caps mirroring the published-page contract (and the validators). The
# model is asked for 5 per group; these are the enforcement, not the request.
MAX_INSIGHTS_TOTAL = 20
MAX_REPOS = 10
GROUP_CHUNK_LIMIT = 5
GITHUB_CHUNK_LIMIT = 10


def main() -> int:
    ctx = stepctx.load()
    tools = ctx.path("{tools_dir}")
    raw_path = tools / "data" / "raw" / f"{ctx.target_date}.json"
    canonical = tools / "data" / f"{ctx.target_date}.json"
    content_path = ctx.run_dir / "content.json"
    mode = (ctx.compose_mode or "").strip() or str(
        ctx.task_raw.get("context", {}).get("compose_default", "replay")
    )
    if mode == "auto":
        mode = "llm"

    print(json.dumps({"compose_mode": mode, "strategy": ctx.context_strategy}))
    if mode == "replay":
        return _replay(ctx, canonical=canonical, content_path=content_path)
    if mode == "llm":
        return _llm_chunked(ctx, raw_path=raw_path, content_path=content_path)
    if mode == "llm-single":
        return _llm(ctx, raw_path=raw_path, content_path=content_path, tools=tools)
    if mode == "delegate":
        return _delegate(ctx, raw_path=raw_path, content_path=content_path, tools=tools)
    return stepctx.fail(ctx, f"未知的 compose 模式：{mode}")


def _llm_chunked(ctx, *, raw_path: Path, content_path: Path) -> int:
    """Map-reduce composition.

    A 20-item bilingual article does not fit in one response: on 2026-09-22 the
    single-shot path was cut off at the 16,384-token output ceiling and produced
    unusable JSON. So the article is composed in six bounded calls (one overview
    + four insight groups + one GitHub chunk) and merged here.
    """
    if not raw_path.is_file():
        return stepctx.fail(ctx, f"缺少 raw 抓取结果：{raw_path}")

    provider_name = str(ctx.task_raw.get("context", {}).get("provider", "")) or None
    provider = providers.get(ctx.config, provider_name)
    available, note = provider.available()
    if not available:
        return stepctx.fail(ctx, f"provider {provider.name} 不可用：{note}")

    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw_urls = _raw_url_index(raw)
    evidence = build_evidence(raw, strategy=ctx.context_strategy)[:MAX_CHUNK_EVIDENCE]
    memory_block = stepctx.memory_block(ctx)

    tokens_in = tokens_out = 0
    cost_usd = None
    cost_usd = None
    notes: list[str] = []

    overview, ti, to, cost, problem = _call_chunk(
        ctx, provider, kind="overview", scope="全站总览", limit=0,
        evidence=evidence, memory=memory_block,
    )
    tokens_in += ti
    tokens_out += to
    cost_usd = _merge_cost(cost_usd, cost)
    if overview is None:
        return stepctx.fail(
            ctx,
            f"总览分段失败：{problem}",
            metrics={"tokens_in": tokens_in, "tokens_out": tokens_out, "compose_mode": "llm"},
        )

    group_payloads: dict[str, dict] = {}
    for group_id, scope_zh, scope_en in INSIGHTS_GROUPS:
        payload, ti, to, cost, problem = _call_chunk(
            ctx, provider, kind="group", scope=f"{scope_zh} / {scope_en}",
            limit=GROUP_CHUNK_LIMIT,
            evidence=evidence, memory=memory_block,
        )
        tokens_in += ti
        tokens_out += to
        cost_usd = _merge_cost(cost_usd, cost)
        if payload is None:
            notes.append(f"{group_id} 分段失败：{problem}")
            group_payloads[group_id] = {"items": [], "references": []}
        else:
            group_payloads[group_id] = payload

    github_payload, ti, to, cost, problem = _call_chunk(
        ctx, provider, kind="github", scope="GitHub 热门仓库", limit=GITHUB_CHUNK_LIMIT,
        evidence=evidence, memory=memory_block,
    )
    tokens_in += ti
    tokens_out += to
    cost_usd = _merge_cost(cost_usd, cost)
    if github_payload is None:
        notes.append(f"github 分段失败：{problem}")
        github_payload = {"items": [], "references": []}

    content, merge_notes = _merge_chunks(
        date=ctx.target_date,
        overview=overview,
        group_payloads=group_payloads,
        github_payload=github_payload,
        raw_urls=raw_urls,
    )
    notes += merge_notes

    content_path.parent.mkdir(parents=True, exist_ok=True)
    content_path.write_text(
        json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    failures = _check(ctx, content, raw_path=raw_path, content_path=content_path)
    metrics = {
        "compose_mode": "llm",
        "context_strategy": ctx.context_strategy,
        "chunks": 2 + len(INSIGHTS_GROUPS),
        "evidence_chars": len(evidence),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
    }
    if cost_usd is not None:
        metrics["cost_usd"] = cost_usd
    metrics.update(_metrics(content_path, raw_path))
    if not failures:
        _accept_content(ctx, content)
    if failures:
        return stepctx.degrade(
            ctx,
            "分片撰写产出未通过校验：" + "；".join(failures[:5]),
            artifacts=[str(content_path)],
            metrics=metrics,
        )
    if notes:
        return stepctx.degrade(
            ctx,
            "；".join(notes[:5]),
            artifacts=[str(content_path)],
            metrics=metrics,
        )
    return stepctx.finish(
        ctx,
        "ok",
        artifacts=[str(content_path)],
        metrics=metrics,
        notes=f"{provider.name} 分片撰写完成（{ctx.context_strategy} 上下文，"
        f"{metrics['chunks']} 次调用）",
    )


def _call_chunk(
    ctx,
    provider,
    *,
    kind: str,
    scope: str,
    limit: int,
    evidence: str,
    memory: str,
) -> tuple[dict | None, int, int, float | None, str]:
    """One bounded model call.

    Returns (payload, tokens_in, tokens_out, cost_usd, problem).  ``cost_usd``
    is ``None`` when provider unit prices are not configured, so the ledger can
    distinguish "free" from "cost unknown".
    """
    system = (ctx.task_dir / str(ctx.task_raw.get("context", {}).get("prompt_dir", "prompts")) / "system.md").read_text(encoding="utf-8")
    template = (ctx.task_dir / str(ctx.task_raw.get("context", {}).get("prompt_dir", "prompts")) / "chunk.md").read_text(encoding="utf-8")
    base_prompt = (
        template.replace("{DATE}", ctx.target_date)
        .replace("{CHUNK_KIND}", kind)
        .replace("{CHUNK_SCOPE}", scope)
        .replace("{CHUNK_LIMIT}", str(limit) if limit else "—")
        .replace("{EVIDENCE}", evidence)
        .replace("{MEMORY}", memory)
    )

    tokens_in = tokens_out = 0
    cost_usd = None
    problem = "未调用"
    prompt = base_prompt
    for attempt in range(2):
        try:
            response = provider.chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=CHUNK_MAX_TOKENS,
            )
        except Exception as exc:  # noqa: BLE001 - a chunk failure must not kill the run
            problem = f"{type(exc).__name__}: {exc}"
            return None, tokens_in, tokens_out, cost_usd, problem
        tokens_in += response.tokens_in
        tokens_out += response.tokens_out
        cost_usd = _merge_cost(cost_usd, response.cost_usd)
        cost_usd = _merge_cost(cost_usd, response.cost_usd)
        payload = _parse_json(response.text)
        if payload is not None and response.finish_reason != "length":
            return payload, tokens_in, tokens_out, cost_usd, ""
        if response.finish_reason == "length":
            problem = f"输出被截断（达到 {CHUNK_MAX_TOKENS} token 上限）"
        else:
            problem = "输出不是合法 JSON"
        if attempt == 0:
            prompt = (
                base_prompt
                + f"\n\n上一次失败：{problem}。请更精简：缩短 summary/comment，减少条目数，"
                "但仍必须是完整可解析的单个 JSON 对象。"
            )
    return None, tokens_in, tokens_out, cost_usd, problem


def _merge_chunks(
    *,
    date: str,
    overview: dict,
    group_payloads: dict[str, dict],
    github_payload: dict,
    raw_urls: dict[str, str],
) -> tuple[dict, list[str]]:
    """Assemble the final article and convert per-chunk citation numbers to global ids."""
    notes: list[str] = []
    references: list[dict] = []
    url_to_id: dict[str, int] = {}
    dropped_citations = 0
    dropped_items = 0
    dropped_by_source: dict[str, int] = {}

    def global_id(url: str) -> int | None:
        nonlocal dropped_citations
        normalised = _normalise_url(url)
        canonical = raw_urls.get(normalised)
        if canonical is None:
            dropped_citations += 1
            return None
        if canonical not in url_to_id:
            url_to_id[canonical] = len(references) + 1
            references.append({"id": url_to_id[canonical], "url": canonical})
        return url_to_id[canonical]

    def convert_items(payload: dict, *, with_meta: bool, label: str) -> list[dict]:
        nonlocal dropped_items
        local_refs: dict[int, dict] = {}
        for entry in payload.get("references") or []:
            if isinstance(entry, dict) and isinstance(entry.get("id"), int):
                local_refs[entry["id"]] = entry
        items: list[dict] = []
        for raw_item in payload.get("items") or []:
            if not isinstance(raw_item, dict):
                continue
            mapping: dict[int, int] = {}
            for local_id, entry in local_refs.items():
                gid = global_id(str(entry.get("url", "")))
                if gid:
                    mapping[local_id] = gid
                    record = references[gid - 1]
                    record.setdefault("title", entry.get("title", ""))
                    record.setdefault("source", entry.get("source", ""))
                    record.setdefault("date", entry.get("date", ""))

            cited: set[int] = set()
            item: dict = {}
            for field in ("title", "summary", "comment"):
                block = raw_item.get(field) or {}
                if not isinstance(block, dict):
                    block = {}
                converted = {}
                for lang in ("zh", "en"):
                    text = str(block.get(lang, ""))
                    text, found = _rewrite_markers(text, mapping)
                    cited.update(found)
                    converted[lang] = text
                item[field] = converted
            for source in raw_item.get("sources") or []:
                if isinstance(source, int) and source in mapping:
                    cited.add(mapping[source])
                elif isinstance(source, str) and source.startswith(("http://", "https://")):
                    # Models sometimes cite the URL directly instead of an index.
                    gid = global_id(source)
                    if gid:
                        cited.add(gid)
            if not cited:
                dropped_items += 1
                dropped_by_source[label] = dropped_by_source.get(label, 0) + 1
                continue
            item["sources"] = sorted(cited)
            if with_meta and isinstance(raw_item.get("meta"), dict):
                item["meta"] = raw_item["meta"]
            items.append(item)
        return items

    groups = []
    per_group_items: list[list[dict]] = []
    all_insight_items: list[dict] = []
    for group_id, scope_zh, scope_en in INSIGHTS_GROUPS:
        payload = group_payloads.get(group_id) or {"items": [], "references": []}
        items = convert_items(payload, with_meta=False, label=group_id)
        per_group_items.append(items)
        groups.append(
            {
                "id": group_id,
                "title": {"zh": scope_zh.split("（")[0].strip(), "en": scope_en},
                "items": items,
            }
        )
    github_items = convert_items(github_payload, with_meta=True, label="github")

    # Enforce the caps instead of trusting the model to respect them: take items
    # round-robin across groups so every group stays represented.
    trimmed_insights = 0
    if sum(len(items) for items in per_group_items) > MAX_INSIGHTS_TOTAL:
        kept: list[list[dict]] = [[] for _ in per_group_items]
        cursors = [0] * len(per_group_items)
        total = 0
        while total < MAX_INSIGHTS_TOTAL:
            progressed = False
            for index, items in enumerate(per_group_items):
                if cursors[index] < len(items) and total < MAX_INSIGHTS_TOTAL:
                    kept[index].append(items[cursors[index]])
                    cursors[index] += 1
                    total += 1
                    progressed = True
            if not progressed:
                break
        trimmed_insights = sum(len(items) for items in per_group_items) - total
        for index, group in enumerate(groups):
            group["items"] = kept[index]
    all_insight_items = [item for items in (group["items"] for group in groups) for item in items]

    trimmed_repos = 0
    if len(github_items) > MAX_REPOS:
        trimmed_repos = len(github_items) - MAX_REPOS
        github_items = github_items[:MAX_REPOS]

    cited_ids = {
        source for item in all_insight_items + github_items for source in item["sources"]
    }
    kept = [ref for ref in references if ref["id"] in cited_ids]
    renumber = {ref["id"]: index + 1 for index, ref in enumerate(kept)}
    for ref in kept:
        ref["id"] = renumber[ref["id"]]
        ref.setdefault("title", "")
        ref.setdefault("source", "")
        ref.setdefault("date", "")
    for item in all_insight_items + github_items:
        item["sources"] = sorted(renumber[source] for source in item["sources"] if source in renumber)
        for field in ("summary", "comment", "title"):
            block = item.get(field) or {}
            for lang in ("zh", "en"):
                block[lang] = _remap_markers(str(block.get(lang, "")), renumber)

    intros = (overview.get("section_intros") or {})
    content = {
        "date": date,
        "generated_at": _now_local(),
        "title": overview.get("title") or {"zh": f"每日技术趋势 · {date}", "en": f"Daily Tech Trends · {date}"},
        "summary": _pair(overview.get("summary")),
        "tldr": overview.get("tldr") or {"zh": [], "en": []},
        "sections": [
            {
                "id": "insights",
                "title": {
                    "zh": "一、行业热点：Agent 工程 · 机器人 · AI 提效 · 公司与人物动向",
                    "en": "Part 1 · Industry signals",
                },
                "intro": _pair(intros.get("insights")),
                "groups": groups,
            },
            {
                "id": "github",
                "title": {"zh": "二、GitHub 热点", "en": "Part 2 · GitHub"},
                "intro": _pair(intros.get("github")),
                "items": github_items,
            },
        ],
        "references": kept,
        "notes": _pair(overview.get("notes")),
        "stats": {"insights": len(all_insight_items), "repos": len(github_items)},
    }
    if dropped_citations:
        notes.append(f"丢弃 {dropped_citations} 条无法回溯到当日抓取结果的引用")
    if dropped_items:
        notes.append(f"丢弃 {dropped_items} 条没有任何可核验来源的条目")
        detail = ", ".join(f"{key}:{value}" for key, value in sorted(dropped_by_source.items()))
        if detail:
            notes.append(f"（分布 {detail}）")
    if trimmed_insights:
        notes.append(f"按上限裁剪 {trimmed_insights} 条行业热点（上限 {MAX_INSIGHTS_TOTAL}）")
    if trimmed_repos:
        notes.append(f"按上限裁剪 {trimmed_repos} 条 GitHub 热点（上限 {MAX_REPOS}）")
    return content, notes


def _rewrite_markers(text: str, mapping: dict[int, int]) -> tuple[str, set[int]]:
    found: set[int] = set()

    def replace(match: re.Match[str]) -> str:
        local = int(match.group(1))
        if local in mapping:
            found.add(mapping[local])
            return f"[[{mapping[local]}]]"
        return ""

    return re.sub(r"\[\[(\d+)\]\]", replace, text), found


def _remap_markers(text: str, renumber: dict[int, int]) -> str:
    def replace(match: re.Match[str]) -> str:
        target = renumber.get(int(match.group(1)))
        return f"[[{target}]]" if target else ""

    return re.sub(r"\[\[(\d+)\]\]", replace, text)


def _pair(value) -> dict:
    if isinstance(value, dict):
        return {"zh": str(value.get("zh", "")), "en": str(value.get("en", ""))}
    if isinstance(value, str):
        return {"zh": value, "en": value}
    return {"zh": "", "en": ""}


def _merge_cost(total: float | None, value: float | None) -> float | None:
    """Accumulate known costs while preserving the unknown-cost signal."""
    if value is None:
        return total
    return round((total or 0.0) + float(value), 6)


def _now_local() -> str:
    import datetime as dt

    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _normalise_url(url: str) -> str:
    text = url.strip().rstrip("/")
    for prefix in ("https://", "http://", "www."):
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text.lower()


def _raw_url_index(raw: dict) -> dict[str, str]:
    """Map normalised url -> canonical url for every url in the capture."""
    index: dict[str, str] = {}
    for url in _urls(raw):
        index.setdefault(_normalise_url(url), url)
    return index


def _replay(ctx, *, canonical: Path, content_path: Path) -> int:
    if not canonical.is_file():
        return stepctx.fail(
            ctx, f"replay 模式需要已存在的 {canonical}；请改用 --compose llm 重新撰写"
        )
    content = json.loads(canonical.read_text(encoding="utf-8"))
    content_path.write_text(
        json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return stepctx.finish(
        ctx,
        "ok",
        artifacts=[str(content_path)],
        metrics={"compose_mode": "replay", "tokens_in": 0, "tokens_out": 0},
        notes="复用已有内容（确定性基线）",
    )


def _llm(ctx, *, raw_path: Path, content_path: Path, tools: Path) -> int:
    if not raw_path.is_file():
        return stepctx.fail(ctx, f"缺少 raw 抓取结果：{raw_path}")

    provider_name = str(ctx.task_raw.get("context", {}).get("provider", "")) or None
    provider = providers.get(ctx.config, provider_name)
    available, note = provider.available()
    if not available:
        return stepctx.fail(ctx, f"provider {provider.name} 不可用：{note}")

    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    evidence = build_evidence(raw, strategy=ctx.context_strategy)
    system_prompt, user_prompt = _render_prompts(ctx, evidence=evidence)

    tokens_in = tokens_out = 0
    repair_rounds = int(ctx.task_raw.get("context", {}).get("repair_rounds", 1))
    failures: list[str] = []
    content: dict | None = None

    for _attempt in range(repair_rounds + 1):
        response = provider.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        tokens_in += response.tokens_in
        tokens_out += response.tokens_out
        content = _parse_json(response.text)
        if content is None:
            failures = ["模型输出不是合法 JSON 对象"]
        else:
            content_path.parent.mkdir(parents=True, exist_ok=True)
            content_path.write_text(
                json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            failures = _check(ctx, content, raw_path=raw_path, content_path=content_path)
        if content is not None and not failures:
            break
        user_prompt = (
            user_prompt
            + "\n\n上一次输出未通过校验，请修正后重新输出完整 JSON。问题清单：\n- "
            + "\n- ".join(failures[:12])
        )

    if content is None:
        return stepctx.fail(
            ctx,
            "模型未能产出可解析的 JSON",
            metrics={"tokens_in": tokens_in, "tokens_out": tokens_out, "compose_mode": "llm"},
        )

    content_path.write_text(
        json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metrics = {
        "compose_mode": "llm",
        "context_strategy": ctx.context_strategy,
        "evidence_chars": len(evidence),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
    }
    if cost_usd is not None:
        metrics["cost_usd"] = cost_usd
    metrics.update(_metrics(content_path, raw_path))
    _accept_content(ctx, content)
    if failures:
        return stepctx.degrade(
            ctx,
            "模型产出未通过校验：" + "；".join(failures[:5]),
            artifacts=[str(content_path)],
            metrics=metrics,
        )
    return stepctx.finish(
        ctx,
        "ok",
        artifacts=[str(content_path)],
        metrics=metrics,
        notes=f"{provider.name} 撰写完成（{ctx.context_strategy} 上下文）",
    )


def _delegate(ctx, *, raw_path: Path, content_path: Path, tools: Path) -> int:
    executor = ctx.executor if ctx.executor in ("codex", "claude") else "codex"
    tool_name = "codex_exec" if executor == "codex" else "claude_exec"
    if tool_name not in ctx.registry.tools:
        return stepctx.fail(ctx, f"任务未允许 {tool_name} 工具")

    prompt = _delegation_prompt(ctx, raw_path=raw_path, content_path=content_path, tools=tools)
    if executor == "codex":
        result = ctx.registry.call(
            "codex_exec",
            {
                "prompt": prompt,
                "workdir": str(tools),
                "sandbox": "workspace-write",
                "timeout_sec": 3000,
            },
        )
    else:
        result = ctx.registry.call(
            "claude_exec",
            {"prompt": prompt, "workdir": str(tools), "timeout_sec": 3000},
        )

    metrics = {
        "compose_mode": "delegate",
        "executor": executor,
        "tokens_in": result.get("tokens_in", 0),
        "tokens_out": result.get("tokens_out", 0),
    }
    if result.get("cost_usd") is not None:
        metrics["cost_usd"] = result["cost_usd"]
    if result.get("dry_run") or result.get("executed") is False:
        return stepctx.fail(
            ctx,
            f"dry-run 下不会启动 {executor}；去掉 --dry-run 才能跑 delegate",
            metrics=metrics,
        )
    if not content_path.is_file():
        return stepctx.fail(
            ctx,
            f"{executor} 未写出 {content_path}（exit {result.get('returncode')}）",
            metrics=metrics,
        )
    content = json.loads(content_path.read_text(encoding="utf-8"))
    failures = _check(ctx, content, raw_path=raw_path, content_path=content_path)
    metrics.update(_metrics(content_path, raw_path))
    _accept_content(ctx, content)
    if failures:
        return stepctx.degrade(
            ctx,
            f"{executor} 产出未通过校验：" + "；".join(failures[:5]),
            artifacts=[str(content_path)],
            metrics=metrics,
        )
    return stepctx.finish(
        ctx,
        "ok",
        artifacts=[str(content_path)],
        metrics=metrics,
        notes=f"由 {executor} 完成撰写",
    )


# -- helpers ---------------------------------------------------------------
def _accept_content(ctx, content: dict) -> None:
    """Copy into the canonical content store unless this is an experiment arm."""
    if not ctx.tool_ctx.data.get("accept_content", True):
        return
    canonical = ctx.path("{tools_dir}") / "data" / f"{ctx.target_date}.json"
    ctx.registry.call(
        "fs_write",
        {
            "path": str(canonical),
            "content": json.dumps(content, ensure_ascii=False, indent=2),
        },
    )


def _check(ctx, content: dict, *, raw_path: Path, content_path: Path) -> list[str]:
    failures, _metrics_out = validators.check_structure(content)
    issues = [str(entry.get("issue")) for entry in failures]
    issues += [str(entry) for entry in validators.daily_trends_references(content_path).get("failures", [])]
    issues += [
        str(entry)
        for entry in validators.daily_trends_verifiable(content_path, raw_path).get("failures", [])
    ]
    return issues


def _metrics(content_path: Path, raw_path: Path) -> dict:
    structure = validators.daily_trends_structure(content_path)
    verifiable = validators.daily_trends_verifiable(content_path, raw_path)
    metrics = dict(structure.get("metrics", {}))
    metrics.update(verifiable.get("metrics", {}))
    return metrics


def _render_prompts(ctx, *, evidence: str) -> tuple[str, str]:
    prompt_dir = ctx.task_dir / str(ctx.task_raw.get("context", {}).get("prompt_dir", "prompts"))
    system = (prompt_dir / "system.md").read_text(encoding="utf-8")
    template = (prompt_dir / "compose.md").read_text(encoding="utf-8")
    user = (
        template.replace("{DATE}", ctx.target_date)
        .replace("{EVIDENCE}", evidence)
        .replace("{MEMORY}", stepctx.memory_block(ctx))
    )
    return system, user


def _delegation_prompt(ctx, *, raw_path: Path, content_path: Path, tools: Path) -> str:
    prompt_dir = ctx.task_dir / str(ctx.task_raw.get("context", {}).get("prompt_dir", "prompts"))
    system = (prompt_dir / "system.md").read_text(encoding="utf-8")
    template = (prompt_dir / "compose.md").read_text(encoding="utf-8")
    spec = (
        template.replace("{DATE}", ctx.target_date)
        .replace("{EVIDENCE}", f"（证据文件：{raw_path}，请自行读取）")
        .replace("{MEMORY}", "")
    )
    return "\n\n".join(
        [
            system,
            spec,
            "## 执行方式",
            f"1. 读取证据文件：{raw_path}",
            f"2. 按上述结构把最终 JSON 写入：{content_path}（只写 JSON 对象）",
            f"3. 内容数据目录：{tools}；不要修改 tools/ 下其他文件，不要执行 git 操作。",
            "4. 完成后只输出一行说明，不要粘贴 JSON 正文。",
        ]
    )


def _parse_json(text: str) -> dict | None:
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", cleaned, re.S)
    if fence:
        cleaned = fence.group(1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def build_evidence(raw: dict, *, strategy: str) -> str:
    return _prefiltered(raw) if strategy == "prefilter" else _full(raw)


def _full(raw: dict) -> str:
    text = json.dumps(raw, ensure_ascii=False, default=str)
    if len(text) > MAX_CHARS_FULL:
        text = text[:MAX_CHARS_FULL]
    return "```json\n" + text + "\n```"


def _prefiltered(raw: dict) -> str:
    """Pick the highest-signal rows instead of dumping the whole capture."""
    lines: list[str] = []

    def block(title: str, rows: list[dict]) -> None:
        if not rows:
            return
        lines.append(f"\n### {title}")
        for row in rows:
            lines.append(json.dumps(row, ensure_ascii=False, default=str)[:700])

    hn = sorted(raw.get("hn") or [], key=lambda r: -(r.get("points") or 0))[:25]
    block(
        "Hacker News (top 25 by points)",
        [
            {k: r.get(k) for k in ("title", "url", "points", "comments", "created_at")}
            for r in hn
        ],
    )

    gh = sorted(raw.get("github") or [], key=lambda r: -(r.get("stars_per_day") or 0))[:30]
    block(
        "GitHub (top 30 by stars/day)",
        [
            {
                k: r.get(k)
                for k in (
                    "full_name",
                    "html_url",
                    "description",
                    "stars",
                    "stars_per_day",
                    "language",
                    "created_at",
                )
            }
            for r in gh
        ],
    )

    arxiv = sorted(raw.get("arxiv") or [], key=lambda r: str(r.get("published", "")), reverse=True)[:20]
    block(
        "arXiv (20 newest)",
        [
            {
                "title": r.get("title"),
                "url": r.get("url"),
                "published": r.get("published"),
                "categories": r.get("categories"),
                "summary": str(r.get("summary", ""))[:400],
            }
            for r in arxiv
        ],
    )

    for name, rows in (raw.get("feeds") or {}).items():
        if not isinstance(rows, list) or not rows:
            continue
        block(
            f"feed:{name} (8 newest)",
            [
                {
                    "title": r.get("title"),
                    "url": r.get("url"),
                    "date": r.get("date") or r.get("published"),
                    "summary": str(r.get("summary", ""))[:300],
                }
                for r in rows[:8]
            ],
        )

    block(
        "Techmeme (top 12)",
        [
            {"title": r.get("title"), "url": r.get("url"), "hour": r.get("hour")}
            for r in (raw.get("techmeme") or [])[:12]
        ],
    )

    block(
        "X posts in window",
        [
            {
                "url": r.get("url"),
                "author": r.get("author"),
                "text": str(r.get("text", ""))[:300],
                "likes": r.get("likes"),
                "date": r.get("date"),
            }
            for r in (raw.get("tweets_recent") or [])[:20]
        ],
    )

    evergreen = sorted(
        raw.get("tweets_evergreen") or [], key=lambda r: -(r.get("likes") or 0)
    )[:15]
    block(
        "X evergreen (top 15 by likes)",
        [
            {
                "url": r.get("url"),
                "author": r.get("author"),
                "text": str(r.get("text", ""))[:300],
                "likes": r.get("likes"),
                "date": r.get("date"),
            }
            for r in evergreen
        ],
    )

    trends = (raw.get("x_trends") or {}).get("tech_trends") or []
    block(
        "X tech trends",
        [{"name": t.get("name"), "snapshots": t.get("snapshots")} for t in trends],
    )

    body = "\n".join(lines)
    body += "\n\n### 可引用的 URL 全集（references 的 url 必须来自这里）\n"
    body += "\n".join(sorted(_urls(raw))[:600])
    if len(body) > MAX_CHARS_PREFILTERED:
        body = body[:MAX_CHARS_PREFILTERED]
    return body


def _urls(node, acc: set[str] | None = None) -> set[str]:
    acc = acc if acc is not None else set()
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, str) and key in (
                "url",
                "html_url",
                "hn_url",
                "link",
                "source_url",
            ):
                if value.startswith(("http://", "https://")):
                    acc.add(value)
            else:
                _urls(value, acc)
    elif isinstance(node, list):
        for value in node:
            _urls(value, acc)
    return acc


if __name__ == "__main__":
    raise SystemExit(main())
