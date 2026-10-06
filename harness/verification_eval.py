"""Measure the acceptance gate itself.

An acceptance gate that has never been measured is a belief. This module builds
a labelled corpus from real captures plus synthetic attacks, runs matchers over
it, and reports the two numbers that decide whether the gate is worth anything:

    false_pass_rate   attacks it let through (the costly direction)
    false_fail_rate   legitimate inputs it wrongly rejected (the noisy direction)

Matchers are compared on the same corpus so a claim like "typed decisions beat
the previous heuristic" is a measurement, not an opinion.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from . import paths
from .decisions import Decision, canonical_url, classify_url
from .errors import HarnessError


def _default_data_dir() -> Path:
    """The sibling daily-trends checkout used by this task.

    The repository does not vendor that checkout, but it must not hard-code a
    single author's absolute home path either.
    """
    return paths.repo_root().parent / "daily-trends" / "data"


DEFAULT_DATA_DIR = str(_default_data_dir())

# Kinds, and what a correct gate must say about them.
KIND_LEGIT = "legit"
KIND_LEGIT_DRIFT = "legit_drift"
KIND_PREFIX_EXTENSION = "prefix_extension"
KIND_FABRICATED_SUFFIX = "fabricated_suffix"
KIND_UNRELATED = "unrelated"
# A real URL from a *different* day's capture. The prefix is genuine and the
# domain is trustworthy, so nothing about the string is obviously wrong — only
# the scoping is. This is the attack that a naively-implemented gate misses.
KIND_STALE_EVIDENCE = "stale_evidence"
# A URL that really is in today's capture but that the article never cited.
# It is a *should pass* sample: the gate can only prove provenance, and asking
# it to prove that a source supports a claim is asking for something else.
KIND_PLAUSIBLE_UNCITED = "plausible_uncited"

EXPECT_PASS = "pass"
EXPECT_NOT_PASS = "not_pass"

_KIND_EXPECT = {
    KIND_LEGIT: EXPECT_PASS,
    KIND_LEGIT_DRIFT: EXPECT_PASS,
    KIND_PLAUSIBLE_UNCITED: EXPECT_PASS,
    KIND_PREFIX_EXTENSION: EXPECT_NOT_PASS,
    KIND_FABRICATED_SUFFIX: EXPECT_NOT_PASS,
    KIND_STALE_EVIDENCE: EXPECT_NOT_PASS,
    KIND_UNRELATED: EXPECT_NOT_PASS,
}

# Classes that exist to bound the claim, not to score it. They are accounted for
# normally; the report calls them out so nobody reads a green number as
# "every citation is supported".
_LIMIT_KINDS = (KIND_PLAUSIBLE_UNCITED,)


@dataclass
class Sample:
    sample_id: str
    kind: str
    url: str
    day: str
    expect: str
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def data_dir() -> Path:
    return Path(os.environ.get("DAILY_TRENDS_DATA_DIR", DEFAULT_DATA_DIR))


# ---------------------------------------------------------------------------
# Corpus construction
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Any:
    target = Path(path)
    if not target.is_file():
        raise HarnessError(
            f"missing file: {target}. Set DAILY_TRENDS_DATA_DIR to the directory "
            "holding <date>.json and raw/<date>.json."
        )
    return json.loads(target.read_text(encoding="utf-8"))


def _raw_urls(day: str, *, root: Path | None = None) -> set[str]:
    raw = _load_json((root or data_dir()) / "raw" / f"{day}.json")
    from .validators import _collect_urls

    return {canonical_url(u) for u in _collect_urls(raw)}


def _cited_reference_urls(day: str, *, root: Path | None = None) -> list[str]:
    content = _load_json((root or data_dir()) / f"{day}.json")
    references = {ref.get("id"): ref for ref in content.get("references") or []}
    used: set[int] = set()
    for section in content.get("sections") or []:
        items = list(section.get("items") or [])
        for group in section.get("groups") or []:
            items.extend(group.get("items") or [])
        for item in items:
            used.update(s for s in (item.get("sources") or []) if isinstance(s, int))
    urls: list[str] = []
    for ref_id in sorted(used):
        ref = references.get(ref_id)
        if ref and ref.get("url"):
            urls.append(str(ref["url"]))
    return urls


def _path_of(canonical: str) -> str:
    """Canonical URL with the query stripped.

    Used when labelling cross-day negatives. The check must be *independent* of
    the matcher under test — calling classify_url here would make the corpus
    agree with the matcher by construction and no leak could ever be found. It
    is also deliberately conservative: if the same path is already in today's
    capture, the sample is not a valid negative and gets dropped.
    """
    return canonical.split("?", 1)[0]


def _drift_scheme(url: str) -> str:
    if url.startswith("https://"):
        return url.replace("https://", "http://", 1)
    if url.startswith("http://"):
        return url.replace("http://", "https://", 1)
    return url


def _drift_trailing_slash(url: str) -> str | None:
    """Add a trailing slash to the path.

    Only safe when there is no query string: inserting the slash before a query
    (`/x.html/?q` vs the real `/x.html?q`) produces a different URL shape rather
    than an enumerated drift, and labelling that "should pass" would be wrong.
    """
    if "?" in url:
        return None
    return url.rstrip("/") + "/"


def _drift_tracking(url: str) -> str:
    """Append a tracking param without clobbering an existing query string."""
    return url + ("&" if "?" in url else "?") + "utm_source=daily-trends"


def _drift_www(url: str) -> str:
    """Toggle the www prefix — in either direction, and only once."""
    if url.startswith(("https://www.", "http://www.")):
        return url.replace("://www.", "://", 1)
    return re.sub(r"^(https?://)", r"\1www.", url, count=1)


_DRIFT_BUILDERS: tuple[tuple[str, Callable[[str], str | None]], ...] = (
    ("scheme", _drift_scheme),
    ("trailing_slash", _drift_trailing_slash),
    ("tracking_params", _drift_tracking),
    ("www", _drift_www),
)


def build_corpus(
    days: Iterable[str],
    *,
    root: Path | None = None,
    max_per_kind_per_day: int = 25,
) -> dict[str, Any]:
    """Build a labelled corpus. Deterministic: same inputs, same corpus."""
    base = root or data_dir()
    known_by_day: dict[str, set[str]] = {}
    cited_by_day: dict[str, list[str]] = {}
    for day in days:
        if not (base / f"{day}.json").is_file() or not (
            base / "raw" / f"{day}.json"
        ).is_file():
            continue
        known_by_day[day] = _raw_urls(day, root=root)
        cited_by_day[day] = _cited_reference_urls(day, root=root)

    samples: list[Sample] = []
    used_days = list(known_by_day)

    for day, known in known_by_day.items():

        legit = [u for u in cited_by_day[day] if canonical_url(u) in known]
        cited_here = {canonical_url(u) for u in legit}
        known_paths = {_path_of(u) for u in known}
        for index, url in enumerate(legit[:max_per_kind_per_day]):
            samples.append(
                Sample(
                    sample_id=f"{day}-legit-{index:03d}",
                    kind=KIND_LEGIT,
                    url=url,
                    day=day,
                    expect=EXPECT_PASS,
                    note="a reference the published article actually cited",
                )
            )

        for index, url in enumerate(legit[:max_per_kind_per_day]):
            # Rotate the starting drift so kinds stay balanced, but fall through
            # when a drift is not applicable to this URL.
            chosen: tuple[str, str] | None = None
            for offset in range(len(_DRIFT_BUILDERS)):
                label, builder = _DRIFT_BUILDERS[(index + offset) % len(_DRIFT_BUILDERS)]
                candidate = builder(url)
                # The drift must be *textually* different (otherwise it is a
                # duplicate), but it is expected to canonicalise to the same
                # URL — that equivalence is what "enumerated drift" means.
                if candidate is not None and candidate != url:
                    chosen = (label, candidate)
                    break
            if chosen is None:
                continue
            label, drifted = chosen
            samples.append(
                Sample(
                    sample_id=f"{day}-drift-{index:03d}",
                    kind=KIND_LEGIT_DRIFT,
                    url=drifted,
                    day=day,
                    expect=EXPECT_PASS,
                    note=f"enumerated drift: {label}",
                )
            )

        # Attacks are built from real URLs, so the *prefix* is always genuine —
        # that is exactly why a prefix-based matcher waves them through.
        bases = sorted(u for u in known if len(u) > 20)[:max_per_kind_per_day]
        for index, real in enumerate(bases):
            attack = f"https://{real}/this-page-never-existed"
            if canonical_url(attack) in known:
                continue
            samples.append(
                Sample(
                    sample_id=f"{day}-ext-{index:03d}",
                    kind=KIND_PREFIX_EXTENSION,
                    url=attack,
                    day=day,
                    expect=EXPECT_NOT_PASS,
                    note="real URL + fabricated path",
                )
            )
        for index, real in enumerate(bases):
            attack = f"https://{real}-v99-fake"
            if canonical_url(attack) in known:
                continue
            samples.append(
                Sample(
                    sample_id=f"{day}-fab-{index:03d}",
                    kind=KIND_FABRICATED_SUFFIX,
                    url=attack,
                    day=day,
                    expect=EXPECT_NOT_PASS,
                    note="real URL + fabricated suffix",
                )
            )
        for index in range(min(5, max_per_kind_per_day)):
            samples.append(
                Sample(
                    sample_id=f"{day}-unrel-{index:03d}",
                    kind=KIND_UNRELATED,
                    url=f"https://example.invalid/{day}/invented-{index}",
                    day=day,
                    expect=EXPECT_NOT_PASS,
                    note="no relationship to the capture at all",
                )
            )

        # Real citation, wrong capture. Nothing about the URL is fabricated —
        # the domain, the path and the article all exist — only the day is wrong.
        stale_pool: list[tuple[str, str]] = []
        seen_stale: set[str] = set()
        for other in used_days:
            if other == day:
                continue
            for url in cited_by_day[other]:
                canonical = canonical_url(url)
                if canonical in known or canonical in cited_here:
                    continue
                if _path_of(canonical) in known_paths:
                    continue
                if canonical in seen_stale:
                    continue
                seen_stale.add(canonical)
                stale_pool.append((url, other))
        for index, (url, other) in enumerate(stale_pool[:max_per_kind_per_day]):
            samples.append(
                Sample(
                    sample_id=f"{day}-stale-{index:03d}",
                    kind=KIND_STALE_EVIDENCE,
                    url=url,
                    day=day,
                    expect=EXPECT_NOT_PASS,
                    note=f"genuinely cited on {other}, but not fetched on {day}",
                )
            )

        # Real URL from today that the article never used. The gate passes these
        # by design; they exist to put a number on what provenance cannot prove.
        uncited = sorted(u for u in known if u not in cited_here)
        for index, canonical in enumerate(uncited[:max_per_kind_per_day]):
            samples.append(
                Sample(
                    sample_id=f"{day}-uncited-{index:03d}",
                    kind=KIND_PLAUSIBLE_UNCITED,
                    url=f"https://{canonical}",
                    day=day,
                    expect=EXPECT_PASS,
                    note="in today's capture but never cited by the article",
                )
            )

    return {
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "days": used_days,
        "samples": [s.to_dict() for s in samples],
    }


def save_corpus(path: str | Path, corpus: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return target


def load_corpus(path: str | Path) -> dict[str, Any]:
    return _load_json(Path(path))


def default_corpus_path() -> Path:
    from . import paths

    return paths.runs_dir() / "verification" / "corpus.json"


def subsample(corpus: dict[str, Any], per_kind: int) -> dict[str, Any]:
    """Deterministically trim a corpus, keeping the kinds balanced.

    Used to cap how many samples a paid matcher is asked to judge. Every matcher
    in a comparison must see the *same* subset, or the numbers are not comparable.
    """
    if per_kind <= 0:
        return corpus
    kept: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for sample in corpus.get("samples", []):
        kind = sample["kind"]
        if seen.get(kind, 0) >= per_kind:
            continue
        seen[kind] = seen.get(kind, 0) + 1
        kept.append(sample)
    return {
        **corpus,
        "samples": kept,
        "subsampled": {"per_kind": per_kind, "from": len(corpus.get("samples", []))},
    }


def available_days(*, root: Path | None = None) -> list[str]:
    """Days that have both a composed article and its raw capture."""
    base = root or data_dir()
    raw_dir = base / "raw"
    if not raw_dir.is_dir():
        return []
    days: list[str] = []
    for raw_file in sorted(raw_dir.glob("*.json")):
        day = raw_file.stem
        if (base / f"{day}.json").is_file():
            days.append(day)
    return days


_SHORT_NAMES = {
    "daily_trends_structure": "structure",
    "daily_trends_references": "references",
    "daily_trends_verifiable": "verifiable",
    "daily_trends_no_duplicates": "duplicates",
    "daily_trends_brief": "brief",
    "daily_trends_depth": "depth",
    "daily_trends_coverage": "coverage",
}


def _short_validator_name(name: str) -> str:
    """Map a task-declared validator name to the short key used in this report."""
    return _SHORT_NAMES.get(name, name)


def content_debt(
    days: Iterable[str],
    *,
    root: Path | None = None,
    required: Iterable[str] | None = None,
    tools_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the content gates across days and report what is still broken.

    The gate blocks a run at the time it happens. That leaves history: days that
    were published before the gate got strict, and whose content the current
    gate would reject. This turns that backlog into one table instead of a
    discovery you make one failed run at a time.
    """
    from . import validators as validators_mod

    base = root or data_dir()
    rows: list[dict[str, Any]] = []
    for day in days:
        content = base / f"{day}.json"
        raw = base / "raw" / f"{day}.json"
        if not content.is_file() or not raw.is_file():
            continue
        structure = validators_mod.daily_trends_structure(content)
        references = validators_mod.daily_trends_references(content)
        verifiable = validators_mod.daily_trends_verifiable(content, raw)
        duplicates = validators_mod.daily_trends_no_duplicates(content)
        # tools 路径优先来自 task.toml（唯一事实来源）；只有没给时才从数据目录推断。
        tools_root = tools_dir or (root or data_dir()).parent
        brief = validators_mod.daily_trends_brief(content, tools_dir=tools_root)
        depth = validators_mod.daily_trends_depth(content, tools_dir=tools_root)
        coverage = validators_mod.daily_trends_coverage(
            content, raw, tools_dir=tools_root
        )
        structure_metrics = structure.get("metrics") or {}
        duplicate_metrics = duplicates.get("metrics") or {}
        brief_metrics = brief.get("metrics") or {}
        depth_metrics = depth.get("metrics") or {}
        coverage_metrics = coverage.get("metrics") or {}
        # `cannot_verify` on structure is the signature of the gate being stale
        # about the article's shape, which is a different problem from content
        # that actually violates the contract. Conflating the two is what made
        # this report say "3/16 clean" for nine days.
        gate_stale = structure.get("decision") == "cannot_verify"
        # 严重度只有一个事实来源：task.toml 里的 `required`。以前这里硬编码了一份
        # 名单，于是「required=false 的闸门」在 run 路径上是告警、在 release 路径上
        # 却可能被当成拦截，同一个闸门两种含义。
        evaluated = {
            "structure": structure,
            "references": references,
            "verifiable": verifiable,
            "duplicates": duplicates,
            "brief": brief,
            "depth": depth,
            "coverage": coverage,
        }
        # task.toml 写的是完整校验器名（daily_trends_structure），这里用短名分桶；
        # 两边名字不一致会让 `name in required_names` 永远为假 —— 于是所有天都
        # 「没有 blocker」，09-20 从 FAIL 变成 ok*。必须显式映射。
        required_names = (
            {_short_validator_name(name) for name in required}
            if required is not None
            else {"structure", "references", "verifiable", "duplicates"}
        )
        blockers = [
            name
            for name, result in evaluated.items()
            if name in required_names and not result.get("ok")
        ]
        rows.append(
            {
                "day": day,
                "structure": structure.get("ok"),
                "structure_decision": structure.get("decision"),
                "gate_stale": gate_stale,
                "shape": structure_metrics.get("shape"),
                "body_variants": structure_metrics.get("body_variants"),
                "unknown_sections": structure_metrics.get("unknown_sections"),
                "references": references.get("ok"),
                "verifiable": verifiable.get("ok"),
                "stale_citations": (verifiable.get("metrics") or {}).get("stale_citations"),
                "unknown_citations": (
                    (verifiable.get("metrics") or {}).get("fail", 0)
                    - (verifiable.get("metrics") or {}).get("stale_citations", 0)
                ),
                "duplicates": duplicates.get("ok"),
                "duplicate_pairs": duplicate_metrics.get("duplicate_pairs"),
                "borderline_pairs": duplicate_metrics.get("borderline_pairs"),
                "brief_entries": brief_metrics.get("entries"),
                "brief_excluded": brief_metrics.get("excluded"),
                "depth_limit_ratio": depth_metrics.get("limit_ratio"),
                "depth_without_substance": depth_metrics.get("without_substance"),
                "coverage_pool": coverage_metrics.get("pool"),
                "coverage_missed": coverage_metrics.get("missed"),
                "coverage_on_topic_missed": coverage_metrics.get("missed_on_topic"),
                "coverage_top3_missed": coverage_metrics.get("top3_missed"),
                "coverage_examples": [
                    w.get("title", "")[:48] for w in (coverage.get("warnings") or [])[:3]
                ],
                "verifiable_ratio": (verifiable.get("metrics") or {}).get(
                    "verifiable_ratio"
                ),
                "cannot_verify": (verifiable.get("metrics") or {}).get("cannot_verify"),
                "fail": (verifiable.get("metrics") or {}).get("fail"),
                "orphan_references": (references.get("metrics") or {}).get("orphans"),
                "warnings": len(references.get("warnings") or []),
                "blockers": blockers,
                "clean": not blockers,
            }
        )
    dirty = [row for row in rows if not row["clean"]]
    stale = [row for row in rows if row["gate_stale"]]
    warned = [row for row in rows if row["clean"] and row.get("warnings")]
    return {
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "days": len(rows),
        "clean": len(rows) - len(dirty),
        "dirty": len(dirty),
        "clean_with_warnings": len(warned),
        "warned": [row["day"] for row in warned],
        "gate_stale_days": len(stale),
        "gate_stale": [row["day"] for row in stale],
        "rows": rows,
    }


def content_debt_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 历史内容债",
        "",
        f"- 检查 {report['days']} 天：{report['clean']} 天通过全部内容闸门，{report['dirty']} 天未通过",
        f"- 生成时间：{report['generated_at']}",
        "",
        f"- 其中 {report.get('gate_stale_days', 0)} 天的结构闸门**不认识内容形状**"
        "（契约变化、闸门滞后），不是内容缺陷",
        f"- {report.get('clean_with_warnings', 0)} 天通过但有告警（如未引用的孤立参考文献）："
        "这类问题不拦发布，只记录，避免把闸门变成必须被绕过的噪声",
        "",
        "| 日期 | 形状 | 结构 | 引用 | 可核验 | 可核验率 | 重复 | 灰区 | 速读 | 深度 | 覆盖漏前3 | 孤儿引用 | 告警 | 卡在哪 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["rows"]:
        mark = lambda ok: "ok" if ok else "**FAIL**"  # noqa: E731
        ratio = row["verifiable_ratio"]
        depth = row.get("depth_limit_ratio")
        depth_text = "—" if depth is None else f"{depth:.2f}"
        variants = row.get("body_variants") or {}
        shape = (
            "未声明"
            if row.get("gate_stale")
            else ",".join(f"{k}:{v}" for k, v in variants.items()) or "—"
        )
        lines.append(
            f"| {row['day']} | {shape} | {mark(row['structure'])} | {mark(row['references'])} | "
            f"{mark(row['verifiable'])} | {'—' if ratio is None else f'{ratio:.2f}'} | "
            f"{row.get('duplicate_pairs') if row.get('duplicate_pairs') is not None else '—'} | "
            f"{row.get('borderline_pairs') if row.get('borderline_pairs') is not None else '—'} | "
            f"{row.get('brief_entries') if row.get('brief_entries') is not None else '—'} | "
            f"{depth_text} | "
            f"{row.get('coverage_top3_missed') if row.get('coverage_top3_missed') is not None else '—'} | "
            f"{row['orphan_references']} | {row.get('warnings', 0)} | "
            f"{', '.join(row['blockers']) or '—'} |"
        )
    lines += [
        "",
        "## 说明",
        "",
        "- 这张表只做诊断，不改已发布内容。是否回修历史稿件是编辑决定，不是工程决定。",
        "- `形状` 是闸门认出的内容形态（prose / fields / summary_comment）。显示「未声明」",
        "  表示产线换了形状而 `harness/content_schema.py` 还不知道它——那是契约漂移，",
        "  应当去补契约或改产线，而不是去修历史稿件。",
        "- `重复` 是同一篇里同一件事出现两次的硬重复（字符串可证）；`灰区` 是需要人工/模型",
        "  判断的相似对，闸门不做模型调用，所以只报数不判。",
        "- `速读` 是页面顶部「今日速读」选出的条数；`深度` 是正文里写了边界/限制的条目占比。",
        "  这两列衡量的是**读得懂吗**：结构全过但深度很低，说明这天只有事件没有信息量。",
        "- `可核验`列的失败信息会区分两种情况：引用的 URL 出现在**别的某一天**的抓取里",
        "  （产线用了陈旧素材，例如 10-05 引了 10-04 的 arXiv），或**任何一天都没有**",
        "  （编造风险更高）。这两种问题要分开修。",
        "- 可核验率下降有两种原因，需要分开看：引用确实不在当天抓取里，或当天的抓取文件",
        "  被后续运行覆盖过（`fetch` 曾经在 dry-run 下也执行）。后者属于可复现性事故。",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Matchers
# ---------------------------------------------------------------------------


def legacy_matcher(url: str, known: set[str]) -> str:
    """Reproduce the original boolean gate, faults included.

    Kept deliberately: an unflattering baseline measured on the same corpus is
    the only thing that turns "I improved it" into evidence.
    """
    from .validators import _url_in_raw

    return Decision.PASS.value if _url_in_raw(url, known) else Decision.FAIL.value


def typed_matcher(url: str, known: set[str]) -> str:
    return classify_url(url, known).decision.value


def llm_matcher(provider: Any, *, max_tokens: int = 8) -> Callable[[str, set[str]], str]:
    """Build a matcher backed by a chat provider.

    This is the LLM-as-judge baseline the industry is moving away from. It is
    wired the same way as the others so it lands in the same table — but it is
    never selected unless the caller passes --allow-llm.
    """

    usage: dict[str, Any] = {
        "calls": 0,
        "tokens_in": 0,
        "tokens_out": 0,
        "cost_usd": 0.0,
        "cost_known": False,
    }

    def match(url: str, known: set[str]) -> str:
        # The judge gets the *whole* capture. Truncating it would make the
        # baseline lose for the wrong reason and inflate the typed matcher's win.
        sample = "\n".join(sorted(known))
        try:
            response = provider.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "You verify citations. Given a cited URL and the URLs that "
                            "were actually fetched, reply with exactly one word: "
                            "PASS if the cited URL was fetched, "
                            "FAIL if it clearly was not, "
                            "CANNOT_VERIFY if the list is insufficient to decide."
                        ),
                    },
                    {"role": "user", "content": f"cited: {url}\n\nfetched:\n{sample}"},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
            )
        except Exception:  # noqa: BLE001 - one bad call must not abort the eval
            # An unavailable judge is an unverifiable judge, not a passing one.
            usage["calls"] += 1
            return Decision.CANNOT_VERIFY.value
        usage["calls"] += 1
        usage["tokens_in"] += int(response.tokens_in or 0)
        usage["tokens_out"] += int(response.tokens_out or 0)
        if response.cost_usd is not None:
            usage["cost_usd"] += float(response.cost_usd)
            usage["cost_known"] = True
        text = (response.text or "").strip().upper()
        for token, decision in (
            ("CANNOT_VERIFY", Decision.CANNOT_VERIFY),
            ("PASS", Decision.PASS),
            ("FAIL", Decision.FAIL),
        ):
            if token in text:
                return decision.value
        return Decision.CANNOT_VERIFY.value

    match.usage = usage  # type: ignore[attr-defined]
    return match


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score(
    corpus: dict[str, Any],
    matcher: Callable[[str, set[str]], str],
    *,
    root: Path | None = None,
    matcher_name: str = "matcher",
) -> dict[str, Any]:
    """Run one matcher over the corpus and return a confusion matrix."""
    usage = getattr(matcher, "usage", None)
    if isinstance(usage, dict):
        for key in ("calls", "tokens_in", "tokens_out"):
            usage[key] = 0
        usage["cost_usd"] = 0.0
        usage["cost_known"] = False

    known_cache: dict[str, set[str]] = {}
    per_kind: dict[str, dict[str, int]] = {}
    verdicts: list[dict[str, Any]] = []
    attacks = 0
    attacks_passed = 0
    legit = 0
    legit_rejected = 0
    uncertain = 0

    for sample in corpus.get("samples", []):
        day = sample["day"]
        if day not in known_cache:
            known_cache[day] = _raw_urls(day, root=root)
        known = known_cache[day]
        verdict = matcher(sample["url"], known)
        expected = sample["expect"]

        bucket = per_kind.setdefault(sample["kind"], {})
        bucket[verdict] = bucket.get(verdict, 0) + 1
        if verdict == Decision.CANNOT_VERIFY.value:
            uncertain += 1

        if expected == EXPECT_NOT_PASS:
            attacks += 1
            if verdict == Decision.PASS.value:
                attacks_passed += 1
        else:
            legit += 1
            if verdict == Decision.FAIL.value:
                legit_rejected += 1

        verdicts.append(
            {
                "sample_id": sample["sample_id"],
                "kind": sample["kind"],
                "expect": expected,
                "verdict": verdict,
                # Two different questions, deliberately kept apart:
                #   safe   — the gate neither leaked nor rejected real content
                #   strict — the gate committed to a definite verdict
                # CANNOT_VERIFY is safe against an attack but not strict; that is
                # exactly the trade-off the policy layer exists to price.
                "safe": (verdict == Decision.PASS.value)
                == (expected == EXPECT_PASS),
                "strict": (
                    verdict == Decision.PASS.value
                    if expected == EXPECT_PASS
                    else verdict == Decision.FAIL.value
                ),
            }
        )

    total = len(verdicts)
    limits = {
        kind: sum(1 for s in corpus.get("samples", []) if s["kind"] == kind)
        for kind in _LIMIT_KINDS
    }
    return {
        "matcher": matcher_name,
        "samples": total,
        "attacks": attacks,
        "legit": legit,
        "limit_samples": limits,
        "usage": dict(usage) if isinstance(usage, dict) else None,
        "false_pass_rate": round(attacks_passed / attacks, 4) if attacks else None,
        "false_pass_count": attacks_passed,
        "false_fail_rate": round(legit_rejected / legit, 4) if legit else None,
        "false_fail_count": legit_rejected,
        "cannot_verify_rate": round(uncertain / total, 4) if total else None,
        "per_kind": per_kind,
        "verdicts": verdicts,
    }


def compare_matchers(
    corpus: dict[str, Any],
    matchers: dict[str, Callable[[str, set[str]], str]],
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    results = [
        score(corpus, matcher, root=root, matcher_name=name)
        for name, matcher in matchers.items()
    ]
    return {
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "days": corpus.get("days", []),
        "samples": len(corpus.get("samples", [])),
        "corpus_fingerprint": corpus_fingerprint(corpus),
        "matchers": results,
    }


def corpus_fingerprint(corpus: dict[str, Any]) -> str:
    """Stable identity of the labelled corpus, not just its size."""
    rows = [
        f"{sample.get('sample_id')}|{sample.get('kind')}|{sample.get('day')}"
        for sample in corpus.get("samples", [])
    ]
    return hashlib.sha256("\n".join(sorted(rows)).encode("utf-8")).hexdigest()[:16]


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 验收闸门自身的测量",
        "",
        f"- 语料：{report['samples']} 条，来自 {', '.join(report['days']) or '（无）'}",
        f"- 生成时间：{report['generated_at']}",
        "",
        "| matcher | 漏检率 (false pass) | 误杀率 (false fail) | cannot_verify | 漏检/总数 | 成本 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for entry in report["matchers"]:
        fpr = entry["false_pass_rate"]
        ffr = entry["false_fail_rate"]
        cvr = entry["cannot_verify_rate"]
        usage = entry.get("usage") or {}
        cost = "免费" if not usage.get("calls") else (
            f"${usage['cost_usd']:.4f}"
            if usage.get("cost_known")
            else f"{usage.get('tokens_in', 0)} tok in"
        )
        lines.append(
            f"| {entry['matcher']} | "
            f"{'—' if fpr is None else f'{fpr:.1%}'} | "
            f"{'—' if ffr is None else f'{ffr:.1%}'} | "
            f"{'—' if cvr is None else f'{cvr:.1%}'} | "
            f"{entry['false_pass_count']}/{entry['attacks']} | {cost} |"
        )
    # Aggregate rates depend on the class mix: adding a class that every matcher
    # gets right dilutes the leak rate without anything improving. Per-kind is
    # the number that cannot be gamed that way.
    kinds = sorted({kind for entry in report["matchers"] for kind in entry["per_kind"]})
    if kinds:
        lines += [
            "",
            "### 分类别判决（避免用易样本稀释总漏检率）",
            "",
            "| matcher | " + " | ".join(kinds) + " |",
            "| --- | " + " | ".join("---" for _ in kinds) + " |",
        ]
        for entry in report["matchers"]:
            cells = []
            for kind in kinds:
                bucket = entry["per_kind"].get(kind, {})
                cells.append(
                    ", ".join(f"{name}×{count}" for name, count in sorted(bucket.items()))
                    or "—"
                )
            lines.append(f"| {entry['matcher']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 口径",
        "",
        "- 语料分七类：真实引用、可枚举漂移、真实但未被引用（应通过）；前缀延伸、后缀伪造、跨天取证、完全无关（应**不**通过）。",
        "- `漏检率` = 应拒绝的样本里被判为 PASS 的比例。这是有代价的方向：放行一条编造来源。",
        "- `误杀率` = 应通过的样本里被判为 FAIL 的比例。这是噪声方向：拦下真实内容。",
        "- `cannot_verify` 不计入漏检（它不是放行），但也**不是**通过——由任务策略决定是否阻断。",
        "- `成本`：确定性 matcher 免费；LLM 裁判按实际 token 计费，未配置单价时只报 token。",
        "- 判决的**果断程度**与准确率同等重要：同样是 0% 漏检，24 次 `FAIL` 和 16 次",
        "  `CANNOT_VERIFY` 对下游是完全不同的负担。",
        "- `plausible_uncited` 是**天花板样本**：URL 确实在当天抓取里、文章却从未引用它。",
        "  闸门判 PASS 是正确的——它证明的是 provenance（来源确实被抓到过），不是 support",
        "  （来源支撑了那句话）。要证明 support 需要另一层，不在本闸门的能力范围内。",
        "",
    ]
    return "\n".join(lines)


BASELINE_VERSION = 2

# Both directions are failures, and both are worse when they go up. A leak lets
# a fabricated source through; a false reject blocks a real one.
_REGRESSION_METRICS = ("false_pass_rate", "false_fail_rate")


def default_baseline_path() -> Path:
    from . import paths

    return paths.runs_dir() / "verification" / "baseline.json"


def baseline_from(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": BASELINE_VERSION,
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "samples": report.get("samples"),
        "days": report.get("days"),
        "corpus_fingerprint": report.get("corpus_fingerprint"),
        "matchers": {
            entry["matcher"]: {
                metric: entry.get(metric) for metric in _REGRESSION_METRICS
            }
            for entry in report.get("matchers", [])
        },
    }


def check_baseline(
    report: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    """Compare a fresh report against a frozen one.

    Guards the failure mode this whole module exists to prevent: a gate that
    quietly gets worse while every test still passes.
    """
    regressions: list[dict[str, Any]] = []
    baseline_fingerprint = baseline.get("corpus_fingerprint")
    current_fingerprint = report.get("corpus_fingerprint")
    if baseline_fingerprint and current_fingerprint and baseline_fingerprint != current_fingerprint:
        return {
            "ok": True,
            "corpus_changed": True,
            "regressions": [],
            "compared_matchers": [],
            "baseline_fingerprint": baseline_fingerprint,
            "current_fingerprint": current_fingerprint,
            "baseline_samples": baseline.get("samples"),
            "current_samples": report.get("samples"),
        }
    if not baseline_fingerprint and (
        baseline.get("samples") != report.get("samples")
        or baseline.get("days") != report.get("days")
    ):
        return {
            "ok": True,
            "corpus_changed": True,
            "regressions": [],
            "compared_matchers": [],
            "baseline_fingerprint": None,
            "current_fingerprint": current_fingerprint,
            "baseline_samples": baseline.get("samples"),
            "current_samples": report.get("samples"),
        }
    for entry in report.get("matchers", []):
        name = entry["matcher"]
        recorded = (baseline.get("matchers") or {}).get(name)
        if not recorded:
            continue
        for metric in _REGRESSION_METRICS:
            now, before = entry.get(metric), recorded.get(metric)
            if now is None or before is None:
                continue
            if now > before + 1e-9:
                regressions.append(
                    {
                        "matcher": name,
                        "metric": metric,
                        "baseline": before,
                        "now": now,
                    }
                )
    return {
        "ok": not regressions,
        "corpus_changed": False,
        "regressions": regressions,
        "compared_matchers": sorted(
            set(baseline.get("matchers") or {})
            & {entry["matcher"] for entry in report.get("matchers", [])}
        ),
    }


def save_baseline(path: str | Path, report: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(baseline_from(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target


def load_baseline(path: str | Path) -> dict[str, Any]:
    return _load_json(Path(path))


__all__ = [
    "Sample",
    "build_corpus",
    "save_corpus",
    "load_corpus",
    "default_corpus_path",
    "available_days",
    "legacy_matcher",
    "typed_matcher",
    "llm_matcher",
    "score",
    "compare_matchers",
    "markdown",
    "baseline_from",
    "check_baseline",
    "save_baseline",
    "load_baseline",
    "default_baseline_path",
]
