"""A test double for `daily-trends/tools/brief.py`.

Why this file exists instead of importing the real selector: the release-gate
and severity tests assert how the harness *reacts* to a brief, not how the
editorial selector chooses one. Those tests used to reach for the author's
checkout via an absolute path under `/Users/...`, which passed on exactly one
machine and made CI red for nine days.

So this is a deliberate contract double, not a copy. The real selector keeps its
own tests in `tests/test_brief_and_depth.py`, which skip when the checkout is
absent.

KNOWN DIVERGENCES from the real module. An independent review measured all four
against the real checkout, so they are listed rather than discovered again:

1. `select_brief` does not apply the interest profile. When every item would be
   filtered out, the real module fails the gate and this double passes. Do not
   write a test here that depends on filtering — use the real module.
2. `depth_signals` uses its own marker lists and does not implement
   `extract_limit`. `daily_trends_depth` is advisory (`required = false`), so a
   divergence changes a warning count, not a release decision.
3. `load_interests` / `topic_hits_loose` are absent, so `daily_trends_coverage`
   falls back to `topic_rank = off` instead of `brief`.

`bi_text` and `iter_items` are kept faithful on purpose: they decide which item
the gate sees, which can flip a release-blocking verdict.

If you change the harness's expectations of the brief module, change this file
with it — a double that drifts from the contract is worse than no double.
`tests/test_brief_contract.py` checks this file against the real module whenever
the checkout is present.
"""

from __future__ import annotations

from typing import Any, Iterator

#: Two words that make a summary count as "deep" in the real selector.
_MECHANISM = ("通过", "采用", "实现", "机制", "because", "by ", "using")
_LIMIT = ("限制", "不足", "仅在", "但", "however", "only", "limit")


def bi_title(item: dict[str, Any]) -> dict[str, str]:
    value = item.get("title")
    return value if isinstance(value, dict) else {"zh": str(value or ""), "en": str(value or "")}


def bi_text(item: dict[str, Any]) -> dict[str, str]:
    """Faithful to the real module's order: prose -> body -> summary(+comment) -> fields."""
    for key in ("prose", "body", "summary"):
        block = item.get(key)
        if isinstance(block, dict) and (block.get("zh") or block.get("en")):
            zh = str(block.get("zh", ""))
            en = str(block.get("en", ""))
            comment = item.get("comment")
            if key == "summary" and isinstance(comment, dict):
                zh = (zh + " " + str(comment.get("zh", ""))).strip()
                en = (en + " " + str(comment.get("en", ""))).strip()
            return {"zh": zh, "en": en}
    zh_parts: list[str] = []
    en_parts: list[str] = []
    for field in item.get("fields") or []:
        value = field.get("value") if isinstance(field, dict) else None
        if isinstance(value, dict):
            zh_parts.append(str(value.get("zh", "")))
            en_parts.append(str(value.get("en", "")))
    return {"zh": " ".join(p for p in zh_parts if p), "en": " ".join(p for p in en_parts if p)}


def iter_items(content: dict[str, Any]) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """Yield (section_id, location, item) in document order, groups first.

    Order matters: `select_brief` takes the first ``limit`` items, so a double
    that walks the document differently selects different items.
    """
    for section in content.get("sections") or []:
        section_id = str(section.get("id") or "?")
        index = 0
        for group in section.get("groups") or []:
            group_id = str(group.get("id") or "?")
            for item in group.get("items") or []:
                index += 1
                yield section_id, f"{section_id}/{group_id}[{index}]", item
        for item in section.get("items") or []:
            index += 1
            yield section_id, f"{section_id}[{index}]", item


def depth_signals(text: str) -> dict[str, bool]:
    lowered = (text or "").lower()
    return {
        "has_substance": any(word.lower() in lowered for word in _MECHANISM),
        "has_limit": any(word.lower() in lowered for word in _LIMIT),
    }


def select_brief(content: dict[str, Any], *, limit: int = 5) -> dict[str, Any]:
    """Pick up to ``limit`` items, shaped the way the gate expects."""
    rows = list(iter_items(content))
    entries: list[dict[str, Any]] = []
    for _section, _location, item in rows[:limit]:
        entries.append(
            {
                "title": bi_title(item),
                "body": bi_text(item),
                "why_zh": "stub: matched the interest profile",
            }
        )
    return {
        "entries": entries,
        "candidates": len(rows),
        "excluded": 0,
        "without_limit": 0,
    }
