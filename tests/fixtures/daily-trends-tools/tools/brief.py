"""A test double for `daily-trends/tools/brief.py`.

Why this file exists instead of importing the real selector: the release-gate
and severity tests assert how the harness *reacts* to a brief, not how the
editorial selector chooses one. Those tests used to reach for the author's
checkout via an absolute path under `/Users/...`, which passed on exactly one
machine and made CI red for nine days.

So this is a deliberate contract double, not a copy: it implements only the
five functions the harness calls, with the payload shape the harness expects.
The real selector keeps its own tests in `tests/test_brief_and_depth.py`, which
skip when the checkout is absent.

If you change the harness's expectations of the brief module, change this file
with it — a double that drifts from the contract is worse than no double.
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
    for key in ("prose", "summary", "comment", "text"):
        value = item.get(key)
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            return {"zh": value, "en": value}
    return {"zh": "", "en": ""}


def iter_items(content: dict[str, Any]) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """Yield (section_id, location, item) for every item, groups included."""
    for section in content.get("sections") or []:
        section_id = str(section.get("id") or "?")
        for item in section.get("items") or []:
            yield section_id, section_id, item
        for group in section.get("groups") or []:
            group_id = str(group.get("id") or "?")
            for item in group.get("items") or []:
                yield section_id, f"{section_id}/{group_id}", item


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
