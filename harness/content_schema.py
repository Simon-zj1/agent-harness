"""The article contract, written down.

Why this module exists
----------------------

For nine days the *producer* of the daily article changed the JSON shape while
the gate kept validating the oldest one:

    09-20 -> 09-26   title / summary / comment / sources   (v1)
    09-27 -> 10-01   title / fields[] / sources            (v2)
    10-02 -> 10-05   title / prose / sources               (v3) + papers section

Nothing failed loudly, because producer and gate are separate code paths:
`./agent verify content` simply reported "3/16 days clean", and every one of
those 13 "failures" was the gate being stale rather than the content being
wrong. A gate that reports failure for the wrong reason is worse than no gate,
because it trains the operator to ignore it.

The rule enforced here: **a shape the gate does not know is not a pass.** It is
CANNOT_VERIFY, naming the offending section or item, so the operator either
extends this module or regenerates the article. Undeclared *sections* are
treated the same way: adding a section to the producer without adding it here
is a contract change, and contract changes must be visible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# --- what the renderer actually supports -----------------------------------
#
# tools/render_site.py renders an item body through one of three variants,
# falling back in this order: prose -> fields -> summary(+comment). Those are
# the variants the site can show, so they are the variants the gate accepts.

VARIANT_PROSE = "prose"
VARIANT_FIELDS = "fields"
VARIANT_SUMMARY_COMMENT = "summary_comment"

BODY_VARIANTS = (VARIANT_PROSE, VARIANT_FIELDS, VARIANT_SUMMARY_COMMENT)

#: Item keys the renderer reads. Anything else is reported (not fatal) so an
#: accidental extra key is visible instead of silently ignored.
KNOWN_ITEM_KEYS = frozenset(
    {"title", "prose", "fields", "summary", "comment", "body", "sources", "meta"}
)


@dataclass(frozen=True)
class SectionSpec:
    id: str
    cap: int
    grouped: bool = False
    note: str = ""


#: Sections the published pages can render. `insights` is grouped; the others
#: are flat. Caps are declared limits, not observations: papers has been 8 in
#: practice and the cap leaves room to grow without leaving the gate behind.
SECTIONS: dict[str, SectionSpec] = {
    "insights": SectionSpec("insights", cap=20, grouped=True, note="行业热点，分四组"),
    "github": SectionSpec("github", cap=10, note="GitHub 热点"),
    "papers": SectionSpec("papers", cap=12, note="论文与基准，2026-09-29 起出现"),
}


@dataclass
class ItemView:
    """One item, normalised across every supported body variant."""

    location: str  # e.g. "insights/agent-engineering[3]" — for reports
    section: str
    group: str | None
    variant: str
    title_zh: str = ""
    title_en: str = ""
    body_zh: str = ""
    body_en: str = ""
    sources: list[int] = field(default_factory=list)
    extra_keys: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    @property
    def display_title(self) -> str:
        return (self.title_zh or self.title_en or self.location)[:60]

    def to_dict(self) -> dict[str, Any]:
        return {
            "location": self.location,
            "section": self.section,
            "group": self.group,
            "variant": self.variant,
            "title": self.display_title,
            "sources": self.sources,
            "extra_keys": self.extra_keys,
            "issues": self.issues,
        }


@dataclass
class ArticleView:
    date: str
    title_zh: str
    title_en: str
    summary_zh: str
    summary_en: str
    notes_zh: str
    notes_en: str
    section_ids: list[str]
    items: list[ItemView]
    references: list[dict[str, Any]]
    unknown_sections: list[str] = field(default_factory=list)
    unknown_top_level_keys: list[str] = field(default_factory=list)

    @property
    def variants(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.items:
            counts[item.variant] = counts.get(item.variant, 0) + 1
        return counts

    def items_in(self, section: str) -> list[ItemView]:
        return [item for item in self.items if item.section == section]


def _bi(block: Any) -> tuple[str, str]:
    """Read a bilingual block in any of the shapes the article has used."""
    if isinstance(block, str):
        return block.strip(), ""
    if isinstance(block, dict):
        return (
            str(block.get("zh", "") or "").strip(),
            str(block.get("en", "") or "").strip(),
        )
    return "", ""


def _iter_section_items(section: dict[str, Any]) -> Iterable[tuple[str | None, int, dict]]:
    index = 0
    for group in section.get("groups") or []:
        group_id = group.get("id")
        for item in group.get("items") or []:
            if isinstance(item, dict):
                index += 1
                yield group_id, index, item
    for item in section.get("items") or []:
        if isinstance(item, dict):
            index += 1
            yield None, index, item


def iter_items(content: dict[str, Any]) -> Iterable[tuple[str, str | None, list, int, dict]]:
    """Yield (location, group_id, container_list, index_in_container, item).

    `container_list` is the actual list object from the parsed JSON, so a caller
    that wants to drop an item can mutate it in place. Locations match the ones
    `normalize()` reports, which is what makes "drop these locations" safe.
    """
    for section in content.get("sections") or []:
        if not isinstance(section, dict):
            continue
        section_id = str(section.get("id") or "")
        index = 0
        for group in section.get("groups") or []:
            group_id = group.get("id")
            container = group.get("items") or []
            for position, item in enumerate(container):
                if not isinstance(item, dict):
                    continue
                index += 1
                yield (
                    section_id + (f"/{group_id}" if group_id else "") + f"[{index}]",
                    group_id,
                    container,
                    position,
                    item,
                )
        container = section.get("items") or []
        for position, item in enumerate(container):
            if not isinstance(item, dict):
                continue
            index += 1
            yield section_id + f"[{index}]", None, container, position, item


def drop_locations(content: dict[str, Any], locations: Iterable[str]) -> int:
    """Remove items by location, in place. Returns how many were removed."""
    wanted = set(locations)
    if not wanted:
        return 0
    found = [entry for entry in iter_items(content) if entry[0] in wanted]
    for _location, _group, container, position, item in found:
        try:
            container.remove(item)
        except ValueError:  # pragma: no cover - same object, so this cannot happen
            continue
    return len(found)


def _classify_item(item: dict[str, Any]) -> tuple[str, list[str], str, str]:
    """Return (variant, issues, body_zh, body_en) for one item."""
    issues: list[str] = []

    prose_zh, prose_en = _bi(item.get("prose"))
    if prose_zh or prose_en:
        if not (prose_zh and prose_en):
            issues.append("prose is not bilingual")
        return VARIANT_PROSE, issues, prose_zh, prose_en

    fields = item.get("fields")
    if isinstance(fields, list) and fields:
        zh_parts: list[str] = []
        en_parts: list[str] = []
        for entry in fields:
            if not isinstance(entry, dict):
                issues.append("fields entry is not an object")
                continue
            label_zh, label_en = _bi(entry.get("label"))
            value_zh, value_en = _bi(entry.get("value"))
            if label_zh and not label_en:
                issues.append(f"field label not bilingual: {label_zh[:24]}")
            if value_zh and not value_en:
                issues.append(f"field value not bilingual: {value_zh[:24]}")
            if value_zh:
                zh_parts.append(value_zh)
            if value_en:
                en_parts.append(value_en)
        body_zh = " ".join(zh_parts)
        body_en = " ".join(en_parts)
        if not (body_zh and body_en):
            issues.append("fields are not bilingual")
        return VARIANT_FIELDS, issues, body_zh, body_en

    summary_zh, summary_en = _bi(item.get("summary"))
    if summary_zh or summary_en:
        comment_zh, comment_en = _bi(item.get("comment"))
        if not (summary_zh and summary_en):
            issues.append("summary is not bilingual")
        if comment_zh or comment_en:
            if not (comment_zh and comment_en):
                issues.append("comment is not bilingual")
            return (
                VARIANT_SUMMARY_COMMENT,
                issues,
                (summary_zh + " " + comment_zh).strip(),
                (summary_en + " " + comment_en).strip(),
            )
        return VARIANT_SUMMARY_COMMENT, issues, summary_zh, summary_en

    body_zh, body_en = _bi(item.get("body"))
    if body_zh or body_en:
        if not (body_zh and body_en):
            issues.append("body is not bilingual")
        return VARIANT_PROSE, issues, body_zh, body_en

    return "unknown", issues + ["no recognised body variant (prose/fields/summary)"], "", ""


def normalize(content: dict[str, Any]) -> ArticleView:
    """Build the normalised view. Never raises on unknown shapes."""
    title_zh, title_en = _bi(content.get("title"))
    summary_zh, summary_en = _bi(content.get("summary"))
    notes_zh, notes_en = _bi(content.get("notes"))

    section_ids: list[str] = []
    unknown_sections: list[str] = []
    items: list[ItemView] = []
    for section in content.get("sections") or []:
        if not isinstance(section, dict):
            continue
        section_id = str(section.get("id") or "")
        section_ids.append(section_id)
        if section_id not in SECTIONS:
            unknown_sections.append(section_id or "<missing id>")
        for group_id, index, item in _iter_section_items(section):
            variant, issues, body_zh, body_en = _classify_item(item)
            item_title_zh, item_title_en = _bi(item.get("title"))
            if item_title_zh and not item_title_en:
                issues.append("title is not bilingual")
            location = section_id + (f"/{group_id}" if group_id else "") + f"[{index}]"
            sources = [s for s in (item.get("sources") or []) if isinstance(s, int)]
            if not sources:
                issues.append("no sources")
            items.append(
                ItemView(
                    location=location,
                    section=section_id,
                    group=group_id,
                    variant=variant,
                    title_zh=item_title_zh,
                    title_en=item_title_en,
                    body_zh=body_zh,
                    body_en=body_en,
                    sources=sources,
                    extra_keys=sorted(set(item.keys()) - KNOWN_ITEM_KEYS),
                    issues=issues,
                )
            )

    known_top = {
        "date",
        "generated_at",
        "title",
        "summary",
        "tldr",
        "sections",
        "references",
        "notes",
        "stats",
    }
    return ArticleView(
        date=str(content.get("date") or ""),
        title_zh=title_zh,
        title_en=title_en,
        summary_zh=summary_zh,
        summary_en=summary_en,
        notes_zh=notes_zh,
        notes_en=notes_en,
        section_ids=section_ids,
        items=items,
        references=[r for r in (content.get("references") or []) if isinstance(r, dict)],
        unknown_sections=unknown_sections,
        unknown_top_level_keys=sorted(set(content.keys()) - known_top),
    )


@dataclass
class ContractReport:
    """What the gate understood about this article's shape."""

    view: ArticleView | None
    unknown_shape: bool
    reasons: list[str] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def has_failures(self) -> bool:
        return bool(self.failures)


def check(content: dict[str, Any]) -> ContractReport:
    """Validate structure against the declared contract.

    Failures are things the contract *forbids* (missing bilingual text,
    over-cap sections, items without sources). `unknown_shape=True` is for
    things the contract does not describe (an undeclared section, an unknown
    item body variant). The caller decides how unknown shape is treated;
    `daily_trends_structure` treats it as CANNOT_VERIFY, never as PASS.
    """
    view = normalize(content)
    failures: list[dict[str, Any]] = []
    reasons: list[str] = []

    if view.unknown_sections:
        reasons.append(
            "undeclared section(s): "
            + ", ".join(view.unknown_sections)
            + " — add them to harness/content_schema.SECTIONS or stop emitting them"
        )
    unknown_variant_items = [item for item in view.items if item.variant == "unknown"]
    if unknown_variant_items:
        reasons.append(
            "unrecognised item body variant in "
            + ", ".join(item.location for item in unknown_variant_items[:5])
            + " — supported variants: "
            + ", ".join(BODY_VARIANTS)
        )

    if not view.date:
        failures.append({"issue": "missing date"})
    for label, zh, en in (
        ("title", view.title_zh, view.title_en),
        ("summary", view.summary_zh, view.summary_en),
        ("notes", view.notes_zh, view.notes_en),
    ):
        if not zh:
            failures.append({"issue": f"{label}.zh is empty"})
        if not en:
            failures.append({"issue": f"{label}.en is empty"})

    if len(view.section_ids) < 2:
        failures.append({"issue": f"expected 2 sections, found {len(view.section_ids)}"})

    per_section: dict[str, int] = {}
    for item in view.items:
        per_section[item.section] = per_section.get(item.section, 0) + 1
        for issue in item.issues:
            failures.append({"issue": f"item {issue}", "title": item.display_title})

    for section_id, spec in SECTIONS.items():
        count = per_section.get(section_id, 0)
        if count > spec.cap:
            failures.append({"issue": f"{section_id} has {count} items (cap {spec.cap})"})

    for section in content.get("sections") or []:
        if not isinstance(section, dict):
            continue
        if section.get("id") == "insights" and not (section.get("groups") or []):
            failures.append({"issue": "insights section has no groups"})
        for group in section.get("groups") or []:
            if not group.get("id"):
                failures.append({"issue": "insights group without id"})
            group_title_zh, _ = _bi(group.get("title"))
            if not group_title_zh:
                failures.append({"issue": f"group {group.get('id')} has no title"})

    if not view.references:
        failures.append({"issue": "no references"})

    metrics = {
        "shape": "unknown" if reasons else "known",
        "sections": view.section_ids,
        "unknown_sections": view.unknown_sections,
        "body_variants": view.variants,
        "items": len(view.items),
        "insights": per_section.get("insights", 0),
        "repos": per_section.get("github", 0),
        "papers": per_section.get("papers", 0),
        "references": len(view.references),
        "item_issues": sum(len(item.issues) for item in view.items),
        "unknown_top_level_keys": view.unknown_top_level_keys,
    }
    return ContractReport(
        view=view,
        unknown_shape=bool(reasons),
        reasons=reasons,
        failures=failures,
        metrics=metrics,
    )


def load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
