"""Chunked composition: merging per-chunk citations into one validated article."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

COMPOSE = (
    Path(__file__).resolve().parent.parent
    / "tasks"
    / "daily-trends"
    / "steps"
    / "20_compose.py"
)


def _load_compose():
    spec = importlib.util.spec_from_file_location("daily_trends_compose", COMPOSE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["daily_trends_compose"] = module
    spec.loader.exec_module(module)
    return module


compose = _load_compose()


def _raw_index(*urls: str) -> dict[str, str]:
    return {compose._normalise_url(url): url for url in urls}


def _item(local_ids, *, zh_url: str | None = None, en_url: str | None = None) -> dict:
    summary = {"zh": "摘要", "en": "summary"}
    if zh_url:
        summary["zh"] = f"摘要 [[{zh_url}]]"
    if en_url:
        summary["en"] = f"summary [[{en_url}]]"
    return {
        "title": {"zh": "标题", "en": "title"},
        "summary": summary,
        "comment": {"zh": "点评", "en": "comment"},
        "sources": list(local_ids),
    }


OVERVIEW = {
    "title": {"zh": "每日技术趋势 · 2026-09-22", "en": "Daily Tech Trends · 2026-09-22"},
    "summary": {"zh": "总览", "en": "overview"},
    "tldr": {"zh": ["a", "b"], "en": ["a", "b"]},
    "notes": {"zh": "口径", "en": "scope"},
    "section_intros": {"insights": {"zh": "一节", "en": "one"}, "github": {"zh": "二节", "en": "two"}},
}


class MergeTests(unittest.TestCase):
    def _merge(self, *, group_payloads, github_payload, raw_urls):
        return compose._merge_chunks(
            date="2026-09-22",
            overview=OVERVIEW,
            group_payloads=group_payloads,
            github_payload=github_payload,
            raw_urls=raw_urls,
        )

    def test_local_citation_numbers_become_global_and_text_markers_follow(self) -> None:
        raw = _raw_index("https://example.com/a", "https://example.com/b")
        payload = {
            "items": [_item([1], zh_url=1, en_url=1)],
            "references": [{"id": 1, "title": "A", "url": "https://example.com/a", "source": "X", "date": "2026-09-22"}],
        }
        content, notes = self._merge(
            group_payloads={"agent-engineering": payload, "robotics": {"items": [], "references": []},
                            "ai-productivity": {"items": [], "references": []},
                            "industry-moves": {"items": [], "references": []}},
            github_payload={"items": [], "references": []},
            raw_urls=raw,
        )
        self.assertEqual(notes, [])
        item = content["sections"][0]["groups"][0]["items"][0]
        self.assertEqual(item["sources"], [1])
        self.assertIn("[[1]]", item["summary"]["zh"])
        self.assertEqual(content["references"], [
            {"id": 1, "url": "https://example.com/a", "title": "A", "source": "X", "date": "2026-09-22"}
        ])
        self.assertEqual(content["stats"]["insights"], 1)

    def test_citations_that_do_not_come_from_the_capture_are_dropped(self) -> None:
        raw = _raw_index("https://example.com/a")
        payload = {
            "items": [_item([1, 2], zh_url=2)],
            "references": [
                {"id": 1, "title": "A", "url": "https://example.com/a", "source": "X", "date": "2026-09-22"},
                {"id": 2, "title": "Fabricated", "url": "https://fabricated.invalid/x", "source": "?", "date": ""},
            ],
        }
        content, notes = self._merge(
            group_payloads={"agent-engineering": payload, "robotics": {"items": [], "references": []},
                            "ai-productivity": {"items": [], "references": []},
                            "industry-moves": {"items": [], "references": []}},
            github_payload={"items": [], "references": []},
            raw_urls=raw,
        )
        self.assertTrue(any("无法回溯" in note for note in notes), notes)
        item = content["sections"][0]["groups"][0]["items"][0]
        self.assertEqual(item["sources"], [1])
        self.assertNotIn("[[", item["summary"]["zh"])
        locs = [ref["url"] for ref in content["references"]]
        self.assertEqual(locs, ["https://example.com/a"])

    def test_items_without_any_verifiable_source_are_dropped(self) -> None:
        payload = {
            "items": [_item([1])],
            "references": [{"id": 1, "title": "bad", "url": "https://nope.invalid/1", "source": "?", "date": ""}],
        }
        content, notes = self._merge(
            group_payloads={"agent-engineering": payload, "robotics": {"items": [], "references": []},
                            "ai-productivity": {"items": [], "references": []},
                            "industry-moves": {"items": [], "references": []}},
            github_payload={"items": [], "references": []},
            raw_urls=_raw_index("https://example.com/a"),
        )
        self.assertEqual(content["sections"][0]["groups"][0]["items"], [])
        self.assertEqual(content["references"], [])
        self.assertTrue(any("没有任何可核验来源" in note for note in notes), notes)

    def test_github_items_keep_their_meta_block(self) -> None:
        raw = _raw_index("https://github.com/x/y")
        payload = {
            "items": [{**_item([1]), "meta": {"stars": 10, "stars_per_day": 2.0, "language": "Go"}}],
            "references": [{"id": 1, "title": "repo", "url": "https://github.com/x/y", "source": "GitHub", "date": "2026-09-22"}],
        }
        content, _ = self._merge(
            group_payloads={gid: {"items": [], "references": []} for gid, _, _ in compose.INSIGHTS_GROUPS},
            github_payload=payload,
            raw_urls=raw,
        )
        github_section = next(s for s in content["sections"] if s["id"] == "github")
        self.assertEqual(github_section["items"][0]["meta"]["language"], "Go")
        self.assertEqual(content["stats"]["repos"], 1)

    def test_scaffolding_is_fixed_even_when_overview_is_thin(self) -> None:
        content, _ = compose._merge_chunks(
            date="2026-09-22",
            overview={},
            group_payloads={gid: {"items": [], "references": []} for gid, _, _ in compose.INSIGHTS_GROUPS},
            github_payload={"items": [], "references": []},
            raw_urls={},
        )
        self.assertEqual([s["id"] for s in content["sections"]], ["insights", "github"])
        self.assertEqual(
            [g["id"] for g in content["sections"][0]["groups"]],
            [gid for gid, _, _ in compose.INSIGHTS_GROUPS],
        )
        self.assertEqual(content["date"], "2026-09-22")

    def test_caps_are_enforced_instead_of_trusting_the_model(self) -> None:
        raw = _raw_index("https://example.com/a")
        payload = {
            "items": [_item([1]) for _ in range(8)],
            "references": [
                {"id": 1, "title": "A", "url": "https://example.com/a", "source": "X", "date": "2026-09-22"}
            ],
        }
        github_payload = {
            "items": [_item([1]) for _ in range(14)],
            "references": [
                {"id": 1, "title": "A", "url": "https://example.com/a", "source": "X", "date": "2026-09-22"}
            ],
        }
        content, notes = self._merge(
            group_payloads={gid: payload for gid, _, _ in compose.INSIGHTS_GROUPS},
            github_payload=github_payload,
            raw_urls=raw,
        )
        insight_total = sum(len(g["items"]) for g in content["sections"][0]["groups"])
        self.assertEqual(insight_total, compose.MAX_INSIGHTS_TOTAL)
        self.assertEqual(len(content["sections"][1]["items"]), compose.MAX_REPOS)
        self.assertEqual(content["stats"]["insights"], compose.MAX_INSIGHTS_TOTAL)
        self.assertEqual(content["stats"]["repos"], compose.MAX_REPOS)
        self.assertTrue(any("裁剪" in note for note in notes), notes)
        self.assertEqual(
            [g["id"] for g in content["sections"][0]["groups"] if g["items"]],
            [gid for gid, _, _ in compose.INSIGHTS_GROUPS],
        )


if __name__ == "__main__":
    unittest.main()
