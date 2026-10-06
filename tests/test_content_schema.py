"""The content contract: known shapes validate, unknown shapes never pass."""

from __future__ import annotations

import unittest

from harness import content_schema as cs


def _v1_item(title="标题", sources=(1,)) -> dict:
    return {
        "title": {"zh": title, "en": "title"},
        "summary": {"zh": "摘要", "en": "summary"},
        "comment": {"zh": "点评", "en": "comment"},
        "sources": list(sources),
    }


def _v2_item() -> dict:
    return {
        "title": {"zh": "标题", "en": "title"},
        "fields": [
            {"label": {"zh": "摘要", "en": "Summary"}, "value": {"zh": "内容", "en": "body"}},
        ],
        "sources": [1],
    }


def _v3_item() -> dict:
    return {
        "title": {"zh": "标题", "en": "title"},
        "prose": {"zh": "正文", "en": "prose"},
        "sources": [1],
    }


def _article(*, item: dict | None = None, sections=None) -> dict:
    return {
        "date": "2026-01-01",
        "title": {"zh": "标题", "en": "title"},
        "summary": {"zh": "总览", "en": "overview"},
        "notes": {"zh": "口径", "en": "scope"},
        "sections": sections
        or [
            {
                "id": "insights",
                "groups": [
                    {"id": "agent-engineering", "title": {"zh": "组"}, "items": [item or _v1_item()]}
                ],
            },
            {"id": "github", "items": [item or _v1_item("repo")]},
        ],
        "references": [{"id": 1, "title": "src", "url": "https://example.com/a"}],
    }


class ShapeDetectionTests(unittest.TestCase):
    def test_all_three_body_variants_are_recognised(self) -> None:
        for item, expected in (
            (_v1_item(), cs.VARIANT_SUMMARY_COMMENT),
            (_v2_item(), cs.VARIANT_FIELDS),
            (_v3_item(), cs.VARIANT_PROSE),
        ):
            report = cs.check(_article(item=item))
            self.assertFalse(report.unknown_shape, report.reasons)
            self.assertEqual(report.view.variants, {expected: 2})
            self.assertEqual(report.failures, [])

    def test_papers_section_is_declared(self) -> None:
        article = _article()
        article["sections"].append({"id": "papers", "items": [_v3_item()]})
        report = cs.check(article)
        self.assertFalse(report.unknown_shape)
        self.assertEqual(report.metrics["papers"], 1)

    def test_undeclared_section_is_unknown_not_pass(self) -> None:
        article = _article()
        article["sections"].append({"id": "podcasts", "items": [_v3_item()]})
        report = cs.check(article)
        self.assertTrue(report.unknown_shape)
        self.assertIn("podcasts", " ".join(report.reasons))
        self.assertEqual(report.metrics["unknown_sections"], ["podcasts"])

    def test_unrecognised_item_body_is_unknown_not_pass(self) -> None:
        article = _article()
        article["sections"][0]["groups"][0]["items"] = [
            {"title": {"zh": "标题", "en": "title"}, "blurb": {"zh": "x", "en": "y"}, "sources": [1]}
        ]
        report = cs.check(article)
        self.assertTrue(report.unknown_shape)
        self.assertTrue(
            any("variant" in reason for reason in report.reasons), report.reasons
        )

    def test_missing_bilingual_text_is_a_failure(self) -> None:
        article = _article(item={**_v1_item(), "summary": {"zh": "只有中文"}})
        report = cs.check(article)
        self.assertFalse(report.unknown_shape)
        self.assertTrue(any("not bilingual" in str(f) for f in report.failures))

    def test_over_cap_section_is_a_failure(self) -> None:
        article = _article()
        article["sections"][1]["items"] = [_v1_item(f"repo{i}") for i in range(11)]
        report = cs.check(article)
        self.assertTrue(any("cap 10" in str(f) for f in report.failures), report.failures)

    def test_item_without_sources_is_a_failure(self) -> None:
        article = _article(item=_v1_item(sources=()))
        report = cs.check(article)
        self.assertTrue(any("no sources" in str(f) for f in report.failures))

    def test_unknown_top_level_keys_are_reported_not_fatal(self) -> None:
        article = _article()
        article["extra_thing"] = 1
        report = cs.check(article)
        self.assertFalse(report.unknown_shape)
        self.assertEqual(report.metrics["unknown_top_level_keys"], ["extra_thing"])

    def test_iter_items_locations_match_normalize(self) -> None:
        article = _article()
        locations = [entry[0] for entry in cs.iter_items(article)]
        view = cs.normalize(article)
        self.assertEqual(locations, [item.location for item in view.items])

    def test_drop_locations_removes_only_named_items(self) -> None:
        article = _article()
        view = cs.normalize(article)
        target = view.items[0].location
        removed = cs.drop_locations(article, [target])
        self.assertEqual(removed, 1)
        remaining = [item.location for item in cs.normalize(article).items]
        self.assertEqual(len(remaining), 1)
        self.assertNotIn(target, remaining)


if __name__ == "__main__":
    unittest.main()
