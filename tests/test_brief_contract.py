"""Check the fixture double against the real brief selector.

The release-gate tests run against `tests/fixtures/daily-trends-tools/tools/brief.py`
so they work on any machine. That buys portability at a price: a double that
drifts from the real module would let a release-blocking gate pass in tests while
failing in production. An independent review measured exactly that (the double
never applies the interest filter, so "every item filtered out" passes here and
fails for real).

This module pins the parts that can flip a verdict -- which item the gate sees,
and what counts as its body text -- and skips when the real checkout is absent,
the same way `test_brief_and_depth.py` does.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOUBLE = REPO / "tests" / "fixtures" / "daily-trends-tools" / "tools" / "brief.py"
REAL = REPO.parent / "daily-trends" / "tools" / "brief.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prose(title: str = "t") -> dict:
    return {"title": {"zh": title, "en": title}, "prose": {"zh": "正文", "en": "body"}, "sources": [1]}


def _fields(title: str = "t") -> dict:
    return {
        "title": {"zh": title, "en": title},
        "fields": [{"name": "why", "value": {"zh": "为什么", "en": "why"}}],
        "sources": [1],
    }


def _summary_comment(title: str = "t") -> dict:
    return {
        "title": {"zh": title, "en": title},
        "summary": {"zh": "摘要", "en": "summary"},
        "comment": {"zh": "点评", "en": "comment"},
        "sources": [1],
    }


def _content(items: list[dict]) -> dict:
    return {
        "date": "2026-01-01",
        "sections": [
            {"id": "insights", "groups": [{"id": "g", "items": items}]},
        ],
    }


class _DoubleOnly(unittest.TestCase):
    """Runs without the real checkout: the double must stand on its own."""

    def test_double_is_importable_and_shaped(self) -> None:
        double = _load(DOUBLE, "double_brief")
        payload = double.select_brief(_content([_prose("a"), _fields("b")]), limit=5)
        self.assertEqual(len(payload["entries"]), 2)
        for entry in payload["entries"]:
            self.assertTrue(entry["body"]["zh"])
            self.assertTrue(entry["why_zh"])


@unittest.skipUnless(REAL.is_file(), "requires the sibling daily-trends checkout")
class DoubleMatchesRealTest(unittest.TestCase):
    def setUp(self) -> None:
        self.double = _load(DOUBLE, "double_brief")
        self.real = _load(REAL, "real_brief")

    def test_bi_text_agrees_on_every_body_variant(self) -> None:
        for item in (_prose(), _fields(), _summary_comment()):
            self.assertEqual(
                self.double.bi_text(item),
                self.real.bi_text(item),
                f"bi_text diverged on {sorted(item)}",
            )

    def test_iter_items_agrees_on_order_and_locations(self) -> None:
        content = _content([_prose("a")])
        content["sections"].append({"id": "github", "items": [_prose("b")]})
        self.assertEqual(
            list(self.double.iter_items(content)),
            list(self.real.iter_items(content)),
        )

    def test_select_brief_sees_the_same_items_in_the_agreement_region(self) -> None:
        """The release tests use plain, on-topic, prose items -- pin that region."""
        content = _content([_prose(f"Agent harness 评测 {i}") for i in range(4)])
        mine = self.double.select_brief(content, limit=5)
        theirs = self.real.select_brief(content, limit=5)
        self.assertEqual(
            [entry["title"] for entry in mine["entries"]],
            [entry["title"] for entry in theirs["entries"]],
        )

    def test_the_known_divergence_is_real_and_documented(self) -> None:
        """Assert the gap exists, so it can only be closed on purpose.

        When every item is filtered out the real selector fails the gate (the
        page would ship without its 今日速读). The double passes because it has no
        interest profile. That gap is why release tests may not rely on filtering;
        if this ever starts passing, the double grew a filter and the file's
        divergence list needs updating.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tools = Path(tmp)
            (tools / "config").mkdir(parents=True)
            (tools / "tools").mkdir(parents=True)
            (tools / "tools" / "brief.py").write_text(DOUBLE.read_text(encoding="utf-8"), encoding="utf-8")
            (tools / "config" / "interests.json").write_text(
                json.dumps({"limit": 5, "exclude": [{"id": "x", "patterns": ["公司"]}]}),
                encoding="utf-8",
            )
            noise = _prose("某公司估值达 10 亿美元")
            self.assertTrue(self.double.select_brief(_content([noise]), limit=5)["entries"])
            self.assertFalse(self.real.select_brief(_content([noise]), limit=5)["entries"])


if __name__ == "__main__":
    unittest.main()
