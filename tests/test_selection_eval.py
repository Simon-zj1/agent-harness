"""The selection sheet: balanced candidates, empty gold, honest scoring."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from harness import selection_eval


def _content() -> dict:
    def item(title: str, ref: int) -> dict:
        return {
            "title": {"zh": title, "en": title},
            "prose": {"zh": "正文", "en": "body"},
            "sources": [ref],
        }

    return {
        "date": "2026-01-01",
        "title": {"zh": "标题", "en": "title"},
        "summary": {"zh": "总览", "en": "overview"},
        "notes": {"zh": "口径", "en": "scope"},
        "sections": [
            {"id": "insights", "groups": [{"id": "g", "items": [item(f"选中{i}", i) for i in (1, 2, 3)]}]},
            {"id": "github", "items": [item("repo", 4)]},
        ],
        "references": [
            {"id": 1, "title": "a", "url": "https://example.com/1"},
            {"id": 2, "title": "b", "url": "https://example.com/2"},
            {"id": 3, "title": "c", "url": "https://example.com/3"},
            {"id": 4, "title": "d", "url": "https://example.com/4"},
        ],
    }


def _raw() -> dict:
    return {
        "hn": [
            {"title": f"hn {i}", "url": f"https://news.example/{i}", "points": 100 - i}
            for i in range(6)
        ],
        "github": [
            {"full_name": f"org/repo{i}", "html_url": f"https://github.com/org/repo{i}", "stars_per_day": 10 - i}
            for i in range(4)
        ],
        "arxiv": [{"title": "paper", "url": "https://arxiv.org/abs/1", "published": "2026-01-01"}],
        "feeds": {"wired": [{"title": "feed item", "url": "https://wired.example/1", "date": "2026-01-01"}]},
    }


class SelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "raw").mkdir(parents=True)
        (self.root / "2026-01-01.json").write_text(
            json.dumps(_content(), ensure_ascii=False), encoding="utf-8"
        )
        (self.root / "raw" / "2026-01-01.json").write_text(
            json.dumps(_raw(), ensure_ascii=False), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_sheet_is_balanced_and_gold_is_empty(self) -> None:
        rows = selection_eval.build_sheet(["2026-01-01"], sample=10, root=self.root)
        buckets = {}
        for row in rows:
            buckets[row.bucket] = buckets.get(row.bucket, 0) + 1
            self.assertEqual(row.gold, "", "gold must ship empty — only a human fills it")
            self.assertTrue(row.draft, "a draft suggestion should exist")
            self.assertTrue(row.case_id)
        self.assertEqual(buckets.get("selected"), 4)
        self.assertEqual(buckets.get("unselected"), 6)

    def test_unselected_rows_exclude_what_the_article_cited(self) -> None:
        rows = selection_eval.build_sheet(["2026-01-01"], sample=20, root=self.root)
        cited = {f"https://example.com/{i}" for i in range(1, 5)}
        unselected = [row for row in rows if row.bucket == "unselected"]
        self.assertTrue(unselected)
        for row in unselected:
            self.assertNotIn(row.url, cited)

    def test_sheet_round_trips_through_csv_and_jsonl(self) -> None:
        rows = selection_eval.build_sheet(["2026-01-01"], sample=6, root=self.root)
        written = selection_eval.write_sheet(rows, self.root / "out")
        for path in written.values():
            loaded = selection_eval.load_sheet(path)
            self.assertEqual(len(loaded), len(rows))
            self.assertIn("gold", loaded[0])

    def test_unlabelled_sheet_produces_no_rates(self) -> None:
        rows = selection_eval.build_sheet(["2026-01-01"], sample=6, root=self.root)
        report = selection_eval.score_sheet([row.to_dict() for row in rows])
        self.assertEqual(report["labelled"], 0)
        self.assertIsNone(report["precision"])
        self.assertIsNone(report["recall"])

    def test_labelled_sheet_scores_the_pipelines_choices(self) -> None:
        rows = selection_eval.build_sheet(["2026-01-01"], sample=10, root=self.root)
        payload = [row.to_dict() for row in rows]
        selected = [row for row in payload if row["bucket"] == "selected"]
        unselected = [row for row in payload if row["bucket"] == "unselected"]
        # human verdict: all four published items were right; two skipped ones were wrong
        for row in selected:
            row["gold"] = "select"
        for row in unselected[:2]:
            row["gold"] = "select"
        for row in unselected[2:]:
            row["gold"] = "reject"
        report = selection_eval.score_sheet(payload)
        self.assertEqual(report["counts"], {"tp": 4, "fp": 0, "fn": 2, "tn": 4})
        self.assertEqual(report["precision"], 1.0)
        self.assertAlmostEqual(report["recall"], 4 / 6)

    def test_either_is_excluded_from_the_rates(self) -> None:
        rows = [row.to_dict() for row in selection_eval.build_sheet(["2026-01-01"], sample=6, root=self.root)]
        rows[0]["gold"] = "either"
        rows[1]["gold"] = "select"
        report = selection_eval.score_sheet(rows)
        self.assertEqual(report["either"], 1)
        self.assertEqual(report["labelled"], 1)

    def test_markdown_names_the_thin_evidence_caveat(self) -> None:
        rows = [row.to_dict() for row in selection_eval.build_sheet(["2026-01-01"], sample=6, root=self.root)]
        rows[0]["gold"] = "select"
        text = selection_eval.markdown(selection_eval.score_sheet(rows))
        self.assertIn("标注少于 30 条", text)


if __name__ == "__main__":
    unittest.main()
