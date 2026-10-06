"""Validators are the acceptance gate, so they get the strictest tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from harness import taskspec, validators

# Frozen fixture rather than the live capture. The live file is rewritten by the
# fetch step, and a test that silently changes its own ground truth is worse than
# no test: an earlier version of this suite started failing only because a
# dry-run had refreshed the capture underneath it.
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "daily-trends-2026-09-25"
REAL_CONTENT = FIXTURES / "content.json"
REAL_RAW = FIXTURES / "raw.json"


def _content() -> dict:
    return json.loads(REAL_CONTENT.read_text(encoding="utf-8"))


class ValidatorTests(unittest.TestCase):
    def test_real_content_passes_structure_and_references(self) -> None:
        result = validators.daily_trends_structure(REAL_CONTENT)
        self.assertTrue(result["ok"], result.get("failures"))
        refs = validators.daily_trends_references(REAL_CONTENT)
        self.assertTrue(refs["ok"], refs.get("failures"))
        self.assertGreaterEqual(result["metrics"]["insights"], 1)
        self.assertGreaterEqual(result["metrics"]["repos"], 1)

    def test_real_content_is_traceable_to_raw(self) -> None:
        result = validators.daily_trends_verifiable(REAL_CONTENT, REAL_RAW)
        self.assertIn("verifiable_ratio", result["metrics"])
        self.assertGreater(result["metrics"]["verifiable_ratio"], 0.9, result.get("failures"))

    def test_missing_bilingual_field_is_rejected(self) -> None:
        content = _content()
        del content["sections"][0]["groups"][0]["items"][0]["summary"]["en"]
        failures, metrics = validators.check_structure(content)
        self.assertTrue(any("summary" in str(f) for f in failures), failures)
        self.assertGreater(metrics["item_issues"], 0)

    def test_missing_section_is_rejected(self) -> None:
        content = _content()
        content["sections"] = content["sections"][:1]
        failures, _ = validators.check_structure(content)
        self.assertTrue(any("expected 2 sections" in str(f) for f in failures), failures)

    def test_unresolvable_reference_is_rejected(self) -> None:
        content = _content()
        content["sections"][0]["groups"][0]["items"][0]["sources"] = [99999]
        path = Path(self._tmp) / "content.json"
        path.write_text(json.dumps(content, ensure_ascii=False))
        result = validators.daily_trends_references(path)
        self.assertFalse(result["ok"])
        self.assertTrue(any("no reference" in str(f) for f in result["failures"]))

    def test_fabricated_reference_fails_verifiable(self) -> None:
        content = _content()
        content["references"][0]["url"] = "https://example.invalid/made-up-story"
        content["sections"][0]["groups"][0]["items"][0]["sources"] = [content["references"][0]["id"]]
        path = Path(self._tmp) / "fabricated.json"
        path.write_text(json.dumps(content, ensure_ascii=False))
        result = validators.daily_trends_verifiable(path, REAL_RAW)
        self.assertFalse(result["ok"])
        self.assertLess(result["metrics"]["verifiable_ratio"], 1.0)

    def test_unknown_validator_name_is_reported(self) -> None:
        with self.assertRaises(Exception):
            validators.get("does_not_exist")

    def test_real_sitemap_passes_the_drift_gate(self) -> None:
        """A real hexo-generator-sitemap output, frozen as a fixture.

        This used to read the author's live site checkout by absolute path, so it
        could not pass anywhere else - CI caught it. The property under test is
        "the shipped generator's baseline whitespace is tolerated", and a frozen
        copy tests exactly that while remaining portable.
        """
        result = validators.sitemap_sane(
            Path(__file__).resolve().parent / "fixtures" / "sitemap-hexo-baseline.xml"
        )
        self.assertTrue(result["ok"], result.get("failures"))
        self.assertGreater(result["metrics"]["urls"], 0)
        self.assertLessEqual(result["metrics"]["max_blank_run"], 8)

    def test_duplicate_sitemap_urls_are_rejected(self) -> None:
        path = Path(self._tmp) / "dup.xml"
        path.write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n<urlset>\n'
            "  <url><loc>https://example.com/a</loc></url>\n"
            "  <url><loc>https://example.com/a</loc></url>\n"
            "</urlset>\n",
            encoding="utf-8",
        )
        result = validators.sitemap_sane(path)
        self.assertFalse(result["ok"])
        self.assertTrue(any("duplicate" in str(f) for f in result["failures"]))

    def test_blank_line_drift_is_rejected(self) -> None:
        path = Path(self._tmp) / "drift.xml"
        path.write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n<urlset>\n'
            + "\n" * 12
            +
            "  <url><loc>https://example.com/a</loc></url>\n"
            "</urlset>\n",
            encoding="utf-8",
        )
        result = validators.sitemap_sane(path)
        self.assertFalse(result["ok"])
        self.assertGreater(result["metrics"]["max_blank_run"], 8)

    def test_hexo_baseline_whitespace_is_tolerated(self) -> None:
        """hexo-generator-sitemap emits a 4-blank-line run before </urlset>."""
        path = Path(self._tmp) / "baseline.xml"
        path.write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n<urlset>\n'
            "  <url><loc>https://example.com/a</loc></url>\n"
            "\n  \n\n  \n"
            "</urlset>\n",
            encoding="utf-8",
        )
        result = validators.sitemap_sane(path)
        self.assertTrue(result["ok"], result.get("failures"))
        self.assertLessEqual(result["metrics"]["max_blank_run"], 4)

    def test_broken_xml_is_rejected(self) -> None:
        path = Path(self._tmp) / "broken.xml"
        path.write_text("<urlset><url>", encoding="utf-8")
        result = validators.sitemap_sane(path)
        self.assertFalse(result["ok"])

    def setUp(self) -> None:
        import tempfile

        self._tmpdir = tempfile.TemporaryDirectory()
        self._tmp = self._tmpdir.name

    def tearDown(self) -> None:
        self._tmpdir.cleanup()


class ValidatorArgTests(unittest.TestCase):
    """`run_all` 的模板替换：只有路径参数才当路径解析。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _index(self, body: str) -> Path:
        path = self.root / "index.html"
        path.write_text(body, encoding="utf-8")
        return path

    def _run(self, index: Path, **args) -> dict:
        # The fixture dates are in January, before the real project start, so the
        # plausible-date window is opened for the test instead of hard-coding
        # 2026-09-01 into the assertions.
        args.setdefault("earliest_date", "2026-01-01")
        spec = taskspec.ValidatorSpec(
            name="daily_trends_index",
            args={"path": str(index), "expected_date": "{date}", **args},
        )
        results = validators.run_all(
            [spec], task_dir=self.root, date="2026-01-02", run_dir=self.root
        )
        return results[0]

    def test_date_argument_is_not_treated_as_a_path(self) -> None:
        """A date passed through `{date}` must arrive as "2026-01-02", not task_dir/2026-01-02."""
        result = self._run(self._index('<a href="/trends/2026-01-02/">期号</a>'))
        self.assertTrue(result["ok"], result.get("failures"))

    def test_index_missing_the_current_issue_fails(self) -> None:
        result = self._run(self._index('<a href="/trends/2026-01-01/">上一期</a>'))
        self.assertFalse(result["ok"])
        self.assertTrue(
            any("does not list" in str(f.get("issue")) for f in result["failures"]),
            result["failures"],
        )

    def test_index_with_a_stray_old_date_fails(self) -> None:
        result = self._run(
            self._index(
                '<a href="/trends/2026-01-02/">本期</a><a href="/trends/2026-01-01/">上一期</a>'
                '<a href="/trends/2020-01-01/">测试日期</a>'
            )
        )
        self.assertFalse(result["ok"])
        self.assertTrue(
            any("out-of-range" in str(f.get("issue")) for f in result["failures"]),
            result["failures"],
        )


if __name__ == "__main__":
    unittest.main()
