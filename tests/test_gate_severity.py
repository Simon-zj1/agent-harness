"""Severity calibration and staleness attribution.

Two properties are pinned here, both learned from real incidents:

* a citation that only exists in *another* day's capture must say so, because
  "reused stale material" and "fabricated" are different problems;
* an orphan reference (declared, never cited) is a warning, not a blocked
  publish — blocking on cosmetics is how a gate teaches people to bypass it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from harness import validators

REPO = Path(__file__).resolve().parent.parent


def _item(title: str, sources: list[int]) -> dict:
    return {
        "title": {"zh": title, "en": title},
        "prose": {"zh": "正文", "en": "body"},
        "sources": sources,
    }


def _content(*, references: list[dict], sources: list[int], orphan: int | None = None) -> dict:
    refs = list(references)
    if orphan is not None:
        refs.append({"id": orphan, "title": "unused", "url": f"https://example.com/unused{orphan}"})
    return {
        "date": "2026-01-02",
        "title": {"zh": "标题", "en": "title"},
        "summary": {"zh": "总览", "en": "overview"},
        "notes": {"zh": "口径", "en": "scope"},
        "sections": [
            {
                "id": "insights",
                "groups": [
                    {"id": "g", "title": {"zh": "组", "en": "Group"}, "items": [_item("条目", sources)]}
                ],
            },
            {"id": "github", "items": [_item("repo", [sources[0]])]},
        ],
        "references": refs,
    }


class StaleAttributionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "raw").mkdir(parents=True)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_raw(self, day: str, urls: list[str]) -> Path:
        path = self.root / "raw" / f"{day}.json"
        path.write_text(
            json.dumps({"hn": [{"title": "t", "url": u} for u in urls]}),
            encoding="utf-8",
        )
        return path

    def test_citation_from_a_previous_day_is_named(self) -> None:
        self._write_raw("2026-01-01", ["https://arxiv.org/abs/2610.03715v1"])
        today_raw = self._write_raw("2026-01-02", ["https://example.com/today"])
        content = _content(
            references=[
                {
                    "id": 1,
                    "title": "yesterday's paper",
                    "url": "https://arxiv.org/abs/2610.03715v1",
                }
            ],
            sources=[1],
        )
        path = self.root / "2026-01-02.json"
        path.write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")

        result = validators.daily_trends_verifiable(path, today_raw)
        self.assertFalse(result["ok"])
        self.assertEqual(result["metrics"]["stale_citations"], 1)
        self.assertEqual(result["metrics"]["stale_from_days"], ["2026-01-01"])
        self.assertTrue(
            any("2026-01-01" in str(f.get("issue", "")) for f in result["failures"]),
            result["failures"],
        )

    def test_citation_found_nowhere_says_so(self) -> None:
        self._write_raw("2026-01-01", ["https://example.com/other"])
        today_raw = self._write_raw("2026-01-02", ["https://example.com/today"])
        content = _content(
            references=[{"id": 1, "title": "ghost", "url": "https://example.com/ghost"}],
            sources=[1],
        )
        path = self.root / "2026-01-02.json"
        path.write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")

        result = validators.daily_trends_verifiable(path, today_raw)
        self.assertEqual(result["metrics"]["stale_citations"], 0)
        self.assertTrue(
            any("no capture at all" in str(f.get("issue", "")) for f in result["failures"]),
            result["failures"],
        )


class ReferenceSeverityTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, content: dict) -> Path:
        path = self.root / "content.json"
        path.write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")
        return path

    def test_orphan_reference_is_a_warning_not_a_blocker(self) -> None:
        content = _content(
            references=[{"id": 1, "title": "cited", "url": "https://example.com/a"}],
            sources=[1],
            orphan=9,
        )
        result = validators.daily_trends_references(self._write(content))
        self.assertTrue(result["ok"], result.get("failures"))
        self.assertEqual(result["metrics"]["orphans"], 1)
        self.assertEqual(result["metrics"]["warnings"], 1)
        self.assertEqual(result["warnings"][0]["ids"], [9])

    def test_source_id_without_reference_still_blocks(self) -> None:
        content = _content(
            references=[{"id": 1, "title": "cited", "url": "https://example.com/a"}],
            sources=[1, 42],
        )
        result = validators.daily_trends_references(self._write(content))
        self.assertFalse(result["ok"])
        self.assertTrue(
            any("no reference" in str(f.get("issue", "")) for f in result["failures"]),
            result["failures"],
        )

    def test_non_http_reference_still_blocks(self) -> None:
        content = _content(
            references=[{"id": 1, "title": "local", "url": "file:///etc/passwd"}],
            sources=[1],
        )
        result = validators.daily_trends_references(self._write(content))
        self.assertFalse(result["ok"])


class ReleaseGateTests(unittest.TestCase):
    """The gate the deploy script calls: newest day must pass, and be bypassable."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "raw").mkdir(parents=True)
        self.env = dict(os.environ)
        self.env["DAILY_TRENDS_DATA_DIR"] = str(self.root)
        self.env["PYTHONPATH"] = str(REPO)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_day(self, day: str, *, cite_ghost: bool) -> None:
        (self.root / "raw" / f"{day}.json").write_text(
            json.dumps({"hn": [{"title": "t", "url": "https://example.com/today"}]}),
            encoding="utf-8",
        )
        url = "https://example.com/ghost" if cite_ghost else "https://example.com/today"
        content = _content(references=[{"id": 1, "title": "src", "url": url}], sources=[1])
        content["date"] = day
        (self.root / f"{day}.json").write_text(
            json.dumps(content, ensure_ascii=False), encoding="utf-8"
        )

    def _gate(self, *args: str, env: dict | None = None) -> subprocess.CompletedProcess:
        merged = dict(self.env)
        merged.update(env or {})
        return subprocess.run(
            [sys.executable, "-m", "harness.cli", "verify", "release", *args],
            cwd=str(REPO),
            env=merged,
            capture_output=True,
            text=True,
            timeout=120,
        )

    def test_clean_newest_day_passes(self) -> None:
        self._write_day("2026-01-02", cite_ghost=False)
        result = self._gate("--days", "1")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("通过全部内容闸门", result.stdout)

    def test_failing_newest_day_blocks(self) -> None:
        self._write_day("2026-01-02", cite_ghost=True)
        result = self._gate("--days", "1")
        self.assertEqual(result.returncode, 1)
        self.assertIn("阻止发布", result.stdout)
        self.assertIn("AGENT_RELEASE_GATE=off", result.stdout)

    def test_gate_can_be_skipped_explicitly(self) -> None:
        self._write_day("2026-01-02", cite_ghost=True)
        result = self._gate("--days", "1", env={"AGENT_RELEASE_GATE": "off"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("skipped", result.stdout)


if __name__ == "__main__":
    unittest.main()
