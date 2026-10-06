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
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from harness import validators

REPO = Path(__file__).resolve().parent.parent
#: 工具仓库的位置。变异测试会把本仓库复制到 /tmp 下跑，那时 REPO.parent 里没有
#: daily-trends —— 于是 CoverageTests 会被整体 skip，变异测试看起来「全绿」。
#: 加一个绝对路径兜底，保证复制出去的副本仍然真的执行这些用例。
_TOOLS_CANDIDATES = (
    REPO.parent / "daily-trends",
    Path("/Users/simon-zj/Documents/ChatGPT/daily-trends"),
)
TOOLS_SOURCE = next(
    (path for path in _TOOLS_CANDIDATES if (path / "tools" / "brief.py").is_file()),
    _TOOLS_CANDIDATES[0],
)


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


class SeverityDeclarationTests(unittest.TestCase):
    """闸门严重度只能有一个事实来源：task.toml 的 required。

    AGENTS.md 规则 11 说「装饰性问题记为告警」。把这条变成可执行的断言，否则
    把 coverage 的 required 从 false 改成 true 不会有任何测试变红（评审的 M3 变异）。
    """

    def test_reading_and_advisory_gates_have_the_declared_severity(self) -> None:
        from harness.cli import _required_validators

        required = _required_validators() or set()
        for name in ("daily_trends_structure", "daily_trends_references",
                     "daily_trends_verifiable", "daily_trends_brief"):
            self.assertIn(name, required, f"{name} 应当拦发布")
        for name in ("daily_trends_depth", "daily_trends_coverage"):
            self.assertNotIn(name, required, f"{name} 是告警级，不该拦发布")


class ReleaseGateTests(unittest.TestCase):
    """The gate the deploy script calls: newest day must pass, and be bypassable."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "raw").mkdir(parents=True)
        self.env = dict(os.environ)
        self.env["DAILY_TRENDS_DATA_DIR"] = str(self.root)
        self.env["PYTHONPATH"] = str(REPO)
        # 任务里的 tools_dir 默认相对仓库位置解析；仓库被复制到 /tmp 做变异测试时
        # 那个相对路径指向不存在的目录，brief/depth/coverage 会全部 CANNOT_VERIFY，
        # 于是 release 用例在副本里必然红（与被测变异无关）。显式给定工具仓库路径，
        # 让这些用例只对被测行为敏感。
        self.env["DAILY_TRENDS_DIR"] = str(TOOLS_SOURCE)

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

    def test_stale_from_another_day_warns_instead_of_blocking(self) -> None:
        """Reused material from yesterday degrades the page; it does not break it.

        A citation that exists in *no* capture is a fabrication risk and blocks;
        one that exists in yesterday's capture is a producer bug and only warns.
        """
        (self.root / "raw" / "2026-01-01.json").write_text(
            json.dumps({"hn": [{"title": "t", "url": "https://example.com/yesterday"}]}),
            encoding="utf-8",
        )
        (self.root / "raw" / "2026-01-02.json").write_text(
            json.dumps({"hn": [{"title": "t", "url": "https://example.com/today"}]}),
            encoding="utf-8",
        )
        content = _content(
            references=[{"id": 1, "title": "yesterday", "url": "https://example.com/yesterday"}],
            sources=[1],
        )
        content["date"] = "2026-01-02"
        (self.root / "2026-01-02.json").write_text(
            json.dumps(content, ensure_ascii=False), encoding="utf-8"
        )
        result = self._gate("--days", "1")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("来自其它日期的抓取", result.stdout)


@unittest.skipUnless(
    (TOOLS_SOURCE / "tools" / "brief.py").is_file(),
    "requires the sibling daily-trends checkout",
)
class CoverageTests(unittest.TestCase):
    """覆盖度：报告「今天最热的那批里，哪些没写进去」。

    这一组用例是被红队评审逼出来的：第一版三个用例都不传 tools_dir（主题排序根本不执行）、
    跨天夹具只有 29 个字符（先被长度过滤丢掉，日期检查没被隔离），变异测试下三个变异全绿。
    现在每个用例都对应一个可被杀死的变异。
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.tools = self.root / "tools-checkout"
        (self.tools / "tools").mkdir(parents=True)
        (self.tools / "config").mkdir(parents=True)
        shutil.copy(TOOLS_SOURCE / "tools" / "brief.py", self.tools / "tools" / "brief.py")
        shutil.copy(
            TOOLS_SOURCE / "config" / "interests.json",
            self.tools / "config" / "interests.json",
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _content(self, *, cited_url: str | None) -> Path:
        refs = [{"id": 1, "title": "cited", "url": cited_url}] if cited_url else []
        content = {
            "date": "2026-01-02",
            "title": {"zh": "标题", "en": "title"},
            "summary": {"zh": "总览", "en": "overview"},
            "notes": {"zh": "口径", "en": "scope"},
            "sections": [
                {"id": "insights", "groups": [{"id": "g", "title": {"zh": "组", "en": "Group"},
                                              "items": [_item("条目", [1])]}]},
                {"id": "github", "items": [_item("repo", [1])]},
            ],
            "references": refs,
        }
        path = self.root / "content.json"
        path.write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")
        return path

    def _coverage(self, raw: dict, *, cited_url: str | None = None, tools_dir: bool = True) -> dict:
        raw_path = self.root / "raw.json"
        raw_path.write_text(json.dumps(raw), encoding="utf-8")
        content_path = self._content(cited_url=cited_url)
        return validators.daily_trends_coverage(
            content_path, raw_path, tools_dir=self.tools if tools_dir else None
        )

    def test_uncited_on_topic_item_is_reported(self) -> None:
        result = self._coverage(
            {"hn": [{"title": "Agent harness benchmark 发布", "url": "https://ex.com/hot", "points": 900}]}
        )
        self.assertTrue(result["ok"], "覆盖度只报警，不拦发布")
        self.assertEqual(result["metrics"]["top3_missed"], 1)
        self.assertIn("hot", result["warnings"][0]["url"])

    def test_low_signal_still_ranks_by_its_own_signal(self) -> None:
        """arXiv/Techmeme 的信号是字符串，排序必须按时间而不是抓取顺序（评审 F2）。"""
        result = self._coverage(
            {
                "arxiv": [
                    {"title": "old Agent paper", "url": "https://arxiv.org/abs/old", "published": "2026-01-01T00:00:00Z"},
                    {"title": "new Agent paper", "url": "https://arxiv.org/abs/new", "published": "2026-01-02T00:00:00Z"},
                ]
            }
        )
        # 最新的一篇必须排在第 1 名，因此它是「来源前 3 名漏」里的第一条。
        self.assertEqual(result["warnings"][0]["url"], "https://arxiv.org/abs/new")

    def test_cited_on_topic_item_is_not_reported(self) -> None:
        result = self._coverage(
            {"hn": [{"title": "Agent harness benchmark 发布", "url": "https://ex.com/hot", "points": 900}]},
            cited_url="https://ex.com/hot",
        )
        self.assertEqual(result["metrics"]["top3_missed"], 0)
        self.assertEqual(result["warnings"], [])

    def test_url_drift_is_not_a_false_miss(self) -> None:
        """raw 带 query、文章引用无 query 的同路径不算漏（评审 F4）。"""
        result = self._coverage(
            {"hn": [{"title": "Agent 检索 API 发布", "url": "https://ex.com/a?view_token=xyz", "points": 500}]},
            cited_url="https://ex.com/a",
        )
        self.assertEqual(result["warnings"], [], result["warnings"])

    def test_offtopic_miss_is_kept_but_ranked_last(self) -> None:
        """无关条目不再被静默过滤，只是排在后面（评审 F1 的修法）。"""
        result = self._coverage(
            {
                "hn": [
                    {"title": "Bob Cringely has died", "url": "https://ex.com/obit", "points": 900},
                    {"title": "Agent harness benchmark 发布", "url": "https://ex.com/agent", "points": 100},
                ]
            }
        )
        self.assertEqual(result["metrics"]["missed"], 2, "两条都要出现，不许静默丢弃")
        self.assertEqual(result["metrics"]["missed_offtopic"], 1)
        self.assertIn("agent", result["warnings"][0]["url"], "命中方向的排前面")

    def test_evergreen_tweets_from_other_days_are_not_counted(self) -> None:
        """31k 赞的跨天常青帖曾被连着四天报成「今天的漏报」（评审 F3）。"""
        long_text = "这是一条足够长的帖子正文，用来确保长度过滤不会成为真正的过滤条件。" * 2
        result = self._coverage(
            {
                "tweets_evergreen": [
                    {"url": "https://x.com/a/1", "date": "2026-01-01", "likes": 31000, "text": long_text}
                ]
            }
        )
        self.assertEqual(result["metrics"]["pool"], 0)

    def test_recent_tweets_from_other_days_are_not_counted_either(self) -> None:
        long_text = "同样足够长的一条近期窗口帖子正文，用于验证 recent 桶也要做日期过滤。" * 2
        result = self._coverage(
            {
                "tweets_recent": [
                    {"url": "https://x.com/b/2", "date": "2026-01-01", "likes": 1100, "text": long_text}
                ]
            }
        )
        self.assertEqual(result["metrics"]["pool"], 0)

    def test_missing_raw_is_cannot_verify_and_never_blocks(self) -> None:
        content_path = self._content(cited_url=None)
        result = validators.daily_trends_coverage(
            content_path, self.root / "nope.json", tools_dir=self.tools
        )
        self.assertEqual(result["decision"], "cannot_verify")
        self.assertFalse(result["ok"])
        # required=false 的闸门在 release 路径上不得成为 blocker —— 由 CLI 层的
        # _required_validators() 保证，这里只锁住 decision 的形状。
        self.assertEqual(result["metrics"]["pool"], 0)


if __name__ == "__main__":
    unittest.main()
