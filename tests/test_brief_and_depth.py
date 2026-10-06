"""Reader-side gates: the brief must pick, the summaries must be deep enough.

Hermetic: the brief module and its interest config are copied from the sibling
daily-trends checkout into a temp tools dir, so these tests never touch the real
data and never depend on the network. If the sibling checkout is absent the
whole module is skipped — the harness repository alone cannot ship that file.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from harness import validators

REPO = Path(__file__).resolve().parent.parent
TOOLS_SOURCE = REPO.parent / "daily-trends"


def _item(title: str, body: str, *, sources=(1,)) -> dict:
    return {
        "title": {"zh": title, "en": title},
        "prose": {"zh": body, "en": body},
        "sources": list(sources),
    }


def _content(*items: dict) -> dict:
    return {
        "date": "2026-01-01",
        "title": {"zh": "标题", "en": "title"},
        "summary": {"zh": "总览", "en": "overview"},
        "notes": {"zh": "口径", "en": "scope"},
        "sections": [
            {
                "id": "insights",
                "groups": [
                    {"id": "g", "title": {"zh": "组", "en": "Group"}, "items": list(items)}
                ],
            },
            {
                "id": "github",
                "items": [
                    _item(
                        "仓库",
                        "一个 harness 项目，配套评测基准；限制是只覆盖单机场景。",
                    )
                ],
            },
        ],
        "references": [{"id": 1, "title": "src", "url": "https://example.com/a"}],
    }


DEEP = "WING 用交互为中心的谱域引导训练机器人策略，在 LIBERO 上提升了 12 个点；限制是依赖第一人称视频质量，真机成功率仍需独立复现。"
SHALLOW = "某公司发布了新的模型，引起广泛讨论。"
BUSINESS = "某机器人公司估值达 10 亿美元，成为新晋独角兽。"


@unittest.skipUnless(
    (TOOLS_SOURCE / "tools" / "brief.py").is_file(),
    "requires the sibling daily-trends checkout",
)
class BriefGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "tools").mkdir(parents=True)
        (self.root / "config").mkdir(parents=True)
        shutil.copy(TOOLS_SOURCE / "tools" / "brief.py", self.root / "tools" / "brief.py")
        shutil.copy(
            TOOLS_SOURCE / "config" / "interests.json",
            self.root / "config" / "interests.json",
        )
        self.tools = self.root

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, content: dict) -> Path:
        path = self.root / "content.json"
        path.write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")
        return path

    def test_brief_picks_five_and_never_picks_business_noise(self) -> None:
        items = [_item(f"Agent 评测 {i}", DEEP) for i in range(6)]
        items.append(_item("某机器人公司估值达 10 亿美元", BUSINESS))
        path = self._write(_content(*items))

        result = validators.daily_trends_brief(path, tools_dir=self.tools, limit=5)
        self.assertTrue(result["ok"], result.get("failures"))
        self.assertEqual(result["metrics"]["entries"], 5)
        self.assertGreaterEqual(result["metrics"]["excluded"], 1)
        self.assertIn("filtered by interest profile", result["detail"])

    def test_brief_blocks_when_everything_is_filtered(self) -> None:
        content = {
            "date": "2026-01-01",
            "title": {"zh": "标题", "en": "title"},
            "summary": {"zh": "总览", "en": "overview"},
            "notes": {"zh": "口径", "en": "scope"},
            "sections": [
                {"id": "insights", "groups": [{"id": "g", "title": {"zh": "组", "en": "Group"},
                                              "items": [_item("某机器人公司估值达 10 亿美元", BUSINESS)]}]},
                {"id": "github", "items": [_item("又一家公司完成 2 亿美元融资", BUSINESS)]},
            ],
            "references": [{"id": 1, "title": "src", "url": "https://example.com/a"}],
        }
        path = self._write(content)
        result = validators.daily_trends_brief(path, tools_dir=self.tools, limit=5)
        self.assertFalse(result["ok"])
        self.assertEqual(result["metrics"]["entries"], 0)
        self.assertTrue(
            any("selected nothing" in str(f.get("issue")) for f in result["failures"]),
            result["failures"],
        )

    def test_missing_brief_module_is_cannot_verify(self) -> None:
        empty = self.root / "elsewhere"
        empty.mkdir()
        path = self._write(_content(_item("Agent 评测", DEEP)))
        result = validators.daily_trends_brief(path, tools_dir=empty, limit=5)
        self.assertFalse(result["ok"])
        self.assertEqual(result["decision"], "cannot_verify")

    def test_each_entry_says_why_and_has_a_body(self) -> None:
        path = self._write(_content(*[_item(f"Agent harness 评测 {i}", DEEP) for i in range(4)]))
        result = validators.daily_trends_brief(path, tools_dir=self.tools, limit=5)
        self.assertTrue(result["ok"], result.get("failures"))


class DepthGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "tools").mkdir(parents=True)
        (self.root / "config").mkdir(parents=True)
        shutil.copy(TOOLS_SOURCE / "tools" / "brief.py", self.root / "tools" / "brief.py")
        shutil.copy(
            TOOLS_SOURCE / "config" / "interests.json",
            self.root / "config" / "interests.json",
        )
        self.tools = self.root

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, content: dict) -> Path:
        path = self.root / "content.json"
        path.write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")
        return path

    def test_deep_summaries_raise_the_limit_ratio(self) -> None:
        path = self._write(_content(_item("Agent 评测", DEEP), _item("机器人策略", DEEP)))
        result = validators.daily_trends_depth(path, tools_dir=self.tools)
        self.assertTrue(result["ok"])
        self.assertGreaterEqual(result["metrics"]["limit_ratio"], 0.9)

    def test_shallow_summaries_are_reported_as_warnings(self) -> None:
        path = self._write(_content(_item("某公司发布模型", SHALLOW), _item("另一个发布", SHALLOW)))
        result = validators.daily_trends_depth(path, tools_dir=self.tools)
        # 深度不够不该拦发布，但要出现在 warnings 与 metrics 里
        self.assertTrue(result["ok"])
        self.assertLess(result["metrics"]["limit_ratio"], 0.5)
        self.assertTrue(result["warnings"])
        self.assertIn("limitation", result["detail"])


if __name__ == "__main__":
    unittest.main()
