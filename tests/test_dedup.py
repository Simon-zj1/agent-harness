"""Cross-source duplicate detection: cheap first, model only in the grey band."""

from __future__ import annotations

import unittest

from harness import dedup
from harness.decisions import Decision


def _candidate(location: str, title: str, *, url: str | None = None, rank: int = 0) -> dedup.Candidate:
    return dedup.Candidate(
        location=location,
        section="insights",
        group="agent-engineering",
        title_zh=title,
        title_en=title,
        # Body text follows the title so two different stories do not look
        # identical through the body-overlap fallback.
        body_zh=f"{title}。这是这条内容的正文说明。" * 2,
        body_en=f"{title} — body text for this item. " * 2,
        sources=[1],
        urls=[url] if url else [],
        rank=rank,
    )


class ScoringTests(unittest.TestCase):
    def test_identical_titles_are_duplicates(self) -> None:
        plan = dedup.compare(
            [_candidate("a", "OpenAI 暂停最强模型训练", rank=0),
             _candidate("b", "OpenAI 暂停最强模型训练", rank=1)]
        )
        self.assertEqual(len(plan.duplicates), 1)
        self.assertGreaterEqual(plan.duplicates[0].score, dedup.DUP_SCORE)

    def test_same_section_with_shared_source_is_a_hard_duplicate(self) -> None:
        plan = dedup.compare(
            [
                _candidate("a", "RoboECC：边缘-云协同的机器人计算框架", url="https://github.com/x/roboecc", rank=0),
                _candidate("b", "RoboECC：机器人边缘-云协同计算代码", url="https://github.com/x/roboecc", rank=1),
            ]
        )
        self.assertEqual(len(plan.duplicates), 1, plan.pairs)
        self.assertIn("shares cited URL", " ".join(plan.duplicates[0].reasons))

    def test_cross_section_repetition_is_a_judgement_call(self) -> None:
        """A roundup in insights and the repo's own card are not auto-dropped."""
        cross = [
            _candidate("insights/robotics[7]", "世界模型代码开源潮：客体永久性官方实现与 WorldinWorld",
                       url="https://github.com/x/object-permanence", rank=0),
            _candidate("github[2]", "hokindeng/object-permanence — 世界模型客体永久性官方代码",
                       url="https://github.com/x/object-permanence", rank=1),
        ]
        cross[1].section = "github"
        plan = dedup.compare(cross)
        self.assertEqual(len(plan.duplicates), 0, plan.pairs)
        self.assertEqual(len(plan.borderline), 1)

    def test_cross_section_near_identical_titles_are_a_hard_duplicate(self) -> None:
        cross = [
            _candidate("a", "OmniJev：多模态有限选择决策与机器人控制", url="https://github.com/x/omnijev", rank=0),
            _candidate("b", "OmniJev：多模态有限选择决策与机器人控制", url="https://github.com/x/omnijev", rank=1),
        ]
        cross[1].section = "github"
        plan = dedup.compare(cross)
        self.assertEqual(len(plan.duplicates), 1, plan.pairs)

    def test_related_but_distinct_stays_in_the_grey_band(self) -> None:
        """A direction and its awesome-list cite the same page: a judgement call."""
        plan = dedup.compare(
            [
                _candidate(
                    "a",
                    "具身递归自我改进（Embodied RSI）成为新方向：机器人开始自我改进",
                    url="https://github.com/x/awesome-embodied-rsi",
                    rank=0,
                ),
                _candidate(
                    "b",
                    "awesome-embodied-rsi：具身递归自我改进的资料清单",
                    url="https://github.com/x/awesome-embodied-rsi",
                    rank=1,
                ),
            ]
        )
        self.assertEqual(len(plan.duplicates), 0)
        self.assertEqual(len(plan.borderline), 1)

    def test_weak_title_similarity_alone_is_not_reported(self) -> None:
        """~40% token overlap on short Chinese titles is usually coincidence."""
        plan = dedup.compare(
            [
                _candidate("a", "机器人开始自我改进的新方向", rank=0),
                _candidate("b", "机器人自我改进资料清单", rank=1),
            ]
        )
        self.assertEqual(plan.pairs, [])

    def test_unrelated_items_are_ignored(self) -> None:
        plan = dedup.compare(
            [
                _candidate("a", "OpenAI 上线视觉广告", rank=0),
                _candidate("b", "Cloudflare 发布 Web Search API", rank=1),
            ]
        )
        self.assertEqual(plan.pairs, [])

    def test_title_normalisation_ignores_punctuation_and_lead_ins(self) -> None:
        self.assertEqual(
            dedup.normalize_title("消息称：OpenAI 发布新模型！"),
            dedup.normalize_title("OpenAI 发布新模型"),
        )


class ResolveTests(unittest.TestCase):
    def _plan(self):
        return dedup.compare(
            [
                _candidate("first", "同一个标题", rank=0),
                _candidate("second", "同一个标题", rank=1),
            ]
        )

    def test_clear_duplicate_drops_the_later_item(self) -> None:
        outcome = dedup.resolve(self._plan())
        self.assertEqual(outcome.dropped_locations, {"second"})
        self.assertEqual(outcome.dropped[0]["kept"], "first")
        self.assertEqual(outcome.dropped[0]["source"], "deterministic")

    def test_grey_band_without_a_judge_is_recorded_as_unjudged(self) -> None:
        plan = dedup.compare(
            [
                _candidate("a", "具身递归自我改进（Embodied RSI）成为新方向", rank=0),
                _candidate("b", "awesome-embodied-rsi：具身递归自我改进的资料清单", rank=1),
            ]
        )
        outcome = dedup.resolve(plan)
        self.assertEqual(outcome.dropped, [])
        self.assertEqual(len(outcome.unjudged), 1)

    def test_judge_can_settle_a_grey_pair(self) -> None:
        plan = dedup.compare(
            [
                _candidate("a", "具身递归自我改进（Embodied RSI）成为新方向", rank=0),
                _candidate("b", "awesome-embodied-rsi：具身递归自我改进的资料清单", rank=1),
            ]
        )
        outcome = dedup.resolve(plan, judge=lambda left, right: Decision.PASS)
        self.assertEqual(outcome.dropped_locations, {"b"})
        self.assertEqual(outcome.dropped[0]["source"], "model")
        self.assertEqual(outcome.judge_calls, 1)

    def test_judge_saying_distinct_keeps_both(self) -> None:
        plan = dedup.compare(
            [
                _candidate("a", "具身递归自我改进（Embodied RSI）成为新方向", rank=0),
                _candidate("b", "awesome-embodied-rsi：具身递归自我改进的资料清单", rank=1),
            ]
        )
        outcome = dedup.resolve(plan, judge=lambda left, right: Decision.FAIL)
        self.assertEqual(outcome.dropped, [])
        self.assertEqual(len(outcome.judged_distinct), 1)

    def test_judge_calls_are_capped(self) -> None:
        plan = dedup.compare(
            [_candidate(f"item{i}", "同一个标题", rank=i) for i in range(6)]
        )
        calls = []

        def judge(left, right):
            calls.append((left.location, right.location))
            return Decision.FAIL

        outcome = dedup.resolve(plan, judge=judge, max_judge_calls=2)
        self.assertLessEqual(outcome.judge_calls, 2)
        self.assertEqual(len(calls), outcome.judge_calls)


if __name__ == "__main__":
    unittest.main()
