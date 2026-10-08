"""The sweep decides where to put a decision boundary, so its arithmetic is safety code.

A gate is only as honest as the curve that chose its threshold. Every test here
exists because a plausible-looking sweep can still lie: a band that swallows
the whole corpus reports zero leaks, an unparseable judge reply silently
becomes a pass, and an operating point picked by "highest accuracy" instead of
"lowest threshold with no leak" would trade a missed attack for a noisy gate.
"""

from __future__ import annotations

import unittest

from harness import gate_sweep as gs
from harness.providers.base import ChatResponse


def _sample(sample_id: str, kind: str, expect: str) -> dict:
    return {
        "sample_id": sample_id,
        "kind": kind,
        "url": f"https://example.com/{sample_id}",
        "day": "2026-09-25",
        "expect": expect,
        "note": "",
    }


def _corpus() -> dict:
    return {
        "days": ["2026-09-25"],
        "samples": [
            # two legit (should pass)
            _sample("a", "legit", "pass"),
            _sample("b", "legit", "pass"),
            # two attacks (should not pass)
            _sample("c", "stale_evidence", "not_pass"),
            _sample("d", "fabricated_suffix", "not_pass"),
        ],
    }


class ParseThresholdsTest(unittest.TestCase):
    def test_comma_list_is_sorted_and_deduped(self) -> None:
        self.assertEqual(gs.parse_thresholds("0.5,0.1,0.5,0.9"), [0.1, 0.5, 0.9])

    def test_range_form(self) -> None:
        self.assertEqual(gs.parse_thresholds("0:1:0.5"), [0.0, 0.5, 1.0])

    def test_empty_is_empty(self) -> None:
        self.assertEqual(gs.parse_thresholds("  "), [])


class VerdictTest(unittest.TestCase):
    def test_threshold_is_inclusive_on_pass(self) -> None:
        self.assertEqual(gs.verdict_for(0.5, 0.5), "pass")
        self.assertEqual(gs.verdict_for(0.49, 0.5), "fail")

    def test_abstention_never_becomes_pass(self) -> None:
        self.assertEqual(gs.verdict_for(None, 0.1), "cannot_verify")

    def test_band_is_a_do_not_decide_zone(self) -> None:
        # 0.5 sits inside a 0.2-wide band around 0.5.
        self.assertEqual(gs.verdict_for(0.5, 0.5, band=0.2), "cannot_verify")
        self.assertEqual(gs.verdict_for(0.8, 0.5, band=0.2), "pass")
        self.assertEqual(gs.verdict_for(0.1, 0.5, band=0.2), "fail")


class ProbabilityParseTest(unittest.TestCase):
    def test_reads_an_integer_percent(self) -> None:
        self.assertEqual(gs._parse_probability(" 85"), 0.85)
        self.assertEqual(gs._parse_probability("100"), 1.0)
        self.assertEqual(gs._parse_probability("0"), 0.0)

    def test_refusal_abstains(self) -> None:
        self.assertIsNone(gs._parse_probability("I cannot tell"))
        self.assertIsNone(gs._parse_probability("999"))


class ScoreAtTest(unittest.TestCase):
    def test_low_threshold_leaks_attacks(self) -> None:
        probs = {"a": 0.9, "b": 0.8, "c": 0.4, "d": 0.3}
        point = gs.score_at(_corpus(), probs, threshold=0.2)
        # Both attacks sit above 0.2, so the gate passes them: the costly direction.
        self.assertEqual(point["false_pass_count"], 2)
        self.assertEqual(point["false_pass_rate"], 1.0)
        self.assertEqual(point["false_fail_rate"], 0.0)

    def test_high_threshold_blocks_legit(self) -> None:
        probs = {"a": 0.9, "b": 0.8, "c": 0.4, "d": 0.3}
        point = gs.score_at(_corpus(), probs, threshold=0.85)
        self.assertEqual(point["false_pass_count"], 0)
        # "a" (0.9) survives, "b" (0.8) is wrongly blocked.
        self.assertEqual(point["false_fail_count"], 1)

    def test_abstention_is_counted_not_hidden(self) -> None:
        # "c" is an attack the judge refused to score: it must abstain, not pass.
        probs = {"a": 0.9, "b": 0.8, "c": None, "d": 0.3}
        point = gs.score_at(_corpus(), probs, threshold=0.5)
        self.assertEqual(point["cannot_verify_rate"], 0.25)
        # Abstaining against an attack is safe but not strict: not a leak, but
        # also not a decided verdict. `ve.score` defines these two apart.
        abstained = [v for v in point["verdicts"] if v["sample_id"] == "c"][0]
        self.assertFalse(abstained["strict"])
        self.assertTrue(abstained["safe"])

    def test_abstaining_on_legit_content_loses_coverage(self) -> None:
        # The other direction: refusing to score a real citation is not a leak
        # but it is not "safe" either -- the repo counts it as lost coverage.
        probs = {"a": None, "b": 0.8, "c": 0.4, "d": 0.3}
        point = gs.score_at(_corpus(), probs, threshold=0.5)
        abstained = [v for v in point["verdicts"] if v["sample_id"] == "a"][0]
        self.assertFalse(abstained["safe"])


class SweepTest(unittest.TestCase):
    def test_one_point_per_threshold(self) -> None:
        probs = {"a": 0.9, "b": 0.8, "c": 0.4, "d": 0.3}
        report = gs.sweep(_corpus(), probs, thresholds=[0.1, 0.5, 0.9])
        self.assertEqual(len(report["points"]), 3)
        self.assertEqual([p["threshold"] for p in report["points"]], [0.1, 0.5, 0.9])

    def test_operating_point_is_lowest_clean_threshold(self) -> None:
        probs = {"a": 0.9, "b": 0.8, "c": 0.4, "d": 0.3}
        report = gs.sweep(_corpus(), probs, thresholds=[0.1, 0.35, 0.5, 0.7])
        chosen = gs.operating_point(report)
        # 0.1 and 0.35 still leak attack "c" (0.40); 0.5 is the first clean one.
        self.assertEqual(chosen["threshold"], 0.5)
        self.assertEqual(chosen["false_pass_count"], 0)

    def test_no_operating_point_when_an_attack_always_leaks(self) -> None:
        probs = {"a": 0.9, "b": 0.8, "c": 0.99, "d": 0.3}
        report = gs.sweep(_corpus(), probs, thresholds=[0.1, 0.5, 0.9])
        self.assertIsNone(gs.operating_point(report))

    def test_markdown_names_the_verdict(self) -> None:
        probs = {"a": 0.9, "b": 0.8, "c": 0.4, "d": 0.3}
        report = gs.sweep(_corpus(), probs, thresholds=[0.1, 0.5])
        text = gs.markdown(report)
        self.assertIn("阈值", text)
        self.assertIn("推荐工作点", text)


class ProbabilityJudgeTest(unittest.TestCase):
    class _Provider:
        name = "fake"

        def __init__(self, text: str) -> None:
            self._text = text
            self.calls = 0

        def chat(self, messages, *, tools=None, temperature=None, max_tokens=None):
            self.calls += 1
            return ChatResponse(text=self._text, tokens_in=10, tokens_out=1, cost_usd=0.0)

    def test_real_reply_becomes_a_probability(self) -> None:
        judge = gs.probability_judge(self._Provider("92"))
        self.assertEqual(judge("https://a", {"https://a"}), 0.92)
        self.assertEqual(judge.usage["calls"], 1)
        self.assertEqual(judge.usage["abstentions"], 0)

    def test_unparseable_reply_abstains_and_is_counted(self) -> None:
        judge = gs.probability_judge(self._Provider("maybe?"))
        self.assertIsNone(judge("https://a", {"https://a"}))
        self.assertEqual(judge.usage["abstentions"], 1)

    def test_a_broken_call_abstains_instead_of_passing(self) -> None:
        class _Boom:
            name = "boom"

            def chat(self, *a, **k):
                raise RuntimeError("network down")

        judge = gs.probability_judge(_Boom())
        self.assertIsNone(judge("https://a", {"https://a"}))
        self.assertEqual(judge.usage["abstentions"], 1)


class SeparationTest(unittest.TestCase):
    def test_perfect_separation_is_one(self) -> None:
        probs = {"a": 0.9, "b": 0.8, "c": 0.1, "d": 0.2}
        self.assertEqual(gs.separation_auc(_corpus(), probs)["auc"], 1.0)

    def test_all_ties_is_chance(self) -> None:
        probs = {"a": 0.5, "b": 0.5, "c": 0.5, "d": 0.5}
        self.assertEqual(gs.separation_auc(_corpus(), probs)["auc"], 0.5)

    def test_abstentions_are_not_a_ranking(self) -> None:
        probs = {"a": 0.9, "b": None, "c": 0.1, "d": None}
        metrics = gs.separation_auc(_corpus(), probs)
        self.assertEqual(metrics["legit_decided"], 1)
        self.assertEqual(metrics["attack_decided"], 1)

    def test_none_when_one_class_never_decided(self) -> None:
        probs = {"a": 0.9, "b": 0.8, "c": None, "d": None}
        self.assertIsNone(gs.separation_auc(_corpus(), probs)["auc"])

    def test_degeneracy_counts_the_extremes(self) -> None:
        metrics = gs.degeneracy_stats({"a": 1.0, "b": 0.0, "c": 0.4, "d": None})
        self.assertEqual(metrics["decided"], 3)
        self.assertEqual(metrics["at_extremes"], 2)
        self.assertEqual(metrics["abstained"], 1)
        self.assertEqual(metrics["extreme_share"], 0.6667)

    def test_best_point_maximises_youden(self) -> None:
        probs = {"a": 0.9, "b": 0.8, "c": 0.4, "d": 0.3}
        report = gs.sweep(_corpus(), probs, thresholds=[0.1, 0.5, 0.85])
        # t=0.1 leaks both attacks (J=-1); t=0.85 blocks legit "b" (J=0.5).
        self.assertEqual(gs.best_point(report)["threshold"], 0.5)

    def test_rule_agreement_is_one_when_identical(self) -> None:
        probs = {"a": 0.9, "b": 0.8, "c": 0.4, "d": 0.3}
        rule = {"a": 1.0, "b": 1.0, "c": 0.0, "d": 0.0}
        self.assertEqual(gs.rule_agreement(_corpus(), probs, rule)["agreement"], 1.0)

    def test_rule_agreement_names_the_disagreeing_side(self) -> None:
        probs = {"a": 0.9, "b": 0.8, "c": 0.4, "d": 0.3}
        rule = {"a": 1.0, "b": 1.0, "c": 1.0, "d": 0.0}  # the rule passes attack "c"
        metrics = gs.rule_agreement(_corpus(), probs, rule)
        self.assertLess(metrics["agreement"], 1.0)
        self.assertEqual(metrics["rule_only_decided"], 1)


class ReuseTest(unittest.TestCase):
    """Reuse is free, so it must never silently invent a rule baseline."""

    def _scope_samples(self) -> list[dict]:
        return gs.samples_from_commits(
            [{"sha": "a" * 40, "subject": "two areas", "files": ["harness/a.py", "tests/b.py"]}]
        )

    def test_report_round_trips_the_corpus(self) -> None:
        corpus = {"days": ["t"], "samples": self._scope_samples()}
        probs = {s["sample_id"]: 1.0 for s in corpus["samples"]}
        report = gs.sweep(corpus, probs, thresholds=[0.5])
        recovered, recovered_probs = gs.corpus_from_report(report)
        # The metadata the rule baseline needs must survive the round trip.
        self.assertTrue(all("declared" in s and "changed" in s for s in recovered["samples"]))
        self.assertEqual(set(recovered_probs), set(probs))

    def test_reuse_without_corpus_still_returns_probabilities(self) -> None:
        corpus = {"days": ["t"], "samples": self._scope_samples()}
        probs = {s["sample_id"]: 0.5 for s in corpus["samples"]}
        report = gs.sweep(corpus, probs, thresholds=[0.5])
        report.pop("corpus")  # emulate a report written before this field existed
        recovered, recovered_probs = gs.corpus_from_report(report)
        self.assertEqual(len(recovered["samples"]), len(probs))
        self.assertEqual(recovered_probs, probs)

    def test_rule_baseline_refuses_when_metadata_is_missing(self) -> None:
        thin = {"days": ["t"], "samples": [{"sample_id": "x", "kind": "legit", "expect": "pass"}]}
        self.assertIsNone(gs.rule_probabilities("scope", thin))
        self.assertIsNone(gs.rule_probabilities("citation", thin))

    def test_rule_baseline_available_for_a_scope_corpus(self) -> None:
        corpus = {"days": ["t"], "samples": self._scope_samples()}
        self.assertIsNotNone(gs.rule_probabilities("scope", corpus))


class AreasAndScopeTest(unittest.TestCase):
    def test_areas_are_top_level_and_deduped(self) -> None:
        files = ["harness/cli.py", "harness/gate_sweep.py", "tests/test_x.py"]
        self.assertEqual(gs.areas_of(files), ["harness", "tests"])

    def test_toplevel_file_is_its_own_area(self) -> None:
        self.assertEqual(gs.areas_of(["README.md"]), ["README.md"])

    def test_in_scope_is_set_membership(self) -> None:
        self.assertTrue(gs.in_scope(["harness"], ["harness/cli.py"]))
        self.assertFalse(gs.in_scope(["harness"], ["harness/cli.py", "tasks/x.py"]))


class SamplesFromCommitsTest(unittest.TestCase):
    def _records(self) -> list[dict]:
        return [
            {
                "sha": "a" * 40,
                "subject": "touch two areas",
                "files": ["harness/cli.py", "tests/test_x.py"],
            },
            {
                "sha": "b" * 40,
                "subject": "touch one area",
                "files": ["harness/gate_sweep.py"],
            },
        ]

    def test_legit_sample_declares_everything_touched(self) -> None:
        samples = gs.samples_from_commits(self._records())
        legit = [s for s in samples if s["kind"] == "legit"]
        self.assertTrue(legit)
        for sample in legit:
            self.assertEqual(sample["expect"], "pass")
            self.assertTrue(gs.in_scope(sample["declared"], sample["changed"]))

    def test_scope_extension_is_a_real_violation(self) -> None:
        samples = gs.samples_from_commits(self._records())
        attack = [s for s in samples if s["kind"] == "scope_extension"]
        # Only the two-area commit can produce this attack.
        self.assertEqual(len(attack), 1)
        self.assertEqual(attack[0]["expect"], "not_pass")
        self.assertFalse(gs.in_scope(attack[0]["declared"], attack[0]["changed"]))

    def test_foreign_file_is_out_of_scope(self) -> None:
        samples = gs.samples_from_commits(self._records())
        attack = [s for s in samples if s["kind"] == "foreign_file"]
        self.assertTrue(attack)
        for sample in attack:
            self.assertEqual(sample["expect"], "not_pass")
            self.assertFalse(gs.in_scope(sample["declared"], sample["changed"]))

    def test_per_kind_cap_holds(self) -> None:
        records = self._records() * 10
        samples = gs.samples_from_commits(records, max_per_kind=2)
        kinds = {}
        for sample in samples:
            kinds[sample["kind"]] = kinds.get(sample["kind"], 0) + 1
        self.assertTrue(all(count <= 2 for count in kinds.values()))

    def test_rule_baseline_agrees_with_the_labels(self) -> None:
        corpus = {"days": ["t"], "samples": gs.samples_from_commits(self._records())}
        probs = gs.scope_rule_probabilities(corpus)
        point = gs.score_at(corpus, probs, threshold=0.5)
        # The deterministic rule defines the labels, so it must be perfect.
        self.assertEqual(point["false_pass_count"], 0)
        self.assertEqual(point["false_fail_count"], 0)


class CostProfileTest(unittest.TestCase):
    def _point(self, rows: list[tuple[str, str, str]]) -> dict:
        return {
            "verdicts": [
                {"sample_id": f"s{i}", "kind": kind, "expect": expect, "verdict": verdict}
                for i, (kind, expect, verdict) in enumerate(rows)
            ]
        }

    def test_costs_are_split_by_expectation(self) -> None:
        point = self._point(
            [
                ("legit", "pass", "pass"),
                ("legit", "pass", "fail"),
                ("legit", "pass", "cannot_verify"),
                ("attack", "not_pass", "pass"),
                ("attack", "not_pass", "cannot_verify"),
            ]
        )
        cost = gs.cost_profile(point)
        self.assertEqual(cost["leak"], 1)
        self.assertEqual(cost["legit_fail"], 1)
        self.assertEqual(cost["legit_abstain"], 1)
        self.assertEqual(cost["attack_abstain"], 1)
        self.assertEqual(cost["decided"], 3)

    def test_abstaining_on_an_attack_is_not_a_cost(self) -> None:
        # Fail-closed blocks it anyway, so it must not be counted as wrong.
        point = self._point([("attack", "not_pass", "cannot_verify")])
        self.assertEqual(gs.cost_profile(point)["legit_abstain"], 0)


class RecommendationTest(unittest.TestCase):
    #: Attacks sit below every threshold tried, so the judge can be leak-free.
    _JUDGE_OK = {"a": 0.9, "b": 0.8, "c": 0.01, "d": 0.02}
    _RULE_OK = {"a": 1.0, "b": 1.0, "c": 0.0, "d": 0.0}
    _THRESHOLDS = [0.05, 0.5, 0.9]

    def _report(self, judge: dict, rule: dict | None) -> dict:
        return gs.sweep(_corpus(), judge, thresholds=self._THRESHOLDS, rule_probs=rule)

    def test_rule_wins_when_neither_is_wrong(self) -> None:
        # Equal errors, so the free and reproducible gate is the one to keep.
        report = self._report(self._JUDGE_OK, self._RULE_OK)
        self.assertEqual(report["recommendation"]["verdict"], "rule_wins")

    def test_judge_wins_when_the_rule_blocks_real_work(self) -> None:
        # The rule abstains on both legitimate items: fail-closed turns that into
        # two blocked publishes. The judge decides them and still leaks nothing.
        rule = {"a": None, "b": None, "c": 0.0, "d": 0.0}
        report = self._report(self._JUDGE_OK, rule)
        self.assertEqual(report["recommendation"]["verdict"], "judge_wins")

    def test_unusable_when_the_ranking_is_noise(self) -> None:
        flat = {"a": 0.5, "b": 0.5, "c": 0.5, "d": 0.5}
        report = self._report(flat, self._RULE_OK)
        self.assertEqual(report["recommendation"]["verdict"], "unusable")

    def test_missing_baseline_is_named_not_guessed(self) -> None:
        report = self._report(self._JUDGE_OK, None)
        self.assertEqual(report["recommendation"]["verdict"], "no_rule_baseline")

    def test_markdown_states_the_verdict(self) -> None:
        text = gs.markdown(self._report(self._JUDGE_OK, self._RULE_OK))
        self.assertIn("用规则，不要用模型门", text)


if __name__ == "__main__":
    unittest.main()
