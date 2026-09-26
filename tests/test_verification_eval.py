"""The gate's own measurement has to be trustworthy before its numbers are.

Every assertion here exists because the corpus generator shipped a bug that
produced a fake false-fail: a drift that was not actually an equivalence, a
double `www.`, a query string appended with the wrong separator. A corpus is
code, and a wrong corpus silently slanders the thing it measures.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from harness import verification_eval as ve
from harness.decisions import canonical_url
from harness.providers.base import ChatResponse

DATA = Path(ve.DEFAULT_DATA_DIR)


def _has_data() -> bool:
    return bool(ve.available_days())


class _FakeProvider:
    name = "fake"

    def __init__(self, text: str) -> None:
        self._text = text
        self.calls: list[list[dict]] = []

    def chat(self, messages, *, tools=None, temperature=None, max_tokens=None):
        self.calls.append(messages)
        return ChatResponse(text=self._text, model="fake")

    def available(self):
        return True, "fake"


@unittest.skipUnless(_has_data(), "no daily-trends captures to build a corpus from")
class CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.days = ve.available_days()[:3]
        cls.corpus = ve.build_corpus(cls.days, max_per_kind_per_day=8)

    def test_corpus_is_deterministic(self) -> None:
        again = ve.build_corpus(self.days, max_per_kind_per_day=8)
        self.assertEqual(
            [s["url"] for s in self.corpus["samples"]],
            [s["url"] for s in again["samples"]],
        )

    def test_every_kind_is_represented(self) -> None:
        kinds = {s["kind"] for s in self.corpus["samples"]}
        self.assertEqual(
            kinds,
            {
                ve.KIND_LEGIT,
                ve.KIND_LEGIT_DRIFT,
                ve.KIND_PLAUSIBLE_UNCITED,
                ve.KIND_PREFIX_EXTENSION,
                ve.KIND_FABRICATED_SUFFIX,
                ve.KIND_STALE_EVIDENCE,
                ve.KIND_UNRELATED,
            },
        )

    def test_stale_evidence_really_is_absent_from_its_day(self) -> None:
        """A cross-day negative must not be traceable under any equivalence."""
        stale = [s for s in self.corpus["samples"] if s["kind"] == ve.KIND_STALE_EVIDENCE]
        self.assertTrue(stale)
        for sample in stale:
            known = ve._raw_urls(sample["day"])
            with self.subTest(sample=sample["sample_id"]):
                self.assertNotIn(canonical_url(sample["url"]), known)
                self.assertNotIn(ve._path_of(canonical_url(sample["url"])), {ve._path_of(u) for u in known})

    def test_plausible_uncited_really_is_in_its_day(self) -> None:
        """The ceiling class must be genuinely traceable, or it proves nothing."""
        uncited = [
            s for s in self.corpus["samples"] if s["kind"] == ve.KIND_PLAUSIBLE_UNCITED
        ]
        self.assertTrue(uncited)
        for sample in uncited:
            known = ve._raw_urls(sample["day"])
            with self.subTest(sample=sample["sample_id"]):
                self.assertIn(canonical_url(sample["url"]), known)

    def test_labelled_positives_really_are_in_the_capture(self) -> None:
        """A 'should pass' sample that is not actually fetched would be a lie."""
        for sample in self.corpus["samples"]:
            if sample["expect"] != ve.EXPECT_PASS:
                continue
            known = ve._raw_urls(sample["day"])
            with self.subTest(sample=sample["sample_id"], url=sample["url"]):
                self.assertIn(canonical_url(sample["url"]), known)

    def test_labelled_attacks_really_are_absent(self) -> None:
        for sample in self.corpus["samples"]:
            if sample["expect"] != ve.EXPECT_NOT_PASS:
                continue
            known = ve._raw_urls(sample["day"])
            with self.subTest(sample=sample["sample_id"], url=sample["url"]):
                self.assertNotIn(canonical_url(sample["url"]), known)

    def test_no_drift_builder_emits_a_malformed_url(self) -> None:
        """Regression: `www.` was double-applied and tracking params were
        appended with the wrong separator, both of which produced 'drifts' that
        were not equivalent to the original at all."""
        for builder in (ve._drift_scheme, ve._drift_www, ve._drift_tracking):
            for url in (
                "https://example.com/a/b",
                "https://www.example.com/a/b",
                "https://example.com/a?id=7",
                "http://www.example.com/a?x=1&y=2",
            ):
                with self.subTest(builder=builder.__name__, url=url):
                    drifted = builder(url)
                    self.assertIsNotNone(drifted)
                    self.assertEqual(canonical_url(drifted), canonical_url(url))
                    self.assertNotIn("www.www.", drifted)

    def test_trailing_slash_drift_refuses_ambiguous_input(self) -> None:
        self.assertIsNone(ve._drift_trailing_slash("https://example.com/a?x=1"))
        self.assertEqual(
            ve._drift_trailing_slash("https://example.com/a"), "https://example.com/a/"
        )


@unittest.skipUnless(_has_data(), "no daily-trends captures to build a corpus from")
class MatcherComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = ve.build_corpus(ve.available_days(), max_per_kind_per_day=8)
        cls.typed = ve.score(cls.corpus, ve.typed_matcher, matcher_name="typed")
        cls.legacy = ve.score(cls.corpus, ve.legacy_matcher, matcher_name="legacy")

    def test_typed_matcher_never_waves_an_attack_through(self) -> None:
        self.assertEqual(self.typed["false_pass_count"], 0, self.typed["per_kind"])
        self.assertEqual(self.typed["false_pass_rate"], 0.0)

    def test_typed_matcher_does_not_reject_real_citations(self) -> None:
        self.assertEqual(self.typed["false_fail_count"], 0, self.typed["per_kind"])
        self.assertEqual(self.typed["false_fail_rate"], 0.0)

    def test_legacy_matcher_is_measurably_worse(self) -> None:
        """The baseline has to lose on the corpus, or the change bought nothing."""
        self.assertGreater(self.legacy["false_pass_rate"], 0.5)
        self.assertGreater(
            self.legacy["false_pass_count"], self.typed["false_pass_count"]
        )

    def test_uncertainty_is_safe_against_attacks_but_not_strict(self) -> None:
        """A CANNOT_VERIFY on an attack did not leak — but it also did not decide."""
        uncertain = [
            v for v in self.typed["verdicts"] if v["verdict"] == "cannot_verify"
        ]
        self.assertTrue(uncertain, "the corpus must exercise the uncertain path")
        self.assertTrue(all(v["safe"] for v in uncertain))
        self.assertTrue(any(not v["strict"] for v in uncertain))
        self.assertEqual(self.typed["false_pass_count"], 0)
        self.assertEqual(self.typed["false_fail_count"], 0)

    def test_error_counts_match_the_verdicts_they_claim_to_count(self) -> None:
        """Guards the accounting itself: a mis-bucketed verdict would hide a leak."""
        verdicts = self.typed["verdicts"]
        leaks = [
            v for v in verdicts if v["expect"] == ve.EXPECT_NOT_PASS and v["verdict"] == "pass"
        ]
        rejections = [
            v for v in verdicts if v["expect"] == ve.EXPECT_PASS and v["verdict"] == "fail"
        ]
        self.assertEqual(self.typed["false_pass_count"], len(leaks))
        self.assertEqual(self.typed["false_fail_count"], len(rejections))
        self.assertEqual(
            self.typed["cannot_verify_rate"],
            round(
                sum(1 for v in verdicts if v["verdict"] == "cannot_verify") / len(verdicts),
                4,
            ),
        )

    def test_the_typed_matchers_uncertainty_only_lands_on_attacks(self) -> None:
        """The whole justification for shipping a gate that says "I don't know".

        Uncertainty is only acceptable if it is never aimed at real content —
        otherwise the operators learn to ignore it. Here every CANNOT_VERIFY is
        on something that should not have passed.
        """
        uncertain = [
            v for v in self.typed["verdicts"] if v["verdict"] == "cannot_verify"
        ]
        self.assertTrue(uncertain)
        misplaced = [v for v in uncertain if v["expect"] == ve.EXPECT_PASS]
        self.assertEqual(misplaced, [], "uncertainty fired on a legitimate citation")

    def test_uncertainty_is_generated_by_the_two_attack_shapes_it_is_meant_for(self) -> None:
        uncertain_kinds = {
            v["kind"]
            for v in self.typed["verdicts"]
            if v["verdict"] == "cannot_verify"
        }
        self.assertEqual(uncertain_kinds, {ve.KIND_PREFIX_EXTENSION, ve.KIND_FABRICATED_SUFFIX})

    def test_stale_evidence_gets_a_definite_fail_not_a_shrug(self) -> None:
        """Wrong-day evidence is not ambiguous: it is simply not in the capture."""
        stale = [
            v
            for v in self.typed["verdicts"]
            if v["kind"] == ve.KIND_STALE_EVIDENCE
        ]
        self.assertTrue(stale)
        self.assertTrue(all(v["verdict"] == "fail" for v in stale), stale[:3])

    def test_the_report_states_what_provenance_cannot_prove(self) -> None:
        """The ceiling class has to be visible, or a green number overclaims."""
        report = ve.compare_matchers(self.corpus, {"typed": ve.typed_matcher})
        entry = report["matchers"][0]
        self.assertGreater(entry["limit_samples"][ve.KIND_PLAUSIBLE_UNCITED], 0)
        text = ve.markdown(report)
        self.assertIn("provenance", text.lower() + text)

    def test_report_renders_both_matchers(self) -> None:
        report = ve.compare_matchers(
            self.corpus, {"legacy": ve.legacy_matcher, "typed": ve.typed_matcher}
        )
        text = ve.markdown(report)
        self.assertIn("legacy", text)
        self.assertIn("typed", text)
        self.assertIn("漏检率", text)


class LlmJudgePlumbingTests(unittest.TestCase):
    """The LLM baseline is wired but never selected implicitly."""

    def test_parses_a_pass_verdict(self) -> None:
        matcher = ve.llm_matcher(_FakeProvider("PASS"))
        self.assertEqual(matcher("https://example.com/a", {"example.com/a"}), "pass")

    def test_parses_an_explicit_uncertainty(self) -> None:
        matcher = ve.llm_matcher(_FakeProvider("CANNOT_VERIFY"))
        self.assertEqual(
            matcher("https://example.com/a", {"example.com/a"}), "cannot_verify"
        )

    def test_unparseable_output_degrades_to_uncertainty_not_pass(self) -> None:
        matcher = ve.llm_matcher(_FakeProvider("I think it is probably fine?"))
        self.assertEqual(
            matcher("https://example.com/a", {"example.com/a"}), "cannot_verify"
        )

    def test_provider_receives_the_fetched_urls(self) -> None:
        provider = _FakeProvider("FAIL")
        matcher = ve.llm_matcher(provider)
        matcher("https://example.com/a", {"example.com/known"})
        self.assertIn("example.com/known", provider.calls[0][1]["content"])


class CorpusSerialisationTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        import tempfile

        corpus = {
            "generated_at": "2026-09-26T00:00:00+08:00",
            "days": ["2026-09-22"],
            "samples": [
                ve.Sample(
                    sample_id="x-1",
                    kind=ve.KIND_LEGIT,
                    url="https://example.com/a",
                    day="2026-09-22",
                    expect=ve.EXPECT_PASS,
                ).to_dict()
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = ve.save_corpus(Path(tmp) / "corpus.json", corpus)
            reloaded = ve.load_corpus(path)
        self.assertEqual(reloaded, corpus)
        self.assertEqual(json.loads(json.dumps(reloaded)), corpus)


@unittest.skipUnless(_has_data(), "no daily-trends captures to diagnose")
class ContentDebtTests(unittest.TestCase):
    """Diagnosis only — history is reported, never silently rewritten."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.report = ve.content_debt(ve.available_days())

    def test_every_day_is_accounted_for(self) -> None:
        self.assertEqual(self.report["days"], len(ve.available_days()))
        self.assertEqual(
            self.report["clean"] + self.report["dirty"], self.report["days"]
        )

    def test_clean_and_dirty_are_decided_by_the_gates(self) -> None:
        for row in self.report["rows"]:
            with self.subTest(day=row["day"]):
                self.assertEqual(row["clean"], not row["blockers"])
                for gate in ("structure", "references", "verifiable"):
                    if row[gate] is False:
                        self.assertIn(gate, row["blockers"])

    def test_markdown_names_the_blocking_gate(self) -> None:
        text = ve.content_debt_markdown(self.report)
        self.assertIn("历史内容债", text)
        for row in self.report["rows"]:
            self.assertIn(row["day"], text)
        self.assertIn("只做诊断", text)

    def test_a_day_with_no_capture_is_skipped_not_guessed(self) -> None:
        report = ve.content_debt(["1999-01-01"])
        self.assertEqual(report["days"], 0)
        self.assertEqual(report["rows"], [])


class BaselineRegressionTests(unittest.TestCase):
    """A gate that quietly gets worse while every test passes is the failure
    mode this whole module exists to prevent."""

    @staticmethod
    def _report(**matchers):
        return {
            "samples": 100,
            "days": ["2026-09-25"],
            "matchers": [
                {"matcher": name, **metrics} for name, metrics in matchers.items()
            ],
        }

    def test_identical_run_is_not_a_regression(self) -> None:
        report = self._report(typed={"false_pass_rate": 0.0, "false_fail_rate": 0.0})
        baseline = ve.baseline_from(report)
        self.assertTrue(ve.check_baseline(report, baseline)["ok"])

    def test_a_leak_regression_is_caught(self) -> None:
        baseline = ve.baseline_from(
            self._report(typed={"false_pass_rate": 0.0, "false_fail_rate": 0.0})
        )
        worse = self._report(typed={"false_pass_rate": 0.05, "false_fail_rate": 0.0})
        verdict = ve.check_baseline(worse, baseline)
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["regressions"][0]["metric"], "false_pass_rate")
        self.assertEqual(verdict["regressions"][0]["baseline"], 0.0)
        self.assertEqual(verdict["regressions"][0]["now"], 0.05)

    def test_a_false_reject_regression_is_also_caught(self) -> None:
        """Improving leak rate by rejecting everything is not an improvement."""
        baseline = ve.baseline_from(
            self._report(typed={"false_pass_rate": 0.0, "false_fail_rate": 0.0})
        )
        worse = self._report(typed={"false_pass_rate": 0.0, "false_fail_rate": 0.9})
        verdict = ve.check_baseline(worse, baseline)
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["regressions"][0]["metric"], "false_fail_rate")

    def test_an_improvement_is_not_a_regression(self) -> None:
        baseline = ve.baseline_from(
            self._report(legacy={"false_pass_rate": 0.6, "false_fail_rate": 0.0})
        )
        better = self._report(legacy={"false_pass_rate": 0.1, "false_fail_rate": 0.0})
        self.assertTrue(ve.check_baseline(better, baseline)["ok"])

    def test_matchers_absent_from_the_baseline_are_not_compared(self) -> None:
        baseline = ve.baseline_from(
            self._report(legacy={"false_pass_rate": 0.6, "false_fail_rate": 0.0})
        )
        only_new = self._report(llm={"false_pass_rate": 0.9, "false_fail_rate": 0.9})
        verdict = ve.check_baseline(only_new, baseline)
        self.assertTrue(verdict["ok"])
        self.assertEqual(verdict["compared_matchers"], [])

    def test_baseline_round_trips_through_disk(self) -> None:
        import tempfile

        report = self._report(typed={"false_pass_rate": 0.0, "false_fail_rate": 0.0})
        with tempfile.TemporaryDirectory() as tmp:
            path = ve.save_baseline(Path(tmp) / "baseline.json", report)
            loaded = ve.load_baseline(path)
        self.assertEqual(loaded["version"], ve.BASELINE_VERSION)
        self.assertEqual(loaded["samples"], 100)
        self.assertIn("typed", loaded["matchers"])


if __name__ == "__main__":
    unittest.main()
