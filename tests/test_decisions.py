"""Typed decisions are the acceptance gate's own contract, so they get tested
the same way the validators do — including the attack that motivated them."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from harness import validators
from harness.decisions import (
    Decision,
    DecisionPolicy,
    FailureClass,
    canonical_url,
    classify_url,
    combine,
)
from harness.refund_guard import decide

FIXTURES = Path(__file__).resolve().parent / "fixtures"
COLLISION_RAW = FIXTURES / "prefix-collision" / "raw.json"
COLLISION_CONTENT = FIXTURES / "prefix-collision" / "content.json"
REAL_CONTENT = FIXTURES / "daily-trends-2026-09-25" / "content.json"
REAL_RAW = FIXTURES / "daily-trends-2026-09-25" / "raw.json"
REFUND_STEP = (
    Path(__file__).resolve().parent.parent / "tasks" / "refund-guard" / "steps" / "10_decide.py"
)


def _raw_urls(path: Path = COLLISION_RAW) -> set[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {canonical_url(u) for u in validators._collect_urls(raw)}


def _load_refund_step():
    spec = importlib.util.spec_from_file_location("refund_step", REFUND_STEP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class UrlGroundTruthTests(unittest.TestCase):
    """The matcher must separate 'verified' from 'cannot tell'."""

    def setUp(self) -> None:
        self.known = _raw_urls()
        self.base = "simonwillison.net/2026/sep/22/llm"
        self.assertTrue(self.base in self.known, "fixture drift: base url missing")

    def test_exact_match_passes(self) -> None:
        verdict = classify_url("https://simonwillison.net/2026/Sep/22/llm/", self.known)
        self.assertIs(verdict.decision, Decision.PASS)

    def test_enumerated_drift_passes(self) -> None:
        # scheme / www / trailing slash / tracking params are known, harmless.
        for url in (
            "http://simonwillison.net/2026/Sep/22/llm",
            "https://www.simonwillison.net/2026/Sep/22/llm/",
            "https://simonwillison.net/2026/Sep/22/llm/?utm_source=newsletter",
        ):
            with self.subTest(url=url):
                self.assertIs(classify_url(url, self.known).decision, Decision.PASS)

    def test_fabricated_url_extending_a_real_one_is_not_verified(self) -> None:
        """Regression: this used to return True and reach a published page.

        `…/llm` is a real fetched URL, so anything that merely extends it shares
        a prefix with ground truth. The old matcher treated that as a match.
        """
        for url in (
            "https://simonwillison.net/2026/Sep/22/llm-anthropic-v99-fake/",
            "https://simonwillison.net/2026/Sep/22/llm/this-page-never-existed",
        ):
            with self.subTest(url=url):
                verdict = classify_url(url, self.known)
                self.assertIs(verdict.decision, Decision.CANNOT_VERIFY)
                self.assertIs(verdict.failure_class, FailureClass.UNKNOWN_VARIANT)
                self.assertTrue(verdict.matched, "must name the prefix it matched")

    def test_url_absent_from_the_capture_fails(self) -> None:
        verdict = classify_url("https://example.com/totally-made-up", self.known)
        self.assertIs(verdict.decision, Decision.FAIL)
        self.assertIs(verdict.failure_class, FailureClass.FABRICATED_SOURCE)

    def test_arxiv_version_is_a_known_drift(self) -> None:
        known = {"arxiv.org/abs/2609.24974"}
        self.assertIs(
            classify_url("https://arxiv.org/abs/2609.24974v1", known).decision,
            Decision.PASS,
        )

    def test_query_parameters_on_the_fetched_url_are_a_known_drift(self) -> None:
        """Regression: the capture keeps whatever params the fetch saw
        (`?st=…&reflink=…`), while articles cite the bare path. Treating that as
        unverifiable rejected a legitimate citation the old gate accepted."""
        known = {"wsj.com/business/x-c5ebcddc3?st=qk6qch&reflink=desktopwebshare"}
        verdict = classify_url(
            "https://www.wsj.com/business/x-c5ebcddc3", known
        )
        self.assertIs(verdict.decision, Decision.PASS)
        self.assertEqual(verdict.drift, "query_suffix")

    def test_a_fabricated_path_suffix_is_still_not_a_drift(self) -> None:
        """The query rule must not become a general purpose loophole.

        A path extension of a URL that carries a query is not even ambiguous —
        the paths differ, so it is a straight FAIL. A path extension of a bare
        URL is ambiguous (CANNOT_VERIFY) because the prefix is genuinely real.
        """
        with_query = classify_url(
            "https://example.com/page/invented", {"example.com/page?st=abc"}
        )
        self.assertIs(with_query.decision, Decision.FAIL)

        bare = classify_url(
            "https://example.com/some/long/path/invented",
            {"example.com/some/long/path"},
        )
        self.assertIs(bare.decision, Decision.CANNOT_VERIFY)

    def test_a_shared_prefix_on_a_short_url_is_not_treated_as_ambiguous(self) -> None:
        """`x.com/a` is a prefix of half the internet; ambiguity there would be
        noise, so short URLs go straight to FAIL."""
        verdict = classify_url("https://x.com/a/b/c", {"x.com/a"})
        self.assertIs(verdict.decision, Decision.FAIL)


class PolicyTests(unittest.TestCase):
    def test_fail_closed_is_the_default(self) -> None:
        policy = DecisionPolicy()
        self.assertFalse(policy.allows(Decision.CANNOT_VERIFY))
        self.assertFalse(policy.allows(Decision.ABSTAIN))
        self.assertFalse(policy.allows(Decision.FAIL))
        self.assertTrue(policy.allows(Decision.PASS))

    def test_warn_records_uncertainty_without_blocking(self) -> None:
        policy = DecisionPolicy.from_table({"on_cannot_verify": "warn"})
        result = {
            "name": "x",
            "decision": "cannot_verify",
            "ok": False,
        }
        from harness.decisions import apply_policy

        apply_policy(result, policy)
        self.assertTrue(result["ok"])
        self.assertEqual(result["policy_action"], "warn")
        self.assertEqual(result["decision"], "cannot_verify")

    def test_rejects_unknown_policy_value(self) -> None:
        with self.assertRaises(ValueError):
            DecisionPolicy.from_table({"on_fail": "ignore"})


class AggregationTests(unittest.TestCase):
    def test_failure_beats_uncertainty_beats_pass(self) -> None:
        results = [
            {"name": "a", "decision": "pass", "ok": True, "policy_action": "allow"},
            {"name": "b", "decision": "cannot_verify", "ok": False, "policy_action": "block"},
            {"name": "c", "decision": "fail", "ok": False, "policy_action": "block"},
        ]
        summary = combine(results)
        self.assertEqual(summary["decision"], "fail")
        self.assertEqual(summary["blocking"], ["b", "c"])
        self.assertEqual(summary["counts"]["pass"], 1)

    def test_one_uncertainty_makes_the_run_uncertain(self) -> None:
        results = [
            {"name": "a", "decision": "pass", "ok": True, "policy_action": "allow"},
            {"name": "b", "decision": "cannot_verify", "ok": False, "policy_action": "block"},
        ]
        self.assertEqual(combine(results)["decision"], "cannot_verify")


class ValidatorSurfaceTests(unittest.TestCase):
    """The typed fields are additive: old callers keep working."""

    def test_legacy_keys_survive(self) -> None:
        result = validators.daily_trends_verifiable(REAL_CONTENT, REAL_RAW)
        for key in ("name", "ok", "detail", "failures", "metrics"):
            self.assertIn(key, result)
        for key in ("decision", "failure_class", "evidence", "policy_action"):
            self.assertIn(key, result)
        self.assertEqual(result["decision"], "pass")
        self.assertTrue(result["ok"])

    def test_real_content_still_traces_back(self) -> None:
        result = validators.daily_trends_verifiable(REAL_CONTENT, REAL_RAW)
        self.assertGreaterEqual(result["metrics"]["verifiable_ratio"], 0.99)
        self.assertEqual(result["metrics"]["cannot_verify"], 0)
        self.assertEqual(result["metrics"]["fail"], 0)

    def test_fabricated_variant_is_reported_with_evidence(self) -> None:
        """The exact shape that defeated the old gate, on a fixture that owns
        its own ground truth instead of borrowing today's live capture."""
        content = json.loads(COLLISION_CONTENT.read_text(encoding="utf-8"))
        # ref#1 is the real `.../llm/`; the fake extends a *sibling* that is also
        # fetched, which is why prefix matching used to wave it through.
        content["references"][0]["url"] = (
            "https://simonwillison.net/2026/Sep/22/llm-anthropic-v99-fake/"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "content.json"
            path.write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")
            result = validators.daily_trends_verifiable(path, COLLISION_RAW)
        self.assertFalse(result["ok"], "cannot_verify must not be reported as verified")
        self.assertEqual(result["decision"], "cannot_verify")
        self.assertTrue(result["evidence"], "a non-PASS decision needs receipts")
        self.assertTrue(result["remediation"])


class RefundGuardTests(unittest.TestCase):
    """Same kernel, decision that moves money."""

    @classmethod
    def setUpClass(cls) -> None:
        module = _load_refund_step()
        book = json.loads(
            (REFUND_STEP.parent.parent / "fixtures" / "orders.json").read_text(
                encoding="utf-8"
            )
        )
        cls.ledger = set(book["payment_ledger"])
        cls.policy = book["policy"]
        cls.orders = book["orders"]
        cls.decide = staticmethod(decide)

    def test_eligible_order_is_approved(self) -> None:
        verdict = self.decide(self.orders[0], self.policy, self.ledger)
        self.assertEqual(verdict["decision"], "pass")
        self.assertEqual(verdict["action"], "approve")

    def test_out_of_window_is_denied_with_a_named_rule(self) -> None:
        verdict = self.decide(self.orders[1], self.policy, self.ledger)
        self.assertEqual(verdict["decision"], "fail")
        self.assertEqual(verdict["action"], "deny")
        self.assertIn("refund_window", verdict["failed_checks"])

    def test_unresolvable_payment_reference_denies(self) -> None:
        verdict = self.decide(self.orders[2], self.policy, self.ledger)
        self.assertEqual(verdict["decision"], "cannot_verify")
        self.assertEqual(verdict["action"], "deny", "fail-closed: unknown denies")
        self.assertIn("payment_reference", verdict["unverified_checks"])

    def test_validator_rejects_approving_an_unverifiable_refund(self) -> None:
        checks = [
            {"check": name, "decision": "pass", "detail": "stub"}
            for name in ("order_status", "refund_window", "currency")
        ] + [
            {
                "check": "payment_reference",
                "decision": "cannot_verify",
                "detail": "not found in ledger",
            }
        ]
        bad = {
            "results": [
                {
                    "order_id": "ord_x",
                    "decision": "cannot_verify",
                    "action": "approve",
                    "checks": checks,
                    "unverified_checks": ["payment_reference"],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decisions.json"
            path.write_text(json.dumps(bad), encoding="utf-8")
            result = validators.refund_decisions_fail_closed(path)
        self.assertFalse(result["ok"])
        self.assertEqual(result["failure_class"], "policy_violation")
        self.assertIn("fail-closed violated", result["failures"][0]["issue"])

    def test_validator_accepts_the_real_fixture_output(self) -> None:
        import os

        with tempfile.TemporaryDirectory() as tmp:
            os.environ["AGENT_RUN_DIR"] = tmp
            os.environ["AGENT_STEP_ID"] = "10_decide"
            try:
                module = _load_refund_step()
                module.main()
                result = validators.refund_decisions_fail_closed(
                    Path(tmp) / "decisions.json"
                )
            finally:
                os.environ.pop("AGENT_RUN_DIR", None)
                os.environ.pop("AGENT_STEP_ID", None)
        self.assertTrue(result["ok"], result.get("failures"))
        self.assertEqual(result["metrics"]["cannot_verify"], 2)


class GateCompletenessTests(unittest.TestCase):
    """A gate must reject missing evidence, not only contradictory evidence."""

    def test_pr_merge_gate_requires_all_three_judgements(self) -> None:
        payload = {
            "range": "HEAD~1..HEAD",
            "scope": ["harness/"],
            "checks": [
                {"check": "TESTS_PASS", "decision": "pass", "detail": "ok"},
                {"check": "ARCHITECTURE_OK", "decision": "pass", "detail": "ok"},
            ],
            "merge": "allow",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "review.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = validators.pr_merge_gate(path)
        self.assertFalse(result["ok"])
        self.assertIn("missing required checks", result["detail"] + str(result["failures"]))

    def test_refund_gate_recomputes_ground_truth(self) -> None:
        orders = REFUND_STEP.parent.parent / "fixtures" / "orders.json"
        checks = [
            {"check": name, "decision": "pass", "detail": "fake"}
            for name in ("order_status", "refund_window", "payment_reference", "currency")
        ]
        payload = {
            "results": [
                {
                    "order_id": "ord_1002",
                    "decision": "pass",
                    "action": "approve",
                    "checks": checks,
                    "failed_checks": [],
                    "unverified_checks": [],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decisions.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = validators.refund_decisions_fail_closed(path, orders_path=orders)
        # ord_1002 is out of the refund window, so an all-pass artifact is a
        # fabricated decision. The other fixture orders are also missing.
        self.assertFalse(result["ok"])
        self.assertIn("decision mismatch", str(result["failures"]))
        self.assertIn("missing decisions for orders", str(result["failures"]))


if __name__ == "__main__":
    unittest.main()
