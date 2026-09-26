"""Probes are only worth anything if they can fail.

Each test here either runs the real probes, or swaps in a deliberately broken
validator and asserts the probes notice. A probe suite that passes against
*everything* is decoration.
"""

from __future__ import annotations

import unittest
from unittest import mock

from harness import validator_probes as vp
from harness import validators as validators_mod


class ProbeSuiteHealthTests(unittest.TestCase):
    def test_the_real_refund_gate_satisfies_every_probe(self) -> None:
        report = vp.run_probes(vp.refund_probes())
        self.assertTrue(report["ok"], report["outcomes"])
        self.assertEqual(report["missed"], 0)
        self.assertEqual(report["overblocked"], 0)

    def test_the_suite_covers_both_directions(self) -> None:
        """A suite of only-negative probes would pass for a gate that rejects
        everything, which is not a gate anyone can ship."""
        probes = vp.refund_probes()
        expects = {probe.expect for probe in probes}
        self.assertEqual(expects, {vp.EXPECT_PASS, vp.EXPECT_NOT_PASS})

    def test_a_permissive_validator_is_caught(self) -> None:
        permissive = lambda *a, **k: {"ok": True, "decision": "pass", "detail": "sure"}
        with mock.patch.object(validators_mod, "get", return_value=permissive):
            report = vp.run_probes(vp.refund_probes())
        self.assertFalse(report["ok"])
        self.assertGreater(report["missed"], 0)
        self.assertEqual(report["overblocked"], 0)

    def test_an_overblocking_validator_is_caught(self) -> None:
        reject_all = lambda *a, **k: {"ok": False, "decision": "fail", "detail": "no"}
        with mock.patch.object(validators_mod, "get", return_value=reject_all):
            report = vp.run_probes(vp.refund_probes())
        self.assertFalse(report["ok"])
        self.assertGreater(report["overblocked"], 0)
        self.assertEqual(report["missed"], 0)

    def test_a_crashing_validator_is_recorded_not_propagated(self) -> None:
        def boom(*a, **k):
            raise RuntimeError("validator exploded")

        with mock.patch.object(validators_mod, "get", return_value=boom):
            report = vp.run_probes(vp.refund_probes())
        self.assertFalse(report["ok"])
        self.assertIn("RuntimeError", report["outcomes"][0]["detail"])


class ProbeRegistryTests(unittest.TestCase):
    def test_the_merge_gate_has_its_own_probe_group(self) -> None:
        probes = vp.all_probes("merge")
        self.assertTrue(probes)
        self.assertTrue(all(p.validator == "pr_merge_gate" for p in probes))

    def test_the_merge_group_covers_the_case_ci_cannot_see(self) -> None:
        """A green test bar plus an undecidable blast radius must still block.

        That single case is the whole reason the merge decision was split into
        three judgements instead of trusting the CI result.
        """
        probes = vp.all_probes("merge")
        scope = [p for p in probes if p.probe_id == "merge-scope-undecidable"]
        self.assertEqual(len(scope), 1)
        decisions = {c["check"]: c["decision"] for c in scope[0].payload["checks"]}
        self.assertEqual(decisions["TESTS_PASS"], "pass")
        self.assertEqual(decisions["NO_UNINTENDED_SCOPE"], "cannot_verify")
        self.assertEqual(scope[0].expect, vp.EXPECT_NOT_PASS)

    def test_both_probe_groups_hold_on_the_real_validators(self) -> None:
        report = vp.run_probes(vp.all_probes())
        self.assertTrue(report["ok"], report["outcomes"])

    def test_unknown_group_is_rejected(self) -> None:
        with self.assertRaises(KeyError):
            vp.all_probes("does_not_exist")

    def test_all_probes_are_named_and_unique(self) -> None:
        probes = vp.all_probes()
        ids = [probe.probe_id for probe in probes]
        self.assertEqual(len(ids), len(set(ids)), "duplicate probe ids")
        self.assertTrue(all(probe.validator for probe in probes))

    def test_every_probe_points_at_a_registered_validator(self) -> None:
        known = set(validators_mod.names())
        for probe in vp.all_probes():
            with self.subTest(probe=probe.probe_id):
                self.assertIn(probe.validator, known)


class ProbeReportTests(unittest.TestCase):
    def test_markdown_lists_every_probe_and_both_error_directions(self) -> None:
        report = vp.run_probes(vp.refund_probes())
        text = vp.markdown(report)
        for probe in vp.refund_probes():
            self.assertIn(probe.probe_id, text)
        self.assertIn("漏判", text)
        self.assertIn("误拦", text)


if __name__ == "__main__":
    unittest.main()
