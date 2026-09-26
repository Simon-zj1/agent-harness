"""When the gate blocks, it must leave something actionable behind.

Typed results carry the receipts (evidence) and the fix (remediation). If those
are dropped at the ledger boundary, a blocked run is a dead end and the operator
has to reverse-engineer which claim broke.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from harness import decisions, paths
from harness.decisions import DecisionPolicy
from harness.ledger import Ledger

from .helpers import Sandbox


class RepairBriefTests(unittest.TestCase):
    def test_only_failing_validators_appear(self) -> None:
        results = [
            {"name": "ok_one", "ok": True, "decision": "pass"},
            {
                "name": "bad_one",
                "ok": False,
                "decision": "fail",
                "failure_class": "fabricated_source",
                "detail": "1/2 traced",
                "remediation": "re-fetch or replace the citation",
                "policy_action": "block",
                "evidence": [{"ref": "ref#1", "detail": "no match", "url": "https://x/y"}],
                "metrics": {"cited": 2},
            },
        ]
        brief = decisions.repair_brief(results)
        self.assertEqual(brief["blocking"], ["bad_one"])
        self.assertEqual(brief["count"], 1)
        entry = brief["entries"][0]
        self.assertEqual(entry["failure_class"], "fabricated_source")
        self.assertEqual(entry["remediation"], "re-fetch or replace the citation")
        self.assertEqual(entry["evidence"][0]["ref"], "ref#1")

    def test_a_clean_run_produces_an_empty_brief(self) -> None:
        brief = decisions.repair_brief([{"name": "a", "ok": True, "decision": "pass"}])
        self.assertEqual(brief["count"], 0)
        self.assertEqual(brief["blocking"], [])

    def test_legacy_results_without_a_decision_still_appear(self) -> None:
        """A validator predating typed decisions must not vanish from the brief."""
        brief = decisions.repair_brief([{"name": "old", "ok": False, "detail": "nope"}])
        self.assertEqual(brief["count"], 1)
        self.assertEqual(brief["entries"][0]["decision"], "fail")

    def test_uncertainty_is_included_but_marked(self) -> None:
        brief = decisions.repair_brief(
            [
                {
                    "name": "v",
                    "ok": False,
                    "decision": "cannot_verify",
                    "failure_class": "unknown_variant",
                    "policy_action": "block",
                }
            ]
        )
        entry = brief["entries"][0]
        self.assertEqual(entry["decision"], "cannot_verify")
        self.assertEqual(entry["policy_action"], "block")


class RemediationPersistenceTests(unittest.TestCase):
    def test_remediation_survives_the_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(Path(tmp) / "runs.db")
            ledger.record_validation(
                "run-1",
                {
                    "name": "v",
                    "ok": False,
                    "detail": "blocked",
                    "decision": "fail",
                    "failure_class": "fabricated_source",
                    "evidence": [{"ref": "ref#1", "detail": "no match"}],
                    "policy_action": "block",
                    "remediation": "replace the citation",
                },
            )
            row = ledger._conn.execute(
                "SELECT decision, failure_class, policy_action, remediation "
                "FROM validations WHERE run_id='run-1'"
            ).fetchone()
        self.assertEqual(row["decision"], "fail")
        self.assertEqual(row["failure_class"], "fabricated_source")
        self.assertEqual(row["policy_action"], "block")
        self.assertEqual(row["remediation"], "replace the citation")


class BlockedRunArtifactTests(unittest.TestCase):
    """Integration: a blocked run writes runs/<id>/validation-failures.json."""

    TASK = """
name = "demo"
date_mode = "today"
timezone = "Asia/Shanghai"
allowed_tools = ["fs_read", "fs_write"]
readable_paths = ["{run_dir}"]
writable_paths = ["{run_dir}"]

[[steps]]
id = "first"
command = ["python3", "steps/first.py"]
"""

    STEP = (
        "from harness import stepctx\n"
        "ctx = stepctx.load()\n"
        "(ctx.run_dir / 'artifact.json').write_text('{}')\n"
        "raise SystemExit(stepctx.finish(ctx, 'ok'))\n"
    )

    def setUp(self) -> None:
        self.sandbox = Sandbox()
        self.sandbox.activate()
        self.work = self.sandbox.root / "work"
        self.work.mkdir(parents=True, exist_ok=True)
        self.sandbox.write_task("demo", self.TASK)
        self.sandbox.step("demo", "first", self.STEP)

    def tearDown(self) -> None:
        self.sandbox.deactivate()

    def _add_validator(self, body: str) -> None:
        task = self.sandbox.task_dir("demo") / "task.toml"
        task.write_text(task.read_text(encoding="utf-8") + body, encoding="utf-8")

    def _last_run_dir(self):
        from harness.cli import build_parser, cmd_run

        args = build_parser().parse_args(["run", "demo", "--date", "2026-09-25", "--dry-run"])
        cmd_run(args)
        runs = sorted(paths.runs_dir().glob("2026-09-25-demo-*"), key=lambda p: p.name)
        return runs[-1] if runs else None

    def test_blocked_run_writes_a_repair_brief(self) -> None:
        # files_exist on a path the run never creates: a deterministic block.
        self._add_validator(
            '\n[[validators]]\nname = "files_exist"\n'
            'args = { paths = ["{run_dir}/never-written.txt"] }\n'
        )
        run_dir = self._last_run_dir()
        self.assertIsNotNone(run_dir, "no run directory was produced")
        brief_path = run_dir / "validation-failures.json"
        self.assertTrue(brief_path.is_file(), f"missing {brief_path}")
        brief = json.loads(brief_path.read_text(encoding="utf-8"))
        self.assertIn("files_exist", brief["blocking"])
        self.assertGreaterEqual(brief["count"], 1)
        self.assertEqual(brief["entries"][0]["policy_action"], "block")

    def test_a_clean_run_writes_no_brief(self) -> None:
        self._add_validator(
            "\n[[validators]]\nname = \"files_exist\"\n"
            "args = { paths = [\"{run_dir}/artifact.json\"] }\n"
        )
        run_dir = self._last_run_dir()
        self.assertIsNotNone(run_dir)
        self.assertFalse((run_dir / "validation-failures.json").exists())


if __name__ == "__main__":
    unittest.main()
