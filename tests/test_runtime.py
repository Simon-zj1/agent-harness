from __future__ import annotations

import json
import unittest
from pathlib import Path

from harness import memory
from harness.errors import LockBusy
from harness.ledger import Ledger
from harness.locks import FileLock
from harness.runtime import RunOptions, Runner

from .helpers import BODY_STEP, DEGRADE_STEP, FAIL_STEP, OK_STEP, Sandbox, demo_task_body


class RuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sandbox = Sandbox()
        self.sandbox.activate()
        self.work = self.sandbox.root / "work"
        self.work.mkdir(parents=True, exist_ok=True)
        self.sandbox.write_task("demo", demo_task_body(self.work))
        self.sandbox.step("demo", "first", OK_STEP.replace("{name}", "first"))
        self.sandbox.step(
            "demo", "second", BODY_STEP.replace("{CONTENT!r}", repr({"ok": True}))
        )
        self.runner = Runner()

    def tearDown(self) -> None:
        self.runner.ledger.close()
        self.sandbox.deactivate()

    def _run(self, **kwargs):
        options = RunOptions(task="demo", date="2026-09-22", notify=False, **kwargs)
        return self.runner.run(options)

    def test_successful_run_records_steps_validators_and_memory(self) -> None:
        outcome = self._run()
        self.assertTrue(outcome.ok, outcome.run.error)
        self.assertEqual(outcome.run.status, "success")
        self.assertEqual([s["id"] for s in outcome.run.steps], ["first", "second"])
        self.assertTrue(all(v["ok"] for v in outcome.run.validators))
        self.assertTrue(list((self.sandbox.home / "memory" / "runs").glob("*.md")))
        self.assertTrue((self.sandbox.home / "runs" / outcome.run.run_id / "context.json").is_file())

    def test_second_run_without_force_is_skipped(self) -> None:
        first = self._run()
        second = self._run()
        self.assertTrue(second.skipped)
        self.assertEqual(second.run.run_id, first.run.run_id)
        self.assertIn("already finished", second.message)

    def test_force_creates_a_new_run(self) -> None:
        first = self._run()
        second = self._run(force=True)
        self.assertFalse(second.skipped)
        self.assertNotEqual(second.run.run_id, first.run.run_id)

    def test_dry_run_is_not_blocked_and_does_not_mark_done(self) -> None:
        dry = self._run(dry_run=True)
        self.assertTrue(dry.run.dry_run)
        ledger = Ledger()
        self.assertIsNone(ledger.find_done("demo", "2026-09-22"))
        ledger.close()
        real = self._run()
        self.assertFalse(real.skipped)

    def test_degraded_step_degrades_the_run_but_keeps_going(self) -> None:
        self.sandbox.write_task(
            "demo",
            demo_task_body(
                self.work,
                extra_step='\n[[steps]]\nid = "flaky"\ncommand = ["python3", "steps/flaky.py"]\nallow_degrade = true\n',
            ),
        )
        self.sandbox.step("demo", "flaky", DEGRADE_STEP.replace("{name}", "flaky"))
        outcome = self._run()
        self.assertEqual(outcome.run.status, "degraded")
        self.assertTrue(outcome.ok)
        self.assertIn("source unavailable", outcome.run.notes)

    def test_failing_step_fails_the_run(self) -> None:
        self.sandbox.write_task(
            "demo",
            demo_task_body(
                self.work,
                extra_step='\n[[steps]]\nid = "boom"\ncommand = ["python3", "steps/boom.py"]\n',
            ),
        )
        self.sandbox.step("demo", "boom", FAIL_STEP.replace("{name}", "boom"))
        outcome = self._run()
        self.assertEqual(outcome.run.status, "failed")
        self.assertEqual(outcome.run.failure_class, "step_failed")
        self.assertIn("boom", outcome.run.error)

    def test_failed_required_validator_blocks_publish_step(self) -> None:
        self.sandbox.write_task(
            "demo",
            demo_task_body(
                self.work,
                extra_step='\n[[steps]]\nid = "publish"\ncommand = ["python3", "steps/publish.py"]\nwhen = "validation_passed"\nrequires_publish = true\n',
            )
            .replace(
                'args = { path = "{run_dir}/content.json" }',
                'args = { path = "{run_dir}/missing.json" }',
            ),
        )
        self.sandbox.step("demo", "publish", OK_STEP.replace("{name}", "published"))
        outcome = self._run(publish=True)
        self.assertEqual(outcome.run.status, "failed")
        self.assertEqual(outcome.run.failure_class, "validation_failed")
        self.assertNotIn("publish", [s["id"] for s in outcome.run.steps if s["status"] == "ok"])

    def test_publish_step_is_skipped_when_publish_disabled(self) -> None:
        self.sandbox.write_task(
            "demo",
            demo_task_body(
                self.work,
                extra_step='\n[[steps]]\nid = "publish"\ncommand = ["python3", "steps/publish.py"]\nwhen = "validation_passed"\nrequires_publish = true\n',
            ),
        )
        self.sandbox.step("demo", "publish", OK_STEP.replace("{name}", "published"))
        outcome = self._run(publish=False)
        publish_step = next(s for s in outcome.run.steps if s["id"] == "publish")
        self.assertEqual(publish_step["status"], "skipped")
        self.assertFalse(outcome.run.published)

    def test_lock_scope_blocks_concurrent_run(self) -> None:
        lock = FileLock("demo.task")
        lock.acquire()
        try:
            with self.assertRaises(LockBusy):
                self._run()
        finally:
            lock.release()

    def test_step_metrics_are_aggregated(self) -> None:
        outcome = self._run()
        self.assertEqual(outcome.run.metrics.get("first"), 1)
        self.assertIn("duration_ms", outcome.run.metrics)

    def test_memory_summary_is_searchable(self) -> None:
        outcome = self._run()
        memory.reindex()
        results = memory.search("demo", limit=5)
        self.assertTrue(results)
        self.assertTrue(
            any(outcome.run.run_id in str(entry.get("path", "")) for entry in results)
            or results
        )
        self.assertGreaterEqual(memory.stats()["entries"], 1)

    def test_run_artifacts_are_recorded_on_disk(self) -> None:
        outcome = self._run()
        run_dir = self.sandbox.home / "runs" / outcome.run.run_id
        first = Path(run_dir / "artifact-first.json")
        self.assertTrue(first.is_file())
        self.assertEqual(json.loads(first.read_text())["date"], "2026-09-22")


if __name__ == "__main__":
    unittest.main()
