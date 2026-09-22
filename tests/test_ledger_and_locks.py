from __future__ import annotations

import os
import unittest

from harness import paths
from harness.errors import LockBusy
from harness.ledger import Ledger, RunRow
from harness.locks import FileLock

from .helpers import Sandbox


class LedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sandbox = Sandbox()
        self.sandbox.activate()
        self.ledger = Ledger()

    def tearDown(self) -> None:
        self.ledger.close()
        self.sandbox.deactivate()

    def _row(self, run_id: str, date: str = "2026-09-22", status: str = "success") -> RunRow:
        row = RunRow(
            run_id=run_id,
            task="demo",
            target_date=date,
            status=status,
            started_at="2026-09-22T23:00:00+08:00",
        )
        self.ledger.insert_run(row)
        return row

    def test_successful_run_is_found_for_idempotency(self) -> None:
        self._row("r1")
        self.assertIsNotNone(self.ledger.find_done("demo", "2026-09-22"))

    def test_second_success_for_same_task_date_is_rejected_by_index(self) -> None:
        self._row("r1")
        with self.assertRaises(Exception):
            self._row("r2")

    def test_dry_runs_do_not_block_idempotency(self) -> None:
        row = RunRow(
            run_id="dry1",
            task="demo",
            target_date="2026-09-22",
            status="success",
            dry_run=True,
            started_at="2026-09-22T23:00:00+08:00",
        )
        self.ledger.insert_run(row)
        self.assertIsNone(self.ledger.find_done("demo", "2026-09-22"))

    def test_failed_run_does_not_block_rerun(self) -> None:
        self._row("r1", status="failed")
        self.assertIsNone(self.ledger.find_done("demo", "2026-09-22"))

    def test_stale_running_run_is_marked_crashed(self) -> None:
        row = RunRow(
            run_id="stale",
            task="demo",
            target_date="2026-09-22",
            status="running",
            started_at="2026-09-22T23:00:00+08:00",
            pid=999_999_999,
        )
        self.ledger.insert_run(row)
        stale = self.ledger.mark_stale_running("demo")
        self.assertEqual(stale, ["stale"])
        self.assertEqual(self.ledger.get_run("stale").status, "crashed")

    def test_current_process_run_is_not_marked_stale(self) -> None:
        row = RunRow(
            run_id="live",
            task="demo",
            target_date="2026-09-22",
            status="running",
            started_at="2026-09-22T23:00:00+08:00",
            pid=os.getpid(),
        )
        self.ledger.insert_run(row)
        self.assertEqual(self.ledger.mark_stale_running("demo"), [])

    def test_totals_count_outcomes(self) -> None:
        self._row("r1")
        self._row("r2", date="2026-09-21", status="failed")
        totals = self.ledger.totals("demo")
        self.assertEqual(totals["runs"], 2)
        self.assertEqual(totals["ok"], 1)
        self.assertEqual(totals["failed"], 1)


class LockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sandbox = Sandbox()
        self.sandbox.activate()

    def tearDown(self) -> None:
        self.sandbox.deactivate()

    def test_second_acquisition_without_wait_fails(self) -> None:
        first = FileLock("demo.task")
        second = FileLock("demo.task")
        first.acquire()
        try:
            with self.assertRaises(LockBusy):
                second.acquire()
        finally:
            first.release()

    def test_lock_is_reusable_after_release(self) -> None:
        first = FileLock("demo.task")
        first.acquire()
        first.release()
        second = FileLock("demo.task")
        second.acquire()
        second.release()

    def test_lock_file_records_holder(self) -> None:
        lock = FileLock("demo.task")
        lock.acquire()
        try:
            self.assertIn(str(os.getpid()), lock.holder)
        finally:
            lock.release()
        self.assertTrue(paths.locks_dir().is_dir())


if __name__ == "__main__":
    unittest.main()
