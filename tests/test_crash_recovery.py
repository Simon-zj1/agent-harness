"""Kill a run mid-flight, then prove a re-run recovers instead of duplicating work."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import unittest
from pathlib import Path

from .helpers import BODY_STEP, Sandbox, demo_task_body

SLOW_STEP = """
import time
from harness import stepctx

def main():
    ctx = stepctx.load()
    time.sleep({seconds})
    return stepctx.finish(ctx, "ok")

raise SystemExit(main())
"""

SLEEP_SECONDS = 6


class CrashRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sandbox = Sandbox()
        self.work = self.sandbox.root / "work"
        self.work.mkdir(parents=True, exist_ok=True)
        self.sandbox.write_task(
            "slow",
            demo_task_body(self.work).replace('name = "demo"', 'name = "slow"'),
        )
        self.sandbox.step(
            "slow", "first", SLOW_STEP.replace("{seconds}", str(SLEEP_SECONDS))
        )
        self.sandbox.step(
            "slow",
            "second",
            BODY_STEP.replace("{CONTENT!r}", repr({"ok": True})),
        )

    def tearDown(self) -> None:
        self.sandbox.deactivate()

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update(
            {
                "AGENT_HOME": str(self.sandbox.home),
                "AGENT_TASKS_DIR": str(self.sandbox.tasks),
                "AGENT_CONFIG": str(self.sandbox.config),
                "PYTHONPATH": str(Path(__file__).resolve().parent.parent),
            }
        )
        return env

    def _run_cli(self, *args: str, timeout: int = 120) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "harness.cli", *args],
            cwd=str(Path(__file__).resolve().parent.parent),
            env=self._env(),
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def test_killed_run_is_marked_crashed_and_rerun_succeeds(self) -> None:
        repo = Path(__file__).resolve().parent.parent
        proc = subprocess.Popen(
            [sys.executable, "-m", "harness.cli", "run", "slow", "--date", "2026-09-22", "--no-notify"],
            cwd=str(repo),
            env=self._env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        # Wait until the step actually started, then kill the whole process group.
        lock = self.sandbox.home / "runs" / "locks" / "slow.task.lock"
        deadline = time.time() + 20
        while time.time() < deadline and not lock.exists():
            time.sleep(0.2)
        time.sleep(1.5)
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=30)
        if proc.stdout is not None:
            proc.stdout.close()

        ledger_path = self.sandbox.home / "runs" / "runs.db"
        self.assertTrue(ledger_path.is_file())

        second = self._run_cli("run", "slow", "--date", "2026-09-22", "--no-notify")
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        self.assertIn("success", second.stdout)

        statuses = self._run_cli("runs", "--limit", "5")
        self.assertIn("crashed", statuses.stdout)
        self.assertIn("success", statuses.stdout)

        report = self._run_cli("report", "--json", self._latest_run_id())
        payload = json.loads(report.stdout)
        self.assertEqual(payload["status"], "success")
        self.assertFalse(payload["published"])
        self.assertEqual([step["status"] for step in payload["steps"]], ["ok", "ok"])
        self.assertTrue(all(result["ok"] for result in payload["validators"]))

    def test_killed_run_releases_its_lock(self) -> None:
        """A killed process must not leave the task permanently locked."""
        repo = Path(__file__).resolve().parent.parent
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "harness.cli",
                "run",
                "slow",
                "--date",
                "2026-09-23",
                "--no-notify",
            ],
            cwd=str(repo),
            env=self._env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        time.sleep(2.0)
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=30)
        if proc.stdout is not None:
            proc.stdout.close()

        rerun = self._run_cli(
            "run", "slow", "--date", "2026-09-23", "--no-notify", "--wait-lock", "10"
        )
        self.assertEqual(rerun.returncode, 0, rerun.stdout + rerun.stderr)
        self.assertNotIn("blocked", rerun.stdout)

    def _latest_run_id(self) -> str:
        runs = sorted((self.sandbox.home / "runs").glob("2026-09-22-slow-*"))
        self.assertTrue(runs, "no run directories found")
        return runs[-1].name


if __name__ == "__main__":
    unittest.main()
