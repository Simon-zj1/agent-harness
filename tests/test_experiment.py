from __future__ import annotations

import json
import unittest

from harness.experiment import load as load_experiment
from harness.experiment import run_experiment

from .helpers import BODY_STEP, Sandbox, demo_task_body


class ExperimentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sandbox = Sandbox()
        self.sandbox.activate()
        self.work = self.sandbox.root / "work"
        self.work.mkdir(parents=True, exist_ok=True)
        self.sandbox.write_task("demo", demo_task_body(self.work))
        self.sandbox.step(
            "demo", "first", "from harness import stepctx\nraise SystemExit(stepctx.finish(stepctx.load(), 'ok'))\n"
        )
        self.sandbox.step(
            "demo", "second", BODY_STEP.replace("{CONTENT!r}", repr({"ok": True}))
        )
        self.experiments = self.sandbox.root / "experiments"
        self.experiments.mkdir(parents=True, exist_ok=True)
        (self.experiments / "demo-compare.toml").write_text(
            """
name = "demo-compare"
task = "demo"

[[arms]]
name = "offline-arm"
compose_mode = "replay"

[[arms]]
name = "paid-arm"
compose_mode = "llm"
requires = ["llm"]
""",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.sandbox.deactivate()

    def test_offline_arm_runs_and_paid_arm_is_skipped_by_default(self) -> None:
        experiment = load_experiment(self.experiments / "demo-compare.toml")
        report = run_experiment(experiment, date="2026-09-22", allow_llm=False)
        self.assertEqual(len(report["arms"]), 1)
        self.assertEqual(report["arms"][0]["arm"], "offline-arm")
        self.assertEqual(report["skipped"][0]["arm"], "paid-arm")
        self.assertTrue(report["report_md"].endswith("report.md"))

    def test_report_files_are_written(self) -> None:
        experiment = load_experiment(self.experiments / "demo-compare.toml")
        report = run_experiment(experiment, date="2026-09-22", allow_llm=False)
        from pathlib import Path

        md = Path(report["report_md"])
        csv = Path(report["report_csv"])
        payload = json.loads(Path(md.parent / "report.json").read_text(encoding="utf-8"))
        self.assertTrue(md.is_file())
        self.assertTrue(csv.is_file())
        self.assertIn("对比表", md.read_text(encoding="utf-8"))
        self.assertEqual(payload["date"], "2026-09-22")

    def test_experiments_skip_shared_state_steps_by_default(self) -> None:
        experiment = load_experiment(self.experiments / "demo-compare.toml")
        self.assertEqual(
            experiment.skipped_steps, ["sync-site", "fetch", "publish"]
        )
        report = run_experiment(experiment, date="2026-09-22", allow_llm=False)
        self.assertTrue(report["arms"])

    def test_arms_do_not_overwrite_canonical_content(self) -> None:
        experiment = load_experiment(self.experiments / "demo-compare.toml")
        run_experiment(experiment, date="2026-09-22", allow_llm=False)
        canonical = self.work / "content.json"
        self.assertFalse(canonical.exists())


if __name__ == "__main__":
    unittest.main()
