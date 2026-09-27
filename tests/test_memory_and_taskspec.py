from __future__ import annotations

import unittest
from pathlib import Path

from harness import memory, taskspec
from harness.errors import TaskError

from .helpers import Sandbox, demo_task_body


class TaskSpecTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sandbox = Sandbox()
        self.sandbox.activate()
        self.work = self.sandbox.root / "work"
        self.work.mkdir(parents=True, exist_ok=True)
        self.sandbox.write_task("demo", demo_task_body(self.work))

    def tearDown(self) -> None:
        self.sandbox.deactivate()

    def test_task_is_discovered_and_parsed(self) -> None:
        self.assertIn("demo", taskspec.available())
        task = taskspec.load("demo")
        self.assertEqual(task.name, "demo")
        self.assertEqual([s.id for s in task.steps], ["first", "second"])
        self.assertEqual(task.validators[0].name, "json_parseable")

    def test_paths_table_and_date_placeholders_resolve(self) -> None:
        task = taskspec.load("demo")
        vars_ = task.vars(date="2026-09-22")
        self.assertEqual(vars_["work_dir"], str(self.work))
        self.assertEqual(
            taskspec.render("{work_dir}/{date}.json", date="2026-09-22", extra=vars_),
            f"{self.work}/2026-09-22.json",
        )

    def test_env_placeholders_keep_paths_portable(self) -> None:
        import os

        self.assertEqual(
            taskspec.render("${NOPE_DIR:-/tmp/fallback}", date="2026-09-22"),
            "/tmp/fallback",
        )
        os.environ["HARNESS_TEST_DIR"] = "/tmp/explicit"
        try:
            self.assertEqual(
                taskspec.render("${HARNESS_TEST_DIR:-/tmp/fallback}", date="2026-09-22"),
                "/tmp/explicit",
            )
            self.assertEqual(
                taskspec.render("${HARNESS_TEST_DIR}/data/{date}.json", date="2026-09-22"),
                "/tmp/explicit/data/2026-09-22.json",
            )
        finally:
            os.environ.pop("HARNESS_TEST_DIR", None)

    def test_unknown_task_raises(self) -> None:
        with self.assertRaises(Exception):
            taskspec.load("nope")

    def test_task_without_steps_is_rejected(self) -> None:
        self.sandbox.write_task("empty", 'name = "empty"\n')
        with self.assertRaises(Exception):
            taskspec.load("empty")

    def test_enabled_publish_step_requires_a_remote(self) -> None:
        body = (
            demo_task_body(self.work)
            .replace("[publish]\ndefault_enabled = false", "[publish]\ndefault_enabled = true")
            + '\n[[steps]]\nid = "publish"\ncommand = ["python3", "steps/publish.py"]\n'
            "when = \"validation_passed\"\nrequires_publish = true\n"
        )
        self.sandbox.write_task("publish-no-remote", body)
        with self.assertRaises(TaskError):
            taskspec.load("publish-no-remote")


class MemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sandbox = Sandbox()
        self.sandbox.activate()

    def tearDown(self) -> None:
        self.sandbox.deactivate()

    def test_run_summary_is_written_and_indexed(self) -> None:
        path = memory.write_run_summary(
            run_id="2026-09-22-demo-230000-ab12cd",
            task="demo",
            target_date="2026-09-22",
            status="success",
            summary="一切正常",
            metrics={"tokens_in": 100, "tokens_out": 50},
            artifacts=["/tmp/artifact.json"],
            degradations=["代理不可用"],
        )
        self.assertTrue(path.is_file())
        entries = memory.search("一切正常", limit=5)
        self.assertTrue(entries)
        self.assertEqual(entries[0]["kind"], "runs")

    def test_notes_are_separate_from_run_facts(self) -> None:
        memory.add_note("harness 取舍", "少即是多", tags=["harness"])
        self.assertTrue(list((self.sandbox.home / "memory" / "notes").glob("*.md")))
        stats = memory.stats()
        self.assertEqual(stats["by_kind"].get("notes"), 1)
        self.assertEqual(stats["by_kind"].get("runs", 0), 0)

    def test_front_matter_is_parsed(self) -> None:
        memory.add_note("记忆格式", "正文内容", tags=["a", "b"])
        entry = next(memory.iter_entries())
        self.assertEqual(entry.title, "记忆格式")
        self.assertEqual(entry.tags, ["a", "b"])
        self.assertIn("正文内容", entry.body)

    def test_reindex_reports_entry_count(self) -> None:
        memory.add_note("标题", "正文")
        self.assertEqual(memory.reindex(), 1)

    def test_promote_requires_a_known_run(self) -> None:
        from harness.ledger import Ledger

        memory.add_note("占位", "内容")
        ledger = Ledger()
        try:
            with self.assertRaises(Exception):
                memory.promote_run("unknown-run", ledger=ledger, title="x")
        finally:
            ledger.close()


if __name__ == "__main__":
    unittest.main()
