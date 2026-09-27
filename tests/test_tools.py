"""Permission model and tool auditing."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from harness import config as config_mod
from harness.errors import PermissionDenied, ToolError
from harness.ledger import Ledger
from harness.registry import ToolContext, _validate_args
from harness.tools import build_registry
from harness.tools import shell_tool

from .helpers import Sandbox


class ToolPermissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sandbox = Sandbox()
        self.sandbox.activate()
        self.config = config_mod.load(self.sandbox.config)
        self.writable = self.sandbox.root / "work"
        self.writable.mkdir(parents=True, exist_ok=True)
        self.run_dir = self.sandbox.root / "run"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = Ledger()
        self.ctx = ToolContext(
            run_id="run-1",
            run_dir=self.run_dir,
            task_name="demo",
            target_date="2026-09-22",
            dry_run=False,
            readable_paths=[self.writable],
            writable_paths=[self.writable],
            data={"task_dir": str(self.writable)},
        )
        self.registry = build_registry(self.ctx, config=self.config, ledger=self.ledger)

    def tearDown(self) -> None:
        self.ledger.close()
        self.sandbox.deactivate()

    def test_write_inside_declared_path_succeeds_and_is_audited(self) -> None:
        target = self.writable / "note.txt"
        result = self.registry.call("fs_write", {"path": str(target), "content": "hi"})
        self.assertTrue(result["written"])
        self.assertEqual(target.read_text(), "hi")
        self.assertEqual(self.ledger.totals()["runs"], 0)
        rows = self.ledger._conn.execute("SELECT tool, ok FROM tool_calls").fetchall()  # noqa: SLF001
        self.assertEqual(rows[0]["tool"], "fs_write")
        self.assertEqual(rows[0]["ok"], 1)

    def test_write_outside_declared_path_is_denied(self) -> None:
        with self.assertRaises(PermissionDenied):
            self.registry.call("fs_write", {"path": "/tmp/elsewhere.txt", "content": "nope"})

    def test_read_outside_declared_path_is_denied(self) -> None:
        with self.assertRaises(PermissionDenied):
            self.registry.call("fs_read", {"path": "/etc/hosts"})

    def test_shell_allowlist_denies_unknown_program(self) -> None:
        with self.assertRaises(PermissionDenied):
            self.registry.call("shell_run", {"argv": ["whoami"]})

    def test_shell_runs_allowlisted_program_and_captures_output(self) -> None:
        result = self.registry.call("shell_run", {"argv": ["ls", str(self.writable)]})
        self.assertEqual(result["returncode"], 0)

    def test_dry_run_suppresses_mutating_commands(self) -> None:
        ctx = ToolContext(
            run_id="run-2",
            run_dir=self.run_dir,
            task_name="demo",
            target_date="2026-09-22",
            dry_run=True,
            readable_paths=[self.writable],
            writable_paths=[self.writable],
            data={"task_dir": str(self.writable)},
        )
        registry = build_registry(ctx, config=self.config, ledger=self.ledger)
        result = registry.call(
            "shell_run",
            {"argv": ["git", "add", "."], "cwd": str(self.writable), "writes": [str(self.writable)]},
        )
        self.assertFalse(result["executed"])
        self.assertTrue(result["dry_run"])

    def test_shell_write_requires_declared_paths(self) -> None:
        with self.assertRaises(PermissionDenied):
            self.registry.call(
                "shell_run",
                {"argv": ["python3", "-c", "pass"], "cwd": str(self.writable)},
            )

        with self.assertRaises(PermissionDenied):
            self.registry.call(
                "shell_run",
                {
                    "argv": ["python3", "-c", "pass"],
                    "cwd": str(self.writable),
                    "writes": ["/tmp/outside-declared-write"],
                },
            )

    def test_notify_tool_is_silent_in_dry_run(self) -> None:
        ctx = ToolContext(
            run_id="run-3",
            run_dir=self.run_dir,
            task_name="demo",
            target_date="2026-09-22",
            dry_run=True,
            writable_paths=[self.writable],
        )
        registry = build_registry(ctx, config=self.config)
        result = registry.call("notify", {"title": "t", "message": "m"})
        self.assertFalse(result["delivered"])

    def test_tool_not_in_allowlist_is_rejected(self) -> None:
        self.registry.allow(["fs_read"])
        with self.assertRaises(PermissionDenied):
            self.registry.call("shell_run", {"argv": ["ls"]})

    def test_unknown_tool_in_allowlist_is_rejected_at_setup(self) -> None:
        with self.assertRaises(ToolError):
            self.registry.allow(["no_such_tool"])

    def test_codex_executor_is_not_spawned_on_dry_run(self) -> None:
        ctx = ToolContext(
            run_id="run-4",
            run_dir=self.run_dir,
            task_name="demo",
            target_date="2026-09-22",
            dry_run=True,
            writable_paths=[self.writable],
        )
        registry = build_registry(ctx, config=self.config)
        result = registry.call(
            "codex_exec", {"prompt": "hello", "workdir": str(self.writable)}
        )
        self.assertFalse(result["executed"])
        self.assertIn("exec", result["argv"])

    def test_dry_run_does_not_require_the_executor_to_be_installed(self) -> None:
        """Dry-run is for machines that do not have the executor.

        The binary lookup used to run before the dry-run short-circuit, so
        `codex_exec` raised `executor binary not found` on any clean checkout -
        which is the one situation dry-run exists for. CI caught it.
        """
        self.config.executors["codex"].command = "definitely-not-installed-xyz"
        self.config.executors["claude"].command = "definitely-not-installed-xyz"
        ctx = ToolContext(
            run_id="run-dry",
            run_dir=self.run_dir,
            task_name="demo",
            target_date="2026-09-25",
            dry_run=True,
            writable_paths=[self.writable],
        )
        registry = build_registry(ctx, config=self.config)
        for tool in ("codex_exec", "claude_exec"):
            with self.subTest(tool=tool):
                result = registry.call(
                    tool, {"prompt": "hello", "workdir": str(self.writable)}
                )
                self.assertFalse(result["executed"])
                self.assertEqual(result["argv"][0], "definitely-not-installed-xyz")

    def test_a_real_run_still_refuses_a_missing_executor(self) -> None:
        """The check moved, it was not deleted."""
        self.config.executors["codex"].command = "definitely-not-installed-xyz"
        ctx = ToolContext(
            run_id="run-live",
            run_dir=self.run_dir,
            task_name="demo",
            target_date="2026-09-25",
            dry_run=False,
            writable_paths=[self.writable],
        )
        registry = build_registry(ctx, config=self.config)
        with self.assertRaises(ToolError):
            registry.call("codex_exec", {"prompt": "hello", "workdir": str(self.writable)})

    def test_freeform_urls_are_rejected_by_http_tool(self) -> None:
        with self.assertRaises(ToolError):
            self.registry.call("http_get", {"url": "file:///etc/passwd"})

    def test_argument_validation_rejects_wrong_types(self) -> None:
        schema = {"type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"]}
        with self.assertRaises(ToolError):
            _validate_args("demo", schema, {"count": "many"})
        with self.assertRaises(ToolError):
            _validate_args("demo", schema, {})


class ShellEnvironmentTests(unittest.TestCase):
    def test_secret_environment_is_not_forwarded_to_shell(self) -> None:
        old = os.environ.get("DEEPSEEK_API_KEY")
        os.environ["DEEPSEEK_API_KEY"] = "super-secret"
        try:
            cleaned = shell_tool._clean_env()
        finally:
            if old is None:
                os.environ.pop("DEEPSEEK_API_KEY", None)
            else:
                os.environ["DEEPSEEK_API_KEY"] = old
        self.assertNotIn("DEEPSEEK_API_KEY", cleaned)
        self.assertIn("PATH", cleaned)


class NotificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sandbox = Sandbox()
        self.sandbox.activate()
        self.config = config_mod.load(self.sandbox.config)
        self.config.notify.macos = False
        self.config.notify.webhook = ""
        self.run_dir = Path(self.sandbox.root / "run2")
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.sandbox.deactivate()

    def test_notification_is_attempted_when_enabled(self) -> None:
        ctx = ToolContext(
            run_id="run-5",
            run_dir=self.run_dir,
            task_name="demo",
            target_date="2026-09-22",
        )
        registry = build_registry(ctx, config=self.config)
        result = registry.call("notify", {"title": "t", "message": "m"})
        self.assertIn("delivered", result)
        self.assertFalse(result["delivered"]["macos"])


if __name__ == "__main__":
    unittest.main()
