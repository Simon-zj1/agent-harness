from __future__ import annotations

import unittest

from harness import config as config_mod
from harness.loop import run_loop
from harness.providers import get as get_provider
from harness.registry import ToolContext
from harness.tools import build_registry

from .helpers import Sandbox


class ProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sandbox = Sandbox()
        self.sandbox.activate()
        self.config = config_mod.load(self.sandbox.config)

    def tearDown(self) -> None:
        self.sandbox.deactivate()

    def test_replay_provider_is_always_available(self) -> None:
        provider = get_provider(self.config, "replay")
        ok, note = provider.available()
        self.assertTrue(ok, note)

    def test_replay_provider_is_deterministic(self) -> None:
        provider = get_provider(self.config, "replay")
        first = provider.chat([{"role": "user", "content": "同样的输入"}])
        second = provider.chat([{"role": "user", "content": "同样的输入"}])
        self.assertEqual(first.text, second.text)

    def test_local_provider_reports_missing_server(self) -> None:
        provider = get_provider(self.config, "local")
        ok, note = provider.available()
        self.assertFalse(ok)
        self.assertTrue("no server listening" in note or "not set" in note)

    def test_deepseek_provider_requires_a_key(self) -> None:
        provider = get_provider(self.config, "deepseek")
        ok, note = provider.available()
        if not ok:
            self.assertIn("DEEPSEEK_API_KEY", note)

    def test_unknown_provider_is_rejected(self) -> None:
        with self.assertRaises(Exception):
            get_provider(self.config, "nope")


class LoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sandbox = Sandbox()
        self.sandbox.activate()
        self.config = config_mod.load(self.sandbox.config)
        self.work = self.sandbox.root / "work"
        self.work.mkdir(parents=True, exist_ok=True)
        self.run_dir = self.sandbox.root / "run"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ctx = ToolContext(
            run_id="loop-run",
            run_dir=self.run_dir,
            task_name="demo",
            target_date="2026-09-22",
            readable_paths=[self.work],
            writable_paths=[self.work],
            data={"task_dir": str(self.work)},
        )
        self.registry = build_registry(self.ctx, config=self.config)
        self.registry.allow(["fs_write", "fs_read", "fs_exists"])
        self.provider = get_provider(self.config, "replay")

    def tearDown(self) -> None:
        self.sandbox.deactivate()

    def test_loop_executes_a_tool_call_then_finishes(self) -> None:
        target = self.work / "loop.txt"
        prompt = (
            "写一个文件。\nTOOL_CALL: "
            '{"name": "fs_write", "arguments": {"path": "%s", "content": "from-loop"}}' % target
        )
        result = run_loop(
            self.provider,
            self.registry,
            system_prompt="你是测试 agent。",
            user_prompt=prompt,
            max_steps=3,
        )
        self.assertEqual(result.status, "done")
        self.assertEqual(len(result.tool_calls), 1)
        self.assertTrue(result.tool_calls[0]["ok"])
        self.assertEqual(target.read_text(), "from-loop")
        self.assertGreater(result.tokens_in, 0)

    def test_loop_respects_max_steps(self) -> None:
        target = self.work / "loop2.txt"
        prompt = (
            "TOOL_CALL: "
            '{"name": "fs_exists", "arguments": {"path": "%s"}}' % target
        )
        result = run_loop(
            self.provider,
            self.registry,
            system_prompt="s",
            user_prompt=prompt,
            max_steps=2,
        )
        self.assertIn(result.status, ("done", "max_steps"))
        self.assertLessEqual(result.steps, 2)

    def test_loop_reports_tool_errors_without_crashing(self) -> None:
        prompt = (
            "TOOL_CALL: "
            '{"name": "fs_read", "arguments": {"path": "/etc/hosts"}}'
        )
        result = run_loop(
            self.provider,
            self.registry,
            system_prompt="s",
            user_prompt=prompt,
            max_steps=2,
        )
        self.assertEqual(result.status, "done")
        self.assertFalse(result.tool_calls[0]["ok"])


if __name__ == "__main__":
    unittest.main()
