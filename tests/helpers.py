"""Test sandbox: relocate AGENT_HOME so tests never touch real state."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

TEST_CONFIG = """
[runtime]
default_provider = "replay"
default_executor = "builtin"
mutating_commands = ["git", "rm"]

[budget]
run_timeout_sec = 120
step_timeout_sec = 60
step_retries = 0

[notify]
enabled = true
macos = false
webhook = ""
on = "failure"

[providers.replay]
type = "replay"
model = "replay"

[providers.deepseek]
type = "openai_compat"
base_url = "https://api.deepseek.com/v1"
model = "deepseek-flash"
api_key_env = "DEEPSEEK_API_KEY"

[providers.local]
type = "openai_compat"
base_url = "http://127.0.0.1:11434/v1"
model = "qwen3:8b"
api_key_env = "LOCAL_API_KEY"

[executors.codex]
command = "codex"
sandbox = "workspace-write"
ephemeral = true

[executors.claude]
command = "claude"
permission_mode = "acceptEdits"

[tools.shell]
allowlist = ["python3", "git", "ls", "mkdir"]
"""

OK_STEP = """
import json
from harness import stepctx

def main():
    ctx = stepctx.load()
    out = ctx.run_dir / "artifact-{name}.json"
    out.write_text(json.dumps({"name": "{name}", "date": ctx.target_date}))
    return stepctx.finish(ctx, "ok", artifacts=[str(out)], metrics={"{name}": 1})

raise SystemExit(main())
"""

DEGRADE_STEP = """
from harness import stepctx

def main():
    ctx = stepctx.load()
    return stepctx.degrade(ctx, "source unavailable for {name}", metrics={"degraded_source": 1})

raise SystemExit(main())
"""

FAIL_STEP = """
from harness import stepctx

def main():
    ctx = stepctx.load()
    return stepctx.fail(ctx, "hard failure in {name}")

raise SystemExit(main())
"""

BODY_STEP = """
import json
from pathlib import Path
from harness import stepctx

def main():
    ctx = stepctx.load()
    (ctx.run_dir / "content.json").write_text(json.dumps({CONTENT!r}))
    return stepctx.finish(ctx, "ok")

raise SystemExit(main())
"""


class Sandbox:
    def __init__(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.home = self.root / "home"
        self.tasks = self.home / "tasks"
        self.tasks.mkdir(parents=True)
        self.home.mkdir(parents=True, exist_ok=True)
        self.config = self.root / "agent.toml"
        self.config.write_text(TEST_CONFIG, encoding="utf-8")
        self._previous: dict[str, str | None] = {}

    def activate(self) -> None:
        overrides = {
            "AGENT_HOME": str(self.home),
            "AGENT_TASKS_DIR": str(self.tasks),
            "AGENT_CONFIG": str(self.config),
        }
        for key, value in overrides.items():
            self._previous[key] = os.environ.get(key)
            os.environ[key] = value

    def deactivate(self) -> None:
        for key, value in self._previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._tmp.cleanup()

    # -- fixtures ----------------------------------------------------------
    def task_dir(self, name: str = "demo") -> Path:
        path = self.tasks / name
        (path / "steps").mkdir(parents=True, exist_ok=True)
        return path

    def step(self, task: str, name: str, code: str) -> None:
        target = self.tasks / task / "steps" / f"{name}.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(code, encoding="utf-8")

    def write_task(self, name: str, body: str) -> Path:
        path = self.task_dir(name)
        (path / "task.toml").write_text(body, encoding="utf-8")
        return path


DEMO_TASK = """
name = "demo"
title = "demo task"
date_mode = "today"
lock_scope = ["task"]
allowed_tools = ["fs_read", "fs_write", "fs_exists", "shell_run", "notify"]
readable_paths = ["{work_dir}"]
writable_paths = ["{work_dir}"]

[paths]
work_dir = "{work_dir}"

[publish]
default_enabled = false

[[steps]]
id = "first"
command = ["python3", "steps/first.py"]

[[steps]]
id = "second"
command = ["python3", "steps/second.py"]

[[validators]]
name = "json_parseable"
args = {{ path = "{{run_dir}}/content.json" }}
"""


def demo_task_body(work_dir: Path, *, extra_step: str = "") -> str:
    return DEMO_TASK.format(work_dir=work_dir) + extra_step
