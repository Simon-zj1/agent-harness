"""Harness configuration (config/agent.toml).

Secrets are never stored in this file: providers name the environment
variable that holds their key, and `.env` files are read as a fallback.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import paths
from .errors import ConfigError

_ENV_FILE_CANDIDATES = ("~/.codex/.env", "~/.config/agent/.env", ".env")


@dataclass
class ProviderConfig:
    name: str
    type: str = "openai_compat"
    base_url: str = ""
    model: str = ""
    api_key_env: str = ""
    timeout_sec: int = 180
    max_retries: int = 2
    enabled: bool = True
    temperature: float = 0.2
    max_output_tokens: int | None = None
    input_per_mtok: float | None = None
    output_per_mtok: float | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)


@dataclass
class ExecutorConfig:
    name: str
    command: str
    sandbox: str | None = None
    permission_mode: str | None = None
    ephemeral: bool = False
    model: str | None = None
    extra_args: list[str] = field(default_factory=list)
    timeout_sec: int = 3600


@dataclass
class BudgetConfig:
    run_timeout_sec: int = 5400
    step_timeout_sec: int = 1800
    step_retries: int = 1
    max_tokens: int | None = None
    monthly_limit_usd: float | None = None


@dataclass
class NotifyConfig:
    enabled: bool = True
    macos: bool = True
    webhook: str = ""
    on: str = "failure"


@dataclass
class AgentConfig:
    path: Path
    runtime: dict[str, Any] = field(default_factory=dict)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    notify: NotifyConfig = field(default_factory=NotifyConfig)
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    executors: dict[str, ExecutorConfig] = field(default_factory=dict)
    shell_allowlist: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def default_provider(self) -> str:
        return str(self.runtime.get("default_provider", "deepseek"))

    @property
    def default_executor(self) -> str:
        return str(self.runtime.get("default_executor", "codex"))

    def provider(self, name: str | None = None) -> ProviderConfig:
        key = name or self.default_provider
        if key not in self.providers:
            raise ConfigError(
                f"unknown provider {key!r}; configured: {sorted(self.providers)}"
            )
        return self.providers[key]

    def executor(self, name: str | None = None) -> ExecutorConfig:
        key = name or self.default_executor
        if key not in self.executors:
            raise ConfigError(
                f"unknown executor {key!r}; configured: {sorted(self.executors)}"
            )
        return self.executors[key]


def _env_file_values() -> dict[str, str]:
    values: dict[str, str] = {}
    for candidate in _ENV_FILE_CANDIDATES:
        path = Path(candidate).expanduser()
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and value and key not in values:
                values[key] = value
    return values


def secret(env_name: str) -> str | None:
    """Read a secret from the environment, then from known .env files."""
    if not env_name:
        return None
    value = os.environ.get(env_name)
    if value:
        return value
    return _env_file_values().get(env_name)


def secret_source(env_name: str) -> str | None:
    if not env_name:
        return None
    if os.environ.get(env_name):
        return "env"
    if env_name in _env_file_values():
        for candidate in _ENV_FILE_CANDIDATES:
            path = Path(candidate).expanduser()
            if path.is_file() and env_name in _env_file_values():
                return f"file:{path}"
    return None


def redact(value: str | None) -> str:
    if not value:
        return "<unset>"
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-2:]} (len={len(value)})"


DEFAULT_SHELL_ALLOWLIST = [
    "python3",
    "git",
    "curl",
    "ls",
    "cp",
    "mkdir",
    "rm",
    "touch",
    "cat",
    "sed",
    "grep",
    "osascript",
    "codex",
    "claude",
    "plutil",
    "launchctl",
    "date",
    "rsync",
]


def load(path: Path | None = None) -> AgentConfig:
    target = path or paths.config_path()
    if not target.is_file():
        return AgentConfig(path=target, shell_allowlist=list(DEFAULT_SHELL_ALLOWLIST))
    try:
        raw = tomllib.loads(target.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"cannot read config {target}: {exc}") from exc

    budget_raw = raw.get("budget", {})
    notify_raw = raw.get("notify", {})
    providers = {
        name: ProviderConfig(name=name, **body)
        for name, body in raw.get("providers", {}).items()
    }
    executors = {
        name: ExecutorConfig(name=name, **body)
        for name, body in raw.get("executors", {}).items()
    }
    shell_raw = raw.get("tools", {}).get("shell", {})
    allowlist = shell_raw.get("allowlist") or DEFAULT_SHELL_ALLOWLIST

    return AgentConfig(
        path=target,
        runtime=raw.get("runtime", {}),
        budget=BudgetConfig(**budget_raw),
        notify=NotifyConfig(**notify_raw),
        providers=providers,
        executors=executors,
        shell_allowlist=list(allowlist),
        raw=raw,
    )
