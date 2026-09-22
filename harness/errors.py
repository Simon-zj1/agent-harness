"""Error taxonomy shared by the kernel, tools and steps."""

from __future__ import annotations


class HarnessError(Exception):
    """Base class for every harness failure."""

    failure_class = "harness_error"

    def __init__(self, message: str, *, failure_class: str | None = None) -> None:
        super().__init__(message)
        if failure_class:
            self.failure_class = failure_class


class ConfigError(HarnessError):
    failure_class = "config_error"


class TaskError(HarnessError):
    failure_class = "task_error"


class LockBusy(HarnessError):
    failure_class = "lock_busy"


class ToolError(HarnessError):
    failure_class = "tool_error"


class PermissionDenied(ToolError):
    failure_class = "permission_denied"


class ProviderError(HarnessError):
    failure_class = "provider_error"


class StepFailed(HarnessError):
    failure_class = "step_failed"


class ValidationFailed(HarnessError):
    failure_class = "validation_failed"


class AlreadyDone(HarnessError):
    """Raised when an idempotency guard sees a successful run for the same task+date."""

    failure_class = "already_done"

    def __init__(self, message: str, run_id: str) -> None:
        super().__init__(message)
        self.run_id = run_id
