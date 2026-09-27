#!/usr/bin/env python3
"""Three independent judgements for a change, none of them "CI is green".

`TESTS_PASS` answers "did the tests stay green". It does not answer the two
questions that actually decide whether a merge is safe:

  ARCHITECTURE_OK      did the change respect the layering rules this repo
                       declares in AGENTS.md, or did it quietly reach across?
  NO_UNINTENDED_SCOPE  did it touch only what it claimed to touch?

Each is answered pass / fail / cannot_verify. Anything that cannot be decided
blocks: a change whose blast radius is unknown is not a change you merge at 3am.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import sys
from pathlib import Path

from harness import stepctx

DEFAULT_RANGE = "HEAD~1..HEAD"


def _repo_root() -> Path:
    """The repository under review.

    `agent` exports AGENT_REPO_ROOT, so the harness never has to guess where it
    lives; the task's readable_paths use the same variable.
    """
    return Path(os.environ["AGENT_REPO_ROOT"]).expanduser().resolve()


def _git(ctx, *args: str) -> tuple[int, str, str]:
    result = ctx.registry.call(
        "shell_run",
        {
            "argv": ["git", *args],
            "cwd": str(_repo_root()),
            "timeout_sec": 120,
            "allow_failure": True,
        },
    )
    return (
        result["returncode"],
        result.get("stdout") or "",
        result.get("stderr") or "",
    )


def check_tests(ctx, repo: Path) -> dict:
    """TESTS_PASS - run the suite. Cannot run means cannot verify."""
    result = ctx.registry.call(
        "shell_run",
        {
            "argv": ["python3", "-m", "unittest", "discover", "-t", ".", "-s", "tests"],
            "cwd": str(repo),
            "timeout_sec": 900,
            "allow_failure": True,
            "writes": [str(ctx.run_dir)],
            "env": {"PYTHONDONTWRITEBYTECODE": "1"},
        },
    )
    output = (result.get("stderr") or "") + (result.get("stdout") or "")
    ran = re.search(r"Ran (\d+) tests?", output)
    if ran is None:
        return {
            "check": "TESTS_PASS",
            "decision": "cannot_verify",
            "detail": "test runner produced no summary; the suite may not have run",
            "evidence": output.strip()[-300:],
        }
    tests = int(ran.group(1))
    if tests == 0:
        return {
            "check": "TESTS_PASS",
            "decision": "cannot_verify",
            "detail": "discovered 0 tests; a green bar over nothing is not evidence",
            "evidence": "",
        }
    if result["returncode"] == 0:
        return {
            "check": "TESTS_PASS",
            "decision": "pass",
            "detail": f"{tests} tests passed",
            "evidence": "",
        }
    return {
        "check": "TESTS_PASS",
        "decision": "fail",
        "detail": f"{tests} tests ran, suite exited {result['returncode']}",
        "evidence": output.strip()[-500:],
    }


# Rules the repository states about itself in AGENTS.md. Each is mechanical, so
# the verdict is a lookup rather than an opinion.
_ARCH_RULES = (
    (
        "harness must not depend on tasks",
        "harness/**/*.py",
        re.compile(r"^\s*(?:from|import)\s+tasks\b", re.M),
    ),
    (
        "step scripts must not call subprocess directly",
        "tasks/*/steps/*.py",
        re.compile(r"^\s*import\s+subprocess|^\s*from\s+subprocess\s+import", re.M),
    ),
    (
        "tasks must not reach into harness.tools internals",
        "tasks/**/*.py",
        re.compile(r"^\s*from\s+harness\.tools\s+import|^\s*import\s+harness\.tools\b", re.M),
    ),
)


def check_architecture(ctx, repo: Path) -> dict:
    violations: list[str] = []
    inspected = 0
    for label, pattern, regex in _ARCH_RULES:
        for path in repo.glob(pattern):
            if "__pycache__" in path.parts:
                continue
            inspected += 1
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if regex.search(text):
                violations.append(f"{label}: {path.relative_to(repo)}")
    if inspected == 0:
        return {
            "check": "ARCHITECTURE_OK",
            "decision": "cannot_verify",
            "detail": "no source files matched the layering rules; nothing was checked",
            "evidence": "",
        }
    return {
        "check": "ARCHITECTURE_OK",
        "decision": "fail" if violations else "pass",
        "detail": (
            f"{len(violations)} layering violation(s) across {inspected} files"
            if violations
            else f"{len(_ARCH_RULES)} layering rules hold across {inspected} files"
        ),
        "evidence": "; ".join(violations[:5]),
    }


def check_scope(ctx, repo: Path, *, revision_range: str, scope: list[str]) -> dict:
    code, out, err = _git(ctx, "diff", "--name-only", revision_range)
    if code != 0:
        return {
            "check": "NO_UNINTENDED_SCOPE",
            "decision": "cannot_verify",
            "detail": f"cannot resolve {revision_range!r}, so the blast radius is unknown",
            "evidence": (err or out).strip()[-300:],
        }
    changed = [line.strip() for line in out.splitlines() if line.strip()]
    if not changed:
        return {
            "check": "NO_UNINTENDED_SCOPE",
            "decision": "cannot_verify",
            "detail": f"{revision_range!r} contains no changed files",
            "evidence": "",
        }
    if not scope:
        return {
            "check": "NO_UNINTENDED_SCOPE",
            "decision": "cannot_verify",
            "detail": "no scope was declared, so nothing can be called out of scope",
            "evidence": "",
        }
    outside = [
        path
        for path in changed
        if not any(
            root in (".", "./", "*")
            or path == root
            or path.startswith(root.rstrip("/") + "/")
            or fnmatch.fnmatch(path, root)
            for root in scope
        )
    ]
    return {
        "check": "NO_UNINTENDED_SCOPE",
        "decision": "fail" if outside else "pass",
        "detail": (
            f"{len(outside)} of {len(changed)} changed files fall outside "
            f"{scope}: {', '.join(outside[:4])}"
            if outside
            else f"all {len(changed)} changed files stay inside {scope}"
        ),
        "evidence": "; ".join(outside[:5]),
    }


def main() -> int:
    ctx = stepctx.load()
    repo = _repo_root()
    revision_range = os.environ.get("AGENT_PR_RANGE", DEFAULT_RANGE)
    raw_scope = os.environ.get("AGENT_PR_SCOPE", "")
    scope = [part.strip() for part in raw_scope.split(",") if part.strip()]

    checks = [
        check_scope(ctx, repo, revision_range=revision_range, scope=scope),
        check_architecture(ctx, repo),
        check_tests(ctx, repo),
    ]

    tally: dict[str, int] = {}
    for entry in checks:
        tally[entry["decision"]] = tally.get(entry["decision"], 0) + 1

    payload = {
        "range": revision_range,
        "scope": scope,
        "checks": checks,
        "tally": tally,
        # The whole point: anything undecided blocks.
        "merge": "allow" if tally.get("pass") == len(checks) else "block",
    }
    out_path = ctx.run_dir / "review.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        "review "
        + revision_range
        + ": "
        + ", ".join(f"{c['check']}={c['decision']}" for c in checks)
    )
    return stepctx.finish(
        ctx,
        "ok",
        artifacts=[str(out_path)],
        metrics={
            "merge": payload["merge"],
            **{f"check_{k}": v for k, v in tally.items()},
        },
        notes=f"merge={payload['merge']}",
    )


if __name__ == "__main__":
    sys.exit(main())
