"""Adversarial probes for any validator, not just the URL gate.

`verification_eval` measures one validator by generating a corpus from real
captures. Some invariants have no corpus to generate from — the refund gate's
rule is "an unverifiable precondition must never be approved", and the way to
test that is to hand it payloads that break it.

A probe is a payload plus what the validator is supposed to say about it. The
interesting probes are the ones that *look* fine: a well-formed decisions file
where one entry quietly approves something it could not check.
"""

from __future__ import annotations

import datetime as dt
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import validators as validators_mod
from .decisions import Decision
from .refund_guard import REQUIRED_CHECKS

EXPECT_PASS = "pass"
EXPECT_NOT_PASS = "not_pass"
REFUND_FIXTURES = (
    Path(__file__).resolve().parent.parent / "tasks" / "refund-guard" / "fixtures"
)
ORDERS_PATH = REFUND_FIXTURES / "orders.json"
EXPECTED_PATH = REFUND_FIXTURES / "expected_decisions.json"


@dataclass
class Probe:
    probe_id: str
    validator: str
    payload: Any
    expect: str
    note: str = ""
    # Extra keyword arguments for the validator, besides the payload path.
    kwargs: dict[str, Any] = field(default_factory=dict)


def _refund_entry(
    order_id: str = "ord_1",
    decision: str = "pass",
    action: str = "approve",
    **extra: Any,
) -> dict[str, Any]:
    failed = set(extra.get("failed_checks") or [])
    unverified = set(extra.get("unverified_checks") or [])
    if decision == Decision.CANNOT_VERIFY.value and not unverified:
        unverified = {"payment_reference"}
    if decision == Decision.FAIL.value and not failed:
        failed = {"refund_window"}
    checks = []
    for name in REQUIRED_CHECKS:
        if name in failed:
            check_decision = Decision.FAIL.value
        elif name in unverified:
            check_decision = Decision.CANNOT_VERIFY.value
        else:
            check_decision = Decision.PASS.value
        checks.append(
            {"check": name, "decision": check_decision, "detail": "probe stub"}
        )
    entry = {
        "order_id": order_id,
        "decision": decision,
        "action": action,
        "checks": checks,
        "failed_checks": sorted(failed),
        "unverified_checks": sorted(unverified),
    }
    entry.update(extra)
    return entry


def refund_probes() -> list[Probe]:
    """Probes for `refund_decisions_fail_closed`.

    Payloads that violate the invariant are marked not_pass; payloads that are
    legitimately fine — including a *denied* refund — must pass, or the gate is
    just rejecting everything.
    """
    return [
        Probe(
            "all-checks-pass",
            "refund_decisions_fail_closed",
            {"results": [_refund_entry()]},
            EXPECT_PASS,
            "every precondition verified: approve",
        ),
        Probe(
            "policy-violation-denied",
            "refund_decisions_fail_closed",
            {"results": [_refund_entry(decision="fail", action="deny", failed_checks=["refund_window"])]},
            EXPECT_PASS,
            "a rule failed and the agent denied: correct",
        ),
        Probe(
            "unverifiable-escalated",
            "refund_decisions_fail_closed",
            {"results": [_refund_entry(decision="cannot_verify", action="escalate")]},
            EXPECT_PASS,
            "escalating an unverifiable refund is allowed",
        ),
        Probe(
            "unverifiable-approved",
            "refund_decisions_fail_closed",
            {
                "results": [
                    _refund_entry(
                        decision="cannot_verify",
                        action="approve",
                        unverified_checks=["payment_reference"],
                    )
                ]
            },
            EXPECT_NOT_PASS,
            "the invariant this validator exists for: unknown must not approve",
        ),
        Probe(
            "policy-violation-approved",
            "refund_decisions_fail_closed",
            {"results": [_refund_entry(decision="fail", action="approve", failed_checks=["refund_window"])]},
            EXPECT_NOT_PASS,
            "a failed rule was approved anyway",
        ),
        Probe(
            "pass-but-denied",
            "refund_decisions_fail_closed",
            {"results": [_refund_entry(decision="pass", action="deny")]},
            EXPECT_NOT_PASS,
            "everything verified yet denied: the record contradicts itself",
        ),
        Probe(
            "unknown-decision-word",
            "refund_decisions_fail_closed",
            {"results": [_refund_entry(decision="probably_fine", action="approve")]},
            EXPECT_NOT_PASS,
            "a decision outside the vocabulary must not be silently accepted",
        ),
        Probe(
            "unknown-action-word",
            "refund_decisions_fail_closed",
            {"results": [_refund_entry(decision="pass", action="refund_now")]},
            EXPECT_NOT_PASS,
            "an action outside the vocabulary must not be silently accepted",
        ),
        Probe(
            "empty-results",
            "refund_decisions_fail_closed",
            {"results": []},
            EXPECT_NOT_PASS,
            "an empty file must not read as 'nothing violated'",
        ),
        Probe(
            "one-bad-entry-among-many",
            "refund_decisions_fail_closed",
            {
                "results": [
                    _refund_entry("ord_1"),
                    _refund_entry("ord_2", decision="cannot_verify", action="deny"),
                    _refund_entry("ord_3", decision="cannot_verify", action="approve"),
                ]
            },
            EXPECT_NOT_PASS,
            "a single bad entry must not be averaged away by good neighbours",
        ),
        Probe(
            "refund-oracle-ground-truth",
            "refund_decisions_fail_closed",
            {
                "results": [
                    _refund_entry("ord_1001", "pass", "approve"),
                    _refund_entry("ord_1002", "fail", "deny", failed_checks=["refund_window"]),
                    _refund_entry(
                        "ord_1003",
                        "cannot_verify",
                        "deny",
                        unverified_checks=["payment_reference"],
                    ),
                    _refund_entry("ord_1004", "fail", "deny", failed_checks=["order_status"]),
                    _refund_entry(
                        "ord_1005",
                        "cannot_verify",
                        "deny",
                        unverified_checks=["payment_reference", "currency"],
                    ),
                ]
            },
            EXPECT_PASS,
            "the artifact matches both the recomputed decisions and the independent expected fixture",
            kwargs={"orders_path": str(ORDERS_PATH), "expected_path": str(EXPECTED_PATH)},
        ),
        Probe(
            "refund-oracle-mismatch",
            "refund_decisions_fail_closed",
            {
                "results": [
                    _refund_entry("ord_1001", "pass", "approve"),
                    _refund_entry("ord_1002", "pass", "approve"),
                    _refund_entry(
                        "ord_1003",
                        "cannot_verify",
                        "deny",
                        unverified_checks=["payment_reference"],
                    ),
                    _refund_entry("ord_1004", "fail", "deny", failed_checks=["order_status"]),
                    _refund_entry(
                        "ord_1005",
                        "cannot_verify",
                        "deny",
                        unverified_checks=["payment_reference", "currency"],
                    ),
                ]
            },
            EXPECT_NOT_PASS,
            "a self-consistent artifact must still fail against the independent expected fixture",
            kwargs={"orders_path": str(ORDERS_PATH), "expected_path": str(EXPECTED_PATH)},
        ),
    ]


def _merge_review(*decisions: tuple[str, str]) -> dict[str, Any]:
    """Build a review artifact from (check name, decision) pairs."""
    checks = [
        {"check": name, "decision": decision, "detail": f"stub {decision}"}
        for name, decision in decisions
    ]
    return {
        "range": "HEAD~1..HEAD",
        "scope": ["harness/"],
        "checks": checks,
        "merge": "allow"
        if all(decision == "pass" for _, decision in decisions)
        else "block",
    }


def merge_probes() -> list[Probe]:
    """Probes for `pr_merge_gate`.

    The invariant under test is the one that separates this from "CI is green":
    a judgement that could not be made must not become an auto-merge.
    """
    return [
        Probe(
            "merge-all-three-pass",
            "pr_merge_gate",
            _merge_review(
                ("NO_UNINTENDED_SCOPE", "pass"),
                ("ARCHITECTURE_OK", "pass"),
                ("TESTS_PASS", "pass"),
            ),
            EXPECT_PASS,
            "every judgement decided, all green: merge",
        ),
        Probe(
            "merge-scope-undecidable",
            "pr_merge_gate",
            _merge_review(
                ("NO_UNINTENDED_SCOPE", "cannot_verify"),
                ("ARCHITECTURE_OK", "pass"),
                ("TESTS_PASS", "pass"),
            ),
            EXPECT_NOT_PASS,
            "the case a green CI bar cannot see: tests pass, blast radius unknown",
        ),
        Probe(
            "merge-architecture-violated",
            "pr_merge_gate",
            _merge_review(
                ("NO_UNINTENDED_SCOPE", "pass"),
                ("ARCHITECTURE_OK", "fail"),
                ("TESTS_PASS", "pass"),
            ),
            EXPECT_NOT_PASS,
            "a layering rule was crossed",
        ),
        Probe(
            "merge-tests-uncertain",
            "pr_merge_gate",
            _merge_review(
                ("NO_UNINTENDED_SCOPE", "pass"),
                ("ARCHITECTURE_OK", "pass"),
                ("TESTS_PASS", "cannot_verify"),
            ),
            EXPECT_NOT_PASS,
            "zero tests discovered is not a green bar",
        ),
        Probe(
            "merge-abstained",
            "pr_merge_gate",
            _merge_review(
                ("NO_UNINTENDED_SCOPE", "pass"),
                ("ARCHITECTURE_OK", "abstain"),
                ("TESTS_PASS", "pass"),
            ),
            EXPECT_NOT_PASS,
            "abstaining is treated as blocking, not as consent",
        ),
        Probe(
            "merge-unknown-decision-word",
            "pr_merge_gate",
            _merge_review(
                ("NO_UNINTENDED_SCOPE", "pass"),
                ("ARCHITECTURE_OK", "probably_fine"),
                ("TESTS_PASS", "pass"),
            ),
            EXPECT_NOT_PASS,
            "a decision outside the vocabulary must not merge",
        ),
        Probe(
            "merge-no-checks-at-all",
            "pr_merge_gate",
            {"range": "HEAD~1..HEAD", "scope": [], "checks": [], "merge": "allow"},
            EXPECT_NOT_PASS,
            "an empty review must not read as 'nothing objected'",
        ),
        Probe(
            "merge-missing-required-check",
            "pr_merge_gate",
            _merge_review(
                ("NO_UNINTENDED_SCOPE", "pass"),
                ("ARCHITECTURE_OK", "pass"),
            ),
            EXPECT_NOT_PASS,
            "a missing required judgement must not be silently treated as consent",
        ),
        Probe(
            "merge-artifact-contradicts-itself",
            "pr_merge_gate",
            {
                "range": "HEAD~1..HEAD",
                "scope": ["harness/"],
                "checks": [
                    {"check": "TESTS_PASS", "decision": "fail", "detail": "stub"}
                ],
                "merge": "allow",
            },
            EXPECT_NOT_PASS,
            "the artifact claims allow while a check blocks",
        ),
    ]


def probe_registry() -> dict[str, Callable[[], list[Probe]]]:
    return {"refund": refund_probes, "merge": merge_probes}


def all_probes(name: str | None = None) -> list[Probe]:
    registry = probe_registry()
    if name and name not in registry:
        raise KeyError(f"unknown probe group {name!r}; known: {sorted(registry)}")
    groups = [name] if name else sorted(registry)
    probes: list[Probe] = []
    for group in groups:
        probes.extend(registry[group]())
    return probes


def run_probes(probes: list[Probe]) -> dict[str, Any]:
    """Run each probe and report whether the validator said what it should."""
    outcomes: list[dict[str, Any]] = []
    for probe in probes:
        func = validators_mod.get(probe.validator)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "payload.json"
            path.write_text(
                json.dumps(probe.payload, ensure_ascii=False), encoding="utf-8"
            )
            try:
                result = func(path, **probe.kwargs)
            except Exception as exc:  # noqa: BLE001 - a crashing validator is a result
                result = {
                    "ok": False,
                    "decision": None,
                    "detail": f"raised {type(exc).__name__}: {exc}",
                }
        blocked = not bool(result.get("ok"))
        expected_block = probe.expect == EXPECT_NOT_PASS
        outcomes.append(
            {
                "probe_id": probe.probe_id,
                "validator": probe.validator,
                "expect": probe.expect,
                "blocked": blocked,
                "decision": result.get("decision"),
                "detail": result.get("detail"),
                "correct": blocked == expected_block,
                "note": probe.note,
            }
        )

    missed = [o for o in outcomes if o["expect"] == EXPECT_NOT_PASS and not o["blocked"]]
    overblocked = [o for o in outcomes if o["expect"] == EXPECT_PASS and o["blocked"]]
    return {
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "probes": len(outcomes),
        "missed": len(missed),
        "overblocked": len(overblocked),
        "ok": not missed and not overblocked,
        "outcomes": outcomes,
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 校验器对抗探针",
        "",
        f"- 探针数：{report['probes']}",
        f"- 漏判（该拦没拦）：{report['missed']}",
        f"- 误拦（不该拦却拦）：{report['overblocked']}",
        f"- 生成时间：{report['generated_at']}",
        "",
        "| 探针 | 校验器 | 期望 | 实际 | 判决 | 结论 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for outcome in report["outcomes"]:
        lines.append(
            f"| {outcome['probe_id']} | {outcome['validator']} | "
            f"{outcome['expect']} | {'拦下' if outcome['blocked'] else '放行'} | "
            f"{outcome['decision'] or '—'} | {'✅' if outcome['correct'] else '❌'} |"
        )
    lines += [
        "",
        "## 口径",
        "",
        "- 探针是「一份载荷 + 校验器应该怎么说」。`not_pass` 表示该载荷必须被拦下。",
        "- `误拦` 与 `漏判` 同等重要：一个只会拒绝的校验器也会拿到 0 漏判。",
        "- 这些不变式没有可生成的语料（不像 URL 可以造前缀），所以用人工构造的边界载荷。",
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "Probe",
    "refund_probes",
    "probe_registry",
    "all_probes",
    "run_probes",
    "markdown",
]
