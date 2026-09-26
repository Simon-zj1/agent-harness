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

EXPECT_PASS = "pass"
EXPECT_NOT_PASS = "not_pass"


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
    entry = {"order_id": order_id, "decision": decision, "action": action}
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
    ]


def probe_registry() -> dict[str, Callable[[], list[Probe]]]:
    return {"refund": refund_probes}


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
