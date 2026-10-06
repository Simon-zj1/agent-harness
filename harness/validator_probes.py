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
    #: 额外输入文件（文件名 -> JSON 内容）。覆盖度这类需要「两个文件」的校验器
    #: 以前没法做探针；`{tmp}` 会在 kwargs 里替换成临时目录。
    extra_files: dict[str, Any] = field(default_factory=dict)


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


def _article(*, sections=None, notes=("口径", "scope")) -> dict[str, Any]:
    def item(title: str) -> dict:
        return {
            "title": {"zh": title, "en": title},
            "summary": {"zh": "摘要", "en": "summary"},
            "comment": {"zh": "点评", "en": "comment"},
            "sources": [1],
        }

    return {
        "date": "2026-01-01",
        "title": {"zh": "标题", "en": "title"},
        "summary": {"zh": "总览", "en": "overview"},
        "notes": {"zh": notes[0], "en": notes[1]},
        "sections": sections
        or [
            {
                "id": "insights",
                "groups": [
                    {"id": "g", "title": {"zh": "组", "en": "Group"}, "items": [item("选中")]}
                ],
            },
            {"id": "github", "items": [item("repo")]},
        ],
        "references": [{"id": 1, "title": "src", "url": "https://example.com/a"}],
    }


def schema_probes() -> list[Probe]:
    """Probes for the article contract (`daily_trends_structure`).

    The invariant that matters most here is the one whose absence cost nine
    days: a shape the gate does not recognise must not be reported as passing.
    The probes below are the shapes that *look* like a valid article.
    """
    unknown_section = _article()
    unknown_section["sections"].append(
        {"id": "podcasts", "items": [{"title": {"zh": "播客", "en": "podcast"}, "prose": {"zh": "x", "en": "y"}, "sources": [1]}]}
    )
    unknown_variant = _article()
    unknown_variant["sections"][0]["groups"][0]["items"] = [
        {"title": {"zh": "标题", "en": "title"}, "blurb": {"zh": "x", "en": "y"}, "sources": [1]}
    ]
    over_cap = _article()
    over_cap["sections"][1]["items"] = [
        {
            "title": {"zh": f"repo{i}", "en": f"repo{i}"},
            "prose": {"zh": "x", "en": "y"},
            "sources": [1],
        }
        for i in range(11)
    ]
    missing_en = _article()
    missing_en["sections"][1]["items"][0]["summary"] = {"zh": "只有中文"}
    legacy = _article()
    legacy["sections"][1]["items"][0] = {
        "title": {"zh": "旧形状", "en": "legacy"},
        "fields": [
            {"label": {"zh": "摘要", "en": "Summary"}, "value": {"zh": "a", "en": "b"}}
        ],
        "sources": [1],
    }

    return [
        Probe(
            "schema-known-shape-passes",
            "daily_trends_structure",
            _article(),
            EXPECT_PASS,
            "the shape the producer actually emits must validate",
        ),
        Probe(
            "schema-legacy-fields-passes",
            "daily_trends_structure",
            legacy,
            EXPECT_PASS,
            "v2 (fields) is still a supported renderer shape",
        ),
        Probe(
            "schema-undeclared-section",
            "daily_trends_structure",
            unknown_section,
            EXPECT_NOT_PASS,
            "a new section the gate does not know is a contract change, not a pass",
        ),
        Probe(
            "schema-unknown-item-variant",
            "daily_trends_structure",
            unknown_variant,
            EXPECT_NOT_PASS,
            "an item body the renderer cannot render must not slip through",
        ),
        Probe(
            "schema-over-cap-section",
            "daily_trends_structure",
            over_cap,
            EXPECT_NOT_PASS,
            "11 repos is over the declared cap of 10",
        ),
        Probe(
            "schema-missing-english",
            "daily_trends_structure",
            missing_en,
            EXPECT_NOT_PASS,
            "a one-language item is not bilingual",
        ),
    ]


def duplicate_probes() -> list[Probe]:
    """Probes for `daily_trends_no_duplicates`."""
    pair = _article()
    pair["sections"][0]["groups"][0]["items"] = [
        {
            "title": {"zh": "RoboECC：边缘-云协同的机器人计算框架", "en": "RoboECC framework"},
            "prose": {"zh": "正文", "en": "body"},
            "sources": [1],
        },
        {
            "title": {"zh": "RoboECC：机器人边缘-云协同计算代码", "en": "RoboECC code"},
            "prose": {"zh": "正文", "en": "body"},
            "sources": [1],
        },
    ]
    cross = _article()
    cross["sections"][0]["groups"][0]["items"] = [
        {
            "title": {
                "zh": "世界模型代码开源潮：客体永久性官方实现与 WorldinWorld",
                "en": "World model open-source wave: object permanence and WorldinWorld",
            },
            "prose": {"zh": "正文", "en": "body"},
            "sources": [1],
        }
    ]
    cross["sections"][1]["items"] = [
        {
            "title": {
                "zh": "hokindeng/object-permanence — 世界模型客体永久性官方代码",
                "en": "hokindeng/object-permanence",
            },
            "prose": {"zh": "正文", "en": "body"},
            "sources": [1],
        }
    ]

    return [
        Probe(
            "dedup-same-section-repeat",
            "daily_trends_no_duplicates",
            pair,
            EXPECT_NOT_PASS,
            "the same story twice in one group is a hard duplicate",
        ),
        Probe(
            "dedup-cross-section-roundup",
            "daily_trends_no_duplicates",
            cross,
            EXPECT_PASS,
            "a roundup and one of the repos it cites is a judgement call, not an auto-drop",
        ),
    ]


def probe_registry() -> dict[str, Callable[[], list[Probe]]]:
    return {
        "refund": refund_probes,
        "merge": merge_probes,
        "schema": schema_probes,
        "dedup": duplicate_probes,
        "references": reference_probes,
        "coverage": coverage_probes,
    }


def coverage_probes() -> list[Probe]:
    """Probes for `daily_trends_coverage`.

    不变量：这是**告警级**闸门——它会把没写进去的高信号条目报出来，但不能因此
    变成拦截（严重度由 task.toml 的 required 决定，见 SeverityDeclarationTests）。
    """
    content = _article()
    raw = {
        "hn": [
            {"title": "Agent harness benchmark 发布", "url": "https://ex.com/agent", "points": 900},
            {"title": "Bob Cringely has died", "url": "https://ex.com/obit", "points": 800},
        ]
    }
    covered_content = _article()
    covered_content["references"] = [
        {"id": 1, "title": "cited", "url": "https://ex.com/agent"}
    ]

    return [
        Probe(
            "coverage-uncited-top-item-is-advisory",
            "daily_trends_coverage",
            content,
            EXPECT_PASS,
            "高信号条目没写进去只告警：报出来但不拦发布",
            kwargs={"raw_path": "{tmp}/raw.json"},
            extra_files={"raw.json": raw},
        ),
        Probe(
            "coverage-cited-top-item-is-quiet",
            "daily_trends_coverage",
            covered_content,
            EXPECT_PASS,
            "被引用的高信号条目不进漏报名单",
            kwargs={"raw_path": "{tmp}/raw.json"},
            extra_files={"raw.json": raw},
        ),
    ]


def reference_probes() -> list[Probe]:
    """Probes for `daily_trends_references`.

    These pin the severity calibration: a broken citation graph blocks, a
    cosmetic orphan does not. Getting this the other way round is what makes an
    operator reach for the bypass flag.
    """
    with_orphan = _article()
    with_orphan["references"] = [
        {"id": 1, "title": "cited", "url": "https://example.com/a"},
        {"id": 2, "title": "never cited", "url": "https://example.com/b"},
    ]
    dangling = _article()
    dangling["sections"][1]["items"][0] = {
        "title": {"zh": "repo", "en": "repo"},
        "prose": {"zh": "x", "en": "y"},
        "sources": [7],
    }
    non_http = _article()
    non_http["references"] = [{"id": 1, "title": "local file", "url": "file:///etc/passwd"}]

    return [
        Probe(
            "refs-orphan-is-a-warning",
            "daily_trends_references",
            with_orphan,
            EXPECT_PASS,
            "an unused reference row is cosmetic: report it, do not stop the publish",
        ),
        Probe(
            "refs-dangling-source-id",
            "daily_trends_references",
            dangling,
            EXPECT_NOT_PASS,
            "a cited id with no reference breaks the page",
        ),
        Probe(
            "refs-non-http-url",
            "daily_trends_references",
            non_http,
            EXPECT_NOT_PASS,
            "only http(s) sources are publishable",
        ),
    ]


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
            for filename, content in (probe.extra_files or {}).items():
                extra = Path(tmp) / filename
                extra.write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")
            kwargs = {
                key: (
                    value.replace("{tmp}/", str(Path(tmp)) + "/")
                    if isinstance(value, str) and "{tmp}" in value
                    else value
                )
                for key, value in (probe.kwargs or {}).items()
            }
            try:
                result = func(path, **kwargs)
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
