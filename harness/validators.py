"""Named validators: acceptance criteria live in code, not in a prompt.

Every validator answers with a typed decision (see `harness.decisions`), not a
boolean. The boolean is still returned as `ok` so old callers keep working —
but it is derived, and `CANNOT_VERIFY` is deliberately *not* ok.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .decisions import (
    Decision,
    DecisionPolicy,
    DecisionResult,
    Evidence,
    FailureClass,
    apply_policy,
    canonical_url,
    classify_url,
    combine,
)
from . import content_schema, dedup, refund_guard
from .errors import ValidationFailed

Validator = Callable[..., dict[str, Any]]
_REGISTRY: dict[str, Validator] = {}


def validator(name: str) -> Callable[[Validator], Validator]:
    def register(func: Validator) -> Validator:
        _REGISTRY[name] = func
        return func

    return register


def get(name: str) -> Validator:
    if name not in _REGISTRY:
        raise ValidationFailed(f"unknown validator {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def names() -> list[str]:
    return sorted(_REGISTRY)


def _result(
    name: str,
    ok: bool,
    detail: str,
    *,
    decision: Decision | None = None,
    failure_class: FailureClass | None = None,
    evidence_items: list[Evidence] | None = None,
    remediation: str | None = None,
    **rest: Any,
) -> dict[str, Any]:
    """Build a typed result, then let legacy keys (`failures`, `metrics`) win.

    Keeping the legacy keys authoritative means none of the existing validators
    or their tests had to change shape.
    """
    resolved = decision or (Decision.PASS if ok else Decision.FAIL)
    if resolved is not Decision.PASS and failure_class is None:
        failure_class = FailureClass.POLICY_VIOLATION
    result = DecisionResult(
        name=name,
        decision=resolved,
        detail=detail,
        failure_class=failure_class,
        evidence=evidence_items or [],
        metrics=rest.get("metrics", {}),
        remediation=remediation,
    ).to_dict()
    result.update(rest)
    return result


@validator("json_parseable")
def json_parseable(path: str | Path, *, name: str = "json_parseable") -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        return _result(name, False, f"missing file: {target}")
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return _result(name, False, f"invalid JSON: {exc}")
    return _result(name, True, f"parsed {target.name}", metrics={"top_level_keys": len(data)})


@validator("files_exist")
def files_exist(paths: list[str], *, name: str = "files_exist") -> dict[str, Any]:
    missing = [p for p in paths if not Path(p).exists()]
    return _result(
        name,
        not missing,
        f"{len(paths) - len(missing)}/{len(paths)} present",
        failures=[{"missing": p} for p in missing],
        metrics={"expected": len(paths), "missing": len(missing)},
    )


def load_content(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise ValidationFailed(f"missing content file: {target}")
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationFailed(f"content is not valid JSON: {exc}") from exc


@validator("daily_trends_structure")
def daily_trends_structure(
    path: str | Path,
    *,
    max_insights: int | None = None,
    max_repos: int | None = None,
    name: str = "daily_trends_structure",
) -> dict[str, Any]:
    """Check the article contract the published pages depend on.

    The contract itself lives in `harness/content_schema.py` and covers every
    body variant the renderer supports (prose / fields / summary+comment) plus
    the declared section set. An **undeclared shape is CANNOT_VERIFY, not
    PASS**: that is the fix for the nine days when the producer silently
    changed shape and this gate kept answering as if it understood the file.
    """
    try:
        content = load_content(path)
    except ValidationFailed as exc:
        return _result(name, False, str(exc))

    report = content_schema.check(content)
    metrics = dict(report.metrics)
    failures = list(report.failures)
    if max_insights is not None and metrics["insights"] > max_insights:
        failures.append(
            {"issue": f"insights has {metrics['insights']} items (max {max_insights})"}
        )
    if max_repos is not None and metrics["repos"] > max_repos:
        failures.append(
            {"issue": f"github has {metrics['repos']} items (max {max_repos})"}
        )

    if report.unknown_shape:
        return _result(
            name,
            False,
            "; ".join(report.reasons),
            decision=Decision.CANNOT_VERIFY,
            failure_class=FailureClass.UNKNOWN_VARIANT,
            remediation=(
                "The gate does not know this shape, so it cannot claim the article "
                "is fine. Either declare the new section/variant in "
                "harness/content_schema.py (this is a contract change and should be "
                "deliberate) or regenerate the article in a supported shape."
            ),
            failures=[{"issue": reason} for reason in report.reasons],
            metrics=metrics,
        )

    ok = not failures
    return _result(
        name,
        ok,
        "structure ok" if ok else f"{len(failures)} structural problem(s)",
        remediation=(
            "Every item needs a bilingual body in one of the supported variants "
            "(prose / fields / summary+comment), a bilingual title and at least "
            "one source."
        )
        if not ok
        else None,
        failures=failures[:40],
        metrics=metrics,
    )


def check_structure(
    content: dict[str, Any],
    *,
    max_insights: int | None = None,
    max_repos: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """In-memory version used by compose for its self-repair round."""
    report = content_schema.check(content)
    failures = list(report.failures)
    failures += [{"issue": reason} for reason in report.reasons]
    metrics = dict(report.metrics)
    if max_insights is not None and metrics["insights"] > max_insights:
        failures.append(
            {"issue": f"insights has {metrics['insights']} items (max {max_insights})"}
        )
    if max_repos is not None and metrics["repos"] > max_repos:
        failures.append(
            {"issue": f"github has {metrics['repos']} items (max {max_repos})"}
        )
    return failures, metrics


@validator("daily_trends_no_duplicates")
def daily_trends_no_duplicates(
    path: str | Path,
    *,
    borderline: str = "warn",
    name: str = "daily_trends_no_duplicates",
) -> dict[str, Any]:
    """The same story must not appear twice in one article.

    Clear duplicates (score >= 0.86) fail: they are a real editorial defect the
    strings can prove. Grey-band pairs (0.60-0.86) need a judgement, and this
    gate has no model, so by default they are *reported, not judged* — the
    count and examples go into metrics and the run detail. Pass
    `borderline = "cannot_verify"` to make unjudged pairs block instead.
    """
    try:
        content = load_content(path)
    except ValidationFailed as exc:
        return _result(name, False, str(exc))

    report = content_schema.check(content)
    view = report.view
    if view is None or report.unknown_shape:
        return _result(
            name,
            False,
            "cannot check duplicates: the article shape is not declared",
            decision=Decision.CANNOT_VERIFY,
            failure_class=FailureClass.UNKNOWN_VARIANT,
            metrics={"duplicate_pairs": 0, "borderline_pairs": 0},
        )

    candidates = dedup.build_candidates(
        view, reference_urls=dedup.reference_url_map(view.references)
    )
    plan = dedup.compare(candidates)
    outcome = dedup.resolve(plan)  # the gate never calls a model

    metrics = {
        "items": len(candidates),
        "duplicate_pairs": len(plan.duplicates),
        "borderline_pairs": len(plan.borderline),
        "max_pair_score": round(plan.pairs[0].score, 3) if plan.pairs else 0.0,
        "duplicates": [entry["location"] for entry in outcome.dropped],
    }
    if outcome.unjudged:
        metrics["unjudged_pairs"] = [pair["score"] for pair in outcome.unjudged]
        metrics["unjudged_examples"] = [
            {
                "score": pair["score"],
                "a": pair["a"]["title"],
                "b": pair["b"]["title"],
            }
            for pair in outcome.unjudged[:5]
        ]

    if outcome.dropped:
        return _result(
            name,
            False,
            f"{len(outcome.dropped)} item(s) duplicate an earlier item",
            decision=Decision.FAIL,
            failure_class=FailureClass.POLICY_VIOLATION,
            failures=[
                {
                    "issue": f"duplicate of {entry['kept']} (score {entry['score']})",
                    "title": entry["title"],
                    "reasons": entry["reasons"],
                }
                for entry in outcome.dropped
            ],
            remediation=(
                "Drop the duplicate in compose (dedup runs there by default) or "
                "merge the two items."
            ),
            metrics=metrics,
        )

    if plan.borderline and borderline == "cannot_verify":
        return _result(
            name,
            False,
            f"{len(plan.borderline)} grey-band pair(s) were never judged",
            decision=Decision.CANNOT_VERIFY,
            failure_class=FailureClass.PRECONDITION_UNKNOWN,
            metrics=metrics,
        )

    detail = "no duplicates"
    if plan.borderline:
        detail = (
            f"no clear duplicates; {len(plan.borderline)} grey-band pair(s) not "
            "judged (see metrics.unjudged_examples)"
        )
    return _result(name, True, detail, metrics=metrics)


@validator("daily_trends_references")
def daily_trends_references(
    path: str | Path, *, name: str = "daily_trends_references"
) -> dict[str, Any]:
    """Reference ids must resolve and urls must be http(s).

    Severity is calibrated to consequence. A source id with no reference, a
    duplicate id or a non-http url breaks the page and blocks. An *orphan*
    reference (declared, never cited) only leaves an unused row in the source
    list: it is reported as a warning, not as a reason to stop a daily publish.
    Blocking on cosmetics is how a gate teaches its operator to bypass it.
    """
    try:
        content = load_content(path)
    except ValidationFailed as exc:
        return _result(name, False, str(exc))

    references = content.get("references") or []
    ids = [ref.get("id") for ref in references]
    failures: list[dict[str, Any]] = []

    if len(set(ids)) != len(ids):
        failures.append({"issue": "duplicate reference ids"})
    id_set = {i for i in ids if isinstance(i, int)}
    for ref in references:
        url = str(ref.get("url", ""))
        if not url.startswith(("http://", "https://")):
            failures.append({"issue": "reference without http(s) url", "id": ref.get("id"), "url": url})
        if not str(ref.get("title", "")).strip():
            failures.append({"issue": "reference without title", "id": ref.get("id")})

    used: set[int] = set()
    for section in content.get("sections") or []:
        items = list(section.get("items") or [])
        for group in section.get("groups") or []:
            items.extend(group.get("items") or [])
        for item in items:
            for source in item.get("sources") or []:
                if not isinstance(source, int):
                    failures.append({"issue": "non-integer source id", "source": source})
                    continue
                used.add(source)
                if source not in id_set:
                    failures.append({"issue": "source id has no reference", "id": source})

    orphans = sorted(id_set - used)
    ok = not failures
    warnings = (
        [{"issue": "reference never cited", "ids": orphans}] if orphans else []
    )
    if ok and orphans:
        detail = f"references ok ({len(orphans)} orphan reference(s) reported)"
    else:
        detail = "references ok" if ok else f"{len(failures)} reference problem(s)"
    return _result(
        name,
        ok,
        detail,
        remediation=(
            "Every cited source id must resolve and every reference url must be "
            "http(s)."
        )
        if not ok
        else None,
        failures=failures[:40],
        warnings=warnings,
        metrics={
            "references": len(references),
            "cited": len(used),
            "orphans": len(orphans),
            "warnings": len(warnings),
        },
    )


@validator("daily_trends_verifiable")
def daily_trends_verifiable(
    content_path: str | Path,
    raw_path: str | Path,
    *,
    min_ratio: float = 1.0,
    name: str = "daily_trends_verifiable",
) -> dict[str, Any]:
    """Every cited reference must trace back to the raw capture — with receipts.

    This is the anti-fabrication gate. It used to answer a boolean, which meant a
    URL that merely *extended* a fetched URL (…/llm → …/llm-something-invented)
    was reported as verified. It now separates three cases:

        PASS            exact match, or an enumerated drift (scheme, www,
                        trailing slash, tracking params, arxiv version)
        CANNOT_VERIFY   the cited URL extends (or truncates) a fetched one, so we
                        know the prefix is real but not the remainder
        FAIL            nothing in the capture matches

    See `harness.decisions.classify_url` for the matcher itself.
    """
    try:
        content = load_content(content_path)
    except ValidationFailed as exc:
        return _result(name, False, str(exc))
    raw_file = Path(raw_path)
    if not raw_file.is_file():
        return _result(name, False, f"missing raw capture: {raw_file}")
    raw = json.loads(raw_file.read_text(encoding="utf-8"))

    raw_urls = {canonical_url(u) for u in _collect_urls(raw)}
    references = content.get("references") or []
    used_ids: set[int] = set()
    for section in content.get("sections") or []:
        items = list(section.get("items") or [])
        for group in section.get("groups") or []:
            items.extend(group.get("items") or [])
        for item in items:
            used_ids.update(s for s in (item.get("sources") or []) if isinstance(s, int))

    cited = [ref for ref in references if ref.get("id") in used_ids]
    verdicts: dict[int, Any] = {}
    evidence_items: list[Evidence] = []
    tally = {"pass": 0, "cannot_verify": 0, "fail": 0}
    stale: dict[int, list[str]] = {}
    other_days: dict[str, set[str]] | None = None
    for ref in cited:
        ref_id = ref.get("id")
        url = str(ref.get("url", ""))
        verdict = classify_url(url, raw_urls)
        verdicts[ref_id] = verdict
        tally[verdict.decision.value] = tally.get(verdict.decision.value, 0) + 1
        if verdict.decision is not Decision.PASS:
            detail = verdict.reason
            if verdict.decision is Decision.FAIL:
                # "matches nothing" and "matches yesterday's capture" are very
                # different problems: the first is a fabrication risk, the
                # second is a producer that reused stale material (2026-10-05
                # cited 8 arXiv papers from 10-04, when that day's arXiv fetch
                # had returned nothing). Name the day instead of making the
                # reader guess which one it is.
                if other_days is None:
                    other_days = _other_capture_index(raw_file)
                days = sorted(
                    day for day, urls in other_days.items() if canonical_url(url) in urls
                )
                if days:
                    stale[ref_id] = days
                    detail = (
                        "cited url is not in this day's capture but is in the capture "
                        f"of {', '.join(days)}"
                    )
                else:
                    detail = "cited url is in no capture at all"
            evidence_items.append(
                Evidence(
                    ref=f"ref#{ref_id}",
                    detail=detail,
                    url=url,
                )
            )

    traced = tally["pass"]
    ratio = 1.0 if not cited else traced / len(cited)
    if tally["fail"]:
        decision = Decision.FAIL
        failure_class = FailureClass.FABRICATED_SOURCE
    elif tally["cannot_verify"]:
        decision = Decision.CANNOT_VERIFY
        failure_class = FailureClass.UNKNOWN_VARIANT
    else:
        decision = Decision.PASS
        failure_class = None

    ok = bool(cited) and decision is Decision.PASS and ratio >= min_ratio
    if not cited:
        decision = Decision.FAIL
        failure_class = FailureClass.MALFORMED
        detail = "no cited references to verify"
    else:
        detail = (
            f"{traced}/{len(cited)} cited references trace back to the raw capture"
            f" (cannot_verify={tally['cannot_verify']}, fail={tally['fail']})"
        )

    return _result(
        name,
        ok,
        detail,
        decision=decision,
        failure_class=failure_class,
        evidence_items=evidence_items[:40],
        remediation=(
            "Re-fetch the cited page, or replace the citation with a URL that is "
            "present in the raw capture."
        )
        if decision is not Decision.PASS
        else None,
        failures=[
            {"issue": item.detail, "ref": item.ref, "url": item.url}
            for item in evidence_items[:40]
        ],
        metrics={
            "cited": len(cited),
            "traced": traced,
            "verifiable_ratio": round(ratio, 4),
            "cannot_verify": tally["cannot_verify"],
            "fail": tally["fail"],
            "stale_citations": len(stale),
            "stale_from_days": sorted({day for days in stale.values() for day in days}),
        },
    )


def _other_capture_index(raw_path: Path, *, limit: int = 30) -> dict[str, set[str]]:
    """Map day -> canonical urls for neighbouring captures.

    Built lazily (only when a citation fails today's capture) and limited to the
    most recent `limit` days, so a long history does not make every validation
    read every file.
    """
    raw_dir = raw_path.parent
    if not raw_dir.is_dir():
        return {}
    today = raw_path.stem
    days = [
        path.stem
        for path in sorted(raw_dir.glob("20*.json"), reverse=True)
        if path.stem != today
    ][:limit]
    index: dict[str, set[str]] = {}
    for day in days:
        try:
            payload = json.loads((raw_dir / f"{day}.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        index[day] = {canonical_url(u) for u in _collect_urls(payload)}
    return index


@validator("refund_decisions_fail_closed")
def refund_decisions_fail_closed(
    path: str | Path,
    *,
    orders_path: str | Path | None = None,
    expected_path: str | Path | None = None,
    name: str = "refund_decisions_fail_closed",
) -> dict[str, Any]:
    """High-stakes variant: an unverifiable refund must never be approved.

    This is the same kernel applied to a decision that moves money. The only
    difference from the publishing gate is the policy: here CANNOT_VERIFY is not
    merely "block the step", it is *the decision* — the answer is deny.

    Invariants checked:
      decision=pass            -> action=approve
      decision=fail            -> action=deny      (a rule was violated)
      decision=cannot_verify   -> action must NOT be approve (fail-closed)
    """
    try:
        payload = load_content(path)
    except ValidationFailed as exc:
        return _result(name, False, str(exc), failure_class=FailureClass.MALFORMED)

    results = payload.get("results") or []
    if not results:
        return _result(
            name,
            False,
            "no decisions to check",
            failure_class=FailureClass.MALFORMED,
        )

    valid_decisions = {
        Decision.PASS.value,
        Decision.FAIL.value,
        Decision.CANNOT_VERIFY.value,
    }
    valid_actions = {"approve", "deny", "escalate"}
    violations: list[Evidence] = []
    tally: dict[str, int] = {}
    expected_by_id: dict[str, dict[str, Any]] | None = None
    if orders_path is not None:
        try:
            book = load_content(orders_path)
        except ValidationFailed as exc:
            return _result(name, False, str(exc), failure_class=FailureClass.MALFORMED)
        ledger = set(book.get("payment_ledger") or [])
        expected_by_id = {
            str(order.get("order_id")): refund_guard.decide(
                order, book.get("policy") or {}, ledger
            )
            for order in (book.get("orders") or [])
        }
    oracle_by_id: dict[str, dict[str, Any]] | None = None
    if expected_path is not None:
        try:
            oracle = load_content(expected_path)
        except ValidationFailed as exc:
            return _result(name, False, str(exc), failure_class=FailureClass.MALFORMED)
        oracle_by_id = {
            str(order_id): value
            for order_id, value in (oracle.get("orders") or {}).items()
        }
    seen_order_ids: set[str] = set()
    seen_oracle_ids: set[str] = set()

    for entry in results:
        order_id = str(entry.get("order_id", "?"))
        decision = str(entry.get("decision", ""))
        action = str(entry.get("action", ""))
        tally[decision] = tally.get(decision, 0) + 1
        if expected_by_id is not None:
            expected = expected_by_id.get(order_id)
            if expected is None:
                violations.append(
                    Evidence(
                        ref=order_id,
                        detail="decision for an order that is not in the ground-truth file",
                    )
                )
            else:
                seen_order_ids.add(order_id)
                if decision != expected["decision"]:
                    violations.append(
                        Evidence(
                            ref=order_id,
                            detail=(
                                f"decision mismatch: artifact says {decision!r}, "
                                f"ground truth says {expected['decision']!r}"
                            ),
                        )
                    )
                if action != expected["action"]:
                    violations.append(
                        Evidence(
                            ref=order_id,
                            detail=(
                                f"action mismatch: artifact says {action!r}, "
                                f"ground truth says {expected['action']!r}"
                            ),
                        )
                    )
        if oracle_by_id is not None:
            oracle_entry = oracle_by_id.get(order_id)
            if oracle_entry is None:
                violations.append(
                    Evidence(
                        ref=order_id,
                        detail="decision for an order that is not in the expected-decisions fixture",
                    )
                )
            else:
                seen_oracle_ids.add(order_id)
                if decision != oracle_entry.get("decision"):
                    violations.append(
                        Evidence(
                            ref=order_id,
                            detail=(
                                f"oracle decision mismatch: artifact says {decision!r}, "
                                f"fixture says {oracle_entry.get('decision')!r}"
                            ),
                        )
                    )
                if action != oracle_entry.get("action"):
                    violations.append(
                        Evidence(
                            ref=order_id,
                            detail=(
                                f"oracle action mismatch: artifact says {action!r}, "
                                f"fixture says {oracle_entry.get('action')!r}"
                            ),
                        )
                    )

        if decision not in valid_decisions:
            violations.append(
                Evidence(ref=order_id, detail=f"unknown decision {decision!r}")
            )
            continue
        if action not in valid_actions:
            violations.append(
                Evidence(ref=order_id, detail=f"unknown action {action!r}")
            )
            continue

        checks = entry.get("checks")
        if not isinstance(checks, list) or not checks:
            violations.append(
                Evidence(ref=order_id, detail="missing the four ground-truth checks")
            )
            continue
        names: list[str] = []
        check_decisions: dict[str, str] = {}
        for check in checks:
            if not isinstance(check, dict):
                violations.append(
                    Evidence(ref=order_id, detail="check entry is not an object")
                )
                continue
            check_name = str(check.get("check", ""))
            names.append(check_name)
            if check_name not in refund_guard.REQUIRED_CHECKS:
                violations.append(
                    Evidence(ref=order_id, detail=f"unknown check {check_name!r}")
                )
                continue
            check_decision = str(check.get("decision", ""))
            if check_decision not in valid_decisions:
                violations.append(
                    Evidence(
                        ref=order_id,
                        detail=f"check {check_name!r} has unknown decision {check_decision!r}",
                    )
                )
                continue
            check_decisions[check_name] = check_decision
        duplicate_checks = sorted({name for name in names if names.count(name) > 1})
        if duplicate_checks:
            violations.append(
                Evidence(ref=order_id, detail=f"duplicate checks: {duplicate_checks}")
            )
        missing_checks = sorted(set(refund_guard.REQUIRED_CHECKS) - set(check_decisions))
        if missing_checks:
            violations.append(
                Evidence(ref=order_id, detail=f"missing checks: {missing_checks}")
            )
            continue
        computed_values = set(check_decisions.values())
        if Decision.FAIL.value in computed_values:
            computed_decision = Decision.FAIL.value
        elif Decision.CANNOT_VERIFY.value in computed_values:
            computed_decision = Decision.CANNOT_VERIFY.value
        else:
            computed_decision = Decision.PASS.value
        if decision != computed_decision:
            violations.append(
                Evidence(
                    ref=order_id,
                    detail=(
                        f"decision {decision!r} contradicts its checks "
                        f"({computed_decision!r})"
                    ),
                )
            )
        computed_failed = sorted(
            name for name, value in check_decisions.items() if value == Decision.FAIL.value
        )
        computed_unverified = sorted(
            name
            for name, value in check_decisions.items()
            if value == Decision.CANNOT_VERIFY.value
        )
        if sorted(entry.get("failed_checks") or []) != computed_failed:
            violations.append(
                Evidence(
                    ref=order_id,
                    detail=f"failed_checks does not match checks: {entry.get('failed_checks')}",
                )
            )
        if sorted(entry.get("unverified_checks") or []) != computed_unverified:
            violations.append(
                Evidence(
                    ref=order_id,
                    detail=(
                        "unverified_checks does not match checks: "
                        f"{entry.get('unverified_checks')}"
                    ),
                )
            )
        if decision == Decision.PASS.value and action != "approve":
            violations.append(
                Evidence(
                    ref=order_id,
                    detail=f"every check passed but action is {action!r}",
                )
            )
        if decision == Decision.FAIL.value and action != "deny":
            violations.append(
                Evidence(
                    ref=order_id,
                    detail=f"a rule failed but action is {action!r}: "
                    f"{entry.get('failed_checks')}",
                )
            )
        if decision == Decision.CANNOT_VERIFY.value and action == "approve":
            violations.append(
                Evidence(
                    ref=order_id,
                    detail="fail-closed violated: unverifiable precondition was "
                    f"approved (unverified={entry.get('unverified_checks')})",
                )
            )

    if expected_by_id is not None:
        missing_orders = sorted(set(expected_by_id) - seen_order_ids)
        if missing_orders:
            violations.append(
                Evidence(
                    ref="decisions.json",
                    detail=f"missing decisions for orders: {missing_orders}",
                )
            )
    if oracle_by_id is not None:
        missing_oracle = sorted(set(oracle_by_id) - seen_oracle_ids)
        if missing_oracle:
            violations.append(
                Evidence(
                    ref="decisions.json",
                    detail=f"missing decisions for expected orders: {missing_oracle}",
                )
            )

    ok = not violations
    detail = (
        f"{len(results)} decisions respect the fail-closed invariant"
        if ok
        else f"{len(violations)} fail-closed violation(s)"
    )
    return _result(
        name,
        ok,
        detail,
        decision=Decision.PASS if ok else Decision.FAIL,
        failure_class=FailureClass.POLICY_VIOLATION if not ok else None,
        evidence_items=violations[:40],
        remediation=(
            "A precondition could not be checked against ground truth. Deny or "
            "escalate; never approve."
        ),
        failures=[
            {"issue": item.detail, "ref": item.ref} for item in violations[:40]
        ],
        metrics={
            "orders": len(results),
            "decisions": tally,
            "cannot_verify": tally.get(Decision.CANNOT_VERIFY.value, 0),
            "violations": len(violations),
        },
    )


REQUIRED_MERGE_CHECKS = ("TESTS_PASS", "ARCHITECTURE_OK", "NO_UNINTENDED_SCOPE")


@validator("pr_merge_gate")
def pr_merge_gate(
    path: str | Path,
    *,
    name: str = "pr_merge_gate",
) -> dict[str, Any]:
    """Auto-merge only when every judgement returned a definite pass.

    This is the whole point of splitting the decision. "CI is green" is one
    judgement; a change can pass it and still be unmergeable because its blast
    radius is unknown or it reached across the architecture.

    The invariant: `cannot_verify` on any check blocks. A change nobody can
    reason about is not a change that merges unattended.
    """
    try:
        payload = load_content(path)
    except ValidationFailed as exc:
        return _result(name, False, str(exc), failure_class=FailureClass.MALFORMED)

    checks = payload.get("checks") or []
    if not isinstance(checks, list) or not checks:
        return _result(
            name,
            False,
            "no checks were produced",
            failure_class=FailureClass.MALFORMED,
        )

    valid = {member.value for member in Decision}
    blocking: list[Evidence] = []
    tally: dict[str, int] = {}
    names: list[str] = []
    for entry in checks:
        if not isinstance(entry, dict):
            blocking.append(Evidence(ref="review.json", detail="check entry is not an object"))
            continue
        label = str(entry.get("check", "?"))
        names.append(label)
        decision = str(entry.get("decision", ""))
        tally[decision] = tally.get(decision, 0) + 1
        if label not in REQUIRED_MERGE_CHECKS:
            blocking.append(
                Evidence(ref=label, detail=f"unknown check {label!r}; refusing to guess")
            )
            continue
        if decision not in valid:
            blocking.append(Evidence(ref=label, detail=f"unknown decision {decision!r}"))
            continue
        if decision == Decision.FAIL.value:
            blocking.append(
                Evidence(ref=label, detail=f"failed: {entry.get('detail', '')}")
            )
        elif decision == Decision.CANNOT_VERIFY.value:
            blocking.append(
                Evidence(
                    ref=label,
                    detail=(
                        "fail-closed: could not be decided, so it does not merge. "
                        + str(entry.get("detail", ""))
                    ),
                )
            )
        elif decision == Decision.ABSTAIN.value:
            blocking.append(
                Evidence(ref=label, detail="abstained; treated as blocking")
            )

    duplicates = sorted({label for label in names if names.count(label) > 1})
    if duplicates:
        blocking.append(
            Evidence(ref="review.json", detail=f"duplicate checks: {duplicates}")
        )
    missing = [label for label in REQUIRED_MERGE_CHECKS if label not in names]
    if missing:
        blocking.append(
            Evidence(
                ref="review.json",
                detail=f"missing required checks: {missing}",
            )
        )

    declared = payload.get("merge")
    if declared != "allow":
        blocking.append(
            Evidence(
                ref="review.json",
                detail=f"the artifact does not claim allow (merge={declared!r})",
            )
        )
    elif blocking:
        blocking.append(
            Evidence(
                ref="review.json",
                detail="the artifact claims merge=allow while a check blocks",
            )
        )

    ok = not blocking
    return _result(
        name,
        ok,
        (
            f"{len(checks)} judgements all passed: "
            + ", ".join(str(entry.get("check")) for entry in checks)
            if ok
            else f"{len(blocking)} blocking judgement(s); auto-merge refused"
        ),
        decision=Decision.PASS if ok else Decision.FAIL,
        failure_class=FailureClass.POLICY_VIOLATION if not ok else None,
        evidence_items=blocking[:40],
        remediation=(
            "Resolve every FAIL, and replace every CANNOT_VERIFY with a definite "
            "answer (declare a scope, restore the revision range, or make the "
            "suite runnable). Do not merge on an undecided change."
        )
        if not ok
        else None,
        failures=[{"issue": item.detail, "ref": item.ref} for item in blocking[:40]],
        metrics={"checks": len(checks), "decisions": tally, "blocking": len(blocking)},
    )


def _load_brief_module(tools_dir: str | Path):
    """Import the brief selector that lives in the daily-trends tools checkout.

    The brief is editorial logic (what to show), so it lives with the renderer
    and the interests config; the gate imports the same module so "the brief on
    the page" and "the brief that was verified" cannot diverge.
    """
    import importlib.util

    path = Path(tools_dir) / "tools" / "brief.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("daily_trends_brief_module", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@validator("daily_trends_brief")
def daily_trends_brief(
    path: str | Path,
    *,
    tools_dir: str | Path,
    limit: int = 5,
    name: str = "daily_trends_brief",
) -> dict[str, Any]:
    """今日速读：能不能挑出「你真正要读的 5 条」，且每条结构完整。

    This is the gate for the reading half of the product, not the publishing
    half: the article may be complete and still useless if the top of the page
    is 36 unsorted items. Structure blocks; how *deep* each summary is does not
    (that is `daily_trends_depth`, and it warns).
    """
    try:
        content = load_content(path)
    except ValidationFailed as exc:
        return _result(name, False, str(exc))
    module = _load_brief_module(tools_dir)
    if module is None:
        return _result(
            name,
            False,
            f"brief selector not found under {tools_dir}",
            decision=Decision.CANNOT_VERIFY,
            failure_class=FailureClass.PRECONDITION_UNKNOWN,
            remediation="Check the task's {tools_dir} path: tools/brief.py must exist.",
            metrics={"entries": 0},
        )
    try:
        payload = module.select_brief(content, limit=limit)
    except Exception as exc:  # noqa: BLE001 - a crashing selector is a result
        return _result(
            name,
            False,
            f"brief selection raised {type(exc).__name__}: {exc}",
            decision=Decision.CANNOT_VERIFY,
            failure_class=FailureClass.INTERNAL_ERROR,
            metrics={"entries": 0},
        )

    entries = payload.get("entries") or []
    failures: list[dict[str, Any]] = []
    if not entries:
        failures.append(
            {
                "issue": "brief selected nothing — every candidate was filtered out "
                "or no item matched the interest profile"
            }
        )
    if len(entries) > limit:
        failures.append({"issue": f"brief has {len(entries)} entries (limit {limit})"})
    for entry in entries:
        title = (entry.get("title") or {}).get("zh") or (entry.get("title") or {}).get("en") or "?"
        if not (entry.get("body") or {}).get("zh"):
            failures.append({"issue": "brief entry has no zh body", "title": title[:40]})
        if not entry.get("why_zh"):
            failures.append({"issue": "brief entry does not say why it was picked", "title": title[:40]})

    metrics = {
        "entries": len(entries),
        "candidates": payload.get("candidates"),
        "excluded": payload.get("excluded"),
        "without_limit_statement": payload.get("without_limit"),
    }
    ok = not failures
    return _result(
        name,
        ok,
        f"brief ok ({len(entries)} entries from {payload.get('candidates')} candidates, "
        f"{payload.get('excluded')} filtered by interest profile)"
        if ok
        else f"{len(failures)} brief problem(s)",
        failures=failures[:20],
        remediation="Fix the interest config or the content before publishing."
        if not ok
        else None,
        metrics=metrics,
    )


@validator("daily_trends_depth")
def daily_trends_depth(
    path: str | Path,
    *,
    tools_dir: str | Path,
    min_ratio: float = 0.6,
    name: str = "daily_trends_depth",
) -> dict[str, Any]:
    """摘要深度：读完能说出这项技术在做什么吗。

    Measured per item: does the text contain a mechanism/evidence word and a
    limitation? A day can be structurally perfect and still be unreadable
    ("某公司发布了某模型"), which is the failure the reader actually feels.
    Reported as warnings, not blockers: the fix is a rewrite, not a stop-the-line.
    """
    try:
        content = load_content(path)
    except ValidationFailed as exc:
        return _result(name, False, str(exc))
    module = _load_brief_module(tools_dir)
    if module is None:
        return _result(
            name,
            False,
            f"depth signals unavailable: {tools_dir}/tools/brief.py not found",
            decision=Decision.CANNOT_VERIFY,
            failure_class=FailureClass.PRECONDITION_UNKNOWN,
            metrics={"items": 0},
        )

    shallow: list[dict[str, Any]] = []
    no_limit: list[str] = []
    items = 0
    for _section, _location, item in module.iter_items(content):
        items += 1
        body = module.bi_text(item)
        signals = module.depth_signals(body.get("zh") or body.get("en") or "")
        title = (module.bi_title(item).get("zh") or "")[:40]
        if not signals["has_substance"]:
            shallow.append({"issue": "no mechanism or evidence in the summary", "title": title})
        if not signals["has_limit"]:
            no_limit.append(title)

    ratio = 1.0 if not items else (items - len(no_limit)) / items
    warnings = [{"issue": "summary states no limitation", "title": t} for t in no_limit[:10]]
    metrics = {
        "items": items,
        "with_limit": items - len(no_limit),
        "limit_ratio": round(ratio, 3),
        "without_substance": len(shallow),
    }
    detail = (
        f"{items - len(no_limit)}/{items} summaries state a limitation "
        f"(mechanism/evidence missing in {len(shallow)})"
    )
    if ratio < min_ratio:
        detail += f" — below the {min_ratio:.0%} target"
    return _result(name, True, detail, warnings=warnings, metrics=metrics)


@validator("sitemap_sane")
def sitemap_sane(
    path: str | Path,
    *,
    max_bytes: int = 200_000,
    # A freshly generated Hexo sitemap legitimately contains a run of 4
    # whitespace-only lines before </urlset>; the drift we guard against grows
    # without bound, so the threshold sits well above the baseline.
    max_blank_run: int = 8,
    name: str = "sitemap_sane",
) -> dict[str, Any]:
    """Catch generated-file drift before it gets committed and pushed.

    Written after a real incident: the renderer's sitemap update appended blank
    lines on every run (see `update_sitemap` in the daily-trends tools). This
    validator fails on that class of drift - duplicate URLs, runaway blank
    lines, unparseable XML, or an implausibly large file.
    """
    import xml.etree.ElementTree as ET

    target = Path(path)
    if not target.is_file():
        return _result(name, False, f"missing sitemap: {target}")
    text = target.read_text(encoding="utf-8", errors="replace")
    failures: list[dict[str, Any]] = []

    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        return _result(name, False, f"sitemap is not valid XML: {exc}")

    locs = [
        (element.text or "").strip()
        for element in root.iter()
        if element.tag.endswith("loc")
    ]
    duplicates = sorted({loc for loc in locs if loc and locs.count(loc) > 1})
    if duplicates:
        failures.append({"issue": "duplicate <loc> entries", "locs": duplicates[:10]})

    blank_runs = _max_blank_run(text)
    if blank_runs > max_blank_run:
        failures.append(
            {
                "issue": f"{blank_runs} consecutive blank lines (max {max_blank_run}) - "
                "the sitemap is accumulating whitespace instead of being rewritten in place"
            }
        )

    size = len(text.encode("utf-8"))
    if size > max_bytes:
        failures.append({"issue": f"sitemap grew to {size} bytes (max {max_bytes})"})

    ok = not failures
    return _result(
        name,
        ok,
        "sitemap ok" if ok else f"{len(failures)} sitemap problem(s)",
        remediation=(
            "The sitemap is accumulating instead of being rewritten in place. "
            "Re-run the renderer, or restore the file from git if it drifted."
        )
        if not ok
        else None,
        failures=failures,
        metrics={
            "urls": len(locs),
            "trends_urls": sum(1 for loc in locs if "/trends/" in loc),
            "max_blank_run": blank_runs,
            "bytes": size,
        },
    )


def _max_blank_run(text: str) -> int:
    longest = current = 0
    for line in text.splitlines():
        if line.strip() == "":
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _collect_urls(node: Any, acc: set[str] | None = None) -> set[str]:
    acc = acc if acc is not None else set()
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, str) and key in ("url", "html_url", "hn_url", "link", "source_url"):
                acc.add(_normalise(value))
            else:
                _collect_urls(value, acc)
    elif isinstance(node, list):
        for value in node:
            _collect_urls(value, acc)
    return acc


def _normalise(url: str) -> str:
    text = url.strip().rstrip("/")
    for prefix in ("https://", "http://", "www."):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    return text.lower()


def _url_in_raw(url: str, raw_urls: set[str]) -> bool:
    target = _normalise(url)
    if target in raw_urls:
        return True
    # Tolerate tracking params / http-vs-https and arxiv abs-vs-v1 drift.
    base = target.split("?", 1)[0]
    if base in raw_urls:
        return True
    return any(raw.startswith(base) or base.startswith(raw) for raw in raw_urls if len(base) > 20)


def run_all(
    specs: list[Any],
    *,
    task_dir: Path,
    date: str,
    run_dir: Path,
    extra: dict[str, Any] | None = None,
    logger: Any | None = None,
    policy: DecisionPolicy | None = None,
) -> list[dict[str, Any]]:
    """Execute declared validators, resolving `{date}` templates in their args.

    The task's policy is applied last: a validator answers with a decision, the
    task decides whether CANNOT_VERIFY blocks. Fail-closed is the default.
    """
    from .taskspec import render

    policy = policy or DecisionPolicy()
    results: list[dict[str, Any]] = []
    for spec in specs:
        func = get(spec.name)
        kwargs = {}
        for key, value in (spec.args or {}).items():
            if isinstance(value, str):
                rendered = render(value, date=date, run_dir=run_dir, extra=extra)
                path = Path(rendered).expanduser()
                kwargs[key] = path if path.is_absolute() else task_dir / rendered
            elif isinstance(value, list):
                resolved = []
                for entry in value:
                    rendered = render(str(entry), date=date, run_dir=run_dir, extra=extra)
                    path = Path(rendered).expanduser()
                    resolved.append(str(path if path.is_absolute() else task_dir / rendered))
                kwargs[key] = resolved
            else:
                kwargs[key] = value
        try:
            result = func(**kwargs)
        except Exception as exc:  # a validator crash must not crash the run
            result = _result(spec.name, False, f"validator raised {type(exc).__name__}: {exc}")
        result = apply_policy(result, policy)
        if logger:
            label = result.get("decision", "ok" if result.get("ok") else "fail")
            logger.info(
                "validator %s: %s%s (%s)",
                result["name"],
                label,
                "" if result.get("ok") else " -> blocked",
                result.get("detail", ""),
            )
        results.append(result)
    return results
