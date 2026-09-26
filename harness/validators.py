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
    max_insights: int = 20,
    max_repos: int = 20,
    name: str = "daily_trends_structure",
) -> dict[str, Any]:
    """Check the article contract the published pages depend on."""
    try:
        content = load_content(path)
    except ValidationFailed as exc:
        return _result(name, False, str(exc))
    failures, metrics = check_structure(
        content, max_insights=max_insights, max_repos=max_repos
    )
    ok = not failures
    return _result(
        name,
        ok,
        "structure ok" if ok else f"{len(failures)} structural problem(s)",
        remediation=(
            "The published pages depend on this shape. Re-run compose, or fix the "
            "offending items: every item needs zh+en title/summary/comment and at "
            "least one source."
        )
        if not ok
        else None,
        failures=failures[:40],
        metrics=metrics,
    )


def check_structure(
    content: dict[str, Any],
    *,
    max_insights: int = 20,
    max_repos: int = 20,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """In-memory version of the structural contract (used for self-repair)."""
    failures: list[dict[str, Any]] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append({"issue": message})

    require(bool(content.get("date")), "missing date")
    for key in ("title", "summary", "notes"):
        block = content.get(key) or {}
        for lang in ("zh", "en"):
            require(bool(str(block.get(lang, "")).strip()), f"{key}.{lang} is empty")

    sections = content.get("sections") or []
    require(len(sections) >= 2, f"expected 2 sections, found {len(sections)}")

    insights_count = 0
    repos_count = 0
    item_issues = 0
    for section in sections:
        sid = section.get("id")
        items: list[dict[str, Any]] = []
        for group in section.get("groups") or []:
            items.extend(group.get("items") or [])
        items.extend(section.get("items") or [])
        if sid == "insights":
            insights_count = len(items)
            require(
                insights_count <= max_insights,
                f"insights has {insights_count} items (max {max_insights})",
            )
            groups = section.get("groups") or []
            require(bool(groups), "insights section has no groups")
            for group in groups:
                require(bool(group.get("id")), "insights group without id")
                require(bool((group.get("title") or {}).get("zh")), "insights group without title")
        elif sid == "github":
            repos_count = len(items)
            require(
                repos_count <= max_repos,
                f"github has {repos_count} items (max {max_repos})",
            )
        for item in items:
            for field in ("title", "summary", "comment"):
                block = item.get(field) or {}
                if not str(block.get("zh", "")).strip() or not str(block.get("en", "")).strip():
                    item_issues += 1
                    failures.append(
                        {"issue": f"item missing bilingual {field}", "title": (item.get("title") or {}).get("zh", "")[:40]}
                    )
            sources = item.get("sources")
            if not isinstance(sources, list) or not sources:
                item_issues += 1
                failures.append(
                    {"issue": "item has no sources", "title": (item.get("title") or {}).get("zh", "")[:40]}
                )

    references = content.get("references") or []
    require(bool(references), "no references")

    metrics = {
        "insights": insights_count,
        "repos": repos_count,
        "references": len(references),
        "item_issues": item_issues,
    }
    return failures, metrics


@validator("daily_trends_references")
def daily_trends_references(
    path: str | Path, *, name: str = "daily_trends_references"
) -> dict[str, Any]:
    """Reference ids must resolve, urls must be http(s), and nothing may be orphaned."""
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
    if orphans:
        failures.append({"issue": "reference never cited", "ids": orphans})

    ok = not failures
    return _result(
        name,
        ok,
        "references ok" if ok else f"{len(failures)} reference problem(s)",
        remediation=(
            "Every cited source id must resolve, every reference must be cited at "
            "least once, and urls must be http(s). Drop the orphan references or "
            "cite them."
        )
        if not ok
        else None,
        failures=failures[:40],
        metrics={
            "references": len(references),
            "cited": len(used),
            "orphans": len(orphans),
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
    for ref in cited:
        ref_id = ref.get("id")
        url = str(ref.get("url", ""))
        verdict = classify_url(url, raw_urls)
        verdicts[ref_id] = verdict
        tally[verdict.decision.value] = tally.get(verdict.decision.value, 0) + 1
        if verdict.decision is not Decision.PASS:
            evidence_items.append(
                Evidence(
                    ref=f"ref#{ref_id}",
                    detail=verdict.reason,
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
        },
    )


@validator("refund_decisions_fail_closed")
def refund_decisions_fail_closed(
    path: str | Path,
    *,
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

    valid_decisions = {member.value for member in Decision}
    valid_actions = {"approve", "deny", "escalate"}
    violations: list[Evidence] = []
    tally: dict[str, int] = {}

    for entry in results:
        order_id = str(entry.get("order_id", "?"))
        decision = str(entry.get("decision", ""))
        action = str(entry.get("action", ""))
        tally[decision] = tally.get(decision, 0) + 1

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
    if not checks:
        return _result(
            name,
            False,
            "no checks were produced",
            failure_class=FailureClass.MALFORMED,
        )

    valid = {member.value for member in Decision}
    blocking: list[Evidence] = []
    tally: dict[str, int] = {}
    for entry in checks:
        label = str(entry.get("check", "?"))
        decision = str(entry.get("decision", ""))
        tally[decision] = tally.get(decision, 0) + 1
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

    declared = payload.get("merge")
    if declared == "allow" and blocking:
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
