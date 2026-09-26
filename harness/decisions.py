"""Typed decisions: the vocabulary that replaces boolean pass/fail.

A validator that can only answer True/False is forced to guess when it does not
know. That is how a fabricated URL slips through an "anti-fabrication" gate: the
gate cannot express "this is neither the source I fetched, nor obviously not
the source I fetched", so it picks the generous answer.

This module adds the missing answers and a failure taxonomy:

    PASS            ground truth confirms the claim
    FAIL            ground truth contradicts the claim
    CANNOT_VERIFY   ground truth is insufficient to decide either way
    ABSTAIN         this validator does not apply to this input

The default policy is fail-closed: CANNOT_VERIFY and ABSTAIN block. A task can
opt out with an explicit ``[policy]`` table in task.toml, which keeps the
relaxation visible in the task declaration instead of buried in a heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable


class Decision(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    CANNOT_VERIFY = "cannot_verify"
    ABSTAIN = "abstain"


class FailureClass(str, Enum):
    """Why a decision is not PASS.

    A stable vocabulary is what makes these decisions countable over time:
    without it, "validation failed" is a single opaque bucket.
    """

    FABRICATED_SOURCE = "fabricated_source"
    UNKNOWN_VARIANT = "unknown_variant"
    SOURCE_MISMATCH = "source_mismatch"
    STALE_EVIDENCE = "stale_evidence"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    PRECONDITION_UNKNOWN = "precondition_unknown"
    POLICY_VIOLATION = "policy_violation"
    SCOPE_VIOLATION = "scope_violation"
    MALFORMED = "malformed"
    NOT_APPLICABLE = "not_applicable"
    INTERNAL_ERROR = "internal_error"


@dataclass
class Evidence:
    """One concrete item the decision was made from."""

    ref: str
    detail: str
    url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"ref": self.ref, "detail": self.detail, "url": self.url}


def evidence(ref: str, detail: str, url: str | None = None) -> Evidence:
    return Evidence(ref=ref, detail=detail, url=url)


@dataclass
class DecisionResult:
    """A validator's answer, with the reason and the receipts attached."""

    name: str
    decision: Decision
    detail: str
    failure_class: FailureClass | None = None
    evidence: list[Evidence] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    remediation: str | None = None
    policy_action: str = "allow"

    @property
    def ok(self) -> bool:
        """Fail-closed view: only PASS is ok before policy is applied."""
        return self.decision is Decision.PASS

    def to_dict(self) -> dict[str, Any]:
        """Legacy dict shape plus the typed fields.

        `ok`/`failures`/`metrics` keep working for existing callers (the ledger,
        the CLI report, the step gate); the typed keys are additive.
        """
        failures = [
            {
                "issue": item.detail,
                "ref": item.ref,
                **({"url": item.url} if item.url else {}),
            }
            for item in self.evidence
        ]
        return {
            # -- legacy surface ------------------------------------------
            "name": self.name,
            "ok": self.ok,
            "detail": self.detail,
            "failures": failures,
            "metrics": self.metrics,
            # -- typed surface -------------------------------------------
            "decision": self.decision.value,
            "failure_class": self.failure_class.value if self.failure_class else None,
            "evidence": [item.to_dict() for item in self.evidence],
            "remediation": self.remediation,
            "policy_action": self.policy_action,
        }


@dataclass
class DecisionPolicy:
    """What a task does with an uncertain answer.

    `block` (the default) keeps the step gate shut; `warn` lets the run continue
    but still records the decision, so the uncertainty is never erased.
    """

    on_cannot_verify: str = "block"
    on_abstain: str = "block"
    on_fail: str = "block"

    _ALLOWED = ("block", "warn")

    @classmethod
    def from_table(cls, table: dict[str, Any] | None) -> "DecisionPolicy":
        table = table or {}
        policy = cls(
            on_cannot_verify=str(table.get("on_cannot_verify", "block")),
            on_abstain=str(table.get("on_abstain", "block")),
            on_fail=str(table.get("on_fail", "block")),
        )
        for field_name in ("on_cannot_verify", "on_abstain", "on_fail"):
            value = getattr(policy, field_name)
            if value not in cls._ALLOWED:
                raise ValueError(
                    f"policy.{field_name}={value!r} is not one of {cls._ALLOWED}"
                )
        return policy

    def action(self, decision: Decision) -> str:
        if decision is Decision.FAIL:
            return self.on_fail
        if decision is Decision.CANNOT_VERIFY:
            return self.on_cannot_verify
        if decision is Decision.ABSTAIN:
            return self.on_abstain
        return "allow"

    def allows(self, decision: Decision) -> bool:
        return self.action(decision) != "block"

    def to_dict(self) -> dict[str, str]:
        return {
            "on_fail": self.on_fail,
            "on_cannot_verify": self.on_cannot_verify,
            "on_abstain": self.on_abstain,
        }


def apply_policy(result: dict[str, Any], policy: DecisionPolicy) -> dict[str, Any]:
    """Rewrite the legacy `ok` flag according to the task's policy."""
    raw = result.get("decision")
    if raw is None:
        return result
    decision = Decision(raw)
    action = policy.action(decision)
    result["policy_action"] = action
    result["ok"] = decision is Decision.PASS or action == "warn"
    return result


def combine(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate typed decisions into one run-level verdict.

    Precedence is FAIL > CANNOT_VERIFY > ABSTAIN > PASS: a single failure is
    never averaged away by passing neighbours.
    """
    counts = {member.value: 0 for member in Decision}
    blocking: list[str] = []
    uncertain: list[str] = []
    for result in results:
        raw = result.get("decision")
        if raw is None:
            counts[Decision.PASS.value if result.get("ok") else Decision.FAIL.value] += 1
            if not result.get("ok"):
                blocking.append(result.get("name", "?"))
            continue
        decision = Decision(raw)
        counts[decision.value] += 1
        if decision is Decision.FAIL:
            blocking.append(result.get("name", "?"))
        elif decision is Decision.CANNOT_VERIFY:
            uncertain.append(result.get("name", "?"))
            if result.get("policy_action") == "block":
                blocking.append(result.get("name", "?"))
        elif decision is Decision.ABSTAIN and result.get("policy_action") == "block":
            blocking.append(result.get("name", "?"))

    if counts[Decision.FAIL.value]:
        verdict = Decision.FAIL
    elif counts[Decision.CANNOT_VERIFY.value]:
        verdict = Decision.CANNOT_VERIFY
    elif counts[Decision.ABSTAIN.value]:
        verdict = Decision.ABSTAIN
    else:
        verdict = Decision.PASS
    return {
        "decision": verdict.value,
        "counts": counts,
        "blocking": blocking,
        "uncertain": uncertain,
    }


def repair_brief(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Turn a blocked run into something actionable.

    A gate that only says "no" leaves the operator to reverse-engineer which
    claim broke and what to do about it. The typed result already carries the
    receipts (evidence) and the fix (remediation) — this collects them into one
    machine-readable brief so a human or a repair step has somewhere to start.
    """
    entries: list[dict[str, Any]] = []
    for result in results:
        if result.get("ok"):
            continue
        raw = result.get("decision")
        decision = raw if raw is not None else Decision.FAIL.value
        entries.append(
            {
                "validator": result.get("name", "?"),
                "decision": decision,
                "failure_class": result.get("failure_class"),
                "detail": result.get("detail", ""),
                "remediation": result.get("remediation"),
                "policy_action": result.get("policy_action"),
                "evidence": result.get("evidence", []),
                "metrics": result.get("metrics", {}),
            }
        )
    return {
        "blocking": [entry["validator"] for entry in entries],
        "count": len(entries),
        "entries": entries,
    }


# ---------------------------------------------------------------------------
# Ground-truth URL matching
#
# This is the piece that the boolean gate got wrong. "Tolerate http/https and
# trailing slashes" is a known, enumerable set of drifts. "Anything that shares
# a prefix" is not — it is an open set, and treating it as equal is how a
# fabricated URL passes an anti-fabrication check.
# ---------------------------------------------------------------------------

_TRACKING_PARAMS = ("utm_", "gclid", "fbclid", "ref=", "source=", "mc_cid", "mc_eid")

_KNOWN_DRIFT = ("scheme", "www", "trailing_slash", "tracking_params", "arxiv_version")

# Below this length a shared prefix is not evidence of anything: `x.com/a` is a
# prefix of half the internet. Short URLs therefore go straight to FAIL instead
# of CANNOT_VERIFY. This is a deliberate floor, not an oversight — the corpus
# only builds attacks from URLs longer than it.
_MIN_PREFIX_LENGTH = 20


def canonical_url(url: str) -> str:
    """Normalise only the drifts we can enumerate."""
    text = (url or "").strip()
    text = text.split("#", 1)[0]
    if "?" in text:
        base, _, query = text.partition("?")
        kept = [
            part
            for part in query.split("&")
            if part and not part.lower().startswith(_TRACKING_PARAMS)
        ]
        text = base + (("?" + "&".join(kept)) if kept else "")
    for prefix in ("https://", "http://"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    if text.startswith("www."):
        text = text[4:]
    return text.rstrip("/").lower()


def _strip_arxiv_version(text: str) -> str:
    """arxiv.org/abs/2609.24974v1 and .../2609.24974 are the same paper."""
    marker = "arxiv.org/abs/"
    if marker not in text:
        return text
    head, _, tail = text.partition(marker)
    digits = tail.split("v", 1)
    if len(digits) == 2 and digits[1].isdigit():
        return f"{head}{marker}{digits[0]}"
    return text


@dataclass
class UrlVerdict:
    decision: Decision
    reason: str
    failure_class: FailureClass | None = None
    matched: str | None = None
    drift: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "reason": self.reason,
            "failure_class": self.failure_class.value if self.failure_class else None,
            "matched": self.matched,
            "drift": self.drift,
        }


def classify_url(cited: str, raw_urls: set[str], *, known: set[str] | None = None) -> UrlVerdict:
    """Decide whether a cited URL traces back to the raw capture.

    Returns PASS only for an exact match or an enumerated drift. A URL that
    merely extends (or truncates) a real one is CANNOT_VERIFY, not PASS.
    """
    target = canonical_url(cited)
    if not target:
        return UrlVerdict(Decision.FAIL, "empty url", FailureClass.MALFORMED)

    known = known if known is not None else set(raw_urls)

    if target in known:
        return UrlVerdict(Decision.PASS, "exact match", matched=target)

    stripped = _strip_arxiv_version(target)
    if stripped != target and stripped in known:
        return UrlVerdict(
            Decision.PASS,
            "arxiv version drift",
            matched=stripped,
            drift="arxiv_version",
        )

    # The capture keeps whatever query string the fetch saw (`?st=…&reflink=…`),
    # while an article often cites the bare path. Same path, extra parameters is
    # an enumerable drift — but only that: the remainder must start at the query
    # boundary, so a fabricated *path* suffix stays CANNOT_VERIFY.
    if "?" not in target:
        for real in known:
            if real.startswith(target + "?"):
                return UrlVerdict(
                    Decision.PASS,
                    "cited url matches a fetched url that carries extra query parameters",
                    matched=real,
                    drift="query_suffix",
                )

    # Ambiguous: one side is a prefix of the other. We know the prefix is real,
    # we do not know that the remainder is.
    for real in known:
        if len(real) < _MIN_PREFIX_LENGTH:
            continue
        if target.startswith(real):
            return UrlVerdict(
                Decision.CANNOT_VERIFY,
                f"cited url extends a fetched url ({real}) by {target[len(real):]!r}",
                FailureClass.UNKNOWN_VARIANT,
                matched=real,
            )
        if real.startswith(target):
            return UrlVerdict(
                Decision.CANNOT_VERIFY,
                f"cited url is a truncation of a longer fetched url ({real})",
                FailureClass.UNKNOWN_VARIANT,
                matched=real,
            )

    return UrlVerdict(Decision.FAIL, "no fetched url matches", FailureClass.FABRICATED_SOURCE)


__all__ = [
    "Decision",
    "FailureClass",
    "Evidence",
    "DecisionResult",
    "DecisionPolicy",
    "UrlVerdict",
    "evidence",
    "apply_policy",
    "combine",
    "repair_brief",
    "canonical_url",
    "classify_url",
]
