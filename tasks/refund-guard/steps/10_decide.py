#!/usr/bin/env python3
"""Deterministic refund verifier: the agent proposes, this decides.

Nothing here calls a model. That is the point: eligibility is decided by
ground-truth checks against the order record and the payment ledger, so the
answer is either PASS, FAIL, or an explicit CANNOT_VERIFY — never a confidence
score that a threshold turns into money leaving the company.

The fail-closed rule is the whole reason this layer exists: a refund can only be
approved when every check passed on real data. "I could not check" denies.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent.parent
RUN_DIR = Path(os.environ.get("AGENT_RUN_DIR", TASK_DIR / "out"))
STEP_ID = os.environ.get("AGENT_STEP_ID", "10_decide")


def load(name: str) -> dict:
    return json.loads((TASK_DIR / "fixtures" / name).read_text(encoding="utf-8"))


def decide(order: dict, policy: dict, ledger: set[str]) -> dict:
    """Return a typed decision for one order."""
    checks: list[dict] = []

    status = order.get("status")
    if status is None:
        checks.append({"check": "order_status", "decision": "cannot_verify",
                       "detail": "order record has no status field"})
    elif status in policy["refundable_statuses"]:
        checks.append({"check": "order_status", "decision": "pass",
                       "detail": f"status={status!r} is refundable"})
    else:
        checks.append({"check": "order_status", "decision": "fail",
                       "detail": f"status={status!r} is not in {policy['refundable_statuses']}"})

    days = order.get("days_since_delivery")
    window = policy["refund_window_days"]
    if not isinstance(days, int):
        checks.append({"check": "refund_window", "decision": "cannot_verify",
                       "detail": "delivery date unknown, window cannot be evaluated"})
    elif days <= window:
        checks.append({"check": "refund_window", "decision": "pass",
                       "detail": f"{days}d <= {window}d"})
    else:
        checks.append({"check": "refund_window", "decision": "fail",
                       "detail": f"{days}d > {window}d"})

    payment_ref = order.get("payment_ref")
    if payment_ref is None:
        checks.append({"check": "payment_reference", "decision": "cannot_verify",
                       "detail": "no payment reference on the order"})
    elif payment_ref in ledger:
        checks.append({"check": "payment_reference", "decision": "pass",
                       "detail": f"{payment_ref} found in payment ledger"})
    else:
        checks.append({"check": "payment_reference", "decision": "cannot_verify",
                       "detail": f"{payment_ref} not found in payment ledger"})

    amount = order.get("amount")
    currency = order.get("currency")
    if currency != policy["currency"]:
        checks.append({"check": "currency", "decision": "cannot_verify",
                       "detail": f"currency={currency!r} has no conversion rule"})
    else:
        checks.append({"check": "currency", "decision": "pass",
                       "detail": f"currency={currency}"})

    decisions = {c["decision"] for c in checks}
    if "fail" in decisions:
        verdict, action, failure_class = "fail", "deny", "policy_violation"
    elif "cannot_verify" in decisions:
        verdict, action, failure_class = "cannot_verify", "deny", "precondition_unknown"
    else:
        verdict, action, failure_class = "pass", "approve", None

    return {
        "order_id": order.get("order_id"),
        "amount": amount,
        "decision": verdict,
        "action": action,
        "failure_class": failure_class,
        "checks": checks,
        "failed_checks": [c["check"] for c in checks if c["decision"] == "fail"],
        "unverified_checks": [c["check"] for c in checks if c["decision"] == "cannot_verify"],
    }


def main() -> int:
    book = load("orders.json")
    proposals = {
        p["order_id"]: p for p in load("agent_proposals.json").get("proposals", [])
    }
    ledger = set(book.get("payment_ledger") or [])
    policy = book["policy"]

    results = []
    overrides = []
    for order in book["orders"]:
        verdict = decide(order, policy, ledger)
        proposal = proposals.get(verdict["order_id"])
        verdict["proposed_action"] = proposal["proposed_action"] if proposal else None
        verdict["proposed_reason"] = proposal.get("reason") if proposal else None
        verdict["verifier_overrode_agent"] = bool(
            proposal and proposal["proposed_action"] != verdict["action"]
        )
        if verdict["verifier_overrode_agent"]:
            overrides.append(verdict["order_id"])
        results.append(verdict)

    # A proposal for an order that does not exist is itself a finding.
    known = {r["order_id"] for r in results}
    orphan_proposals = sorted(set(proposals) - known)

    tally: dict[str, int] = {}
    for entry in results:
        tally[entry["decision"]] = tally.get(entry["decision"], 0) + 1

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "results": results,
        "tally": tally,
        "overridden_by_verifier": overrides,
        "orphan_proposals": orphan_proposals,
    }
    (RUN_DIR / "decisions.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"decided {len(results)} orders: "
        + ", ".join(f"{k}={v}" for k, v in sorted(tally.items()))
        + f"; verifier overrode the agent on {len(overrides)} order(s)"
    )
    result_path = RUN_DIR / "steps" / f"{STEP_ID}.result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "artifacts": [str(RUN_DIR / "decisions.json")],
                "metrics": {
                    "orders": len(results),
                    "approved": sum(1 for r in results if r["action"] == "approve"),
                    "denied": sum(1 for r in results if r["action"] == "deny"),
                    "cannot_verify": tally.get("cannot_verify", 0),
                    "agent_overrides": len(overrides),
                    **{f"decision_{k}": v for k, v in tally.items()},
                },
                "notes": f"fail-closed refund verification over {len(results)} orders",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
