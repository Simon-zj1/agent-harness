"""Deterministic refund eligibility checks.

This module is shared by the refund task step and its validator.  The validator
must be able to recompute the decision from the order record instead of trusting
the artifact produced by the step.
"""

from __future__ import annotations

from typing import Any

REQUIRED_CHECKS = ("order_status", "refund_window", "payment_reference", "currency")


def decide(order: dict[str, Any], policy: dict[str, Any], ledger: set[str]) -> dict[str, Any]:
    """Return a typed, fail-closed decision for one refund order."""
    checks: list[dict[str, str]] = []

    status = order.get("status")
    if status is None:
        checks.append(
            {
                "check": "order_status",
                "decision": "cannot_verify",
                "detail": "order record has no status field",
            }
        )
    elif status in policy["refundable_statuses"]:
        checks.append(
            {
                "check": "order_status",
                "decision": "pass",
                "detail": f"status={status!r} is refundable",
            }
        )
    else:
        checks.append(
            {
                "check": "order_status",
                "decision": "fail",
                "detail": f"status={status!r} is not in {policy['refundable_statuses']}",
            }
        )

    days = order.get("days_since_delivery")
    window = policy["refund_window_days"]
    if not isinstance(days, int):
        checks.append(
            {
                "check": "refund_window",
                "decision": "cannot_verify",
                "detail": "delivery date unknown, window cannot be evaluated",
            }
        )
    elif days <= window:
        checks.append(
            {"check": "refund_window", "decision": "pass", "detail": f"{days}d <= {window}d"}
        )
    else:
        checks.append(
            {"check": "refund_window", "decision": "fail", "detail": f"{days}d > {window}d"}
        )

    payment_ref = order.get("payment_ref")
    if payment_ref is None:
        checks.append(
            {
                "check": "payment_reference",
                "decision": "cannot_verify",
                "detail": "no payment reference on the order",
            }
        )
    elif payment_ref in ledger:
        checks.append(
            {
                "check": "payment_reference",
                "decision": "pass",
                "detail": f"{payment_ref} found in payment ledger",
            }
        )
    else:
        checks.append(
            {
                "check": "payment_reference",
                "decision": "cannot_verify",
                "detail": f"{payment_ref} not found in payment ledger",
            }
        )

    amount = order.get("amount")
    currency = order.get("currency")
    if currency != policy["currency"]:
        checks.append(
            {
                "check": "currency",
                "decision": "cannot_verify",
                "detail": f"currency={currency!r} has no conversion rule",
            }
        )
    else:
        checks.append(
            {"check": "currency", "decision": "pass", "detail": f"currency={currency}"}
        )

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
        "unverified_checks": [
            c["check"] for c in checks if c["decision"] == "cannot_verify"
        ],
    }


__all__ = ["REQUIRED_CHECKS", "decide"]
