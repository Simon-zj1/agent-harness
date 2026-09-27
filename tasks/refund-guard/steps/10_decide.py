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

from harness.refund_guard import decide

TASK_DIR = Path(__file__).resolve().parent.parent
RUN_DIR = Path(os.environ.get("AGENT_RUN_DIR", TASK_DIR / "out"))
STEP_ID = os.environ.get("AGENT_STEP_ID", "10_decide")


def load(name: str) -> dict:
    return json.loads((TASK_DIR / "fixtures" / name).read_text(encoding="utf-8"))


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
