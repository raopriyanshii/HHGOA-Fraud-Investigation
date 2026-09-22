"""
Phase G2 -- policy & next-best-action layer, implementing the approved
Revision 2 specification exactly. Pure function over a G1 ledger
(src.agent.workflow.investigate()'s return value): no TigerGraph, no
MCP, no LLM, no risk_score, no invented fraud_probability/exposure_usd/
pattern/customer-response. Every action is traced to a literal condition
from the organizer's official Fraud Policy (D:\\hackathon\\task_5\\README.md)
or classified DEFERRED with the exact missing input named.

This module does not import or call src.agent.workflow -- it only
consumes that function's return value, passed in by the caller.
"""
from __future__ import annotations

# Official Fraud Policy action vocabulary (organizer README, "Fraud
# Policy" section 1), in the document's own listed order -- this is the
# one fixed canonical order every G2 output list follows, always.
CANONICAL_ACTION_ORDER = (
    "ALLOW_TRANSACTION",
    "DECLINE_TRANSACTION",
    "MONITOR_CARD",
    "MONITOR_CONNECTED_CARDS",
    "WARN_CUSTOMER",
    "VERIFY_WITH_CUSTOMER",
    "STEP_UP_AUTH",
    "BLOCK_CARD",
    "BLOCK_ALL_CARDS",
    "GENERATE_REPORT",
    "CREATE_CASE",
    "FILE_REPORT",
    "ESCALATE_TO_ANALYST",
    "CLOSE_NO_FRAUD",
)
OFFICIAL_ACTIONS = frozenset(CANONICAL_ACTION_ORDER)
assert len(CANONICAL_ACTION_ORDER) == 14
assert len(OFFICIAL_ACTIONS) == 14  # no duplicates in the vocabulary

RECOMMENDED = "RECOMMENDED"
PROHIBITED = "PROHIBITED"
ELIGIBLE_NOT_RECOMMENDED = "ELIGIBLE_NOT_RECOMMENDED"
DEFERRED = "DEFERRED"

# Every one of the 12 actions with no literal, satisfiable condition in
# this version -- each reason names the specific missing input, per the
# approved spec's decision-rule table. Not a fallback/default: these are
# the documented, final determination for V1, not a placeholder.
_DEFERRED_REASONS = {
    "ALLOW_TRANSACTION": "no policy rule cites a positive trigger condition for this action",
    "DECLINE_TRANSACTION": "R1/R4/R5 require fraud_probability or a customer-reply state, not computed by G1/G2",
    "MONITOR_CARD": "R4 requires a no-reply state, not tracked by G1/G2",
    "MONITOR_CONNECTED_CARDS": "R6 requires confirmed fraud on other cards sharing a named element (device profile, billing region, or recipient email); G1 never checks other cards' fraud status",
    "WARN_CUSTOMER": "R7 requires a customer response and a recurring-pattern match, not available",
    "VERIFY_WITH_CUSTOMER": "R1 requires fraud_probability, not computed by G1/G2",
    "STEP_UP_AUTH": "R1/R5 require fraud_probability or a card-testing pattern, not available",
    "BLOCK_CARD": "R1/R2/R5 require fraud_probability or a customer response, not available",
    "GENERATE_REPORT": "no policy rule cites a positive trigger condition for this action",
    "FILE_REPORT": "R2/R6/R9 require a customer response, cross-card fraud verification for a named shared element, or pattern judgment, none available",
    "ESCALATE_TO_ANALYST": "R8/R9 require fraud_probability and exposure_usd, or pattern judgment, none available",
    "CLOSE_NO_FRAUD": "R3 requires a customer confirmation response, not available",
}


def _get_trigger_type(ledger: dict) -> tuple[str | None, str]:
    """Returns (trigger_type_or_None, status). status is one of:
      "known"   -- a successful benchmark_case_resolution call recorded trigger_type
      "failed"  -- a benchmark_case_resolution call exists but failed
      "missing" -- no such call exists (a flagged_txn_id trigger was used)
    """
    for tc in ledger.get("tool_calls", []):
        if tc.get("tool") == "benchmark_case_resolution":
            if tc.get("success"):
                result = tc.get("curated_result_summary") or {}
                return result.get("trigger_type"), "known"
            return None, "failed"
    return None, "missing"


def _evaluate_create_case(ledger: dict) -> tuple[str, str | None, str]:
    trigger_type, status = _get_trigger_type(ledger)
    if status == "known":
        if trigger_type == "customer_report":
            return RECOMMENDED, "auto", "§3a: customer disputes a charge (trigger_type == 'customer_report')"
        return (
            DEFERRED,
            None,
            f"§3a: trigger_type == {trigger_type!r} confirmed not a customer dispute; "
            f"no other rule grounds CREATE_CASE from available evidence",
        )
    if status == "failed":
        return (
            DEFERRED,
            None,
            "§3a: trigger_type unavailable -- benchmark_case_resolution call failed (see ledger uncertainties)",
        )
    return (
        DEFERRED,
        None,
        "§3a: trigger_type unavailable -- investigation used a flagged_txn_id trigger, no case resolution was performed",
    )


def _evaluate_block_all_cards(ledger: dict) -> tuple[str, str | None, str]:
    # Unconditional in V1: R10's exception can never be confirmed true from
    # G1's evidence (see module docstring / spec Revision 2), regardless of
    # what the ledger contains -- this function intentionally ignores its
    # `ledger` argument's content beyond accepting it, for exactly that
    # reason: no evidence shape changes this determination.
    return (
        PROHIBITED,
        None,
        "R10: exception (two of the customer's own cards independently confirmed fraud, "
        "or confirmed compromised credentials) is not verifiable from G1's evidence -- G1 "
        "never enumerates a customer's other cards or checks their individual histories, "
        "and has no credential-compromise signal. Unprovable exception -> prohibition holds.",
    )


def evaluate(ledger: dict) -> dict:
    """Pure function: G1 ledger -> G2 policy decision.

    Every one of the 14 official actions is evaluated exactly once, in
    CANONICAL_ACTION_ORDER, and placed into exactly one of the four
    policy-state buckets -- each bucket list is built by iterating the
    fixed CANONICAL_ACTION_ORDER tuple (never a dict/set), so repeated
    calls on the same ledger always produce identical output, including
    list order.
    """
    recommended: list[dict] = []
    prohibited: list[dict] = []
    eligible_not_recommended: list[dict] = []
    deferred: list[dict] = []

    for action in CANONICAL_ACTION_ORDER:
        if action == "CREATE_CASE":
            state, route, reason = _evaluate_create_case(ledger)
        elif action == "BLOCK_ALL_CARDS":
            state, route, reason = _evaluate_block_all_cards(ledger)
        else:
            state, route, reason = DEFERRED, None, _DEFERRED_REASONS[action]

        entry = {"action": action, "reason": reason}
        if state == RECOMMENDED:
            entry["route"] = route
            recommended.append(entry)
        elif state == PROHIBITED:
            prohibited.append(entry)
        elif state == ELIGIBLE_NOT_RECOMMENDED:
            eligible_not_recommended.append(entry)
        elif state == DEFERRED:
            deferred.append(entry)
        else:
            raise AssertionError(f"unknown policy state {state!r} for action {action!r}")

    return {
        "investigation_recommendation": ledger.get("recommendation"),
        "recommended": recommended,
        "prohibited": prohibited,
        "eligible_not_recommended": eligible_not_recommended,
        "deferred": deferred,
        "uncertainties": list(ledger.get("uncertainties", [])),
    }
