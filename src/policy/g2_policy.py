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
    "MONITOR_CARD": "R4 requires a no-reply state, not tracked by G1/G2",
    "WARN_CUSTOMER": "R7 requires a customer response and a recurring-pattern match, not available",
    "VERIFY_WITH_CUSTOMER": "R1 requires fraud_probability, not computed by G1/G2",
    "BLOCK_CARD": (
        "R5's >$100-cleared-purchase trigger condition can be checked once pattern_evidence "
        "exists, but BLOCK_CARD's required route (L1 vs L2) depends on exposure_usd, which "
        "remains unavailable; R1/R2 also require fraud_probability or a customer response, "
        "neither available either -- BLOCK_CARD remains DEFERRED regardless of R5 evidence"
    ),
    "GENERATE_REPORT": "no policy rule cites a positive trigger condition for this action",
    "ESCALATE_TO_ANALYST": "R8/R9 require fraud_probability and exposure_usd, or pattern judgment, none available",
    "CLOSE_NO_FRAUD": "R3 requires a customer confirmation response, not available",
}
# MONITOR_CONNECTED_CARDS, FILE_REPORT, DECLINE_TRANSACTION, and
# STEP_UP_AUTH are not in this table -- they have dedicated evaluators
# below (the same pattern CREATE_CASE and BLOCK_ALL_CARDS already use),
# since R6/R5 make them conditionally RECOMMENDED rather than
# unconditionally DEFERRED. BLOCK_CARD stays here: its trigger is
# partially checkable via R5 now, but its route can never be determined
# without exposure_usd, so it is unconditionally DEFERRED regardless of
# ledger content -- a static reason is accurate, a dedicated evaluator
# would not change the outcome and is not added.


def _r6_evidence_present(ledger: dict) -> bool:
    """R6 (organizer Fraud Policy): "several cards show fraud from the
    same device profile, billing region, or recipient email." Sufficient
    ONLY when a successful cross_card_fraud_verification (Q9) result
    contains at least one other_cards entry, under via_device or
    via_region, with has_confirmed_fraud == True. Mere connectivity
    (a shared element with no confirmed fraud on the other card) is not
    enough -- G1's Q9 gate already filters to non-hub elements before any
    result reaches the ledger, so a non-null via_device/via_region here
    already means the underlying element was non-hub; this function does
    not need to, and does not, re-check hub_flag. Never reads
    HHG_CONNECTED_TO-derived evidence (connected_via_case) -- Q9 never
    gathers that, so it cannot leak in here. Never reads risk_score.

    G8 defect fix: `tool_calls[].curated_result_summary` is intentionally
    _trim()-collapsed by workflow.py (_MAX_LIST_SAMPLE=5) -- for a real,
    highly-connected card (HHG-011's device alone returns 795 other_cards
    entries), `other_cards` becomes {"count": 795, "sample": [...]}
    instead of a list, and iterating a dict yields its keys as bare
    strings, crashing on entry.get(...) (discovered via a live G8 run,
    not previously covered by any fixture-based test, all of which use
    <=5-entry lists that _trim() never touches). Reads `full_evidence`
    (untrimmed, added in G7) when present; falls back to the old
    tool_calls-based path only for ledgers that predate that key, so
    every existing hand-built test fixture keeps working unchanged.
    isinstance(entry, dict) is a defensive guard in both paths so a
    similarly-shaped collapsed dict can never crash this function again.
    """
    full_evidence = ledger.get("full_evidence")
    if full_evidence is not None:
        cross_card = full_evidence.get("cross_card_fraud_verification") or {}
        for path in ("via_device", "via_region"):
            path_result = cross_card.get(path)
            if path_result:
                for entry in path_result.get("other_cards", []):
                    if isinstance(entry, dict) and entry.get("has_confirmed_fraud"):
                        return True
        return False

    for tc in ledger.get("tool_calls", []):
        if tc.get("tool") == "cross_card_fraud_verification" and tc.get("success"):
            result = tc.get("curated_result_summary") or {}
            for path in ("via_device", "via_region"):
                path_result = result.get(path)
                if path_result:
                    for entry in path_result.get("other_cards", []):
                        if isinstance(entry, dict) and entry.get("has_confirmed_fraud"):
                            return True
    return False


def _r5_evidence_present(ledger: dict) -> bool:
    """R5 (organizer Fraud Policy): "Three or more small online
    authorizations on one card within an hour, followed by a larger
    purchase." Sufficient ONLY when G1's `pattern_evidence` field (from
    src.investigation.patterns.classify_card_testing -- pure local
    analysis after combined_evidence, zero tool calls) reports
    matched == True. Does not rank, score, or select among multiple
    candidates G5-A may have found; any match is enough, since R5's base
    clause ("recommend DECLINE_TRANSACTION and STEP_UP_AUTH") attaches no
    further condition beyond the sequence existing. Never reads
    risk_score. Never infers a dollar threshold -- classify_card_testing
    itself never asserts one, and this function does not either.
    """
    pattern_evidence = ledger.get("pattern_evidence") or {}
    return bool(pattern_evidence.get("matched"))


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

    # Existing §3a customer-dispute path -- condition and reason text
    # unchanged from the approved Revision 2 spec.
    if status == "known" and trigger_type == "customer_report":
        return RECOMMENDED, "auto", "§3a: customer disputes a charge (trigger_type == 'customer_report')"

    # Additional, independent R6 path (G4): does not replace or alter the
    # customer_report check above, which is always evaluated first.
    if _r6_evidence_present(ledger):
        return RECOMMENDED, "auto", "R6: shared non-hub device/region evidence shows independently confirmed fraud on another card"

    if status == "known":
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


def _evaluate_monitor_connected_cards(ledger: dict) -> tuple[str, str | None, str]:
    if _r6_evidence_present(ledger):
        return RECOMMENDED, "auto", "R6: shared non-hub device/region evidence shows independently confirmed fraud on another card"
    return (
        DEFERRED,
        None,
        "R6 requires confirmed fraud on another card sharing a named element (device profile, billing region, or recipient email); no such evidence present",
    )


def _evaluate_file_report(ledger: dict) -> tuple[str, str | None, str]:
    if _r6_evidence_present(ledger):
        return RECOMMENDED, "L2", "R6: shared non-hub device/region evidence shows independently confirmed fraud on another card"
    return (
        DEFERRED,
        None,
        "R2/R6/R9 require a customer response, cross-card fraud verification for a named shared element, or pattern judgment; only the R6 check is currently available and its evidence is absent here",
    )


def _evaluate_decline_transaction(ledger: dict) -> tuple[str, str | None, str]:
    if _r5_evidence_present(ledger):
        return RECOMMENDED, "L1", "R5: three or more small online authorizations within an hour, followed by a larger purchase"
    return (
        DEFERRED,
        None,
        "R1/R4/R5 require fraud_probability, a customer-reply state, or a matched card-testing sequence; none available",
    )


def _evaluate_step_up_auth(ledger: dict) -> tuple[str, str | None, str]:
    if _r5_evidence_present(ledger):
        return RECOMMENDED, "auto", "R5: three or more small online authorizations within an hour, followed by a larger purchase"
    return (
        DEFERRED,
        None,
        "R1/R5 require fraud_probability or a matched card-testing sequence; none available",
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
        elif action == "MONITOR_CONNECTED_CARDS":
            state, route, reason = _evaluate_monitor_connected_cards(ledger)
        elif action == "FILE_REPORT":
            state, route, reason = _evaluate_file_report(ledger)
        elif action == "DECLINE_TRANSACTION":
            state, route, reason = _evaluate_decline_transaction(ledger)
        elif action == "STEP_UP_AUTH":
            state, route, reason = _evaluate_step_up_auth(ledger)
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
