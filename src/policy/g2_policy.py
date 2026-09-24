"""
Phase G2 -- policy & next-best-action layer, implementing the approved
Revision 2 specification exactly. Pure function over a G1 ledger
(src.agent.workflow.investigate()'s return value): no TigerGraph, no
MCP, no LLM, no risk_score. Every action is traced to a literal condition
from the organizer's official Fraud Policy (D:\\hackathon\\task_5\\README.md)
or classified DEFERRED with the exact missing input named.

This module does not import or call src.agent.workflow -- it only
consumes that function's return value, passed in by the caller.

G11-D: `evaluate()` takes an optional second argument, `validated` --
the already-validated reasoning output (reasoning_validator.py's
accepted {"ok": True, "verdict", "fraud_probability", "pattern",
"exposure_usd", "affected_txn_ids", ...} dict). When omitted (the
default, `None`), every rule below behaves EXACTLY as it always has --
this preserves the pre-reasoning ("initial") G2 pass byte-for-byte, used
by src.agent.case_output.investigate_and_assemble to build the LLM's
evidence package before any reasoning output exists. When `validated` is
supplied (the "final", post-reasoning pass, called only after
validation succeeds), a small, explicitly-listed set of rules that the
organizer's policy ties to verdict/fraud_probability/pattern/exposure_usd
gain an additional, deterministic check -- see _evaluate_create_case,
_evaluate_file_report, _evaluate_escalate_to_analyst, _evaluate_block_card.
No rule here ever reads raw LLM text or makes a judgment call itself:
every check is a plain comparison against already-validated, already-
grounded numbers/enums, exactly like every other rule in this module.

G12: `evaluate()` also takes an optional third argument,
`customer_response` -- one of "confirmed" / "denied" / "no_reply", or
`None` (the default). This is never real customer input: the organizer
README states plainly that customer/analyst replies are not provided in
this round and must be simulated (see src.agent.evidence_loop, the one
caller that ever passes a non-None value). When omitted, every rule
below behaves EXACTLY as before this change -- `evaluate(ledger)` and
`evaluate(ledger, validated)` remain byte-for-byte unaffected. When
supplied, R2 ("customer denies"), R3 ("customer confirms"), and R4
("no reply within 24 hours") become checkable for the first time --
see _evaluate_close_no_fraud, _evaluate_monitor_card, the new branches
in _evaluate_create_case/_evaluate_block_card/_evaluate_file_report/
_evaluate_decline_transaction/_evaluate_escalate_to_analyst.

G13 (SAR audit fix): _evaluate_file_report's R6 and R9 branches now
additionally require validated["verdict"] == "fraud" -- organizer §3a's
own "file when fraud is confirmed or strongly suspected" precondition,
which neither branch previously checked (see those functions'
docstrings). R6's reason text also now names the actual shared device
profile / billing region value (see _r6_shared_element_text) instead of
the generic "device/region" text. R2's FILE_REPORT branch is unchanged.
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

# Every one of these actions has no literal, satisfiable condition given
# ONLY the G1 ledger -- each reason names the specific missing input, per
# the approved spec's decision-rule table. This is the exact fallback
# text used for the pre-reasoning pass (validated=None) for every action
# below, and remains the FINAL determination even post-reasoning for the
# actions whose missing input is still not one of verdict/
# fraud_probability/pattern/exposure_usd/affected_txn_ids (a customer
# response, a no-reply state, or a recurring-pattern match -- none of
# which this project has ever gathered).
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
# unconditionally DEFERRED.
#
# G11-D: BLOCK_CARD and ESCALATE_TO_ANALYST also moved to dedicated
# evaluators below (_evaluate_block_card, _evaluate_escalate_to_analyst)
# -- their entries in this dict remain as the exact fallback text used
# whenever `validated` is None (preserving the pre-reasoning pass
# byte-for-byte) or when validated is present but their specific
# condition still doesn't hold.


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


def _r6_qualifying_paths(ledger: dict) -> set[str]:
    """Same evidence _r6_evidence_present reads, but returns WHICH of
    {"via_device", "via_region"} individually satisfy R6, rather than a
    single boolean -- used only to decide which shared-element value(s)
    to name in the R6 FILE_REPORT reason (see _r6_shared_element_text).
    Mirrors _r6_evidence_present's exact dual-path (full_evidence /
    tool_calls fallback) structure; reads no new evidence.
    """
    paths: set[str] = set()
    full_evidence = ledger.get("full_evidence")
    if full_evidence is not None:
        cross_card = full_evidence.get("cross_card_fraud_verification") or {}
        for path in ("via_device", "via_region"):
            path_result = cross_card.get(path)
            if path_result and any(
                isinstance(entry, dict) and entry.get("has_confirmed_fraud")
                for entry in path_result.get("other_cards", [])
            ):
                paths.add(path)
        return paths

    for tc in ledger.get("tool_calls", []):
        if tc.get("tool") == "cross_card_fraud_verification" and tc.get("success"):
            result = tc.get("curated_result_summary") or {}
            for path in ("via_device", "via_region"):
                path_result = result.get(path)
                if path_result and any(
                    isinstance(entry, dict) and entry.get("has_confirmed_fraud")
                    for entry in path_result.get("other_cards", [])
                ):
                    paths.add(path)
    return paths


def _r6_shared_element_text(ledger: dict) -> str:
    """Deterministic, concise naming of the actual shared element(s) R6
    fired on -- for the FILE_REPORT reason only (README R6: "name the
    shared element"). Q9's own result never carries the element's VALUE
    itself, only which OTHER cards share it (see
    src.investigation.queries._extract_cross_card_results) -- the value
    lives in this investigation's own full_evidence.combined_evidence
    (the flagged transaction's own device_profile / billing region),
    read here, never invented. A qualifying path whose value isn't
    available in this ledger shape (e.g. a pre-G7 fixture with no
    full_evidence.combined_evidence) falls back to a plain, honest label
    rather than fabricating one.
    """
    paths = _r6_qualifying_paths(ledger)
    if not paths:
        return "device/region"

    combined = (ledger.get("full_evidence") or {}).get("combined_evidence") or {}
    parts: list[str] = []
    if "via_device" in paths:
        device_profile = (combined.get("device_evidence") or {}).get("device_profile")
        parts.append(f"device profile '{device_profile}'" if device_profile else "device profile")
    if "via_region" in paths:
        region = (combined.get("billing_region") or {}).get("addr1")
        parts.append(f"billing region {region}" if region is not None else "billing region")
    return " and ".join(parts)


def _r9_cross_customer_evidence_present(ledger: dict) -> bool:
    """R9's "coordinated or repeated abuse across customers" clause is
    strictly narrower than R6's own evidence: R6 fires on "several cards"
    sharing an element (which src.investigation.queries.py's Q9 result
    never requires to belong to different customers -- see
    _extract_cross_card_results's `same_customer` field), but R9 names
    "across customers" explicitly. Reuses the EXACT same Q9
    cross_card_fraud_verification data _r6_evidence_present reads --
    gathers no new evidence, adds no new tool call -- and additionally
    requires at least one has_confirmed_fraud entry with
    same_customer == False (a field Q9 already computes and _r6_
    evidence_present already ignores). Mirrors _r6_evidence_present's
    exact full_evidence/tool_calls dual-path structure for consistency.
    """
    full_evidence = ledger.get("full_evidence")
    if full_evidence is not None:
        cross_card = full_evidence.get("cross_card_fraud_verification") or {}
        for path in ("via_device", "via_region"):
            path_result = cross_card.get(path)
            if path_result:
                for entry in path_result.get("other_cards", []):
                    if isinstance(entry, dict) and entry.get("has_confirmed_fraud") and entry.get("same_customer") is False:
                        return True
        return False

    for tc in ledger.get("tool_calls", []):
        if tc.get("tool") == "cross_card_fraud_verification" and tc.get("success"):
            result = tc.get("curated_result_summary") or {}
            for path in ("via_device", "via_region"):
                path_result = result.get(path)
                if path_result:
                    for entry in path_result.get("other_cards", []):
                        if isinstance(entry, dict) and entry.get("has_confirmed_fraud") and entry.get("same_customer") is False:
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


def _evaluate_create_case(
    ledger: dict, validated: dict | None = None, customer_response: str | None = None,
) -> tuple[str, str | None, str]:
    trigger_type, status = _get_trigger_type(ledger)

    # Existing §3a customer-dispute path -- condition and reason text
    # unchanged from the approved Revision 2 spec.
    if status == "known" and trigger_type == "customer_report":
        return RECOMMENDED, "auto", "§3a: customer disputes a charge (trigger_type == 'customer_report')"

    # Additional, independent R6 path (G4): does not replace or alter the
    # customer_report check above, which is always evaluated first.
    if _r6_evidence_present(ledger):
        return RECOMMENDED, "auto", "R6: shared non-hub device/region evidence shows independently confirmed fraud on another card"

    # G11-D: §3a's own third, explicit trigger ("Open one whenever fraud
    # probability reaches 0.30") -- only checkable post-reasoning, since
    # fraud_probability does not exist in the G1 ledger. validated=None
    # (the pre-reasoning pass) always skips this, unchanged from before.
    if validated is not None and validated.get("fraud_probability") is not None and validated["fraud_probability"] >= 0.30:
        return (
            RECOMMENDED, "auto",
            f"§3a: fraud probability {validated['fraud_probability']} reaches the 0.30 case-opening threshold",
        )

    # G12: R2's own CREATE_CASE trigger ("Customer denies the transaction
    # -> recommend BLOCK_CARD and CREATE_CASE") -- only reachable when
    # src.agent.evidence_loop has simulated a "denied" response;
    # customer_response is None for every pre-loop call, so this is
    # unreachable before this change, exactly like the §3a check above.
    if customer_response == "denied":
        return RECOMMENDED, "auto", "R2: customer denies the transaction (simulated response)"

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


def _evaluate_file_report(
    ledger: dict, validated: dict | None = None, customer_response: str | None = None,
) -> tuple[str, str | None, str]:
    # G11-D: R9's own path is checked FIRST, and only when `validated` is
    # present -- "activity fits none of the known patterns but the
    # evidence shows coordinated or repeated abuse across customers."
    # R9's evidence (_r9_cross_customer_evidence_present) is a STRICT
    # SUBSET of R6's (same has_confirmed_fraud check, plus same_customer
    # is False), so it is always also valid R6 evidence; checking it
    # first lets the more specific, more accurate citation win whenever
    # `pattern` is genuinely "undocumented", without changing the result
    # for any ledger that doesn't also satisfy this narrower condition
    # (those still fall through to the unchanged R6 check below).
    # `pattern` does not exist pre-reasoning, so validated=None always
    # skips straight to the R6 check, unchanged from before this change.
    #
    # G13 (SAR audit fix): BOTH the R9 and R6 branches below now
    # additionally require validated["verdict"] == "fraud" -- organizer
    # §3a: "File one when fraud is confirmed or strongly suspected..."
    # Before this change, neither branch read the CURRENT case's own
    # verdict at all, so a case this project's own reasoning layer judged
    # "legitimate" could still trigger FILE_REPORT purely because some
    # OTHER card shared a device/region with confirmed fraud elsewhere --
    # treating another customer's confirmed fraud as sufficient by
    # itself, never checking whether THIS case is even suspected. Uses
    # only the existing validated["verdict"] enum (no new probability
    # threshold invented) -- "confirmed" maps directly to this project's
    # own "fraud" verdict; a "strongly suspected" case falls under R8/R9's
    # own escalation paths instead, not a new number here. `validated is
    # None` (pre-reasoning) always fails this check, same as every other
    # validated-gated rule in this module -- FILE_REPORT's R6/R9 paths
    # are no longer available pre-reasoning even when Q9 evidence exists,
    # since this case's own verdict genuinely isn't known yet.
    if (
        validated is not None
        and validated.get("verdict") == "fraud"
        and validated.get("pattern") == "undocumented"
        and _r9_cross_customer_evidence_present(ledger)
    ):
        return (
            RECOMMENDED, "L2",
            "R9: undocumented pattern with confirmed fraud on another customer's card sharing a named element (device/region)",
        )

    if validated is not None and validated.get("verdict") == "fraud" and _r6_evidence_present(ledger):
        # G13: names the actual shared element (README R6: "name the
        # shared element") -- read from this investigation's own
        # full_evidence, never from Q9 itself (Q9 never returns the
        # element's value, only which other cards share it).
        shared = _r6_shared_element_text(ledger)
        return RECOMMENDED, "L2", f"R6: shared {shared} shows independently confirmed fraud on another card"

    # G12: R2's own FILE_REPORT clause -- "Add FILE_REPORT if exposure
    # exceeds $1,000 or the case connects to a shared device profile or
    # another card's fraud." The second half of that clause is exactly
    # the R6 check above (already evaluated first); only the exposure
    # sub-condition is new here, and only reachable once
    # src.agent.evidence_loop has simulated a "denied" response. R2's own
    # policy text has no "confirmed/strongly suspected" precondition of
    # its own (a customer denial is itself the triggering evidence), so
    # this branch is intentionally NOT gated on validated["verdict"] --
    # unchanged from before G13.
    if customer_response == "denied" and validated is not None and validated.get("exposure_usd") is not None:
        exposure = validated["exposure_usd"]
        if exposure > 1000:
            return RECOMMENDED, "L2", f"R2: customer denies the transaction (simulated response); exposure ${exposure} exceeds $1,000"

    return (
        DEFERRED,
        None,
        "R2/R6/R9 require a customer response, a confirmed-fraud verdict on this case together with cross-card fraud "
        "verification for a named shared element, or pattern judgment; none of these currently hold",
    )


def _evaluate_decline_transaction(ledger: dict, customer_response: str | None = None) -> tuple[str, str | None, str]:
    if _r5_evidence_present(ledger):
        return RECOMMENDED, "L1", "R5: three or more small online authorizations within an hour, followed by a larger purchase"
    # G12: R4's own DECLINE_TRANSACTION clause ("No reply within 24 hours
    # -> recommend MONITOR_CARD and DECLINE_TRANSACTION for pending
    # authorizations") -- only reachable once src.agent.evidence_loop has
    # simulated a "no_reply" response.
    if customer_response == "no_reply":
        return RECOMMENDED, "L1", "R4: no reply within 24 hours (simulated benchmark response); decline pending authorizations"
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


def _evaluate_escalate_to_analyst(
    ledger: dict, validated: dict | None = None, customer_response: str | None = None,
) -> tuple[str, str | None, str]:
    """R8: "If the verdict is uncertain and exposure exceeds $500, or the
    evidence conflicts, recommend ESCALATE_TO_ANALYST." Both clauses are
    gated behind `validated is not None`, even the "evidence conflicts"
    alternative (whose `ledger.uncertainties` input technically exists
    pre-reasoning too) -- so the pre-reasoning pass (validated=None) is
    unconditionally DEFERRED, byte-for-byte identical to before this
    change, and ESCALATE_TO_ANALYST is only ever evaluated post-
    reasoning, per the required architecture.

    G12: R4's own escalation clause ("Escalate if exposure exceeds $500")
    is checked FIRST, ahead of R8's "uncertain verdict" clause, whenever
    src.agent.evidence_loop has simulated a "no_reply" response --
    src.agent.evidence_loop only ever simulates a response when
    validated["verdict"] == "uncertain" in the first place (its own
    R1 gate), so R8's "verdict == 'uncertain' and exposure > 500" clause
    would otherwise always be true first and silently shadow R4's more
    specific, actually-applicable citation for every real no-reply case.
    Checking R4 first fixes that mis-attribution without changing the
    recommended ACTION either way (both cite ESCALATE_TO_ANALYST) and
    without touching R8's own behavior when customer_response is None
    (the check below is a no-op in that case) or when R4's own condition
    doesn't hold (exposure <= $500, or customer_response != "no_reply") --
    R8's two original clauses still run exactly as before.
    """
    if validated is not None:
        verdict = validated.get("verdict")
        exposure = validated.get("exposure_usd")
        if customer_response == "no_reply" and exposure is not None and exposure > 500:
            return (
                RECOMMENDED, "auto",
                f"R4: no reply within 24 hours (simulated benchmark response); exposure ${exposure} exceeds $500",
            )
        if verdict == "uncertain" and exposure is not None and exposure > 500:
            return RECOMMENDED, "auto", f"R8: verdict 'uncertain' with exposure ${exposure} exceeding $500"
        if ledger.get("uncertainties"):
            return RECOMMENDED, "auto", "R8: evidence conflicts (ledger uncertainties present)"
    return DEFERRED, None, _DEFERRED_REASONS["ESCALATE_TO_ANALYST"]


def _evaluate_block_card(
    ledger: dict, validated: dict | None = None, customer_response: str | None = None,
) -> tuple[str, str | None, str]:
    """R5: "If a purchase over $100 has already cleared, recommend
    BLOCK_CARD." The $100+ cleared-purchase condition is checkable from
    G1's own pattern_evidence (src.investigation.patterns.
    classify_card_testing already reports each candidate's
    larger_purchase_amount -- no new evidence needed), but the route
    (L1 vs L2, README section 2: L1 when exposure <= $2,500, L2
    otherwise) can only be set once exposure_usd exists, i.e. post-
    reasoning. When validated is None, or the R5 condition isn't met,
    falls through to the R2 check below (G12), then to the exact
    original DEFERRED text -- unchanged from before this change.
    """
    pattern_evidence = ledger.get("pattern_evidence") or {}
    if pattern_evidence.get("matched"):
        cleared_amount = next(
            (
                c.get("larger_purchase_amount")
                for c in (pattern_evidence.get("candidates") or [])
                if isinstance(c, dict) and c.get("larger_purchase_amount") is not None and c["larger_purchase_amount"] > 100
            ),
            None,
        )
        if cleared_amount is not None and validated is not None and validated.get("exposure_usd") is not None:
            exposure = validated["exposure_usd"]
            route = "L1" if exposure <= 2500 else "L2"
            return (
                RECOMMENDED, route,
                f"R5: a ${cleared_amount} purchase has already cleared (over $100); exposure ${exposure} sets the L1/L2 route",
            )

    # G12: R2's own BLOCK_CARD trigger ("Customer denies the transaction
    # -> recommend BLOCK_CARD and CREATE_CASE") -- independent of R5's
    # cleared-purchase condition above; the route still follows the same
    # L1/<=$2,500 vs L2/>$2,500 split from README section 2. Only
    # reachable once src.agent.evidence_loop has simulated a "denied"
    # response.
    if customer_response == "denied" and validated is not None and validated.get("exposure_usd") is not None:
        exposure = validated["exposure_usd"]
        route = "L1" if exposure <= 2500 else "L2"
        return (
            RECOMMENDED, route,
            f"R2: customer denies the transaction (simulated response); exposure ${exposure} sets the L1/L2 route",
        )

    return DEFERRED, None, _DEFERRED_REASONS["BLOCK_CARD"]


def _evaluate_close_no_fraud(customer_response: str | None) -> tuple[str, str | None, str]:
    """R3: "Customer confirms the transaction -> recommend CLOSE_NO_FRAUD."
    Only ever RECOMMENDED when customer_response == "confirmed" -- a
    simulated response produced by src.agent.evidence_loop, since this
    project has no live customer channel (see module docstring, G12).
    customer_response is None for every pre-loop call, so this remains
    DEFERRED with the exact original fallback text whenever no evidence
    response has been simulated -- unaffected by this change.
    """
    if customer_response == "confirmed":
        return RECOMMENDED, "auto", "R3: customer confirmed the transaction (simulated response)"
    return DEFERRED, None, _DEFERRED_REASONS["CLOSE_NO_FRAUD"]


def _evaluate_monitor_card(customer_response: str | None) -> tuple[str, str | None, str]:
    """R4: "No reply within 24 hours -> recommend MONITOR_CARD and
    DECLINE_TRANSACTION." (DECLINE_TRANSACTION's half of this rule lives
    in _evaluate_decline_transaction.) Only ever RECOMMENDED when
    customer_response == "no_reply" -- see _evaluate_close_no_fraud's
    docstring for why this is gated the same way.
    """
    if customer_response == "no_reply":
        return RECOMMENDED, "auto", "R4: no reply within 24 hours (simulated benchmark response)"
    return DEFERRED, None, _DEFERRED_REASONS["MONITOR_CARD"]


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


def evaluate(ledger: dict, validated: dict | None = None, customer_response: str | None = None) -> dict:
    """Pure function: G1 ledger (+ optional validated reasoning output,
    + optional simulated customer_response) -> G2 policy decision.

    Every one of the 14 official actions is evaluated exactly once, in
    CANONICAL_ACTION_ORDER, and placed into exactly one of the four
    policy-state buckets -- each bucket list is built by iterating the
    fixed CANONICAL_ACTION_ORDER tuple (never a dict/set), so repeated
    calls on the same ledger (and the same validated/customer_response,
    or their absence) always produce identical output, including order.

    G11-D: `validated` defaults to None. `evaluate(ledger)` -- the exact
    call every existing caller and test already makes -- is completely
    unaffected: every evaluator below either ignores `validated` or
    explicitly checks `validated is not None` before using it, so the
    pre-reasoning ("initial") pass is byte-for-byte identical to before
    this change. Passing the validated reasoning output produces the
    post-reasoning ("final") pass, used only after reasoning succeeds.

    G12: `customer_response` defaults to None. `evaluate(ledger)` and
    `evaluate(ledger, validated)` are unaffected by this parameter's
    existence -- every evaluator that reads it only acts when it is
    explicitly one of "confirmed"/"denied"/"no_reply", never on `None`.
    Only src.agent.evidence_loop ever passes a non-None value, and only
    after simulating a response (this project has no live customer or
    analyst channel).
    """
    recommended: list[dict] = []
    prohibited: list[dict] = []
    eligible_not_recommended: list[dict] = []
    deferred: list[dict] = []

    for action in CANONICAL_ACTION_ORDER:
        if action == "CREATE_CASE":
            state, route, reason = _evaluate_create_case(ledger, validated, customer_response)
        elif action == "BLOCK_ALL_CARDS":
            state, route, reason = _evaluate_block_all_cards(ledger)
        elif action == "MONITOR_CONNECTED_CARDS":
            state, route, reason = _evaluate_monitor_connected_cards(ledger)
        elif action == "MONITOR_CARD":
            state, route, reason = _evaluate_monitor_card(customer_response)
        elif action == "FILE_REPORT":
            state, route, reason = _evaluate_file_report(ledger, validated, customer_response)
        elif action == "DECLINE_TRANSACTION":
            state, route, reason = _evaluate_decline_transaction(ledger, customer_response)
        elif action == "STEP_UP_AUTH":
            state, route, reason = _evaluate_step_up_auth(ledger)
        elif action == "BLOCK_CARD":
            state, route, reason = _evaluate_block_card(ledger, validated, customer_response)
        elif action == "ESCALATE_TO_ANALYST":
            state, route, reason = _evaluate_escalate_to_analyst(ledger, validated, customer_response)
        elif action == "CLOSE_NO_FRAUD":
            state, route, reason = _evaluate_close_no_fraud(customer_response)
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
