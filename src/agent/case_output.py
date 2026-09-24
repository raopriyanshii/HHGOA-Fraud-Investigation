"""
G7 -- output assembly (isolated adapter).

Combines the G1 ledger, the G2 policy result, and the validated LLM
reasoning output into the organizer-required `case` object and
`next_best_actions` object. It only assembles already-computed,
already-validated pieces, and it is the one place these three sources
are combined: G1, G2, and the reasoning layer never see each other's
output shape.

G14: this module now makes AT MOST ONE additional MCP tool call and AT
MOST ONE additional LLM call, and only when the caller opts in by
passing `call_tool` to investigate_and_assemble AND round 1's own
validated output sets needs_more_evidence=true -- see
_run_evidence_request_round. When `call_tool` is omitted (every
pre-G14 caller and test), behavior is byte-for-byte the original
zero-MCP-call, single-LLM-call guarantee, unconditionally.

Scope: produces `case` (Part 1), `evidence_requests`, `next_best_actions`
(Part 3), and `sar` (Part 2, G13). `next_best_actions.initial` still
equals `next_best_actions.final` whenever nothing changed the
pre-reasoning G2 pass's own recommendations -- matching the organizer's
own rule "If you requested nothing, final equals initial" -- but they
now genuinely CAN differ for two independent reasons: the validated
verdict/fraud_probability/pattern/exposure_usd unlocking a rule (R8,
§3a, R9, or R5's BLOCK_CARD route) the pre-reasoning ledger alone could
not satisfy, and/or the G12 evidence-request loop (see
src.agent.evidence_loop) simulating a response that unlocks R2/R3/R4.

G13: `sar` (README "Answer Format", Part 2) is built here, deterministically,
from data this module already has -- no new graph/LLM call. `sar.file`/
`sar.reason` are read directly from the FILE_REPORT entry in
`next_best_actions.final`, if any (must agree with whether FILE_REPORT
appears there, per the organizer's own field definition). When it does,
`total_amount_usd` is `validated["exposure_usd"]` (the reasoning
validator's own deterministic sum -- never recalculated here),
`activity_dates` is derived only from the timestamps `ledger.
full_evidence.combined_evidence` already carries for
`validated["affected_txn_ids"]`, `subjects` from entity IDs already
present in that same evidence, and `narrative` is a deterministic,
templated sentence sequence built only from already-validated/grounded
fields -- never a second LLM call, never an invented fact. When
FILE_REPORT is absent, `sar` is the organizer's exact required empty
shape (`narrative=""`, `subjects=[]`, `total_amount_usd=0`,
`activity_dates=[]`). Does NOT produce `stop_reason` (beyond the
failure-path one below), `tool_calls`, `tokens`, `latency_s`,
`connected_card_ids`, `connected_device_profiles`, `written_to_graph`,
or `graph_case_id` -- these remain later-phase work, not invented here.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable

from src.agent import evidence_loop
from src.agent.reasoning import DEFAULT_LLM_TIMEOUT_S, CallLLM, build_evidence_package, get_llm_reasoning
from src.agent.reasoning_validator import validate_reasoning_output
from src.graph.tigergraph_connection import scrub_secret
from src.policy import g2_policy

# G14 -- same structural shape as src.agent.workflow.CallTool /
# src.agent.case_writer.CallTool, redefined locally rather than imported
# cross-module (the established convention in this project -- G1, G2,
# and the reasoning/output layer never import each other's internals).
CallTool = Callable[[str, dict], Awaitable[tuple[bool, Any, str | None]]]

# G14: the one registered evidence type this phase supports, and the one
# real MCP tool it maps to -- never an LLM-chosen tool name, never
# arbitrary GSQL. Extending this dict (never inferring a mapping) is how
# a future evidence type would be added.
_EVIDENCE_TYPE_TOOL_NAMES = {
    "historical_case_evidence": "historical_case_evidence",
}


def _status_for(verdict: str, g1_recommendation: str | None) -> str:
    if verdict == "fraud":
        return "closed_fraud"
    if verdict == "legitimate":
        return "closed_legitimate"
    if g1_recommendation == "escalate":
        return "escalated"
    return "open"


def _similar_prior_cases(ledger: dict) -> list[str]:
    """Deterministic only -- never asks the LLM to select or invent case
    IDs (see project's own G6 field analysis: this is the one field
    where a fixed rule over already-retrieved Q4 evidence is preferable
    to an LLM judgment call). Cites every retrieved historical case
    (direct or connected) the closed-case record itself marks
    confirmed_fraud.
    """
    combined = (ledger.get("full_evidence") or {}).get("combined_evidence") or {}
    historical = combined.get("historical_cases") or {}
    case_ids: list[str] = []
    for c in (historical.get("direct_cases") or []) + (historical.get("connected_cases") or []):
        if c.get("outcome") == "confirmed_fraud" and c.get("case_id") not in case_ids:
            case_ids.append(c["case_id"])
    return case_ids


def _actions_from_g2_result(g2_result: dict) -> list[dict]:
    return [
        {"action": e["action"], "route": e.get("route"), "reason": e["reason"]}
        for e in g2_result.get("recommended", [])
    ]


def _describe_action_diff(initial: list[dict], final: list[dict]) -> str:
    """Deterministic, from the actual action-set difference only -- never
    a generic/hardcoded claim. `initial`/`final` are both already in
    CANONICAL_ACTION_ORDER (g2_policy.evaluate builds `recommended` by
    iterating that fixed tuple), so iterating each list in order and
    comparing against the other's action-name set preserves that order
    with no additional sort needed.
    """
    initial_actions = {a["action"] for a in initial}
    final_actions = {a["action"] for a in final}
    added = [a["action"] for a in final if a["action"] not in initial_actions]
    removed = [a["action"] for a in initial if a["action"] not in final_actions]
    if not added and not removed:
        return "nothing"
    parts = []
    if added:
        parts.append(f"added {', '.join(added)}")
    if removed:
        parts.append(f"removed {', '.join(removed)}")
    return "; ".join(parts) + " once the validated reasoning output (verdict/fraud_probability/pattern/exposure_usd) was available"


def _next_best_actions(g2_result: dict, g2_result_final: dict | None = None) -> dict:
    """Entirely deterministic, entirely from G2's own `recommended`
    bucket -- G2's `prohibited`/`deferred` buckets are never read here,
    which is what structurally guarantees a PROHIBITED action (e.g.
    BLOCK_ALL_CARDS) can never appear as a recommended next-best-action:
    there is no code path from `prohibited` to this output at all. The
    LLM's raw output is never consulted here directly -- only through
    already-validated, already-grounded fields (verdict/fraud_probability/
    pattern/exposure_usd) that g2_policy.evaluate treats as plain,
    deterministic comparisons, exactly like every other rule it applies.

    `g2_result` is always the pre-reasoning ("initial") G2 pass.
    `g2_result_final`, when given, is the post-reasoning ("final") pass
    (g2_policy.evaluate(ledger, validated)) -- see investigate_and_
    assemble below. When omitted, `final` equals `initial` exactly as
    before this phase (used only by callers/tests that never run a
    second G2 pass).
    """
    initial = _actions_from_g2_result(g2_result)
    if g2_result_final is None:
        return {"initial": initial, "final": initial, "what_changed": "nothing"}
    final = _actions_from_g2_result(g2_result_final)
    return {"initial": initial, "final": final, "what_changed": _describe_action_diff(initial, final)}


def _evidence_delta(g2_result_final: dict, g2_result_after_response: dict) -> dict:
    """Internal-only audit field (not part of the organizer's required
    schema): isolates exactly what the G12 evidence-request/simulated-
    response pass itself changed, separately from next_best_actions'
    own initial/final (which also includes G11-D's reasoning-availability
    delta, a different, earlier cause). `pre_evidence_actions` is
    g2_result_final's own `recommended` -- the post-reasoning, PRE-
    evidence-response G2 pass computed in investigate_and_assemble below
    -- never the original pre-reasoning `g2_result` and never
    next_best_actions["initial"]. `post_evidence_actions` is the SAME
    g2_result the evidence loop actually produced
    (evidence_outcome["g2_result"]). `added_by_evidence_response` is a
    deterministic set-difference, ordered exactly as `post_evidence_
    actions` already is (CANONICAL_ACTION_ORDER, via g2_policy.evaluate)
    -- never re-sorted.
    """
    pre = _actions_from_g2_result(g2_result_final)
    post = _actions_from_g2_result(g2_result_after_response)
    pre_actions = {a["action"] for a in pre}
    added = [a for a in post if a["action"] not in pre_actions]
    return {
        "pre_evidence_actions": pre,
        "post_evidence_actions": post,
        "added_by_evidence_response": added,
    }


def _historical_case_ids(cases: list, outcome: str = "confirmed_fraud") -> list[str]:
    return sorted({
        c["case_id"] for c in cases
        if isinstance(c, dict) and c.get("outcome") == outcome and c.get("case_id")
    })


def _sibling_confirmed_fraud_case_ids(ledger: dict, card_key: str) -> list[str]:
    """The Round-2 historical_case_evidence result for this specific
    card is never carried into full_evidence (see workflow.py) -- only
    the ledger's own tool_calls list has it, and that list's
    curated_result_summary already went through workflow.py's _trim(),
    so a card with more than 5 direct/connected cases arrives here as
    {"count": N, "sample": [...]} rather than a plain list. Reads
    whichever shape is actually present and cites only what it contains
    -- never invents an id beyond what was actually returned.
    """
    for tc in ledger.get("tool_calls", []):
        if tc.get("tool") == "historical_case_evidence" and (tc.get("arguments") or {}).get("card_key") == card_key:
            result = tc.get("curated_result_summary") or {}
            ids: set[str] = set()
            for key in ("direct_cases", "connected_cases"):
                cases = result.get(key)
                if isinstance(cases, dict):  # _trim()-collapsed {count, sample}
                    cases = cases.get("sample")
                for c in cases or []:
                    if isinstance(c, dict) and c.get("outcome") == "confirmed_fraud" and c.get("case_id"):
                        ids.add(c["case_id"])
            return sorted(ids)
    return []


def _recommendation_basis_evidence(ledger: dict) -> list[dict]:
    """G1's own recommendation_basis.reasons (src.agent.workflow._assess's
    escalate/monitor determination) is entirely deterministic Python --
    computed before this function is ever called, never LLM output.
    Templated exactly like the existing R5/R6 entries below: the LLM
    never controls whether these entries exist or what they cite.
    Historical case IDs are read only from data the ledger already
    carries (full_evidence.combined_evidence.historical_cases for the
    flagged customer's own history; the ledger's tool_calls for a
    sibling card's Round-2 result) -- nothing here performs a new graph
    query, and no id is ever invented.
    """
    reasons = (ledger.get("recommendation_basis") or {}).get("reasons") or []
    if not reasons:
        return []

    full_evidence = ledger.get("full_evidence") or {}
    combined = full_evidence.get("combined_evidence") or {}
    historical = combined.get("historical_cases") or {}
    device = combined.get("device_evidence") or {}

    evidence: list[dict] = []
    seen_claims: set[str] = set()

    def _add(claim: str, entity_ids: list[str]) -> None:
        if claim in seen_claims:
            return  # requirement 6: no duplicate entries for the same underlying evidence
        seen_claims.add(claim)
        evidence.append({"claim": claim, "source": "graph", "ref": "g1:recommendation_basis", "entity_ids": entity_ids})

    for reason in reasons:
        if reason == "direct_confirmed_fraud":
            _add(
                "G1 escalation basis: the customer's own historical case record includes at least one confirmed-fraud case",
                _historical_case_ids(historical.get("direct_cases") or []),
            )
        elif reason == "connected_confirmed_fraud":
            _add(
                "G1 escalation basis: a case connected to this customer's history includes at least one confirmed-fraud case",
                _historical_case_ids(historical.get("connected_cases") or []),
            )
        elif reason.startswith("sibling_confirmed_fraud:"):
            card_key = reason.split(":", 1)[1]
            _add(
                f"G1 escalation basis: card {card_key}, connected via shared closed-case history, has at least one confirmed-fraud case",
                _sibling_confirmed_fraud_case_ids(ledger, card_key),
            )
        elif reason == "narrow_device_evidence":
            device_profile = device.get("device_profile")
            _add(
                "G1 monitoring basis: the flagged transaction's device profile is not shared/hub (narrow device evidence)",
                [device_profile] if device_profile else [],
            )

    return evidence


def _q9_r6_entity_ids(ledger: dict) -> list[str]:
    """Real entity_ids for the R6 evidence entry below, read from Q9's
    own untrimmed result (ledger["full_evidence"]["cross_card_fraud_
    verification"] -- see workflow.py, never _trim()-collapsed there).
    Both card_key and confirmed_fraud_case_ids are included for every
    other_cards entry with has_confirmed_fraud == True, from both
    via_device and via_region -- both are real graph entities/records
    Q9 already returned, never invented, never customer_id. Deduplicated
    and sorted for a deterministic result. Handles the {"count": N,
    "sample": [...]} trimmed shape defensively (matching every other
    other_cards reader in this project, e.g. g2_policy._r6_evidence_
    present) even though this particular value is not expected to be
    trimmed; returns [] for anything missing, None, or malformed --
    never raises, never performs a graph query.
    """
    cross_card = (ledger.get("full_evidence") or {}).get("cross_card_fraud_verification")
    if not isinstance(cross_card, dict):
        return []

    ids: set[str] = set()
    for path in ("via_device", "via_region"):
        path_result = cross_card.get(path)
        if not isinstance(path_result, dict):
            continue
        other_cards = path_result.get("other_cards")
        if isinstance(other_cards, dict):  # _trim()-collapsed {count, sample}
            other_cards = other_cards.get("sample")
        for entry in other_cards or []:
            if not (isinstance(entry, dict) and entry.get("has_confirmed_fraud")):
                continue
            if entry.get("card_key"):
                ids.add(entry["card_key"])
            for case_id in entry.get("confirmed_fraud_case_ids") or []:
                ids.add(case_id)

    return sorted(ids)


def _evidence_list(ledger: dict, g2_result: dict) -> list[dict]:
    """Deterministically templated (no LLM), mirroring the organizer's
    `evidence` shape ({claim, source, ref, entity_ids}). Covers the
    deterministic findings this project establishes: R5 (card-testing),
    R6 (cross-card fraud verification), and G1's own recommendation_basis
    reasons (direct/connected/sibling confirmed fraud, narrow device
    evidence) -- does not attempt to restate every raw evidence field as
    a claim.
    """
    evidence: list[dict] = []
    pattern_evidence = ledger.get("pattern_evidence") or {}
    if pattern_evidence.get("matched"):
        candidate_ids = sorted({
            tid
            for c in pattern_evidence.get("candidates", [])
            for tid in (c["authorization_txn_ids"] + [c["larger_purchase_txn_id"]])
        })
        evidence.append({
            "claim": (
                "R5 card-testing shape detected: three or more small online authorizations "
                "within an hour, followed by a strictly larger purchase (see candidates for exact windows)"
            ),
            "source": "graph",
            "ref": "pattern:classify_card_testing",
            "entity_ids": [str(i) for i in candidate_ids],
        })
    for entry in g2_result.get("recommended", []):
        if entry["action"] in ("MONITOR_CONNECTED_CARDS", "FILE_REPORT", "CREATE_CASE") and "R6" in entry["reason"]:
            evidence.append({
                "claim": entry["reason"],
                "source": "graph",
                "ref": "tool:cross_card_fraud_verification",
                "entity_ids": _q9_r6_entity_ids(ledger),
            })
            break  # one R6 citation is enough -- all three actions share the same underlying evidence
    evidence.extend(_recommendation_basis_evidence(ledger))
    return evidence


# --- G13: SAR (Part 2) construction -- deterministic, evidence-grounded,
# no LLM call. Only ever invoked when FILE_REPORT is in next_best_actions
# .final (see _build_sar); the organizer's exact empty shape is used
# otherwise. ---

_EMPTY_SAR = {"file": False, "reason": "", "narrative": "", "subjects": [], "total_amount_usd": 0, "activity_dates": []}


def _sar_ts_by_txn_id(ledger: dict) -> dict[int, str]:
    """txn_id -> full timestamp string, built the same way reasoning_
    validator.py's own amount_by_txn_id map is (full_evidence.combined_
    evidence.transaction + .temporal_activity) -- reused here only for
    activity_dates. No new query, no invented timestamp.
    """
    combined = (ledger.get("full_evidence") or {}).get("combined_evidence") or {}
    ts_by_id: dict[int, str] = {}
    txn = combined.get("transaction")
    if txn and txn.get("TransactionID") is not None and txn.get("ts"):
        ts_by_id[txn["TransactionID"]] = txn["ts"]
    for t in combined.get("temporal_activity") or []:
        if t.get("TransactionID") is not None and t.get("ts"):
            ts_by_id[t["TransactionID"]] = t["ts"]
    return ts_by_id


def _sar_activity_dates(ledger: dict, validated: dict) -> list[str]:
    """README: "First and last date of the activity, YYYY-MM-DD" -- a
    two-element list. Scope is validated["affected_txn_ids"] only. Built
    from the deduplicated, sorted SET of calendar dates found among those
    transactions' own timestamps (never a new query), then reduced to
    [first, last] -- repeating the single date when there is only one,
    matching the organizer's own worked example. [] when no timestamp is
    known for any affected transaction -- never a fabricated date.
    """
    ts_by_id = _sar_ts_by_txn_id(ledger)
    dates = sorted({
        ts_by_id[tid][:10] for tid in (validated.get("affected_txn_ids") or [])
        if tid in ts_by_id and ts_by_id[tid]
    })
    if not dates:
        return []
    return [dates[0], dates[-1]]


def _sar_connected_card_keys(ledger: dict) -> list[str]:
    """Real, confirmed-fraud card_keys from Q9's own result (never a
    closed-case id) -- reuses _q9_r6_entity_ids's already-gathered ids
    and keeps only entries shaped like this project's own established
    card_key convention (customer_id:card1, confirmed since Phase C --
    see src.agent.reasoning's own docstring), since README `subjects`
    names customers/cards/merchants/devices, not case ids.
    """
    return [i for i in _q9_r6_entity_ids(ledger) if ":" in i]


_SAR_MAX_NAMED_CONNECTED_CARDS = 5  # matches _sar_narrative's own connected[:5] slice exactly


def _sar_subjects(ledger: dict, file_report_reason: str) -> list[str]:
    """README: "IDs of the customers, cards, merchants, and devices named
    in the narrative." Only entity IDs already present in
    ledger.full_evidence.combined_evidence (the flagged transaction's own
    customer/card/device) plus, when Q9 evidence actually exists, the
    real confirmed-fraud card_keys it returned -- capped to the same
    _SAR_MAX_NAMED_CONNECTED_CARDS slice _sar_narrative itself names in
    the narrative text (submission-audit fix: subjects must be entities
    the narrative actually names, not every real card Q9 happens to
    return -- a highly-connected card can have hundreds of confirmed-
    fraud siblings, and only the first 5 are ever named in prose). Never
    invented, deduplicated, order-preserving (primary customer/card
    first).

    `file_report_reason` is accepted for interface consistency with
    _sar_narrative but is NEVER inspected here (G13 review fix): whether
    connected cards are cited depends only on _sar_connected_card_keys(
    ledger) actually returning real evidence, never on which rule's text
    happens to appear in the reason string -- this applies equally to R6
    and R9 (R9's own cross-customer evidence is a strict subset of R6's,
    so the same real cards are just as legitimately part of an R9-
    triggered report's subjects).
    """
    combined = (ledger.get("full_evidence") or {}).get("combined_evidence") or {}
    subjects: list[str] = []

    def _add(value):
        if value and value not in subjects:
            subjects.append(value)

    _add((combined.get("customer") or {}).get("customer_id"))
    _add((combined.get("card") or {}).get("card_key"))
    _add((combined.get("device_evidence") or {}).get("device_profile"))

    for card_key in _sar_connected_card_keys(ledger)[:_SAR_MAX_NAMED_CONNECTED_CARDS]:
        _add(card_key)

    return subjects


def _sar_narrative(ledger: dict, validated: dict, case: dict, file_report_reason: str) -> str:
    """README: 6-12 sentences, "who, what, when, where, how, and why it
    is suspicious." Deterministic sentence sequence, no LLM call, built
    only from already-validated/grounded fields -- a dimension whose
    grounding data isn't available in this ledger is simply omitted, per
    caller's instruction, rather than fabricated. `file_report_reason` is
    the FILE_REPORT entry's own already-computed `reason` (from
    g2_policy._evaluate_file_report, G13) -- for R6 it already names the
    actual shared device profile/billing region value, so this function
    does not re-derive that naming itself (avoids duplicating g2_policy's
    own logic). For R9, `validated["pattern_description"]` (already
    validated case output, never re-derived) supplies the "how". The
    verdict sentence below is always literally `validated["verdict"]` --
    never a hardcoded assumption about what that verdict must be (G13
    review fix).
    """
    combined = (ledger.get("full_evidence") or {}).get("combined_evidence") or {}
    customer_id = (combined.get("customer") or {}).get("customer_id")
    card_key = (combined.get("card") or {}).get("card_key")
    pattern = validated.get("pattern")
    exposure = validated.get("exposure_usd")
    affected = validated.get("affected_txn_ids") or []

    sentences: list[str] = []

    if customer_id and card_key:
        sentences.append(f"This report concerns customer {customer_id}, card {card_key}.")
    elif card_key:
        sentences.append(f"This report concerns card {card_key}.")
    elif customer_id:
        sentences.append(f"This report concerns customer {customer_id}.")

    sentences.append(
        f"The investigation's reasoning layer reached a '{validated.get('verdict')}' verdict "
        f"with a fraud probability of {validated.get('fraud_probability')}."
    )

    if affected and exposure is not None:
        sentences.append(f"{len(affected)} transaction(s) were identified as part of this episode, totaling ${exposure}.")
    elif exposure is not None:
        sentences.append(f"The total exposure identified is ${exposure}.")

    if pattern and pattern != "none":
        sentences.append(f"The identified pattern is {pattern}.")

    if case.get("first_suspicious_txn_id"):
        sentences.append(f"The earliest transaction identified as part of this episode is {case['first_suspicious_txn_id']}.")

    dates = _sar_activity_dates(ledger, validated)
    if dates:
        if dates[0] == dates[1]:
            sentences.append(f"The activity occurred on {dates[0]}.")
        else:
            sentences.append(f"The activity occurred between {dates[0]} and {dates[1]}.")

    region = (combined.get("billing_region") or {}).get("addr1")
    if region is not None:
        sentences.append(f"The flagged transaction is billed in region {region}.")

    if pattern == "card_testing":
        sentences.append(
            "The pattern matches card testing: several small online authorizations preceded a "
            "strictly larger purchase on the same card."
        )
    elif pattern == "undocumented" and validated.get("pattern_description"):
        sentences.append(validated["pattern_description"])

    sentences.append(f"{file_report_reason.rstrip('.')}.")

    # G13 review fix: gated on real evidence (_sar_connected_card_keys
    # actually returning something), never on parsing file_report_reason
    # for "R6" -- applies equally to R6 and R9, whose cross-customer
    # evidence is a strict subset of R6's own.
    connected = _sar_connected_card_keys(ledger)
    if connected:
        shown = connected[:_SAR_MAX_NAMED_CONNECTED_CARDS]
        more = ", and others" if len(connected) > _SAR_MAX_NAMED_CONNECTED_CARDS else ""
        sentences.append(f"{len(connected)} other card(s) sharing this element already show confirmed fraud: {', '.join(shown)}{more}.")

    similar = case.get("similar_prior_cases") or []
    if similar:
        shown = similar[:5]
        more = ", and others" if len(similar) > 5 else ""
        sentences.append(f"This activity resembles {len(similar)} previously closed fraud case(s): {', '.join(shown)}{more}.")

    if case.get("summary"):
        sentences.append(case["summary"])

    return " ".join(sentences[:12])


def _build_sar(ledger: dict, case: dict, validated: dict, next_best_actions_final: list[dict]) -> dict:
    """Organizer README Part 2 (`sar`). `file`/`reason` are read directly
    from the FILE_REPORT entry in `next_best_actions_final`, if any --
    must agree with whether FILE_REPORT appears there (organizer's own
    field definition), never re-decided here. When absent, returns the
    organizer's exact required empty shape. When present, `narrative`/
    `subjects`/`total_amount_usd`/`activity_dates` are built
    deterministically from data already computed above -- no new graph
    or LLM call.
    """
    file_report = next((a for a in next_best_actions_final if a.get("action") == "FILE_REPORT"), None)
    if file_report is None:
        return dict(_EMPTY_SAR)

    reason = file_report.get("reason", "")
    return {
        "file": True,
        "reason": reason,
        "narrative": _sar_narrative(ledger, validated, case, reason),
        "subjects": _sar_subjects(ledger, reason),
        "total_amount_usd": validated.get("exposure_usd", 0),
        "activity_dates": _sar_activity_dates(ledger, validated),
    }


def _stop_reason_for_success(
    ledger: dict,
    validated: dict,
    evidence_requested: bool,
    evidence_request_outcome: dict | None = None,
) -> str:
    """Organizer README, Answer Format top-level fields: `stop_reason` --
    "Why the investigation ended here." Built only from vocabulary this
    project already established (G1's own `recommendation` values and
    the reasoning layer's own validated `verdict`) -- no new semantic
    meaning invented for this field. `evidence_requested` is the G12
    evidence-request loop's own honest signal (src.agent.evidence_loop)
    of whether it found a policy-relevant gap for this case -- never
    invented here.

    `evidence_request_outcome` is G14's own record (see
    _run_evidence_request_round) of whether an additional graph evidence
    request was made. Bug fix: this parameter did not exist before, so a
    case where G14's own second reasoning round ran and changed the
    verdict still produced "No evidence-request gap existed" here --
    factually wrong. Checked ONLY for round2_ok (a fully successful round
    2): a rejected or failed request already left `validated` completely
    unaffected by design, so it is correctly left unmentioned here too.
    """
    g14_completed = bool(evidence_request_outcome and evidence_request_outcome.get("round2_ok"))
    g12_note = (
        "One evidence request was simulated and its assumed response applied to the final "
        "next-best-action recommendation (see evidence_requests)."
    )
    g14_note = (
        f"An additional historical_case_evidence request for card "
        f"{evidence_request_outcome['target_card_key']} was made and a second reasoning "
        f"pass incorporated it before this verdict was reached."
        if g14_completed
        else ""
    )
    if evidence_requested and g14_completed:
        # Both loops fired for this case (e.g. G14's own round 2 still
        # left the verdict "uncertain" enough to also satisfy G12's R1
        # gate) -- neither is more authoritative than the other, so both
        # are stated rather than one silently overriding the other.
        loop_note = f"{g12_note} {g14_note}"
    elif evidence_requested:
        loop_note = g12_note
    elif g14_completed:
        loop_note = g14_note
    else:
        loop_note = "No evidence-request gap existed, so no further steps were taken."
    return (
        f"G1 investigation completed (recommendation: {ledger.get('recommendation')}); "
        f"the reasoning layer produced a validated '{validated['verdict']}' verdict from the "
        f"evidence gathered. {loop_note}"
    )


def assemble_case(ledger: dict, g2_result: dict, validated: dict) -> dict:
    """`validated` is the accepted output of
    reasoning_validator.validate_reasoning_output (an {"ok": True, ...}
    dict) -- callers must check "ok" before calling this.
    """
    return {
        "status": _status_for(validated["verdict"], ledger.get("recommendation")),
        "verdict": validated["verdict"],
        "fraud_probability": validated["fraud_probability"],
        "pattern": validated["pattern"],
        "pattern_description": validated["pattern_description"],
        "affected_txn_ids": [str(i) for i in validated["affected_txn_ids"]],
        "first_suspicious_txn_id": (
            str(validated["first_suspicious_txn_id"]) if validated["first_suspicious_txn_id"] is not None else ""
        ),
        "exposure_usd": validated["exposure_usd"],
        "evidence": _evidence_list(ledger, g2_result),
        "similar_prior_cases": _similar_prior_cases(ledger),
        "summary": validated["summary"],
    }


def _connected_via_device_card_keys(ledger: dict) -> set[str]:
    """The exact, full (never sampled/trimmed) set the anti-fabrication
    check validates a requested target_card_key against -- read from the
    same full_evidence.combined_evidence.connected_cards.connected_via_device
    path reasoning.py's own evidence_request_candidates is built from,
    but here unbounded: a real card the LLM names correctly is never
    rejected merely for falling outside the prompt's rendering cap.
    """
    combined = (ledger.get("full_evidence") or {}).get("combined_evidence") or {}
    connected = combined.get("connected_cards") or {}
    return {
        c["card_key"]
        for c in (connected.get("connected_via_device") or [])
        if isinstance(c, dict) and c.get("card_key")
    }


def _ledger_with_additional_evidence(ledger: dict, target_card_key: str, q4_result: dict) -> dict:
    """Shallow-copies the ledger and its full_evidence dict only, for
    round 2's own build_evidence_package call -- never mutates the
    caller's original ledger (g2_policy.evaluate, assemble_case, and
    every other consumer of the original `ledger` argument see it
    completely unchanged), and never overwrites the flagged card's own
    combined_evidence.historical_cases. The new evidence lives only in
    this new, clearly separated key.
    """
    full_evidence = dict(ledger.get("full_evidence") or {})
    full_evidence["additional_evidence"] = {
        "target_card_key": target_card_key,
        "historical_case_evidence": q4_result,
    }
    enriched = dict(ledger)
    enriched["full_evidence"] = full_evidence
    return enriched


async def _run_evidence_request_round(
    ledger: dict, g2_result: dict, validated: dict, *, call_llm: CallLLM, call_tool: CallTool, timeout_s: float,
) -> tuple[dict, dict | None]:
    """G14: at most ONE additional evidence request, at most ONE
    additional tool call, at most ONE additional LLM call -- returns
    (final_validated, evidence_request_outcome). `final_validated` is
    round 2's validated result on a fully successful round 2, or
    `validated` (round 1's, unchanged) on any rejection or failure --
    an optional enrichment can only ever improve on round 1's already-
    valid result, never replace it with a failure. `evidence_request_
    outcome` is None only when round 1 didn't ask for anything; every
    other outcome (rejected, tool-failed, round-2-failed, or accepted)
    is recorded, for auditability, never silently dropped.
    """
    if not validated.get("needs_more_evidence"):
        return validated, None

    request = validated.get("evidence_request") or {}
    target_card_key = request.get("target_card_key")
    reason = request.get("reason")
    outcome: dict = {
        "requested": True,
        "type": request.get("type"),
        "target_card_key": target_card_key,
        "reason": reason,
        "accepted": False,
        "rejection_reason": None,
        "round2_attempted": False,
        "round2_ok": None,
    }

    # Anti-fabrication gate: reasoning_validator.py already rejected any
    # `type` other than historical_case_evidence outright (the whole
    # round-1 response fails validation in that case, so this function is
    # never reached) -- the ONE thing left to check here, because it
    # needs the ledger, is whether target_card_key names a card this
    # investigation's own graph evidence actually surfaced.
    if target_card_key not in _connected_via_device_card_keys(ledger):
        outcome["rejection_reason"] = "target_card_key_not_in_connected_via_device"
        return validated, outcome

    tool_name = _EVIDENCE_TYPE_TOOL_NAMES[request["type"]]
    ok, q4_result, err = await call_tool(tool_name, {"card_key": target_card_key})
    if not ok:
        outcome["accepted"] = True
        outcome["rejection_reason"] = None
        outcome["tool_error"] = scrub_secret(err or "")
        return validated, outcome

    outcome["accepted"] = True
    enriched_ledger = _ledger_with_additional_evidence(ledger, target_card_key, q4_result)
    round2_evidence_package = build_evidence_package(enriched_ledger, g2_result)

    round2_outcome = await get_llm_reasoning(round2_evidence_package, call_llm=call_llm, timeout_s=timeout_s)
    outcome["round2_attempted"] = True
    if not round2_outcome["ok"]:
        outcome["round2_ok"] = False
        outcome["round2_failure_reason"] = round2_outcome["failure_reason"]
        return validated, outcome

    round2_validated = validate_reasoning_output(round2_outcome["raw"], round2_evidence_package)
    if not round2_validated["ok"]:
        outcome["round2_ok"] = False
        outcome["round2_failure_reason"] = round2_validated["failure_reason"]
        return validated, outcome

    outcome["round2_ok"] = True
    return round2_validated, outcome


async def investigate_and_assemble(
    ledger: dict, g2_result: dict, *, call_llm: CallLLM | None = None, timeout_s: float = DEFAULT_LLM_TIMEOUT_S,
    call_tool: CallTool | None = None,
) -> dict:
    """Top-level orchestration for this phase: runs the LLM reasoning
    layer over the already-completed G1/G2 results, validates its
    output, and assembles the final case object. On any failure (LLM call
    failure, timeout, invalid JSON, failed validation), returns an
    explicit {"ok": False, ...} stop-state -- never a fabricated case
    object.

    G14: makes zero MCP/TigerGraph calls itself UNLESS round 1's
    validated output sets needs_more_evidence=true AND `call_tool` is
    given -- then, and only then, at most ONE additional call
    (src.agent.workflow's own historical_case_evidence tool, never a
    tool name the LLM chose) may run, followed by at most ONE additional
    LLM call. `call_tool` defaults to None: every existing caller that
    doesn't pass it (every test written before this phase, and any
    caller that doesn't wire it) gets the exact original zero-MCP-call,
    single-LLM-call behavior, unconditionally -- this is not merely the
    common case, it is the ONLY reachable path when call_tool is None,
    regardless of what the LLM's own output says.
    """
    evidence_package = build_evidence_package(ledger, g2_result)

    outcome = await get_llm_reasoning(evidence_package, call_llm=call_llm, timeout_s=timeout_s)
    if not outcome["ok"]:
        return {
            "ok": False,
            "failure_reason": outcome["failure_reason"],
            "detail": outcome.get("detail", ""),
            "stop_reason": (
                f"LLM reasoning layer failed ({outcome['failure_reason']}); "
                "no case object produced rather than fabricating one"
            ),
        }

    validated = validate_reasoning_output(outcome["raw"], evidence_package)
    if not validated["ok"]:
        return {
            "ok": False,
            "failure_reason": validated["failure_reason"],
            "detail": validated.get("detail", ""),
            "stop_reason": (
                f"LLM output failed validation ({validated['failure_reason']}); "
                "no case object produced rather than fabricating one"
            ),
        }

    # G14: at most one additional evidence request/tool call/LLM call --
    # see _run_evidence_request_round's own docstring. A no-op (returns
    # `validated` unchanged, outcome=None) whenever call_tool is None or
    # round 1 didn't ask for anything.
    if call_tool is not None:
        final_validated, evidence_request_outcome = await _run_evidence_request_round(
            ledger, g2_result, validated, call_llm=call_llm, call_tool=call_tool, timeout_s=timeout_s,
        )
    else:
        final_validated, evidence_request_outcome = validated, None

    # G11-D: the post-reasoning ("final") G2 pass -- called ONLY here,
    # after both the LLM call and validation have succeeded, using the
    # SAME g2_policy.evaluate() the pre-reasoning g2_result above already
    # came from, now additionally given the validated reasoning output
    # (round 2's, if G14's one optional additional round ran and
    # succeeded; round 1's unchanged otherwise). On any failure path
    # above, this line is never reached -- there is no other call to
    # g2_policy.evaluate with a second argument anywhere in this module.
    g2_result_final = g2_policy.evaluate(ledger, final_validated)

    # G12: evidence-request / simulated-response loop -- runs only after
    # both the LLM call and validation succeeded, over the SAME
    # post-reasoning G2 pass just computed above. Adds no graph/LLM call
    # of its own; see src.agent.evidence_loop.apply.
    evidence_outcome = evidence_loop.apply(ledger, final_validated, g2_result_final)

    case = assemble_case(ledger, g2_result, final_validated)
    next_best_actions = _next_best_actions(g2_result, evidence_outcome["g2_result"])

    # Internal-only audit field (not part of the organizer's 9-key
    # submission schema): isolates what the evidence-request/simulated-
    # response pass itself changed, separate from next_best_actions'
    # own initial/final. Present only when an evidence request actually
    # happened; None otherwise -- every other field above is unaffected
    # either way.
    evidence_delta = (
        _evidence_delta(g2_result_final, evidence_outcome["g2_result"])
        if evidence_outcome["evidence_requests"]
        else None
    )

    return {
        "ok": True,
        "case": case,
        "evidence_requests": evidence_outcome["evidence_requests"],
        "next_best_actions": next_best_actions,
        "sar": _build_sar(ledger, case, final_validated, next_best_actions["final"]),
        "stripped_ids": final_validated.get("stripped_ids", []),
        "stop_reason": _stop_reason_for_success(
            ledger, final_validated, bool(evidence_outcome["evidence_requests"]), evidence_request_outcome,
        ),
        "evidence_delta": evidence_delta,
        # G14: internal-only audit field (not part of the organizer's
        # 9-key submission schema, same discipline as evidence_delta) --
        # None when round 1 never asked for more evidence; a full record
        # (requested/accepted/rejection_reason/round2_ok/...) otherwise,
        # so a rejected or failed request is never silently invisible.
        "evidence_request_outcome": evidence_request_outcome,
    }
