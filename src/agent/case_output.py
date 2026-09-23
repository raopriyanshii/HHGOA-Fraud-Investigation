"""
G7 -- output assembly (isolated adapter).

Combines the G1 ledger, the G2 policy result, and the validated LLM
reasoning output into the organizer-required `case` object and
`next_best_actions` object. Performs zero graph/LLM calls itself -- it
only assembles already-computed, already-validated pieces, and it is the
one place these three sources are combined: G1, G2, and the reasoning
layer never see each other's output shape.

Scope (deliberately limited this phase): produces `case` (Part 1) and
`next_best_actions` (Part 3, `initial` == `final` -- no evidence-request
loop exists yet, matching the organizer's own rule "If you requested
nothing, final equals initial"). Does NOT produce `sar`, `evidence_
requests`, `stop_reason` (beyond the failure-path one below), `tool_
calls`, `tokens`, `latency_s`, `connected_card_ids`,
`connected_device_profiles`, `written_to_graph`, or `graph_case_id` --
these remain later-phase work, not invented here.
"""
from __future__ import annotations

from src.agent.reasoning import DEFAULT_LLM_TIMEOUT_S, CallLLM, build_evidence_package, get_llm_reasoning
from src.agent.reasoning_validator import validate_reasoning_output


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


def _next_best_actions(g2_result: dict) -> dict:
    """Entirely deterministic, entirely from G2's own `recommended`
    bucket. G2's `prohibited`/`deferred` buckets are never read here --
    that is what structurally guarantees a PROHIBITED action (e.g.
    BLOCK_ALL_CARDS) can never appear as a recommended next-best-action:
    there is no code path from `prohibited` to this output at all, let
    alone one that passes through the LLM. The LLM's output is not
    consulted for this field at all in this phase.
    """
    actions = [
        {"action": e["action"], "route": e.get("route"), "reason": e["reason"]}
        for e in g2_result.get("recommended", [])
    ]
    return {"initial": actions, "final": actions, "what_changed": "nothing"}


def _evidence_list(ledger: dict, g2_result: dict) -> list[dict]:
    """Deterministically templated (no LLM), mirroring the organizer's
    `evidence` shape ({claim, source, ref, entity_ids}). Only covers the
    deterministic findings this project currently establishes (R5/R6) --
    does not attempt to restate every raw evidence field as a claim.
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
                "entity_ids": [],
            })
            break  # one R6 citation is enough -- all three actions share the same underlying evidence
    return evidence


def _stop_reason_for_success(ledger: dict, validated: dict) -> str:
    """Organizer README, Answer Format top-level fields: `stop_reason` --
    "Why the investigation ended here." Built only from vocabulary this
    project already established (G1's own `recommendation` values and
    the reasoning layer's own validated `verdict`) -- no new semantic
    meaning invented for this field. This phase has no evidence-request/
    response loop (see spec Section 13), so "further steps" always means
    "none were taken," stated plainly rather than implying a loop ran.
    """
    return (
        f"G1 investigation completed (recommendation: {ledger.get('recommendation')}); "
        f"the reasoning layer produced a validated '{validated['verdict']}' verdict from the "
        "evidence gathered. No evidence-request loop exists in this phase, so no further "
        "steps were taken."
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


async def investigate_and_assemble(
    ledger: dict, g2_result: dict, *, call_llm: CallLLM | None = None, timeout_s: float = DEFAULT_LLM_TIMEOUT_S,
) -> dict:
    """Top-level orchestration for this phase: runs the LLM reasoning
    layer over the already-completed G1/G2 results, validates its
    output, and assembles the final case object. Makes zero MCP/
    TigerGraph calls itself -- call_llm is the only external call this
    function, or anything it calls, can make, and it is bounded by
    timeout_s (see reasoning.get_llm_reasoning) so a non-responding
    provider cannot hang this call indefinitely. On any failure (LLM call
    failure, timeout, invalid JSON, failed validation), returns an
    explicit {"ok": False, ...} stop-state -- never a fabricated case
    object.
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

    return {
        "ok": True,
        "case": assemble_case(ledger, g2_result, validated),
        "next_best_actions": _next_best_actions(g2_result),
        "stripped_ids": validated.get("stripped_ids", []),
        "stop_reason": _stop_reason_for_success(ledger, validated),
    }
