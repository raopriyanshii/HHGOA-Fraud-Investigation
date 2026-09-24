"""
G12 -- evidence-request / simulated-response loop.

The organizer README requires three things this project did not yet
have: `evidence_requests` (customer_validation / step_up_auth /
analyst_info, each with `asked_after_step` and `assumed_response`), and
a `next_best_actions.final` that is genuinely recalculated after that
simulated response, with `what_changed` reflecting it. The README also
states plainly (glossary + policy §5): "Customer and analyst replies are
not provided. If your agent asks the customer or requests step-up
authentication, simulate the response in your own system and record what
you assumed."

This module is a small, deterministic layer over already-computed,
already-validated data -- src.agent.case_output.investigate_and_assemble
is the only caller. It:
  1. decides whether the case genuinely rests on a policy-relevant
     evidence gap (R1: "the case rests on a single signal ... and your
     assessed fraud probability is below 0.70");
  2. picks which of R1's two named actions the request corresponds to
     (VERIFY_WITH_CUSTOMER -> "customer_validation", or STEP_UP_AUTH ->
     "step_up_auth" when R5's own card-testing evidence is present --
     never an invented trigger);
  3. simulates a response -- always the fixed "no_reply" state (see
     _simulate_response's docstring for why this project does not use
     fraud_probability, or any other validated field, to pick among
     confirmed/denied/no_reply);
  4. re-evaluates G2 (src.policy.g2_policy.evaluate) with that simulated
     response, so R4 becomes checkable.

Makes zero graph/MCP/LLM calls itself -- the one g2_policy.evaluate call
it may perform is the same pure, already-tested function case_output.py
already calls, given one additional argument.
"""
from __future__ import annotations

from src.policy import g2_policy

# R1's own literal number ("fraud probability is below 0.70").
_R1_PROBABILITY_CEILING = 0.70

# The one, fixed simulated response state this project ever assumes --
# see _simulate_response's docstring for why.
_SIMULATED_RESPONSE_STATE = "no_reply"


def _r5_pattern_matched(ledger: dict) -> bool:
    # Mirrors g2_policy._r5_evidence_present's exact condition -- reused
    # by name rather than imported (that helper is private to g2_policy;
    # this project's own convention elsewhere is to re-read the same
    # ledger field directly rather than reach into another module's
    # underscore-prefixed internals).
    return bool((ledger.get("pattern_evidence") or {}).get("matched"))


def _should_request_evidence(ledger: dict, validated: dict) -> dict | None:
    """R1: "If the case rests on a single signal (including a risk score
    alone) and your assessed fraud probability is below 0.70, recommend
    VERIFY_WITH_CUSTOMER or STEP_UP_AUTH before any block."

    "Rests on a single signal" is operationalized here as: the reasoning
    layer's own validated verdict is "uncertain". A "fraud" or
    "legitimate" verdict means the reasoning layer itself judged the
    evidence as no longer genuinely ambiguous -- the honest signal this
    project already produces for whether a second, independent piece of
    evidence settled the question. This is deliberately conservative:
    a "legitimate" case is never asked about (no gap to close), and a
    confident "fraud" case is never asked about either (R1 governs
    verification BEFORE a block, not after the case is already settled).

    Returns None when no gap exists. Returns {"type": ...} when one does
    -- "step_up_auth" when R5's card-testing evidence already grounds
    STEP_UP_AUTH as a live G2 action (the one condition the task's own
    scope allows for this type, "R1/R5" -- never an invented trigger),
    "customer_validation" (R1's other named action) otherwise.
    """
    if validated.get("verdict") != "uncertain":
        return None

    fraud_probability = validated.get("fraud_probability")
    if fraud_probability is None or fraud_probability >= _R1_PROBABILITY_CEILING:
        return None

    if _r5_pattern_matched(ledger):
        return {"type": "step_up_auth"}
    return {"type": "customer_validation"}


def _simulate_response() -> str:
    """Fixed, deterministic, benchmark-only simulation -- there is no
    live customer or analyst channel in this round (README §5 / Notes:
    "Customer and analyst replies are not provided ... simulate the
    response in your own system"). This project deliberately does NOT
    derive the simulated state from fraud_probability (or any other
    validated field): the R1 gate that decides WHETHER to request
    evidence already runs on that same probability, so using it a
    second time to also decide WHAT the (fictitious) customer said would
    manufacture agreement with the model's own uncertain estimate --
    laundering one ungrounded signal into apparent corroborating
    evidence, and risking inflated/incorrect final recommendations on
    exactly the cases the reasoning layer was least sure about. Always
    returning "no_reply" is the one assumption that adds no fabricated
    information either way: it is R4's own designated "we don't know"
    state, not a claim that the customer said anything in particular.
    """
    return _SIMULATED_RESPONSE_STATE


def _assumed_response_text(request_type: str, state: str) -> str:
    subject = "step-up authentication" if request_type == "step_up_auth" else "customer validation"
    return (
        f"Simulated {subject} response: no reply within the 24-hour window. Benchmark "
        "simulation only -- no real 24 hours elapsed and no real customer/analyst reply "
        "was received; this project has no live customer or analyst channel (organizer "
        "README, policy §5). This is the one, fixed assumption always made when evidence "
        f"is requested, recorded here honestly as state={state!r}."
    )


def apply(ledger: dict, validated: dict, g2_result_final: dict) -> dict:
    """Top-level entry point, called once per case, only after reasoning
    and validation have both already succeeded.

    `g2_result_final` is the already-computed post-reasoning G2 pass
    (g2_policy.evaluate(ledger, validated), performed by the caller
    before this function runs -- never repeated here).

    Returns {"evidence_requests": [...], "g2_result": ...}:
      - no evidence gap: evidence_requests is [], g2_result is
        g2_result_final, unchanged -- the common case, and the only
        shape possible before this module existed.
      - a gap exists: evidence_requests has exactly one organizer-shaped
        entry ({"type", "asked_after_step", "assumed_response"}), and
        g2_result is a NEW g2_policy.evaluate(ledger, validated,
        customer_response=...) pass -- one additional, real G2
        re-evaluation of all 14 actions, not a patch onto
        g2_result_final.

    `asked_after_step` is the count of G1 investigation steps already
    recorded in the ledger (len(ledger["tool_calls"])) -- a real,
    grounded number describing when in the investigation this request
    would have been made, never invented.

    Performs no graph/MCP/LLM call of its own.
    """
    request = _should_request_evidence(ledger, validated)
    if request is None:
        return {"evidence_requests": [], "g2_result": g2_result_final}

    state = _simulate_response()
    evidence_request = {
        "type": request["type"],
        "asked_after_step": len(ledger.get("tool_calls") or []),
        "assumed_response": _assumed_response_text(request["type"], state),
    }

    g2_result_after_response = g2_policy.evaluate(ledger, validated, customer_response=state)

    return {"evidence_requests": [evidence_request], "g2_result": g2_result_after_response}
