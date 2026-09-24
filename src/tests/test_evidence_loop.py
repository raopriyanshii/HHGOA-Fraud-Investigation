"""
G12 unit tests for src/agent/evidence_loop.py.

Fixtures are hand-built G1-ledger-shaped dicts and validated-reasoning-
shaped dicts, matching the same conventions test_case_output.py and
test_g2_policy.py already use. No LLM, no MCP, no TigerGraph -- evidence_
loop.apply is a pure function over already-computed, already-validated
data plus one real (but MCP/LLM-free) call into g2_policy.evaluate.
"""
from src.agent import evidence_loop as el
from src.policy import g2_policy as g2


def _ledger(**overrides):
    ledger = {
        "investigation_id": "test-fixture",
        "trigger": {"type": "flagged_txn_id", "value": 100},
        "tool_calls": [],
        "uncertainties": [],
        "recommendation": "insufficient_evidence",
        "recommendation_basis": {"reasons": [], "tool_call_orders": []},
        "pattern_evidence": {"matched": False, "pattern": "card_testing", "candidates": [], "uncertainty": []},
    }
    ledger.update(overrides)
    return ledger


def _validated(**overrides):
    validated = {
        "ok": True,
        "verdict": "uncertain",
        "fraud_probability": 0.3,
        "pattern": "none",
        "pattern_description": "",
        "affected_txn_ids": [],
        "first_suspicious_txn_id": None,
        "exposure_usd": 0,
        "summary": "a summary",
        "stripped_ids": [],
    }
    validated.update(overrides)
    return validated


# --- gate: when no evidence request is warranted ---

def test_no_request_when_verdict_is_legitimate():
    result = el.apply(_ledger(), _validated(verdict="legitimate", fraud_probability=0.05), g2.evaluate(_ledger()))
    assert result["evidence_requests"] == []


def test_no_request_when_verdict_is_confident_fraud():
    result = el.apply(_ledger(), _validated(verdict="fraud", fraud_probability=0.95), g2.evaluate(_ledger()))
    assert result["evidence_requests"] == []


def test_no_request_when_probability_at_the_r1_ceiling():
    # R1's own literal text: "below 0.70" -- exactly 0.70 does not qualify.
    result = el.apply(_ledger(), _validated(verdict="uncertain", fraud_probability=0.70), g2.evaluate(_ledger()))
    assert result["evidence_requests"] == []


def test_no_request_when_probability_above_the_r1_ceiling():
    result = el.apply(_ledger(), _validated(verdict="uncertain", fraud_probability=0.9), g2.evaluate(_ledger()))
    assert result["evidence_requests"] == []


def test_g2_result_returned_unchanged_when_no_request():
    ledger = _ledger()
    g2_result_final = g2.evaluate(ledger, _validated(verdict="fraud", fraud_probability=0.95))
    result = el.apply(ledger, _validated(verdict="fraud", fraud_probability=0.95), g2_result_final)
    assert result["g2_result"] is g2_result_final


# --- gate: when a genuine gap exists ---

def test_request_made_when_uncertain_and_below_ceiling():
    result = el.apply(_ledger(), _validated(verdict="uncertain", fraud_probability=0.3), g2.evaluate(_ledger()))
    assert len(result["evidence_requests"]) == 1


def test_request_type_is_customer_validation_without_r5_pattern_match():
    ledger = _ledger()  # pattern_evidence.matched is False in the default fixture
    result = el.apply(ledger, _validated(verdict="uncertain", fraud_probability=0.3), g2.evaluate(ledger))
    assert result["evidence_requests"][0]["type"] == "customer_validation"


def test_request_type_is_step_up_auth_when_r5_pattern_matched():
    ledger = _ledger(pattern_evidence={"matched": True, "pattern": "card_testing", "candidates": [], "uncertainty": []})
    result = el.apply(ledger, _validated(verdict="uncertain", fraud_probability=0.3), g2.evaluate(ledger))
    assert result["evidence_requests"][0]["type"] == "step_up_auth"


def test_asked_after_step_reflects_number_of_recorded_tool_calls():
    ledger = _ledger(tool_calls=[
        {"order": 1, "tool": "combined_evidence", "arguments": {}, "success": True, "curated_result_summary": {}},
        {"order": 2, "tool": "historical_case_evidence", "arguments": {}, "success": True, "curated_result_summary": {}},
        {"order": 3, "tool": "cross_card_fraud_verification", "arguments": {}, "success": True, "curated_result_summary": {}},
    ])
    result = el.apply(ledger, _validated(verdict="uncertain", fraud_probability=0.3), g2.evaluate(ledger))
    assert result["evidence_requests"][0]["asked_after_step"] == 3


def test_asked_after_step_is_zero_with_no_recorded_tool_calls():
    ledger = _ledger()  # tool_calls == []
    result = el.apply(ledger, _validated(verdict="uncertain", fraud_probability=0.3), g2.evaluate(ledger))
    assert result["evidence_requests"][0]["asked_after_step"] == 0


def test_evidence_request_shape_matches_organizer_schema_exactly():
    ledger = _ledger()
    result = el.apply(ledger, _validated(verdict="uncertain", fraud_probability=0.3), g2.evaluate(ledger))
    request = result["evidence_requests"][0]
    assert set(request.keys()) == {"type", "asked_after_step", "assumed_response"}
    assert request["type"] in ("customer_validation", "step_up_auth", "analyst_info")
    assert isinstance(request["asked_after_step"], int)
    assert isinstance(request["assumed_response"], str) and request["assumed_response"]


def test_no_fabricated_graph_evidence_in_a_request():
    # The evidence_request object itself carries no entity_ids/claim/ref --
    # it is never treated as a graph-evidence citation, and nothing here
    # invents a transaction/card/customer/case id.
    ledger = _ledger()
    result = el.apply(ledger, _validated(verdict="uncertain", fraud_probability=0.3), g2.evaluate(ledger))
    request = result["evidence_requests"][0]
    assert "entity_ids" not in request
    assert "claim" not in request
    assert "ref" not in request


# --- simulated response is always the fixed "no_reply" state, never
# derived from fraud_probability (see g2_policy_audit follow-up fix) ---

def test_simulated_response_is_always_no_reply_regardless_of_probability():
    for fraud_probability in (0.01, 0.1, 0.2, 0.3, 0.49, 0.5, 0.51, 0.65, 0.69):
        ledger = _ledger()
        validated = _validated(verdict="uncertain", fraud_probability=fraud_probability)
        result = el.apply(ledger, validated, g2.evaluate(ledger, validated))
        assert "no reply" in result["evidence_requests"][0]["assumed_response"]


def test_simulation_does_not_depend_on_fraud_probability():
    # Two cases that would previously have landed in different
    # confirmed/denied/no_reply bands must now produce byte-identical
    # evidence_requests (module-internal fields only -- asked_after_step
    # and type do not depend on probability either).
    ledger = _ledger()
    low = el.apply(ledger, _validated(verdict="uncertain", fraud_probability=0.1), g2.evaluate(ledger))
    high = el.apply(ledger, _validated(verdict="uncertain", fraud_probability=0.65), g2.evaluate(ledger))
    assert low["evidence_requests"][0]["assumed_response"] == high["evidence_requests"][0]["assumed_response"]


def test_evidence_requests_records_the_simulation_assumption_honestly():
    ledger = _ledger()
    validated = _validated(verdict="uncertain", fraud_probability=0.3)
    result = el.apply(ledger, validated, g2.evaluate(ledger, validated))
    text = result["evidence_requests"][0]["assumed_response"]
    assert "Simulated" in text
    assert "no reply" in text
    assert "no real" in text  # discloses this is a benchmark assumption, not a real reply


# --- simulated response actually reaches G2 and changes the outcome (R4 only) ---

def test_simulated_no_reply_response_grounds_r4_monitor_and_decline():
    ledger = _ledger()
    validated = _validated(verdict="uncertain", fraud_probability=0.35, exposure_usd=100)
    result = el.apply(ledger, validated, g2.evaluate(ledger, validated))
    recommended = {e["action"] for e in result["g2_result"]["recommended"]}
    assert "MONITOR_CARD" in recommended
    assert "DECLINE_TRANSACTION" in recommended
    assert "ESCALATE_TO_ANALYST" not in recommended  # exposure $100 is under R4's $500 escalation figure


def test_no_reply_plus_exposure_over_500_escalates_with_r4_citation():
    ledger = _ledger()
    validated = _validated(verdict="uncertain", fraud_probability=0.35, exposure_usd=600)
    result = el.apply(ledger, validated, g2.evaluate(ledger, validated))
    entry = next(e for e in result["g2_result"]["recommended"] if e["action"] == "ESCALATE_TO_ANALYST")
    assert entry["reason"].startswith("R4")  # not R8 -- see g2_policy._evaluate_escalate_to_analyst fix


def test_no_reply_plus_exposure_at_or_below_500_does_not_escalate():
    ledger = _ledger()
    validated = _validated(verdict="uncertain", fraud_probability=0.35, exposure_usd=500)
    result = el.apply(ledger, validated, g2.evaluate(ledger, validated))
    assert not any(e["action"] == "ESCALATE_TO_ANALYST" for e in result["g2_result"]["recommended"])


# --- determinism ---

def test_apply_is_deterministic_across_repeated_calls():
    ledger = _ledger()
    validated = _validated(verdict="uncertain", fraud_probability=0.4, exposure_usd=200)
    g2_result_final = g2.evaluate(ledger, validated)
    result_1 = el.apply(ledger, validated, g2_result_final)
    result_2 = el.apply(ledger, validated, g2_result_final)
    assert result_1 == result_2
