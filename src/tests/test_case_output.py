"""
G7 unit tests for src/agent/case_output.py.

Every test uses a fake `call_llm` (no live LLM dependency) and hand-built
ledger/G2-result fixtures matching the real shapes produced by
src.agent.workflow.investigate() and src.policy.g2_policy.evaluate().
"""
import asyncio
import json

from src.agent import case_output as co
from src.policy import g2_policy


def run(coro):
    return asyncio.run(coro)


def _base_ledger(**overrides):
    ledger = {
        "investigation_id": "test-fixture",
        "trigger": {"type": "flagged_txn_id", "value": 100},
        "tool_calls": [],
        "uncertainties": [],
        "recommendation": "insufficient_evidence",
        "recommendation_basis": {"reasons": [], "tool_call_orders": []},
        "pattern_evidence": {"matched": False, "pattern": "card_testing", "candidates": [], "uncertainty": []},
        "full_evidence": {
            "combined_evidence": {
                "transaction": {"TransactionID": 100, "ts": "2016-01-01 10:00:00", "TransactionAmt": 50.0},
                "card": {"card_key": "C00001:1"},
                "customer": {"customer_id": "C00001"},
                "temporal_activity": [{"TransactionID": 100, "ts": "2016-01-01 10:00:00", "TransactionAmt": 50.0, "channel": "online"}],
                "device_evidence": None,
                "billing_region": {"hub_flag": True},
                "historical_cases": {"direct_cases": [{"case_id": "CC-1", "outcome": "confirmed_fraud"}, {"case_id": "CC-2", "outcome": "cleared"}], "connected_cases": []},
                "connected_cards": {"connected_via_case": [], "connected_via_device": []},
            },
            "cross_card_fraud_verification": None,
        },
    }
    ledger.update(overrides)
    return ledger


def _base_g2_result(**overrides):
    result = {
        "investigation_recommendation": "insufficient_evidence",
        "recommended": [],
        "prohibited": [{"action": "BLOCK_ALL_CARDS", "reason": "R10: exception not verifiable"}],
        "eligible_not_recommended": [],
        "deferred": [],
        "uncertainties": [],
    }
    result.update(overrides)
    return result


def _valid_llm_response(**overrides):
    body = {
        "verdict": "uncertain",
        "fraud_probability": 0.3,
        "affected_txn_ids": [],
        "first_suspicious_txn_id": None,
        "pattern": "none",
        "pattern_description": "",
        "summary": "a short summary",
        "reasoning": "some reasoning",
        "uncertainties": [],
    }
    body.update(overrides)
    return body


def _fake_call_llm(response_dict):
    async def call_llm(prompt):
        return json.dumps(response_dict)
    return call_llm


# --- similar_prior_cases: deterministic, never from the LLM ---

def test_similar_prior_cases_cites_only_confirmed_fraud_and_is_deterministic():
    ledger = _base_ledger()
    result = co._similar_prior_cases(ledger)
    assert result == ["CC-1"]  # CC-2 is cleared, must not appear


# --- next_best_actions: PROHIBITED can never appear (item L) ---

def test_prohibited_action_never_appears_in_next_best_actions():
    g2_result = _base_g2_result(recommended=[{"action": "DECLINE_TRANSACTION", "route": "L1", "reason": "R5"}])
    actions = co._next_best_actions(g2_result)
    all_actions = [a["action"] for a in actions["initial"]] + [a["action"] for a in actions["final"]]
    assert "BLOCK_ALL_CARDS" not in all_actions
    assert "DECLINE_TRANSACTION" in all_actions


def test_next_best_actions_initial_equals_final_with_no_evidence_request_loop():
    g2_result = _base_g2_result(recommended=[{"action": "CREATE_CASE", "route": "auto", "reason": "R6"}])
    actions = co._next_best_actions(g2_result)
    assert actions["initial"] == actions["final"]
    assert actions["what_changed"] == "nothing"


# --- deterministic pattern override survives full assembly (item K) ---

def test_r5_matched_pattern_survives_assembly_even_if_llm_disagreed():
    ledger = _base_ledger(pattern_evidence={"matched": True, "pattern": "card_testing", "candidates": [{"authorization_txn_ids": [100], "larger_purchase_txn_id": 100}], "uncertainty": []})
    call_llm = _fake_call_llm(_valid_llm_response(pattern="account_takeover"))
    outcome = run(co.investigate_and_assemble(ledger, _base_g2_result(), call_llm=call_llm))
    assert outcome["ok"] is True
    assert outcome["case"]["pattern"] == "card_testing"


# --- end-to-end success path ---

def test_end_to_end_success_produces_expected_case_shape():
    ledger = _base_ledger()
    call_llm = _fake_call_llm(_valid_llm_response(verdict="legitimate", fraud_probability=0.05))
    outcome = run(co.investigate_and_assemble(ledger, _base_g2_result(), call_llm=call_llm))
    assert outcome["ok"] is True
    case = outcome["case"]
    for key in ("status", "verdict", "fraud_probability", "pattern", "pattern_description", "affected_txn_ids", "first_suspicious_txn_id", "exposure_usd", "evidence", "similar_prior_cases", "summary"):
        assert key in case
    assert case["status"] == "closed_legitimate"
    assert case["affected_txn_ids"] == []
    assert case["exposure_usd"] == 0.0


def test_status_escalated_when_g1_recommendation_is_escalate_and_verdict_uncertain():
    ledger = _base_ledger(recommendation="escalate")
    call_llm = _fake_call_llm(_valid_llm_response(verdict="uncertain"))
    outcome = run(co.investigate_and_assemble(ledger, _base_g2_result(), call_llm=call_llm))
    assert outcome["ok"] is True
    assert outcome["case"]["status"] == "escalated"


# --- failure handling: fail safe, never fabricate (items B-E, N) ---

def test_malformed_json_produces_explicit_failure_not_a_fabricated_case():
    async def bad_call_llm(prompt):
        return "not json"
    outcome = run(co.investigate_and_assemble(_base_ledger(), _base_g2_result(), call_llm=bad_call_llm))
    assert outcome["ok"] is False
    assert outcome["failure_reason"] == "invalid_json"
    assert "case" not in outcome


def test_invalid_verdict_produces_explicit_failure():
    call_llm = _fake_call_llm(_valid_llm_response(verdict="definitely_fraud"))
    outcome = run(co.investigate_and_assemble(_base_ledger(), _base_g2_result(), call_llm=call_llm))
    assert outcome["ok"] is False
    assert outcome["failure_reason"] == "invalid_verdict"


def test_llm_raising_timeout_error_produces_explicit_failure():
    async def timing_out(prompt):
        raise TimeoutError("no response")
    outcome = run(co.investigate_and_assemble(_base_ledger(), _base_g2_result(), call_llm=timing_out))
    assert outcome["ok"] is False
    assert outcome["failure_reason"] == "llm_timeout"
    assert "stop_reason" in outcome
    assert "case" not in outcome


# --- G7 defect fix: enforced timeout terminates a genuinely hanging call_llm ---

def test_genuinely_hanging_call_llm_is_terminated_end_to_end():
    async def hanging_call_llm(prompt):
        await asyncio.sleep(60)
        return "unreachable"

    outcome = run(co.investigate_and_assemble(_base_ledger(), _base_g2_result(), call_llm=hanging_call_llm, timeout_s=0.05))
    assert outcome["ok"] is False
    assert outcome["failure_reason"] == "llm_timeout"
    assert "case" not in outcome


# --- G7 defect fix: stop_reason present and consistent on both paths ---

def test_stop_reason_present_on_success_and_uses_established_vocabulary():
    ledger = _base_ledger(recommendation="monitor")
    call_llm = _fake_call_llm(_valid_llm_response(verdict="uncertain"))
    outcome = run(co.investigate_and_assemble(ledger, _base_g2_result(), call_llm=call_llm))
    assert outcome["ok"] is True
    assert "stop_reason" in outcome
    # built only from already-established vocabulary (G1's recommendation
    # value, the reasoning layer's own verdict) -- nothing invented:
    assert "monitor" in outcome["stop_reason"]
    assert "uncertain" in outcome["stop_reason"]


def test_stop_reason_present_and_unchanged_wording_on_validation_failure():
    call_llm = _fake_call_llm(_valid_llm_response(verdict="not_a_real_verdict"))
    outcome = run(co.investigate_and_assemble(_base_ledger(), _base_g2_result(), call_llm=call_llm))
    assert outcome["ok"] is False
    assert outcome["stop_reason"] == (
        "LLM output failed validation (invalid_verdict); no case object produced rather than fabricating one"
    )


# --- item M: zero MCP/TigerGraph calls made by the reasoning stage ---

def test_reasoning_stage_makes_zero_mcp_or_tigergraph_calls():
    llm_call_count = {"n": 0}

    async def counting_call_llm(prompt):
        llm_call_count["n"] += 1
        return json.dumps(_valid_llm_response())

    outcome = run(co.investigate_and_assemble(_base_ledger(), _base_g2_result(), call_llm=counting_call_llm))
    assert outcome["ok"] is True
    assert llm_call_count["n"] == 1


def test_zero_mcp_tigergraph_calls_is_an_active_guard_not_an_inference(monkeypatch):
    # Stronger than the count-based test above: actively monkeypatches
    # the real MCP/TigerGraph entry points to fail loudly if reached, so
    # this test would catch a future accidental call, not just assert
    # today's absence of one. Covers both entry points G1 itself goes
    # through: the mcp server's own call_tool, and the raw TigerGraph
    # connection factory beneath it.
    from src.graph import tigergraph_connection as tgc
    from src.mcp_server import server as mcp_server_module

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("G7 reasoning/output-assembly must never call TigerGraph/MCP directly")

    monkeypatch.setattr(tgc, "get_connection", _fail_if_called)
    monkeypatch.setattr(mcp_server_module.mcp, "call_tool", _fail_if_called)

    call_llm = _fake_call_llm(_valid_llm_response())
    outcome = run(co.investigate_and_assemble(_base_ledger(), _base_g2_result(), call_llm=call_llm))
    assert outcome["ok"] is True  # would have raised AssertionError above if either entry point were touched


# --- item P: structural output contract (representative fixtures, not live 20-case run) ---

def test_output_contract_holds_across_representative_verdicts():
    for verdict, prob in (("fraud", 0.9), ("legitimate", 0.05), ("uncertain", 0.5)):
        call_llm = _fake_call_llm(_valid_llm_response(verdict=verdict, fraud_probability=prob))
        outcome = run(co.investigate_and_assemble(_base_ledger(), _base_g2_result(), call_llm=call_llm))
        assert outcome["ok"] is True
        case = outcome["case"]
        assert case["status"] in ("closed_fraud", "closed_legitimate", "escalated", "open")
        assert isinstance(case["affected_txn_ids"], list) and all(isinstance(i, str) for i in case["affected_txn_ids"])
        assert isinstance(case["first_suspicious_txn_id"], str)
        assert isinstance(case["exposure_usd"], (int, float))
        assert isinstance(case["evidence"], list)
        assert isinstance(case["similar_prior_cases"], list)
        assert "next_best_actions" in outcome
        assert set(outcome["next_best_actions"].keys()) == {"initial", "final", "what_changed"}


# --- G11-D: post-reasoning ("final") G2 pass wired into investigate_and_assemble ---

def test_final_next_best_actions_reflects_post_reasoning_g2_pass():
    ledger = _base_ledger()  # no tool_calls, no cross_card evidence -- nothing fires pre-reasoning
    g2_result = g2_policy.evaluate(ledger)
    assert g2_result["recommended"] == []

    call_llm = _fake_call_llm(_valid_llm_response(fraud_probability=0.5))  # clears the section-3a 0.30 threshold
    outcome = run(co.investigate_and_assemble(ledger, g2_result, call_llm=call_llm))
    assert outcome["ok"] is True
    nba = outcome["next_best_actions"]
    assert nba["initial"] == []
    assert any(a["action"] == "CREATE_CASE" for a in nba["final"])
    assert nba["what_changed"] != "nothing"
    assert "CREATE_CASE" in nba["what_changed"]


def test_what_changed_is_nothing_when_final_equals_initial():
    ledger = _base_ledger()
    g2_result = g2_policy.evaluate(ledger)
    call_llm = _fake_call_llm(_valid_llm_response(verdict="legitimate", fraud_probability=0.1))  # below every new threshold
    outcome = run(co.investigate_and_assemble(ledger, g2_result, call_llm=call_llm))
    assert outcome["ok"] is True
    nba = outcome["next_best_actions"]
    assert nba["initial"] == nba["final"]
    assert nba["what_changed"] == "nothing"


def test_what_changed_is_deterministic_across_repeated_calls():
    ledger = _base_ledger()
    g2_result = g2_policy.evaluate(ledger)
    call_llm = _fake_call_llm(_valid_llm_response(fraud_probability=0.5))
    outcome_1 = run(co.investigate_and_assemble(ledger, g2_result, call_llm=call_llm))
    outcome_2 = run(co.investigate_and_assemble(ledger, g2_result, call_llm=call_llm))
    assert outcome_1["next_best_actions"] == outcome_2["next_best_actions"]


def test_final_g2_pass_evaluated_exactly_once_on_success(monkeypatch):
    real_evaluate = g2_policy.evaluate
    calls = []

    def counting_evaluate(ledger, validated=None, customer_response=None):
        calls.append((validated, customer_response))
        return real_evaluate(ledger, validated, customer_response)

    monkeypatch.setattr(co.g2_policy, "evaluate", counting_evaluate)

    ledger = _base_ledger()
    # fraud_probability=0.9 keeps this fixture above the G12 evidence-loop
    # gate's own <0.70 ceiling (src.agent.evidence_loop._should_request_
    # evidence) -- the point of THIS test is the no-evidence-request case,
    # covered separately below (test_final_g2_pass_evaluated_twice_when_
    # evidence_request_is_simulated) for the case where it fires.
    call_llm = _fake_call_llm(_valid_llm_response(fraud_probability=0.9))
    outcome = run(co.investigate_and_assemble(ledger, _base_g2_result(), call_llm=call_llm))
    assert outcome["ok"] is True
    assert len(calls) == 1  # investigate_and_assemble calls g2_policy.evaluate exactly once itself
    assert calls[0][0] is not None  # always the post-reasoning call, with validated
    assert calls[0][1] is None  # no evidence request simulated -- no customer_response


# --- G12: evidence-request loop wired into investigate_and_assemble ---

def test_final_g2_pass_evaluated_twice_when_evidence_request_is_simulated(monkeypatch):
    real_evaluate = g2_policy.evaluate
    calls = []

    def counting_evaluate(ledger, validated=None, customer_response=None):
        calls.append((validated, customer_response))
        return real_evaluate(ledger, validated, customer_response)

    monkeypatch.setattr(co.g2_policy, "evaluate", counting_evaluate)

    ledger = _base_ledger()
    # verdict "uncertain" (default) + fraud_probability 0.3 (default) is
    # exactly the G12 gate's own weak-single-signal condition.
    call_llm = _fake_call_llm(_valid_llm_response())
    outcome = run(co.investigate_and_assemble(ledger, _base_g2_result(), call_llm=call_llm))
    assert outcome["ok"] is True
    assert len(calls) == 2
    assert calls[0][1] is None  # first pass: no customer_response
    assert calls[1][1] in ("confirmed", "denied", "no_reply")  # second pass: simulated response applied
    assert len(outcome["evidence_requests"]) == 1


def test_evidence_requests_absent_for_confident_legitimate_verdict():
    ledger = _base_ledger()
    call_llm = _fake_call_llm(_valid_llm_response(verdict="legitimate", fraud_probability=0.05))
    outcome = run(co.investigate_and_assemble(ledger, _base_g2_result(), call_llm=call_llm))
    assert outcome["ok"] is True
    assert outcome["evidence_requests"] == []
    assert outcome["next_best_actions"]["what_changed"] == "nothing"


def test_evidence_requests_absent_for_confident_fraud_verdict():
    ledger = _base_ledger()
    call_llm = _fake_call_llm(_valid_llm_response(verdict="fraud", fraud_probability=0.95))
    outcome = run(co.investigate_and_assemble(ledger, _base_g2_result(), call_llm=call_llm))
    assert outcome["ok"] is True
    assert outcome["evidence_requests"] == []


def test_evidence_requests_present_and_shaped_for_weak_uncertain_signal():
    ledger = _base_ledger()
    call_llm = _fake_call_llm(_valid_llm_response(verdict="uncertain", fraud_probability=0.3))
    outcome = run(co.investigate_and_assemble(ledger, _base_g2_result(), call_llm=call_llm))
    assert outcome["ok"] is True
    assert len(outcome["evidence_requests"]) == 1
    request = outcome["evidence_requests"][0]
    assert set(request.keys()) == {"type", "asked_after_step", "assumed_response"}
    assert request["type"] == "customer_validation"  # no R5 pattern match in the default fixture
    assert request["asked_after_step"] == 0  # default fixture's tool_calls is []
    assert isinstance(request["assumed_response"], str) and request["assumed_response"]


def test_stop_reason_mentions_evidence_request_when_one_was_simulated():
    ledger = _base_ledger()
    call_llm = _fake_call_llm(_valid_llm_response(verdict="uncertain", fraud_probability=0.3))
    outcome = run(co.investigate_and_assemble(ledger, _base_g2_result(), call_llm=call_llm))
    assert outcome["ok"] is True
    assert "evidence request" in outcome["stop_reason"]


def test_final_g2_pass_not_evaluated_when_llm_call_fails(monkeypatch):
    calls = []
    monkeypatch.setattr(co.g2_policy, "evaluate", lambda *a, **k: calls.append((a, k)))

    async def failing_call_llm(prompt):
        raise RuntimeError("simulated provider failure")

    outcome = run(co.investigate_and_assemble(_base_ledger(), _base_g2_result(), call_llm=failing_call_llm))
    assert outcome["ok"] is False
    assert calls == []


def test_final_g2_pass_not_evaluated_when_validation_fails(monkeypatch):
    calls = []
    monkeypatch.setattr(co.g2_policy, "evaluate", lambda *a, **k: calls.append((a, k)))

    call_llm = _fake_call_llm(_valid_llm_response(verdict="definitely_fraud"))  # rejected by reasoning_validator
    outcome = run(co.investigate_and_assemble(_base_ledger(), _base_g2_result(), call_llm=call_llm))
    assert outcome["ok"] is False
    assert calls == []


def test_post_reasoning_g2_pass_makes_zero_mcp_or_tigergraph_calls(monkeypatch):
    from src.graph import tigergraph_connection as tgc
    from src.mcp_server import server as mcp_server_module

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("post-reasoning G2 pass must never call TigerGraph/MCP directly")

    monkeypatch.setattr(tgc, "get_connection", _fail_if_called)
    monkeypatch.setattr(mcp_server_module.mcp, "call_tool", _fail_if_called)

    ledger = _base_ledger()
    g2_result = g2_policy.evaluate(ledger)
    call_llm = _fake_call_llm(_valid_llm_response(fraud_probability=0.9))  # exercises the new section-3a branch
    outcome = run(co.investigate_and_assemble(ledger, g2_result, call_llm=call_llm))
    assert outcome["ok"] is True  # would have raised above if either entry point were touched
    assert any(a["action"] == "CREATE_CASE" for a in outcome["next_best_actions"]["final"])


def test_next_best_actions_with_explicit_final_argument_differs_from_initial_only_argument():
    g2_initial = _base_g2_result(recommended=[])
    g2_final = _base_g2_result(recommended=[{"action": "ESCALATE_TO_ANALYST", "route": "auto", "reason": "R8: ..."}])
    actions = co._next_best_actions(g2_initial, g2_final)
    assert actions["initial"] == []
    assert actions["final"] == [{"action": "ESCALATE_TO_ANALYST", "route": "auto", "reason": "R8: ..."}]
    assert actions["what_changed"] == "added ESCALATE_TO_ANALYST once the validated reasoning output (verdict/fraud_probability/pattern/exposure_usd) was available"


def test_evidence_list_returns_empty_when_no_applicable_evidence_exists():
    ledger = _base_ledger()  # default: recommendation_basis.reasons == [], pattern not matched, no R6
    assert co._evidence_list(ledger, _base_g2_result()) == []


def test_evidence_list_r5_behavior_unchanged():
    ledger = _base_ledger(pattern_evidence={
        "matched": True, "pattern": "card_testing",
        "candidates": [{"authorization_txn_ids": [1, 2, 3], "larger_purchase_txn_id": 4}],
        "uncertainty": [],
    })
    evidence = co._evidence_list(ledger, _base_g2_result())
    assert len(evidence) == 1
    assert evidence[0]["ref"] == "pattern:classify_card_testing"
    assert evidence[0]["entity_ids"] == ["1", "2", "3", "4"]


def test_evidence_list_r6_behavior_unchanged():
    g2_result = _base_g2_result(recommended=[
        {"action": "CREATE_CASE", "route": "auto", "reason": "R6: shared non-hub device/region evidence shows independently confirmed fraud on another card"},
        {"action": "FILE_REPORT", "route": "L2", "reason": "R6: shared non-hub device/region evidence shows independently confirmed fraud on another card"},
    ])
    evidence = co._evidence_list(_base_ledger(), g2_result)
    assert len(evidence) == 1  # one R6 citation, not one per action
    assert evidence[0]["ref"] == "tool:cross_card_fraud_verification"
    assert evidence[0]["entity_ids"] == []  # no Q9 data in the default fixture -- nothing to cite


_R6_G2_RESULT = _base_g2_result(recommended=[
    {"action": "CREATE_CASE", "route": "auto", "reason": "R6: shared non-hub device/region evidence shows independently confirmed fraud on another card"},
])


def _ledger_with_cross_card(cross_card: dict) -> dict:
    return _base_ledger(full_evidence={
        "combined_evidence": _base_ledger()["full_evidence"]["combined_evidence"],
        "cross_card_fraud_verification": cross_card,
    })


# --- G11-I: R6 evidence entity_ids populated from Q9's real result ---

def test_r6_via_device_confirmed_fraud_cites_real_card_key_and_case_ids():
    ledger = _ledger_with_cross_card({
        "via_device": {"other_cards": [
            {"card_key": "C99999:1", "customer_id": "C99999", "same_customer": False, "has_confirmed_fraud": True, "confirmed_fraud_case_ids": ["CC-9"]},
        ]},
        "via_region": None,
    })
    evidence = co._evidence_list(ledger, _R6_G2_RESULT)
    assert evidence[0]["entity_ids"] == ["C99999:1", "CC-9"]


def test_r6_via_region_confirmed_fraud_cites_real_card_key_and_case_ids():
    ledger = _ledger_with_cross_card({
        "via_device": None,
        "via_region": {"other_cards": [
            {"card_key": "C88888:2", "customer_id": "C88888", "same_customer": False, "has_confirmed_fraud": True, "confirmed_fraud_case_ids": ["CC-5"]},
        ]},
    })
    evidence = co._evidence_list(ledger, _R6_G2_RESULT)
    assert evidence[0]["entity_ids"] == ["C88888:2", "CC-5"]


def test_r6_overlapping_ids_across_paths_are_deduplicated_and_sorted():
    ledger = _ledger_with_cross_card({
        "via_device": {"other_cards": [
            {"card_key": "C99999:1", "customer_id": "C99999", "same_customer": False, "has_confirmed_fraud": True, "confirmed_fraud_case_ids": ["CC-9"]},
        ]},
        "via_region": {"other_cards": [
            {"card_key": "C99999:1", "customer_id": "C99999", "same_customer": False, "has_confirmed_fraud": True, "confirmed_fraud_case_ids": ["CC-9", "CC-2"]},
        ]},
    })
    evidence = co._evidence_list(ledger, _R6_G2_RESULT)
    assert evidence[0]["entity_ids"] == ["C99999:1", "CC-2", "CC-9"]  # deduplicated, sorted


def test_r6_no_confirmed_fraud_entries_cites_no_ids():
    ledger = _ledger_with_cross_card({
        "via_device": {"other_cards": [
            {"card_key": "C77777:1", "customer_id": "C77777", "same_customer": False, "has_confirmed_fraud": False, "confirmed_fraud_case_ids": []},
        ]},
        "via_region": None,
    })
    evidence = co._evidence_list(ledger, _R6_G2_RESULT)
    assert evidence[0]["entity_ids"] == []


def test_r6_missing_q9_result_cites_no_ids():
    ledger = _ledger_with_cross_card(None)
    evidence = co._evidence_list(ledger, _R6_G2_RESULT)
    assert evidence[0]["entity_ids"] == []


def test_r6_trimmed_sample_shape_still_cites_available_ids():
    ledger = _ledger_with_cross_card({
        "via_device": {"other_cards": {"count": 800, "sample": [
            {"card_key": "C11111:1", "customer_id": "C11111", "same_customer": False, "has_confirmed_fraud": True, "confirmed_fraud_case_ids": ["CC-3"]},
        ]}},
        "via_region": None,
    })
    evidence = co._evidence_list(ledger, _R6_G2_RESULT)
    assert evidence[0]["entity_ids"] == ["C11111:1", "CC-3"]


def test_r6_customer_id_never_included_in_entity_ids():
    ledger = _ledger_with_cross_card({
        "via_device": {"other_cards": [
            {"card_key": "C99999:1", "customer_id": "C99999", "same_customer": False, "has_confirmed_fraud": True, "confirmed_fraud_case_ids": []},
        ]},
        "via_region": None,
    })
    evidence = co._evidence_list(ledger, _R6_G2_RESULT)
    assert "C99999" not in evidence[0]["entity_ids"]  # bare customer_id never cited


def test_r6_claim_and_ref_unchanged_when_entity_ids_are_populated():
    ledger = _ledger_with_cross_card({
        "via_device": {"other_cards": [
            {"card_key": "C99999:1", "customer_id": "C99999", "same_customer": False, "has_confirmed_fraud": True, "confirmed_fraud_case_ids": ["CC-9"]},
        ]},
        "via_region": None,
    })
    evidence = co._evidence_list(ledger, _R6_G2_RESULT)
    assert evidence[0]["claim"] == _R6_G2_RESULT["recommended"][0]["reason"]
    assert evidence[0]["ref"] == "tool:cross_card_fraud_verification"
    assert evidence[0]["source"] == "graph"


def test_r5_and_recommendation_basis_ordering_unaffected_by_r6_entity_id_change():
    ledger = _ledger_with_cross_card({
        "via_device": {"other_cards": [
            {"card_key": "C99999:1", "customer_id": "C99999", "same_customer": False, "has_confirmed_fraud": True, "confirmed_fraud_case_ids": ["CC-9"]},
        ]},
        "via_region": None,
    })
    ledger["pattern_evidence"] = {"matched": True, "pattern": "card_testing", "candidates": [{"authorization_txn_ids": [1], "larger_purchase_txn_id": 2}], "uncertainty": []}
    ledger["recommendation_basis"] = {"reasons": ["direct_confirmed_fraud"], "tool_call_orders": [1]}
    evidence = co._evidence_list(ledger, _R6_G2_RESULT)
    assert len(evidence) == 3
    assert evidence[0]["ref"] == "pattern:classify_card_testing"
    assert evidence[1]["ref"] == "tool:cross_card_fraud_verification"
    assert evidence[1]["entity_ids"] == ["C99999:1", "CC-9"]
    assert evidence[2]["ref"] == "g1:recommendation_basis"


# --- G11-H: recommendation_basis-driven evidence ---

def test_direct_confirmed_fraud_produces_evidence_with_real_case_id():
    ledger = _base_ledger(recommendation_basis={"reasons": ["direct_confirmed_fraud"], "tool_call_orders": [1]})
    evidence = co._evidence_list(ledger, _base_g2_result())
    assert len(evidence) == 1
    entry = evidence[0]
    assert entry["source"] == "graph"
    assert entry["ref"] == "g1:recommendation_basis"
    assert entry["entity_ids"] == ["CC-1"]  # only the confirmed_fraud one -- CC-2 (cleared) is excluded


def test_connected_confirmed_fraud_produces_evidence_with_real_case_id():
    ledger = _base_ledger(
        recommendation_basis={"reasons": ["connected_confirmed_fraud"], "tool_call_orders": [1]},
        full_evidence={
            "combined_evidence": {
                **_base_ledger()["full_evidence"]["combined_evidence"],
                "historical_cases": {
                    "direct_cases": [],
                    "connected_cases": [{"case_id": "CC-9", "outcome": "confirmed_fraud"}, {"case_id": "CC-8", "outcome": "cleared"}],
                },
            },
            "cross_card_fraud_verification": None,
        },
    )
    evidence = co._evidence_list(ledger, _base_g2_result())
    assert len(evidence) == 1
    assert evidence[0]["entity_ids"] == ["CC-9"]
    assert "CC-8" not in evidence[0]["entity_ids"]


def test_sibling_confirmed_fraud_produces_evidence_with_real_case_id_from_round2_tool_call():
    ledger = _base_ledger(
        recommendation_basis={"reasons": ["sibling_confirmed_fraud:C99999:1"], "tool_call_orders": [3]},
        tool_calls=[{
            "order": 3,
            "tool": "historical_case_evidence",
            "arguments": {"card_key": "C99999:1"},
            "success": True,
            "curated_result_summary": {
                "direct_cases": [{"case_id": "CC-77", "outcome": "confirmed_fraud"}],
                "connected_cases": [],
            },
        }],
    )
    evidence = co._evidence_list(ledger, _base_g2_result())
    assert len(evidence) == 1
    assert "C99999:1" in evidence[0]["claim"]
    assert evidence[0]["entity_ids"] == ["CC-77"]


def test_sibling_confirmed_fraud_with_trimmed_sample_shape_still_cites_available_ids():
    # workflow.py's _trim() collapses a >5-entry list to {count, sample} --
    # the sibling lookup must read whatever is actually available in
    # "sample" rather than skip the whole entry.
    ledger = _base_ledger(
        recommendation_basis={"reasons": ["sibling_confirmed_fraud:C777:1"], "tool_call_orders": [3]},
        tool_calls=[{
            "order": 3,
            "tool": "historical_case_evidence",
            "arguments": {"card_key": "C777:1"},
            "success": True,
            "curated_result_summary": {
                "direct_cases": {"count": 8, "sample": [{"case_id": "CC-1", "outcome": "confirmed_fraud"}]},
                "connected_cases": [],
            },
        }],
    )
    evidence = co._evidence_list(ledger, _base_g2_result())
    assert evidence[0]["entity_ids"] == ["CC-1"]


def test_sibling_confirmed_fraud_with_missing_tool_call_result_cites_no_fabricated_ids():
    # The reason references a card whose historical_case_evidence result
    # isn't in tool_calls at all (e.g. trimmed away or never recorded) --
    # must return [], never invent an id.
    ledger = _base_ledger(recommendation_basis={"reasons": ["sibling_confirmed_fraud:C000:1"], "tool_call_orders": [3]})
    evidence = co._evidence_list(ledger, _base_g2_result())
    assert evidence[0]["entity_ids"] == []


def test_narrow_device_evidence_produces_evidence_with_device_profile():
    ledger = _base_ledger(
        recommendation_basis={"reasons": ["narrow_device_evidence"], "tool_call_orders": []},
        full_evidence={
            "combined_evidence": {
                **_base_ledger()["full_evidence"]["combined_evidence"],
                "device_evidence": {"device_profile": "SAMSUNG SM-G892A", "hub_flag": False},
            },
            "cross_card_fraud_verification": None,
        },
    )
    evidence = co._evidence_list(ledger, _base_g2_result())
    assert len(evidence) == 1
    assert evidence[0]["entity_ids"] == ["SAMSUNG SM-G892A"]


def test_narrow_device_evidence_without_device_profile_cites_no_fabricated_id():
    ledger = _base_ledger(recommendation_basis={"reasons": ["narrow_device_evidence"], "tool_call_orders": []})
    # default fixture's device_evidence is None -- no device_profile to cite
    evidence = co._evidence_list(ledger, _base_g2_result())
    assert evidence[0]["entity_ids"] == []


def test_multiple_recommendation_basis_reasons_produce_multiple_distinct_entries():
    ledger = _base_ledger(recommendation_basis={
        "reasons": ["direct_confirmed_fraud", "sibling_confirmed_fraud:C99999:1"],
        "tool_call_orders": [1, 3],
    }, tool_calls=[{
        "order": 3, "tool": "historical_case_evidence", "arguments": {"card_key": "C99999:1"}, "success": True,
        "curated_result_summary": {"direct_cases": [{"case_id": "CC-77", "outcome": "confirmed_fraud"}], "connected_cases": []},
    }])
    evidence = co._evidence_list(ledger, _base_g2_result())
    assert len(evidence) == 2
    assert evidence[0]["entity_ids"] == ["CC-1"]
    assert evidence[1]["entity_ids"] == ["CC-77"]


def test_duplicate_recommendation_basis_reason_does_not_produce_duplicate_entries():
    ledger = _base_ledger(recommendation_basis={
        "reasons": ["direct_confirmed_fraud", "direct_confirmed_fraud"], "tool_call_orders": [1],
    })
    evidence = co._evidence_list(ledger, _base_g2_result())
    assert len(evidence) == 1


def test_recommendation_basis_evidence_appended_after_existing_r5_and_r6_entries():
    ledger = _base_ledger(
        pattern_evidence={"matched": True, "pattern": "card_testing", "candidates": [{"authorization_txn_ids": [1], "larger_purchase_txn_id": 2}], "uncertainty": []},
        recommendation_basis={"reasons": ["direct_confirmed_fraud"], "tool_call_orders": [1]},
    )
    g2_result = _base_g2_result(recommended=[
        {"action": "CREATE_CASE", "route": "auto", "reason": "R6: shared non-hub device/region evidence shows independently confirmed fraud on another card"},
    ])
    evidence = co._evidence_list(ledger, g2_result)
    assert len(evidence) == 3
    assert evidence[0]["ref"] == "pattern:classify_card_testing"  # R5 first, unchanged position
    assert evidence[1]["ref"] == "tool:cross_card_fraud_verification"  # R6 second, unchanged position
    assert evidence[2]["ref"] == "g1:recommendation_basis"  # new entries appended last


def test_no_new_evidence_when_recommendation_basis_reasons_are_unrecognized():
    # Defensive: an unrecognized reason string (e.g. from a future G1
    # change) must never crash this function or fabricate an entry.
    ledger = _base_ledger(recommendation_basis={"reasons": ["some_future_reason"], "tool_call_orders": []})
    assert co._evidence_list(ledger, _base_g2_result()) == []


def test_next_best_actions_describes_removed_actions_too():
    # Constructed only to prove _describe_action_diff's "removed" branch --
    # g2_policy.evaluate itself never removes an action between passes (a
    # RECOMMENDED action's condition stays true once met), this is a
    # direct unit test of the diff-description helper, not a claim about
    # real G2 behavior.
    g2_initial = _base_g2_result(recommended=[{"action": "DECLINE_TRANSACTION", "route": "L1", "reason": "R5"}])
    g2_final = _base_g2_result(recommended=[])
    actions = co._next_best_actions(g2_initial, g2_final)
    assert actions["what_changed"] == "removed DECLINE_TRANSACTION once the validated reasoning output (verdict/fraud_probability/pattern/exposure_usd) was available"


# --- G13: SAR (Part 2) construction ---

_R6_QUALIFYING = {"other_cards": [
    {"card_key": "C99999:1", "customer_id": "C99999", "same_customer": False, "has_confirmed_fraud": True, "confirmed_fraud_case_ids": ["CC-9"]},
]}


def _sar_ready_ledger(**overrides):
    return _base_ledger(
        tool_calls=[{
            "order": 1, "tool": "cross_card_fraud_verification", "arguments": {"card_key": "C00001:1"},
            "success": True, "curated_result_summary": {"via_device": _R6_QUALIFYING, "via_region": None},
        }],
        full_evidence={
            "combined_evidence": {
                "transaction": {"TransactionID": 100, "ts": "2016-11-14 09:12:00", "TransactionAmt": 50.0},
                "card": {"card_key": "C00001:1"},
                "customer": {"customer_id": "C00001"},
                "temporal_activity": [
                    {"TransactionID": 100, "ts": "2016-11-14 09:12:00", "TransactionAmt": 50.0, "channel": "online"},
                    {"TransactionID": 101, "ts": "2016-11-15 10:00:00", "TransactionAmt": 20.0, "channel": "online"},
                ],
                "device_evidence": {"device_profile": "SAMSUNG SM-G892A Build/NRD90M", "hub_flag": False},
                "billing_region": {"addr1": 444.0, "hub_flag": True},
                "historical_cases": {"direct_cases": [], "connected_cases": []},
                "connected_cards": {"connected_via_case": [], "connected_via_device": []},
            },
            "cross_card_fraud_verification": {"via_device": _R6_QUALIFYING, "via_region": None},
        },
        **overrides,
    )


def test_sar_file_true_produces_grounded_narrative_subjects_amount_and_dates():
    ledger = _sar_ready_ledger()
    call_llm = _fake_call_llm(_valid_llm_response(
        verdict="fraud", fraud_probability=0.9, affected_txn_ids=[100, 101], first_suspicious_txn_id=100,
        pattern="account_takeover", summary="Two transactions on card C00001:1 show mixed-channel activity inconsistent with the cardholder.",
    ))
    outcome = run(co.investigate_and_assemble(ledger, _base_g2_result(), call_llm=call_llm))
    assert outcome["ok"] is True
    sar = outcome["sar"]
    assert sar["file"] is True
    assert sar["reason"].startswith("R6")
    assert "SAMSUNG SM-G892A Build/NRD90M" in sar["reason"]  # G13: names the actual shared element

    # narrative: 6-12 sentences, non-empty, grounded
    sentence_count = sar["narrative"].count(". ") + 1
    assert 6 <= sentence_count <= 12
    assert "C00001" in sar["narrative"]
    assert "SAMSUNG SM-G892A Build/NRD90M" in sar["narrative"]

    # subjects: populated, deduplicated, no invented ids
    assert "C00001" in sar["subjects"]
    assert "C00001:1" in sar["subjects"]
    assert "SAMSUNG SM-G892A Build/NRD90M" in sar["subjects"]
    assert "C99999:1" in sar["subjects"]  # the real R6-connected card
    assert len(sar["subjects"]) == len(set(sar["subjects"]))  # deduplicated

    # total_amount_usd: exactly the validator's own exposure_usd, never recomputed
    assert sar["total_amount_usd"] == outcome["case"]["exposure_usd"]

    # activity_dates: chronological [first, last], derived from real timestamps
    assert sar["activity_dates"] == ["2016-11-14", "2016-11-15"]


def test_sar_file_false_produces_exact_empty_values():
    ledger = _base_ledger()  # no cross_card evidence -- nothing grounds FILE_REPORT
    call_llm = _fake_call_llm(_valid_llm_response(verdict="legitimate", fraud_probability=0.05))
    outcome = run(co.investigate_and_assemble(ledger, _base_g2_result(), call_llm=call_llm))
    assert outcome["ok"] is True
    assert outcome["sar"] == {"file": False, "reason": "", "narrative": "", "subjects": [], "total_amount_usd": 0, "activity_dates": []}


def test_sar_narrative_contains_no_ids_outside_known_grounded_set():
    # Every card_key/customer_id/device string that appears in the
    # narrative must come from this ledger's own known evidence -- never
    # a fabricated id.
    ledger = _sar_ready_ledger()
    call_llm = _fake_call_llm(_valid_llm_response(
        verdict="fraud", fraud_probability=0.9, affected_txn_ids=[100, 101], first_suspicious_txn_id=100,
        pattern="account_takeover",
    ))
    outcome = run(co.investigate_and_assemble(ledger, _base_g2_result(), call_llm=call_llm))
    sar = outcome["sar"]
    known_ids = {"C00001", "C00001:1", "C99999:1", "SAMSUNG SM-G892A Build/NRD90M", "CC-9"}
    assert set(sar["subjects"]) <= known_ids


def test_sar_activity_dates_repeat_single_date_when_only_one_date_present():
    ledger = _sar_ready_ledger()
    ledger["full_evidence"]["combined_evidence"]["temporal_activity"] = [
        {"TransactionID": 100, "ts": "2016-11-14 09:12:00", "TransactionAmt": 50.0, "channel": "online"},
    ]
    call_llm = _fake_call_llm(_valid_llm_response(
        verdict="fraud", fraud_probability=0.9, affected_txn_ids=[100], first_suspicious_txn_id=100, pattern="account_takeover",
    ))
    outcome = run(co.investigate_and_assemble(ledger, _base_g2_result(), call_llm=call_llm))
    assert outcome["sar"]["activity_dates"] == ["2016-11-14", "2016-11-14"]


def test_sar_total_amount_usd_never_recalculated_from_raw_transactions():
    ledger = _sar_ready_ledger()
    call_llm = _fake_call_llm(_valid_llm_response(
        verdict="fraud", fraud_probability=0.9, affected_txn_ids=[100], first_suspicious_txn_id=100, pattern="account_takeover",
    ))
    outcome = run(co.investigate_and_assemble(ledger, _base_g2_result(), call_llm=call_llm))
    # Only txn 100 ($50.0) was grounded into affected_txn_ids -- txn 101
    # ($20.0) exists in temporal_activity but was never claimed by the
    # LLM, so it must not silently inflate total_amount_usd.
    assert outcome["sar"]["total_amount_usd"] == 50.0
    assert outcome["sar"]["total_amount_usd"] == outcome["case"]["exposure_usd"]


# --- G13 review fix 1: narrative verdict sentence is grounded in
# validated["verdict"], never a hardcoded "confirmed fraud" claim ---

def test_sar_narrative_verdict_sentence_reflects_fraud_verdict():
    ledger = _base_ledger()
    validated = {"verdict": "fraud", "fraud_probability": 0.9, "pattern": "none", "pattern_description": "",
                 "affected_txn_ids": [100], "exposure_usd": 50.0}
    case = {"first_suspicious_txn_id": "100", "similar_prior_cases": [], "summary": ""}
    narrative = co._sar_narrative(ledger, validated, case, "R2: customer denies the transaction")
    assert "reached a 'fraud' verdict" in narrative
    assert "confirmed fraud" not in narrative


def test_sar_narrative_verdict_sentence_reflects_uncertain_verdict_never_claims_confirmed_fraud():
    # The exact gap the G13 review flagged: R2's FILE_REPORT branch does
    # NOT require validated["verdict"] == "fraud" (unlike R6/R9), so a
    # FILE_REPORT-triggering result can carry any verdict. The narrative
    # must say so honestly, never assert "confirmed fraud" regardless.
    ledger = _base_ledger()
    validated = {"verdict": "uncertain", "fraud_probability": 0.55, "pattern": "none", "pattern_description": "",
                 "affected_txn_ids": [100], "exposure_usd": 1500.0}
    case = {"first_suspicious_txn_id": "100", "similar_prior_cases": [], "summary": ""}
    narrative = co._sar_narrative(ledger, validated, case, "R2: customer denies the transaction; exposure $1500.0 exceeds $1,000")
    assert "reached a 'uncertain' verdict" in narrative
    assert "confirmed fraud" not in narrative
    assert "0.55" in narrative


# --- G13 review fix 2: connected-card evidence is included by real
# evidence presence, never by parsing the reason string for "R6" -- must
# work identically for R9 ---

def _r9_ready_ledger(**overrides):
    ledger = _sar_ready_ledger()
    ledger.update(overrides)
    return ledger


def test_r9_file_report_includes_connected_card_keys_in_subjects():
    ledger = _r9_ready_ledger()  # same Q9 evidence as the R6 fixture; same_customer=False satisfies R9 too
    call_llm = _fake_call_llm(_valid_llm_response(
        verdict="fraud", fraud_probability=0.9, affected_txn_ids=[100], first_suspicious_txn_id=100,
        pattern="undocumented", pattern_description="A coordinated pattern across unrelated customers.",
    ))
    outcome = run(co.investigate_and_assemble(ledger, _base_g2_result(), call_llm=call_llm))
    assert outcome["sar"]["reason"].startswith("R9")  # confirms this went through the R9 path, not R6
    assert "C99999:1" in outcome["sar"]["subjects"]  # the real R9/Q9-connected card, previously omitted


def test_r9_file_report_includes_connected_card_sentence_in_narrative():
    ledger = _r9_ready_ledger()
    call_llm = _fake_call_llm(_valid_llm_response(
        verdict="fraud", fraud_probability=0.9, affected_txn_ids=[100], first_suspicious_txn_id=100,
        pattern="undocumented", pattern_description="A coordinated pattern across unrelated customers.",
    ))
    outcome = run(co.investigate_and_assemble(ledger, _base_g2_result(), call_llm=call_llm))
    assert outcome["sar"]["reason"].startswith("R9")
    assert "C99999:1" in outcome["sar"]["narrative"]
    assert "already show confirmed fraud" in outcome["sar"]["narrative"]


def test_r6_connected_card_subjects_and_sentence_still_present_after_fix():
    # R6 behavior must be unchanged by switching the gate from a reason-
    # string check to a direct evidence check -- R6's own Q9 evidence is
    # real, so the connected card must still appear both places.
    ledger = _sar_ready_ledger()
    call_llm = _fake_call_llm(_valid_llm_response(
        verdict="fraud", fraud_probability=0.9, affected_txn_ids=[100], first_suspicious_txn_id=100,
        pattern="account_takeover",  # not "undocumented" -- takes the R6 path, not R9
    ))
    outcome = run(co.investigate_and_assemble(ledger, _base_g2_result(), call_llm=call_llm))
    assert outcome["sar"]["reason"].startswith("R6")
    assert "C99999:1" in outcome["sar"]["subjects"]
    assert "C99999:1" in outcome["sar"]["narrative"]
    assert "already show confirmed fraud" in outcome["sar"]["narrative"]


def test_sar_subjects_and_narrative_omit_connected_cards_when_no_qualifying_q9_evidence():
    # R2 path: FILE_REPORT fires from a customer denial, not Q9 evidence.
    # No cross_card_fraud_verification data exists in this ledger at all,
    # so _sar_connected_card_keys must be empty and neither subjects nor
    # narrative should mention any connected card.
    ledger = _base_ledger()
    validated = {"verdict": "uncertain", "fraud_probability": 0.5, "pattern": "none", "pattern_description": "",
                 "affected_txn_ids": [100], "exposure_usd": 1500.0}
    case = {"first_suspicious_txn_id": "100", "similar_prior_cases": [], "summary": ""}
    reason = "R2: customer denies the transaction (simulated response); exposure $1500.0 exceeds $1,000"
    assert co._sar_subjects(ledger, reason) == ["C00001", "C00001:1"]
    assert "already show confirmed fraud" not in co._sar_narrative(ledger, validated, case, reason)


# --- submission-audit fix: sar.subjects capped to the same 5 connected
# cards _sar_narrative actually names in prose ---

_R6_QUALIFYING_MANY = {"other_cards": [
    {"card_key": f"C9000{i}:1", "customer_id": f"C9000{i}", "same_customer": False, "has_confirmed_fraud": True, "confirmed_fraud_case_ids": [f"CC-{i}"]}
    for i in range(7)
]}


def _sar_ready_ledger_many_connected(**overrides):
    ledger = _sar_ready_ledger()
    ledger["tool_calls"] = [{
        "order": 1, "tool": "cross_card_fraud_verification", "arguments": {"card_key": "C00001:1"},
        "success": True, "curated_result_summary": {"via_device": _R6_QUALIFYING_MANY, "via_region": None},
    }]
    ledger["full_evidence"]["cross_card_fraud_verification"] = {"via_device": _R6_QUALIFYING_MANY, "via_region": None}
    ledger.update(overrides)
    return ledger


def test_sar_subjects_capped_at_five_connected_cards_with_more_than_five_available():
    ledger = _sar_ready_ledger_many_connected()  # 7 real confirmed-fraud connected cards
    call_llm = _fake_call_llm(_valid_llm_response(
        verdict="fraud", fraud_probability=0.9, affected_txn_ids=[100], first_suspicious_txn_id=100, pattern="account_takeover",
    ))
    outcome = run(co.investigate_and_assemble(ledger, _base_g2_result(), call_llm=call_llm))
    sar = outcome["sar"]
    connected_in_subjects = [s for s in sar["subjects"] if s.startswith("C9000")]
    assert len(connected_in_subjects) == 5  # capped, not all 7


def test_sar_subjects_connected_cards_are_exactly_the_ones_named_in_narrative():
    ledger = _sar_ready_ledger_many_connected()
    call_llm = _fake_call_llm(_valid_llm_response(
        verdict="fraud", fraud_probability=0.9, affected_txn_ids=[100], first_suspicious_txn_id=100, pattern="account_takeover",
    ))
    outcome = run(co.investigate_and_assemble(ledger, _base_g2_result(), call_llm=call_llm))
    sar = outcome["sar"]
    connected_in_subjects = {s for s in sar["subjects"] if s.startswith("C9000")}
    connected_in_narrative = {s for s in co._sar_connected_card_keys(ledger)[:5] if s in sar["narrative"]}
    assert connected_in_subjects == connected_in_narrative
    assert len(connected_in_narrative) == 5
    assert "and others" in sar["narrative"]  # confirms the narrative itself also discloses truncation


def test_sar_no_connected_card_evidence_fabricated_or_lost_underlying_ledger():
    # The underlying ledger's own Q9 evidence must be untouched by the cap
    # -- _sar_connected_card_keys(ledger) itself still returns all 7 real
    # entries; only the SAR-facing subjects/narrative are capped for
    # display, nothing is deleted from the source of truth.
    ledger = _sar_ready_ledger_many_connected()
    all_connected = co._sar_connected_card_keys(ledger)
    assert len(all_connected) == 7
    assert all(cid.startswith("C9000") for cid in all_connected)
    call_llm = _fake_call_llm(_valid_llm_response(
        verdict="fraud", fraud_probability=0.9, affected_txn_ids=[100], first_suspicious_txn_id=100, pattern="account_takeover",
    ))
    outcome = run(co.investigate_and_assemble(ledger, _base_g2_result(), call_llm=call_llm))
    subjects = set(outcome["sar"]["subjects"])
    # every subject that IS present must be a real, ungrounded-nothing id
    assert subjects <= set(all_connected) | {"C00001", "C00001:1", "SAMSUNG SM-G892A Build/NRD90M"}


def test_r9_subjects_also_capped_at_five_connected_cards():
    ledger = _sar_ready_ledger_many_connected()
    call_llm = _fake_call_llm(_valid_llm_response(
        verdict="fraud", fraud_probability=0.9, affected_txn_ids=[100], first_suspicious_txn_id=100,
        pattern="undocumented", pattern_description="A coordinated pattern across unrelated customers.",
    ))
    outcome = run(co.investigate_and_assemble(ledger, _base_g2_result(), call_llm=call_llm))
    assert outcome["sar"]["reason"].startswith("R9")
    connected_in_subjects = [s for s in outcome["sar"]["subjects"] if s.startswith("C9000")]
    assert len(connected_in_subjects) == 5


def test_r6_still_names_connected_cards_after_cap_fix_when_five_or_fewer():
    # R6 with <=5 connected cards (the common case) must still include all
    # of them, unchanged from before this fix -- the cap only matters once
    # there are more than 5.
    ledger = _sar_ready_ledger()  # exactly 1 connected card
    call_llm = _fake_call_llm(_valid_llm_response(
        verdict="fraud", fraud_probability=0.9, affected_txn_ids=[100], first_suspicious_txn_id=100, pattern="account_takeover",
    ))
    outcome = run(co.investigate_and_assemble(ledger, _base_g2_result(), call_llm=call_llm))
    assert "C99999:1" in outcome["sar"]["subjects"]


def test_sar_file_false_subjects_unaffected_by_cap_fix():
    ledger = _base_ledger()
    call_llm = _fake_call_llm(_valid_llm_response(verdict="legitimate", fraud_probability=0.05))
    outcome = run(co.investigate_and_assemble(ledger, _base_g2_result(), call_llm=call_llm))
    assert outcome["sar"] == {"file": False, "reason": "", "narrative": "", "subjects": [], "total_amount_usd": 0, "activity_dates": []}
