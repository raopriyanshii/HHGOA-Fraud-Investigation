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

    def counting_evaluate(ledger, validated=None):
        calls.append(validated)
        return real_evaluate(ledger, validated)

    monkeypatch.setattr(co.g2_policy, "evaluate", counting_evaluate)

    ledger = _base_ledger()
    call_llm = _fake_call_llm(_valid_llm_response())
    outcome = run(co.investigate_and_assemble(ledger, _base_g2_result(), call_llm=call_llm))
    assert outcome["ok"] is True
    assert len(calls) == 1  # investigate_and_assemble calls g2_policy.evaluate exactly once itself
    assert calls[0] is not None  # always the post-reasoning call, with validated


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
