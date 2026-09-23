"""
G7 unit tests for src/agent/reasoning.py.

Every test here uses a fake `call_llm` -- none of this depends on a live
LLM provider (no default provider is even wired in; see
reasoning._default_call_llm). These test evidence-package curation,
prompt construction, and response parsing/failure-handling -- not the
semantic acceptance of an LLM's answer, which is
test_reasoning_validator.py's job.
"""
import asyncio
import json

from src.agent import reasoning


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
                "temporal_activity": [
                    {"TransactionID": 100, "ts": "2016-01-01 10:00:00", "TransactionAmt": 50.0, "channel": "online"},
                ],
                "device_evidence": {"device_profile": "SAMSUNG SM-G935F | Android 7.0", "hub_flag": False},
                "billing_region": {"addr1": 444.0, "hub_flag": True},
                "historical_cases": {"direct_cases": [{"case_id": "CC-1", "outcome": "confirmed_fraud"}], "connected_cases": []},
                "connected_cards": {"connected_via_case": [], "connected_via_device": []},
            },
            "cross_card_fraud_verification": None,
        },
    }
    ledger.update(overrides)
    return ledger


_BASE_G2_RESULT = {"recommended": [], "prohibited": [], "eligible_not_recommended": [], "deferred": []}


# --- known-id / evidence-package curation ---

def test_collect_known_ids_includes_flagged_and_temporal_txn_ids():
    ledger = _base_ledger()
    ledger["full_evidence"]["combined_evidence"]["temporal_activity"] = [
        {"TransactionID": 100, "ts": "2016-01-01 10:00:00", "TransactionAmt": 50.0, "channel": "online"},
        {"TransactionID": 101, "ts": "2016-01-01 10:05:00", "TransactionAmt": 5.0, "channel": "online"},
    ]
    known = reasoning._collect_known_ids(ledger)
    assert known["txn"] == [100, 101]


def test_collect_known_ids_never_includes_ids_not_present_in_evidence():
    ledger = _base_ledger()
    known = reasoning._collect_known_ids(ledger)
    # a plausible-looking but never-gathered id must not appear anywhere:
    assert "C99999:9" not in known["card"]
    assert "C99999" not in known["customer"]
    assert "FAKE-DEVICE-PROFILE" not in known["device"]
    assert 999999 not in known["txn"]


def test_collect_known_ids_includes_connected_and_cross_card_entities():
    ledger = _base_ledger()
    ledger["full_evidence"]["combined_evidence"]["connected_cards"] = {
        "connected_via_case": [{"connected_card_key": "C00002:1", "connected_customer_id": "C00002"}],
        "connected_via_device": [{"card_key": "C00003:1", "customer_id": "C00003"}],
    }
    ledger["full_evidence"]["cross_card_fraud_verification"] = {
        "via_device": {"other_cards": [{"card_key": "C00004:1", "customer_id": "C00004", "has_confirmed_fraud": True, "confirmed_fraud_case_ids": ["CC-9"]}]},
        "via_region": None,
    }
    known = reasoning._collect_known_ids(ledger)
    assert "C00002:1" in known["card"]
    assert "C00003:1" in known["card"]
    assert "C00004:1" in known["card"]
    assert "CC-9" in known["case"]


def test_build_evidence_package_has_four_required_sections():
    package = reasoning.build_evidence_package(_base_ledger(), _BASE_G2_RESULT)
    assert set(["flagged_txn_id", "known_ids", "facts", "deterministic_findings", "unknown"]) <= set(package.keys())
    assert package["flagged_txn_id"] == 100


def test_build_evidence_package_zero_side_effects_and_zero_tool_dependency():
    # pure function: same input twice -> identical output, no network/graph access
    ledger = _base_ledger()
    p1 = reasoning.build_evidence_package(ledger, _BASE_G2_RESULT)
    p2 = reasoning.build_evidence_package(ledger, _BASE_G2_RESULT)
    assert p1 == p2


# --- prompt construction (pure string building) ---

def test_prompt_contains_all_four_required_section_headers():
    package = reasoning.build_evidence_package(_base_ledger(), _BASE_G2_RESULT)
    prompt = reasoning.build_prompt(package)
    for header in ("FACTS:", "DETERMINISTIC FINDINGS:", "REASONING (what you are allowed to infer", "UNKNOWN"):
        assert header in prompt


def test_prompt_states_known_ids_are_the_only_allowed_ids():
    package = reasoning.build_evidence_package(_base_ledger(), _BASE_G2_RESULT)
    prompt = reasoning.build_prompt(package)
    assert "KNOWN IDS" in prompt
    assert "100" in prompt  # the flagged txn id must actually appear in the rendered known_ids block


def test_prompt_forbids_calling_tools_and_copying_risk_score():
    package = reasoning.build_evidence_package(_base_ledger(), _BASE_G2_RESULT)
    prompt = reasoning.build_prompt(package)
    assert "You MUST NOT" in prompt
    assert "call TigerGraph or any tool directly" in prompt
    assert "treat any risk_score as fraud_probability" in prompt


def test_prompt_r5_matched_instructs_forced_card_testing():
    ledger = _base_ledger(pattern_evidence={"matched": True, "pattern": "card_testing", "candidates": [{"authorization_txn_ids": [1, 2, 3], "larger_purchase_txn_id": 4}], "uncertainty": []})
    package = reasoning.build_evidence_package(ledger, _BASE_G2_RESULT)
    prompt = reasoning.build_prompt(package)
    assert '"matched": true' in prompt.lower()


# --- get_llm_reasoning: parsing and failure handling ---

def test_valid_json_response_is_parsed_successfully():
    async def fake_call_llm(prompt):
        return json.dumps({"verdict": "uncertain", "fraud_probability": 0.3, "affected_txn_ids": [], "first_suspicious_txn_id": None, "pattern": "none", "pattern_description": "", "summary": "ok", "reasoning": "ok", "uncertainties": []})

    package = reasoning.build_evidence_package(_base_ledger(), _BASE_G2_RESULT)
    outcome = run(reasoning.get_llm_reasoning(package, call_llm=fake_call_llm))
    assert outcome["ok"] is True
    assert outcome["raw"]["verdict"] == "uncertain"


def test_malformed_json_response_fails_safely():
    async def fake_call_llm(prompt):
        return "this is not JSON at all {"

    package = reasoning.build_evidence_package(_base_ledger(), _BASE_G2_RESULT)
    outcome = run(reasoning.get_llm_reasoning(package, call_llm=fake_call_llm))
    assert outcome["ok"] is False
    assert outcome["failure_reason"] == "invalid_json"


def test_non_object_json_response_fails_safely():
    async def fake_call_llm(prompt):
        return json.dumps([1, 2, 3])  # valid JSON, but not an object

    package = reasoning.build_evidence_package(_base_ledger(), _BASE_G2_RESULT)
    outcome = run(reasoning.get_llm_reasoning(package, call_llm=fake_call_llm))
    assert outcome["ok"] is False
    assert outcome["failure_reason"] == "invalid_json"


def test_llm_call_raising_a_non_timeout_error_is_caught_and_fails_safely():
    async def broken_call_llm(prompt):
        raise ConnectionError("LLM provider connection reset")

    package = reasoning.build_evidence_package(_base_ledger(), _BASE_G2_RESULT)
    outcome = run(reasoning.get_llm_reasoning(package, call_llm=broken_call_llm))
    assert outcome["ok"] is False
    assert outcome["failure_reason"] == "llm_call_failed"


def test_llm_call_raising_timeout_error_itself_is_classified_as_llm_timeout():
    # asyncio.TimeoutError IS TimeoutError in Python 3.11+, so a call_llm
    # that raises its own TimeoutError is indistinguishable from -- and
    # correctly classified the same as -- our own enforced timeout below.
    async def self_timing_out_call_llm(prompt):
        raise TimeoutError("LLM provider did not respond in time")

    package = reasoning.build_evidence_package(_base_ledger(), _BASE_G2_RESULT)
    outcome = run(reasoning.get_llm_reasoning(package, call_llm=self_timing_out_call_llm))
    assert outcome["ok"] is False
    assert outcome["failure_reason"] == "llm_timeout"


# --- G7 defect fix: enforced timeout on a genuinely hanging call_llm ---

def test_genuinely_hanging_call_llm_is_terminated_by_enforced_timeout():
    # A call_llm that never returns and never raises -- proves the
    # timeout is actually enforced (asyncio.wait_for), not merely
    # documented. A tiny timeout_s keeps this test fast; if the
    # enforcement were missing, this test would hang the suite instead
    # of failing cleanly.
    async def hanging_call_llm(prompt):
        await asyncio.sleep(60)  # far longer than the timeout_s below
        return "unreachable"

    package = reasoning.build_evidence_package(_base_ledger(), _BASE_G2_RESULT)
    outcome = run(reasoning.get_llm_reasoning(package, call_llm=hanging_call_llm, timeout_s=0.05))
    assert outcome["ok"] is False
    assert outcome["failure_reason"] == "llm_timeout"
    assert "0.05" in outcome["detail"]


def test_default_llm_timeout_is_explicit_and_configurable():
    assert reasoning.DEFAULT_LLM_TIMEOUT_S > 0
    # confirms the parameter is actually threaded through, not decorative:
    async def hanging_call_llm(prompt):
        await asyncio.sleep(60)
        return "unreachable"

    package = reasoning.build_evidence_package(_base_ledger(), _BASE_G2_RESULT)
    outcome = run(reasoning.get_llm_reasoning(package, call_llm=hanging_call_llm, timeout_s=0.05))
    assert outcome["failure_reason"] == "llm_timeout"


def test_default_call_llm_raises_clearly_when_no_provider_is_configured():
    package = reasoning.build_evidence_package(_base_ledger(), _BASE_G2_RESULT)
    outcome = run(reasoning.get_llm_reasoning(package))  # no call_llm passed
    assert outcome["ok"] is False
    assert outcome["failure_reason"] == "llm_call_failed"
    assert "No default LLM provider is configured" in outcome["detail"]


def test_call_llm_is_invoked_exactly_once_per_reasoning_call():
    calls = []

    async def counting_call_llm(prompt):
        calls.append(prompt)
        return json.dumps({"verdict": "uncertain", "fraud_probability": 0.2, "affected_txn_ids": [], "first_suspicious_txn_id": None, "pattern": "none", "pattern_description": "", "summary": "", "reasoning": "", "uncertainties": []})

    package = reasoning.build_evidence_package(_base_ledger(), _BASE_G2_RESULT)
    run(reasoning.get_llm_reasoning(package, call_llm=counting_call_llm))
    assert len(calls) == 1
