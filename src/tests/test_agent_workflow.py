"""
Phase G1 unit tests for src/agent/workflow.py.

Every test here uses a fake `call_tool` (mocked MCP dispatch) so none of
this depends on a live TigerGraph connection -- these test the workflow's
own orchestration logic (round structure, hard limits, the V1
recommendation rule, error handling), not Q1-Q8's graph correctness,
which Phase E's own test suite already covers separately.

No asyncio test plugin is required or added (requirements.txt is frozen
for this phase) -- every test is a plain sync `def`, driving the async
workflow with asyncio.run().
"""
import asyncio

from src.agent import workflow as wf


def run(coro):
    return asyncio.run(coro)


def _base_evidence(**overrides):
    evidence = {
        "transaction": {"TransactionID": 999, "ts": "2016-01-01 00:00:00"},
        "card": {"card_key": "C00001:1"},
        "customer": {"customer_id": "C00001"},
        "temporal_activity": [{"TransactionID": 999, "ts": "2016-01-01 00:00:00"}],
        "device_evidence": None,
        "historical_cases": {"direct_cases": [], "connected_cases": []},
        "connected_cards": {"connected_via_case": [], "connected_via_device": []},
        "billing_region": {"hub_flag": True},
    }
    evidence.update(overrides)
    return evidence


def make_call_tool(responses: dict):
    """responses: {tool_name: value} for a fixed response, or
    {tool_name: {arg_key_value: value}} for per-argument responses
    (used for historical_case_evidence, keyed by card_key). A value of
    ("error", "message text") produces a failed call instead.
    """
    calls = []

    async def call_tool(name, arguments):
        calls.append((name, dict(arguments)))
        entry = responses[name]
        if isinstance(entry, dict) and name == "historical_case_evidence":
            value = entry[arguments["card_key"]]
        else:
            value = entry
        if isinstance(value, tuple) and value[0] == "error":
            return False, None, value[1]
        return True, value, None

    call_tool.calls = calls
    return call_tool


# --- 1. flagged transaction trigger ---

def test_flagged_txn_trigger_calls_combined_evidence_once():
    call_tool = make_call_tool({"combined_evidence": _base_evidence()})
    result = run(wf.investigate({"type": "flagged_txn_id", "value": 3514030}, call_tool=call_tool))
    assert call_tool.calls == [("combined_evidence", {"flagged_txn_id": 3514030, "window_hours": 24.0})]
    assert result["trigger"] == {"type": "flagged_txn_id", "value": 3514030}
    assert len(result["tool_calls"]) == 1


# --- 2. case-ID trigger ---

def test_case_id_trigger_resolves_then_calls_combined_evidence():
    call_tool = make_call_tool({
        "benchmark_case_resolution": {"flagged_txn_id": 3514030, "case_id": "HHG-001"},
        "combined_evidence": _base_evidence(),
    })
    result = run(wf.investigate({"type": "case_id", "value": "HHG-001"}, call_tool=call_tool))
    assert call_tool.calls == [
        ("benchmark_case_resolution", {"case_id": "HHG-001"}),
        ("combined_evidence", {"flagged_txn_id": 3514030, "window_hours": 24.0}),
    ]
    assert result["tool_calls"][0]["order"] == 1
    assert result["tool_calls"][1]["order"] == 2


# --- 3. no connected cards -> no Round 2 ---

def test_empty_connected_via_case_skips_round_2():
    call_tool = make_call_tool({"combined_evidence": _base_evidence()})
    result = run(wf.investigate({"type": "flagged_txn_id", "value": 1}, call_tool=call_tool))
    assert not any(tc["tool"] == "historical_case_evidence" for tc in result["tool_calls"])


# --- 4. connected cards -> maximum 3 follow-ups ---

def test_more_than_three_connected_cards_caps_at_three():
    connected = [
        {"connected_card_key": f"C{i:05d}:1", "connected_customer_id": f"C{i:05d}"}
        for i in range(5)
    ]
    evidence = _base_evidence(connected_cards={"connected_via_case": connected, "connected_via_device": []})
    call_tool = make_call_tool({
        "combined_evidence": evidence,
        "historical_case_evidence": {c["connected_card_key"]: {"direct_cases": [], "connected_cases": []} for c in connected},
    })
    result = run(wf.investigate({"type": "flagged_txn_id", "value": 1}, call_tool=call_tool))
    followups = [tc for tc in result["tool_calls"] if tc["tool"] == "historical_case_evidence"]
    assert len(followups) == 3


# --- 5. deterministic card ordering ---

def test_follow_up_selection_is_sorted_ascending_by_card_key():
    connected = [
        {"connected_card_key": "C00003:1", "connected_customer_id": "C00003"},
        {"connected_card_key": "C00001:1", "connected_customer_id": "C00001"},
        {"connected_card_key": "C00002:1", "connected_customer_id": "C00002"},
        {"connected_card_key": "C00005:1", "connected_customer_id": "C00005"},
    ]
    evidence = _base_evidence(connected_cards={"connected_via_case": connected, "connected_via_device": []})
    call_tool = make_call_tool({
        "combined_evidence": evidence,
        "historical_case_evidence": {c["connected_card_key"]: {"direct_cases": [], "connected_cases": []} for c in connected},
    })
    result = run(wf.investigate({"type": "flagged_txn_id", "value": 1}, call_tool=call_tool))
    followup_keys = [tc["arguments"]["card_key"] for tc in result["tool_calls"] if tc["tool"] == "historical_case_evidence"]
    assert followup_keys == ["C00001:1", "C00002:1", "C00003:1"]


# --- 6. confirmed fraud -> escalate ---

def test_direct_confirmed_fraud_escalates():
    evidence = _base_evidence(historical_cases={
        "direct_cases": [{"case_id": "CC-1", "outcome": "confirmed_fraud"}],
        "connected_cases": [],
    })
    call_tool = make_call_tool({"combined_evidence": evidence})
    result = run(wf.investigate({"type": "flagged_txn_id", "value": 1}, call_tool=call_tool))
    assert result["recommendation"] == "escalate"
    assert "direct_confirmed_fraud" in result["recommendation_basis"]["reasons"]


# --- 7. non-hub device evidence without fraud -> monitor ---

def test_narrow_device_evidence_without_fraud_monitors():
    evidence = _base_evidence(device_evidence={"hub_flag": False, "n_distinct_customers": 3, "cards": []})
    # Non-hub device_evidence trips the G4 Q9 gate -- register a neutral
    # (no cross-card evidence) response so this test still isolates G1's
    # own monitor-tier logic, not G4's R6 evidence.
    call_tool = make_call_tool({"combined_evidence": evidence, "cross_card_fraud_verification": {"via_device": None, "via_region": None}})
    result = run(wf.investigate({"type": "flagged_txn_id", "value": 1}, call_tool=call_tool))
    assert result["recommendation"] == "monitor"
    assert result["recommendation_basis"]["reasons"] == ["narrow_device_evidence"]


# --- 7b. recommendation precedence: confirmed_fraud beats a monitor-level signal ---

def test_confirmed_fraud_takes_precedence_over_device_monitor_signal():
    # Both a priority-1 (escalate) and a priority-2 (monitor) condition are
    # true at once. Flagged in the G1 review audit: no prior test proved
    # branch 1's early-return actually wins -- each branch was only ever
    # tested in isolation.
    evidence = _base_evidence(
        historical_cases={"direct_cases": [{"case_id": "CC-1", "outcome": "confirmed_fraud"}], "connected_cases": []},
        device_evidence={"hub_flag": False, "n_distinct_customers": 3, "cards": []},
    )
    # Non-hub device_evidence trips the G4 Q9 gate here too -- neutral response,
    # same reasoning as test_narrow_device_evidence_without_fraud_monitors above.
    call_tool = make_call_tool({"combined_evidence": evidence, "cross_card_fraud_verification": {"via_device": None, "via_region": None}})
    result = run(wf.investigate({"type": "flagged_txn_id", "value": 1}, call_tool=call_tool))
    assert result["recommendation"] == "escalate"
    assert "direct_confirmed_fraud" in result["recommendation_basis"]["reasons"]
    # branch 2 must never even be considered once branch 1 matches:
    assert "narrow_device_evidence" not in result["recommendation_basis"]["reasons"]


# --- 8. no qualifying evidence -> insufficient_evidence ---

def test_no_evidence_at_all_is_insufficient_evidence():
    call_tool = make_call_tool({"combined_evidence": _base_evidence()})
    result = run(wf.investigate({"type": "flagged_txn_id", "value": 1}, call_tool=call_tool))
    assert result["recommendation"] == "insufficient_evidence"


# --- 9. cleared history does not escalate (nor monitor) ---

def test_cleared_history_alone_is_insufficient_evidence():
    evidence = _base_evidence(historical_cases={
        "direct_cases": [{"case_id": "CC-1", "outcome": "cleared"}],
        "connected_cases": [],
    })
    call_tool = make_call_tool({"combined_evidence": evidence})
    result = run(wf.investigate({"type": "flagged_txn_id", "value": 1}, call_tool=call_tool))
    assert result["recommendation"] == "insufficient_evidence"


def test_cleared_connected_case_alone_is_insufficient_evidence():
    # Same property as the direct-cases test above, but for connected_cases --
    # _has_confirmed_fraud is applied identically to both lists, but this
    # was previously untested for connected_cases specifically (flagged in
    # the G1 review audit as a coverage gap, not a known defect).
    evidence = _base_evidence(historical_cases={
        "direct_cases": [],
        "connected_cases": [{"case_id": "CC-2", "outcome": "cleared", "relationship": "connected_to"}],
    })
    call_tool = make_call_tool({"combined_evidence": evidence})
    result = run(wf.investigate({"type": "flagged_txn_id", "value": 1}, call_tool=call_tool))
    assert result["recommendation"] == "insufficient_evidence"


# --- 10. Round-2 confirmed fraud escalates ---

def test_round_2_sibling_confirmed_fraud_escalates():
    connected = [{"connected_card_key": "C00001:1", "connected_customer_id": "C00001"}]
    evidence = _base_evidence(connected_cards={"connected_via_case": connected, "connected_via_device": []})
    call_tool = make_call_tool({
        "combined_evidence": evidence,
        "historical_case_evidence": {"C00001:1": {"direct_cases": [{"case_id": "CC-9", "outcome": "confirmed_fraud"}], "connected_cases": []}},
    })
    result = run(wf.investigate({"type": "flagged_txn_id", "value": 1}, call_tool=call_tool))
    assert result["recommendation"] == "escalate"
    assert any(r.startswith("sibling_confirmed_fraud") for r in result["recommendation_basis"]["reasons"])


# --- 11. Round-2 cannot trigger another round ---

def test_round_2_finding_does_not_trigger_a_third_round():
    connected = [{"connected_card_key": "C00001:1", "connected_customer_id": "C00001"}]
    evidence = _base_evidence(connected_cards={"connected_via_case": connected, "connected_via_device": []})
    call_tool = make_call_tool({
        "combined_evidence": evidence,
        "historical_case_evidence": {"C00001:1": {"direct_cases": [{"case_id": "CC-9", "outcome": "confirmed_fraud"}], "connected_cases": []}},
    })
    result = run(wf.investigate({"type": "flagged_txn_id", "value": 1}, call_tool=call_tool))
    # exactly 1 (combined_evidence) + 1 (the single follow-up) -- nothing more,
    # even though that follow-up revealed confirmed fraud.
    assert len(result["tool_calls"]) == 2


# --- 12. MCP failure is safely represented ---

def test_mcp_failure_is_captured_and_does_not_crash_the_investigation():
    call_tool = make_call_tool({"combined_evidence": ("error", "connection reset by peer")})
    result = run(wf.investigate({"type": "flagged_txn_id", "value": 1}, call_tool=call_tool))
    assert result["recommendation"] == "insufficient_evidence"
    assert result["tool_calls"][0]["success"] is False
    assert result["tool_calls"][0]["error"] == "connection reset by peer"
    assert len(result["uncertainties"]) == 1


def test_default_call_tool_scrubs_secrets_before_they_reach_the_ledger(monkeypatch):
    # Exercises the real _default_call_tool (production call_tool), not the
    # test double above -- this is the one place scrub_secret is actually
    # applied, so it needs its own direct test rather than relying on the
    # mocked-call_tool tests (which bypass _default_call_tool entirely).
    async def fake_mcp_call_tool(name, arguments):
        raise RuntimeError("connection failed, Authorization: Bearer sk-fake-not-a-real-token-abcdef123456")

    monkeypatch.setattr(wf.mcp, "call_tool", fake_mcp_call_tool)
    ok, value, error = run(wf._default_call_tool("combined_evidence", {"flagged_txn_id": 1, "window_hours": 24.0}))
    assert ok is False
    assert value is None
    assert "sk-fake-not-a-real-token-abcdef123456" not in error
    assert "[REDACTED]" in error


# --- 13. hard limits cannot be exceeded ---

def test_hard_limit_cannot_be_exceeded_even_with_many_connected_cards():
    connected = [
        {"connected_card_key": f"C{i:05d}:1", "connected_customer_id": f"C{i:05d}"}
        for i in range(50)
    ]
    evidence = _base_evidence(connected_cards={"connected_via_case": connected, "connected_via_device": []})
    call_tool = make_call_tool({
        "combined_evidence": evidence,
        "historical_case_evidence": {c["connected_card_key"]: {"direct_cases": [], "connected_cases": []} for c in connected},
    })
    result = run(wf.investigate({"type": "flagged_txn_id", "value": 1}, call_tool=call_tool))
    followups = [tc for tc in result["tool_calls"] if tc["tool"] == "historical_case_evidence"]
    assert len(followups) == wf.MAX_FOLLOW_UP_CARDS
    assert len(result["tool_calls"]) == 1 + wf.MAX_FOLLOW_UP_CARDS


# --- G4: Q9 gate (Option B) + effective Round-2 cap, ceiling validation ---

def _three_connected_cards():
    return [{"connected_card_key": f"C{i:05d}:1", "connected_customer_id": f"C{i:05d}"} for i in range(3)]


def test_ceiling_case_id_trigger_q9_gate_true_three_connected_cards_stays_at_five():
    connected = _three_connected_cards()
    evidence = _base_evidence(
        device_evidence={"hub_flag": False, "n_distinct_customers": 3, "cards": []},
        connected_cards={"connected_via_case": connected, "connected_via_device": []},
    )
    call_tool = make_call_tool({
        "benchmark_case_resolution": {"flagged_txn_id": 1, "case_id": "HHG-001"},
        "combined_evidence": evidence,
        "cross_card_fraud_verification": {"via_device": None, "via_region": None},
        "historical_case_evidence": {c["connected_card_key"]: {"direct_cases": [], "connected_cases": []} for c in connected},
    })
    result = run(wf.investigate({"type": "case_id", "value": "HHG-001"}, call_tool=call_tool))
    assert len(result["tool_calls"]) == 5
    followups = [tc for tc in result["tool_calls"] if tc["tool"] == "historical_case_evidence"]
    assert len(followups) == 2, "effective cap must reduce from 3 to 2 when Q9 fires"
    assert any(tc["tool"] == "cross_card_fraud_verification" for tc in result["tool_calls"])


def test_ceiling_flagged_txn_id_trigger_q9_gate_true_three_connected_cards():
    connected = _three_connected_cards()
    evidence = _base_evidence(
        device_evidence={"hub_flag": False, "n_distinct_customers": 3, "cards": []},
        connected_cards={"connected_via_case": connected, "connected_via_device": []},
    )
    call_tool = make_call_tool({
        "combined_evidence": evidence,
        "cross_card_fraud_verification": {"via_device": None, "via_region": None},
        "historical_case_evidence": {c["connected_card_key"]: {"direct_cases": [], "connected_cases": []} for c in connected},
    })
    result = run(wf.investigate({"type": "flagged_txn_id", "value": 1}, call_tool=call_tool))
    assert len(result["tool_calls"]) == 4  # no resolution call for this trigger type -- below the 5 ceiling
    followups = [tc for tc in result["tool_calls"] if tc["tool"] == "historical_case_evidence"]
    assert len(followups) == 2


def test_ceiling_q9_gate_false_three_connected_cards_keeps_full_round_2():
    connected = _three_connected_cards()
    evidence = _base_evidence(connected_cards={"connected_via_case": connected, "connected_via_device": []})
    # device_evidence stays None and billing_region stays hub-flagged (the
    # _base_evidence defaults) -- the gate must not fire.
    call_tool = make_call_tool({
        "combined_evidence": evidence,
        "historical_case_evidence": {c["connected_card_key"]: {"direct_cases": [], "connected_cases": []} for c in connected},
    })
    result = run(wf.investigate({"type": "flagged_txn_id", "value": 1}, call_tool=call_tool))
    assert not any(tc["tool"] == "cross_card_fraud_verification" for tc in result["tool_calls"])
    followups = [tc for tc in result["tool_calls"] if tc["tool"] == "historical_case_evidence"]
    assert len(followups) == 3, "full cap must be unreduced when Q9 never fires"


def test_q9_failure_is_recorded_and_investigation_continues_with_reduced_cap():
    connected = _three_connected_cards()
    evidence = _base_evidence(
        device_evidence={"hub_flag": False, "n_distinct_customers": 3, "cards": []},
        connected_cards={"connected_via_case": connected, "connected_via_device": []},
    )
    call_tool = make_call_tool({
        "combined_evidence": evidence,
        "cross_card_fraud_verification": ("error", "connection reset by peer"),
        "historical_case_evidence": {c["connected_card_key"]: {"direct_cases": [], "connected_cases": []} for c in connected},
    })
    result = run(wf.investigate({"type": "flagged_txn_id", "value": 1}, call_tool=call_tool))
    q9_call = next(tc for tc in result["tool_calls"] if tc["tool"] == "cross_card_fraud_verification")
    assert q9_call["success"] is False
    assert any("cross_card_fraud_verification failed" in u for u in result["uncertainties"])
    followups = [tc for tc in result["tool_calls"] if tc["tool"] == "historical_case_evidence"]
    assert len(followups) == 2, "cap stays reduced even though the Q9 call itself failed"
    assert result["recommendation"] in ("escalate", "monitor", "insufficient_evidence")  # did not crash


def test_q9_gate_true_but_no_connected_cards_still_calls_q9_and_skips_round_2():
    evidence = _base_evidence(device_evidence={"hub_flag": False, "n_distinct_customers": 3, "cards": []})
    call_tool = make_call_tool({
        "combined_evidence": evidence,
        "cross_card_fraud_verification": {"via_device": None, "via_region": None},
    })
    result = run(wf.investigate({"type": "flagged_txn_id", "value": 1}, call_tool=call_tool))
    assert any(tc["tool"] == "cross_card_fraud_verification" for tc in result["tool_calls"])
    assert not any(tc["tool"] == "historical_case_evidence" for tc in result["tool_calls"])
    assert len(result["tool_calls"]) == 2


# --- extra: large lists are trimmed in the ledger, not in the recommendation logic ---

def test_large_list_is_trimmed_in_ledger_but_still_used_for_recommendation():
    many_devices = [{"card_key": f"C{i:05d}:1", "customer_id": f"C{i:05d}"} for i in range(200)]
    evidence = _base_evidence(
        device_evidence={"hub_flag": False, "n_distinct_customers": 200, "cards": many_devices},
    )
    # Non-hub device_evidence trips the G4 Q9 gate -- neutral response, same
    # reasoning as the other two fixed-up tests above.
    call_tool = make_call_tool({"combined_evidence": evidence, "cross_card_fraud_verification": {"via_device": None, "via_region": None}})
    result = run(wf.investigate({"type": "flagged_txn_id", "value": 1}, call_tool=call_tool))
    # recommendation logic still sees the real (untrimmed) evidence:
    assert result["recommendation"] == "monitor"
    # but the ledger's stored copy is bounded:
    stored_device_evidence = result["tool_calls"][0]["curated_result_summary"]["device_evidence"]
    stored_cards = stored_device_evidence["cards"]
    assert isinstance(stored_cards, dict) and stored_cards["count"] == 200
    assert len(stored_cards["sample"]) == wf._MAX_LIST_SAMPLE
