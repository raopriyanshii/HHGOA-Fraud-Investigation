"""
Phase G2 unit tests for src/policy/g2_policy.py, covering the approved
Revision 2 specification's test matrix exactly, plus the canonical-order
and repeated-evaluation determinism requirements added at approval time.

g2_policy.evaluate() is a pure function -- no MCP, no TigerGraph, no
mocking needed. Fixtures here are plain G1-ledger-shaped dicts, built by
hand to match src.agent.workflow's actual ledger shape (order/tool/
arguments/success/curated_result_summary for tool_calls; recommendation/
recommendation_basis/uncertainties at the top level).
"""
import copy
import json

from src.policy import g2_policy as g2


def _resolution_call(order, trigger_type, success=True):
    if success:
        return {
            "order": order,
            "tool": "benchmark_case_resolution",
            "arguments": {"case_id": "HHG-001"},
            "success": True,
            "curated_result_summary": {
                "case_id": "HHG-001",
                "customer_id": "C12382",
                "card_key": "C12382:21139",
                "trigger_type": trigger_type,
                "trigger_text": "irrelevant for this test",
                "flagged_txn_id": 3514030,
                "opened_at": "2016-12-05 01:55:28",
                "risk_score": 0.61,
                "status": "READY",
            },
        }
    return {
        "order": order,
        "tool": "benchmark_case_resolution",
        "arguments": {"case_id": "HHG-001"},
        "success": False,
        "error": "Error executing tool benchmark_case_resolution",
    }


def _base_ledger(**overrides):
    ledger = {
        "investigation_id": "test-fixture",
        "trigger": {"type": "flagged_txn_id", "value": 1},
        "tool_calls": [
            {
                "order": 1,
                "tool": "combined_evidence",
                "arguments": {"flagged_txn_id": 1, "window_hours": 24.0},
                "success": True,
                "curated_result_summary": {"device_evidence": None, "historical_cases": {"direct_cases": [], "connected_cases": []}},
            }
        ],
        "uncertainties": [],
        "recommendation": "insufficient_evidence",
        "recommendation_basis": {"reasons": [], "tool_call_orders": []},
    }
    ledger.update(overrides)
    return ledger


# --- 1. case_id trigger, trigger_type == customer_report -> CREATE_CASE RECOMMENDED ---

def test_customer_report_trigger_recommends_create_case():
    ledger = _base_ledger(tool_calls=[
        _resolution_call(1, "customer_report"),
        _base_ledger()["tool_calls"][0],
    ])
    result = g2.evaluate(ledger)
    actions = {e["action"]: e for e in result["recommended"]}
    assert "CREATE_CASE" in actions
    assert actions["CREATE_CASE"]["route"] == "auto"
    assert "customer_report" in actions["CREATE_CASE"]["reason"] or "customer disputes" in actions["CREATE_CASE"]["reason"]
    assert not any(e["action"] == "CREATE_CASE" for e in result["deferred"])


# --- 2/3. case_id trigger, trigger_type == risk_score / analyst_request -> CREATE_CASE DEFERRED ---

def test_risk_score_trigger_defers_create_case_with_distinguishable_reason():
    ledger = _base_ledger(tool_calls=[_resolution_call(1, "risk_score")])
    result = g2.evaluate(ledger)
    entry = next(e for e in result["deferred"] if e["action"] == "CREATE_CASE")
    assert "risk_score" in entry["reason"]
    assert not any(e["action"] == "CREATE_CASE" for e in result["recommended"])


def test_analyst_request_trigger_defers_create_case_with_distinguishable_reason():
    ledger = _base_ledger(tool_calls=[_resolution_call(1, "analyst_request")])
    result = g2.evaluate(ledger)
    entry = next(e for e in result["deferred"] if e["action"] == "CREATE_CASE")
    assert "analyst_request" in entry["reason"]


def test_risk_score_and_analyst_request_produce_different_deferred_reasons():
    r1 = g2.evaluate(_base_ledger(tool_calls=[_resolution_call(1, "risk_score")]))
    r2 = g2.evaluate(_base_ledger(tool_calls=[_resolution_call(1, "analyst_request")]))
    reason1 = next(e for e in r1["deferred"] if e["action"] == "CREATE_CASE")["reason"]
    reason2 = next(e for e in r2["deferred"] if e["action"] == "CREATE_CASE")["reason"]
    assert reason1 != reason2


# --- 4. flagged_txn_id trigger (no resolution call at all) -> CREATE_CASE DEFERRED, distinguishable ---

def test_flagged_txn_id_trigger_defers_create_case_as_missing_not_false():
    ledger = _base_ledger()  # default fixture: no benchmark_case_resolution call present
    result = g2.evaluate(ledger)
    entry = next(e for e in result["deferred"] if e["action"] == "CREATE_CASE")
    assert "flagged_txn_id" in entry["reason"] or "no case resolution" in entry["reason"]


# --- 5. benchmark_case_resolution present but failed -> CREATE_CASE DEFERRED, uncertainties passed through ---

def test_failed_resolution_call_defers_create_case_and_preserves_uncertainties():
    ledger = _base_ledger(
        tool_calls=[_resolution_call(1, None, success=False)],
        uncertainties=["trigger resolution failed: Error executing tool benchmark_case_resolution"],
    )
    result = g2.evaluate(ledger)
    entry = next(e for e in result["deferred"] if e["action"] == "CREATE_CASE")
    assert "failed" in entry["reason"]
    assert result["uncertainties"] == ["trigger resolution failed: Error executing tool benchmark_case_resolution"]


# --- 6. any ledger -> BLOCK_ALL_CARDS always PROHIBITED ---

def test_block_all_cards_always_prohibited_on_default_ledger():
    result = g2.evaluate(_base_ledger())
    assert any(e["action"] == "BLOCK_ALL_CARDS" for e in result["prohibited"])
    assert not any(e["action"] == "BLOCK_ALL_CARDS" for e in result["recommended"])
    assert not any(e["action"] == "BLOCK_ALL_CARDS" for e in result["eligible_not_recommended"])
    assert not any(e["action"] == "BLOCK_ALL_CARDS" for e in result["deferred"])


# --- 7. strongest possible G1 evidence -> BLOCK_ALL_CARDS still PROHIBITED ---

def test_block_all_cards_prohibited_even_with_strongest_g1_evidence():
    ledger = _base_ledger(
        recommendation="escalate",
        recommendation_basis={"reasons": ["direct_confirmed_fraud", "sibling_confirmed_fraud:C00001:1"], "tool_call_orders": [1, 2]},
        tool_calls=[
            {
                "order": 1,
                "tool": "combined_evidence",
                "arguments": {"flagged_txn_id": 1, "window_hours": 24.0},
                "success": True,
                "curated_result_summary": {
                    "historical_cases": {"direct_cases": [{"case_id": "CC-1", "outcome": "confirmed_fraud"}], "connected_cases": []},
                    "connected_cards": {"connected_via_case": [{"connected_card_key": "C00001:1", "connected_customer_id": "C00001"}], "connected_via_device": []},
                },
            },
            {
                "order": 2,
                "tool": "historical_case_evidence",
                "arguments": {"card_key": "C00001:1"},
                "success": True,
                "curated_result_summary": {"direct_cases": [{"case_id": "CC-9", "outcome": "confirmed_fraud"}], "connected_cases": []},
            },
        ],
    )
    result = g2.evaluate(ledger)
    assert any(e["action"] == "BLOCK_ALL_CARDS" for e in result["prohibited"])


# --- 7b. G1's recommendation tier must never drive CREATE_CASE (audit WARN #2) ---

def _strong_escalate_ledger(resolution_call=None):
    # Same strong-evidence shape as test 7 above (direct_confirmed_fraud +
    # sibling_confirmed_fraud, the strongest ledger G1 can produce), with an
    # optional benchmark_case_resolution call prepended so the trigger_type
    # can be varied independently of how strong the escalate evidence is.
    tool_calls = []
    order = 1
    if resolution_call is not None:
        resolution_call = dict(resolution_call, order=order)
        tool_calls.append(resolution_call)
        order += 1
    tool_calls.append({
        "order": order,
        "tool": "combined_evidence",
        "arguments": {"flagged_txn_id": 1, "window_hours": 24.0},
        "success": True,
        "curated_result_summary": {
            "historical_cases": {"direct_cases": [{"case_id": "CC-1", "outcome": "confirmed_fraud"}], "connected_cases": []},
            "connected_cards": {"connected_via_case": [{"connected_card_key": "C00001:1", "connected_customer_id": "C00001"}], "connected_via_device": []},
        },
    })
    order += 1
    tool_calls.append({
        "order": order,
        "tool": "historical_case_evidence",
        "arguments": {"card_key": "C00001:1"},
        "success": True,
        "curated_result_summary": {"direct_cases": [{"case_id": "CC-9", "outcome": "confirmed_fraud"}], "connected_cases": []},
    })
    return _base_ledger(
        recommendation="escalate",
        recommendation_basis={"reasons": ["direct_confirmed_fraud", "sibling_confirmed_fraud:C00001:1"], "tool_call_orders": [1, 2]},
        tool_calls=tool_calls,
    )


def test_strong_escalate_with_non_customer_report_trigger_does_not_recommend_create_case():
    # The exact adversarial combination the G2 audit flagged as untested:
    # the strongest possible G1 evidence (escalate, direct + sibling
    # confirmed fraud) paired with a trigger_type that is confirmed NOT a
    # customer dispute. If CREATE_CASE were ever accidentally coupled to
    # G1's recommendation tier instead of trigger_type alone, this is the
    # case that would catch it.
    ledger = _strong_escalate_ledger(resolution_call=_resolution_call(1, "risk_score"))
    result = g2.evaluate(ledger)
    assert not any(e["action"] == "CREATE_CASE" for e in result["recommended"])
    assert any(e["action"] == "CREATE_CASE" for e in result["deferred"])


def test_strong_escalate_with_missing_trigger_type_does_not_recommend_create_case():
    # Same strong evidence, but no benchmark_case_resolution call at all
    # (a flagged_txn_id-triggered investigation) -- trigger_type is
    # unknown, not confirmed non-customer_report, but CREATE_CASE must
    # still not be recommended.
    ledger = _strong_escalate_ledger(resolution_call=None)
    result = g2.evaluate(ledger)
    assert not any(e["action"] == "CREATE_CASE" for e in result["recommended"])
    assert any(e["action"] == "CREATE_CASE" for e in result["deferred"])


# --- 8. the other 12 actions -> always DEFERRED, on any ledger ---

def test_all_non_create_case_non_block_all_cards_actions_are_deferred():
    always_deferred = g2.OFFICIAL_ACTIONS - {"CREATE_CASE", "BLOCK_ALL_CARDS"}
    for ledger in (
        _base_ledger(),
        _base_ledger(tool_calls=[_resolution_call(1, "customer_report")]),
        _base_ledger(recommendation="monitor"),
    ):
        result = g2.evaluate(ledger)
        deferred_actions = {e["action"] for e in result["deferred"]}
        assert always_deferred <= deferred_actions
        for bucket in ("recommended", "prohibited", "eligible_not_recommended"):
            assert not (always_deferred & {e["action"] for e in result[bucket]})


# --- 9. malformed/unknown action names are rejected -- vocabulary integrity ---

def test_action_vocabulary_is_closed_and_matches_official_14():
    assert len(g2.CANONICAL_ACTION_ORDER) == 14
    assert len(g2.OFFICIAL_ACTIONS) == 14
    assert "NOT_A_REAL_ACTION" not in g2.OFFICIAL_ACTIONS
    assert "block_all_cards" not in g2.OFFICIAL_ACTIONS  # case-sensitive, no fuzzy matching
    expected = {
        "ALLOW_TRANSACTION", "DECLINE_TRANSACTION", "MONITOR_CARD", "MONITOR_CONNECTED_CARDS",
        "WARN_CUSTOMER", "VERIFY_WITH_CUSTOMER", "STEP_UP_AUTH", "BLOCK_CARD", "BLOCK_ALL_CARDS",
        "GENERATE_REPORT", "CREATE_CASE", "FILE_REPORT", "ESCALATE_TO_ANALYST", "CLOSE_NO_FRAUD",
    }
    assert g2.OFFICIAL_ACTIONS == expected


def test_evaluate_output_never_contains_an_action_outside_the_official_14():
    result = g2.evaluate(_base_ledger(tool_calls=[_resolution_call(1, "customer_report")]))
    all_actions = [e["action"] for bucket in ("recommended", "prohibited", "eligible_not_recommended", "deferred") for e in result[bucket]]
    assert all(a in g2.OFFICIAL_ACTIONS for a in all_actions)
    assert len(all_actions) == 14  # every action appears exactly once, total


# --- 10. risk_score is provably unused ---

def test_risk_score_value_never_affects_the_decision():
    low = _base_ledger(tool_calls=[
        {**_resolution_call(1, "risk_score")},
        {
            "order": 2, "tool": "combined_evidence", "arguments": {"flagged_txn_id": 3514030, "window_hours": 24.0},
            "success": True,
            "curated_result_summary": {"transaction": {"risk_score": 0.01}, "historical_cases": {"direct_cases": [], "connected_cases": []}},
        },
    ])
    high = _base_ledger(tool_calls=[
        {**_resolution_call(1, "risk_score")},
        {
            "order": 2, "tool": "combined_evidence", "arguments": {"flagged_txn_id": 3514030, "window_hours": 24.0},
            "success": True,
            "curated_result_summary": {"transaction": {"risk_score": 0.99}, "historical_cases": {"direct_cases": [], "connected_cases": []}},
        },
    ])
    assert g2.evaluate(low) == g2.evaluate(high)


# --- 11. canonical order: every output list follows CANONICAL_ACTION_ORDER ---

def test_deferred_actions_appear_in_exact_canonical_order():
    result = g2.evaluate(_base_ledger(tool_calls=[_resolution_call(1, "customer_report")]))
    expected_deferred_order = [a for a in g2.CANONICAL_ACTION_ORDER if a not in ("CREATE_CASE", "BLOCK_ALL_CARDS")]
    assert [e["action"] for e in result["deferred"]] == expected_deferred_order


def test_full_action_set_across_all_buckets_equals_canonical_order_when_concatenated_by_position():
    # Every action's bucket membership, checked against its canonical index --
    # concatenating recommended+prohibited+eligible_not_recommended+deferred
    # and sorting by each action's index in CANONICAL_ACTION_ORDER must
    # reproduce CANONICAL_ACTION_ORDER exactly (every action present, no
    # duplicates, no extras).
    result = g2.evaluate(_base_ledger(tool_calls=[_resolution_call(1, "customer_report")]))
    all_actions = [e["action"] for bucket in ("recommended", "prohibited", "eligible_not_recommended", "deferred") for e in result[bucket]]
    reconstructed = tuple(sorted(all_actions, key=g2.CANONICAL_ACTION_ORDER.index))
    assert reconstructed == g2.CANONICAL_ACTION_ORDER


# --- 12. repeated evaluation of the same ledger is byte-identical, including order ---

def test_repeated_evaluation_is_deterministic_including_order():
    ledger = _base_ledger(tool_calls=[_resolution_call(1, "customer_report")])
    results = [g2.evaluate(ledger) for _ in range(5)]
    for r in results[1:]:
        assert r == results[0]
    # also compare via JSON serialization (order-sensitive for lists) as an
    # independent check that doesn't rely solely on Python dict/list ==
    serialized = [json.dumps(r, sort_keys=False) for r in results]
    assert len(set(serialized)) == 1


def test_repeated_evaluation_deterministic_across_distinct_but_equal_ledger_objects():
    # Two separately-constructed ledger dicts with identical content (not
    # the same object) must still produce identical, identically-ordered output.
    ledger_a = _base_ledger(tool_calls=[_resolution_call(1, "risk_score")])
    ledger_b = _base_ledger(tool_calls=[_resolution_call(1, "risk_score")])
    assert ledger_a is not ledger_b
    assert g2.evaluate(ledger_a) == g2.evaluate(ledger_b)


# --- evaluate() must not mutate its input ledger (audit WARN #4) ---

def test_evaluate_does_not_mutate_the_input_ledger():
    ledger = _strong_escalate_ledger(resolution_call=_resolution_call(1, "customer_report"))
    ledger["uncertainties"] = ["a pre-existing uncertainty, to make sure this list isn't touched either"]
    before = copy.deepcopy(ledger)

    g2.evaluate(ledger)

    assert ledger == before


# --- investigation_recommendation pass-through ---

def test_investigation_recommendation_is_passed_through_unchanged():
    for tier in ("escalate", "monitor", "insufficient_evidence"):
        result = g2.evaluate(_base_ledger(recommendation=tier))
        assert result["investigation_recommendation"] == tier


# --- G4: R6 (cross_card_fraud_verification) tests ---

def _q9_call(order, via_device=None, via_region=None, success=True):
    if not success:
        return {
            "order": order,
            "tool": "cross_card_fraud_verification",
            "arguments": {"card_key": "C12382:21139"},
            "success": False,
            "error": "connection reset by peer",
        }
    return {
        "order": order,
        "tool": "cross_card_fraud_verification",
        "arguments": {"card_key": "C12382:21139"},
        "success": True,
        "curated_result_summary": {"via_device": via_device, "via_region": via_region},
    }


_QUALIFYING_VIA_DEVICE = {"other_cards": [{"card_key": "C99999:1", "customer_id": "C99999", "same_customer": False, "has_confirmed_fraud": True, "confirmed_fraud_case_ids": ["CC-9"]}]}
_NON_QUALIFYING_VIA_DEVICE = {"other_cards": [{"card_key": "C99999:1", "customer_id": "C99999", "same_customer": False, "has_confirmed_fraud": False, "confirmed_fraud_case_ids": []}]}


# 6. customer_report CREATE_CASE still works, unaffected by R6 existing
def test_customer_report_create_case_path_still_works_with_r6_present():
    ledger = _base_ledger(tool_calls=[
        _resolution_call(1, "customer_report"),
        _base_ledger()["tool_calls"][0],
        _q9_call(3, via_device=_QUALIFYING_VIA_DEVICE),
    ])
    result = g2.evaluate(ledger)
    entry = next(e for e in result["recommended"] if e["action"] == "CREATE_CASE")
    assert entry["route"] == "auto"
    assert "customer disputes" in entry["reason"]  # the ORIGINAL §3a reason, not R6's


# 7. R6 CREATE_CASE works when Q9 evidence qualifies (non-customer_report trigger)
def test_r6_create_case_recommended_when_q9_evidence_qualifies():
    ledger = _base_ledger(tool_calls=[
        _resolution_call(1, "risk_score"),  # confirmed NOT a customer dispute
        _q9_call(2, via_device=_QUALIFYING_VIA_DEVICE),
    ])
    result = g2.evaluate(ledger)
    entry = next(e for e in result["recommended"] if e["action"] == "CREATE_CASE")
    assert entry["route"] == "auto"
    assert entry["reason"].startswith("R6")
    assert not any(e["action"] == "CREATE_CASE" for e in result["deferred"])


# 8. R6 MONITOR_CONNECTED_CARDS works
def test_r6_monitor_connected_cards_recommended_when_q9_evidence_qualifies():
    ledger = _base_ledger(tool_calls=[_q9_call(2, via_region=_QUALIFYING_VIA_DEVICE)])
    result = g2.evaluate(ledger)
    entry = next(e for e in result["recommended"] if e["action"] == "MONITOR_CONNECTED_CARDS")
    assert entry["route"] == "auto"
    assert entry["reason"].startswith("R6")


# 9. R6 FILE_REPORT works (route L2)
def test_r6_file_report_recommended_when_q9_evidence_qualifies():
    ledger = _base_ledger(tool_calls=[_q9_call(2, via_device=_QUALIFYING_VIA_DEVICE)])
    result = g2.evaluate(ledger)
    entry = next(e for e in result["recommended"] if e["action"] == "FILE_REPORT")
    assert entry["route"] == "L2"
    assert entry["reason"].startswith("R6")


# 10. Q9 evidence without confirmed fraud remains insufficient
def test_q9_shared_element_without_confirmed_fraud_is_not_sufficient_for_r6():
    ledger = _base_ledger(tool_calls=[_q9_call(2, via_device=_NON_QUALIFYING_VIA_DEVICE)])
    result = g2.evaluate(ledger)
    for action in ("CREATE_CASE", "MONITOR_CONNECTED_CARDS", "FILE_REPORT"):
        assert any(e["action"] == action for e in result["deferred"]), f"{action} should be deferred without confirmed fraud"
        assert not any(e["action"] == action for e in result["recommended"])


def test_q9_call_present_but_failed_does_not_satisfy_r6():
    ledger = _base_ledger(tool_calls=[_q9_call(2, success=False)])
    result = g2.evaluate(ledger)
    for action in ("CREATE_CASE", "MONITOR_CONNECTED_CARDS", "FILE_REPORT"):
        assert any(e["action"] == action for e in result["deferred"])


def test_q9_absent_from_ledger_entirely_does_not_satisfy_r6():
    # The common case today: G1 never called Q9 (gate was false).
    result = g2.evaluate(_base_ledger())
    for action in ("CREATE_CASE", "MONITOR_CONNECTED_CARDS", "FILE_REPORT"):
        assert any(e["action"] == action for e in result["deferred"])


# 11. existing BLOCK_ALL_CARDS behavior remains unchanged, even with qualifying R6 evidence
def test_block_all_cards_still_prohibited_even_with_qualifying_r6_evidence():
    ledger = _base_ledger(tool_calls=[_q9_call(2, via_device=_QUALIFYING_VIA_DEVICE, via_region=_QUALIFYING_VIA_DEVICE)])
    result = g2.evaluate(ledger)
    assert any(e["action"] == "BLOCK_ALL_CARDS" for e in result["prohibited"])
    assert not any(e["action"] == "BLOCK_ALL_CARDS" for e in result["recommended"])


# --- G8 defect fix: _r6_evidence_present must read full_evidence
# (untrimmed) at real HHG-011 scale (795 device + 197 region other_cards)
# without crashing on workflow.py's _trim()-collapsed tool_calls copy. ---

def _synthetic_other_cards_g2(total: int, fraud_positive_indices: set[int]) -> list[dict]:
    return [
        {
            "card_key": f"C{i:05d}:1",
            "customer_id": f"C{i:05d}",
            "same_customer": False,
            "has_confirmed_fraud": i in fraud_positive_indices,
            "confirmed_fraud_case_ids": [f"CC-{i:04d}"] if i in fraud_positive_indices else [],
        }
        for i in range(total)
    ]


def test_r6_reads_full_evidence_at_hhg011_scale_without_crashing():
    # 795 other_cards, matching workflow.py's _MAX_LIST_SAMPLE=5 trim
    # threshold being far exceeded -- the trimmed tool_calls copy would be
    # {"count": 795, "sample": [...]}, which crashed this function before
    # the G8 fix (iterating a dict yields its keys as bare strings).
    # fraud-positive card placed near the END, past any trimmed sample:
    other_cards = _synthetic_other_cards_g2(795, fraud_positive_indices={790})
    ledger = _base_ledger(
        tool_calls=[{
            "order": 2,
            "tool": "cross_card_fraud_verification",
            "arguments": {"card_key": "C12382:21139"},
            "success": True,
            # simulates workflow.py's real _trim() output for this size:
            "curated_result_summary": {
                "via_device": {"other_cards": {"count": 795, "sample": other_cards[:5]}},
                "via_region": None,
            },
        }],
        full_evidence={
            "combined_evidence": {},
            "cross_card_fraud_verification": {"via_device": {"other_cards": other_cards}, "via_region": None},
        },
    )
    assert g2._r6_evidence_present(ledger) is True


def test_r6_full_evidence_path_finds_fraud_card_beyond_any_trimmed_sample():
    # the trimmed sample only ever has 5 entries -- if this function fell
    # back to reading the trimmed copy, it would never see index 790.
    other_cards = _synthetic_other_cards_g2(795, fraud_positive_indices={790})
    ledger = _base_ledger(
        tool_calls=[_q9_call(2, via_device={"other_cards": {"count": 795, "sample": other_cards[:5]}})],
        full_evidence={
            "combined_evidence": {},
            "cross_card_fraud_verification": {"via_device": {"other_cards": other_cards}, "via_region": None},
        },
    )
    assert g2._r6_evidence_present(ledger) is True


def test_r6_full_evidence_path_correctly_absent_when_no_fraud_at_scale():
    other_cards = _synthetic_other_cards_g2(795, fraud_positive_indices=set())  # zero fraud-positive
    ledger = _base_ledger(
        tool_calls=[_q9_call(2, via_device={"other_cards": {"count": 795, "sample": other_cards[:5]}})],
        full_evidence={
            "combined_evidence": {},
            "cross_card_fraud_verification": {"via_device": {"other_cards": other_cards}, "via_region": None},
        },
    )
    assert g2._r6_evidence_present(ledger) is False


def test_r6_falls_back_to_tool_calls_path_when_full_evidence_absent():
    # pre-G7 ledger shape (no full_evidence key at all) must still work
    # unchanged -- this is the original, small-fixture behavior.
    ledger = _base_ledger(tool_calls=[_q9_call(2, via_device=_QUALIFYING_VIA_DEVICE)])
    assert "full_evidence" not in ledger
    assert g2._r6_evidence_present(ledger) is True


def test_r6_defensive_guard_against_non_dict_entries_in_full_evidence():
    # defense-in-depth: even a still-malformed/trimmed-shaped entry inside
    # full_evidence itself (should never happen, since full_evidence is
    # never passed through _trim(), but guarded anyway) does not crash.
    ledger = _base_ledger(
        tool_calls=[_q9_call(2, via_device=_QUALIFYING_VIA_DEVICE)],
        full_evidence={
            "combined_evidence": {},
            "cross_card_fraud_verification": {"via_device": {"other_cards": ["count", "sample"]}, "via_region": None},
        },
    )
    assert g2._r6_evidence_present(ledger) is False


def test_r6_evidence_present_ignores_connected_via_case_shaped_data():
    # Sanity check that HHG_CONNECTED_TO-derived evidence (which Q9 never
    # produces) cannot accidentally satisfy _r6_evidence_present even if a
    # tool_calls entry happens to carry a similarly-shaped connected_cards
    # field under a different tool name.
    ledger = _base_ledger(tool_calls=[{
        "order": 2,
        "tool": "connected_cards",  # not cross_card_fraud_verification
        "arguments": {"card_key": "C12382:21139"},
        "success": True,
        "curated_result_summary": {"connected_via_case": [{"connected_card_key": "C99999:1", "connected_customer_id": "C99999"}], "connected_via_device": []},
    }])
    assert g2._r6_evidence_present(ledger) is False


def test_canonical_order_preserved_with_r6_actions_recommended():
    ledger = _base_ledger(tool_calls=[_q9_call(2, via_device=_QUALIFYING_VIA_DEVICE)])
    result = g2.evaluate(ledger)
    all_actions = [e["action"] for bucket in ("recommended", "prohibited", "eligible_not_recommended", "deferred") for e in result[bucket]]
    reconstructed = tuple(sorted(all_actions, key=g2.CANONICAL_ACTION_ORDER.index))
    assert reconstructed == g2.CANONICAL_ACTION_ORDER
    assert len(all_actions) == 14


# --- G5-B: R5 (card_testing) tests ---

_R5_MATCHED = {
    "matched": True,
    "pattern": "card_testing",
    "candidates": [
        {
            "authorization_txn_ids": [1, 2, 3],
            "authorization_count": 3,
            "authorization_span_minutes": 10.0,
            "larger_purchase_txn_id": 4,
            "larger_purchase_amount": 150.0,  # > $100, relevant to the BLOCK_CARD test below
        }
    ],
    "uncertainty": ["No numeric definition of 'small' is asserted -- ..."],
}
_R5_NOT_MATCHED = {"matched": False, "pattern": "card_testing", "candidates": [], "uncertainty": []}


def test_r5_matched_recommends_decline_transaction_route_l1():
    ledger = _base_ledger(pattern_evidence=_R5_MATCHED)
    result = g2.evaluate(ledger)
    entry = next(e for e in result["recommended"] if e["action"] == "DECLINE_TRANSACTION")
    assert entry["route"] == "L1"
    assert entry["reason"].startswith("R5")
    assert not any(e["action"] == "DECLINE_TRANSACTION" for e in result["deferred"])


def test_r5_matched_recommends_step_up_auth_route_auto():
    ledger = _base_ledger(pattern_evidence=_R5_MATCHED)
    result = g2.evaluate(ledger)
    entry = next(e for e in result["recommended"] if e["action"] == "STEP_UP_AUTH")
    assert entry["route"] == "auto"
    assert entry["reason"].startswith("R5")


def test_r5_absent_leaves_decline_transaction_and_step_up_auth_deferred():
    ledger = _base_ledger()  # no pattern_evidence key at all -- the common, unwired-gate case
    result = g2.evaluate(ledger)
    for action in ("DECLINE_TRANSACTION", "STEP_UP_AUTH"):
        assert any(e["action"] == action for e in result["deferred"])
        assert not any(e["action"] == action for e in result["recommended"])


def test_r5_not_matched_explicitly_leaves_both_deferred():
    ledger = _base_ledger(pattern_evidence=_R5_NOT_MATCHED)
    result = g2.evaluate(ledger)
    for action in ("DECLINE_TRANSACTION", "STEP_UP_AUTH"):
        assert any(e["action"] == action for e in result["deferred"])


def test_block_card_remains_deferred_even_with_r5_match_over_100():
    # The candidate's larger_purchase_amount ($150.0) exceeds R5's own
    # literal $100 escalation figure, but exposure_usd still doesn't
    # exist -- BLOCK_CARD must stay DEFERRED, never RECOMMENDED.
    ledger = _base_ledger(pattern_evidence=_R5_MATCHED)
    result = g2.evaluate(ledger)
    assert any(e["action"] == "BLOCK_CARD" for e in result["deferred"])
    assert not any(e["action"] == "BLOCK_CARD" for e in result["recommended"])
    reason = next(e for e in result["deferred"] if e["action"] == "BLOCK_CARD")["reason"]
    assert "exposure_usd" in reason


def test_block_all_cards_still_prohibited_with_r5_match():
    ledger = _base_ledger(pattern_evidence=_R5_MATCHED)
    result = g2.evaluate(ledger)
    assert any(e["action"] == "BLOCK_ALL_CARDS" for e in result["prohibited"])
    assert not any(e["action"] == "BLOCK_ALL_CARDS" for e in result["recommended"])


def test_canonical_order_preserved_with_r5_actions_recommended():
    ledger = _base_ledger(pattern_evidence=_R5_MATCHED)
    result = g2.evaluate(ledger)
    all_actions = [e["action"] for bucket in ("recommended", "prohibited", "eligible_not_recommended", "deferred") for e in result[bucket]]
    reconstructed = tuple(sorted(all_actions, key=g2.CANONICAL_ACTION_ORDER.index))
    assert reconstructed == g2.CANONICAL_ACTION_ORDER
    assert len(all_actions) == 14


def test_four_state_model_unchanged_with_r5_evidence():
    ledger = _base_ledger(pattern_evidence=_R5_MATCHED)
    result = g2.evaluate(ledger)
    assert set(result.keys()) == {
        "investigation_recommendation", "recommended", "prohibited",
        "eligible_not_recommended", "deferred", "uncertainties",
    }
    # no action ever appears in more than one bucket
    seen = set()
    for bucket in ("recommended", "prohibited", "eligible_not_recommended", "deferred"):
        for e in result[bucket]:
            assert e["action"] not in seen
            seen.add(e["action"])


def test_hhg_011_real_r5_detector_output_grounds_decline_and_step_up_auth():
    # Real, already-verified G5-A output for HHG-011 (card C11923:21363) --
    # 22 candidates, unranked, exactly as classify_card_testing actually
    # produces it. No candidate is singled out as "best"; matched=True
    # alone is what G2 consumes here.
    from src.investigation.patterns import classify_card_testing

    temporal = [
        {"TransactionID": 3580907, "ts": "2016-12-28 03:33:38", "TransactionAmt": 16.17, "channel": "online"},
        {"TransactionID": 3580933, "ts": "2016-12-28 03:57:10", "TransactionAmt": 23.78, "channel": "online"},
        {"TransactionID": 3580951, "ts": "2016-12-28 04:21:00", "TransactionAmt": 52.17, "channel": "online"},
        {"TransactionID": 3582082, "ts": "2016-12-28 18:54:14", "TransactionAmt": 62.11, "channel": "online"},
        {"TransactionID": 3583411, "ts": "2016-12-29 04:13:36", "TransactionAmt": 6.33, "channel": "online"},
        {"TransactionID": 3583415, "ts": "2016-12-29 04:16:10", "TransactionAmt": 6.39, "channel": "online"},
        {"TransactionID": 3583417, "ts": "2016-12-29 04:22:35", "TransactionAmt": 6.35, "channel": "online"},
        {"TransactionID": 3583670, "ts": "2016-12-29 13:23:32", "TransactionAmt": 227.32, "channel": "online"},
    ]
    real_pattern_evidence = classify_card_testing(temporal, flagged_txn_id=3583368)
    assert real_pattern_evidence["matched"] is True
    assert len(real_pattern_evidence["candidates"]) > 1  # genuinely multiple, unranked

    ledger = _base_ledger(pattern_evidence=real_pattern_evidence)
    result = g2.evaluate(ledger)
    assert any(e["action"] == "DECLINE_TRANSACTION" and e["route"] == "L1" for e in result["recommended"])
    assert any(e["action"] == "STEP_UP_AUTH" and e["route"] == "auto" for e in result["recommended"])
