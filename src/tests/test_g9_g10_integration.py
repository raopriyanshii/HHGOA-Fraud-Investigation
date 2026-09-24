"""
G10 integration tests: the G9 <-> G10 wiring in src/benchmark/g9_runner.py
specifically (run_single_case/run_all_cases calling
src.agent.case_writer.write_case_memory). Not a re-test of G9's own
reasoning/validation/policy behavior (test_g9_runner.py) or G10's own
write-mutation behavior (test_case_memory.py/test_case_writer.py) --
those are untouched and unweakened.

Every test mocks wf.investigate() (no live TigerGraph) and injects a
fake call_llm and a fake write_call_tool (no live MCP/TigerGraph write)
-- this suite never depends on either.
"""
import asyncio
import json

from src.benchmark import g9_runner


def run(coro):
    return asyncio.run(coro)


def _fake_ledger(case_id: str, flagged_txn_id: int = 3530164) -> dict:
    return {
        "investigation_id": f"inv-{case_id}",
        "trigger": {"type": "case_id", "value": case_id},
        "tool_calls": [{"order": 1, "tool": "benchmark_case_resolution", "arguments": {"case_id": case_id}, "success": True, "curated_result_summary": {"flagged_txn_id": flagged_txn_id}}],
        "uncertainties": [],
        "recommendation": "insufficient_evidence",
        "recommendation_basis": {"reasons": [], "tool_call_orders": []},
        "pattern_evidence": {"matched": False, "pattern": "card_testing", "candidates": [], "uncertainty": []},
        "full_evidence": {
            "combined_evidence": {
                "transaction": {"TransactionID": flagged_txn_id, "ts": "2016-01-01 10:00:00", "TransactionAmt": 50.0},
                "card": {"card_key": "C00001:1"},
                "customer": {"customer_id": "C00001"},
                "temporal_activity": [{"TransactionID": flagged_txn_id, "ts": "2016-01-01 10:00:00", "TransactionAmt": 50.0, "channel": "online"}],
                "device_evidence": None,
                "billing_region": {"hub_flag": True},
                "historical_cases": {"direct_cases": [], "connected_cases": []},
                "connected_cards": {"connected_via_case": [], "connected_via_device": []},
            },
            "cross_card_fraud_verification": None,
        },
    }


_VALID_LLM_RESPONSE = json.dumps({
    "verdict": "uncertain", "fraud_probability": 0.45, "affected_txn_ids": ["3530164"],
    "first_suspicious_txn_id": "3530164", "pattern": "none", "pattern_description": "",
    "summary": "a real-shaped summary", "reasoning": "some reasoning", "uncertainties": [],
})


def _patch_investigate(monkeypatch, raise_for=None):
    raise_for = raise_for or set()

    async def fake_investigate(trigger, *, window_hours=24.0, call_tool=None):
        case_id = trigger["value"]
        if case_id in raise_for:
            raise RuntimeError(f"simulated G1 failure for {case_id}")
        return _fake_ledger(case_id)

    monkeypatch.setattr(g9_runner.wf, "investigate", fake_investigate)


def _fake_write_call_tool(response=None, error=None):
    calls = []

    async def call_tool(name, arguments):
        calls.append((name, arguments))
        if error is not None:
            return False, None, error
        return True, response or {"written": True, "graph_case_id": arguments["case_id"], "steps": {"vertex": {"ok": True}}}, None

    call_tool.calls = calls
    return call_tool


async def _successful_call_llm(prompt):
    return _VALID_LLM_RESPONSE


# --- 1/3/4/14. successful G9 case -> writer called exactly once, receives real fields, graph_case_id=case_id ---

def test_successful_case_invokes_writer_exactly_once_with_correct_case_id(monkeypatch):
    _patch_investigate(monkeypatch)
    write_call_tool = _fake_write_call_tool()
    result = run(g9_runner.run_single_case("HHG-003", _successful_call_llm, write_call_tool=write_call_tool))
    assert result["success"] is True
    assert len(write_call_tool.calls) == 1
    name, payload = write_call_tool.calls[0]
    assert name == "write_case_memory"
    assert payload["case_id"] == "HHG-003"


def test_writer_receives_the_actual_assembled_g9_verdict_and_probability(monkeypatch):
    _patch_investigate(monkeypatch)
    write_call_tool = _fake_write_call_tool()
    run(g9_runner.run_single_case("HHG-003", _successful_call_llm, write_call_tool=write_call_tool))
    _, payload = write_call_tool.calls[0]
    # these are the REAL values the fake LLM response produced, not invented by the integration:
    assert payload["verdict"] == "uncertain"
    assert payload["fraud_probability"] == 0.45
    assert payload["affected_txn_ids"] == [3530164]
    assert payload["summary"] == "a real-shaped summary"


# --- 2/3. successful writer -> written_to_graph=true, graph_case_id=case_id ---

def test_successful_write_sets_written_to_graph_true_and_correct_graph_case_id(monkeypatch):
    _patch_investigate(monkeypatch)
    write_call_tool = _fake_write_call_tool()
    result = run(g9_runner.run_single_case("HHG-003", _successful_call_llm, write_call_tool=write_call_tool))
    assert result["written_to_graph"] is True
    assert result["graph_case_id"] == "HHG-003"
    # organizer-format compliance: mirrored into the case object itself:
    assert result["outcome"]["case"]["written_to_graph"] is True
    assert result["outcome"]["case"]["graph_case_id"] == "HHG-003"


# --- 5/6. G9/validator failure -> writer called zero times ---

def test_g1_pipeline_failure_never_invokes_writer(monkeypatch):
    _patch_investigate(monkeypatch, raise_for={"HHG-003"})
    write_call_tool = _fake_write_call_tool()
    try:
        run(g9_runner.run_single_case("HHG-003", _successful_call_llm, write_call_tool=write_call_tool))
    except RuntimeError:
        pass
    assert write_call_tool.calls == []


def test_validator_failure_never_invokes_writer(monkeypatch):
    _patch_investigate(monkeypatch)

    async def bad_call_llm(prompt):
        return json.dumps({"verdict": "not_a_real_verdict", "fraud_probability": 0.5, "affected_txn_ids": [], "first_suspicious_txn_id": None, "pattern": "none", "pattern_description": "", "summary": "", "reasoning": "", "uncertainties": []})

    write_call_tool = _fake_write_call_tool()
    result = run(g9_runner.run_single_case("HHG-003", bad_call_llm, write_call_tool=write_call_tool))
    assert result["success"] is False
    assert write_call_tool.calls == []
    assert result["written_to_graph"] is False
    assert result["graph_case_id"] == ""


# --- 7/8. graph-write failure -> written_to_graph=false, graph_case_id="" ---

def test_graph_write_failure_reports_false_and_empty_id(monkeypatch):
    _patch_investigate(monkeypatch)
    write_call_tool = _fake_write_call_tool(error="503 UNAVAILABLE: TigerGraph temporarily unreachable")
    result = run(g9_runner.run_single_case("HHG-003", _successful_call_llm, write_call_tool=write_call_tool))
    assert result["success"] is True  # the INVESTIGATION still succeeded
    assert result["written_to_graph"] is False
    assert result["graph_case_id"] == ""
    assert result["outcome"]["case"]["written_to_graph"] is False
    assert result["outcome"]["case"]["graph_case_id"] == ""


def test_graph_write_partial_success_is_reported_as_failure(monkeypatch):
    _patch_investigate(monkeypatch)
    write_call_tool = _fake_write_call_tool(response={"written": False, "graph_case_id": "", "steps": {"vertex": {"ok": True}, "involves_edges": {"ok": False}}})
    result = run(g9_runner.run_single_case("HHG-003", _successful_call_llm, write_call_tool=write_call_tool))
    assert result["written_to_graph"] is False
    assert result["graph_case_id"] == ""


# --- 9/10. graph-write failure does not cause an LLM retry or investigation retry ---

def test_graph_write_failure_does_not_retry_the_llm_call(monkeypatch):
    _patch_investigate(monkeypatch)
    llm_calls = []

    async def counting_call_llm(prompt):
        llm_calls.append(prompt)
        return _VALID_LLM_RESPONSE

    write_call_tool = _fake_write_call_tool(error="boom")
    run(g9_runner.run_single_case("HHG-003", counting_call_llm, write_call_tool=write_call_tool))
    assert len(llm_calls) == 1  # not retried because the graph write failed


def test_graph_write_failure_does_not_rerun_investigation(monkeypatch):
    investigate_calls = []

    async def counting_investigate(trigger, *, window_hours=24.0, call_tool=None):
        investigate_calls.append(trigger)
        return _fake_ledger(trigger["value"])

    monkeypatch.setattr(g9_runner.wf, "investigate", counting_investigate)
    write_call_tool = _fake_write_call_tool(error="boom")
    run(g9_runner.run_single_case("HHG-003", _successful_call_llm, write_call_tool=write_call_tool))
    assert len(investigate_calls) == 1


# --- 11. graph-write failure does not stop remaining benchmark cases ---

def test_graph_write_failure_for_one_case_does_not_stop_the_run(monkeypatch, tmp_path):
    case_ids = ["HHG-001", "HHG-002", "HHG-003"]
    case_pack = tmp_path / "case_pack.csv"
    with open(case_pack, "w", newline="") as f:
        f.write("case_id,opened_at,trigger_type,trigger_text,flagged_txn_id,customer_id,card_key,risk_score\n")
        for cid in case_ids:
            f.write(f"{cid},2016-01-01 00:00:00,risk_score,irrelevant,3530164,C00001,C00001:1,0.5\n")

    _patch_investigate(monkeypatch)

    write_calls = []

    async def sometimes_failing_write_call_tool(name, arguments):
        write_calls.append(arguments["case_id"])
        if arguments["case_id"] == "HHG-002":
            return False, None, "simulated graph write failure"
        return True, {"written": True, "graph_case_id": arguments["case_id"], "steps": {}}, None

    outcome = run(g9_runner.run_all_cases(
        _successful_call_llm, case_pack_path=case_pack, output_dir=tmp_path,
        write_call_tool=sometimes_failing_write_call_tool,
    ))
    assert outcome["summary"]["total_cases"] == 3
    assert len(write_calls) == 3  # every case still ran, including after HHG-002's failure
    by_case = {r["case_id"]: r for r in outcome["results"]}
    assert by_case["HHG-001"]["written_to_graph"] is True
    assert by_case["HHG-002"]["written_to_graph"] is False
    assert by_case["HHG-002"]["success"] is True  # investigation itself still succeeded
    assert by_case["HHG-003"]["written_to_graph"] is True


# --- 12/13. existing G9 fields remain unchanged after success/failure ---

def test_existing_g9_fields_unchanged_after_successful_graph_write(monkeypatch):
    _patch_investigate(monkeypatch)
    write_call_tool = _fake_write_call_tool()
    result = run(g9_runner.run_single_case("HHG-003", _successful_call_llm, write_call_tool=write_call_tool))
    case = result["outcome"]["case"]
    # all pre-existing case_output fields are exactly what case_output.py produces, untouched:
    assert case["verdict"] == "uncertain"
    assert case["fraud_probability"] == 0.45
    assert case["pattern"] == "none"
    assert case["affected_txn_ids"] == ["3530164"]
    assert case["summary"] == "a real-shaped summary"
    assert "next_best_actions" in result["outcome"]
    assert "stop_reason" in result["outcome"]


def test_existing_g9_fields_unchanged_after_failed_graph_write(monkeypatch):
    _patch_investigate(monkeypatch)
    write_call_tool = _fake_write_call_tool(error="boom")
    result = run(g9_runner.run_single_case("HHG-003", _successful_call_llm, write_call_tool=write_call_tool))
    case = result["outcome"]["case"]
    assert case["verdict"] == "uncertain"
    assert case["fraud_probability"] == 0.45
    assert case["summary"] == "a real-shaped summary"


# --- 15/16/17. no direct TigerGraph write, no generic mutation API, no second writer ---

def test_g9_runner_never_imports_get_connection_or_a_second_writer():
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(g9_runner))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported.update(f"{node.module}.{alias.name}" for alias in node.names)
    assert not any(name.endswith("get_connection") for name in imported)
    assert "src.agent.case_writer" in imported  # the ONE existing writer, reused
    # no second/competing writer module was introduced:
    assert not any("writer" in name and name != "src.agent.case_writer" for name in imported)


def test_real_write_dispatcher_is_scoped_to_exactly_one_tool_name():
    import asyncio as _asyncio
    with __import__("pytest").raises(ValueError):
        _asyncio.run(g9_runner._real_write_call_tool("some_other_tool", {}))


def test_only_one_write_case_memory_tool_name_is_ever_dispatched(monkeypatch):
    _patch_investigate(monkeypatch)
    write_call_tool = _fake_write_call_tool()
    run(g9_runner.run_single_case("HHG-003", _successful_call_llm, write_call_tool=write_call_tool))
    tool_names = {name for name, _ in write_call_tool.calls}
    assert tool_names == {"write_case_memory"}


# --- 18. no Gemini/Groq call introduced by the G10 integration itself ---

def test_g10_integration_adds_zero_llm_calls_beyond_the_existing_one(monkeypatch):
    _patch_investigate(monkeypatch)
    llm_calls = []

    async def counting_call_llm(prompt):
        llm_calls.append(prompt)
        return _VALID_LLM_RESPONSE

    write_call_tool = _fake_write_call_tool()
    run(g9_runner.run_single_case("HHG-003", counting_call_llm, write_call_tool=write_call_tool))
    assert len(llm_calls) == 1  # exactly the one G9 already made -- G10 added none


def test_case_writer_module_has_no_llm_provider_import():
    import ast
    import inspect

    from src.agent import case_writer

    tree = ast.parse(inspect.getsource(case_writer))
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert not any("llm_providers" in name for name in imported)


# --- runner_exception path also carries the G10 fields consistently ---

def test_runner_exception_path_reports_written_to_graph_false(monkeypatch, tmp_path):
    case_pack = tmp_path / "case_pack.csv"
    with open(case_pack, "w", newline="") as f:
        f.write("case_id,opened_at,trigger_type,trigger_text,flagged_txn_id,customer_id,card_key,risk_score\n")
        f.write("HHG-999,2016-01-01 00:00:00,risk_score,irrelevant,1,C1,C1:1,0.5\n")

    _patch_investigate(monkeypatch, raise_for={"HHG-999"})
    write_call_tool = _fake_write_call_tool()
    outcome = run(g9_runner.run_all_cases(_successful_call_llm, case_pack_path=case_pack, output_dir=tmp_path, write_call_tool=write_call_tool))
    result = outcome["results"][0]
    assert result["stage"] == "runner_exception"
    assert result["written_to_graph"] is False
    assert result["graph_case_id"] == ""
    assert write_call_tool.calls == []


# --- G14 end-to-end integration through run_single_case (fake data only, no live calls) ---

def _fake_ledger_with_connected_via_device(case_id: str, target_card_key: str, flagged_txn_id: int = 3530164) -> dict:
    ledger = _fake_ledger(case_id, flagged_txn_id)
    ledger["full_evidence"]["combined_evidence"]["connected_cards"] = {
        "connected_via_case": [],
        "connected_via_device": [{"card_key": target_card_key, "customer_id": target_card_key.split(":")[0]}],
    }
    return ledger


def _evidence_request_llm_response(target_card_key: str) -> str:
    return json.dumps({
        "verdict": "uncertain", "fraud_probability": 0.4, "affected_txn_ids": [],
        "first_suspicious_txn_id": None, "pattern": "none", "pattern_description": "",
        "summary": "needs more evidence", "reasoning": "checking sibling card", "uncertainties": [],
        "needs_more_evidence": True,
        "evidence_request": {"type": "historical_case_evidence", "target_card_key": target_card_key, "reason": "check sibling history"},
    })


_G14_FINAL_LLM_RESPONSE = json.dumps({
    "verdict": "fraud", "fraud_probability": 0.9, "affected_txn_ids": ["3530164"],
    "first_suspicious_txn_id": "3530164", "pattern": "none", "pattern_description": "",
    "summary": "confirmed after sibling check", "reasoning": "sibling had confirmed fraud", "uncertainties": [],
})


def _sequenced_call_llm(*responses):
    calls = []

    async def call_llm(prompt):
        calls.append(prompt)
        return responses[len(calls) - 1]

    call_llm.calls = calls
    return call_llm


def _fake_evidence_request_call_tool(response=None, ok=True, error=None):
    calls = []

    async def call_tool(name, arguments):
        calls.append((name, arguments))
        if not ok:
            return False, None, error or "simulated Q4 tool failure"
        return True, response if response is not None else {"direct_cases": [{"case_id": "CC-1", "outcome": "confirmed_fraud"}], "connected_cases": []}, None

    call_tool.calls = calls
    return call_tool


# 1. evidence requested successfully -> Q4 runs once, round 2 runs once, final result from round 2

def test_g14_evidence_requested_successfully_full_g9_integration(monkeypatch):
    target_card_key = "C99999:1"

    async def fake_investigate_with_device(trigger, *, window_hours=24.0, call_tool=None):
        return _fake_ledger_with_connected_via_device(trigger["value"], target_card_key)

    monkeypatch.setattr(g9_runner.wf, "investigate", fake_investigate_with_device)

    call_llm = _sequenced_call_llm(_evidence_request_llm_response(target_card_key), _G14_FINAL_LLM_RESPONSE)
    write_call_tool = _fake_write_call_tool()
    evidence_call_tool = _fake_evidence_request_call_tool()

    result = run(g9_runner.run_single_case(
        "HHG-003", call_llm, write_call_tool=write_call_tool, evidence_request_call_tool=evidence_call_tool,
    ))

    assert result["success"] is True
    assert len(call_llm.calls) == 2  # LLM calls = 2 only because G14 succeeded
    assert len(evidence_call_tool.calls) == 1  # Q4 called exactly once
    assert evidence_call_tool.calls[0] == ("historical_case_evidence", {"card_key": target_card_key})
    # both raw responses preserved, in chronological order:
    assert result["llm"]["raw_responses"] == [_evidence_request_llm_response(target_card_key), _G14_FINAL_LLM_RESPONSE]
    assert result["outcome"]["case"]["verdict"] == "fraud"  # round 2's result used, not round 1's

    info = result["outcome"]["evidence_request_outcome"]
    assert info["accepted"] is True
    assert info["round2_attempted"] is True
    assert info["round2_ok"] is True

    from src.benchmark import submission_adapter as sa
    submission = sa.build_submission_case(result)
    assert set(submission.keys()) == {
        "case_id", "case", "evidence_requests", "next_best_actions", "sar",
        "stop_reason", "tool_calls", "tokens", "latency_s",
    }
    # tool_calls counted correctly: 1 fake G1 call + 1 graph write + 1 accepted evidence-request call
    assert submission["tool_calls"] == 3


# 2. invalid card requested -> Q4 NOT called, round 2 NOT called, round 1's result used

def test_g14_invalid_card_requested_no_tool_call_no_round2(monkeypatch):
    _patch_investigate(monkeypatch)  # default fake ledger's connected_via_device is []
    call_llm = _sequenced_call_llm(_evidence_request_llm_response("C_NOT_REAL:1"))
    write_call_tool = _fake_write_call_tool()
    evidence_call_tool = _fake_evidence_request_call_tool()

    result = run(g9_runner.run_single_case(
        "HHG-003", call_llm, write_call_tool=write_call_tool, evidence_request_call_tool=evidence_call_tool,
    ))

    assert result["success"] is True
    assert len(call_llm.calls) == 1  # no second LLM call
    assert evidence_call_tool.calls == []  # Q4 never called
    assert result["llm"]["raw_responses"] == [_evidence_request_llm_response("C_NOT_REAL:1")]
    assert result["outcome"]["case"]["verdict"] == "uncertain"  # round 1's own result, unchanged

    info = result["outcome"]["evidence_request_outcome"]
    assert info["accepted"] is False
    assert info["rejection_reason"] == "target_card_key_not_in_connected_via_device"

    from src.benchmark import submission_adapter as sa
    submission = sa.build_submission_case(result)
    assert set(submission.keys()) == {
        "case_id", "case", "evidence_requests", "next_best_actions", "sar",
        "stop_reason", "tool_calls", "tokens", "latency_s",
    }
    assert submission["tool_calls"] == 2  # 1 G1 call + 1 graph write -- NOT +1 for the rejected request


# 3. historical evidence tool fails -> round 2 NOT called, falls back to round 1

def test_g14_historical_tool_failure_falls_back_to_round1(monkeypatch):
    target_card_key = "C99999:1"

    async def fake_investigate_with_device(trigger, *, window_hours=24.0, call_tool=None):
        return _fake_ledger_with_connected_via_device(trigger["value"], target_card_key)

    monkeypatch.setattr(g9_runner.wf, "investigate", fake_investigate_with_device)

    call_llm = _sequenced_call_llm(_evidence_request_llm_response(target_card_key))
    write_call_tool = _fake_write_call_tool()
    evidence_call_tool = _fake_evidence_request_call_tool(ok=False, error="simulated Q4 tool failure")

    result = run(g9_runner.run_single_case(
        "HHG-003", call_llm, write_call_tool=write_call_tool, evidence_request_call_tool=evidence_call_tool,
    ))

    assert result["success"] is True
    assert len(call_llm.calls) == 1  # round 2 never happened
    assert len(evidence_call_tool.calls) == 1  # Q4 WAS attempted -- that's how the failure is known
    assert result["llm"]["raw_responses"] == [_evidence_request_llm_response(target_card_key)]
    assert result["outcome"]["case"]["verdict"] == "uncertain"  # safe fallback to round 1's result

    info = result["outcome"]["evidence_request_outcome"]
    assert info["accepted"] is True  # the call was genuinely attempted
    assert info["round2_attempted"] is False  # never reached round 2 since the tool itself failed
    assert "tool_error" in info

    from src.benchmark import submission_adapter as sa
    submission = sa.build_submission_case(result)
    assert set(submission.keys()) == {
        "case_id", "case", "evidence_requests", "next_best_actions", "sar",
        "stop_reason", "tool_calls", "tokens", "latency_s",
    }
    # an attempted-but-failed tool call is still counted (matches the
    # existing convention: G1's own tool_calls list counts attempted
    # calls regardless of success/failure too):
    assert submission["tool_calls"] == 3
