"""
G11-B integration tests: the wiring of src/benchmark/submission_adapter.py
into src/benchmark/g9_runner.py's run_all_cases loop (submission_dir
parameter). Not a re-test of the adapter's own field-mapping logic
(test_submission_adapter.py) or of G9/G10's own reasoning/policy/
graph-write behavior (test_g9_runner.py/test_g9_g10_integration.py) --
those are untouched and unweakened.

Every test mocks wf.investigate() (no live TigerGraph) and injects a
fake call_llm and a fake write_call_tool (no live MCP/TigerGraph write,
no live LLM provider) -- this suite never depends on either.
"""
import asyncio
import json

from src.benchmark import g9_runner, submission_adapter as sub


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


def _small_case_pack(tmp_path, case_ids):
    path = tmp_path / "case_pack.csv"
    with open(path, "w", newline="") as f:
        f.write("case_id,opened_at,trigger_type,trigger_text,flagged_txn_id,customer_id,card_key,risk_score\n")
        for cid in case_ids:
            f.write(f"{cid},2016-01-01 00:00:00,risk_score,irrelevant,1000,C00001,C00001:1,0.5\n")
    return path


async def _successful_call_llm(prompt):
    return _VALID_LLM_RESPONSE


# 1. successful internal result creates cases/<case_id>.json
def test_successful_case_creates_submission_file(monkeypatch, tmp_path):
    _patch_investigate(monkeypatch)
    case_pack = _small_case_pack(tmp_path, ["HHG-001"])
    submission_dir = tmp_path / "cases"

    run(g9_runner.run_all_cases(
        _successful_call_llm, case_pack_path=case_pack, output_dir=tmp_path / "outputs",
        write_call_tool=_fake_write_call_tool(), submission_dir=submission_dir,
    ))

    assert (submission_dir / "HHG-001.json").exists()


# 2. generated file follows the exact adapter schema
def test_generated_submission_file_matches_adapter_schema(monkeypatch, tmp_path):
    _patch_investigate(monkeypatch)
    case_pack = _small_case_pack(tmp_path, ["HHG-001"])
    submission_dir = tmp_path / "cases"

    run(g9_runner.run_all_cases(
        _successful_call_llm, case_pack_path=case_pack, output_dir=tmp_path / "outputs",
        write_call_tool=_fake_write_call_tool(), submission_dir=submission_dir,
    ))

    written = json.loads((submission_dir / "HHG-001.json").read_text())
    assert set(written.keys()) == {
        "case_id", "case", "evidence_requests", "next_best_actions", "sar",
        "stop_reason", "tool_calls", "tokens", "latency_s",
    }
    assert written["case_id"] == "HHG-001"


# 3/4. G10 written_to_graph / graph_case_id preserved
def test_written_to_graph_and_graph_case_id_preserved_from_g10(monkeypatch, tmp_path):
    _patch_investigate(monkeypatch)
    case_pack = _small_case_pack(tmp_path, ["HHG-002"])
    submission_dir = tmp_path / "cases"

    run(g9_runner.run_all_cases(
        _successful_call_llm, case_pack_path=case_pack, output_dir=tmp_path / "outputs",
        write_call_tool=_fake_write_call_tool(), submission_dir=submission_dir,
    ))

    written = json.loads((submission_dir / "HHG-002.json").read_text())
    assert written["case"]["written_to_graph"] is True
    assert written["case"]["graph_case_id"] == "HHG-002"


# 5. investigation failure still produces an honest submission file
def test_investigation_failure_still_produces_honest_submission_file(monkeypatch, tmp_path):
    _patch_investigate(monkeypatch, raise_for={"HHG-005"})
    case_pack = _small_case_pack(tmp_path, ["HHG-005"])
    submission_dir = tmp_path / "cases"

    run(g9_runner.run_all_cases(
        _successful_call_llm, case_pack_path=case_pack, output_dir=tmp_path / "outputs",
        write_call_tool=_fake_write_call_tool(), submission_dir=submission_dir,
    ))

    written = json.loads((submission_dir / "HHG-005.json").read_text())
    assert written["case"] == sub._EMPTY_CASE
    assert written["case_id"] == "HHG-005"


# 6. graph-write failure still produces an honest submission file
def test_graph_write_failure_still_produces_honest_submission_file(monkeypatch, tmp_path):
    _patch_investigate(monkeypatch)
    case_pack = _small_case_pack(tmp_path, ["HHG-006"])
    submission_dir = tmp_path / "cases"

    run(g9_runner.run_all_cases(
        _successful_call_llm, case_pack_path=case_pack, output_dir=tmp_path / "outputs",
        write_call_tool=_fake_write_call_tool(error="simulated graph write failure"), submission_dir=submission_dir,
    ))

    written = json.loads((submission_dir / "HHG-006.json").read_text())
    # investigation itself succeeded, so the case object is real, but the
    # graph write failed -- written_to_graph/graph_case_id must reflect
    # that failure honestly, never claim a successful write.
    assert written["case"]["verdict"] == "uncertain"
    assert written["case"]["written_to_graph"] is False
    assert written["case"]["graph_case_id"] == ""


# 7. existing outputs/case_results output remains unchanged
def test_case_results_output_identical_with_or_without_submission_dir(monkeypatch, tmp_path):
    _patch_investigate(monkeypatch)
    case_pack = _small_case_pack(tmp_path, ["HHG-007"])

    run(g9_runner.run_all_cases(
        _successful_call_llm, case_pack_path=case_pack, output_dir=tmp_path / "without",
        write_call_tool=_fake_write_call_tool(),
    ))
    run(g9_runner.run_all_cases(
        _successful_call_llm, case_pack_path=case_pack, output_dir=tmp_path / "with",
        write_call_tool=_fake_write_call_tool(), submission_dir=tmp_path / "cases",
    ))

    without = json.loads((tmp_path / "without" / "case_results" / "HHG-007.json").read_text())
    with_ = json.loads((tmp_path / "with" / "case_results" / "HHG-007.json").read_text())
    without.pop("timing_s", None)  # real wall-clock timing, expected to vary run to run
    with_.pop("timing_s", None)
    assert without == with_


# 7b. submission_dir=None (the default) never touches disk under cases/
def test_submission_dir_none_by_default_writes_no_submission_file(monkeypatch, tmp_path):
    _patch_investigate(monkeypatch)
    case_pack = _small_case_pack(tmp_path, ["HHG-008"])

    run(g9_runner.run_all_cases(
        _successful_call_llm, case_pack_path=case_pack, output_dir=tmp_path / "outputs",
        write_call_tool=_fake_write_call_tool(),
    ))

    assert not (tmp_path / "cases").exists()


# 8. cases/ directory is created when absent
def test_submission_dir_created_automatically_when_absent(monkeypatch, tmp_path):
    _patch_investigate(monkeypatch)
    case_pack = _small_case_pack(tmp_path, ["HHG-009"])
    submission_dir = tmp_path / "brand_new_cases_dir"
    assert not submission_dir.exists()

    run(g9_runner.run_all_cases(
        _successful_call_llm, case_pack_path=case_pack, output_dir=tmp_path / "outputs",
        write_call_tool=_fake_write_call_tool(), submission_dir=submission_dir,
    ))

    assert submission_dir.exists()
    assert (submission_dir / "HHG-009.json").exists()


# 9. adapter is called exactly once per benchmark case
def test_adapter_called_exactly_once_per_case(monkeypatch, tmp_path):
    _patch_investigate(monkeypatch)
    case_ids = ["HHG-001", "HHG-002", "HHG-003"]
    case_pack = _small_case_pack(tmp_path, case_ids)
    submission_dir = tmp_path / "cases"

    calls = []
    real_write = sub.write_submission_file

    def counting_write(result, output_dir=sub.DEFAULT_SUBMISSION_DIR):
        calls.append(result["case_id"])
        return real_write(result, output_dir=output_dir)

    monkeypatch.setattr(g9_runner.sub, "write_submission_file", counting_write)

    run(g9_runner.run_all_cases(
        _successful_call_llm, case_pack_path=case_pack, output_dir=tmp_path / "outputs",
        write_call_tool=_fake_write_call_tool(), submission_dir=submission_dir,
    ))

    assert calls == case_ids  # exactly one call per case, in order, no duplicates


# 10. no Gemini call is introduced
def test_no_extra_llm_call_introduced_by_submission_wiring(monkeypatch, tmp_path):
    _patch_investigate(monkeypatch)
    case_ids = ["HHG-001", "HHG-002"]
    case_pack = _small_case_pack(tmp_path, case_ids)

    llm_calls = []

    async def counting_call_llm(prompt):
        llm_calls.append(prompt)
        return _VALID_LLM_RESPONSE

    run(g9_runner.run_all_cases(
        counting_call_llm, case_pack_path=case_pack, output_dir=tmp_path / "outputs",
        write_call_tool=_fake_write_call_tool(), submission_dir=tmp_path / "cases",
    ))

    assert len(llm_calls) == len(case_ids)  # exactly one per case -- unchanged by the submission wiring


# 11. no TigerGraph call is introduced
def test_no_extra_graph_write_call_introduced_by_submission_wiring(monkeypatch, tmp_path):
    _patch_investigate(monkeypatch)
    case_ids = ["HHG-001", "HHG-002"]
    case_pack = _small_case_pack(tmp_path, case_ids)
    write_call_tool = _fake_write_call_tool()

    run(g9_runner.run_all_cases(
        _successful_call_llm, case_pack_path=case_pack, output_dir=tmp_path / "outputs",
        write_call_tool=write_call_tool, submission_dir=tmp_path / "cases",
    ))

    assert len(write_call_tool.calls) == len(case_ids)  # exactly one graph write per case -- unchanged


# 12. deterministic repeated conversion produces identical JSON content
def test_submission_file_content_deterministic_across_runs(monkeypatch, tmp_path):
    _patch_investigate(monkeypatch)
    case_pack = _small_case_pack(tmp_path, ["HHG-010"])

    run(g9_runner.run_all_cases(
        _successful_call_llm, case_pack_path=case_pack, output_dir=tmp_path / "outputs1",
        write_call_tool=_fake_write_call_tool(response={"written": True, "graph_case_id": "HHG-010", "steps": {}}),
        submission_dir=tmp_path / "cases1",
    ))
    run(g9_runner.run_all_cases(
        _successful_call_llm, case_pack_path=case_pack, output_dir=tmp_path / "outputs2",
        write_call_tool=_fake_write_call_tool(response={"written": True, "graph_case_id": "HHG-010", "steps": {}}),
        submission_dir=tmp_path / "cases2",
    ))

    content_1 = json.loads((tmp_path / "cases1" / "HHG-010.json").read_text())
    content_2 = json.loads((tmp_path / "cases2" / "HHG-010.json").read_text())
    content_1.pop("latency_s", None)  # real wall-clock timing, expected to vary run to run
    content_2.pop("latency_s", None)
    assert content_1 == content_2
