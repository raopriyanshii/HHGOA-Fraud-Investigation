"""
G10 unit tests for src/agent/case_writer.py (the deterministic
orchestration layer). Every test uses a fake call_tool -- zero live MCP
or TigerGraph dependency.
"""
import asyncio

import pytest

from src.agent import case_writer


def run(coro):
    return asyncio.run(coro)


def _successful_outcome(**case_overrides):
    case = {
        "status": "closed_fraud",
        "verdict": "fraud",
        "fraud_probability": 0.8,
        "pattern": "card_testing",
        "pattern_description": "",
        "affected_txn_ids": ["1001", "1002"],
        "first_suspicious_txn_id": "1001",
        "exposure_usd": 55.0,
        "evidence": [{"claim": "x", "source": "graph", "ref": "y", "entity_ids": []}],
        "similar_prior_cases": ["CC-1"],
        "summary": "a summary",
    }
    case.update(case_overrides)
    return {
        "ok": True,
        "case": case,
        "next_best_actions": {"initial": [], "final": [], "what_changed": "nothing"},
        "stripped_ids": [],
        "stop_reason": "a stop reason",
    }


def _failed_outcome(failure_reason="invalid_verdict"):
    return {
        "ok": False,
        "failure_reason": failure_reason,
        "detail": "simulated failure",
        "stop_reason": f"LLM output failed validation ({failure_reason})",
    }


def _make_call_tool(response=None, ok=True, error=None):
    calls = []

    async def call_tool(name, payload):
        calls.append((name, payload))
        return ok, response, error

    call_tool.calls = calls
    return call_tool


# --- 1. successful complete write (through the writer layer) ---

def test_successful_write_through_writer(monkeypatch):
    call_tool = _make_call_tool(response={"written": True, "graph_case_id": "HHG-001", "steps": {}})
    result = run(case_writer.write_case_memory("HHG-001", _successful_outcome(), call_tool=call_tool))
    assert result["attempted"] is True
    assert result["written_to_graph"] is True
    assert result["graph_case_id"] == "HHG-001"
    assert len(call_tool.calls) == 1
    assert call_tool.calls[0][0] == "write_case_memory"


# --- 7/8. validation/policy failure blocks the writer entirely ---

def test_reasoning_validation_failure_blocks_writer():
    call_tool = _make_call_tool()
    result = run(case_writer.write_case_memory("HHG-001", _failed_outcome("invalid_verdict"), call_tool=call_tool))
    assert result["attempted"] is False
    assert result["written_to_graph"] is False
    assert result["graph_case_id"] == ""
    assert result["reason"] == "investigation_not_successful"
    assert call_tool.calls == []  # never even reaches the MCP layer


def test_llm_call_failure_blocks_writer():
    call_tool = _make_call_tool()
    result = run(case_writer.write_case_memory("HHG-001", _failed_outcome("llm_timeout"), call_tool=call_tool))
    assert result["attempted"] is False
    assert call_tool.calls == []


# --- 9. malformed payload rejected before graph mutation ---

def test_malformed_verdict_in_case_output_is_rejected_before_mcp_call():
    outcome = _successful_outcome(verdict="definitely_fraud")  # not a valid enum value
    call_tool = _make_call_tool()
    result = run(case_writer.write_case_memory("HHG-001", outcome, call_tool=call_tool))
    assert result["attempted"] is False
    assert result["reason"] == "malformed_payload"
    assert call_tool.calls == []


def test_malformed_probability_is_rejected_before_mcp_call():
    outcome = _successful_outcome(fraud_probability=1.5)
    call_tool = _make_call_tool()
    result = run(case_writer.write_case_memory("HHG-001", outcome, call_tool=call_tool))
    assert result["attempted"] is False
    assert result["reason"] == "malformed_payload"
    assert call_tool.calls == []


# --- 10. MCP write failure ---

def test_mcp_write_failure_reported_honestly():
    call_tool = _make_call_tool(ok=False, error="connection reset by peer")
    result = run(case_writer.write_case_memory("HHG-001", _successful_outcome(), call_tool=call_tool))
    assert result["attempted"] is True
    assert result["written_to_graph"] is False
    assert result["graph_case_id"] == ""
    assert result["reason"] == "mcp_write_failed"


def test_mcp_partial_write_result_is_reported_as_unsuccessful():
    call_tool = _make_call_tool(response={"written": False, "graph_case_id": "", "steps": {"vertex": {"ok": True}, "involves_edges": {"ok": False}}})
    result = run(case_writer.write_case_memory("HHG-001", _successful_outcome(), call_tool=call_tool))
    assert result["written_to_graph"] is False
    assert result["graph_case_id"] == ""
    assert result["reason"] == "partial_or_failed_graph_write"


# --- 14. secret scrubbing at the writer layer ---

def test_mcp_failure_error_is_scrubbed(monkeypatch):
    fake_key = "AIzaSyDaGmWKa4JsXZ-HjGw7ISLn_3namBGewQe"
    monkeypatch.setenv("GEMINI_API_KEY", fake_key)
    call_tool = _make_call_tool(ok=False, error=f"401 key {fake_key} rejected")
    result = run(case_writer.write_case_memory("HHG-001", _successful_outcome(), call_tool=call_tool))
    assert fake_key not in result["detail"]
    assert "[REDACTED]" in result["detail"]


# --- 16. LLM layer has no graph-write dependency (structural checks) ---

def test_reasoning_module_never_imports_case_writer_or_case_memory():
    import ast
    import inspect

    from src.agent import reasoning

    tree = ast.parse(inspect.getsource(reasoning))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "src.agent.case_writer" not in imported
    assert "src.investigation.case_memory" not in imported


def test_reasoning_validator_never_imports_case_writer_or_case_memory():
    import ast
    import inspect

    from src.agent import reasoning_validator

    tree = ast.parse(inspect.getsource(reasoning_validator))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "src.agent.case_writer" not in imported
    assert "src.investigation.case_memory" not in imported


def test_case_writer_never_imports_llm_provider_modules():
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(case_writer))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any("llm_providers" in name for name in imported)


# --- 17. graph_case_id empty on every unsuccessful path (writer level) ---

def test_graph_case_id_empty_on_every_failure_path_at_writer_level():
    scenarios = [
        _failed_outcome(),
        _successful_outcome(verdict="bad"),
    ]
    for outcome in scenarios:
        call_tool = _make_call_tool()
        result = run(case_writer.write_case_memory("HHG-001", outcome, call_tool=call_tool))
        assert result["graph_case_id"] == ""

    call_tool = _make_call_tool(ok=False, error="boom")
    result = run(case_writer.write_case_memory("HHG-001", _successful_outcome(), call_tool=call_tool))
    assert result["graph_case_id"] == ""


# --- payload construction correctness ---

def test_build_case_memory_payload_reads_only_already_present_fields():
    outcome = _successful_outcome()
    payload = case_writer.build_case_memory_payload("HHG-001", outcome)
    assert payload["case_id"] == "HHG-001"
    assert payload["affected_txn_ids"] == [1001, 1002]  # coerced from strings, matching validator's own int contract
    assert payload["first_suspicious_txn_id"] == 1001
    assert payload["similar_prior_cases"] == ["CC-1"]
    assert payload["stop_reason"] == "a stop reason"


def test_build_case_memory_payload_handles_empty_first_suspicious_txn_id():
    outcome = _successful_outcome(first_suspicious_txn_id="")
    payload = case_writer.build_case_memory_payload("HHG-001", outcome)
    assert payload["first_suspicious_txn_id"] is None


# --- decision 1: writer never consults G2 to decide whether to write ---

def test_writer_never_reads_g2_policy_result():
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(case_writer))
    source_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "g2_result" not in source_names
    assert "g2_policy" not in source_names
