"""
G9 unit tests for src/benchmark/g9_runner.py.

Every test here mocks wf.investigate() (no live TigerGraph) and injects a
fake call_llm (no live LLM provider) -- the test suite never depends on
either. These test the RUNNER's own added behavior (iteration, per-case
isolation, artifact writing, summary construction) -- not G1/G2/G7/G8's
own logic, which is already covered by their own test suites and is
never reimplemented here.
"""
import asyncio
import json

import pytest

from src.benchmark import g9_runner


def run(coro):
    return asyncio.run(coro)


def _fake_ledger(case_id: str, flagged_txn_id: int = 1000) -> dict:
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
    "verdict": "uncertain", "fraud_probability": 0.3, "affected_txn_ids": [],
    "first_suspicious_txn_id": None, "pattern": "none", "pattern_description": "",
    "summary": "ok", "reasoning": "ok", "uncertainties": [],
})


def _patch_investigate(monkeypatch, ledger_for=None, raise_for=None):
    """ledger_for(case_id) -> ledger, or a fixed ledger if callable not given.
    raise_for: a set of case_ids for which wf.investigate itself raises.
    """
    raise_for = raise_for or set()

    async def fake_investigate(trigger, *, window_hours=24.0, call_tool=None):
        case_id = trigger["value"]
        if case_id in raise_for:
            raise RuntimeError(f"simulated G1 failure for {case_id}")
        if ledger_for is not None:
            return ledger_for(case_id)
        return _fake_ledger(case_id)

    monkeypatch.setattr(g9_runner.wf, "investigate", fake_investigate)


def _small_case_pack(tmp_path, case_ids):
    path = tmp_path / "case_pack.csv"
    with open(path, "w", newline="") as f:
        f.write("case_id,opened_at,trigger_type,trigger_text,flagged_txn_id,customer_id,card_key,risk_score\n")
        for cid in case_ids:
            f.write(f"{cid},2016-01-01 00:00:00,risk_score,irrelevant,1000,C00001,C00001:1,0.5\n")
    return path


# --- runner processes all 20 real cases ---

def test_runner_processes_all_20_real_case_pack_rows(monkeypatch, tmp_path):
    _patch_investigate(monkeypatch)

    async def fake_call_llm(prompt):
        return _VALID_LLM_RESPONSE

    outcome = run(g9_runner.run_all_cases(fake_call_llm, output_dir=tmp_path))
    assert outcome["summary"]["total_cases"] == 20
    assert outcome["summary"]["successful_cases"] == 20
    assert len(outcome["results"]) == 20

    case_files = list((tmp_path / "case_results").glob("HHG-*.json"))
    assert len(case_files) == 20
    assert (tmp_path / "evaluation_summary.json").exists()


# --- one case failure does not terminate the whole run ---

def test_one_case_failure_does_not_stop_the_remaining_cases(monkeypatch, tmp_path):
    _patch_investigate(monkeypatch, raise_for={"HHG-005"})

    async def fake_call_llm(prompt):
        return _VALID_LLM_RESPONSE

    outcome = run(g9_runner.run_all_cases(fake_call_llm, output_dir=tmp_path))
    assert outcome["summary"]["total_cases"] == 20
    assert outcome["summary"]["failed_cases"] == 1
    assert outcome["summary"]["successful_cases"] == 19

    failing = next(r for r in outcome["results"] if r["case_id"] == "HHG-005")
    assert failing["success"] is False
    assert failing["stage"] == "runner_exception"


# --- exactly one LLM call per successful case / no retries ---

def test_exactly_one_llm_call_per_case_no_retries(monkeypatch, tmp_path):
    case_ids = ["HHG-001", "HHG-002", "HHG-003"]
    case_pack = _small_case_pack(tmp_path, case_ids)
    _patch_investigate(monkeypatch)

    calls = []

    async def counting_call_llm(prompt):
        calls.append(prompt)
        return _VALID_LLM_RESPONSE

    run(g9_runner.run_all_cases(counting_call_llm, case_pack_path=case_pack, output_dir=tmp_path))
    assert len(calls) == len(case_ids)  # exactly one per case, no more


def test_llm_failure_for_one_case_does_not_trigger_extra_attempts(monkeypatch, tmp_path):
    case_ids = ["HHG-001", "HHG-002"]
    case_pack = _small_case_pack(tmp_path, case_ids)
    _patch_investigate(monkeypatch)

    calls = []

    async def flaky_call_llm(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            raise RuntimeError("simulated transient provider failure")
        return _VALID_LLM_RESPONSE

    outcome = run(g9_runner.run_all_cases(flaky_call_llm, case_pack_path=case_pack, output_dir=tmp_path))
    assert len(calls) == 2  # exactly one attempt per case -- no retry of the failed one
    assert outcome["summary"]["llm_failures"] == 1
    assert outcome["summary"]["successful_cases"] == 1


# --- provider failure is represented honestly, never fabricated ---

def test_provider_failure_is_recorded_honestly_not_fabricated(monkeypatch, tmp_path):
    case_pack = _small_case_pack(tmp_path, ["HHG-001"])
    _patch_investigate(monkeypatch)

    async def failing_call_llm(prompt):
        raise RuntimeError("503 UNAVAILABLE: model overloaded")

    outcome = run(g9_runner.run_all_cases(failing_call_llm, case_pack_path=case_pack, output_dir=tmp_path))
    result = outcome["results"][0]
    assert result["success"] is False
    assert "case" not in result["outcome"]  # no fabricated case object
    assert result["outcome"]["failure_reason"] == "llm_call_failed"
    assert "503 UNAVAILABLE" in result["outcome"]["detail"]


# --- secrets are never serialized into case artifacts ---

def test_secrets_are_not_serialized_into_case_result_files(monkeypatch, tmp_path):
    fake_secret = "gsk_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789ABCDEFGHIJKLM"
    monkeypatch.setenv("GROQ_API_KEY", fake_secret)
    case_pack = _small_case_pack(tmp_path, ["HHG-001"])
    _patch_investigate(monkeypatch)

    async def leaking_call_llm(prompt):
        raise RuntimeError(f"401 Unauthorized: key {fake_secret} rejected")

    run(g9_runner.run_all_cases(leaking_call_llm, case_pack_path=case_pack, output_dir=tmp_path))

    written = (tmp_path / "case_results" / "HHG-001.json").read_text()
    assert fake_secret not in written
    assert "[REDACTED]" in written

    summary_written = (tmp_path / "evaluation_summary.json").read_text()
    assert fake_secret not in summary_written


def test_runner_exception_message_is_scrubbed(monkeypatch, tmp_path):
    fake_secret = "gsk_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789ABCDEFGHIJKLM"
    monkeypatch.setenv("GROQ_API_KEY", fake_secret)
    case_pack = _small_case_pack(tmp_path, ["HHG-001"])

    async def leaking_investigate(trigger, *, window_hours=24.0, call_tool=None):
        raise RuntimeError(f"connection failed, key={fake_secret}")

    monkeypatch.setattr(g9_runner.wf, "investigate", leaking_investigate)

    async def unreachable_call_llm(prompt):
        raise AssertionError("must not be called if G1 itself failed")

    outcome = run(g9_runner.run_all_cases(unreachable_call_llm, case_pack_path=case_pack, output_dir=tmp_path))
    result = outcome["results"][0]
    assert fake_secret not in result["error"]
    assert "[REDACTED]" in result["error"]

    written = (tmp_path / "case_results" / "HHG-001.json").read_text()
    assert fake_secret not in written


# --- final output remains deterministic after LLM validation ---

def test_output_is_deterministic_across_repeated_runs_with_identical_llm_output(monkeypatch, tmp_path):
    case_pack = _small_case_pack(tmp_path, ["HHG-001"])
    _patch_investigate(monkeypatch)

    async def fixed_call_llm(prompt):
        return _VALID_LLM_RESPONSE

    outcome_1 = run(g9_runner.run_all_cases(fixed_call_llm, case_pack_path=case_pack, output_dir=tmp_path / "run1"))
    outcome_2 = run(g9_runner.run_all_cases(fixed_call_llm, case_pack_path=case_pack, output_dir=tmp_path / "run2"))

    case_1 = outcome_1["results"][0]["outcome"]["case"]
    case_2 = outcome_2["results"][0]["outcome"]["case"]
    assert case_1 == case_2  # same verdict, fraud_probability, exposure_usd, etc.


# --- existing G1/G2 behavior remains unchanged (structural check) ---

def test_runner_does_not_reimplement_g1_or_g2_logic():
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(g9_runner))
    function_defs = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    # none of G1/G2's own internal function names are shadowed/redefined here:
    assert "_assess" not in function_defs
    assert "_r6_evidence_present" not in function_defs
    assert "evaluate" not in function_defs
    assert "investigate" not in function_defs


def test_runner_imports_real_workflow_and_policy_modules():
    assert g9_runner.wf.__name__ == "src.agent.workflow"
    assert g9_runner.g2.__name__ == "src.policy.g2_policy"
    assert g9_runner.co.__name__ == "src.agent.case_output"


# --- summary does not invent an accuracy score ---

def test_summary_never_contains_an_accuracy_or_correctness_score(monkeypatch, tmp_path):
    case_pack = _small_case_pack(tmp_path, ["HHG-001"])
    _patch_investigate(monkeypatch)

    async def fake_call_llm(prompt):
        return _VALID_LLM_RESPONSE

    outcome = run(g9_runner.run_all_cases(fake_call_llm, case_pack_path=case_pack, output_dir=tmp_path))
    summary_keys = set(outcome["summary"].keys())
    for forbidden in ("accuracy", "score", "correct", "correctness", "precision", "recall"):
        assert not any(forbidden in k.lower() for k in summary_keys)
