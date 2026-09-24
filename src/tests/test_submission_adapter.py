"""
G11-A unit tests for src/benchmark/submission_adapter.py.

Every test here uses in-memory fixture dicts shaped exactly like
g9_runner.run_single_case's real return value (see test_g9_runner.py for
the same convention) -- no live TigerGraph, no live LLM provider, no
dependency on outputs/case_results/ containing any particular file.
"""
import json

import pytest

from src.benchmark import submission_adapter as sa


def _success_result(case_id="HHG-999", *, file_report=False, written_to_graph=True):
    final_actions = [
        {"action": "CREATE_CASE", "route": "auto", "reason": "§3a: customer disputes a charge"},
    ]
    if file_report:
        final_actions.append(
            {"action": "FILE_REPORT", "route": "L2", "reason": "R6: shared device links a second card"}
        )
    return {
        "case_id": case_id,
        "investigation_id": f"inv-{case_id}",
        "trigger": {"type": "case_id", "value": case_id},
        "g1": {
            "recommendation": "escalate",
            "tool_calls": [
                {"order": 1, "tool": "benchmark_case_resolution", "arguments": {}, "success": True},
                {"order": 2, "tool": "combined_evidence", "arguments": {}, "success": True},
            ],
        },
        "g2_policy": {"recommended": final_actions, "prohibited": [], "deferred": []},
        "llm": {"raw_response": "{...}", "call_error": None},
        "outcome": {
            "ok": True,
            "case": {
                "status": "escalated",
                "verdict": "uncertain",
                "fraud_probability": 0.45,
                "pattern": "none",
                "pattern_description": "",
                "affected_txn_ids": ["3530164"],
                "first_suspicious_txn_id": "3530164",
                "exposure_usd": 49,
                "evidence": [{"claim": "x", "source": "graph", "ref": "y", "entity_ids": []}],
                "similar_prior_cases": ["CC-2817"],
                "summary": "A short summary.",
            },
            "next_best_actions": {
                "initial": final_actions,
                "final": final_actions,
                "what_changed": "nothing",
            },
            "stripped_ids": [],
            "stop_reason": "G1 investigation completed; validated verdict produced.",
        },
        "success": True,
        "written_to_graph": written_to_graph,
        "graph_case_id": case_id if written_to_graph else "",
        "graph_write": {"attempted": True, "written_to_graph": written_to_graph, "graph_case_id": case_id if written_to_graph else ""},
        "timing_s": {"g1_evidence": 3.1, "llm_and_validation": 17.0, "total": 20.1},
    }


def _investigation_failure_result(case_id="HHG-998"):
    return {
        "case_id": case_id,
        "investigation_id": f"inv-{case_id}",
        "trigger": {"type": "case_id", "value": case_id},
        "g1": {"recommendation": "escalate", "tool_calls": [{"order": 1, "tool": "benchmark_case_resolution", "arguments": {}, "success": True}]},
        "g2_policy": {"recommended": [], "prohibited": [], "deferred": []},
        "llm": {"raw_response": None, "call_error": "503 UNAVAILABLE"},
        "outcome": {
            "ok": False,
            "failure_reason": "llm_call_failed",
            "detail": "503 UNAVAILABLE",
            "stop_reason": "LLM reasoning layer failed (llm_call_failed); no case object produced rather than fabricating one",
        },
        "success": False,
        "written_to_graph": False,
        "graph_case_id": "",
        "graph_write": {"attempted": False, "written_to_graph": False, "graph_case_id": "", "reason": "investigation_not_successful"},
        "timing_s": {"g1_evidence": 3.1, "llm_and_validation": 60.0, "total": 63.1},
    }


def _runner_exception_result(case_id="HHG-997"):
    return {
        "case_id": case_id,
        "success": False,
        "stage": "runner_exception",
        "error": "simulated G1 failure",
        "written_to_graph": False,
        "graph_case_id": "",
        "graph_write": {"attempted": False, "written_to_graph": False, "graph_case_id": "", "reason": "investigation_not_successful"},
    }


_TOP_LEVEL_KEYS = {
    "case_id", "case", "evidence_requests", "next_best_actions", "sar",
    "stop_reason", "tool_calls", "tokens", "latency_s",
}
_CASE_KEYS = {
    "status", "verdict", "fraud_probability", "pattern", "pattern_description",
    "affected_txn_ids", "first_suspicious_txn_id", "connected_card_ids",
    "connected_device_profiles", "exposure_usd", "evidence", "similar_prior_cases",
    "summary", "written_to_graph", "graph_case_id",
}
_SAR_KEYS = {"file", "reason", "narrative", "subjects", "total_amount_usd", "activity_dates"}
_NBA_KEYS = {"initial", "final", "what_changed"}


# 1. exact top-level keys
def test_top_level_keys_match_organizer_schema_exactly():
    out = sa.build_submission_case(_success_result())
    assert set(out.keys()) == _TOP_LEVEL_KEYS


# 2. exact nesting
def test_nested_object_keys_match_organizer_schema_exactly():
    out = sa.build_submission_case(_success_result(file_report=True))
    assert set(out["case"].keys()) == _CASE_KEYS
    assert set(out["sar"].keys()) == _SAR_KEYS
    assert set(out["next_best_actions"].keys()) == _NBA_KEYS


# 3. exact case_id
def test_case_id_is_preserved_exactly():
    out = sa.build_submission_case(_success_result(case_id="HHG-011"))
    assert out["case_id"] == "HHG-011"


# 4. written_to_graph mapping
def test_written_to_graph_mapped_from_top_level_result():
    out_true = sa.build_submission_case(_success_result(written_to_graph=True))
    out_false = sa.build_submission_case(_success_result(written_to_graph=False))
    assert out_true["case"]["written_to_graph"] is True
    assert out_false["case"]["written_to_graph"] is False


# 5. graph_case_id mapping
def test_graph_case_id_mapped_and_empty_when_not_written():
    out_true = sa.build_submission_case(_success_result(case_id="HHG-005", written_to_graph=True))
    out_false = sa.build_submission_case(_success_result(case_id="HHG-005", written_to_graph=False))
    assert out_true["case"]["graph_case_id"] == "HHG-005"
    assert out_false["case"]["graph_case_id"] == ""


# 6. next_best_actions mapping
def test_next_best_actions_copied_verbatim_on_success():
    result = _success_result(file_report=True)
    out = sa.build_submission_case(result)
    assert out["next_best_actions"] == result["outcome"]["next_best_actions"]


# 7. evidence mapping
def test_evidence_copied_verbatim_on_success():
    result = _success_result()
    out = sa.build_submission_case(result)
    assert out["case"]["evidence"] == result["outcome"]["case"]["evidence"]


# 8. SAR mapping
def test_sar_file_true_derived_from_file_report_action():
    result = _success_result(file_report=True)
    out = sa.build_submission_case(result)
    assert out["sar"]["file"] is True
    assert out["sar"]["reason"] == "R6: shared device links a second card"
    assert out["sar"]["narrative"] == ""


def test_sar_file_false_when_no_file_report_action():
    out = sa.build_submission_case(_success_result(file_report=False))
    assert out["sar"] == sa._EMPTY_SAR


# 9. evidence_requests mapping
def test_evidence_requests_always_empty():
    assert sa.build_submission_case(_success_result())["evidence_requests"] == []
    assert sa.build_submission_case(_investigation_failure_result())["evidence_requests"] == []
    assert sa.build_submission_case(_runner_exception_result())["evidence_requests"] == []


# 10. stop_reason mapping
def test_stop_reason_mapped_for_all_three_shapes():
    success_out = sa.build_submission_case(_success_result())
    assert success_out["stop_reason"] == "G1 investigation completed; validated verdict produced."

    failure_out = sa.build_submission_case(_investigation_failure_result())
    assert "llm_call_failed" in failure_out["stop_reason"]

    exc_out = sa.build_submission_case(_runner_exception_result())
    assert exc_out["stop_reason"] == "simulated G1 failure"


# 11. tool_calls mapping
def test_tool_calls_counts_g1_calls_plus_graph_write_attempt():
    out = sa.build_submission_case(_success_result())
    assert out["tool_calls"] == 3  # 2 G1 tool calls + 1 attempted graph write

    failure_out = sa.build_submission_case(_investigation_failure_result())
    assert failure_out["tool_calls"] == 1  # 1 G1 tool call, graph write never attempted


# 12. tokens mapping
def test_tokens_reported_as_zero_not_tracked():
    assert sa.build_submission_case(_success_result())["tokens"] == 0
    assert sa.build_submission_case(_investigation_failure_result())["tokens"] == 0


# 13. latency_s mapping
def test_latency_s_mapped_from_timing_total():
    out = sa.build_submission_case(_success_result())
    assert out["latency_s"] == 20.1

    exc_out = sa.build_submission_case(_runner_exception_result())
    assert exc_out["latency_s"] == 0.0


# 14. no internal debug-only fields leak into submission output
def test_no_internal_fields_leak_into_submission_output():
    out = sa.build_submission_case(_success_result(file_report=True))
    serialized = json.dumps(out)
    for leaked in ("g1_policy", "g2_policy", "raw_response", "investigation_id",
                   "recommendation_basis", "curated_result_summary", "stripped_ids"):
        assert leaked not in serialized


# 15. deterministic output
def test_output_is_deterministic_across_repeated_calls():
    result = _success_result(file_report=True)
    out1 = sa.build_submission_case(result)
    out2 = sa.build_submission_case(result)
    assert out1 == out2


# 16. missing/failed investigation handled honestly
def test_investigation_failure_produces_honest_empty_case_not_a_crash():
    out = sa.build_submission_case(_investigation_failure_result())
    assert out["case"]["verdict"] == "uncertain"
    assert out["case"]["affected_txn_ids"] == []
    assert out["case"]["exposure_usd"] == 0
    assert out["next_best_actions"] == {"initial": [], "final": [], "what_changed": "nothing"}


def test_runner_exception_produces_honest_empty_case_not_a_crash():
    out = sa.build_submission_case(_runner_exception_result())
    assert out["case"] == sa._EMPTY_CASE
    assert out["sar"] == sa._EMPTY_SAR


# 17. no invented values
def test_connected_card_fields_are_never_invented():
    out = sa.build_submission_case(_success_result())
    assert out["case"]["connected_card_ids"] == []
    assert out["case"]["connected_device_profiles"] == []


def test_verdict_and_pattern_values_are_never_altered_from_internal_result():
    result = _success_result()
    out = sa.build_submission_case(result)
    assert out["case"]["verdict"] == result["outcome"]["case"]["verdict"]
    assert out["case"]["pattern"] == result["outcome"]["case"]["pattern"]
    assert out["case"]["fraud_probability"] == result["outcome"]["case"]["fraud_probability"]


# 18. 20 case IDs can be emitted
def test_all_20_case_ids_can_be_written(tmp_path):
    case_results_dir = tmp_path / "case_results"
    case_results_dir.mkdir()
    output_dir = tmp_path / "cases"

    for i in range(1, 21):
        case_id = f"HHG-{i:03d}"
        with open(case_results_dir / f"{case_id}.json", "w") as f:
            json.dump(_success_result(case_id=case_id), f)

    paths = sa.write_submission_files_from_case_results(case_results_dir, output_dir)
    assert len(paths) == 20
    written = sorted(p.name for p in output_dir.glob("HHG-*.json"))
    assert written == [f"HHG-{i:03d}.json" for i in range(1, 21)]


# 19. output directory is exactly cases/
def test_default_submission_dir_is_named_cases():
    assert sa.DEFAULT_SUBMISSION_DIR.name == "cases"
    assert sa.DEFAULT_SUBMISSION_DIR.parent == sa.REPO_ROOT


# 20. existing G9 output remains unchanged
def test_writing_submission_file_does_not_touch_case_results_dir(tmp_path):
    case_results_dir = tmp_path / "case_results"
    case_results_dir.mkdir()
    result = _success_result(case_id="HHG-003")
    internal_path = case_results_dir / "HHG-003.json"
    with open(internal_path, "w") as f:
        json.dump(result, f)
    before = internal_path.read_text()

    output_dir = tmp_path / "cases"
    sa.write_submission_file(result, output_dir=output_dir)

    after = internal_path.read_text()
    assert before == after
    assert (output_dir / "HHG-003.json").exists()
