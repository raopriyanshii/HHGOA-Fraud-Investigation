"""
G11-A -- deterministic organizer-submission adapter.

Reads the already-produced internal per-case result (the exact dict
src.benchmark.g9_runner.run_single_case/run_all_cases already returns
and writes to outputs/case_results/<case_id>.json) and re-shapes it into
the organizer's required submission file:

    cases/<case_id>.json

matching the README's "Answer Format" section (top level +
case/sar/next_best_actions) field-for-field.

This module implements NO new investigation, reasoning, validation,
policy, or graph-write logic. Every field below is either copied
directly from the internal result, or is a disclosed, honest empty/zero
placeholder for a capability this project does not yet implement:

  - `evidence_requests` -- copied verbatim from the internal result
    (case_output.py's G12 evidence-request loop, src.agent.evidence_loop)
    on success; `[]` on any failure path, honestly, never fabricated
  - `sar` -- copied verbatim from the internal result (case_output.py's
    G13 `_build_sar`, the one place `sar` is constructed) on success;
    the organizer's exact required empty shape (`_EMPTY_SAR`) on any
    failure path. This module does not derive or duplicate any SAR
    field itself.
  - `tokens` -- this project never captures LLM usage/token-count
    metadata from either provider; reported as 0, not estimated
  - `case.connected_card_ids`/`connected_device_profiles` -- case_output.py
    explicitly documents these as out of its own scope ("later-phase
    work, not invented here"); left honestly empty rather than
    reconstructed from raw, un-vetted evidence here
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASE_RESULTS_DIR = REPO_ROOT / "outputs" / "case_results"
DEFAULT_SUBMISSION_DIR = REPO_ROOT / "cases"

_EMPTY_CASE = {
    "status": "open",
    "verdict": "uncertain",
    "fraud_probability": 0.0,
    "pattern": "none",
    "pattern_description": "",
    "affected_txn_ids": [],
    "first_suspicious_txn_id": "",
    "connected_card_ids": [],
    "connected_device_profiles": [],
    "exposure_usd": 0,
    "evidence": [],
    "similar_prior_cases": [],
    "summary": "",
    "written_to_graph": False,
    "graph_case_id": "",
}

_EMPTY_SAR = {
    "file": False,
    "reason": "",
    "narrative": "",
    "subjects": [],
    "total_amount_usd": 0,
    "activity_dates": [],
}

_EMPTY_NEXT_BEST_ACTIONS = {"initial": [], "final": [], "what_changed": "nothing"}


def _case_from_success(case: dict, written_to_graph: bool, graph_case_id: str) -> dict:
    return {
        "status": case["status"],
        "verdict": case["verdict"],
        "fraud_probability": case["fraud_probability"],
        "pattern": case["pattern"],
        "pattern_description": case["pattern_description"],
        "affected_txn_ids": case["affected_txn_ids"],
        "first_suspicious_txn_id": case["first_suspicious_txn_id"],
        "connected_card_ids": [],
        "connected_device_profiles": [],
        "exposure_usd": case["exposure_usd"],
        "evidence": case["evidence"],
        "similar_prior_cases": case["similar_prior_cases"],
        "summary": case["summary"],
        "written_to_graph": written_to_graph,
        "graph_case_id": graph_case_id,
    }


def _tool_calls_count(result: dict) -> int:
    g1_tool_calls = (result.get("g1") or {}).get("tool_calls") or []
    graph_write = result.get("graph_write") or {}
    n = len(g1_tool_calls)
    if graph_write.get("attempted"):
        n += 1
    return n


def build_submission_case(result: dict) -> dict:
    """`result` is exactly one entry of the list g9_runner.run_all_cases
    returns (equivalently, one outputs/case_results/<case_id>.json file),
    covering all three shapes that module produces: a full success, an
    investigation failure (outcome["ok"] is False, g1/g2 still present),
    or a runner_exception (only case_id/success/stage/error present).
    """
    case_id = result["case_id"]
    written_to_graph = bool(result.get("written_to_graph", False))
    graph_case_id = result.get("graph_case_id") or ""

    outcome = result.get("outcome") or {}
    if result.get("success") and outcome.get("ok"):
        case = _case_from_success(outcome["case"], written_to_graph, graph_case_id)
        next_best_actions = outcome["next_best_actions"]
        evidence_requests = outcome.get("evidence_requests", [])
        sar = outcome.get("sar") or dict(_EMPTY_SAR)
        stop_reason = outcome.get("stop_reason", "")
    else:
        case = dict(_EMPTY_CASE)
        case["written_to_graph"] = written_to_graph
        case["graph_case_id"] = graph_case_id
        next_best_actions = dict(_EMPTY_NEXT_BEST_ACTIONS)
        evidence_requests = []
        sar = dict(_EMPTY_SAR)
        stop_reason = outcome.get("stop_reason") or result.get("error") or "investigation did not complete"

    timing = result.get("timing_s") or {}

    return {
        "case_id": case_id,
        "case": case,
        "evidence_requests": evidence_requests,
        "next_best_actions": next_best_actions,
        "sar": sar,
        "stop_reason": stop_reason,
        "tool_calls": _tool_calls_count(result),
        "tokens": 0,
        "latency_s": timing.get("total", 0.0),
    }


def write_submission_file(result: dict, output_dir: Path = DEFAULT_SUBMISSION_DIR) -> Path:
    submission = build_submission_case(result)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{submission['case_id']}.json"
    with open(path, "w") as f:
        json.dump(submission, f, indent=2, default=str)
    return path


def write_submission_files_from_case_results(
    case_results_dir: Path = DEFAULT_CASE_RESULTS_DIR,
    output_dir: Path = DEFAULT_SUBMISSION_DIR,
) -> list[Path]:
    paths = []
    for p in sorted(case_results_dir.glob("HHG-*.json")):
        with open(p) as f:
            result = json.load(f)
        paths.append(write_submission_file(result, output_dir=output_dir))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="G11-A organizer-submission adapter")
    parser.add_argument("--case-results-dir", type=Path, default=DEFAULT_CASE_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SUBMISSION_DIR)
    args = parser.parse_args()

    paths = write_submission_files_from_case_results(args.case_results_dir, args.output_dir)
    print(f"wrote {len(paths)} submission file(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
