"""
G9 -- end-to-end benchmark runner for the organizer's 20 case_pack cases.

This module implements NO new investigation, reasoning, validation, or
policy logic. It only orchestrates the already-existing, already-tested
pipeline exactly as it exists:

  workflow.investigate()            -- G1: benchmark resolution -> combined
                                        evidence -> conditional cross-card
                                        evidence -> ledger (real MCP calls,
                                        never TigerGraph directly)
  g2_policy.evaluate()               -- G2: deterministic policy
  case_output.investigate_and_assemble()
                                      -- G7/G8: evidence package -> ONE real
                                        LLM call -> deterministic validator
                                        -> case object / next_best_actions
  case_writer.write_case_memory()    -- G10: the SAME outcome object, if and
                                        only if it succeeded, persisted via
                                        the one existing MCP write tool --
                                        never a second reasoning step, never
                                        a second LLM call

The runner adds only:
  - iteration over the 20 case_pack rows
  - per-case timing
  - a thin, non-invasive recording wrapper around the injected call_llm,
    used ONLY to capture the raw LLM response for the audit artifact --
    it forwards to the real call_llm unchanged and adds no additional
    call, no retry, and no timeout of its own (the existing
    reasoning.DEFAULT_LLM_TIMEOUT_S, 60s, applies exactly as before)
  - a real MCP dispatcher scoped to exactly the one G10 write tool
    (_real_write_call_tool), mirroring workflow._default_call_tool's own
    contract, for the same reason (zero live dependency in tests)
  - JSON artifact writing (outputs/case_results/<CASE_ID>.json,
    outputs/evaluation_summary.json), now including written_to_graph/
    graph_case_id/graph_write alongside the fields already written
  - per-case exception isolation, so one case's failure (G1, G2, LLM, or
    G10 graph write) never stops the remaining 19 cases

Never calls TigerGraph directly and never bypasses MCP: the only paths to
graph data are workflow.investigate()'s own call_tool (defaults to the
real MCP dispatch, src.agent.workflow._default_call_tool ->
src.mcp_server.server.mcp.call_tool, exactly as G1 already established)
and _real_write_call_tool below, which is hard-scoped to the single
literal tool name "write_case_memory" and can never become a
general-purpose mutation dispatcher.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from src.agent import case_output as co
from src.agent import case_writer
from src.agent import workflow as wf
from src.agent.reasoning import CallLLM
from src.graph.tigergraph_connection import scrub_secret
from src.mcp_server.server import mcp
from src.policy import g2_policy as g2

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASE_PACK_PATH = REPO_ROOT / "data" / "processed" / "case_pack_resolved.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"

# The two categories of failure_reason values this project's own code
# actually produces (src.agent.reasoning.get_llm_reasoning and
# src.agent.reasoning_validator.validate_reasoning_output respectively) --
# not invented here, quoted exactly from those modules, used only to
# classify an already-produced failure_reason for the summary.
_LLM_CALL_FAILURE_REASONS = frozenset({
    "llm_timeout", "llm_call_failed", "invalid_response_type", "invalid_json",
})
_VALIDATOR_FAILURE_REASONS = frozenset({
    "malformed_output", "invalid_verdict", "invalid_probability", "invalid_pattern",
})


def _load_case_pack(case_pack_path: Path) -> list[dict]:
    with open(case_pack_path, newline="") as f:
        return list(csv.DictReader(f))


# G10 integration: a NOT-attempted result, used whenever the investigation
# itself didn't succeed -- the writer is never invoked in that case (see
# run_single_case below), so this is the one, fixed shape reported instead.
_GRAPH_WRITE_NOT_ATTEMPTED = {
    "attempted": False,
    "written_to_graph": False,
    "graph_case_id": "",
    "reason": "investigation_not_successful",
    "detail": None,
    "steps": None,
}


async def _real_write_call_tool(name: str, arguments: dict) -> tuple[bool, Any, str | None]:
    """Real MCP dispatch for src.agent.case_writer's one write call --
    mirrors src.agent.workflow._default_call_tool's exact (ok, value,
    error) contract and never-raises guarantee, scoped ONLY to the
    single G10 write tool. `name` is never constructed from data --
    the one call site below always passes the literal string
    "write_case_memory" -- so this can never become a general-purpose
    TigerGraph mutation dispatcher.
    """
    if name != "write_case_memory":
        raise ValueError(f"tool {name!r} is not the G10 write tool this dispatcher is scoped to")
    try:
        result = await mcp.call_tool(name, arguments)
    except Exception as e:
        return False, None, scrub_secret(str(e))
    if result.is_error:
        text_parts = [getattr(c, "text", "") for c in (result.content or [])]
        message = " ".join(p for p in text_parts if p) or f"{name} returned an error"
        return False, None, scrub_secret(message)
    sc = result.structured_content
    if sc is not None:
        value = sc.get("result") if isinstance(sc, dict) and "result" in sc else sc
    else:
        # Some MCP SDK return-type shapes only populate the text content
        # block, not structured_content -- confirmed empirically during
        # G10's own live verification. Parse that instead of losing the
        # result.
        text_parts = [getattr(c, "text", "") for c in (result.content or [])]
        try:
            value = json.loads("".join(text_parts)) if text_parts else None
        except json.JSONDecodeError:
            value = None
    return True, value, None


def _make_recording_call_llm(call_llm: CallLLM) -> tuple[CallLLM, dict]:
    """Wraps a real CallLLM purely for audit capture. Forwards to
    `call_llm` unchanged, adds no additional call, no retry, and no
    timeout -- reasoning.get_llm_reasoning's own asyncio.wait_for
    (DEFAULT_LLM_TIMEOUT_S) still governs the single call exactly as
    before. `record` is populated with whatever this one call actually
    returned or raised, for the case-result artifact only.
    """
    record: dict = {"raw_response": None, "call_error": None}

    async def wrapped(prompt: str) -> str:
        try:
            result = await call_llm(prompt)
        except Exception as e:
            record["call_error"] = scrub_secret(str(e))
            raise
        record["raw_response"] = result
        return result

    return wrapped, record


async def run_single_case(case_id: str, call_llm: CallLLM, *, write_call_tool=None) -> dict:
    """Runs the existing, unmodified pipeline for one case_id trigger.
    Exactly one LLM call is made if G1/G2 succeed (case_output.
    investigate_and_assemble's own existing guarantee, unchanged here).
    Raises on a genuine pipeline exception (G1/G2) -- the caller
    (run_all_cases) is responsible for per-case exception isolation, so
    this function itself stays a direct, unmodified reflection of the
    real pipeline for anyone testing it in isolation.

    G10 integration (added here, the smallest point available): once
    `outcome` exists, if and ONLY if the investigation itself succeeded
    (outcome["ok"]), the already-implemented, already-tested
    src.agent.case_writer.write_case_memory is invoked exactly once,
    with the SAME outcome object -- no new field mapping, no second
    reasoning step, no additional LLM call. On any G9 failure, the
    writer is never invoked at all (not even case_writer's own internal
    short-circuit is relied on here -- the gate is explicit, at this
    call site, so "the writer was never called" is trivially true, not
    just structurally guaranteed one layer down). `write_call_tool`
    defaults to the real MCP dispatch (_real_write_call_tool); tests
    inject a fake to avoid any live TigerGraph/MCP dependency.
    """
    recording_call_llm, record = _make_recording_call_llm(call_llm)
    write_call_tool = write_call_tool or _real_write_call_tool

    t0 = time.perf_counter()
    ledger = await wf.investigate({"type": "case_id", "value": case_id})
    t1 = time.perf_counter()

    g2_result = g2.evaluate(ledger)

    t2 = time.perf_counter()
    outcome = await co.investigate_and_assemble(ledger, g2_result, call_llm=recording_call_llm)
    t3 = time.perf_counter()

    if outcome.get("ok"):
        graph_write = await case_writer.write_case_memory(case_id, outcome, call_tool=write_call_tool)
    else:
        graph_write = dict(_GRAPH_WRITE_NOT_ATTEMPTED)

    # Mirrors the two fields into outcome["case"] itself, matching the
    # organizer's own case.written_to_graph/case.graph_case_id fields
    # (case_output.py is not modified to produce these -- they are
    # merged in here, additively, onto the dict it already returned).
    # Every existing key in outcome["case"] is left exactly as
    # case_output.py produced it.
    if outcome.get("ok"):
        outcome["case"]["written_to_graph"] = graph_write["written_to_graph"]
        outcome["case"]["graph_case_id"] = graph_write["graph_case_id"]

    full_evidence = ledger.get("full_evidence") or {}
    combined_evidence = full_evidence.get("combined_evidence") or {}

    return {
        "case_id": case_id,
        "investigation_id": ledger.get("investigation_id"),
        "trigger": ledger.get("trigger"),
        "flagged_transaction": combined_evidence.get("transaction"),
        "g1": {
            "recommendation": ledger.get("recommendation"),
            "recommendation_basis": ledger.get("recommendation_basis"),
            "tool_calls": ledger.get("tool_calls"),
            "uncertainties": ledger.get("uncertainties"),
            "pattern_evidence": ledger.get("pattern_evidence"),
        },
        "g2_policy": g2_result,
        "llm": {
            "raw_response": record["raw_response"],
            "call_error": record["call_error"],
        },
        "outcome": outcome,
        "success": bool(outcome.get("ok")),
        # G10: always present, regardless of success/failure, so every
        # case result has a stable location for these two fields --
        # reasoning success and graph-write success are reported
        # independently, never conflated.
        "written_to_graph": graph_write["written_to_graph"],
        "graph_case_id": graph_write["graph_case_id"],
        "graph_write": graph_write,
        "timing_s": {
            "g1_evidence": round(t1 - t0, 3),
            "llm_and_validation": round(t3 - t2, 3),
            "total": round(t3 - t0, 3),
        },
    }


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _classify_failure(result: dict) -> str:
    if result.get("success"):
        return "none"
    outcome = result.get("outcome") or {}
    reason = outcome.get("failure_reason")
    if reason in _LLM_CALL_FAILURE_REASONS:
        return "llm_call_failure"
    if reason in _VALIDATOR_FAILURE_REASONS:
        return "validator_failure"
    if result.get("stage") == "runner_exception":
        return "pipeline_exception"
    return "unknown"


def _build_summary(results: list[dict], total_runtime_s: float) -> dict:
    total = len(results)
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    llm_failures = [r for r in failed if _classify_failure(r) == "llm_call_failure"]
    validator_failures = [r for r in failed if _classify_failure(r) == "validator_failure"]
    pipeline_exceptions = [r for r in failed if _classify_failure(r) == "pipeline_exception"]

    failure_reason_counts: dict[str, int] = {}
    for r in failed:
        outcome = r.get("outcome") or {}
        reason = outcome.get("failure_reason") or r.get("error") or "unknown"
        failure_reason_counts[reason] = failure_reason_counts.get(reason, 0) + 1

    policy_results = {
        r["case_id"]: [e["action"] for e in (r.get("g2_policy") or {}).get("recommended", [])]
        for r in results
    }

    execution_time_per_case = {
        r["case_id"]: (r.get("timing_s") or {}).get("total")
        for r in results
    }

    return {
        "total_cases": total,
        "successful_cases": len(successful),
        "failed_cases": len(failed),
        "llm_successes": total - len(llm_failures) - len(pipeline_exceptions),
        "llm_failures": len(llm_failures),
        "validator_failures": len(validator_failures),
        "pipeline_exceptions": len(pipeline_exceptions),
        "policy_results": policy_results,
        "execution_time_per_case_s": execution_time_per_case,
        "total_runtime_s": round(total_runtime_s, 3),
        "failure_reasons": failure_reason_counts,
        # Explicitly NOT computed: no organizer answer key is available in
        # this repository, so no accuracy/correctness score is calculated
        # or invented here -- see docs/AGENT_WORKFLOW_SPEC.md Section 14.
    }


async def run_all_cases(
    call_llm: CallLLM,
    *,
    case_pack_path: Path = DEFAULT_CASE_PACK_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    case_ids: list[str] | None = None,
    write_call_tool=None,
) -> dict:
    """Runs every row in case_pack_path (or only `case_ids`, if given) --
    a single case's failure (G1, G2, LLM/validator, or G10 graph write)
    is caught and recorded, never stopping the remaining cases. Writes
    one JSON file per case plus one evaluation_summary.json under
    output_dir. `write_call_tool` is threaded through to
    run_single_case unchanged (defaults to the real MCP dispatch; tests
    inject a fake).
    """
    rows = _load_case_pack(case_pack_path)
    if case_ids is not None:
        wanted = set(case_ids)
        rows = [r for r in rows if r["case_id"] in wanted]

    results: list[dict] = []
    run_t0 = time.perf_counter()

    for row in rows:
        case_id = row["case_id"]
        try:
            result = await run_single_case(case_id, call_llm, write_call_tool=write_call_tool)
        except Exception as e:
            result = {
                "case_id": case_id,
                "success": False,
                "stage": "runner_exception",
                "error": scrub_secret(str(e)),
                # G10: kept present and consistent even on a G1/G2
                # pipeline exception -- no graph write was ever
                # attempted, and this is the one, fixed shape reported.
                "written_to_graph": False,
                "graph_case_id": "",
                "graph_write": dict(_GRAPH_WRITE_NOT_ATTEMPTED),
            }
        results.append(result)
        _write_json(output_dir / "case_results" / f"{case_id}.json", result)

    total_runtime_s = time.perf_counter() - run_t0
    summary = _build_summary(results, total_runtime_s)
    _write_json(output_dir / "evaluation_summary.json", summary)

    return {"results": results, "summary": summary}


def _build_call_llm(provider: str) -> CallLLM:
    if provider == "gemini":
        from src.agent.llm_providers.gemini_provider import make_gemini_call_llm
        return make_gemini_call_llm()
    if provider == "groq":
        from src.agent.llm_providers.groq_provider import make_groq_call_llm
        return make_groq_call_llm()
    raise ValueError(f"unknown provider: {provider!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="G9 end-to-end benchmark runner")
    parser.add_argument("--provider", choices=["gemini", "groq"], default="gemini")
    parser.add_argument("--case", action="append", dest="cases", default=None, help="restrict to one case_id (repeatable)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case-pack", type=Path, default=DEFAULT_CASE_PACK_PATH)
    args = parser.parse_args()

    call_llm = _build_call_llm(args.provider)
    outcome = asyncio.run(run_all_cases(
        call_llm,
        case_pack_path=args.case_pack,
        output_dir=args.output_dir,
        case_ids=args.cases,
    ))
    print(json.dumps(outcome["summary"], indent=2, default=str))


if __name__ == "__main__":
    main()
