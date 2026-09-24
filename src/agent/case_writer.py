"""
G10 -- deterministic case-memory writer.

Consumes the ALREADY-VALIDATED output of
case_output.investigate_and_assemble() (ok=True only) plus the G1 ledger,
and persists it to TigerGraph via exactly one MCP tool
(write_case_memory) -- never a direct TigerGraph call, mirroring
src.agent.workflow's own call_tool injection pattern exactly, for the
same reasons (zero live dependency in tests, swappable dispatch).

The LLM has no path to this module at all: it never sees a case_id,
never constructs a payload, and never calls a tool. This module builds
the entire write payload itself, from data that is already fully
deterministic by the time it arrives here -- outcome["case"] is the
output of reasoning_validator.py's own grounding/enum/probability
checks, not raw LLM text. No GSQL is ever constructed from LLM-generated
content; only already-validated scalar/list values become fixed API
parameters (case_id, verdict, fraud_probability, ... -- see
write_case_memory's own fixed signature in src/mcp_server/server.py).

Decision (locked, not decided by this module): persist every
reasoning-successful investigation, regardless of what G2 recommends --
case-memory persistence and action execution/approval are unconditionally
separate. This module never reads g2_result to decide WHETHER to write,
only outcome["ok"]. G2's own recommended/prohibited/deferred actions are
never executed by this module, and a successful write never triggers any
of the 14 fraud-policy actions.

A reasoning success (outcome["ok"]) and a graph-write success
(the returned "written_to_graph") are independent states -- one can be
True while the other is False, and this module never conflates them.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Awaitable, Callable

from src.graph.tigergraph_connection import scrub_secret

CallTool = Callable[[str, dict], Awaitable[tuple[bool, Any, str | None]]]

_VALID_VERDICTS = ("fraud", "legitimate", "uncertain")


def _validate_payload(payload: dict) -> str | None:
    """Returns None if the payload is well-formed, else a short
    description of what's wrong. Never raises -- a malformed payload is
    reported as such, not crashed on, and never reaches the MCP call.
    """
    if not payload.get("case_id") or not isinstance(payload["case_id"], str):
        return "missing or invalid case_id"
    if payload.get("verdict") not in _VALID_VERDICTS:
        return f"invalid verdict: {payload.get('verdict')!r}"
    prob = payload.get("fraud_probability")
    if isinstance(prob, bool) or not isinstance(prob, (int, float)) or not (0.0 <= prob <= 1.0):
        return f"invalid fraud_probability: {prob!r}"
    affected = payload.get("affected_txn_ids")
    if not isinstance(affected, list) or not all(isinstance(t, int) and not isinstance(t, bool) for t in affected):
        return "affected_txn_ids must be a list of ints"
    similar = payload.get("similar_prior_cases")
    if not isinstance(similar, list) or not all(isinstance(c, str) for c in similar):
        return "similar_prior_cases must be a list of strings"
    first = payload.get("first_suspicious_txn_id")
    if first is not None and (isinstance(first, bool) or not isinstance(first, int)):
        return "first_suspicious_txn_id must be an int or null"
    if not isinstance(payload.get("evidence"), list):
        return "evidence must be a list"
    if not isinstance(payload.get("next_best_actions"), dict):
        return "next_best_actions must be a dict"
    for key in ("pattern", "pattern_description", "summary", "stop_reason"):
        if not isinstance(payload.get(key), str):
            return f"{key} must be a string"
    exposure = payload.get("exposure_usd")
    if isinstance(exposure, bool) or not isinstance(exposure, (int, float)):
        return "exposure_usd must be a number"
    return None


def build_case_memory_payload(case_id: str, outcome: dict) -> dict:
    """Pure, deterministic construction from already-validated data.
    `outcome` must be a successful (ok=True)
    case_output.investigate_and_assemble() result -- callers must check
    this before calling. Never invents a value: every field is read
    directly from outcome["case"]/outcome["next_best_actions"], never
    guessed or defaulted to something not already present.
    """
    case = outcome["case"]
    affected_txn_ids = [int(t) for t in (case.get("affected_txn_ids") or [])]
    first_suspicious_raw = case.get("first_suspicious_txn_id")
    first_suspicious_txn_id = int(first_suspicious_raw) if first_suspicious_raw else None

    return {
        "case_id": case_id,
        "verdict": case["verdict"],
        "fraud_probability": case["fraud_probability"],
        "pattern": case["pattern"],
        "pattern_description": case["pattern_description"],
        "exposure_usd": case["exposure_usd"],
        "summary": case["summary"],
        "stop_reason": outcome.get("stop_reason") or "",
        "evidence": case.get("evidence") or [],
        "next_best_actions": outcome.get("next_best_actions") or {},
        "affected_txn_ids": affected_txn_ids,
        "first_suspicious_txn_id": first_suspicious_txn_id,
        "similar_prior_cases": case.get("similar_prior_cases") or [],
        "updated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }


async def write_case_memory(case_id: str, outcome: dict, *, call_tool: CallTool) -> dict:
    """Top-level entry point -- the only function anything outside this
    module should call. Never raises; every failure mode is represented
    explicitly, never fabricated into a false success.

    Returns:
      {
        "attempted": bool,
        "written_to_graph": bool,   # False unless every graph step succeeded
        "graph_case_id": str,        # case_id only when written_to_graph is True, else ""
        "reason": str | None,        # why not written, when applicable
        "detail": str | None,        # scrubbed error detail, when applicable
        "steps": dict | None,        # per-step detail from the write, when attempted
      }
    """
    if not outcome.get("ok"):
        return {
            "attempted": False,
            "written_to_graph": False,
            "graph_case_id": "",
            "reason": "investigation_not_successful",
            "detail": None,
            "steps": None,
        }

    payload = build_case_memory_payload(case_id, outcome)
    error = _validate_payload(payload)
    if error is not None:
        return {
            "attempted": False,
            "written_to_graph": False,
            "graph_case_id": "",
            "reason": "malformed_payload",
            "detail": error,
            "steps": None,
        }

    ok, result, err = await call_tool("write_case_memory", payload)
    if not ok:
        return {
            "attempted": True,
            "written_to_graph": False,
            "graph_case_id": "",
            "reason": "mcp_write_failed",
            "detail": scrub_secret(err or ""),
            "steps": None,
        }

    written = bool((result or {}).get("written"))
    return {
        "attempted": True,
        "written_to_graph": written,
        "graph_case_id": (result or {}).get("graph_case_id") if written else "",
        "reason": None if written else "partial_or_failed_graph_write",
        "detail": None,
        "steps": (result or {}).get("steps"),
    }
