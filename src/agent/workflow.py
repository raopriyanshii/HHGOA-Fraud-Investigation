"""
Phase G1 -- bounded investigation workflow, implementing
docs/AGENT_WORKFLOW_SPEC.md (frozen at commit 123473e) exactly.

No LLM. No policy engine. No autonomous action. This module only
orchestrates calls to the existing MCP tools (src/mcp_server/server.py)
and applies the fixed, deterministic V1 recommendation rule from spec
Section 6a. Every fact in the evidence ledger traces back to a real MCP
tool call -- this module never reasons about the graph itself and never
calls the underlying Phase E query functions directly (spec Section 9).

Hard limits (spec Section 4), enforced here, not just documented:
  - MAX_FOLLOW_UP_CARDS caps Round 2 at 3 tool calls.
  - There is no loop or recursion construct that could produce a Round 3:
    Round 1 is a fixed sequence of at most 2 calls (trigger resolution +
    combined_evidence), Round 2 is one bounded loop over <=3 selected
    cards, and nothing feeds Round 2's results back into another tool
    call.
  - Only 3 literal, hardcoded tool names are ever dispatched
    (benchmark_case_resolution, combined_evidence,
    historical_case_evidence) -- no tool name is ever constructed from
    trigger or evidence data.
"""
from __future__ import annotations

import uuid
from typing import Any, Awaitable, Callable

from src.graph.tigergraph_connection import scrub_secret
from src.investigation.patterns import classify_card_testing
from src.mcp_server.server import mcp

MAX_FOLLOW_UP_CARDS = 3

# The only tool names this workflow will ever dispatch. Not a general
# "call anything" surface -- a hardcoded allowlist, checked defensively
# in _default_call_tool below.
_ALLOWED_TOOLS = frozenset(
    {"benchmark_case_resolution", "combined_evidence", "historical_case_evidence", "cross_card_fraud_verification"}
)

CallTool = Callable[[str, dict], Awaitable[tuple[bool, Any, str | None]]]

_MAX_LIST_SAMPLE = 5
_result_wrapped_cache: dict[str, bool] = {}


def _trim(value: Any) -> Any:
    """Recursively bounds any list to a {count, sample} shape once it
    exceeds _MAX_LIST_SAMPLE entries, so the evidence ledger never stores
    an unnecessarily large raw graph payload (spec Section 8's explicit
    example: connected_via_device can run into the hundreds). Only used
    when writing to the ledger -- the recommendation rule itself always
    evaluates the full, untrimmed evidence.
    """
    if isinstance(value, list):
        if len(value) > _MAX_LIST_SAMPLE:
            return {"count": len(value), "sample": [_trim(v) for v in value[:_MAX_LIST_SAMPLE]]}
        return [_trim(v) for v in value]
    if isinstance(value, dict):
        return {k: _trim(v) for k, v in value.items()}
    return value


async def _is_result_wrapped(name: str) -> bool:
    # Mirrors src/mcp_server/validate_tools.py's schema-driven detection:
    # the installed MCP SDK wraps structured_content as {"result": ...}
    # for a `dict | None` return type, but returns the object directly
    # for `dict[str, Any]`. Detected from each tool's own declared
    # schema rather than hardcoding which of the 3 tools behaves which
    # way, so this keeps working if a tool's signature changes later.
    if name not in _result_wrapped_cache:
        tools = await mcp.list_tools()
        schema = next(t.output_schema for t in tools if t.name == name)
        _result_wrapped_cache[name] = list(schema.get("properties", {}).keys()) == ["result"]
    return _result_wrapped_cache[name]


async def _default_call_tool(name: str, arguments: dict) -> tuple[bool, Any, str | None]:
    """Real MCP dispatch. Never calls a Phase E function directly --
    everything goes through mcp.call_tool, the same path Phase F's own
    validation used. Returns (success, value, error) uniformly; never
    raises, and never lets a raw exception (which could carry connection
    details) escape unsanitized.
    """
    if name not in _ALLOWED_TOOLS:
        raise ValueError(f"tool {name!r} is not in this workflow's allowed tool set")
    try:
        result = await mcp.call_tool(name, arguments)
    except Exception as e:
        return False, None, scrub_secret(str(e))
    if result.is_error:
        text_parts = [getattr(c, "text", "") for c in (result.content or [])]
        message = " ".join(p for p in text_parts if p) or f"{name} returned an error"
        return False, None, scrub_secret(message)
    sc = result.structured_content
    if await _is_result_wrapped(name):
        value = sc.get("result") if isinstance(sc, dict) else sc
    else:
        value = sc
    return True, value, None


def _ledger_entry(order: int, tool: str, arguments: dict, success: bool, result: Any, error: str | None) -> dict:
    entry = {"order": order, "tool": tool, "arguments": arguments, "success": success}
    if success:
        entry["curated_result_summary"] = _trim(result)
    else:
        entry["error"] = error
    return entry


def _has_confirmed_fraud(cases: list) -> bool:
    return any(c.get("outcome") == "confirmed_fraud" for c in cases)


def _assess(evidence: dict, round2: list[tuple[str, int, dict | None]]) -> tuple[str, dict]:
    """The V1 rule from spec Section 6a (tightened), applied exactly.
    round2 is a list of (card_key, tool_call_order, result_or_None) for
    the follow-up calls that succeeded (failed ones are excluded here --
    they're already recorded as failures in the ledger and can't supply
    evidence either way).
    """
    historical = evidence.get("historical_cases") or {"direct_cases": [], "connected_cases": []}
    direct = historical.get("direct_cases") or []
    connected = historical.get("connected_cases") or []

    reasons: list[str] = []
    orders: list[int] = []

    if _has_confirmed_fraud(direct):
        reasons.append("direct_confirmed_fraud")
    if _has_confirmed_fraud(connected):
        reasons.append("connected_confirmed_fraud")
    # direct/connected reasons always stem from the single combined_evidence
    # call; its real order (1 or 2, depending on trigger type) is filled in
    # by the caller below, since _assess doesn't know that order itself.

    for card_key, order, result in round2:
        if result is None:
            continue
        sibling_direct = result.get("direct_cases") or []
        sibling_connected = result.get("connected_cases") or []
        if _has_confirmed_fraud(sibling_direct) or _has_confirmed_fraud(sibling_connected):
            reasons.append(f"sibling_confirmed_fraud:{card_key}")
            orders.append(order)

    if reasons:
        return "escalate", {"reasons": reasons, "tool_call_orders": sorted(set(orders))}

    device = evidence.get("device_evidence")
    if device is not None and device.get("hub_flag") is False:
        return "monitor", {"reasons": ["narrow_device_evidence"], "tool_call_orders": []}

    return "insufficient_evidence", {"reasons": [], "tool_call_orders": []}


def _finalize(
    investigation_id: str,
    trigger: dict,
    tool_calls: list,
    uncertainties: list,
    recommendation: str,
    basis: dict,
    pattern_evidence: dict | None = None,
    full_evidence: dict | None = None,
) -> dict:
    return {
        "investigation_id": investigation_id,
        "trigger": trigger,
        "tool_calls": tool_calls,
        "uncertainties": uncertainties,
        "recommendation": recommendation,
        "recommendation_basis": basis,
        "pattern_evidence": pattern_evidence,
        "full_evidence": full_evidence,
    }


async def investigate(trigger: dict, *, window_hours: float = 24.0, call_tool: CallTool | None = None) -> dict:
    """Runs one bounded investigation and returns the evidence ledger
    (spec Section 8). `trigger` is {"type": "flagged_txn_id", "value": int}
    or {"type": "case_id", "value": str} -- the two forms spec Section 2
    defines. `call_tool` defaults to real MCP dispatch; tests inject a
    fake to avoid depending on a live TigerGraph connection.
    """
    if trigger.get("type") not in ("flagged_txn_id", "case_id"):
        raise ValueError(f"unsupported trigger type: {trigger.get('type')!r}")

    call_tool = call_tool or _default_call_tool
    investigation_id = str(uuid.uuid4())
    tool_calls: list[dict] = []
    uncertainties: list[str] = []
    order = 1

    # --- Round 1: resolve trigger (if needed), then combined_evidence exactly once ---
    if trigger["type"] == "case_id":
        ok, resolved, err = await call_tool("benchmark_case_resolution", {"case_id": trigger["value"]})
        tool_calls.append(_ledger_entry(order, "benchmark_case_resolution", {"case_id": trigger["value"]}, ok, resolved, err))
        order += 1
        if not ok:
            uncertainties.append(f"trigger resolution failed: {err}")
            return _finalize(investigation_id, trigger, tool_calls, uncertainties, "insufficient_evidence", {"reasons": ["trigger_resolution_failed"], "tool_call_orders": [1]})
        if resolved is None:
            uncertainties.append("case_id did not resolve to a known investigation case")
            return _finalize(investigation_id, trigger, tool_calls, uncertainties, "insufficient_evidence", {"reasons": ["case_not_found"], "tool_call_orders": [1]})
        flagged_txn_id = resolved["flagged_txn_id"]
    else:
        flagged_txn_id = trigger["value"]

    combined_order = order
    ok, evidence, err = await call_tool("combined_evidence", {"flagged_txn_id": flagged_txn_id, "window_hours": window_hours})
    tool_calls.append(_ledger_entry(combined_order, "combined_evidence", {"flagged_txn_id": flagged_txn_id, "window_hours": window_hours}, ok, evidence, err))
    order += 1
    if not ok:
        uncertainties.append(f"combined_evidence failed: {err}")
        return _finalize(investigation_id, trigger, tool_calls, uncertainties, "insufficient_evidence", {"reasons": ["base_evidence_unavailable"], "tool_call_orders": [combined_order]})
    if evidence is None:
        uncertainties.append("flagged_txn_id did not resolve to a known transaction")
        return _finalize(investigation_id, trigger, tool_calls, uncertainties, "insufficient_evidence", {"reasons": ["transaction_not_found"], "tool_call_orders": [combined_order]})

    # --- Q9 gate: cross_card_fraud_verification (G4 Option B) ---
    # Uses only evidence combined_evidence already returned -- no new query
    # is needed to compute this gate. Costs exactly one call slot when it
    # fires, and reduces Round 2's effective cap by exactly one in that
    # same case, so the total never exceeds the frozen 5-call ceiling
    # (spec Section 4) regardless of trigger type, gate outcome, or ring
    # size:
    #   case_id  + gate True  + ring: 1 + 1 + 1 + 2 = 5
    #   case_id  + gate False + ring: 1 + 1 + 0 + 3 = 5
    #   flagged_txn_id + gate True  + ring: 1 + 1 + 2 = 4
    #   flagged_txn_id + gate False + ring: 1 + 0 + 3 = 4
    # MAX_FOLLOW_UP_CARDS itself is unchanged; only this investigation's
    # *effective* cap varies.
    device_evidence = evidence.get("device_evidence")
    billing_region = evidence.get("billing_region")
    q9_gate = evidence.get("card") is not None and (
        (device_evidence is not None and device_evidence.get("hub_flag") is False)
        or (billing_region is not None and billing_region.get("hub_flag") is False)
    )

    effective_follow_up_cap = MAX_FOLLOW_UP_CARDS
    cross_card = None  # stays None unless the Q9 gate fires below; carried into full_evidence either way
    if q9_gate:
        card_key_for_q9 = evidence["card"]["card_key"]
        q9_order = order
        ok, cross_card, err = await call_tool("cross_card_fraud_verification", {"card_key": card_key_for_q9})
        tool_calls.append(_ledger_entry(q9_order, "cross_card_fraud_verification", {"card_key": card_key_for_q9}, ok, cross_card, err))
        order += 1
        if not ok:
            uncertainties.append(f"cross_card_fraud_verification failed: {err}")
        # Round 2's cap shrinks regardless of whether this call succeeded --
        # the slot was already spent either way.
        effective_follow_up_cap = MAX_FOLLOW_UP_CARDS - 1

    # --- Follow-up decision + Round 2 (spec Section 5's deterministic selection) ---
    connected_via_case = ((evidence.get("connected_cards") or {}).get("connected_via_case") or [])
    round2: list[tuple[str, int, dict | None]] = []
    if connected_via_case:
        selected = sorted(connected_via_case, key=lambda c: c["connected_card_key"])[:effective_follow_up_cap]
        for c in selected:
            card_key = c["connected_card_key"]
            ok, r2, err = await call_tool("historical_case_evidence", {"card_key": card_key})
            tool_calls.append(_ledger_entry(order, "historical_case_evidence", {"card_key": card_key}, ok, r2, err))
            if ok:
                round2.append((card_key, order, r2))
            else:
                uncertainties.append(f"historical_case_evidence failed for {card_key}: {err}")
                round2.append((card_key, order, None))
            order += 1

    # Hard-limit self-check (spec Section 4): defense-in-depth, not the
    # only enforcement -- the [:effective_follow_up_cap] slice above already
    # makes exceeding this structurally impossible. Checked against the
    # effective cap (2 or 3, per the Q9 gate above), which is always
    # <= MAX_FOLLOW_UP_CARDS -- MAX_FOLLOW_UP_CARDS itself is unchanged.
    followups_made = sum(1 for tc in tool_calls if tc["tool"] == "historical_case_evidence")
    assert followups_made <= effective_follow_up_cap, "Round 2 exceeded the effective follow-up-card limit"

    # --- Reassess (no further tool calls triggered by this, ever -- no Round 3) ---
    recommendation, basis = _assess(evidence, round2)
    if combined_order not in basis["tool_call_orders"] and recommendation in ("escalate", "monitor"):
        basis["tool_call_orders"] = sorted(set(basis["tool_call_orders"]) | {combined_order})

    # --- R5 card-testing evidence (G5-B) ---
    # Pure local Python analysis over temporal_activity that combined_evidence
    # already returned -- zero MCP/tool calls, no effect on the 5-call
    # ceiling. Computed AFTER _assess() has already run and returned, so
    # this evidence structurally cannot influence G1's own
    # escalate/monitor/insufficient_evidence tier above -- it is separate
    # evidence for G2's policy layer to consume, not part of G1's own
    # recommendation. Not recorded as a tool_calls entry, since no tool
    # was called.
    pattern_evidence = classify_card_testing(evidence.get("temporal_activity") or [], flagged_txn_id)

    # --- full_evidence (G7 reasoning-layer support) ---
    # Additive only: tool_calls entries above already went through _trim()
    # for ledger compactness, which collapses any list past
    # _MAX_LIST_SAMPLE (5) into {count, sample} -- fine for an audit log,
    # not enough for a reasoning layer to ground transaction-ID citations
    # against (HHG-011 alone has >20 distinct online transactions in one
    # window). This carries the same combined_evidence/cross_card_fraud_
    # verification results already computed above, untouched by _trim.
    # No existing key changes; nothing here triggers another tool call.
    full_evidence = {"combined_evidence": evidence, "cross_card_fraud_verification": cross_card}

    return _finalize(investigation_id, trigger, tool_calls, uncertainties, recommendation, basis, pattern_evidence, full_evidence)
