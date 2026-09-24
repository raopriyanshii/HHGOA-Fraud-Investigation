"""
MCP server exposing the deterministic investigation queries (Q1-Q9) as
read-only MCP tools, plus exactly one write tool (G10, write_case_memory).
Every tool is a thin wrapper around an existing, already-validated
function in src/investigation/queries.py (reads) or
src/investigation/case_memory.py (the one write) -- no graph logic is
reimplemented here, and no LLM/agent/policy code lives in this module.
write_case_memory is a single fixed operation, never generic TigerGraph
mutation and never arbitrary GSQL; no LLM ever calls it directly (see
src/agent/case_writer.py, which is the only caller this project builds).
"""
from typing import Any

from mcp.server.mcpserver import MCPServer

from src.graph.tigergraph_connection import get_connection, scrub_secret
from src.investigation import case_memory as cm
from src.investigation import queries as q

mcp = MCPServer(name="fraud-investigation")

_conn = None


def _get_conn():
    global _conn
    if _conn is None:
        _conn = get_connection(graphname="HHGOA_FraudInvestigation")
    return _conn


def _safe_call(fn, *args, **kwargs):
    try:
        return fn(_get_conn(), *args, **kwargs)
    except Exception as e:
        raise RuntimeError(scrub_secret(str(e)))


@mcp.tool()
def transaction_lookup(flagged_txn_id: int) -> dict | None:
    """Q1: resolve a flagged transaction to its card, customer, device, and billing region."""
    return _safe_call(q.q1_transaction_lookup, flagged_txn_id)


@mcp.tool()
def temporal_neighborhood(flagged_txn_id: int, window_hours: float = 24.0) -> dict | None:
    """Q2: same-card transaction activity within window_hours of the flagged transaction."""
    return _safe_call(q.q2_temporal_neighborhood, flagged_txn_id, window_hours)


@mcp.tool()
def device_sharing(device_profile: str) -> dict | None:
    """Q3: cards and customers that share the given device profile."""
    return _safe_call(q.q3_device_sharing, device_profile)


@mcp.tool()
def historical_case_evidence(card_key: str) -> dict[str, Any]:
    """Q4: closed-case precedent for a card, direct and connected."""
    return _safe_call(q.q4_historical_case_evidence, card_key)


@mcp.tool()
def connected_cards(card_key: str) -> dict[str, Any]:
    """Q5: ring-mate cards via shared closed case and via shared non-hub device."""
    return _safe_call(q.q5_connected_cards, card_key)


@mcp.tool()
def billing_region_evidence(flagged_txn_id: int) -> dict | None:
    """Q6: billing-region facts for a transaction."""
    return _safe_call(q.q6_billing_region_evidence, flagged_txn_id)


@mcp.tool()
def benchmark_case_resolution(case_id: str) -> dict | None:
    """Q7: direct InvestigationCase lookup by id."""
    return _safe_call(q.q7_benchmark_case_resolution, case_id)


@mcp.tool()
def combined_evidence(flagged_txn_id: int, window_hours: float = 24.0) -> dict | None:
    """Q8: Q1-Q6 composed into one evidence object for a flagged transaction."""
    return _safe_call(q.q8_combined_evidence, flagged_txn_id, window_hours)


@mcp.tool()
def cross_card_fraud_verification(card_key: str) -> dict[str, Any]:
    """Q9: cards sharing a non-hub device or billing region with the given card, each tagged with same_customer and its own confirmed-fraud status."""
    return _safe_call(q.q9_cross_card_fraud_verification, card_key)


@mcp.tool()
def prior_investigation_memory(card_key: str, case_id: str) -> list[dict]:
    """Q10: this system's own previously written HHG_InvestigationCase
    records for the given card (excluding case_id itself and any
    unwritten seed row). Registered for unit-testability only -- not
    called by src.agent.workflow, and does not change the 5-tool-call
    investigation budget."""
    return _safe_call(q.q10_prior_investigation_memory, card_key, case_id)


@mcp.tool()
def write_case_memory(
    case_id: str,
    verdict: str,
    fraud_probability: float,
    pattern: str,
    pattern_description: str,
    exposure_usd: float,
    summary: str,
    stop_reason: str,
    evidence: list,
    next_best_actions: dict,
    affected_txn_ids: list,
    first_suspicious_txn_id: int | None,
    similar_prior_cases: list,
    updated_at: str,
) -> dict:
    """G10: the ONLY write-capable tool this server exposes -- a single
    fixed case-memory operation, never generic TigerGraph mutation and
    never arbitrary GSQL. Upserts the named HHG_InvestigationCase
    vertex's content attributes, HHG_INVOLVES edges (affected_txn_ids),
    and HHG_CITES edges (similar_prior_cases); written_to_graph is only
    set True after all steps succeed (src.investigation.case_memory's
    own guarantee, not reimplemented here)."""
    payload = {
        "case_id": case_id,
        "verdict": verdict,
        "fraud_probability": fraud_probability,
        "pattern": pattern,
        "pattern_description": pattern_description,
        "exposure_usd": exposure_usd,
        "summary": summary,
        "stop_reason": stop_reason,
        "evidence": evidence,
        "next_best_actions": next_best_actions,
        "affected_txn_ids": affected_txn_ids,
        "first_suspicious_txn_id": first_suspicious_txn_id,
        "similar_prior_cases": similar_prior_cases,
        "updated_at": updated_at,
    }
    return _safe_call(cm.write_investigation_case, payload)


if __name__ == "__main__":
    mcp.run()
