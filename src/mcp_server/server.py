"""
MCP server exposing the Phase E deterministic investigation queries
(Q1-Q8) as MCP tools. Every tool is a thin wrapper around the existing,
already-validated functions in src/investigation/queries.py -- no graph
logic is reimplemented here, and no LLM/agent/policy code lives in this
module.
"""
from typing import Any

from mcp.server.mcpserver import MCPServer

from src.graph.tigergraph_connection import get_connection, scrub_secret
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


if __name__ == "__main__":
    mcp.run()
