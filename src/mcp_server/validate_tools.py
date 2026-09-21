"""
Validates the MCP tool-exposure layer, not the query logic itself.

Q1-Q8 were already benchmarked and tested directly against the graph in
Phase E (docs/PHASE_E_VALIDATION.md). This script calls the same 8
functions through the MCP protocol (mcp.call_tool), using the same fixed,
real inputs Phase E already used, and checks that the MCP-wrapped result
matches what the direct call already produced. It is a regression check
on the wrapper, not a new discovery benchmark -- no new "expected values"
are invented here.

No LLM/agent involved: this script itself acts as a plain MCP client via
in-process call_tool().
"""
import asyncio
import json

from src.mcp_server.server import mcp
from src.investigation import queries as q
from src.graph.tigergraph_connection import get_connection

TXN_ID = 3514030
CARD_KEY = "C12382:21139"
CASE_ID = "HHG-001"
RING_DEVICE_PROFILE = "SM-G935F Build/NRD90M|Android 7.0|chrome 62.0 for android|1920x1080"
RING_CARD_KEY = "C03528:14407"


def _direct_conn():
    return get_connection(graphname="HHGOA_FraudInvestigation")


def _canonical(obj):
    # TigerGraph does not guarantee list order across separate query
    # executions for unordered SELECTs, so two independent live calls
    # (one direct, one via MCP) can return the same content in a
    # different order. Sort lists by their canonical JSON form at every
    # nesting level so the comparison checks content, not call-to-call
    # ordering -- Phase E's own tests never claimed an order guarantee
    # for these fields (only Q2's ts-sort, which is stable either way).
    if isinstance(obj, dict):
        return {k: _canonical(v) for k, v in obj.items()}
    if isinstance(obj, list):
        items = [_canonical(v) for v in obj]
        return sorted(items, key=lambda x: json.dumps(x, sort_keys=True, default=str))
    return obj


def _matches(direct, via_mcp):
    return _canonical(direct) == _canonical(via_mcp)


_result_wrapped = {}  # tool name -> bool, cached from each tool's declared output_schema


async def _is_result_wrapped(name: str) -> bool:
    # The SDK wraps structured_content as {"result": <value>} for return
    # types that aren't themselves a plain JSON object (e.g. `dict | None`),
    # but returns the object directly for a return type like dict[str, Any]
    # that already maps to an object schema. Detect this per-tool from its
    # own declared schema rather than assuming one convention.
    if name not in _result_wrapped:
        tools = await mcp.list_tools()
        schema = next(t.output_schema for t in tools if t.name == name)
        _result_wrapped[name] = list(schema.get("properties", {}).keys()) == ["result"]
    return _result_wrapped[name]


async def _call(name: str, arguments: dict):
    result = await mcp.call_tool(name, arguments)
    if result.is_error:
        raise AssertionError(f"{name} returned an MCP error: {result.content}")
    sc = result.structured_content
    if await _is_result_wrapped(name):
        return sc.get("result") if isinstance(sc, dict) else sc
    return sc


async def main():
    conn = _direct_conn()
    checks = []

    # Q1
    direct = q.q1_transaction_lookup(conn, TXN_ID)
    via_mcp = await _call("transaction_lookup", {"flagged_txn_id": TXN_ID})
    checks.append(("Q1", _matches(direct, via_mcp)))

    # Q2
    direct = q.q2_temporal_neighborhood(conn, TXN_ID, 24.0)
    via_mcp = await _call("temporal_neighborhood", {"flagged_txn_id": TXN_ID, "window_hours": 24.0})
    checks.append(("Q2", _matches(direct, via_mcp)))

    # Q3
    direct = q.q3_device_sharing(conn, RING_DEVICE_PROFILE)
    via_mcp = await _call("device_sharing", {"device_profile": RING_DEVICE_PROFILE})
    checks.append(("Q3", _matches(direct, via_mcp)))

    # Q4
    direct = q.q4_historical_case_evidence(conn, CARD_KEY)
    via_mcp = await _call("historical_case_evidence", {"card_key": CARD_KEY})
    checks.append(("Q4", _matches(direct, via_mcp)))

    # Q5
    direct = q.q5_connected_cards(conn, RING_CARD_KEY)
    via_mcp = await _call("connected_cards", {"card_key": RING_CARD_KEY})
    checks.append(("Q5", _matches(direct, via_mcp)))

    # Q6
    direct = q.q6_billing_region_evidence(conn, TXN_ID)
    via_mcp = await _call("billing_region_evidence", {"flagged_txn_id": TXN_ID})
    checks.append(("Q6", _matches(direct, via_mcp)))

    # Q7
    direct = q.q7_benchmark_case_resolution(conn, CASE_ID)
    via_mcp = await _call("benchmark_case_resolution", {"case_id": CASE_ID})
    checks.append(("Q7", _matches(direct, via_mcp)))

    # Q8
    direct = q.q8_combined_evidence(conn, TXN_ID, 24.0)
    via_mcp = await _call("combined_evidence", {"flagged_txn_id": TXN_ID, "window_hours": 24.0})
    checks.append(("Q8", _matches(direct, via_mcp)))

    print(json.dumps({name: passed for name, passed in checks}, indent=2))
    all_pass = all(passed for _, passed in checks)
    print("ALL PASS" if all_pass else "SOME FAILED")
    return all_pass


if __name__ == "__main__":
    ok = asyncio.run(main())
    raise SystemExit(0 if ok else 1)
