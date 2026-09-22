"""
Phase G1 integration validation: runs the real investigate() workflow --
real MCP dispatch (src.mcp_server.server.mcp), real TigerGraph connection,
no mocking -- against a known Phase E benchmark case, and prints the
resulting evidence ledger for manual inspection.

Unlike Phase E's benchmark script, this does not check resolved identities
against a hardcoded expected-value table -- there is no separate "answer
key" here. What it does assert is a specific recommendation label,
`escalate`, for HHG-001, because that label follows deterministically from
a fact already established and audited in Phase E/F: HHG-001's card
(C12382:21139) has 4 direct historical cases, all outcome=confirmed_fraud,
and the V1 rule's branch 1 (docs/AGENT_WORKFLOW_SPEC.md Section 6a) turns
that into `escalate` unconditionally. That fact is independent of this
script, not invented here, and this assertion is only meaningful when both
(a) TigerGraph is reachable and (b) that historical data hasn't changed --
if it fails because TigerGraph is unreachable, the failure will show up as
`insufficient_evidence` with a failed tool_calls entry (a connectivity
problem), not as some other recommendation label (which would indicate an
actual logic regression). See the assertion message below for how to tell
the two apart without re-running this script repeatedly while TigerGraph
is known to be down.
"""
import asyncio
import json

from src.agent.workflow import investigate

CASE_ID = "HHG-001"  # resolves to flagged_txn_id 3514030, card C12382:21139 -- Phase E's own benchmark case


async def main():
    result = await investigate({"type": "case_id", "value": CASE_ID})
    print(json.dumps(result, indent=2, default=str))

    assert result["trigger"] == {"type": "case_id", "value": CASE_ID}
    assert 1 <= len(result["tool_calls"]) <= 5, "tool call count outside the spec's documented 1-5 range"
    followups = [tc for tc in result["tool_calls"] if tc["tool"] == "historical_case_evidence"]
    assert len(followups) <= 3, "exceeded the hard follow-up-card limit"
    assert result["recommendation"] == "escalate", (
        f"expected 'escalate' for {CASE_ID} (its card has 4 confirmed_fraud "
        f"direct historical cases, established in Phase E/F audits) but got "
        f"{result['recommendation']!r} instead. Before treating this as a "
        f"logic regression: check result['tool_calls'] for any entry with "
        f"success=False first -- if TigerGraph is unreachable, this fails "
        f"via 'insufficient_evidence' from a failed tool call, which is a "
        f"connectivity problem, not evidence that the recommendation rule "
        f"itself is wrong."
    )

    print("\nStructural checks passed.")
    print(f"Recommendation: {result['recommendation']}")
    print(f"Total tool calls: {len(result['tool_calls'])}")
    return result


if __name__ == "__main__":
    asyncio.run(main())
