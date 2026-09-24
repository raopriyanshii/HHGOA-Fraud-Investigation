"""
G10 -- the ONLY write-capable TigerGraph function in this project.

Deliberately kept out of src/investigation/queries.py, whose own module
docstring states "Every function here is READ-ONLY and returns plain,
JSON-serializable Python structures" -- that promise stays literally true
by keeping this write path in a separate file, not by bending it.

Uses pyTigerGraph 2.0.4's native upsertVertex/upsertEdges REST++ methods
-- their exact signatures were inspected directly (inspect.signature)
against the actually-installed package before this file was written, not
guessed:
  upsertVertex(vertexType, vertexId, attributes=None) -> int
  upsertEdges(sourceVertexType, edgeType, targetVertexType, edges, vertexMustExist=False, atomic=False) -> int
    where edges is a list of (source_id, target_id, {attrs}) tuples.
Never a hand-rolled GSQL UPDATE, never INSTALL QUERY, never any other
graph type touched.

Sequential, non-atomic by construction: TigerGraph gives no cross-call
transaction across independent upsert calls, and this module does not
pretend otherwise. Order: (1) vertex content attributes, with
written_to_graph explicitly False; (2) HHG_INVOLVES edges
(affected_txn_ids); (3) HHG_CITES edges (similar_prior_cases); (4) only
if 1-3 all succeed, one final, minimal vertex upsert flips
written_to_graph to True. Any failure at any step stops immediately,
leaves written_to_graph False (both in the returned result and, for
steps already applied, on the graph itself), and reports exactly which
step failed -- there is no code path that reports `written: True` after
a partial sequence.

vertexMustExist=True on both edge upserts: an id that doesn't already
exist in the graph must never silently create a phantom stub vertex.
payload's ids are expected to already be grounded by
reasoning_validator.py before this function is ever called, so this is a
defensive guard against a bug ever fabricating graph data, not an
expected path.
"""
from __future__ import annotations

import json

from src.graph.tigergraph_connection import scrub_secret

INVESTIGATION_CASE_TYPE = "HHG_InvestigationCase"
TRANSACTION_TYPE = "HHG_Transaction"
CLOSED_CASE_TYPE = "HHG_ClosedCase"
INVOLVES_EDGE = "HHG_INVOLVES"
CITES_EDGE = "HHG_CITES"

# G10 decision (locked): a minimal, explicit graph-lifecycle value,
# deliberately kept separate from the organizer's own answer-file
# case.status enum (open/closed_fraud/closed_legitimate/escalated) --
# see docs/AGENT_WORKFLOW_SPEC.md Section 16. No larger state machine.
LIFECYCLE_INVESTIGATED = "INVESTIGATED"


def write_investigation_case(conn, payload: dict) -> dict:
    """`payload` is assumed already fully validated by the caller
    (src.agent.case_writer.build_case_memory_payload +
    _validate_payload) -- this function only performs the graph
    mutation itself, in the fixed step order documented above.

    Returns:
      {"written": bool, "graph_case_id": str, "steps": {<step>: {"ok": bool, ...}}}
    `written` is True only when every step succeeded; `graph_case_id` is
    the case_id when written, "" otherwise -- this function never
    returns a non-empty graph_case_id for anything less than a fully
    confirmed write.
    """
    case_id = payload["case_id"]
    steps: dict = {}

    content_attrs = {
        "status": LIFECYCLE_INVESTIGATED,
        "verdict": payload["verdict"],
        "fraud_probability": payload["fraud_probability"],
        "pattern": payload["pattern"],
        "pattern_description": payload["pattern_description"],
        "exposure_usd": payload["exposure_usd"],
        "summary": payload["summary"],
        "stop_reason": payload["stop_reason"],
        "evidence_json": json.dumps(payload["evidence"], default=str),
        "next_best_actions_json": json.dumps(payload["next_best_actions"], default=str),
        "first_suspicious_txn_id": payload["first_suspicious_txn_id"] or 0,
        "written_to_graph": False,  # not yet confirmed complete -- flipped True only at the end
        "updated_at": payload["updated_at"],
    }

    try:
        accepted = conn.upsertVertex(INVESTIGATION_CASE_TYPE, case_id, content_attrs)
        steps["vertex"] = {"ok": True, "accepted": accepted}
    except Exception as e:
        steps["vertex"] = {"ok": False, "error": scrub_secret(str(e))}
        return {"written": False, "graph_case_id": "", "steps": steps}

    affected_txn_ids = payload["affected_txn_ids"]
    try:
        accepted = 0
        if affected_txn_ids:
            accepted = conn.upsertEdges(
                INVESTIGATION_CASE_TYPE, INVOLVES_EDGE, TRANSACTION_TYPE,
                [(case_id, txn_id, {}) for txn_id in affected_txn_ids],
                vertexMustExist=True,
            )
        steps["involves_edges"] = {"ok": True, "accepted": accepted, "requested": len(affected_txn_ids)}
    except Exception as e:
        steps["involves_edges"] = {"ok": False, "error": scrub_secret(str(e))}
        return {"written": False, "graph_case_id": "", "steps": steps}

    similar_prior_cases = payload["similar_prior_cases"]
    try:
        accepted = 0
        if similar_prior_cases:
            accepted = conn.upsertEdges(
                INVESTIGATION_CASE_TYPE, CITES_EDGE, CLOSED_CASE_TYPE,
                [(case_id, cc_id, {}) for cc_id in similar_prior_cases],
                vertexMustExist=True,
            )
        steps["cites_edges"] = {"ok": True, "accepted": accepted, "requested": len(similar_prior_cases)}
    except Exception as e:
        steps["cites_edges"] = {"ok": False, "error": scrub_secret(str(e))}
        return {"written": False, "graph_case_id": "", "steps": steps}

    try:
        conn.upsertVertex(INVESTIGATION_CASE_TYPE, case_id, {"written_to_graph": True})
        steps["finalize"] = {"ok": True}
    except Exception as e:
        steps["finalize"] = {"ok": False, "error": scrub_secret(str(e))}
        return {"written": False, "graph_case_id": "", "steps": steps}

    return {"written": True, "graph_case_id": case_id, "steps": steps}
