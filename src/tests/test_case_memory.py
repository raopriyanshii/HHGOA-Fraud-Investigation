"""
G10 unit tests for src/investigation/case_memory.py (the query layer).

Every test uses a fake `conn` object with controllable
upsertVertex/upsertEdges behavior -- zero live TigerGraph dependency.
These test the exact mutation sequence, exact fields written, failure
handling, and the monotonic/no-delete guarantee.
"""
import pytest

from src.investigation import case_memory as cm


class _FakeConn:
    """Records every upsertVertex/upsertEdges call. Deliberately has NO
    delete-shaped method at all (no deleteVertex/deleteEdge/delVertex) --
    if case_memory.py ever tried to call one, it would raise
    AttributeError, proving the monotonic/no-delete guarantee structurally,
    not just by inspection.
    """

    def __init__(self, vertex_exc=None, involves_exc=None, cites_exc=None, finalize_exc=None):
        self.vertex_calls = []
        self.edge_calls = []
        self._vertex_exc = vertex_exc
        self._involves_exc = involves_exc
        self._cites_exc = cites_exc
        self._finalize_exc = finalize_exc
        self._vertex_call_count = 0

    def upsertVertex(self, vertexType, vertexId, attributes=None):
        self._vertex_call_count += 1
        self.vertex_calls.append({"vertexType": vertexType, "vertexId": vertexId, "attributes": dict(attributes or {})})
        is_finalize_call = attributes == {"written_to_graph": True}
        if is_finalize_call and self._finalize_exc:
            raise self._finalize_exc
        if not is_finalize_call and self._vertex_exc:
            raise self._vertex_exc
        return 1

    def upsertEdges(self, sourceVertexType, edgeType, targetVertexType, edges, vertexMustExist=False, atomic=False):
        self.edge_calls.append({
            "sourceVertexType": sourceVertexType, "edgeType": edgeType, "targetVertexType": targetVertexType,
            "edges": list(edges), "vertexMustExist": vertexMustExist,
        })
        if edgeType == cm.INVOLVES_EDGE and self._involves_exc:
            raise self._involves_exc
        if edgeType == cm.CITES_EDGE and self._cites_exc:
            raise self._cites_exc
        return len(edges)


def _payload(**overrides):
    payload = {
        "case_id": "HHG-999",
        "verdict": "fraud",
        "fraud_probability": 0.8,
        "pattern": "card_testing",
        "pattern_description": "",
        "exposure_usd": 123.45,
        "summary": "test summary",
        "stop_reason": "test stop reason",
        "evidence": [{"claim": "x", "source": "graph", "ref": "y", "entity_ids": []}],
        "next_best_actions": {"initial": [], "final": [], "what_changed": "nothing"},
        "affected_txn_ids": [1001, 1002],
        "first_suspicious_txn_id": 1001,
        "similar_prior_cases": ["CC-1"],
        "updated_at": "2016-01-01 00:00:00",
    }
    payload.update(overrides)
    return payload


# --- 1. successful complete write ---

def test_successful_complete_write():
    conn = _FakeConn()
    result = cm.write_investigation_case(conn, _payload())
    assert result["written"] is True
    assert result["graph_case_id"] == "HHG-999"
    assert all(step["ok"] for step in result["steps"].values())


# --- 2/3. same case written twice, attributes updated on rerun ---

def test_same_case_written_twice_updates_same_vertex_id():
    conn = _FakeConn()
    cm.write_investigation_case(conn, _payload(verdict="uncertain"))
    cm.write_investigation_case(conn, _payload(verdict="fraud"))
    vertex_ids = {c["vertexId"] for c in conn.vertex_calls}
    assert vertex_ids == {"HHG-999"}  # always the same vertex id -- never a second/different one
    content_calls = [c for c in conn.vertex_calls if "verdict" in c["attributes"]]
    assert content_calls[0]["attributes"]["verdict"] == "uncertain"
    assert content_calls[1]["attributes"]["verdict"] == "fraud"  # rerun's new value is what gets sent


# --- 4. INVOLVES edges ---

def test_involves_edges_exact_shape():
    conn = _FakeConn()
    cm.write_investigation_case(conn, _payload(affected_txn_ids=[2001, 2002, 2003]))
    involves_calls = [c for c in conn.edge_calls if c["edgeType"] == cm.INVOLVES_EDGE]
    assert len(involves_calls) == 1
    call = involves_calls[0]
    assert call["sourceVertexType"] == "HHG_InvestigationCase"
    assert call["targetVertexType"] == "HHG_Transaction"
    assert call["vertexMustExist"] is True
    assert call["edges"] == [("HHG-999", 2001, {}), ("HHG-999", 2002, {}), ("HHG-999", 2003, {})]


def test_no_involves_edge_call_when_affected_txn_ids_empty():
    conn = _FakeConn()
    cm.write_investigation_case(conn, _payload(affected_txn_ids=[]))
    involves_calls = [c for c in conn.edge_calls if c["edgeType"] == cm.INVOLVES_EDGE]
    assert involves_calls == []


# --- 5. CITES edges ---

def test_cites_edges_exact_shape():
    conn = _FakeConn()
    cm.write_investigation_case(conn, _payload(similar_prior_cases=["CC-1", "CC-2"]))
    cites_calls = [c for c in conn.edge_calls if c["edgeType"] == cm.CITES_EDGE]
    assert len(cites_calls) == 1
    call = cites_calls[0]
    assert call["sourceVertexType"] == "HHG_InvestigationCase"
    assert call["targetVertexType"] == "HHG_ClosedCase"
    assert call["vertexMustExist"] is True
    assert call["edges"] == [("HHG-999", "CC-1", {}), ("HHG-999", "CC-2", {})]


# --- 6. monotonic edge behavior (no delete, ever) ---

def test_no_delete_method_is_ever_called_structurally():
    # _FakeConn has no delete-shaped method at all; a successful run
    # proves case_memory.py never even attempts one.
    conn = _FakeConn()
    result = cm.write_investigation_case(conn, _payload())
    assert result["written"] is True
    assert not hasattr(conn, "deleteVertex")
    assert not hasattr(conn, "deleteEdge")


def test_module_source_contains_no_delete_calls():
    import inspect
    source = inspect.getsource(cm)
    for forbidden in ("delVertex", "deleteVertex", "delEdge", "deleteEdge", "DELETE "):
        assert forbidden not in source


# --- 11. TigerGraph vertex failure ---

def test_vertex_upsert_failure_reports_unsuccessful():
    conn = _FakeConn(vertex_exc=RuntimeError("connection reset"))
    result = cm.write_investigation_case(conn, _payload())
    assert result["written"] is False
    assert result["graph_case_id"] == ""
    assert result["steps"]["vertex"]["ok"] is False
    # edges must never be attempted if the vertex step itself failed:
    assert conn.edge_calls == []


# --- 12. TigerGraph edge failure (both paths) ---

def test_involves_edge_failure_reports_unsuccessful():
    conn = _FakeConn(involves_exc=RuntimeError("edge upsert failed"))
    result = cm.write_investigation_case(conn, _payload())
    assert result["written"] is False
    assert result["graph_case_id"] == ""
    assert result["steps"]["involves_edges"]["ok"] is False
    assert "cites_edges" not in result["steps"]  # never reached


def test_cites_edge_failure_reports_unsuccessful():
    conn = _FakeConn(cites_exc=RuntimeError("edge upsert failed"))
    result = cm.write_investigation_case(conn, _payload())
    assert result["written"] is False
    assert result["graph_case_id"] == ""
    assert result["steps"]["cites_edges"]["ok"] is False


# --- 13. partial-write reporting ---

def test_partial_write_vertex_and_involves_succeed_cites_fails():
    conn = _FakeConn(cites_exc=RuntimeError("boom"))
    result = cm.write_investigation_case(conn, _payload(similar_prior_cases=["CC-1"]))
    assert result["written"] is False  # never reported as successful
    assert result["steps"]["vertex"]["ok"] is True
    assert result["steps"]["involves_edges"]["ok"] is True
    assert result["steps"]["cites_edges"]["ok"] is False
    # the final written_to_graph=True flip must never have been sent:
    finalize_calls = [c for c in conn.vertex_calls if c["attributes"] == {"written_to_graph": True}]
    assert finalize_calls == []


# --- 14. secret scrubbing ---

def test_vertex_failure_error_is_scrubbed(monkeypatch):
    fake_key = "AIzaSyDaGmWKa4JsXZ-HjGw7ISLn_3namBGewQe"
    monkeypatch.setenv("GEMINI_API_KEY", fake_key)
    conn = _FakeConn(vertex_exc=RuntimeError(f"401 Unauthorized: key {fake_key} rejected"))
    result = cm.write_investigation_case(conn, _payload())
    assert fake_key not in result["steps"]["vertex"]["error"]
    assert "[REDACTED]" in result["steps"]["vertex"]["error"]


# --- 15. exact fields written ---

def test_exact_vertex_attributes_written():
    conn = _FakeConn()
    cm.write_investigation_case(conn, _payload())
    content_call = conn.vertex_calls[0]
    assert content_call["attributes"] == {
        "status": "INVESTIGATED",
        "verdict": "fraud",
        "fraud_probability": 0.8,
        "pattern": "card_testing",
        "pattern_description": "",
        "exposure_usd": 123.45,
        "summary": "test summary",
        "stop_reason": "test stop reason",
        "evidence_json": '[{"claim": "x", "source": "graph", "ref": "y", "entity_ids": []}]',
        "next_best_actions_json": '{"initial": [], "final": [], "what_changed": "nothing"}',
        "first_suspicious_txn_id": 1001,
        "written_to_graph": False,
        "updated_at": "2016-01-01 00:00:00",
    }


# --- 17. graph_case_id empty on any unsuccessful write ---

def test_graph_case_id_empty_on_every_failure_mode():
    for conn in (
        _FakeConn(vertex_exc=RuntimeError("x")),
        _FakeConn(involves_exc=RuntimeError("x")),
        _FakeConn(cites_exc=RuntimeError("x")),
        _FakeConn(finalize_exc=RuntimeError("x")),
    ):
        result = cm.write_investigation_case(conn, _payload())
        assert result["graph_case_id"] == ""
        assert result["written"] is False


# --- 18. written_to_graph=True only after the full sequence, in order ---

def test_finalize_flip_happens_last_and_only_on_full_success():
    conn = _FakeConn()
    result = cm.write_investigation_case(conn, _payload())
    assert result["written"] is True
    finalize_calls = [c for c in conn.vertex_calls if c["attributes"] == {"written_to_graph": True}]
    assert len(finalize_calls) == 1
    # it must be the LAST vertex call:
    assert conn.vertex_calls[-1] == finalize_calls[0]


def test_finalize_failure_still_reports_unsuccessful_even_though_content_and_edges_succeeded():
    conn = _FakeConn(finalize_exc=RuntimeError("network blip on the last call"))
    result = cm.write_investigation_case(conn, _payload())
    assert result["written"] is False
    assert result["graph_case_id"] == ""
    assert result["steps"]["vertex"]["ok"] is True
    assert result["steps"]["involves_edges"]["ok"] is True
    assert result["steps"]["cites_edges"]["ok"] is True
    assert result["steps"]["finalize"]["ok"] is False


# --- 19. idempotent call pattern (true vertex-count invariance is a
# TigerGraph primary-key upsert guarantee, verified separately live) ---

def test_rerun_always_targets_the_same_primary_id_never_a_new_one():
    conn = _FakeConn()
    for _ in range(3):
        cm.write_investigation_case(conn, _payload())
    assert {c["vertexId"] for c in conn.vertex_calls} == {"HHG-999"}


# --- 20. Transaction_Fraud is never referenced ---

def test_module_never_references_transaction_fraud():
    import inspect
    source = inspect.getsource(cm)
    assert "Transaction_Fraud" not in source


def test_only_hhg_prefixed_types_are_ever_used():
    conn = _FakeConn()
    cm.write_investigation_case(conn, _payload())
    for call in conn.vertex_calls:
        assert call["vertexType"] == "HHG_InvestigationCase"
    for call in conn.edge_calls:
        assert call["sourceVertexType"].startswith("HHG_")
        assert call["targetVertexType"].startswith("HHG_")
