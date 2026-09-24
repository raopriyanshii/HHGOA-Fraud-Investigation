"""
Q10 unit tests for src.investigation.queries.q10_prior_investigation_memory.

Uses a fake `conn` whose runInterpretedQuery(gsql, params) reproduces the
Q10 GSQL query's own filter (cc.case_id != input_case_id AND
cc.written_to_graph == TRUE) over an in-memory fixture of
HHG_InvestigationCase-shaped vertices already known to be connected to
the given card -- zero live TigerGraph dependency, applying this
project's established fake-connection convention (test_case_memory.py's
_FakeConn, for the write path) to a read query instead. The fake stands
in for the HHG_ON_CARD_reverse traversal itself (a real graph edge, not
reproducible without a live engine); separate structural tests below
assert the actual GSQL text performs that traversal and those filters.
"""
from src.investigation import queries as q


def _case_vertex(case_id, written_to_graph=True, **overrides):
    attrs = {
        "case_id": case_id,
        "verdict": "fraud",
        "fraud_probability": 0.8,
        "pattern": "card_testing",
        "exposure_usd": 100.0,
        "summary": "a summary",
        "updated_at": "2016-01-01 00:00:00",
        "written_to_graph": written_to_graph,
    }
    attrs.update(overrides)
    return {"v_id": case_id, "v_type": "HHG_InvestigationCase", "attributes": attrs}


class _FakeConn:
    def __init__(self, cases_on_card: list[dict]):
        self._cases_on_card = cases_on_card
        self.calls = []

    def runInterpretedQuery(self, gsql, params):
        self.calls.append({"gsql": gsql, "params": params})
        filtered = [
            c for c in self._cases_on_card
            if c["attributes"]["case_id"] != params["input_case_id"]
            and c["attributes"]["written_to_graph"] is True
        ]
        return [{"PriorCases": filtered}]


# --- A. written prior case is returned ---

def test_written_prior_case_is_returned():
    conn = _FakeConn([_case_vertex("HHG-002", written_to_graph=True)])
    result = q.q10_prior_investigation_memory(conn, "C00001:1", "HHG-001")
    assert len(result) == 1
    assert result[0]["case_id"] == "HHG-002"
    assert result[0]["verdict"] == "fraud"
    assert result[0]["written_to_graph"] is True


# --- B. unwritten/seed case is filtered out ---

def test_unwritten_seed_case_is_filtered_out():
    conn = _FakeConn([_case_vertex("HHG-003", written_to_graph=False, verdict="", fraud_probability=-1)])
    result = q.q10_prior_investigation_memory(conn, "C00001:1", "HHG-001")
    assert result == []


# --- C. same case_id is excluded ---

def test_same_case_id_is_excluded():
    conn = _FakeConn([_case_vertex("HHG-001", written_to_graph=True)])
    result = q.q10_prior_investigation_memory(conn, "C00001:1", "HHG-001")
    assert result == []


# --- D. no prior case returns [] ---

def test_no_prior_case_returns_empty_list():
    conn = _FakeConn([])
    result = q.q10_prior_investigation_memory(conn, "C00001:1", "HHG-001")
    assert result == []


# --- E. multiple prior cases are all returned ---

def test_multiple_prior_cases_are_all_returned():
    conn = _FakeConn([
        _case_vertex("HHG-002", written_to_graph=True),
        _case_vertex("HHG-004", written_to_graph=True),
    ])
    result = q.q10_prior_investigation_memory(conn, "C00001:1", "HHG-001")
    assert {r["case_id"] for r in result} == {"HHG-002", "HHG-004"}


def test_mixed_written_and_unwritten_and_self_only_returns_the_valid_one():
    conn = _FakeConn([
        _case_vertex("HHG-001", written_to_graph=True),   # excluded: self
        _case_vertex("HHG-005", written_to_graph=False),  # excluded: never investigated
        _case_vertex("HHG-002", written_to_graph=True),   # kept
    ])
    result = q.q10_prior_investigation_memory(conn, "C00001:1", "HHG-001")
    assert [r["case_id"] for r in result] == ["HHG-002"]


# --- F/G. query structure (does not depend on a live engine) ---

def test_query_does_not_reference_closed_case():
    assert "HHG_ClosedCase" not in q._Q10_GSQL


def test_query_uses_on_card_reverse_traversal():
    assert "HHG_ON_CARD_reverse" in q._Q10_GSQL


def test_query_never_references_involves_or_cites_edges():
    # Neither HHG_INVOLVES nor HHG_CITES has HHG_Card as a valid FROM/TO
    # endpoint (confirmed against gsql/02_edges.gsql) -- Q10 must not
    # reference either.
    assert "HHG_INVOLVES" not in q._Q10_GSQL
    assert "HHG_CITES" not in q._Q10_GSQL


def test_query_filters_on_case_id_and_written_to_graph():
    assert "cc.case_id != input_case_id" in q._Q10_GSQL
    assert "cc.written_to_graph == TRUE" in q._Q10_GSQL


def test_query_input_parameters_match_spec():
    assert "VERTEX<HHG_Card> input_card" in q._Q10_GSQL
    assert "STRING input_case_id" in q._Q10_GSQL


# --- output field contract ---

def test_only_allowed_fields_are_returned():
    conn = _FakeConn([_case_vertex(
        "HHG-002", written_to_graph=True,
        evidence_json="[]", next_best_actions_json="{}", sar_json="", evidence_requests_json="[]",
        risk_score=0.5, trigger_type="risk_score", trigger_text="irrelevant",
    )])
    result = q.q10_prior_investigation_memory(conn, "C00001:1", "HHG-001")
    assert set(result[0].keys()) == {
        "case_id", "verdict", "fraud_probability", "pattern",
        "exposure_usd", "summary", "updated_at", "written_to_graph",
    }


def test_params_threaded_correctly_to_query():
    conn = _FakeConn([])
    q.q10_prior_investigation_memory(conn, "C99999:1", "HHG-020")
    assert conn.calls[0]["params"] == {"input_card": "C99999:1", "input_case_id": "HHG-020"}
