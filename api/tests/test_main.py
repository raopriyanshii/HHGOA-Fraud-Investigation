"""
Tests for api/main.py -- the thin FastAPI wrapper around the existing
run_single_case/build_submission_case pipeline.

Every test here mocks src.benchmark.g9_runner._build_call_llm and
run_single_case, plus src.benchmark.submission_adapter.build_submission_case,
at their import site inside api.main. No real Groq call, no real
TigerGraph call, no GROQ_API_KEY/TigerGraph env var needed to run this
file. Uses fastapi.testclient.TestClient (sync tests over the ASGI app),
so no pytest-asyncio dependency is required.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import api.main as api_main

client = TestClient(api_main.app)


# --- GET /health ---

def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --- POST /investigate: happy path ---

def test_investigate_calls_existing_pipeline_and_returns_submission(monkeypatch):
    fake_call_llm = object()
    build_call_llm_mock = MagicMock(return_value=fake_call_llm)
    fake_internal_result = {"case_id": "HHG-011", "success": True, "outcome": {"ok": True}}
    run_single_case_mock = AsyncMock(return_value=fake_internal_result)
    fake_submission = {
        "case_id": "HHG-011",
        "case": {"status": "closed_fraud"},
        "evidence_requests": [],
        "next_best_actions": {"initial": [], "final": [], "what_changed": "nothing"},
        "sar": {"file": False, "reason": "", "narrative": "", "subjects": [], "total_amount_usd": 0, "activity_dates": []},
        "stop_reason": "x",
        "tool_calls": 3,
        "tokens": 0,
        "latency_s": 1.2,
    }
    build_submission_case_mock = MagicMock(return_value=fake_submission)

    monkeypatch.setattr(api_main, "_build_call_llm", build_call_llm_mock)
    monkeypatch.setattr(api_main, "run_single_case", run_single_case_mock)
    monkeypatch.setattr(api_main, "build_submission_case", build_submission_case_mock)

    response = client.post("/investigate", json={"case_id": "HHG-011"})

    assert response.status_code == 200
    assert response.json() == fake_submission

    build_call_llm_mock.assert_called_once_with("groq")  # the existing Groq closure, not gemini
    run_single_case_mock.assert_awaited_once_with("HHG-011", fake_call_llm)
    build_submission_case_mock.assert_called_once_with(fake_internal_result)


def test_investigate_uses_run_single_case_not_run_all_cases(monkeypatch):
    # Structural guard: this endpoint must never import or call
    # run_all_cases -- api.main doesn't even import it, confirmed here by
    # asserting the module attribute doesn't exist.
    assert not hasattr(api_main, "run_all_cases")


# --- POST /investigate: request validation ---

@pytest.mark.parametrize("bad_case_id", ["HHG-021", "HHG-000", "HHG-1", "HHG-0011", "ABC-001", "hhg-001", "", "HHG-011 "])
def test_investigate_rejects_out_of_range_or_malformed_case_id(monkeypatch, bad_case_id):
    run_single_case_mock = AsyncMock()
    monkeypatch.setattr(api_main, "run_single_case", run_single_case_mock)

    response = client.post("/investigate", json={"case_id": bad_case_id})

    assert response.status_code == 422
    run_single_case_mock.assert_not_called()


@pytest.mark.parametrize("good_case_id", ["HHG-001", "HHG-009", "HHG-010", "HHG-020"])
def test_investigate_accepts_every_valid_case_id_in_range(monkeypatch, good_case_id):
    monkeypatch.setattr(api_main, "_build_call_llm", MagicMock(return_value=object()))
    monkeypatch.setattr(api_main, "run_single_case", AsyncMock(return_value={"case_id": good_case_id}))
    monkeypatch.setattr(api_main, "build_submission_case", MagicMock(return_value={"case_id": good_case_id}))

    response = client.post("/investigate", json={"case_id": good_case_id})
    assert response.status_code == 200


def test_investigate_missing_case_id_field_is_422():
    response = client.post("/investigate", json={})
    assert response.status_code == 422


# --- POST /investigate: failure handling never leaks internals ---

def test_investigate_pipeline_exception_returns_clean_500(monkeypatch):
    monkeypatch.setattr(api_main, "_build_call_llm", MagicMock(return_value=object()))

    async def raising_run_single_case(case_id, call_llm):
        raise RuntimeError("simulated failure containing SECRET_TOKEN_ABC123")

    monkeypatch.setattr(api_main, "run_single_case", raising_run_single_case)

    response = client.post("/investigate", json={"case_id": "HHG-005"})

    assert response.status_code == 500
    body = response.json()
    assert body == {"detail": "Investigation failed"}
    assert "SECRET_TOKEN_ABC123" not in response.text
    assert "RuntimeError" not in response.text


def test_investigate_build_call_llm_exception_returns_clean_500(monkeypatch):
    # e.g. GROQ_API_KEY not configured -- must not leak that detail either.
    def raising_build_call_llm(provider):
        raise RuntimeError("No Groq API key configured -- set GROQ_API_KEY in .env")

    monkeypatch.setattr(api_main, "_build_call_llm", raising_build_call_llm)

    response = client.post("/investigate", json={"case_id": "HHG-005"})

    assert response.status_code == 500
    assert response.json() == {"detail": "Investigation failed"}
    assert "GROQ_API_KEY" not in response.text


def test_investigate_submission_adapter_exception_returns_clean_500(monkeypatch):
    monkeypatch.setattr(api_main, "_build_call_llm", MagicMock(return_value=object()))
    monkeypatch.setattr(api_main, "run_single_case", AsyncMock(return_value={"case_id": "HHG-005"}))

    def raising_build_submission_case(result):
        raise KeyError("case")

    monkeypatch.setattr(api_main, "build_submission_case", raising_build_submission_case)

    response = client.post("/investigate", json={"case_id": "HHG-005"})

    assert response.status_code == 500
    assert response.json() == {"detail": "Investigation failed"}


# --- no real credentials required for any of the above ---

def test_no_groq_or_tigergraph_env_vars_required_for_these_tests(monkeypatch):
    # Confirms this test module's own isolation: deleting any provider/
    # graph env vars (if present) must not affect the mocked test above.
    for var in ("GROQ_API_KEY", "GEMINI_API_KEY", "TG_HOST", "TG_GSQL_SECRET"):
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setattr(api_main, "_build_call_llm", MagicMock(return_value=object()))
    monkeypatch.setattr(api_main, "run_single_case", AsyncMock(return_value={"case_id": "HHG-001"}))
    monkeypatch.setattr(api_main, "build_submission_case", MagicMock(return_value={"case_id": "HHG-001"}))

    response = client.post("/investigate", json={"case_id": "HHG-001"})
    assert response.status_code == 200
