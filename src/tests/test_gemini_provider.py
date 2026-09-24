"""
G8 unit tests for src/agent/llm_providers/gemini_provider.py.

Every test here monkeypatches google.genai.Client with a fake, in-process
stand-in -- none of this makes a real network call, requires
GEMINI_API_KEY, or depends on Gemini's actual availability. google-genai
IS installed in this environment (see requirements.txt), so importing the
real `google.genai` module and patching an attribute on it is safe and
does not itself make any network call.
"""
import asyncio
import json

import pytest

import google.genai as genai_module

from src.agent import reasoning
from src.agent.llm_providers import gemini_provider as gp
from src.graph.tigergraph_connection import scrub_secret


def run(coro):
    return asyncio.run(coro)


_VALID_RESPONSE_BODY = {
    "verdict": "uncertain",
    "fraud_probability": 0.4,
    "affected_txn_ids": [],
    "first_suspicious_txn_id": None,
    "pattern": "none",
    "pattern_description": "",
    "summary": "a summary",
    "reasoning": "some reasoning",
    "uncertainties": [],
}


def _make_fake_client_class(response_text=None, exception=None, candidates=None, usage_metadata=None):
    """Returns a fake replacement for google.genai.Client whose
    .aio.models.generate_content(...) either returns a fixed response
    (an object with a .text attribute, matching the real SDK's return
    shape) or raises a given exception -- and records every call it
    received for assertions. `candidates`/`usage_metadata` are optional
    and default to None (matching every pre-existing call site here, all
    of which only ever set .text) -- they exist only for the G11-C
    diagnostic tests below, which need a response shaped like the real
    SDK's finish_reason/usage_metadata fields.
    """
    calls = []

    class _FakeResponse:
        def __init__(self, text, candidates=None, usage_metadata=None):
            self.text = text
            self.candidates = candidates
            self.usage_metadata = usage_metadata

    class _FakeModels:
        async def generate_content(self, model, contents, config):
            calls.append({"model": model, "contents": contents, "config": config})
            if exception is not None:
                raise exception
            return _FakeResponse(response_text, candidates=candidates, usage_metadata=usage_metadata)

    class _FakeAio:
        def __init__(self):
            self.models = _FakeModels()

    class _FakeClient:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.aio = _FakeAio()

    _FakeClient.calls = calls
    return _FakeClient


def _call_llm_with_fake_client(fake_client_cls, monkeypatch, api_key="test-fake-key"):
    monkeypatch.setattr(genai_module, "Client", fake_client_cls)
    monkeypatch.setenv("GEMINI_API_KEY", api_key)
    return gp.make_gemini_call_llm()


def _base_evidence_package():
    return reasoning.build_evidence_package(
        {"full_evidence": {"combined_evidence": {"transaction": {"TransactionID": 1}}, "cross_card_fraud_verification": None}},
        {"recommended": [], "prohibited": []},
    )


# --- successful structured response / correct JSON-string serialization ---

def test_successful_structured_response_returns_json_string(monkeypatch):
    fake_cls = _make_fake_client_class(response_text=json.dumps(_VALID_RESPONSE_BODY))
    call_llm = _call_llm_with_fake_client(fake_cls, monkeypatch)
    result = run(call_llm("some prompt"))
    assert isinstance(result, str)
    assert json.loads(result) == _VALID_RESPONSE_BODY


def test_response_flows_unmodified_into_get_llm_reasoning(monkeypatch):
    fake_cls = _make_fake_client_class(response_text=json.dumps(_VALID_RESPONSE_BODY))
    call_llm = _call_llm_with_fake_client(fake_cls, monkeypatch)
    outcome = run(reasoning.get_llm_reasoning(_base_evidence_package(), call_llm=call_llm))
    assert outcome["ok"] is True
    assert outcome["raw"] == _VALID_RESPONSE_BODY


# --- malformed / empty provider response ---

def test_empty_response_text_raises(monkeypatch):
    fake_cls = _make_fake_client_class(response_text="")
    call_llm = _call_llm_with_fake_client(fake_cls, monkeypatch)
    with pytest.raises(RuntimeError, match="empty response"):
        run(call_llm("prompt"))


def test_none_response_text_raises(monkeypatch):
    fake_cls = _make_fake_client_class(response_text=None)
    call_llm = _call_llm_with_fake_client(fake_cls, monkeypatch)
    with pytest.raises(RuntimeError, match="empty response"):
        run(call_llm("prompt"))


def test_non_json_response_raises_before_reaching_get_llm_reasoning(monkeypatch):
    # this provider validates JSON-ness itself (json.loads inside call_llm) --
    # confirms that raises cleanly rather than silently passing bad text through.
    fake_cls = _make_fake_client_class(response_text="not valid json {")
    call_llm = _call_llm_with_fake_client(fake_cls, monkeypatch)
    with pytest.raises(json.JSONDecodeError):
        run(call_llm("prompt"))


def test_non_json_response_still_fails_safely_end_to_end(monkeypatch):
    fake_cls = _make_fake_client_class(response_text="not valid json {")
    call_llm = _call_llm_with_fake_client(fake_cls, monkeypatch)
    outcome = run(reasoning.get_llm_reasoning(_base_evidence_package(), call_llm=call_llm))
    assert outcome["ok"] is False
    assert outcome["failure_reason"] == "llm_call_failed"  # caught as a generic call failure, not silently accepted


# --- provider exception / rate-limit-like exception ---

def test_generic_provider_exception_is_caught_and_reraised_as_runtimeerror(monkeypatch):
    fake_cls = _make_fake_client_class(exception=ConnectionError("upstream connection reset"))
    call_llm = _call_llm_with_fake_client(fake_cls, monkeypatch)
    with pytest.raises(RuntimeError, match="upstream connection reset"):
        run(call_llm("prompt"))


def test_rate_limit_like_exception_is_caught_and_reraised(monkeypatch):
    fake_cls = _make_fake_client_class(exception=Exception("429 RESOURCE_EXHAUSTED: Quota exceeded"))
    call_llm = _call_llm_with_fake_client(fake_cls, monkeypatch)
    with pytest.raises(RuntimeError, match="RESOURCE_EXHAUSTED"):
        run(call_llm("prompt"))


def test_provider_exception_flows_into_existing_llm_call_failed_path(monkeypatch):
    fake_cls = _make_fake_client_class(exception=Exception("500 internal server error"))
    call_llm = _call_llm_with_fake_client(fake_cls, monkeypatch)
    outcome = run(reasoning.get_llm_reasoning(_base_evidence_package(), call_llm=call_llm))
    assert outcome["ok"] is False
    assert outcome["failure_reason"] == "llm_call_failed"


# --- timeout propagation (existing 60s G7 timeout, unchanged) ---

def test_slow_provider_call_is_terminated_by_existing_g7_timeout(monkeypatch):
    class _SlowModels:
        async def generate_content(self, model, contents, config):
            await asyncio.sleep(60)
            return None  # unreachable

    class _SlowAio:
        def __init__(self):
            self.models = _SlowModels()

    class _SlowClient:
        def __init__(self, api_key=None):
            self.aio = _SlowAio()

    monkeypatch.setattr(genai_module, "Client", _SlowClient)
    monkeypatch.setenv("GEMINI_API_KEY", "test-fake-key")
    call_llm = gp.make_gemini_call_llm()

    outcome = run(reasoning.get_llm_reasoning(_base_evidence_package(), call_llm=call_llm, timeout_s=0.05))
    assert outcome["ok"] is False
    assert outcome["failure_reason"] == "llm_timeout"


# --- schema mismatch flows to the validator, not rejected here ---

def test_schema_mismatched_but_valid_json_response_still_flows_to_validator(monkeypatch):
    incomplete = {"verdict": "fraud"}  # missing 8 required keys, but still valid JSON
    fake_cls = _make_fake_client_class(response_text=json.dumps(incomplete))
    call_llm = _call_llm_with_fake_client(fake_cls, monkeypatch)
    outcome = run(reasoning.get_llm_reasoning(_base_evidence_package(), call_llm=call_llm))
    assert outcome["ok"] is True  # syntactically valid JSON, parses fine
    assert outcome["raw"] == incomplete  # unmodified -- reasoning_validator rejects it, not this layer


# --- model / max-output-token configuration ---

def test_model_and_max_output_tokens_are_configurable_and_passed_through(monkeypatch):
    fake_cls = _make_fake_client_class(response_text=json.dumps(_VALID_RESPONSE_BODY))
    monkeypatch.setattr(genai_module, "Client", fake_cls)
    monkeypatch.setenv("GEMINI_API_KEY", "test-fake-key")
    call_llm = gp.make_gemini_call_llm(model="gemini-custom-test-model", max_output_tokens=777)
    run(call_llm("prompt"))
    assert len(fake_cls.calls) == 1
    assert fake_cls.calls[0]["model"] == "gemini-custom-test-model"
    assert fake_cls.calls[0]["config"].max_output_tokens == 777


def test_defaults_match_the_intended_model_and_are_explicit_constants():
    assert gp.DEFAULT_GEMINI_MODEL == "gemini-3.6-flash"
    assert gp.DEFAULT_MAX_OUTPUT_TOKENS > 0


def test_default_max_output_tokens_is_high_enough_to_avoid_the_real_g9_truncation_bug():
    # G9 finding: a real HHG-001 call with the old default (1024) produced
    # a JSON parse error ("Expecting property name enclosed in double
    # quotes") -- the schema-conformant response (which includes free-text
    # reasoning/summary fields) was truncated mid-object before the model
    # could close it, despite response_json_schema's usual guarantee of
    # valid JSON. Raised to 4096. This pins the specific value so a future
    # change can't silently regress back toward the value that broke a
    # real case, without at least being a deliberate, visible edit here.
    assert gp.DEFAULT_MAX_OUTPUT_TOKENS == 4096
    assert gp.DEFAULT_MAX_OUTPUT_TOKENS > 1024  # strictly above the value that actually truncated


def test_request_uses_the_shared_reasoning_output_schema_and_json_mime_type(monkeypatch):
    fake_cls = _make_fake_client_class(response_text=json.dumps(_VALID_RESPONSE_BODY))
    call_llm = _call_llm_with_fake_client(fake_cls, monkeypatch)
    run(call_llm("prompt"))
    sent_config = fake_cls.calls[0]["config"]
    assert sent_config.response_mime_type == "application/json"
    assert sent_config.response_json_schema == reasoning.REASONING_OUTPUT_SCHEMA
    assert sent_config.temperature == 0.0


def test_prompt_is_sent_as_contents_unmodified(monkeypatch):
    fake_cls = _make_fake_client_class(response_text=json.dumps(_VALID_RESPONSE_BODY))
    call_llm = _call_llm_with_fake_client(fake_cls, monkeypatch)
    run(call_llm("the exact prompt text"))
    assert fake_cls.calls[0]["contents"] == "the exact prompt text"


# --- no retries by default ---

def test_no_retries_on_failure_exactly_one_attempt(monkeypatch):
    fake_cls = _make_fake_client_class(exception=Exception("transient failure"))
    call_llm = _call_llm_with_fake_client(fake_cls, monkeypatch)
    try:
        run(call_llm("prompt"))
    except RuntimeError:
        pass
    assert len(fake_cls.calls) == 1


# --- no MCP/TigerGraph access, no graph writes ---

def test_no_mcp_or_tigergraph_access(monkeypatch):
    from src.graph import tigergraph_connection as tgc
    from src.mcp_server import server as mcp_server_module

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("gemini_provider.py must never call TigerGraph/MCP directly")

    monkeypatch.setattr(tgc, "get_connection", _fail_if_called)
    monkeypatch.setattr(mcp_server_module.mcp, "call_tool", _fail_if_called)

    fake_cls = _make_fake_client_class(response_text=json.dumps(_VALID_RESPONSE_BODY))
    call_llm = _call_llm_with_fake_client(fake_cls, monkeypatch)
    result = run(call_llm("prompt"))
    assert json.loads(result) == _VALID_RESPONSE_BODY  # would have raised above if either entry point were touched


def test_module_imports_no_mcp_or_tigergraph_symbols():
    import ast
    import inspect

    import src.agent.llm_providers.gemini_provider as module

    tree = ast.parse(inspect.getsource(module))
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_names.add(node.module)
            imported_names.update(f"{node.module}.{alias.name}" for alias in node.names)

    assert not any("mcp" in name for name in imported_names), imported_names
    assert not any(name.endswith("get_connection") for name in imported_names), imported_names


def test_no_graph_write_capability_exists_in_module():
    import inspect
    source = inspect.getsource(gp)
    assert "runInterpretedQuery" not in source
    assert "INSTALL QUERY" not in source
    assert "upsert" not in source.lower()


# --- no secret leakage in errors ---

def test_no_api_key_configured_error_never_contains_a_fake_key_value(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY is not configured"):
        gp.make_gemini_call_llm()


def test_provider_exception_with_embedded_api_key_is_scrubbed(monkeypatch):
    fake_key = "AIzaSyDaGmWKa4JsXZ-HjGw7ISLn_3namBGewQe"
    leaking_message = f"401 Unauthorized: key {fake_key} rejected"
    fake_cls = _make_fake_client_class(exception=Exception(leaking_message))
    call_llm = _call_llm_with_fake_client(fake_cls, monkeypatch, api_key=fake_key)

    with pytest.raises(RuntimeError) as exc_info:
        run(call_llm("prompt"))
    assert fake_key not in str(exc_info.value)
    assert "[REDACTED]" in str(exc_info.value)


def test_provider_exception_with_bearer_header_is_scrubbed(monkeypatch):
    leaking_message = "request failed, Authorization: Bearer sk-fake-not-a-real-token-abcdef123456"
    fake_cls = _make_fake_client_class(exception=Exception(leaking_message))
    call_llm = _call_llm_with_fake_client(fake_cls, monkeypatch)

    with pytest.raises(RuntimeError) as exc_info:
        run(call_llm("prompt"))
    assert "sk-fake-not-a-real-token-abcdef123456" not in str(exc_info.value)
    assert "[REDACTED]" in str(exc_info.value)


def test_scrub_secret_directly_covers_gemini_key_env_var(monkeypatch):
    fake_key = "AIzaSyDaGmWKa4JsXZ-HjGw7ISLn_3namBGewQe"
    monkeypatch.setenv("GEMINI_API_KEY", fake_key)
    text = f"error: invalid key {fake_key}"
    scrubbed = scrub_secret(text)
    assert fake_key not in scrubbed
    assert "[REDACTED]" in scrubbed


def test_module_never_prints_or_logs_anything():
    import inspect
    source = inspect.getsource(gp)
    assert "print(" not in source
    assert "logging." not in source
    assert "logger." not in source


# --- G11-C: parse-failure diagnostics (finish_reason, usage_metadata, raw text) ---
# Real HHG-001 finding: all 20 real cases failed identically with
# "Unterminated string starting at: line 1 column 357 (char 356)" and
# call_error/raw_response gave no way to tell truncation-by-max-tokens
# apart from a safety block or any other cause. These tests prove the
# provider now preserves that information instead of discarding it.

from google.genai import types as _genai_types  # noqa: E402 -- same real SDK types.py imports, confirmed below


def test_diagnostic_message_preserves_raw_truncated_text(monkeypatch):
    truncated = '{"verdict": "frau'  # deliberately cut mid-string, like the real failure
    fake_cls = _make_fake_client_class(response_text=truncated)
    call_llm = _call_llm_with_fake_client(fake_cls, monkeypatch)
    with pytest.raises(json.JSONDecodeError) as exc_info:
        run(call_llm("prompt"))
    assert truncated in str(exc_info.value)


def test_diagnostic_message_includes_real_sdk_finish_reason_enum_value(monkeypatch):
    from types import SimpleNamespace
    candidate = SimpleNamespace(finish_reason=_genai_types.FinishReason.MAX_TOKENS, finish_message=None)
    fake_cls = _make_fake_client_class(response_text='{"verdict": "frau', candidates=[candidate])
    call_llm = _call_llm_with_fake_client(fake_cls, monkeypatch)
    with pytest.raises(json.JSONDecodeError) as exc_info:
        run(call_llm("prompt"))
    assert "finish_reason=MAX_TOKENS" in str(exc_info.value)


def test_diagnostic_message_includes_usage_metadata_token_counts(monkeypatch):
    from types import SimpleNamespace
    usage = SimpleNamespace(
        prompt_token_count=58000, candidates_token_count=4096,
        thoughts_token_count=4090, total_token_count=62096,
    )
    fake_cls = _make_fake_client_class(response_text='{"verdict": "frau', usage_metadata=usage)
    # a short, distinctive api_key: below scrub_secret's 3-char fragment
    # threshold, so it can't collide with substrings of the diagnostic
    # field names themselves (e.g. the default "test-fake-key" fixture
    # value shares "tes" with "candidates", which scrub_secret's
    # fragment-matching would otherwise redact -- a fixture collision,
    # not a bug in the diagnostic code).
    call_llm = _call_llm_with_fake_client(fake_cls, monkeypatch, api_key="ab")
    with pytest.raises(json.JSONDecodeError) as exc_info:
        run(call_llm("prompt"))
    message = str(exc_info.value)
    assert "prompt_token_count=58000" in message
    assert "candidates_token_count=4096" in message
    assert "thoughts_token_count=4090" in message
    assert "total_token_count=62096" in message


def test_diagnostic_message_handles_missing_candidates_and_usage_gracefully(monkeypatch):
    # matches the ORIGINAL _FakeResponse shape used by every other test in
    # this file: no .candidates/.usage_metadata at all, exactly like a
    # bare object with only .text -- must not crash.
    fake_cls = _make_fake_client_class(response_text='{"verdict": "frau')
    call_llm = _call_llm_with_fake_client(fake_cls, monkeypatch)
    with pytest.raises(json.JSONDecodeError) as exc_info:
        run(call_llm("prompt"))
    message = str(exc_info.value)
    assert "finish_reason=None" in message
    assert "usage_metadata=(unavailable)" in message


def test_diagnostic_message_still_reports_the_original_parse_position(monkeypatch):
    fake_cls = _make_fake_client_class(response_text='{"verdict": "frau')
    call_llm = _call_llm_with_fake_client(fake_cls, monkeypatch)
    with pytest.raises(json.JSONDecodeError) as exc_info:
        run(call_llm("prompt"))
    assert "char" in str(exc_info.value)  # the standard JSONDecodeError position suffix is preserved


def test_diagnostic_message_is_scrubbed_for_secrets_in_raw_text(monkeypatch):
    fake_key = "AIzaSyDaGmWKa4JsXZ-HjGw7ISLn_3namBGewQe"
    truncated_with_key = f'{{"verdict": "the key is {fake_key} and then it cuts off'
    fake_cls = _make_fake_client_class(response_text=truncated_with_key)
    call_llm = _call_llm_with_fake_client(fake_cls, monkeypatch, api_key=fake_key)
    with pytest.raises(json.JSONDecodeError) as exc_info:
        run(call_llm("prompt"))
    message = str(exc_info.value)
    assert fake_key not in message
    assert "[REDACTED]" in message


def test_diagnostic_failure_still_flows_into_existing_llm_call_failed_path(monkeypatch):
    # end-to-end: reasoning.get_llm_reasoning's own generic except-Exception
    # still classifies this the same way as before -- the richer message
    # changes only what's inside `detail`, never the failure_reason.
    from types import SimpleNamespace
    candidate = SimpleNamespace(finish_reason=_genai_types.FinishReason.MAX_TOKENS, finish_message=None)
    fake_cls = _make_fake_client_class(response_text='{"verdict": "frau', candidates=[candidate])
    call_llm = _call_llm_with_fake_client(fake_cls, monkeypatch)
    outcome = run(reasoning.get_llm_reasoning(_base_evidence_package(), call_llm=call_llm))
    assert outcome["ok"] is False
    assert outcome["failure_reason"] == "llm_call_failed"
    assert "MAX_TOKENS" in outcome["detail"]


def test_diagnose_parse_failure_helper_direct(monkeypatch):
    from types import SimpleNamespace
    response = SimpleNamespace(
        candidates=[SimpleNamespace(finish_reason=_genai_types.FinishReason.SAFETY, finish_message="blocked")],
        usage_metadata=SimpleNamespace(
            prompt_token_count=10, candidates_token_count=5,
            thoughts_token_count=0, total_token_count=15,
        ),
    )
    try:
        json.loads("not json")
    except json.JSONDecodeError as parse_error:
        message = gp._diagnose_parse_failure("not json", response, parse_error)
    assert "finish_reason=SAFETY" in message
    assert "finish_message='blocked'" in message
    assert "prompt_token_count=10" in message
    assert "raw_text='not json'" in message


def test_genai_types_finish_reason_and_usage_metadata_field_names_are_real():
    # Anchors this module's assumed SDK attribute names against the
    # actually-installed google-genai SDK -- if a future SDK upgrade
    # renames these, this test (not a live Gemini call) catches it.
    assert "finish_reason" in _genai_types.Candidate.model_fields
    assert "finish_message" in _genai_types.Candidate.model_fields
    usage_fields = _genai_types.GenerateContentResponseUsageMetadata.model_fields
    assert "prompt_token_count" in usage_fields
    assert "candidates_token_count" in usage_fields
    assert "thoughts_token_count" in usage_fields
    assert "total_token_count" in usage_fields
