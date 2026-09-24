"""
G8 unit tests for src/agent/llm_providers/groq_provider.py.

Every test here monkeypatches groq.AsyncGroq with a fake, in-process
stand-in -- none of this makes a real network call, requires
GROQ_API_KEY, or depends on Groq's actual availability. groq IS installed
in this environment (see requirements.txt), so importing the real `groq`
module and patching an attribute on it is safe and does not itself make
any network call.
"""
import asyncio
import json

import pytest

import groq as groq_module

from src.agent import reasoning
from src.agent.llm_providers import groq_provider as gp
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


def _make_fake_client_class(response_text=None, exception=None, empty_choices=False):
    """Returns a fake replacement for groq.AsyncGroq whose
    .chat.completions.create(...) either returns a fixed OpenAI-shaped
    response (choices[0].message.content) or raises a given exception --
    and records every call it received for assertions.
    """
    calls = []

    class _FakeMessage:
        def __init__(self, content):
            self.content = content

    class _FakeChoice:
        def __init__(self, content):
            self.message = _FakeMessage(content)

    class _FakeCompletion:
        def __init__(self, content):
            self.choices = [] if empty_choices else [_FakeChoice(content)]

    class _FakeCompletions:
        async def create(self, model, messages, response_format, max_tokens):
            calls.append({
                "model": model, "messages": messages,
                "response_format": response_format, "max_tokens": max_tokens,
            })
            if exception is not None:
                raise exception
            return _FakeCompletion(response_text)

    class _FakeChat:
        def __init__(self):
            self.completions = _FakeCompletions()

    class _FakeClient:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.chat = _FakeChat()

    _FakeClient.calls = calls
    return _FakeClient


def _call_llm_with_fake_client(fake_client_cls, api_key: str = "test-fake-key"):
    original_client = groq_module.AsyncGroq
    groq_module.AsyncGroq = fake_client_cls
    try:
        return gp.make_groq_call_llm(api_key=api_key)
    finally:
        groq_module.AsyncGroq = original_client


def _base_evidence_package():
    return reasoning.build_evidence_package(
        {"full_evidence": {"combined_evidence": {"transaction": {"TransactionID": 1}}, "cross_card_fraud_verification": None}},
        {"recommended": [], "prohibited": []},
    )


# --- A. successful structured response / B. correct JSON-string serialization ---

def test_successful_structured_response_returns_json_string(monkeypatch):
    fake_cls = _make_fake_client_class(response_text=json.dumps(_VALID_RESPONSE_BODY))
    monkeypatch.setattr(groq_module, "AsyncGroq", fake_cls)

    call_llm = gp.make_groq_call_llm(api_key="test-fake-key")
    result = run(call_llm("some prompt"))

    assert isinstance(result, str)
    assert json.loads(result) == _VALID_RESPONSE_BODY


def test_response_flows_unmodified_into_get_llm_reasoning(monkeypatch):
    fake_cls = _make_fake_client_class(response_text=json.dumps(_VALID_RESPONSE_BODY))
    monkeypatch.setattr(groq_module, "AsyncGroq", fake_cls)

    call_llm = gp.make_groq_call_llm(api_key="test-fake-key")
    outcome = run(reasoning.get_llm_reasoning(_base_evidence_package(), call_llm=call_llm))
    assert outcome["ok"] is True
    assert outcome["raw"] == _VALID_RESPONSE_BODY


# --- C. malformed / empty provider response ---

def test_malformed_response_text_is_returned_unmodified_not_rejected_here():
    fake_cls = _make_fake_client_class(response_text="not valid json {")
    call_llm = _call_llm_with_fake_client(fake_cls)
    result = run(call_llm("prompt"))
    assert result == "not valid json {"


def test_empty_response_text_raises():
    fake_cls = _make_fake_client_class(response_text="")
    call_llm = _call_llm_with_fake_client(fake_cls)
    with pytest.raises(RuntimeError, match="empty response"):
        run(call_llm("prompt"))


def test_none_response_text_raises():
    fake_cls = _make_fake_client_class(response_text=None)
    call_llm = _call_llm_with_fake_client(fake_cls)
    with pytest.raises(RuntimeError, match="empty response"):
        run(call_llm("prompt"))


def test_empty_choices_list_raises():
    fake_cls = _make_fake_client_class(response_text="irrelevant", empty_choices=True)
    call_llm = _call_llm_with_fake_client(fake_cls)
    with pytest.raises(RuntimeError, match="no choices"):
        run(call_llm("prompt"))


# --- D. provider exception / E. rate-limit-like exception ---

def test_generic_provider_exception_is_caught_and_reraised_as_runtimeerror():
    fake_cls = _make_fake_client_class(exception=ConnectionError("upstream connection reset"))
    call_llm = _call_llm_with_fake_client(fake_cls)
    with pytest.raises(RuntimeError, match="upstream connection reset"):
        run(call_llm("prompt"))


def test_rate_limit_like_exception_is_caught_and_reraised():
    fake_cls = _make_fake_client_class(exception=Exception("429: rate_limit_exceeded, please retry after 20s"))
    call_llm = _call_llm_with_fake_client(fake_cls)
    with pytest.raises(RuntimeError, match="rate_limit_exceeded"):
        run(call_llm("prompt"))


def test_provider_exception_flows_into_existing_llm_call_failed_path(monkeypatch):
    fake_cls = _make_fake_client_class(exception=Exception("500 internal server error"))
    monkeypatch.setattr(groq_module, "AsyncGroq", fake_cls)
    call_llm = gp.make_groq_call_llm(api_key="test-fake-key")

    outcome = run(reasoning.get_llm_reasoning(_base_evidence_package(), call_llm=call_llm))
    assert outcome["ok"] is False
    assert outcome["failure_reason"] == "llm_call_failed"


# --- F. timeout/failure propagation (existing 60s G7 timeout, unchanged) ---

def test_slow_provider_call_is_terminated_by_existing_g7_timeout(monkeypatch):
    class _SlowCompletions:
        async def create(self, model, messages, response_format, max_tokens):
            await asyncio.sleep(60)
            return None  # unreachable

    class _SlowChat:
        def __init__(self):
            self.completions = _SlowCompletions()

    class _SlowClient:
        def __init__(self, api_key=None):
            self.chat = _SlowChat()

    monkeypatch.setattr(groq_module, "AsyncGroq", _SlowClient)
    call_llm = gp.make_groq_call_llm(api_key="test-fake-key")

    outcome = run(reasoning.get_llm_reasoning(_base_evidence_package(), call_llm=call_llm, timeout_s=0.05))
    assert outcome["ok"] is False
    assert outcome["failure_reason"] == "llm_timeout"


def test_default_timeout_constant_is_unchanged_at_60s():
    # this file adds no timeout of its own -- confirms reasoning.py's
    # existing G7 default (60s) is what actually protects a real call.
    assert reasoning.DEFAULT_LLM_TIMEOUT_S == 60.0


# --- G. schema mismatch ---

def test_schema_mismatched_response_still_flows_to_validator_for_rejection(monkeypatch):
    incomplete = {"verdict": "fraud"}  # missing 8 required keys
    fake_cls = _make_fake_client_class(response_text=json.dumps(incomplete))
    monkeypatch.setattr(groq_module, "AsyncGroq", fake_cls)
    call_llm = gp.make_groq_call_llm(api_key="test-fake-key")

    outcome = run(reasoning.get_llm_reasoning(_base_evidence_package(), call_llm=call_llm))
    assert outcome["ok"] is True  # syntactically valid JSON, parses fine
    assert outcome["raw"] == incomplete  # unmodified -- validator rejects it, not this layer


# --- H. max-output-token configuration / I. model configuration ---

def test_model_and_max_output_tokens_are_configurable_and_passed_through(monkeypatch):
    fake_cls = _make_fake_client_class(response_text=json.dumps(_VALID_RESPONSE_BODY))
    monkeypatch.setattr(groq_module, "AsyncGroq", fake_cls)

    call_llm = gp.make_groq_call_llm(api_key="test-fake-key", model="custom-test-model", max_output_tokens=777)
    run(call_llm("prompt"))

    assert len(fake_cls.calls) == 1
    assert fake_cls.calls[0]["model"] == "custom-test-model"
    assert fake_cls.calls[0]["max_tokens"] == 777


def test_defaults_are_explicit_constants_including_required_model():
    assert gp.DEFAULT_GROQ_MODEL == "openai/gpt-oss-20b"
    assert gp.DEFAULT_MAX_OUTPUT_TOKENS > 0


def test_default_max_output_tokens_raised_after_real_hhg011_failure():
    # G11-G: real HHG-011 diagnostic (G11-F) confirmed prompt_chars=140570
    # (~35,142 estimated tokens) failed with only a 2048-token output
    # budget (400 json_validate_failed, failed_generation_len=0). Raised
    # to 4096, matching the same category of fix already proven correct
    # for Gemini (1024 -> 4096). Pins the exact value so a future change
    # can't silently regress back toward the value that failed on a real
    # case, without at least being a deliberate, visible edit here.
    assert gp.DEFAULT_MAX_OUTPUT_TOKENS == 4096
    assert gp.DEFAULT_MAX_OUTPUT_TOKENS > 2048  # strictly above the value that actually failed


def test_request_uses_exact_specified_response_format_shape(monkeypatch):
    fake_cls = _make_fake_client_class(response_text=json.dumps(_VALID_RESPONSE_BODY))
    monkeypatch.setattr(groq_module, "AsyncGroq", fake_cls)
    call_llm = gp.make_groq_call_llm(api_key="test-fake-key")
    run(call_llm("prompt"))

    sent = fake_cls.calls[0]["response_format"]
    assert sent["type"] == "json_schema"
    assert sent["json_schema"]["name"] == "fraud_reasoning"
    assert sent["json_schema"]["strict"] is True
    assert sent["json_schema"]["schema"] == reasoning.REASONING_OUTPUT_SCHEMA
    assert sent["json_schema"]["schema"]["additionalProperties"] is False


def test_prompt_is_sent_as_a_single_user_message(monkeypatch):
    fake_cls = _make_fake_client_class(response_text=json.dumps(_VALID_RESPONSE_BODY))
    monkeypatch.setattr(groq_module, "AsyncGroq", fake_cls)
    call_llm = gp.make_groq_call_llm(api_key="test-fake-key")
    run(call_llm("the exact prompt text"))
    assert fake_cls.calls[0]["messages"] == [{"role": "user", "content": "the exact prompt text"}]


# --- no retries by default ---

def test_no_retries_on_failure_exactly_one_attempt():
    fake_cls = _make_fake_client_class(exception=Exception("transient failure"))
    call_llm = _call_llm_with_fake_client(fake_cls)
    try:
        run(call_llm("prompt"))
    except RuntimeError:
        pass
    assert len(fake_cls.calls) == 1


# --- J. no MCP/TigerGraph access ---

def test_no_mcp_or_tigergraph_access(monkeypatch):
    from src.graph import tigergraph_connection as tgc
    from src.mcp_server import server as mcp_server_module

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("groq_provider.py must never call TigerGraph/MCP directly")

    monkeypatch.setattr(tgc, "get_connection", _fail_if_called)
    monkeypatch.setattr(mcp_server_module.mcp, "call_tool", _fail_if_called)

    fake_cls = _make_fake_client_class(response_text=json.dumps(_VALID_RESPONSE_BODY))
    call_llm = _call_llm_with_fake_client(fake_cls)
    result = run(call_llm("prompt"))
    assert json.loads(result) == _VALID_RESPONSE_BODY  # would have raised above if either entry point were touched


def test_module_imports_no_mcp_or_tigergraph_symbols():
    import ast
    import inspect

    import src.agent.llm_providers.groq_provider as module

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


# --- K. no secret leakage in errors ---

def test_no_api_key_configured_error_never_contains_a_fake_key_value(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="No Groq API key configured"):
        gp.make_groq_call_llm(api_key=None)


def test_provider_exception_with_embedded_api_key_is_scrubbed(monkeypatch):
    fake_key = "gsk_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789ABCDEFGHIJKLM"
    monkeypatch.setenv("GROQ_API_KEY", fake_key)
    leaking_message = f"401 Unauthorized: key {fake_key} rejected"
    fake_cls = _make_fake_client_class(exception=Exception(leaking_message))
    call_llm = _call_llm_with_fake_client(fake_cls, api_key=fake_key)

    with pytest.raises(RuntimeError) as exc_info:
        run(call_llm("prompt"))
    assert fake_key not in str(exc_info.value)
    assert "[REDACTED]" in str(exc_info.value)


def test_provider_exception_with_bearer_header_is_scrubbed():
    leaking_message = "request failed, Authorization: Bearer sk-fake-not-a-real-token-abcdef123456"
    fake_cls = _make_fake_client_class(exception=Exception(leaking_message))
    call_llm = _call_llm_with_fake_client(fake_cls)

    with pytest.raises(RuntimeError) as exc_info:
        run(call_llm("prompt"))
    assert "sk-fake-not-a-real-token-abcdef123456" not in str(exc_info.value)
    assert "[REDACTED]" in str(exc_info.value)


def test_scrub_secret_directly_covers_groq_key_env_var(monkeypatch):
    fake_key = "gsk_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789ABCDEFGHIJKLM"
    monkeypatch.setenv("GROQ_API_KEY", fake_key)
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


# --- G11-F: HHG-011 failure diagnostics (prompt size, Groq error body) ---
# Real HHG-011 finding: a 400 json_validate_failed with an empty
# failed_generation gave no way to tell prompt size/output budget apart
# from any other cause. These tests prove the provider now preserves
# that context instead of discarding it, exactly as _diagnose_call_
# failure's own docstring describes -- and never for any other reason.

def _real_bad_request_error(error_body: dict) -> "groq.BadRequestError":
    import httpx
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(400, request=request, content=json.dumps(error_body).encode())
    return groq_module.BadRequestError(f"Error code: 400 - {error_body}", response=response, body=error_body)


_REAL_HHG011_ERROR_BODY = {
    "error": {
        "message": "Failed to validate JSON. Please adjust your prompt. See 'failed_generation' for more details.",
        "type": "invalid_request_error",
        "code": "json_validate_failed",
        "failed_generation": "",
    }
}


def test_bad_request_error_diagnostic_includes_prompt_length():
    fake_cls = _make_fake_client_class(exception=_real_bad_request_error(_REAL_HHG011_ERROR_BODY))
    call_llm = _call_llm_with_fake_client(fake_cls)
    long_prompt = "x" * 140_570  # matches the real HHG-011 compacted prompt size measured earlier this project
    with pytest.raises(RuntimeError) as exc_info:
        run(call_llm(long_prompt))
    message = str(exc_info.value)
    assert "prompt_chars=140570" in message
    assert "estimated_prompt_tokens~=35142" in message  # 140570 // 4, disclosed as an estimate


def test_bad_request_error_diagnostic_includes_configured_max_output_tokens():
    fake_cls = _make_fake_client_class(exception=_real_bad_request_error(_REAL_HHG011_ERROR_BODY))
    original_client = groq_module.AsyncGroq
    groq_module.AsyncGroq = fake_cls
    try:
        call_llm = gp.make_groq_call_llm(api_key="test-fake-key", max_output_tokens=777)
        with pytest.raises(RuntimeError) as exc_info:
            run(call_llm("prompt"))
    finally:
        groq_module.AsyncGroq = original_client
    assert "max_output_tokens=777" in str(exc_info.value)


def test_bad_request_error_diagnostic_includes_status_code_and_error_type():
    fake_cls = _make_fake_client_class(exception=_real_bad_request_error(_REAL_HHG011_ERROR_BODY))
    call_llm = _call_llm_with_fake_client(fake_cls)
    with pytest.raises(RuntimeError) as exc_info:
        run(call_llm("prompt"))
    message = str(exc_info.value)
    assert "status_code=400" in message
    assert "BadRequestError" in message


def test_bad_request_error_diagnostic_includes_groq_error_body_fields():
    fake_cls = _make_fake_client_class(exception=_real_bad_request_error(_REAL_HHG011_ERROR_BODY))
    call_llm = _call_llm_with_fake_client(fake_cls)
    with pytest.raises(RuntimeError) as exc_info:
        run(call_llm("prompt"))
    message = str(exc_info.value)
    assert "groq_error_type=invalid_request_error" in message
    assert "groq_error_code=json_validate_failed" in message
    assert "groq_error_message=Failed to validate JSON" in message


def test_bad_request_error_diagnostic_includes_failed_generation_length_and_content():
    error_body = dict(_REAL_HHG011_ERROR_BODY)
    error_body["error"] = dict(error_body["error"])
    error_body["error"]["failed_generation"] = '{"verdict": "frau'  # non-empty, to prove content is captured too
    fake_cls = _make_fake_client_class(exception=_real_bad_request_error(error_body))
    call_llm = _call_llm_with_fake_client(fake_cls)
    with pytest.raises(RuntimeError) as exc_info:
        run(call_llm("prompt"))
    message = str(exc_info.value)
    assert "failed_generation_len=17" in message
    assert "failed_generation='{\"verdict\": \"frau'" in message


def test_bad_request_error_diagnostic_captures_empty_failed_generation_exactly_as_observed():
    # The REAL HHG-011 shape: failed_generation is present but empty --
    # must be reported as length 0, not omitted or confused with "absent".
    fake_cls = _make_fake_client_class(exception=_real_bad_request_error(_REAL_HHG011_ERROR_BODY))
    call_llm = _call_llm_with_fake_client(fake_cls)
    with pytest.raises(RuntimeError) as exc_info:
        run(call_llm("prompt"))
    assert "failed_generation_len=0" in str(exc_info.value)


def test_bad_request_error_diagnostic_captures_unknown_extra_error_fields_generically():
    error_body = {
        "error": {
            "message": "some future error",
            "type": "invalid_request_error",
            "code": "json_validate_failed",
            "failed_generation": "",
            "usage": {"completion_tokens": 42},  # a field this code doesn't know about today
        }
    }
    fake_cls = _make_fake_client_class(exception=_real_bad_request_error(error_body))
    call_llm = _call_llm_with_fake_client(fake_cls)
    with pytest.raises(RuntimeError) as exc_info:
        run(call_llm("prompt"))
    assert "additional_error_fields=" in str(exc_info.value)
    assert "completion_tokens" in str(exc_info.value)


def test_diagnostic_does_not_break_for_exceptions_without_status_code_or_body():
    # Every pre-existing test in this file raises a plain Exception/
    # ConnectionError with no .status_code/.body -- getattr(..., None)
    # throughout must keep this working exactly as before.
    fake_cls = _make_fake_client_class(exception=ConnectionError("upstream connection reset"))
    call_llm = _call_llm_with_fake_client(fake_cls)
    with pytest.raises(RuntimeError, match="upstream connection reset"):
        run(call_llm("prompt"))


def test_diagnostic_message_is_scrubbed_for_secrets(monkeypatch):
    fake_key = "gsk_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789ABCDEFGHIJKLM"
    monkeypatch.setenv("GROQ_API_KEY", fake_key)
    error_body = {
        "error": {
            "message": f"invalid key {fake_key} rejected",
            "type": "invalid_request_error",
            "code": "json_validate_failed",
            "failed_generation": "",
        }
    }
    fake_cls = _make_fake_client_class(exception=_real_bad_request_error(error_body))
    call_llm = _call_llm_with_fake_client(fake_cls, api_key=fake_key)
    with pytest.raises(RuntimeError) as exc_info:
        run(call_llm("prompt"))
    message = str(exc_info.value)
    assert fake_key not in message
    assert "[REDACTED]" in message


def test_diagnostic_does_not_log_or_persist_the_full_prompt_content():
    fake_cls = _make_fake_client_class(exception=_real_bad_request_error(_REAL_HHG011_ERROR_BODY))
    call_llm = _call_llm_with_fake_client(fake_cls)
    distinctive_prompt = "UNIQUE_MARKER_" + ("y" * 1000)
    with pytest.raises(RuntimeError) as exc_info:
        run(call_llm(distinctive_prompt))
    # only the length is reported -- the prompt's actual text never appears
    assert distinctive_prompt not in str(exc_info.value)
    assert "UNIQUE_MARKER_" not in str(exc_info.value)


def test_success_path_unaffected_by_diagnostic_addition():
    fake_cls = _make_fake_client_class(response_text=json.dumps(_VALID_RESPONSE_BODY))
    call_llm = _call_llm_with_fake_client(fake_cls)
    result = run(call_llm("prompt"))
    assert json.loads(result) == _VALID_RESPONSE_BODY


def test_no_retries_still_holds_with_diagnostic_addition():
    fake_cls = _make_fake_client_class(exception=_real_bad_request_error(_REAL_HHG011_ERROR_BODY))
    call_llm = _call_llm_with_fake_client(fake_cls)
    try:
        run(call_llm("prompt"))
    except RuntimeError:
        pass
    assert len(fake_cls.calls) == 1
