"""
G8 -- real Groq provider adapter for the CallLLM interface.

Produces exactly one thing: an `async def call_llm(prompt: str) -> str`
closure matching src.agent.reasoning.CallLLM. Nothing else in this project
imports this module -- it is wired in only by whoever explicitly passes
`call_llm=make_groq_call_llm(...)` into
case_output.investigate_and_assemble(...), the same injection point a
test's fake call_llm already uses.

Isolation (structural, not just documented):
  - No import of src.agent.workflow, src.policy.g2_policy,
    src.investigation.patterns, src.graph.tigergraph_connection.
    get_connection, or src.mcp_server.server.mcp anywhere in this file.
  - No filesystem writes, no graph writes.
  - No retries -- one API call per call_llm invocation, full stop. If the
    call fails, it raises; src.agent.reasoning.get_llm_reasoning's
    existing asyncio.wait_for(DEFAULT_LLM_TIMEOUT_S, 60s) + except
    Exception handling (unchanged by this file) turns that into the
    standard llm_call_failed/llm_timeout stop-state -- this module adds
    no new error-handling path and no timeout of its own.
  - No logging of prompts or responses. The only text this module ever
    touches after a failure is run through scrub_secret() before being
    re-raised, exactly like every other external-call boundary in this
    project. On a failed API call, _diagnose_call_failure adds prompt-
    length/error-body diagnostics to that same scrubbed message (added
    after a real HHG-011 failure gave no way to tell prompt size apart
    from any other cause) -- it captures only the prompt's length, never
    its content, and never a credential.

Structured output: uses Groq's OpenAI-compatible Structured Outputs mode
(response_format={"type": "json_schema", "json_schema": {"name": ...,
"strict": True, "schema": REASONING_OUTPUT_SCHEMA}}) -- the same schema
object src.agent.reasoning defines and documents in its prompt text, now
also carrying "additionalProperties": false as strict mode requires. This
does NOT replace src.agent.reasoning_validator's job: whatever text comes
back is handed to get_llm_reasoning() unmodified, parsed with the same
plain json.loads(), and independently validated exactly as a fake test
double's response would be.
"""
from __future__ import annotations

import os

from src.agent.reasoning import REASONING_OUTPUT_SCHEMA, CallLLM
from src.graph.tigergraph_connection import scrub_secret

DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
# G11-G: raised from 2048 after a real HHG-011 failure (400
# json_validate_failed, failed_generation_len=0) -- diagnostics added in
# G11-F confirmed prompt_chars=140570 (~35,142 estimated tokens) against
# only a 2048-token output budget. This is a single, isolated
# configuration-value experiment: it does not change the model, the
# schema, strict-mode behavior, retries, or anything upstream of this
# provider (evidence compaction, prompt contents, reasoning_validator).
# Matches the same category of fix already proven correct for Gemini
# (1024 -> 4096 after an earlier real truncation defect there).
DEFAULT_MAX_OUTPUT_TOKENS = 4096


def _diagnose_call_failure(e: Exception, prompt: str, model: str, max_output_tokens: int) -> str:
    """Diagnostic-only message builder for any exception raised by
    client.chat.completions.create -- does not change what was sent to
    Groq, does not retry, does not alter the success path below. Built
    to explain a real HHG-011 failure (400 json_validate_failed,
    failed_generation empty) whose cause -- prompt size, output budget,
    or something else -- was previously indistinguishable from
    str(exception) alone.

    Never logs or persists the full prompt -- only its length. Attribute
    names (status_code, message, body) are APIError/APIStatusError's own
    documented fields, confirmed against the installed groq SDK's actual
    exception source (groq._exceptions), not guessed; `body` is Groq's
    already-decoded JSON error response when the response was valid
    JSON (exactly the {'error': {...}} shape seen in the real HHG-011
    failure), read here generically -- including any field beyond the
    ones already known about today (e.g. a future usage/finish-reason
    addition) -- rather than hardcoded to only today's known keys.
    getattr(..., None) throughout so this never raises for an exception
    type that lacks these attributes (e.g. a plain connection error).
    """
    status_code = getattr(e, "status_code", None)
    body = getattr(e, "body", None)
    error_detail = body.get("error") if isinstance(body, dict) else None

    prompt_chars = len(prompt)
    # A rough, disclosed estimate only (~4 chars/token) -- no tokenizer
    # call and no network call is made to compute this.
    estimated_prompt_tokens = prompt_chars // 4

    parts = [
        f"Groq call failed: {type(e).__name__}",
        f"status_code={status_code}",
        f"model={model}",
        f"prompt_chars={prompt_chars}",
        f"estimated_prompt_tokens~={estimated_prompt_tokens}",
        f"max_output_tokens={max_output_tokens}",
    ]

    if isinstance(error_detail, dict):
        parts.append(f"groq_error_type={error_detail.get('type')}")
        parts.append(f"groq_error_code={error_detail.get('code')}")
        parts.append(f"groq_error_message={error_detail.get('message')}")
        failed_generation = error_detail.get("failed_generation")
        if failed_generation is not None:
            parts.append(f"failed_generation_len={len(failed_generation)}")
            parts.append(f"failed_generation={failed_generation!r}")
        extra_keys = {
            k: v for k, v in error_detail.items()
            if k not in ("type", "code", "message", "failed_generation")
        }
        if extra_keys:
            parts.append(f"additional_error_fields={extra_keys}")
    else:
        parts.append(f"message={getattr(e, 'message', str(e))}")

    return " | ".join(parts)


def make_groq_call_llm(
    *,
    model: str = DEFAULT_GROQ_MODEL,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    api_key: str | None = None,
) -> CallLLM:
    """Builds and returns a real, Groq-backed CallLLM closure.

    Reads GROQ_API_KEY from the environment if `api_key` is not passed
    explicitly. Never hardcodes a key, never logs one, and raises
    immediately here -- at factory-call time, before any investigation
    runs -- if none is configured, so a misconfigured run fails fast with
    a clear message instead of hanging on the first real call.

    The groq SDK is imported lazily, inside this function, so importing
    this module (or this project's test suite) never requires groq to be
    installed.
    """
    from groq import AsyncGroq

    resolved_key = api_key or os.environ.get("GROQ_API_KEY")
    if not resolved_key:
        raise RuntimeError(
            "No Groq API key configured -- set GROQ_API_KEY in .env. "
            "This factory never falls back to a default or hardcoded key."
        )

    client = AsyncGroq(api_key=resolved_key)
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "fraud_reasoning",
            "strict": True,
            "schema": REASONING_OUTPUT_SCHEMA,
        },
    }

    async def call_llm(prompt: str) -> str:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format=response_format,
                max_tokens=max_output_tokens,
            )
        except Exception as e:
            # Never let a raw SDK exception escape unsanitized -- some
            # client libraries embed request/auth details in their
            # __str__. Re-raised as a plain RuntimeError (not the SDK's
            # own exception type) so no exception attribute anywhere in
            # this chain can carry an unscrubbed credential fragment
            # forward; get_llm_reasoning's existing `except Exception`
            # handles this identically to any other provider failure.
            #
            # Diagnostic-only enrichment (added after a real HHG-011
            # failure -- 400 json_validate_failed, failed_generation
            # empty -- gave no way to tell prompt size/output budget
            # apart from any other cause): _diagnose_call_failure adds
            # prompt-size and Groq-error-body context to the message.
            # Never changes what was sent, never retries, never touches
            # the success path below.
            raise RuntimeError(scrub_secret(_diagnose_call_failure(e, prompt, model, max_output_tokens))) from None

        choices = response.choices
        if not choices:
            raise RuntimeError("Groq returned no choices in the response")
        text = choices[0].message.content
        if not text:
            raise RuntimeError("Groq returned an empty response (no message content)")
        return text

    return call_llm
