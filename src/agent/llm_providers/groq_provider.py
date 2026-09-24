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
    project.

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
DEFAULT_MAX_OUTPUT_TOKENS = 2048


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
            raise RuntimeError(scrub_secret(str(e))) from None

        choices = response.choices
        if not choices:
            raise RuntimeError("Groq returned no choices in the response")
        text = choices[0].message.content
        if not text:
            raise RuntimeError("Groq returned an empty response (no message content)")
        return text

    return call_llm
