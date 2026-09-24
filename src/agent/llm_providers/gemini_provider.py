from __future__ import annotations

import json
import os

from src.agent.reasoning import CallLLM, REASONING_OUTPUT_SCHEMA
from src.graph.tigergraph_connection import scrub_secret


DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
# G9 finding: 1024 was too low for this schema in practice -- a real
# HHG-001 call truncated mid-object (the free-text reasoning/summary
# fields can run long), producing a JSON parse error despite
# response_json_schema's usual guarantee of valid JSON. Raised to a more
# realistic budget for this 9-field schema; still far under the model's
# documented 65,536-token output ceiling. Not a retry mechanism, not an
# architecture change -- a single numeric configuration value.
DEFAULT_MAX_OUTPUT_TOKENS = 4096


def make_gemini_call_llm(
    model: str = DEFAULT_GEMINI_MODEL,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> CallLLM:
    """
    Create an async CallLLM adapter backed by Google Gemini.

    The adapter is intentionally isolated from the investigation workflow.
    It receives only the rendered reasoning prompt and returns only the
    structured JSON response expected by the existing reasoning layer. No
    import of src.graph.tigergraph_connection.get_connection or
    src.mcp_server.server.mcp anywhere in this file, no filesystem writes,
    no graph writes, no retries -- one call per invocation.
    """

    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    client = genai.Client(api_key=api_key)

    async def call_llm(prompt: str) -> str:
        try:
            response = await client.aio.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=max_output_tokens,
                    response_mime_type="application/json",
                    response_json_schema=REASONING_OUTPUT_SCHEMA,
                ),
            )
        except Exception as e:
            # Defense in depth: get_llm_reasoning's own except-Exception
            # already scrubs whatever this raises, but the source should
            # never depend solely on the caller to sanitize -- matching
            # every other external-call boundary in this project.
            raise RuntimeError(scrub_secret(str(e))) from None

        text = getattr(response, "text", None)
        if not text:
            raise RuntimeError("Gemini returned an empty response")

        # Validate that the provider actually returned JSON before handing
        # it to the existing reasoning/validation layer. On failure, the
        # raised exception's type is kept as json.JSONDecodeError (same as
        # before) so callers that distinguish it are unaffected -- only
        # its message is enriched with the diagnostics below, since the
        # existing call sites (reasoning.py, g9_runner.py) surface only
        # str(exception) and are not modified here.
        try:
            json.loads(text)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                scrub_secret(_diagnose_parse_failure(text, response, e)), text, e.pos,
            ) from None
        return text

    return call_llm


def _diagnose_parse_failure(text: str, response, parse_error: json.JSONDecodeError) -> str:
    """Builds a diagnostic message for a response that failed the JSON
    self-check above -- preserving the actual (truncated) response text
    and the two SDK-provided signals that explain *why* generation ended
    (finish_reason: STOP/MAX_TOKENS/SAFETY/etc., and usage_metadata's
    token accounting, including thoughts_token_count for models that
    consume part of the output budget on internal reasoning). Field
    names (candidates[].finish_reason/finish_message,
    usage_metadata.prompt_token_count/candidates_token_count/
    thoughts_token_count/total_token_count) were confirmed against the
    installed google-genai SDK's actual response model fields
    (google.genai.types.Candidate / GenerateContentResponseUsageMetadata),
    not guessed.
    """
    candidates = getattr(response, "candidates", None) or []
    finish_reason = getattr(candidates[0], "finish_reason", None) if candidates else None
    finish_message = getattr(candidates[0], "finish_message", None) if candidates else None
    finish_reason_value = getattr(finish_reason, "value", finish_reason)

    usage = getattr(response, "usage_metadata", None)
    if usage is not None:
        usage_str = (
            f"prompt_token_count={getattr(usage, 'prompt_token_count', None)} "
            f"candidates_token_count={getattr(usage, 'candidates_token_count', None)} "
            f"thoughts_token_count={getattr(usage, 'thoughts_token_count', None)} "
            f"total_token_count={getattr(usage, 'total_token_count', None)}"
        )
    else:
        usage_str = "unavailable"

    return (
        f"{parse_error.msg} "
        f"finish_reason={finish_reason_value} finish_message={finish_message!r} "
        f"usage_metadata=({usage_str}) raw_text={text!r}"
    )
