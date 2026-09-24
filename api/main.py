"""
Minimal HTTP API wrapper around the existing fraud-investigation backend.

This module implements NO investigation logic of its own. It is a thin
FastAPI adapter over the already-existing, already-tested pipeline:

    src.benchmark.g9_runner.run_single_case
    src.benchmark.g9_runner._build_call_llm
    src.benchmark.submission_adapter.build_submission_case

Every POST /investigate call performs exactly one real LLM call (Groq)
and one real TigerGraph investigation + graph write-back, via the SAME
code path the 20-case benchmark (g9_runner.run_all_cases) already uses
for each case -- nothing here duplicates, reimplements, or bypasses that
pipeline. This module never calls run_all_cases, never writes a
benchmark JSON file to outputs/ or cases/, and never touches
src/agent, src/benchmark, src/policy, or TigerGraph code.

No secret ever appears in a response or a log line here: run_single_case
and build_submission_case already never return one (the backend's own
providers scrub secrets before any exception leaves them -- see
src/agent/llm_providers/*.py), and on any failure this module logs only
the case_id and returns a fixed, generic error message -- never
str(exception).
"""
from __future__ import annotations

import logging
import re

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from src.benchmark.g9_runner import _build_call_llm, run_single_case
from src.benchmark.submission_adapter import build_submission_case

logger = logging.getLogger("fraud_investigation_api")

# Exactly HHG-001 through HHG-020, zero-padded to 3 digits -- the 20
# benchmark case IDs this backend's case_pack actually knows about.
_CASE_ID_PATTERN = re.compile(r"^HHG-(00[1-9]|01[0-9]|020)$")

app = FastAPI(title="Fraud Investigation API", version="0.1.0")

# CORS: allows the local Next.js dev server (http://localhost:3000) to
# call this API from the browser. A specific origin only -- never "*" --
# and only what the existing two routes actually need: GET (/health),
# POST (/investigate), and the Content-Type header for the JSON request
# body. Credentials stay disabled (the default) -- this API uses no
# cookies/session auth, so there is nothing for the browser to send.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class InvestigateRequest(BaseModel):
    case_id: str

    @field_validator("case_id")
    @classmethod
    def _validate_case_id(cls, value: str) -> str:
        if not _CASE_ID_PATTERN.match(value):
            raise ValueError("case_id must be one of HHG-001 through HHG-020")
        return value


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/investigate")
async def investigate(request: InvestigateRequest) -> dict:
    """Runs the existing investigation pipeline end-to-end for one case
    and returns the organizer-shaped submission JSON, plus one additive
    `evidence_delta` field (see src.agent.case_output._evidence_delta)
    for the development dashboard only -- present (non-null) only when
    this case actually triggered an evidence request, present as
    `null` otherwise, never part of build_submission_case's own return
    value or of any file written to outputs/ or cases/. Performs a real
    Groq call and a real TigerGraph read + write, exactly as
    run_single_case already does when called directly -- this endpoint
    adds no retry, no caching, and no additional call of its own.
    """
    try:
        call_llm = _build_call_llm("groq")
        result = await run_single_case(request.case_id, call_llm)
        # Internal-only field for the development dashboard (see
        # src.agent.case_output._evidence_delta) -- build_submission_case
        # itself is never modified, so every real submission artifact
        # (cases/<id>.json, written only by src.benchmark.g9_runner /
        # submission_adapter.write_submission_file, never by this dev-only
        # endpoint) is completely unaffected by this attribute.
        submission = build_submission_case(result)
        submission["evidence_delta"] = result.get("outcome", {}).get("evidence_delta")
        return submission
    except Exception:
        # Never leak str(exception) to the client or the log -- the
        # backend's own providers already scrub secrets before raising,
        # but this boundary stays conservative regardless: only the
        # case_id (never sensitive) is recorded, and the client gets a
        # fixed, generic message.
        logger.exception("Investigation failed for case_id=%s", request.case_id)
        raise HTTPException(status_code=500, detail="Investigation failed") from None
