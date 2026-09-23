"""
G7 -- LLM reasoning layer, isolated from G1/G2.

Consumes a curated evidence package built from the G1 ledger (workflow.py's
investigate() return value) and the G2 policy result (g2_policy.evaluate()'s
return value) -- never calls TigerGraph or an MCP tool itself, never
imports src.graph.tigergraph_connection.get_connection or
src.mcp_server.server.mcp. The LLM's raw output is NOT trusted directly by
this module: get_llm_reasoning() only invokes the model and parses its
response as JSON. Acceptance/grounding of that JSON (enum checks,
probability bounds, entity-ID validation against the evidence actually
gathered) is a separate, deterministic step in
src.agent.reasoning_validator -- this module never decides whether an LLM
response is acceptable, only whether it is syntactically JSON at all.

No LLM provider is wired in by default. call_llm must be supplied by the
caller, mirroring src.agent.workflow's call_tool injection pattern exactly
(same reasons: tests never depend on a live network call, and swapping the
underlying provider never touches this module's logic). _default_call_llm
raises clearly rather than silently picking a provider/SDK/API key -- that
is a deliberate, separate decision this module does not make on its own.
"""
from __future__ import annotations

import asyncio
import json
from typing import Awaitable, Callable

from src.graph.tigergraph_connection import scrub_secret

CallLLM = Callable[[str], Awaitable[str]]

# Explicit, configurable ceiling on the one external call this layer ever
# makes. A call_llm that never returns and never raises would otherwise
# hang the whole investigation indefinitely -- asyncio.wait_for below
# enforces this regardless of what the injected implementation does.
# Override per-call via get_llm_reasoning(..., timeout_s=...); this is a
# safety ceiling, not a calibrated production value.
DEFAULT_LLM_TIMEOUT_S = 60.0

# The 7 organizer-defined pattern enum values (README "pattern values"
# section) -- used only to describe the contract to the LLM in the prompt.
# Enum enforcement itself lives in reasoning_validator.py, not here.
_PATTERN_VALUES = (
    "card_testing", "card_not_present_fraud", "card_not_present_new_device",
    "out_of_region_use", "account_takeover", "undocumented", "none",
)

REQUIRED_OUTPUT_KEYS = (
    "verdict", "fraud_probability", "affected_txn_ids", "first_suspicious_txn_id",
    "pattern", "pattern_description", "summary", "reasoning", "uncertainties",
)


def _collect_known_ids(ledger: dict) -> dict[str, list]:
    """Every entity ID the LLM is allowed to cite, grouped by type. Built
    from full_evidence (untrimmed, see workflow.py) so nothing is missed
    the way a _trim()-collapsed tool_calls summary would miss it. Returns
    sorted lists (not sets) so prompt text is stable/reproducible across
    calls given the same input -- useful for tests and for not burning
    prompt-cache on spurious reordering.
    """
    txn_ids: set[int] = set()
    card_keys: set[str] = set()
    customer_ids: set[str] = set()
    device_profiles: set[str] = set()
    case_ids: set[str] = set()
    regions: set = set()

    full_evidence = ledger.get("full_evidence") or {}
    combined = full_evidence.get("combined_evidence") or {}

    txn = combined.get("transaction")
    if txn and txn.get("TransactionID") is not None:
        txn_ids.add(txn["TransactionID"])

    for t in combined.get("temporal_activity") or []:
        if t.get("TransactionID") is not None:
            txn_ids.add(t["TransactionID"])

    card = combined.get("card")
    if card and card.get("card_key"):
        card_keys.add(card["card_key"])

    customer = combined.get("customer")
    if customer and customer.get("customer_id"):
        customer_ids.add(customer["customer_id"])

    device = combined.get("device_evidence")
    if device and device.get("device_profile"):
        device_profiles.add(device["device_profile"])

    billing = combined.get("billing_region")
    if billing and billing.get("addr1") is not None:
        regions.add(billing["addr1"])

    historical = combined.get("historical_cases") or {}
    for c in (historical.get("direct_cases") or []) + (historical.get("connected_cases") or []):
        if c.get("case_id"):
            case_ids.add(c["case_id"])

    connected = combined.get("connected_cards") or {}
    for c in connected.get("connected_via_case") or []:
        if c.get("connected_card_key"):
            card_keys.add(c["connected_card_key"])
    for c in connected.get("connected_via_device") or []:
        if c.get("card_key"):
            card_keys.add(c["card_key"])

    cross_card = full_evidence.get("cross_card_fraud_verification") or {}
    for path in ("via_device", "via_region"):
        path_result = cross_card.get(path) if cross_card else None
        if path_result:
            for entry in path_result.get("other_cards", []):
                if entry.get("card_key"):
                    card_keys.add(entry["card_key"])
                if entry.get("customer_id"):
                    customer_ids.add(entry["customer_id"])
                for cid in entry.get("confirmed_fraud_case_ids") or []:
                    case_ids.add(cid)

    return {
        "txn": sorted(txn_ids),
        "card": sorted(card_keys),
        "customer": sorted(customer_ids),
        "device": sorted(device_profiles),
        "case": sorted(case_ids),
        "region": sorted(regions, key=str),
    }


def build_evidence_package(ledger: dict, g2_result: dict) -> dict:
    """Pure, deterministic curation of everything the LLM is allowed to
    see, split into the four categories the prompt must distinguish
    (FACTS / DETERMINISTIC FINDINGS / REASONING-ALLOWED / UNKNOWN). Never
    touches TigerGraph/MCP -- operates entirely on the already-gathered
    ledger and the already-computed G2 policy result.
    """
    full_evidence = ledger.get("full_evidence") or {}
    combined = full_evidence.get("combined_evidence") or {}
    pattern_evidence = ledger.get("pattern_evidence") or {"matched": False, "candidates": []}

    flagged_txn_id = None
    txn = combined.get("transaction")
    if txn:
        flagged_txn_id = txn.get("TransactionID")
    if flagged_txn_id is None:
        # Defensive fallback only -- combined_evidence's own transaction
        # should always carry this in practice. Guards a malformed/failed
        # ledger rather than one this workflow actually produces.
        trigger = ledger.get("trigger") or {}
        if trigger.get("type") == "flagged_txn_id":
            flagged_txn_id = trigger.get("value")

    return {
        "flagged_txn_id": flagged_txn_id,
        "known_ids": _collect_known_ids(ledger),
        "facts": {
            "transaction": combined.get("transaction"),
            "card": combined.get("card"),
            "customer": combined.get("customer"),
            "temporal_activity": combined.get("temporal_activity") or [],
            "device_evidence": combined.get("device_evidence"),
            "billing_region": combined.get("billing_region"),
            "historical_cases": combined.get("historical_cases"),
            "connected_cards": combined.get("connected_cards"),
            "cross_card_fraud_verification": full_evidence.get("cross_card_fraud_verification"),
        },
        "deterministic_findings": {
            "g1_recommendation": ledger.get("recommendation"),
            "g1_recommendation_basis": ledger.get("recommendation_basis"),
            "pattern_evidence": pattern_evidence,
            "g2_recommended_actions": g2_result.get("recommended", []),
            "g2_prohibited_actions": g2_result.get("prohibited", []),
        },
        "unknown": [
            "Customer and analyst replies are not available in this round -- no evidence_requests loop has run yet.",
            "risk_score (if present on the flagged transaction) is a model input, never a fraud verdict or probability.",
        ],
    }


def build_prompt(evidence_package: dict) -> str:
    """Pure string construction, no side effects -- fully unit-testable
    without any network/LLM call. Explicitly separates FACTS,
    DETERMINISTIC FINDINGS, what REASONING may infer, and UNKNOWN, per
    the required prompt structure.
    """
    facts_json = json.dumps(evidence_package["facts"], indent=2, default=str, sort_keys=True)
    findings_json = json.dumps(evidence_package["deterministic_findings"], indent=2, default=str, sort_keys=True)
    known_ids_json = json.dumps(evidence_package["known_ids"], indent=2, sort_keys=True)
    unknown_lines = "\n".join(f"- {line}" for line in evidence_package["unknown"])

    return f"""You are the reasoning layer of a bank fraud-investigation agent. All \
graph analysis has already been performed deterministically by a separate query \
layer and a separate rule-based policy layer. You never query the graph yourself, \
and you never override a deterministic finding.

FACTS:
Evidence already retrieved from the graph for this case -- not something you may \
add to or re-derive:
{facts_json}

DETERMINISTIC FINDINGS:
Results already established by code before you were called. Treat these as fixed, \
never as one opinion among several to weigh against your own:
{findings_json}

REASONING (what you are allowed to infer from the above):
- verdict: "fraud", "legitimate", or "uncertain" -- your honest conclusion, never risk_score
- fraud_probability: a number from 0.0 to 1.0, calibrated conservatively and honestly \
(this is scored for calibration); never copy or anchor on any risk_score value in FACTS
- affected_txn_ids: every transaction you believe is part of the same fraud episode as \
the flagged transaction, chosen ONLY from known_ids.txn below -- never invent an ID. If \
the evidence shows several overlapping candidate sequences with no clear single boundary, \
say so explicitly in `reasoning` and `uncertainties` rather than silently picking one
- first_suspicious_txn_id: the transaction where you believe the episode started, chosen \
ONLY from known_ids.txn, or null if you cannot tell
- pattern: one of {list(_PATTERN_VALUES)} -- if deterministic_findings.pattern_evidence.matched \
is true, you MUST answer "card_testing" and must not choose anything else; if it is false, \
you must NOT answer "card_testing" either
- pattern_description: two or three sentences, ONLY when pattern is "undocumented", otherwise ""
- summary: short, and grounded only in FACTS/DETERMINISTIC FINDINGS -- nothing unsupported
- reasoning: your reasoning in your own words, explicitly naming any point where the evidence \
does not clearly support one conclusion
- uncertainties: a list of short strings naming anything you could not resolve from the evidence

UNKNOWN (not available to you -- never convert these into a fabricated fact):
{unknown_lines}

KNOWN IDS (the ONLY entity IDs you may reference in affected_txn_ids or \
first_suspicious_txn_id -- anything outside this list will be discarded, never guessed \
at or silently corrected):
{known_ids_json}

You MUST NOT:
- call TigerGraph or any tool directly
- invent any transaction, customer, card, or device ID not listed above
- override deterministic_findings.pattern_evidence's card_testing result in either direction
- treat any risk_score as fraud_probability
- fabricate exposure or fabricate transactions

Respond with ONLY a single JSON object, no prose outside it, with exactly these keys: \
verdict, fraud_probability, affected_txn_ids, first_suspicious_txn_id, pattern, \
pattern_description, summary, reasoning, uncertainties."""


async def _default_call_llm(prompt: str) -> str:
    raise RuntimeError(
        "No default LLM provider is configured. Pass call_llm explicitly (see the "
        "CallLLM type) -- wiring a real provider (SDK, model, API key) is a "
        "deliberate, separate decision this module does not make on its own."
    )


async def get_llm_reasoning(
    evidence_package: dict, *, call_llm: CallLLM | None = None, timeout_s: float = DEFAULT_LLM_TIMEOUT_S,
) -> dict:
    """Invokes the (injectable) LLM and parses its response as JSON.
    Never raises: every failure mode (call raised, call timed out, wrong
    return type, invalid JSON, non-object JSON) is returned as {"ok":
    False, "failure_reason": ..., "detail": ...} for the caller to handle
    as an explicit stop-state -- never fabricated into a fake success.
    Performs no semantic validation (enums, probability bounds, ID
    grounding); that is reasoning_validator.py's job.

    The call is wrapped in asyncio.wait_for(timeout_s) -- a call_llm that
    hangs (never returns, never raises) is cancelled and treated exactly
    like any other failure, never left to hang the caller indefinitely.
    """
    call_llm = call_llm or _default_call_llm
    prompt = build_prompt(evidence_package)

    try:
        raw_text = await asyncio.wait_for(call_llm(prompt), timeout=timeout_s)
    except asyncio.TimeoutError:
        # Covers both a genuinely hanging call_llm (cancelled by wait_for
        # once timeout_s elapses) and a call_llm that raises TimeoutError
        # itself -- in Python 3.11+ asyncio.TimeoutError IS TimeoutError,
        # so these are indistinguishable by exception type, and are not
        # worth distinguishing: both mean "the LLM did not answer in
        # time," which is exactly what failure_reason should say.
        return {"ok": False, "failure_reason": "llm_timeout", "detail": f"LLM call did not complete within {timeout_s}s"}
    except Exception as e:
        return {"ok": False, "failure_reason": "llm_call_failed", "detail": scrub_secret(str(e))}

    if not isinstance(raw_text, str):
        return {
            "ok": False,
            "failure_reason": "invalid_response_type",
            "detail": f"expected str from call_llm, got {type(raw_text).__name__}",
        }

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as e:
        return {"ok": False, "failure_reason": "invalid_json", "detail": scrub_secret(str(e))}

    if not isinstance(parsed, dict):
        return {"ok": False, "failure_reason": "invalid_json", "detail": "top-level JSON value is not an object"}

    return {"ok": True, "raw": parsed, "prompt": prompt}
