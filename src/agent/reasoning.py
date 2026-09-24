"""
G7 -- LLM reasoning layer, isolated from G1/G2.

Consumes a curated evidence package built from the G1 ledger
(workflow.py's investigate() return value) and the G2 policy result
(g2_policy.evaluate()'s return value).

This module:
- never calls TigerGraph or an MCP tool itself
- never imports the TigerGraph connection or MCP server
- only parses the raw LLM response as JSON
- leaves semantic validation to reasoning_validator.py
- keeps the LLM-facing evidence bounded
- preserves full_evidence upstream for grounding and policy correctness

No LLM provider is wired in by default. call_llm must be supplied by the
caller, mirroring src.agent.workflow's call_tool injection pattern exactly.
"""

from __future__ import annotations

import asyncio
import json
from typing import Awaitable, Callable

from src.graph.tigergraph_connection import scrub_secret


CallLLM = Callable[[str], Awaitable[str]]

# Explicit safety ceiling for the single external LLM call.
DEFAULT_LLM_TIMEOUT_S = 60.0


VALID_VERDICTS = ("fraud", "legitimate", "uncertain")

VALID_PATTERNS = (
    "card_testing",
    "card_not_present_fraud",
    "card_not_present_new_device",
    "out_of_region_use",
    "account_takeover",
    "undocumented",
    "none",
)

REQUIRED_OUTPUT_KEYS = (
    "verdict",
    "fraud_probability",
    "affected_txn_ids",
    "first_suspicious_txn_id",
    "pattern",
    "pattern_description",
    "summary",
    "reasoning",
    "uncertainties",
)


# Canonical structured-output schema used by providers such as Gemini.
REASONING_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "title": "FraudInvestigationReasoning",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": list(VALID_VERDICTS),
        },
        "fraud_probability": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "affected_txn_ids": {
            "type": "array",
            "items": {
                "type": "integer",
            },
        },
        "first_suspicious_txn_id": {
            "anyOf": [
                {
                    "type": "integer",
                },
                {
                    "type": "null",
                },
            ],
        },
        "pattern": {
            "type": "string",
            "enum": list(VALID_PATTERNS),
        },
        "pattern_description": {
            "type": "string",
        },
        "summary": {
            "type": "string",
        },
        "reasoning": {
            "type": "string",
        },
        "uncertainties": {
            "type": "array",
            "items": {
                "type": "string",
            },
        },
    },
    "required": list(REQUIRED_OUTPUT_KEYS),
    "additionalProperties": False,
}


def _compact_cross_card_evidence(
    cross_card: dict | None,
    device_hub_flag: bool | None,
    region_hub_flag: bool | None,
) -> dict | None:
    """
    Build a deterministic, complete LLM-facing representation of Q9.

    The raw cross-card evidence remains untouched in full_evidence. This
    function only changes what gets serialized into the LLM prompt.

    For the LLM:
    - total connected cards are preserved exactly (never sampled)
    - hub status is preserved
    - EVERY confirmed-fraud card is included -- no first-N sampling of
      any kind. A real, highly-connected card (HHG-011) has 301+137
      confirmed-fraud cards; all of them are kept, because a first-N cut
      here would silently discard fraud-positive evidence, which is
      never acceptable regardless of prompt size.
    - non-fraud cards are represented only by the aggregate counts above
      -- that is the actual size reduction: not sampling the fraud-
      positive set, but never serializing the much larger non-fraud set
      individually at all.
    - records are deterministically sorted by card_key for reproducible
      prompt text, not as a relevance ranking
    - no fabricated card/case IDs are created
    """
    if not cross_card:
        return None

    def _compact_path(
        path_result: dict | None,
        hub_flag: bool | None,
    ) -> dict | None:
        if not path_result:
            return None

        other_cards = path_result.get("other_cards") or []

        confirmed = sorted(
            (
                {
                    "card_key": c.get("card_key"),
                    "customer_id": c.get("customer_id"),
                    "same_customer": c.get("same_customer"),
                    "confirmed_fraud_case_ids": list(c.get("confirmed_fraud_case_ids") or []),
                }
                for c in other_cards
                if isinstance(c, dict) and c.get("has_confirmed_fraud")
            ),
            key=lambda c: c["card_key"] or "",
        )

        return {
            "total_connected_card_count": len(other_cards),
            "hub_flag": hub_flag,
            "confirmed_fraud_card_count": len(confirmed),
            "confirmed_fraud_cards": confirmed,  # ALL of them, never truncated
        }

    return {
        "via_device": _compact_path(cross_card.get("via_device"), device_hub_flag),
        "via_region": _compact_path(cross_card.get("via_region"), region_hub_flag),
    }


def _collect_known_ids(ledger: dict) -> dict[str, list]:
    """Every entity ID the LLM is allowed to cite, grouped by type. Built
    from full_evidence (untrimmed, see workflow.py) so nothing is missed
    the way a _trim()-collapsed tool_calls summary would miss it. Returns
    sorted lists (not sets) so prompt text is stable/reproducible across
    calls given the same input.

    All six categories are returned -- txn, card, customer, device, case,
    region -- even though the current REASONING_OUTPUT_SCHEMA only lets
    the LLM cite transaction IDs (affected_txn_ids/first_suspicious_txn_id).
    The other categories remain part of this contract because other code
    (tests, and any future schema field that names a card/customer/case)
    depends on this function returning the complete, grounded picture --
    only the PROMPT (build_prompt) decides which categories are actually
    worth rendering into the KNOWN IDS text; this function never narrows
    itself to match what one particular prompt happens to need today.

    Q9 cross_card_fraud_verification is restricted to has_confirmed_fraud
    == True entries only -- the same real, highly-connected card that
    broke the prompt's size has 795+197 raw other_cards entries; walking
    all of them here would leak hundreds of ids into the prompt even
    after facts.cross_card_fraud_verification itself is compacted. No id
    is included that the compact FACTS representation doesn't already
    surface. connected_via_device (Q5) is deliberately NOT walked here --
    it carries no fraud signal of its own and is redundant with Q9's
    more precise, fraud-tagged pass; for the same real card this alone
    was 795 additional raw entries. connected_via_case IS kept -- it is
    the smaller, shared-closed-case-derived signal (typically a handful
    of cards, 0 for HHG-011 itself) and carries more direct meaning than
    a bare device-sharing connection.
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

    cross_card = full_evidence.get("cross_card_fraud_verification") or {}
    for path in ("via_device", "via_region"):
        path_result = cross_card.get(path) if cross_card else None
        if path_result:
            for entry in path_result.get("other_cards", []):
                if not (isinstance(entry, dict) and entry.get("has_confirmed_fraud")):
                    continue
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


def build_evidence_package(
    ledger: dict,
    g2_result: dict,
) -> dict:
    """
    Pure deterministic curation of the evidence available to the LLM.

    This function:
    - never calls TigerGraph
    - never calls MCP
    - never changes full_evidence
    - separates facts from deterministic findings and unknowns
    """

    full_evidence = ledger.get("full_evidence") or {}
    combined = full_evidence.get("combined_evidence") or {}

    pattern_evidence = ledger.get(
        "pattern_evidence"
    ) or {
        "matched": False,
        "candidates": [],
    }

    flagged_txn_id = None

    txn = combined.get("transaction")

    if txn:
        flagged_txn_id = txn.get("TransactionID")

    if flagged_txn_id is None:
        trigger = ledger.get("trigger") or {}

        if trigger.get("type") == "flagged_txn_id":
            flagged_txn_id = trigger.get("value")

    device_evidence = combined.get("device_evidence")
    billing_region = combined.get("billing_region")
    connected_cards = combined.get("connected_cards") or {}

    return {
        "flagged_txn_id": flagged_txn_id,

        "known_ids": _collect_known_ids(ledger),

        "facts": {
            "transaction": combined.get("transaction"),

            "card": combined.get("card"),

            "customer": combined.get("customer"),

            "temporal_activity": combined.get(
                "temporal_activity"
            ) or [],

            "device_evidence": device_evidence,

            "billing_region": billing_region,

            "historical_cases": combined.get(
                "historical_cases"
            ),

            # Q5 connection information is represented as counts only.
            # Q9 contains the explicit cross-card fraud verification signal.
            "connected_cards": {
                "connected_via_case_count": len(
                    connected_cards.get("connected_via_case") or []
                ),
                "connected_via_device_count": len(
                    connected_cards.get("connected_via_device") or []
                ),
            },

            "cross_card_fraud_verification": _compact_cross_card_evidence(
                full_evidence.get(
                    "cross_card_fraud_verification"
                ),
                (device_evidence or {}).get("hub_flag"),
                (billing_region or {}).get("hub_flag"),
            ),
        },

        "deterministic_findings": {
            "g1_recommendation": ledger.get(
                "recommendation"
            ),

            "g1_recommendation_basis": ledger.get(
                "recommendation_basis"
            ),

            "pattern_evidence": pattern_evidence,

            "g2_recommended_actions": g2_result.get(
                "recommended",
                [],
            ),

            "g2_prohibited_actions": g2_result.get(
                "prohibited",
                [],
            ),
        },

        "unknown": [
            (
                "Customer and analyst replies are not available in this "
                "round -- no evidence_requests loop has run yet."
            ),
            (
                "risk_score (if present on the flagged transaction) is a "
                "model input, never a fraud verdict or probability."
            ),
        ],
    }


def _compact_cross_card_for_prompt(compact: dict | None) -> dict | None:
    """PROMPT-ONLY rendering, not an evidence contract. Converts the
    internal _compact_cross_card_evidence() dict-shaped records (pinned
    by its own tests to card_key/customer_id/same_customer/
    confirmed_fraud_case_ids) into compact [card_key, same_customer,
    case_ids] arrays for the TEXT actually sent to the LLM.

    customer_id is omitted here ONLY -- never in evidence_package['facts']
    itself. It is always derivable from card_key by this project's own
    established card-identity convention (card_key ==
    f"{customer_id}:{card1}", confirmed since Phase C), and no output
    field ever asks the LLM to cite a customer_id. Repeating it verbatim
    for hundreds of real confirmed-fraud cards (HHG-011: 438) adds prompt
    size with zero grounding value. Every other required-for-reasoning
    field (which card, same_customer, associated case ids) is preserved
    in full for every confirmed-fraud card -- nothing is sampled or
    dropped, only re-shaped from a verbose dict to a compact array.
    """
    if not compact:
        return None

    def _render_path(path: dict | None) -> dict | None:
        if not path:
            return None
        return {
            "total_connected_card_count": path["total_connected_card_count"],
            "hub_flag": path["hub_flag"],
            "confirmed_fraud_card_count": path["confirmed_fraud_card_count"],
            # [card_key, same_customer, confirmed_fraud_case_ids] per card --
            # ALL confirmed-fraud cards, never truncated:
            "confirmed_fraud_cards": [
                [c["card_key"], c["same_customer"], c["confirmed_fraud_case_ids"]]
                for c in path["confirmed_fraud_cards"]
            ],
        }

    return {
        "via_device": _render_path(compact.get("via_device")),
        "via_region": _render_path(compact.get("via_region")),
    }


def build_prompt(evidence_package: dict) -> str:
    """
    Pure prompt construction.

    The prompt explicitly separates:
    - FACTS
    - DETERMINISTIC FINDINGS
    - REASONING
    - UNKNOWN
    - KNOWN IDS

    Two PROMPT-ONLY size reductions are applied here, neither of which
    changes evidence_package itself (the dict this function receives is
    never mutated) or any other consumer of it:

    1. cross_card_fraud_verification is re-rendered via
       _compact_cross_card_for_prompt (dict-of-dicts -> dict-of-arrays,
       customer_id dropped -- see that function's docstring for why this
       is correctness-preserving, not evidence-weakening).
    2. KNOWN IDS renders only the `txn` category. REASONING_OUTPUT_SCHEMA
       only ever lets the LLM cite transaction IDs (affected_txn_ids,
       first_suspicious_txn_id) -- it has no card/customer/device/case/
       region output field at all -- so those categories serve no
       grounding purpose in the rendered prompt text. evidence_package
       ["known_ids"] itself (returned by _collect_known_ids) still
       carries all six categories, complete and untouched, exactly as
       reasoning_validator.py and every existing test expect.
    """

    facts_for_prompt = dict(evidence_package["facts"])
    facts_for_prompt["cross_card_fraud_verification"] = _compact_cross_card_for_prompt(
        evidence_package["facts"].get("cross_card_fraud_verification")
    )

    facts_json = json.dumps(
        facts_for_prompt,
        indent=2,
        default=str,
        sort_keys=True,
    )

    findings_json = json.dumps(
        evidence_package["deterministic_findings"],
        indent=2,
        default=str,
        sort_keys=True,
    )

    # Only `txn` is rendered -- see docstring above. evidence_package
    # itself is untouched; this is a local, prompt-only view.
    known_ids_json = json.dumps(
        {"txn": evidence_package["known_ids"]["txn"]},
        indent=2,
        sort_keys=True,
    )

    unknown_lines = "\n".join(
        f"- {line}"
        for line in evidence_package["unknown"]
    )

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
- pattern: one of {list(VALID_PATTERNS)} -- if deterministic_findings.pattern_evidence.matched \
is true, you MUST answer "card_testing" and must not choose anything else; if it is false, \
you must NOT answer "card_testing" either
- pattern_description: two or three sentences, ONLY when pattern is "undocumented", otherwise ""
- summary: short, and grounded only in FACTS/DETERMINISTIC FINDINGS -- nothing unsupported
- reasoning: your reasoning in your own words, explicitly naming any point where the evidence \
does not clearly support one conclusion
- uncertainties: a list of short strings naming anything you could not resolve from the evidence

UNKNOWN (not available to you -- never convert these into a fabricated fact):
{unknown_lines}

KNOWN IDS (the ONLY transaction IDs you may reference in affected_txn_ids or \
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
        "No default LLM provider is configured. Pass call_llm explicitly "
        "(see the CallLLM type) -- wiring a real provider (SDK, model, API key) "
        "is a deliberate, separate decision this module does not make on its own."
    )


async def get_llm_reasoning(
    evidence_package: dict,
    *,
    call_llm: CallLLM | None = None,
    timeout_s: float = DEFAULT_LLM_TIMEOUT_S,
) -> dict:
    """
    Invoke the injected LLM and parse its response as JSON.

    This function never fabricates a successful result.

    Semantic validation such as:
    - enum validation
    - probability bounds
    - transaction-ID grounding

    remains the responsibility of reasoning_validator.py.
    """

    call_llm = call_llm or _default_call_llm

    prompt = build_prompt(evidence_package)

    try:
        raw_text = await asyncio.wait_for(
            call_llm(prompt),
            timeout=timeout_s,
        )

    except asyncio.TimeoutError:
        return {
            "ok": False,
            "failure_reason": "llm_timeout",
            "detail": (
                f"LLM call did not complete within "
                f"{timeout_s}s"
            ),
        }

    except Exception as e:
        return {
            "ok": False,
            "failure_reason": "llm_call_failed",
            "detail": scrub_secret(str(e)),
        }

    if not isinstance(raw_text, str):
        return {
            "ok": False,
            "failure_reason": "invalid_response_type",
            "detail": (
                "expected str from call_llm, got "
                f"{type(raw_text).__name__}"
            ),
        }

    try:
        parsed = json.loads(raw_text)

    except json.JSONDecodeError as e:
        return {
            "ok": False,
            "failure_reason": "invalid_json",
            "detail": scrub_secret(str(e)),
        }

    if not isinstance(parsed, dict):
        return {
            "ok": False,
            "failure_reason": "invalid_json",
            "detail": (
                "top-level JSON value is not an object"
            ),
        }

    return {
        "ok": True,
        "raw": parsed,
        "prompt": prompt,
    }