"""
G7 -- deterministic validator for the LLM reasoning layer's output.

Nothing here calls an LLM or TigerGraph. The LLM's raw JSON output
(src.agent.reasoning.get_llm_reasoning's "raw" value) is never trusted
directly: every entity ID it returns is checked against the evidence
package's known_ids (built by src.agent.reasoning from the already-
gathered, untrimmed ledger) before being accepted. An ID that isn't
grounded is stripped from the result -- never silently swapped for
another ID, and never re-verified against TigerGraph (that would be a
new, ungoverned tool call outside G1's bounded ≤5-call workflow, which
this module has no access to make in the first place).

Two categories of deterministic override live here, both because the
LLM must never be allowed to rewrite a fact code has already
established, in either direction:
  - pattern_evidence.matched == True  -> pattern is forced to
    "card_testing", regardless of what the LLM said.
  - pattern_evidence.matched == False -> the LLM asserting "card_testing"
    anyway is rejected as invalid, not silently accepted or downgraded.
"""
from __future__ import annotations

_VALID_VERDICTS = ("fraud", "legitimate", "uncertain")
_VALID_PATTERNS = (
    "card_testing", "card_not_present_fraud", "card_not_present_new_device",
    "out_of_region_use", "account_takeover", "undocumented", "none",
)
_REQUIRED_KEYS = (
    "verdict", "fraud_probability", "affected_txn_ids", "first_suspicious_txn_id",
    "pattern", "pattern_description", "summary", "reasoning", "uncertainties",
)


def _fail(reason: str, detail: str = "") -> dict:
    return {"ok": False, "failure_reason": reason, "detail": detail}


def _coerce_txn_id(value: object) -> int | None:
    """A real transaction id is always an int (TransactionID, UINT in
    GSQL). The LLM may return it as an int or a numeric string -- both
    are accepted only if the resulting int is actually in known_ids;
    anything else (a non-numeric string, a float, a made-up id, a bool)
    is rejected outright, never guessed at.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    return None


def validate_reasoning_output(raw: object, evidence_package: dict) -> dict:
    """Returns {"ok": True, ...validated fields...} on acceptance, or
    {"ok": False, "failure_reason": ..., "detail": ...} on rejection.
    Rejection means the caller must produce an explicit stop/uncertainty
    state -- never fabricate a replacement answer.
    """
    if not isinstance(raw, dict):
        return _fail("malformed_output", "top-level value is not a JSON object")

    missing = [k for k in _REQUIRED_KEYS if k not in raw]
    if missing:
        return _fail("malformed_output", f"missing required keys: {missing}")

    verdict = raw["verdict"]
    if verdict not in _VALID_VERDICTS:
        return _fail("invalid_verdict", repr(verdict))

    probability = raw["fraud_probability"]
    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        return _fail("invalid_probability", repr(probability))
    probability = float(probability)
    if not (0.0 <= probability <= 1.0):
        return _fail("invalid_probability", repr(probability))

    pattern_evidence = (evidence_package.get("deterministic_findings") or {}).get("pattern_evidence") or {}
    r5_matched = bool(pattern_evidence.get("matched"))

    raw_pattern = raw["pattern"]
    if r5_matched:
        # Deterministic override: G5-A already established this pattern.
        # The LLM's own opinion is discarded entirely here, never merged
        # or compared -- a positive R5 match always wins.
        pattern = "card_testing"
    else:
        if raw_pattern not in _VALID_PATTERNS:
            return _fail("invalid_pattern", repr(raw_pattern))
        if raw_pattern == "card_testing":
            # The LLM asserting the one pattern this project only ever
            # establishes deterministically (G5-A), when G5-A itself
            # found no match, is the same violation in the other
            # direction -- inventing a deterministic-category finding.
            return _fail("invalid_pattern", "LLM asserted card_testing without a G5-A match")
        pattern = raw_pattern

    pattern_description = raw["pattern_description"]
    if not isinstance(pattern_description, str):
        return _fail("malformed_output", "pattern_description must be a string")
    # Deterministic override: only meaningful (and only required) for
    # "undocumented", per the organizer's own field definition
    # ("Otherwise \"\""); forced empty for every other pattern regardless
    # of what the LLM wrote.
    if pattern != "undocumented":
        pattern_description = ""

    summary = raw["summary"]
    reasoning_text = raw["reasoning"]
    if not isinstance(summary, str):
        return _fail("malformed_output", "summary must be a string")
    if not isinstance(reasoning_text, str):
        return _fail("malformed_output", "reasoning must be a string")

    uncertainties = raw["uncertainties"]
    if not isinstance(uncertainties, list) or not all(isinstance(u, str) for u in uncertainties):
        uncertainties = []  # advisory only, never load-bearing for acceptance

    known_ids = evidence_package.get("known_ids") or {}
    known_txn_ids = set(known_ids.get("txn") or [])
    flagged_txn_id = evidence_package.get("flagged_txn_id")

    raw_affected = raw["affected_txn_ids"]
    if not isinstance(raw_affected, list):
        return _fail("malformed_output", "affected_txn_ids must be a list")

    grounded_ids: list[int] = []
    stripped_ids: list = []
    seen: set[int] = set()
    for item in raw_affected:
        coerced = _coerce_txn_id(item)
        if coerced is not None and coerced in known_txn_ids:
            if coerced not in seen:
                seen.add(coerced)
                grounded_ids.append(coerced)
        else:
            stripped_ids.append(item)

    raw_first = raw["first_suspicious_txn_id"]
    first_suspicious_txn_id: int | None = None
    if raw_first is not None:
        coerced_first = _coerce_txn_id(raw_first)
        if coerced_first is not None and coerced_first in known_txn_ids:
            first_suspicious_txn_id = coerced_first
            if first_suspicious_txn_id not in seen:
                seen.add(first_suspicious_txn_id)
                grounded_ids.append(first_suspicious_txn_id)
        else:
            stripped_ids.append(raw_first)

    # Organizer's own field definition: affected_txn_ids includes the
    # flagged transaction "including the flagged one" -- enforced only
    # once the LLM has already asserted SOME episode membership (a
    # non-empty grounded list); an empty list stays empty rather than
    # manufacturing episode membership the LLM never claimed.
    if grounded_ids and verdict != "legitimate" and flagged_txn_id is not None and flagged_txn_id not in seen:
        grounded_ids.insert(0, flagged_txn_id)
        seen.add(flagged_txn_id)

    # Deterministic override for a legitimate verdict (organizer Notes
    # section, stated exactly this way): affected_txn_ids empty,
    # exposure_usd 0 -- regardless of what the LLM proposed.
    if verdict == "legitimate":
        grounded_ids = []
        first_suspicious_txn_id = None

    facts = evidence_package.get("facts") or {}
    amount_by_txn_id: dict[int, float] = {}
    txn = facts.get("transaction")
    if txn and txn.get("TransactionID") is not None:
        amount_by_txn_id[txn["TransactionID"]] = txn.get("TransactionAmt")
    for t in facts.get("temporal_activity") or []:
        if t.get("TransactionID") is not None:
            amount_by_txn_id[t["TransactionID"]] = t.get("TransactionAmt")

    # exposure_usd is computed here, deterministically, from the final
    # validated ID list -- the LLM's own fraud_probability/verdict/etc.
    # never determine this number; it is a pure sum-of-absolute-amounts
    # over IDs that have already survived grounding.
    exposure_usd = round(
        sum(abs(amount_by_txn_id[tid]) for tid in grounded_ids if amount_by_txn_id.get(tid) is not None),
        2,
    )

    return {
        "ok": True,
        "verdict": verdict,
        "fraud_probability": probability,
        "pattern": pattern,
        "pattern_description": pattern_description,
        "affected_txn_ids": grounded_ids,
        "first_suspicious_txn_id": first_suspicious_txn_id,
        "exposure_usd": exposure_usd,
        "summary": summary,
        "reasoning": reasoning_text,
        "uncertainties": uncertainties,
        "stripped_ids": stripped_ids,
    }
