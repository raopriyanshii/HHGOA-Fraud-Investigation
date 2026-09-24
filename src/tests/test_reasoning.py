"""
G7 unit tests for src/agent/reasoning.py.

Every test here uses a fake `call_llm` -- none of this depends on a live
LLM provider (no default provider is even wired in; see
reasoning._default_call_llm). These test evidence-package curation,
prompt construction, and response parsing/failure-handling -- not the
semantic acceptance of an LLM's answer, which is
test_reasoning_validator.py's job.
"""
import asyncio
import json

from src.agent import reasoning


def run(coro):
    return asyncio.run(coro)


def _base_ledger(**overrides):
    ledger = {
        "investigation_id": "test-fixture",
        "trigger": {"type": "flagged_txn_id", "value": 100},
        "tool_calls": [],
        "uncertainties": [],
        "recommendation": "insufficient_evidence",
        "recommendation_basis": {"reasons": [], "tool_call_orders": []},
        "pattern_evidence": {"matched": False, "pattern": "card_testing", "candidates": [], "uncertainty": []},
        "full_evidence": {
            "combined_evidence": {
                "transaction": {"TransactionID": 100, "ts": "2016-01-01 10:00:00", "TransactionAmt": 50.0},
                "card": {"card_key": "C00001:1"},
                "customer": {"customer_id": "C00001"},
                "temporal_activity": [
                    {"TransactionID": 100, "ts": "2016-01-01 10:00:00", "TransactionAmt": 50.0, "channel": "online"},
                ],
                "device_evidence": {"device_profile": "SAMSUNG SM-G935F | Android 7.0", "hub_flag": False},
                "billing_region": {"addr1": 444.0, "hub_flag": True},
                "historical_cases": {"direct_cases": [{"case_id": "CC-1", "outcome": "confirmed_fraud"}], "connected_cases": []},
                "connected_cards": {"connected_via_case": [], "connected_via_device": []},
            },
            "cross_card_fraud_verification": None,
        },
    }
    ledger.update(overrides)
    return ledger


_BASE_G2_RESULT = {"recommended": [], "prohibited": [], "eligible_not_recommended": [], "deferred": []}


# --- known-id / evidence-package curation ---

def test_collect_known_ids_includes_flagged_and_temporal_txn_ids():
    ledger = _base_ledger()
    ledger["full_evidence"]["combined_evidence"]["temporal_activity"] = [
        {"TransactionID": 100, "ts": "2016-01-01 10:00:00", "TransactionAmt": 50.0, "channel": "online"},
        {"TransactionID": 101, "ts": "2016-01-01 10:05:00", "TransactionAmt": 5.0, "channel": "online"},
    ]
    known = reasoning._collect_known_ids(ledger)
    assert known["txn"] == [100, 101]


def test_collect_known_ids_never_includes_ids_not_present_in_evidence():
    ledger = _base_ledger()
    known = reasoning._collect_known_ids(ledger)
    # a plausible-looking but never-gathered id must not appear anywhere:
    assert "C99999:9" not in known["card"]
    assert "C99999" not in known["customer"]
    assert "FAKE-DEVICE-PROFILE" not in known["device"]
    assert 999999 not in known["txn"]


def test_collect_known_ids_includes_connected_and_cross_card_entities():
    ledger = _base_ledger()
    ledger["full_evidence"]["combined_evidence"]["connected_cards"] = {
        "connected_via_case": [{"connected_card_key": "C00002:1", "connected_customer_id": "C00002"}],
        "connected_via_device": [{"card_key": "C00003:1", "customer_id": "C00003"}],
    }
    ledger["full_evidence"]["cross_card_fraud_verification"] = {
        "via_device": {"other_cards": [{"card_key": "C00004:1", "customer_id": "C00004", "has_confirmed_fraud": True, "confirmed_fraud_case_ids": ["CC-9"]}]},
        "via_region": None,
    }
    known = reasoning._collect_known_ids(ledger)
    assert "C00002:1" in known["card"]  # connected_via_case: kept
    # G8 defect fix: connected_via_device is deliberately NOT walked here
    # any more -- it carries no fraud signal and no current output field
    # ever needs to cite one of these ids (see reasoning.py's own comment
    # at this call site); this was 795 raw entries for the real card that
    # broke the prompt's size.
    assert "C00003:1" not in known["card"]
    assert "C00004:1" in known["card"]  # cross_card, confirmed-fraud: kept
    assert "CC-9" in known["case"]


# --- G8 defect fix: compact cross-card/connected-cards evidence for the prompt ---
#
# Real HHG-011 broke this: 795 device-connected + 197 region-connected
# other_cards entries, none of it trimmed anywhere upstream (full_evidence
# is deliberately untrimmed for grounding/policy correctness), all
# serialized raw into the prompt -> a real Groq 400 context_length_exceeded
# error. These tests use a synthetic HHG-011-SHAPED dataset (same order of
# magnitude, deliberately scattered fraud-positive entries) to prove the
# fix without depending on live data.

def _synthetic_other_cards(total: int, fraud_positive_card_keys: set[str]) -> list[dict]:
    """A synthetic, HHG-011-shaped other_cards list: `total` entries, with
    fraud-positive ones deliberately scattered throughout (not clustered
    at the start), to prove no first-N truncation is happening.
    """
    cards = []
    for i in range(total):
        card_key = f"C{i:05d}:1"
        is_fraud = card_key in fraud_positive_card_keys
        cards.append({
            "card_key": card_key,
            "customer_id": f"C{i:05d}",
            "same_customer": False,
            "has_confirmed_fraud": is_fraud,
            "confirmed_fraud_case_ids": [f"CC-{i:04d}"] if is_fraud else [],
        })
    return cards


def _hhg011_shaped_cross_card(device_total=795, region_total=197, device_fraud_count=301, region_fraud_count=137):
    # matches the REAL HHG-011 counts (795/197 total, 301/137 confirmed-
    # fraud, measured live this session) by construction -- evenly spaced
    # indices scattered across the whole range (including near the END,
    # to catch any accidental first-N/last-N selection bug), sliced to
    # the exact target count:
    def _spaced_indices(total: int, count: int) -> set[int]:
        if count <= 0:
            return set()
        step = max(1, total // count)
        return set(list(range(0, total, step))[:count])

    device_fraud_keys = {f"C{i:05d}:1" for i in _spaced_indices(device_total, device_fraud_count)}
    region_fraud_keys = {f"C{i:05d}:1" for i in _spaced_indices(region_total, region_fraud_count)}
    return {
        "via_device": {"other_cards": _synthetic_other_cards(device_total, device_fraud_keys)},
        "via_region": {"other_cards": _synthetic_other_cards(region_total, region_fraud_keys)},
    }, device_fraud_keys, region_fraud_keys


def test_compact_cross_card_evidence_preserves_all_fraud_positive_cards():
    cross_card, device_fraud_keys, region_fraud_keys = _hhg011_shaped_cross_card()
    compact = reasoning._compact_cross_card_evidence(cross_card, device_hub_flag=False, region_hub_flag=None)

    got_device_keys = {c["card_key"] for c in compact["via_device"]["confirmed_fraud_cards"]}
    got_region_keys = {c["card_key"] for c in compact["via_region"]["confirmed_fraud_cards"]}
    assert got_device_keys == device_fraud_keys
    assert got_region_keys == region_fraud_keys
    # every fraud-positive card retains its grounded identity fields:
    sample = compact["via_device"]["confirmed_fraud_cards"][0]
    assert set(sample.keys()) == {"card_key", "customer_id", "same_customer", "confirmed_fraud_case_ids"}
    assert sample["confirmed_fraud_case_ids"]  # never empty for a fraud-positive entry here


def test_compact_cross_card_evidence_separates_device_and_region():
    cross_card, device_fraud_keys, region_fraud_keys = _hhg011_shaped_cross_card()
    compact = reasoning._compact_cross_card_evidence(cross_card, device_hub_flag=False, region_hub_flag=True)
    # no cross-contamination between the two paths:
    device_keys = {c["card_key"] for c in compact["via_device"]["confirmed_fraud_cards"]}
    region_keys = {c["card_key"] for c in compact["via_region"]["confirmed_fraud_cards"]}
    assert device_keys == device_fraud_keys
    assert region_keys == region_fraud_keys
    assert compact["via_device"]["hub_flag"] is False
    assert compact["via_region"]["hub_flag"] is True


def test_compact_cross_card_evidence_preserves_counts():
    cross_card, _, _ = _hhg011_shaped_cross_card(device_total=795, region_total=197)
    compact = reasoning._compact_cross_card_evidence(cross_card, device_hub_flag=False, region_hub_flag=False)
    assert compact["via_device"]["total_connected_card_count"] == 795
    assert compact["via_region"]["total_connected_card_count"] == 197


def test_compact_cross_card_evidence_preserves_hub_flags():
    cross_card, _, _ = _hhg011_shaped_cross_card(device_total=10, region_total=10, device_fraud_count=1, region_fraud_count=1)
    for device_flag, region_flag in ((True, True), (False, False), (None, None), (False, True)):
        compact = reasoning._compact_cross_card_evidence(cross_card, device_hub_flag=device_flag, region_hub_flag=region_flag)
        assert compact["via_device"]["hub_flag"] == device_flag
        assert compact["via_region"]["hub_flag"] == region_flag


def test_compact_cross_card_evidence_no_first_n_selection():
    # fraud-positive cards placed ONLY in the second half of a large list
    # -- a first-N-style bug would miss all of them.
    total = 1000
    late_fraud_keys = {f"C{i:05d}:1" for i in range(600, 620)}
    other_cards = _synthetic_other_cards(total, late_fraud_keys)
    cross_card = {"via_device": {"other_cards": other_cards}, "via_region": None}
    compact = reasoning._compact_cross_card_evidence(cross_card, device_hub_flag=False, region_hub_flag=None)
    got_keys = {c["card_key"] for c in compact["via_device"]["confirmed_fraud_cards"]}
    assert got_keys == late_fraud_keys
    assert compact["via_device"]["total_connected_card_count"] == 1000


def test_compact_cross_card_evidence_never_invents_a_fraud_record():
    other_cards = _synthetic_other_cards(50, fraud_positive_card_keys=set())  # zero fraud-positive
    cross_card = {"via_device": {"other_cards": other_cards}, "via_region": None}
    compact = reasoning._compact_cross_card_evidence(cross_card, device_hub_flag=False, region_hub_flag=None)
    assert compact["via_device"]["confirmed_fraud_cards"] == []
    assert compact["via_device"]["confirmed_fraud_card_count"] == 0
    assert compact["via_device"]["total_connected_card_count"] == 50


def test_compact_cross_card_evidence_none_input_stays_none():
    assert reasoning._compact_cross_card_evidence(None, device_hub_flag=None, region_hub_flag=None) is None


def test_compact_cross_card_evidence_small_list_behaves_identically_to_before():
    # existing small-fixture shape (<=5 entries) -- confirms backward
    # compatibility with every pre-existing test fixture in this suite.
    cross_card = {
        "via_device": {"other_cards": [
            {"card_key": "C00001:1", "customer_id": "C00001", "same_customer": False, "has_confirmed_fraud": True, "confirmed_fraud_case_ids": ["CC-1"]},
            {"card_key": "C00002:1", "customer_id": "C00002", "same_customer": False, "has_confirmed_fraud": False, "confirmed_fraud_case_ids": []},
        ]},
        "via_region": None,
    }
    compact = reasoning._compact_cross_card_evidence(cross_card, device_hub_flag=False, region_hub_flag=None)
    assert compact["via_device"]["total_connected_card_count"] == 2
    assert compact["via_device"]["confirmed_fraud_card_count"] == 1
    assert compact["via_device"]["confirmed_fraud_cards"][0]["card_key"] == "C00001:1"


def test_known_ids_from_cross_card_restricted_to_confirmed_fraud_only():
    ledger = _base_ledger()
    cross_card, device_fraud_keys, region_fraud_keys = _hhg011_shaped_cross_card(device_total=200, region_total=50, device_fraud_count=3, region_fraud_count=2)
    ledger["full_evidence"]["cross_card_fraud_verification"] = cross_card
    known = reasoning._collect_known_ids(ledger)
    # only the confirmed-fraud card_keys leak into known_ids, not all 250:
    assert device_fraud_keys | region_fraud_keys <= set(known["card"])
    assert len(known["card"]) < 250  # nowhere near the raw 200+250 total


def test_full_evidence_unchanged_after_build_evidence_package():
    ledger = _base_ledger()
    cross_card, _, _ = _hhg011_shaped_cross_card(device_total=100, region_total=50, device_fraud_count=2, region_fraud_count=2)
    ledger["full_evidence"]["cross_card_fraud_verification"] = cross_card
    import copy
    before = copy.deepcopy(ledger["full_evidence"])
    reasoning.build_evidence_package(ledger, _BASE_G2_RESULT)
    assert ledger["full_evidence"] == before  # completely untouched, still fully untrimmed


def test_prompt_size_bounded_for_hhg011_shaped_synthetic_evidence():
    # the actual regression test: build a synthetic evidence package at
    # real HHG-011 scale (795 device + 197 region other_cards) and prove
    # the resulting prompt is dramatically smaller than what naively
    # serializing every raw record would produce. The real failure was a
    # ~441,700-character prompt that Groq rejected outright.
    cross_card, _, _ = _hhg011_shaped_cross_card()  # 795 + 197, HHG-011 scale
    connected_via_device = [{"card_key": f"C{i:05d}:1", "customer_id": f"C{i:05d}"} for i in range(795)]
    temporal_activity = [
        {"TransactionID": 1000 + i, "ts": "2016-01-01 10:00:00", "TransactionAmt": 5.0, "channel": "online"}
        for i in range(52)  # HHG-011's real temporal_activity size
    ]

    ledger = _base_ledger()
    ledger["full_evidence"]["cross_card_fraud_verification"] = cross_card
    ledger["full_evidence"]["combined_evidence"]["connected_cards"] = {
        "connected_via_case": [], "connected_via_device": connected_via_device,
    }
    ledger["full_evidence"]["combined_evidence"]["temporal_activity"] = temporal_activity
    package = reasoning.build_evidence_package(ledger, _BASE_G2_RESULT)
    fixed_prompt = reasoning.build_prompt(package)

    # naive baseline: what the OLD (pre-fix) build_evidence_package would
    # have put in `facts` for the identical dataset -- every raw record,
    # uncompacted -- computed directly here rather than re-adding the old
    # code path, so this stays a true regression guard even if the old
    # behavior is ever fully forgotten.
    naive_facts = dict(package["facts"])
    naive_facts["connected_cards"] = ledger["full_evidence"]["combined_evidence"]["connected_cards"]
    naive_facts["cross_card_fraud_verification"] = cross_card
    naive_prompt_facts_len = len(json.dumps(naive_facts, indent=2, default=str))
    fixed_prompt_facts_len = len(json.dumps(package["facts"], indent=2, default=str))

    assert fixed_prompt_facts_len < naive_prompt_facts_len / 2  # at least a 50% reduction in the FACTS payload alone
    assert len(fixed_prompt) < 200_000  # well under the ~441,700 chars that broke the real call, even in this dense synthetic worst case
    # and correctness isn't sacrificed for size -- fraud-positive cards are still there:
    assert package["facts"]["cross_card_fraud_verification"]["via_device"]["confirmed_fraud_card_count"] > 0
    assert package["facts"]["cross_card_fraud_verification"]["via_device"]["total_connected_card_count"] == 795
    assert package["facts"]["connected_cards"] == {"connected_via_case_count": 0, "connected_via_device_count": 795}


def test_prompt_omits_customer_id_from_cross_card_fraud_cards_but_keeps_it_internally():
    # Part 3/4 (Gemini prompt-size fix): customer_id is always derivable
    # from card_key by this project's own card-identity convention
    # (card_key == f"{customer_id}:{card1}") and no output field ever
    # asks the LLM to cite one -- it is dropped from the RENDERED PROMPT
    # only, never from evidence_package["facts"] itself.
    cross_card = {
        "via_device": {"other_cards": [
            {"card_key": "C77777:1", "customer_id": "C77777-UNIQUE-MARKER", "same_customer": False, "has_confirmed_fraud": True, "confirmed_fraud_case_ids": ["CC-1"]},
        ]},
        "via_region": None,
    }
    ledger = _base_ledger()
    ledger["full_evidence"]["cross_card_fraud_verification"] = cross_card
    package = reasoning.build_evidence_package(ledger, _BASE_G2_RESULT)

    # internal contract: customer_id IS present in facts:
    assert package["facts"]["cross_card_fraud_verification"]["via_device"]["confirmed_fraud_cards"][0]["customer_id"] == "C77777-UNIQUE-MARKER"

    # rendered prompt: customer_id string is NOT present, but card_key
    # and the case id (the fields actually needed for grounding) are:
    prompt = reasoning.build_prompt(package)
    assert "C77777-UNIQUE-MARKER" not in prompt
    assert "C77777:1" in prompt
    assert "CC-1" in prompt


def test_prompt_known_ids_section_omits_non_txn_categories_but_package_stays_complete():
    # Part 3 (Gemini prompt-size fix): REASONING_OUTPUT_SCHEMA has no
    # card/customer/device/case/region output field at all, so those
    # categories are dropped from the RENDERED KNOWN IDS text -- but
    # evidence_package["known_ids"] itself must stay complete, since
    # reasoning_validator.py and other tests depend on the full contract.
    ledger = _base_ledger()
    ledger["full_evidence"]["cross_card_fraud_verification"] = {
        "via_device": {"other_cards": [
            {"card_key": "C88888:1", "customer_id": "C88888", "same_customer": False, "has_confirmed_fraud": True, "confirmed_fraud_case_ids": ["CC-UNIQUE-CASE-MARKER"]},
        ]},
        "via_region": None,
    }
    package = reasoning.build_evidence_package(ledger, _BASE_G2_RESULT)

    # internal contract: known_ids is still the complete, 6-category dict:
    assert set(package["known_ids"].keys()) == {"txn", "card", "customer", "device", "case", "region"}
    assert "C88888:1" in package["known_ids"]["card"]
    assert "CC-UNIQUE-CASE-MARKER" in package["known_ids"]["case"]

    # rendered prompt: the KNOWN IDS block only ever shows txn ids --
    # the card/case markers above must not leak into that section (they
    # may still legitimately appear elsewhere, e.g. inside FACTS, so this
    # checks the KNOWN IDS block specifically):
    prompt = reasoning.build_prompt(package)
    known_ids_block = prompt.split("KNOWN IDS")[1].split("You MUST NOT")[0]
    assert "C88888:1" not in known_ids_block
    assert "CC-UNIQUE-CASE-MARKER" not in known_ids_block
    assert "100" in known_ids_block  # the flagged txn id IS still rendered there


def test_build_evidence_package_has_four_required_sections():
    package = reasoning.build_evidence_package(_base_ledger(), _BASE_G2_RESULT)
    assert set(["flagged_txn_id", "known_ids", "facts", "deterministic_findings", "unknown"]) <= set(package.keys())
    assert package["flagged_txn_id"] == 100


def test_build_evidence_package_zero_side_effects_and_zero_tool_dependency():
    # pure function: same input twice -> identical output, no network/graph access
    ledger = _base_ledger()
    p1 = reasoning.build_evidence_package(ledger, _BASE_G2_RESULT)
    p2 = reasoning.build_evidence_package(ledger, _BASE_G2_RESULT)
    assert p1 == p2


# --- prompt construction (pure string building) ---

def test_prompt_contains_all_four_required_section_headers():
    package = reasoning.build_evidence_package(_base_ledger(), _BASE_G2_RESULT)
    prompt = reasoning.build_prompt(package)
    for header in ("FACTS:", "DETERMINISTIC FINDINGS:", "REASONING (what you are allowed to infer", "UNKNOWN"):
        assert header in prompt


def test_prompt_states_known_ids_are_the_only_allowed_ids():
    package = reasoning.build_evidence_package(_base_ledger(), _BASE_G2_RESULT)
    prompt = reasoning.build_prompt(package)
    assert "KNOWN IDS" in prompt
    assert "100" in prompt  # the flagged txn id must actually appear in the rendered known_ids block


def test_prompt_forbids_calling_tools_and_copying_risk_score():
    package = reasoning.build_evidence_package(_base_ledger(), _BASE_G2_RESULT)
    prompt = reasoning.build_prompt(package)
    assert "You MUST NOT" in prompt
    assert "call TigerGraph or any tool directly" in prompt
    assert "treat any risk_score as fraud_probability" in prompt


def test_prompt_r5_matched_instructs_forced_card_testing():
    ledger = _base_ledger(pattern_evidence={"matched": True, "pattern": "card_testing", "candidates": [{"authorization_txn_ids": [1, 2, 3], "larger_purchase_txn_id": 4}], "uncertainty": []})
    package = reasoning.build_evidence_package(ledger, _BASE_G2_RESULT)
    prompt = reasoning.build_prompt(package)
    assert '"matched": true' in prompt.lower()


# --- get_llm_reasoning: parsing and failure handling ---

def test_valid_json_response_is_parsed_successfully():
    async def fake_call_llm(prompt):
        return json.dumps({"verdict": "uncertain", "fraud_probability": 0.3, "affected_txn_ids": [], "first_suspicious_txn_id": None, "pattern": "none", "pattern_description": "", "summary": "ok", "reasoning": "ok", "uncertainties": []})

    package = reasoning.build_evidence_package(_base_ledger(), _BASE_G2_RESULT)
    outcome = run(reasoning.get_llm_reasoning(package, call_llm=fake_call_llm))
    assert outcome["ok"] is True
    assert outcome["raw"]["verdict"] == "uncertain"


def test_malformed_json_response_fails_safely():
    async def fake_call_llm(prompt):
        return "this is not JSON at all {"

    package = reasoning.build_evidence_package(_base_ledger(), _BASE_G2_RESULT)
    outcome = run(reasoning.get_llm_reasoning(package, call_llm=fake_call_llm))
    assert outcome["ok"] is False
    assert outcome["failure_reason"] == "invalid_json"


def test_non_object_json_response_fails_safely():
    async def fake_call_llm(prompt):
        return json.dumps([1, 2, 3])  # valid JSON, but not an object

    package = reasoning.build_evidence_package(_base_ledger(), _BASE_G2_RESULT)
    outcome = run(reasoning.get_llm_reasoning(package, call_llm=fake_call_llm))
    assert outcome["ok"] is False
    assert outcome["failure_reason"] == "invalid_json"


def test_llm_call_raising_a_non_timeout_error_is_caught_and_fails_safely():
    async def broken_call_llm(prompt):
        raise ConnectionError("LLM provider connection reset")

    package = reasoning.build_evidence_package(_base_ledger(), _BASE_G2_RESULT)
    outcome = run(reasoning.get_llm_reasoning(package, call_llm=broken_call_llm))
    assert outcome["ok"] is False
    assert outcome["failure_reason"] == "llm_call_failed"


def test_llm_call_raising_timeout_error_itself_is_classified_as_llm_timeout():
    # asyncio.TimeoutError IS TimeoutError in Python 3.11+, so a call_llm
    # that raises its own TimeoutError is indistinguishable from -- and
    # correctly classified the same as -- our own enforced timeout below.
    async def self_timing_out_call_llm(prompt):
        raise TimeoutError("LLM provider did not respond in time")

    package = reasoning.build_evidence_package(_base_ledger(), _BASE_G2_RESULT)
    outcome = run(reasoning.get_llm_reasoning(package, call_llm=self_timing_out_call_llm))
    assert outcome["ok"] is False
    assert outcome["failure_reason"] == "llm_timeout"


# --- G7 defect fix: enforced timeout on a genuinely hanging call_llm ---

def test_genuinely_hanging_call_llm_is_terminated_by_enforced_timeout():
    # A call_llm that never returns and never raises -- proves the
    # timeout is actually enforced (asyncio.wait_for), not merely
    # documented. A tiny timeout_s keeps this test fast; if the
    # enforcement were missing, this test would hang the suite instead
    # of failing cleanly.
    async def hanging_call_llm(prompt):
        await asyncio.sleep(60)  # far longer than the timeout_s below
        return "unreachable"

    package = reasoning.build_evidence_package(_base_ledger(), _BASE_G2_RESULT)
    outcome = run(reasoning.get_llm_reasoning(package, call_llm=hanging_call_llm, timeout_s=0.05))
    assert outcome["ok"] is False
    assert outcome["failure_reason"] == "llm_timeout"
    assert "0.05" in outcome["detail"]


def test_default_llm_timeout_is_explicit_and_configurable():
    assert reasoning.DEFAULT_LLM_TIMEOUT_S > 0
    # confirms the parameter is actually threaded through, not decorative:
    async def hanging_call_llm(prompt):
        await asyncio.sleep(60)
        return "unreachable"

    package = reasoning.build_evidence_package(_base_ledger(), _BASE_G2_RESULT)
    outcome = run(reasoning.get_llm_reasoning(package, call_llm=hanging_call_llm, timeout_s=0.05))
    assert outcome["failure_reason"] == "llm_timeout"


def test_default_call_llm_raises_clearly_when_no_provider_is_configured():
    package = reasoning.build_evidence_package(_base_ledger(), _BASE_G2_RESULT)
    outcome = run(reasoning.get_llm_reasoning(package))  # no call_llm passed
    assert outcome["ok"] is False
    assert outcome["failure_reason"] == "llm_call_failed"
    assert "No default LLM provider is configured" in outcome["detail"]


def test_call_llm_is_invoked_exactly_once_per_reasoning_call():
    calls = []

    async def counting_call_llm(prompt):
        calls.append(prompt)
        return json.dumps({"verdict": "uncertain", "fraud_probability": 0.2, "affected_txn_ids": [], "first_suspicious_txn_id": None, "pattern": "none", "pattern_description": "", "summary": "", "reasoning": "", "uncertainties": []})

    package = reasoning.build_evidence_package(_base_ledger(), _BASE_G2_RESULT)
    run(reasoning.get_llm_reasoning(package, call_llm=counting_call_llm))
    assert len(calls) == 1
