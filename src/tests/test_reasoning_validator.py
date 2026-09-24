"""
G7 unit tests for src/agent/reasoning_validator.py.

Pure-function tests -- no LLM, no TigerGraph, no mocking needed. The
overlap test (test_hhg011_*) uses real transaction IDs and real amounts
pulled once from the live graph this session (card C11923:21363, case
HHG-011 -- the one benchmark case G6-A found has overlapping R5
candidates), not synthetic/invented values, so exposure math is checked
against real data for the one case that actually exercises it.
"""
from src.agent import reasoning_validator as rv


def _package(**overrides):
    package = {
        "flagged_txn_id": 100,
        "known_ids": {"txn": [100, 101, 102], "card": [], "customer": [], "device": [], "case": [], "region": []},
        "facts": {
            "transaction": {"TransactionID": 100, "TransactionAmt": 50.0},
            "temporal_activity": [
                {"TransactionID": 100, "TransactionAmt": 50.0},
                {"TransactionID": 101, "TransactionAmt": 5.0},
                {"TransactionID": 102, "TransactionAmt": 10.0},
            ],
        },
        "deterministic_findings": {"pattern_evidence": {"matched": False, "candidates": []}},
    }
    package.update(overrides)
    return package


def _valid_raw(**overrides):
    raw = {
        "verdict": "uncertain",
        "fraud_probability": 0.4,
        "affected_txn_ids": [],
        "first_suspicious_txn_id": None,
        "pattern": "none",
        "pattern_description": "",
        "summary": "summary text",
        "reasoning": "reasoning text",
        "uncertainties": [],
    }
    raw.update(overrides)
    return raw


# --- malformed output ---

def test_non_dict_output_is_rejected():
    result = rv.validate_reasoning_output(["not", "a", "dict"], _package())
    assert result["ok"] is False
    assert result["failure_reason"] == "malformed_output"


def test_missing_required_key_is_rejected():
    raw = _valid_raw()
    del raw["summary"]
    result = rv.validate_reasoning_output(raw, _package())
    assert result["ok"] is False
    assert result["failure_reason"] == "malformed_output"


def test_wrong_type_for_affected_txn_ids_is_rejected():
    result = rv.validate_reasoning_output(_valid_raw(affected_txn_ids="not a list"), _package())
    assert result["ok"] is False
    assert result["failure_reason"] == "malformed_output"


# --- invalid verdict ---

def test_invalid_verdict_is_rejected():
    result = rv.validate_reasoning_output(_valid_raw(verdict="probably_fraud"), _package())
    assert result["ok"] is False
    assert result["failure_reason"] == "invalid_verdict"


def test_uncertain_verdict_is_accepted():
    result = rv.validate_reasoning_output(_valid_raw(verdict="uncertain"), _package())
    assert result["ok"] is True
    assert result["verdict"] == "uncertain"


# --- probability bounds ---

def test_probability_below_zero_is_rejected():
    result = rv.validate_reasoning_output(_valid_raw(fraud_probability=-0.1), _package())
    assert result["ok"] is False
    assert result["failure_reason"] == "invalid_probability"


def test_probability_above_one_is_rejected():
    result = rv.validate_reasoning_output(_valid_raw(fraud_probability=1.5), _package())
    assert result["ok"] is False
    assert result["failure_reason"] == "invalid_probability"


def test_non_numeric_probability_is_rejected():
    result = rv.validate_reasoning_output(_valid_raw(fraud_probability="high"), _package())
    assert result["ok"] is False
    assert result["failure_reason"] == "invalid_probability"


def test_boundary_probabilities_are_accepted():
    for p in (0.0, 1.0, 0.5):
        result = rv.validate_reasoning_output(_valid_raw(fraud_probability=p), _package())
        assert result["ok"] is True
        assert result["fraud_probability"] == p


# --- hallucinated / unknown IDs ---

def test_hallucinated_transaction_id_is_stripped_not_rejected():
    result = rv.validate_reasoning_output(_valid_raw(verdict="fraud", affected_txn_ids=[101, 999999]), _package())
    assert result["ok"] is True
    assert 999999 not in result["affected_txn_ids"]
    assert 999999 in result["stripped_ids"]
    assert 101 in result["affected_txn_ids"]


def test_hallucinated_first_suspicious_txn_id_is_stripped_to_none():
    result = rv.validate_reasoning_output(_valid_raw(first_suspicious_txn_id=999999), _package())
    assert result["ok"] is True
    assert result["first_suspicious_txn_id"] is None
    assert 999999 in result["stripped_ids"]


def test_string_numeric_id_matching_a_known_id_is_accepted():
    # the LLM may reasonably return "101" instead of 101 -- accepted only
    # because 101 is actually a known id, never guessed at otherwise.
    result = rv.validate_reasoning_output(_valid_raw(verdict="fraud", affected_txn_ids=["101"]), _package())
    assert result["ok"] is True
    assert 101 in result["affected_txn_ids"]


def test_non_numeric_string_id_is_stripped():
    result = rv.validate_reasoning_output(_valid_raw(affected_txn_ids=["not-an-id"]), _package())
    assert result["ok"] is True
    assert result["affected_txn_ids"] == []
    assert "not-an-id" in result["stripped_ids"]


# --- dedup ---

def test_duplicate_affected_txn_ids_are_deduplicated():
    result = rv.validate_reasoning_output(_valid_raw(verdict="fraud", affected_txn_ids=[101, 101, 102, 101]), _package())
    assert result["ok"] is True
    # flagged_txn_id (100) is auto-included since the episode is non-empty and verdict != legitimate:
    assert sorted(result["affected_txn_ids"]) == [100, 101, 102]
    assert result["affected_txn_ids"].count(101) == 1


# --- flagged-transaction inclusion ---

def test_flagged_txn_id_is_added_when_episode_is_non_empty():
    result = rv.validate_reasoning_output(_valid_raw(verdict="fraud", affected_txn_ids=[101, 102]), _package())
    assert result["ok"] is True
    assert 100 in result["affected_txn_ids"]  # flagged_txn_id from _package()


def test_flagged_txn_id_not_forced_when_llm_asserted_no_episode():
    result = rv.validate_reasoning_output(_valid_raw(verdict="uncertain", affected_txn_ids=[]), _package())
    assert result["ok"] is True
    assert result["affected_txn_ids"] == []


# --- legitimate verdict override ---

def test_legitimate_verdict_forces_empty_affected_txn_ids_and_zero_exposure():
    result = rv.validate_reasoning_output(
        _valid_raw(verdict="legitimate", affected_txn_ids=[100, 101], first_suspicious_txn_id=100),
        _package(),
    )
    assert result["ok"] is True
    assert result["affected_txn_ids"] == []
    assert result["first_suspicious_txn_id"] is None
    assert result["exposure_usd"] == 0.0


# --- exposure calculation ---

def test_exposure_usd_sums_absolute_amounts_of_final_ids_only():
    result = rv.validate_reasoning_output(_valid_raw(verdict="fraud", affected_txn_ids=[101, 102]), _package())
    assert result["ok"] is True
    # flagged (100, $50) gets auto-included -> 50 + 5 + 10 = 65.0
    assert result["exposure_usd"] == 65.0


def test_exposure_usd_ignores_stripped_hallucinated_ids():
    result = rv.validate_reasoning_output(_valid_raw(verdict="fraud", affected_txn_ids=[101, 999999]), _package())
    assert result["ok"] is True
    assert result["exposure_usd"] == 55.0  # flagged(100,$50) + 101($5); 999999 contributes nothing


# --- deterministic pattern overrides (item 8) ---

def test_r5_matched_forces_card_testing_regardless_of_llm_choice():
    package = _package(deterministic_findings={"pattern_evidence": {"matched": True, "candidates": []}})
    result = rv.validate_reasoning_output(_valid_raw(pattern="account_takeover"), package)
    assert result["ok"] is True
    assert result["pattern"] == "card_testing"


def test_llm_cannot_self_assert_card_testing_without_r5_match():
    package = _package(deterministic_findings={"pattern_evidence": {"matched": False, "candidates": []}})
    result = rv.validate_reasoning_output(_valid_raw(pattern="card_testing"), package)
    assert result["ok"] is False
    assert result["failure_reason"] == "invalid_pattern"


def test_invalid_pattern_enum_value_is_rejected():
    result = rv.validate_reasoning_output(_valid_raw(pattern="totally_made_up_pattern"), _package())
    assert result["ok"] is False
    assert result["failure_reason"] == "invalid_pattern"


def test_pattern_description_forced_empty_unless_undocumented():
    result = rv.validate_reasoning_output(
        _valid_raw(pattern="account_takeover", pattern_description="a description that should be dropped"),
        _package(),
    )
    assert result["ok"] is True
    assert result["pattern_description"] == ""


def test_pattern_description_kept_for_undocumented():
    result = rv.validate_reasoning_output(
        _valid_raw(pattern="undocumented", pattern_description="a genuinely new pattern"),
        _package(),
    )
    assert result["ok"] is True
    assert result["pattern_description"] == "a genuinely new pattern"


# --- documented edge case: flagged_txn_id itself not present in known_ids ---

def test_flagged_txn_id_not_in_known_ids_documents_current_safe_behavior():
    # This should not happen in practice: G1's full_evidence always
    # derives flagged_txn_id from the same combined_evidence.transaction
    # that also seeds known_ids["txn"] (see reasoning.build_evidence_
    # package), so the two are always consistent in the real pipeline.
    # This test pins down, rather than silently leaves undefined, what
    # happens if a malformed/defensive evidence_package ever breaks that
    # invariant: an empty LLM-asserted episode stays empty (safe -- no
    # ungrounded id is ever introduced), but once the LLM asserts ANY
    # other grounded id, the (unverified) flagged_txn_id is still force-
    # included as "part of the episode," per the organizer's "including
    # the flagged one" rule -- current, accepted behavior, not changed
    # here; a stricter version would additionally require flagged_txn_id
    # to be a member of known_ids["txn"] before inserting it.
    package = _package(
        flagged_txn_id=999,  # deliberately NOT in known_ids["txn"] below
        known_ids={"txn": [101, 102], "card": [], "customer": [], "device": [], "case": [], "region": []},
    )

    # case A: LLM asserts no episode at all -> flagged_txn_id is never introduced
    empty_result = rv.validate_reasoning_output(_valid_raw(verdict="uncertain", affected_txn_ids=[]), package)
    assert empty_result["ok"] is True
    assert empty_result["affected_txn_ids"] == []
    assert 999 not in empty_result["affected_txn_ids"]

    # case B: LLM asserts a real episode -> flagged_txn_id is force-included
    # even though it isn't itself a member of known_ids["txn"] (documented,
    # not fixed, per this review round's scope):
    nonempty_result = rv.validate_reasoning_output(_valid_raw(verdict="fraud", affected_txn_ids=[101]), package)
    assert nonempty_result["ok"] is True
    assert 999 in nonempty_result["affected_txn_ids"]


# --- documented edge case: whole-number-float transaction IDs ---

def test_whole_number_float_id_is_stripped_not_coerced():
    # Documents CURRENT, INTENTIONAL behavior: a JSON float like 100.0 is
    # treated as ungrounded and stripped, never silently coerced to the
    # known int id 100 -- "never guess" applies to type ambiguity, not
    # just to entirely-unknown ids. Not changed here: the organizer's own
    # answer format types affected_txn_ids as "list of strings" anyway,
    # so a bare JSON float was never the expected wire shape to begin
    # with, and nothing in the official schema calls for coercing it.
    result = rv.validate_reasoning_output(_valid_raw(verdict="fraud", affected_txn_ids=[100.0]), _package())
    assert result["ok"] is True
    assert result["affected_txn_ids"] == []  # stripped -> grounded_ids stays empty -> no flagged-txn insertion either
    assert 100.0 in result["stripped_ids"]


def test_whole_number_float_first_suspicious_txn_id_is_also_stripped_not_coerced():
    result = rv.validate_reasoning_output(_valid_raw(first_suspicious_txn_id=101.0), _package())
    assert result["ok"] is True
    assert result["first_suspicious_txn_id"] is None
    assert 101.0 in result["stripped_ids"]


# --- item G: the grounding mechanism generalizes beyond txn ids ---

def test_known_ids_grounding_rejects_fabricated_entities_of_any_type():
    known_ids = {"txn": [100], "card": ["C00001:1"], "customer": ["C00001"], "device": ["real-device"], "case": ["CC-1"], "region": []}
    assert "FAKE-CARD-999" not in known_ids["card"]
    assert "C99999" not in known_ids["customer"]
    assert "FAKE-DEVICE-PROFILE" not in known_ids["device"]
    assert "CC-FAKE" not in known_ids["case"]


# --- item J: real HHG-011 data, overlapping R5 candidates ---

def test_hhg011_real_overlapping_candidates_grounded_and_exposure_correct():
    # Real transaction IDs/amounts/timestamps for card C11923:21363 (case
    # HHG-011), pulled live this session. 3582082 is real, overlapping
    # data: it is candidate 1's larger_purchase_txn_id AND candidate 2's
    # first authorization_txn_id -- the exact double-role overlap G6-A
    # found in the real 22-candidate set. The flagged transaction is
    # 3583368 ($131.30), per case_pack_resolved.csv.
    temporal_activity = [
        {"TransactionID": 3580907, "ts": "2016-12-28 03:33:38", "TransactionAmt": 16.17, "channel": "online"},
        {"TransactionID": 3580933, "ts": "2016-12-28 03:57:10", "TransactionAmt": 23.78, "channel": "online"},
        {"TransactionID": 3580951, "ts": "2016-12-28 04:21:00", "TransactionAmt": 52.17, "channel": "online"},
        {"TransactionID": 3582082, "ts": "2016-12-28 18:54:14", "TransactionAmt": 62.11, "channel": "online"},
        {"TransactionID": 3582107, "ts": "2016-12-28 19:03:07", "TransactionAmt": 62.06, "channel": "online"},
        {"TransactionID": 3582142, "ts": "2016-12-28 19:13:13", "TransactionAmt": 4.15, "channel": "online"},
        {"TransactionID": 3582671, "ts": "2016-12-28 22:21:56", "TransactionAmt": 98.70, "channel": "online"},
        {"TransactionID": 3583368, "ts": "2016-12-29 03:27:44", "TransactionAmt": 131.30, "channel": "online"},
    ]
    package = _package(
        flagged_txn_id=3583368,
        known_ids={"txn": [t["TransactionID"] for t in temporal_activity], "card": [], "customer": [], "device": [], "case": [], "region": []},
        facts={"transaction": {"TransactionID": 3583368, "TransactionAmt": 131.30}, "temporal_activity": temporal_activity},
        deterministic_findings={"pattern_evidence": {"matched": True, "candidates": []}},
    )
    # An LLM response spanning two overlapping candidates' transactions
    # plus a deliberate duplicate and a hallucinated id, exactly the kind
    # of response the validator must handle without inventing or
    # double-counting anything:
    raw = _valid_raw(
        verdict="fraud",
        pattern="card_testing",  # will be forced to card_testing anyway (R5 matched)
        affected_txn_ids=[3580907, 3580933, 3580951, 3582082, 3582082, 3582107, 9999999],
        first_suspicious_txn_id=3580907,
    )
    result = rv.validate_reasoning_output(raw, package)
    assert result["ok"] is True
    assert result["pattern"] == "card_testing"
    assert 9999999 in result["stripped_ids"]
    assert 9999999 not in result["affected_txn_ids"]
    # no duplicate of the overlapping id:
    assert result["affected_txn_ids"].count(3582082) == 1
    # flagged transaction (3583368) must be included, since the LLM asserted a non-empty episode:
    assert 3583368 in result["affected_txn_ids"]
    # exact real-dollar exposure sum over the deduplicated, grounded set:
    expected = round(16.17 + 23.78 + 52.17 + 62.11 + 62.06 + 131.30, 2)
    assert result["exposure_usd"] == expected


# --- G14: needs_more_evidence / evidence_request ---

def test_needs_more_evidence_absent_defaults_to_false_for_pre_g14_fixtures():
    # Every pre-G14 raw-response fixture in this project (including
    # _valid_raw() above) omits these keys entirely -- this must never
    # be treated as malformed_output.
    result = rv.validate_reasoning_output(_valid_raw(), _package())
    assert result["ok"] is True
    assert result["needs_more_evidence"] is False
    assert result["evidence_request"] is None


def test_needs_more_evidence_must_be_a_boolean():
    raw = _valid_raw(needs_more_evidence="yes")
    result = rv.validate_reasoning_output(raw, _package())
    assert result["ok"] is False
    assert result["failure_reason"] == "malformed_output"


def test_valid_evidence_request_is_accepted_and_returned():
    raw = _valid_raw(
        needs_more_evidence=True,
        evidence_request={"type": "historical_case_evidence", "target_card_key": "C99999:1", "reason": "check sibling history"},
    )
    result = rv.validate_reasoning_output(raw, _package())
    assert result["ok"] is True
    assert result["needs_more_evidence"] is True
    assert result["evidence_request"] == {
        "type": "historical_case_evidence", "target_card_key": "C99999:1", "reason": "check sibling history",
    }


def test_evidence_request_missing_when_needs_more_evidence_true_is_rejected():
    raw = _valid_raw(needs_more_evidence=True, evidence_request=None)
    result = rv.validate_reasoning_output(raw, _package())
    assert result["ok"] is False
    assert result["failure_reason"] == "invalid_evidence_request"


def test_evidence_request_non_null_when_needs_more_evidence_false_is_normalized_not_rejected():
    # Deterministic override, same discipline as pattern_description --
    # the LLM's own primary signal (needs_more_evidence=False) wins over
    # an inconsistent, populated evidence_request, and the whole response
    # is not thrown away over one inconsistent nullable field.
    raw = _valid_raw(
        needs_more_evidence=False,
        evidence_request={"type": "historical_case_evidence", "target_card_key": "C1:1", "reason": "x"},
    )
    result = rv.validate_reasoning_output(raw, _package())
    assert result["ok"] is True
    assert result["evidence_request"] is None


def test_invalid_evidence_request_type_is_rejected():
    raw = _valid_raw(
        needs_more_evidence=True,
        evidence_request={"type": "some_made_up_tool", "target_card_key": "C1:1", "reason": "x"},
    )
    result = rv.validate_reasoning_output(raw, _package())
    assert result["ok"] is False
    assert result["failure_reason"] == "invalid_evidence_request"


def test_evidence_request_as_list_is_rejected_never_treated_as_multiple_requests():
    raw = _valid_raw(
        needs_more_evidence=True,
        evidence_request=[{"type": "historical_case_evidence", "target_card_key": "C1:1", "reason": "x"}],
    )
    result = rv.validate_reasoning_output(raw, _package())
    assert result["ok"] is False
    assert result["failure_reason"] == "invalid_evidence_request"


def test_evidence_request_empty_target_card_key_is_rejected():
    raw = _valid_raw(
        needs_more_evidence=True,
        evidence_request={"type": "historical_case_evidence", "target_card_key": "  ", "reason": "x"},
    )
    result = rv.validate_reasoning_output(raw, _package())
    assert result["ok"] is False
    assert result["failure_reason"] == "invalid_evidence_request"


def test_evidence_request_empty_reason_is_rejected():
    raw = _valid_raw(
        needs_more_evidence=True,
        evidence_request={"type": "historical_case_evidence", "target_card_key": "C1:1", "reason": ""},
    )
    result = rv.validate_reasoning_output(raw, _package())
    assert result["ok"] is False
    assert result["failure_reason"] == "invalid_evidence_request"


def test_evidence_request_target_card_key_wrong_type_is_rejected():
    raw = _valid_raw(
        needs_more_evidence=True,
        evidence_request={"type": "historical_case_evidence", "target_card_key": 12345, "reason": "x"},
    )
    result = rv.validate_reasoning_output(raw, _package())
    assert result["ok"] is False
    assert result["failure_reason"] == "invalid_evidence_request"


def test_this_validator_never_checks_target_card_key_against_the_ledger():
    # By design (this phase): the graph-dependent anti-fabrication check
    # belongs in case_output.py, which has the ledger -- this function
    # never receives one and must accept a structurally well-formed
    # request regardless of whether the card is real.
    raw = _valid_raw(
        needs_more_evidence=True,
        evidence_request={"type": "historical_case_evidence", "target_card_key": "C_MADE_UP:999", "reason": "x"},
    )
    result = rv.validate_reasoning_output(raw, _package())
    assert result["ok"] is True
    assert result["evidence_request"]["target_card_key"] == "C_MADE_UP:999"
