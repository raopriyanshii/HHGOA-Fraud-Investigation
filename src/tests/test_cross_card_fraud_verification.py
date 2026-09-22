"""
Tests for Q9 (src.investigation.queries.q9_cross_card_fraud_verification),
the G3 cross-card fraud-verification capability.

Two groups, matching this project's established split:

1. Live-connection tests (module-scoped `conn` fixture, same pattern as
   src/tests/test_investigation_queries.py) -- exercise the real GSQL
   against HHGOA_FraudInvestigation. These test structural correctness
   and self-consistency properties rather than hardcoded exact counts,
   because this is new code with no prior empirical baseline the way
   Q1-Q8 had from Phase E's own development -- asserting an exact
   discovered-card count here would be guessing, not verifying.

2. Offline tests of _extract_cross_card_results directly -- the
   same_customer/dedup/case-id logic is pure Python and doesn't need a
   live connection to verify precisely and deterministically, including
   the "same customer's own cards share a device" scenario (test matrix
   item 6), which is not something a real example was searched for live.
"""
import time

import pytest

from src.graph.tigergraph_connection import get_connection
from src.investigation import queries as q

CARD_KEY = "C12382:21139"  # HHG-001's card
CUSTOMER_ID = "C12382"
RING_CARD_KEY = "C03528:14407"  # confirmed real 22-card ring member
HIGH_CARDINALITY_CARD_KEY = "C11923:21363"  # 1,439 distinct device profiles (Phase E's N+1 discovery case)


@pytest.fixture(scope="module")
def conn():
    return get_connection(graphname="HHGOA_FraudInvestigation")


def _all_entries(result: dict) -> list:
    entries = []
    for path in ("via_device", "via_region"):
        if result[path] is not None:
            entries.extend(result[path]["other_cards"])
    return entries


# --- structural / shape ---

def test_output_shape_has_no_connected_via_case_or_device_keys(conn):
    result = q.q9_cross_card_fraud_verification(conn, RING_CARD_KEY)
    assert set(result.keys()) == {"via_device", "via_region"}
    assert "connected_via_case" not in result
    assert "connected_via_device" not in result


def test_each_returned_entry_has_the_five_required_fields(conn):
    result = q.q9_cross_card_fraud_verification(conn, RING_CARD_KEY)
    for entry in _all_entries(result):
        assert set(entry.keys()) == {"card_key", "customer_id", "same_customer", "has_confirmed_fraud", "confirmed_fraud_case_ids"}
        assert isinstance(entry["same_customer"], bool)
        assert isinstance(entry["has_confirmed_fraud"], bool)
        assert isinstance(entry["confirmed_fraud_case_ids"], list)


def test_input_card_never_appears_in_its_own_other_cards(conn):
    result = q.q9_cross_card_fraud_verification(conn, RING_CARD_KEY)
    for entry in _all_entries(result):
        assert entry["card_key"] != RING_CARD_KEY


# --- 2/3 (test matrix): malformed / nonexistent card -> both paths None, no raw exception ---

def test_malformed_card_key_returns_none_for_both_paths_not_error(conn):
    result = q.q9_cross_card_fraud_verification(conn, "this-card-key-does-not-exist")
    assert result == {"via_device": None, "via_region": None}


def test_empty_string_card_key_returns_none_for_both_paths_not_error(conn):
    result = q.q9_cross_card_fraud_verification(conn, "")
    assert result == {"via_device": None, "via_region": None}


# --- 1/4 (test matrix): non-hub device / region evidence, self-consistency ---

def test_same_customer_flag_matches_known_input_customer_id(conn):
    result = q.q9_cross_card_fraud_verification(conn, CARD_KEY)
    for entry in _all_entries(result):
        assert entry["same_customer"] == (entry["customer_id"] == CUSTOMER_ID)


def test_has_confirmed_fraud_is_consistent_with_case_ids_list(conn):
    # A property that must hold regardless of what the live data actually
    # contains: has_confirmed_fraud is true iff there's at least one case id.
    result = q.q9_cross_card_fraud_verification(conn, RING_CARD_KEY)
    for entry in _all_entries(result):
        assert entry["has_confirmed_fraud"] == (len(entry["confirmed_fraud_case_ids"]) > 0)


def test_ring_card_via_device_or_region_is_well_formed_if_present(conn):
    # RING_CARD_KEY is a confirmed real member of the 22-card device-sharing
    # ring (Phase E). This checks structural correctness of whatever the
    # live query actually returns -- not a hardcoded count, since no prior
    # run of this specific query exists to check a count against.
    result = q.q9_cross_card_fraud_verification(conn, RING_CARD_KEY)
    for path in ("via_device", "via_region"):
        if result[path] is not None:
            assert isinstance(result[path]["other_cards"], list)
            assert len(result[path]["other_cards"]) > 0
            keys = [e["card_key"] for e in result[path]["other_cards"]]
            assert len(keys) == len(set(keys))  # deduplicated


# --- 9 (test matrix): high-cardinality card, batched not N+1 ---

def test_high_cardinality_card_completes_quickly_batched_not_n_plus_1(conn):
    # C11923's original per-device-loop implementation of the *analogous*
    # Q5 traversal took 661.81s for this exact card (1,439 distinct device
    # profiles) before Phase E's batched-query fix. Q9 reuses that same
    # single-batched-traversal pattern from the start. A generous 30s bound
    # is far below what an N+1 pattern would take here and far above a
    # normal single-batched-call latency -- this guards against Q9
    # regressing into the same bug, not a tight performance benchmark.
    start = time.perf_counter()
    result = q.q9_cross_card_fraud_verification(conn, HIGH_CARDINALITY_CARD_KEY)
    elapsed = time.perf_counter() - start
    assert elapsed < 30.0, f"took {elapsed:.1f}s -- possible N+1 regression"
    assert set(result.keys()) == {"via_device", "via_region"}


# --- offline: pure logic tests of _extract_cross_card_results ---

def _raw_card(card_key, customer_id, fraud_case_ids=()):
    return {
        "attributes": {
            "card_key": card_key,
            "@customer_id_val": customer_id,
            "@confirmed_fraud_case_ids": list(fraud_case_ids),
        }
    }


def test_extract_same_customer_true_for_own_other_card_sharing_a_device():
    # Test matrix item 6: same customer's own cards sharing a device must
    # be flagged same_customer=True, not conflated with a cross-customer ring.
    raw = [_raw_card("C00001:2", "C00001")]
    out = q._extract_cross_card_results(raw, input_customer_id="C00001")
    assert out == [{
        "card_key": "C00001:2",
        "customer_id": "C00001",
        "same_customer": True,
        "has_confirmed_fraud": False,
        "confirmed_fraud_case_ids": [],
    }]


def test_extract_same_customer_false_for_different_customer_ring_member():
    raw = [_raw_card("C99999:1", "C99999", fraud_case_ids=["CC-1"])]
    out = q._extract_cross_card_results(raw, input_customer_id="C00001")
    assert out[0]["same_customer"] is False
    assert out[0]["has_confirmed_fraud"] is True
    assert out[0]["confirmed_fraud_case_ids"] == ["CC-1"]


def test_extract_deduplicates_by_card_key():
    raw = [
        _raw_card("C00002:1", "C00002"),
        _raw_card("C00002:1", "C00002"),  # e.g. reached via two different transactions
    ]
    out = q._extract_cross_card_results(raw, input_customer_id="C00001")
    assert len(out) == 1


def test_extract_sorts_confirmed_fraud_case_ids():
    raw = [_raw_card("C00003:1", "C00003", fraud_case_ids=["CC-9", "CC-1", "CC-5"])]
    out = q._extract_cross_card_results(raw, input_customer_id="C00001")
    assert out[0]["confirmed_fraud_case_ids"] == ["CC-1", "CC-5", "CC-9"]


def test_extract_returns_empty_list_for_no_other_cards():
    assert q._extract_cross_card_results([], input_customer_id="C00001") == []
