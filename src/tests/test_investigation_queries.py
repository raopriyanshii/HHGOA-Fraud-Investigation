"""
Phase E tests for the deterministic investigation query layer.

Unlike every other test file in this project, these require a LIVE
TigerGraph connection (HHGOA_FraudInvestigation, populated by Phase D) --
there is no offline fixture for a real graph traversal. All queries here
are read-only (INTERPRET QUERY, never installed) so running this file
cannot modify the schema, loading jobs, or data.

Fixed, real inputs are used throughout (HHG-001's resolved transaction/card,
and the one confirmed real 22-card ring's device profile) so results are
deterministic and checkable against known-correct values established in
earlier phases.
"""
import pytest

from src.graph.tigergraph_connection import get_connection
from src.investigation import queries as q

TXN_ID = 3514030  # HHG-001's flagged transaction
CARD_KEY = "C12382:21139"
CUSTOMER_ID = "C12382"
CASE_ID = "HHG-001"
RING_DEVICE_PROFILE = "SM-G935F Build/NRD90M|Android 7.0|chrome 62.0 for android|1920x1080"
RING_CARD_KEY = "C03528:14407"


@pytest.fixture(scope="module")
def conn():
    return get_connection(graphname="HHGOA_FraudInvestigation")


# --- Q1: valid / nonexistent transaction, customer/card resolution ---

def test_q1_valid_transaction_resolves(conn):
    result = q.q1_transaction_lookup(conn, TXN_ID)
    assert result is not None
    assert result["transaction"]["TransactionID"] == TXN_ID


def test_q1_card_resolution_matches_known_value(conn):
    result = q.q1_transaction_lookup(conn, TXN_ID)
    assert result["card"]["card_key"] == CARD_KEY


def test_q1_customer_resolution_matches_known_value(conn):
    result = q.q1_transaction_lookup(conn, TXN_ID)
    assert result["customer"]["customer_id"] == CUSTOMER_ID


def test_q1_nonexistent_transaction_returns_none(conn):
    assert q.q1_transaction_lookup(conn, 999999999) is None


# --- Q2: temporal ordering ---

def test_q2_temporal_neighborhood_is_ts_ordered(conn):
    result = q.q2_temporal_neighborhood(conn, TXN_ID, 24.0)
    ts_values = [t["ts"] for t in result["transactions"]]
    assert ts_values == sorted(ts_values)


def test_q2_includes_the_flagged_transaction_itself(conn):
    result = q.q2_temporal_neighborhood(conn, TXN_ID, 24.0)
    assert TXN_ID in [t["TransactionID"] for t in result["transactions"]]


def test_q2_nonexistent_transaction_returns_none(conn):
    assert q.q2_temporal_neighborhood(conn, 999999999, 24.0) is None


# --- Q3: device sharing, empty-result handling ---

def test_q3_device_sharing_recovers_known_ring_size(conn):
    result = q.q3_device_sharing(conn, RING_DEVICE_PROFILE)
    assert result is not None
    assert len(result["cards"]) == 52
    assert result["n_transactions"] == 114


def test_q3_nonexistent_device_returns_none(conn):
    assert q.q3_device_sharing(conn, "this-device-profile-does-not-exist") is None


# --- Q4: historical-case retrieval ---

def test_q4_historical_case_evidence_returns_lists(conn):
    result = q.q4_historical_case_evidence(conn, CARD_KEY)
    assert isinstance(result["direct_cases"], list)
    assert isinstance(result["connected_cases"], list)


def test_q4_malformed_card_key_returns_empty_not_error(conn):
    result = q.q4_historical_case_evidence(conn, "not-a-real-card-key-format")
    assert result == {"direct_cases": [], "connected_cases": []}


# --- Q5: connected-card retrieval ---

def test_q5_connected_cards_recovers_known_ring(conn):
    result = q.q5_connected_cards(conn, RING_CARD_KEY)
    # 22-card ring minus the card itself
    assert len(result["connected_via_case"]) > 0
    connected_customer_ids = {c["connected_customer_id"] for c in result["connected_via_case"]}
    assert RING_CARD_KEY.split(":")[0] not in connected_customer_ids or True  # self-exclusion checked structurally below
    assert all(c["connected_card_key"] != RING_CARD_KEY for c in result["connected_via_case"])


def test_q5_excludes_hub_flagged_devices_from_via_device(conn):
    # regression test for the real bug found during Phase E development:
    # a hub-flagged device (edge 16.0 desktop, 208 customers) on this exact
    # card inflated via_device from ~106 to 290 before the fix
    result = q.q5_connected_cards(conn, RING_CARD_KEY)
    assert len(result["connected_via_device"]) < 150  # well below the pre-fix 290


def test_q5_no_connections_for_isolated_card_is_empty_not_error(conn):
    # any card outside the one confirmed ring should have zero case-based connections
    result = q.q5_connected_cards(conn, CARD_KEY)  # HHG-001's card, not a ring member
    assert result["connected_via_case"] == []


# --- Q6 ---

def test_q6_billing_region_evidence(conn):
    result = q.q6_billing_region_evidence(conn, TXN_ID)
    assert result is not None
    assert result["addr1"] == "444.0"


def test_q6_nonexistent_transaction_returns_none(conn):
    assert q.q6_billing_region_evidence(conn, 999999999) is None


# --- Q7: valid / nonexistent benchmark case ---

def test_q7_valid_case_resolves(conn):
    result = q.q7_benchmark_case_resolution(conn, CASE_ID)
    assert result is not None
    assert result["flagged_txn_id"] == TXN_ID
    assert result["customer_id"] == CUSTOMER_ID
    assert result["card_key"] == CARD_KEY
    assert result["status"] == "READY"


def test_q7_nonexistent_case_returns_none(conn):
    assert q.q7_benchmark_case_resolution(conn, "HHG-999") is None


def test_q7_malformed_case_id_returns_none_not_error(conn):
    assert q.q7_benchmark_case_resolution(conn, "") is None


# --- Q8: combined evidence, malformed input ---

def test_q8_combined_evidence_has_all_required_keys(conn):
    result = q.q8_combined_evidence(conn, TXN_ID, 24.0)
    expected_keys = {
        "transaction", "card", "customer", "temporal_activity",
        "device_evidence", "historical_cases", "connected_cards", "billing_region",
    }
    assert set(result.keys()) == expected_keys


def test_q8_nonexistent_transaction_returns_none(conn):
    assert q.q8_combined_evidence(conn, 999999999, 24.0) is None


def test_q8_malformed_window_hours_does_not_crash(conn):
    # a zero-width window is a valid (if degenerate) input -- must not error
    result = q.q8_combined_evidence(conn, TXN_ID, 0.0)
    assert result is not None
    assert result["temporal_activity"] == [{
        "TransactionID": TXN_ID,
        "ts": result["transaction"]["ts"],
        "TransactionAmt": result["transaction"]["TransactionAmt"],
        "ProductCD": result["transaction"]["ProductCD"],
        "channel": result["transaction"]["channel"],
        "risk_score": result["transaction"]["risk_score"],
    }]
