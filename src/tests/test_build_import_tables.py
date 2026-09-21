"""
Focused tests for the import-table builders, using a small fixture that
exercises every non-obvious behavior:
  CY01 -> online, HAS an identity record (device_profile resolvable),
          is the flagged transaction of a case_pack case, and is the
          subject of its own closed case (CC-X1) -- exercises the
          "primary" resolution path end to end.
  CY02 -> in_person (ProductCD='W') -- must get NO device_profile at all.
  CY03 -> online but with NO identity row -- the ~4.4% real coverage gap;
          must also get NO device_profile, not a fabricated one.
  CY04 -> never has a case of its own; only appears inside CC-X1's
          connected_card_ids -- exercises the fallback resolution that
          Phase C added after finding this exact situation in the real
          22-card ring.
"""
from pathlib import Path

import pandas as pd
import pytest

from src.data.build_import_tables import (
    CURATED_TRANSACTION_COLUMNS,
    apply_hub_flag,
    build_case_pack_resolved_table,
    build_closed_case_connected_to_card_table,
    build_customers_table,
    build_historical_cases_table,
    build_transactions_graph_table,
)

DEFAULT_TXN = {
    "TransactionAmt": 50.0, "risk_score": 0.2, "addr1": 100.0, "addr2": 87.0, "dist1": None,
    "P_emaildomain": "example.com", "R_emaildomain": None,
    **{f"C{i}": 0 for i in range(1, 15)},
    "D1": 0.0, "D2": None, "D3": None, "D4": None, "D10": None, "D11": None, "D15": None,
    "M1": "T", "M2": "T", "M3": "T", "M4": "M0", "M6": "F",
}


def make_txn(**overrides) -> dict:
    row = dict(DEFAULT_TXN)
    row.update(overrides)
    return {col: row.get(col) for col in CURATED_TRANSACTION_COLUMNS}


TRANSACTIONS = pd.DataFrame(
    [
        make_txn(TransactionID=5001, customer_id="CY01", card1=101, card4="visa", card6="credit", ts="2016-07-01 10:00:00", ProductCD="C", channel="online"),
        make_txn(TransactionID=5002, customer_id="CY02", card1=102, card4="visa", card6="debit", ts="2016-07-02 10:00:00", ProductCD="W", channel="in_person"),
        make_txn(TransactionID=5003, customer_id="CY03", card1=103, card4="mastercard", card6="credit", ts="2016-07-03 10:00:00", ProductCD="R", channel="online"),
        make_txn(TransactionID=5004, customer_id="CY04", card1=104, card4="visa", card6="debit", ts="2016-07-04 10:00:00", ProductCD="C", channel="online"),
    ]
)

IDENTITY = pd.DataFrame(
    [
        # only CY01's transaction has an identity record
        {"TransactionID": 5001, "DeviceInfo": "SM-G935F Build/NRD90M", "id_30": "Android 7.0", "id_31": "chrome 62.0 for android", "id_33": "1920x1080", "DeviceType": "mobile", "id_12": "Found", "id_15": "New", "id_23": "IP_PROXY:ANONYMOUS", "id_34": "match_status:2"},
    ]
)

CASE_PACK = pd.DataFrame(
    [{"case_id": "HHG-T1", "opened_at": "2016-07-01 11:00:00", "trigger_type": "risk_score", "trigger_text": "test", "flagged_txn_id": 5001, "customer_id": "CY01", "risk_score": 0.2}]
)

CLOSED_CASES = pd.DataFrame(
    [
        {
            "case_id": "CC-X1", "customer_id": "CY01", "card_id": "CY01-K1",
            "outcome": "confirmed_fraud", "pattern": "card_not_present_fraud",
            "opened_at": "2016-07-01 12:00:00", "closed_at": "2016-07-03 12:00:00",
            "exposure_usd": 50.0, "n_txns": 1, "report_filed": "No",
            "actions_taken": "CREATE_CASE|BLOCK_CARD", "analyst_notes": "test case",
            "txn_ids": "5001",
            # CY04 never has a case of its own -- only shows up here
            "connected_card_ids": "CY04-K1",
        }
    ]
)

CARDS_DF = pd.DataFrame(
    [
        {"card_key": "CY01:101", "customer_id": "CY01", "card1": 101, "card4": "visa", "card4_conflict": False, "card6": "credit", "card6_conflict": False, "first_seen_ts": "2016-07-01 10:00:00", "last_seen_ts": "2016-07-01 10:00:00", "transaction_count": 1, "external_ids": "CY01-K1"},
        {"card_key": "CY02:102", "customer_id": "CY02", "card1": 102, "card4": "visa", "card4_conflict": False, "card6": "debit", "card6_conflict": False, "first_seen_ts": "2016-07-02 10:00:00", "last_seen_ts": "2016-07-02 10:00:00", "transaction_count": 1, "external_ids": ""},
        {"card_key": "CY03:103", "customer_id": "CY03", "card1": 103, "card4": "mastercard", "card4_conflict": False, "card6": "credit", "card6_conflict": False, "first_seen_ts": "2016-07-03 10:00:00", "last_seen_ts": "2016-07-03 10:00:00", "transaction_count": 1, "external_ids": ""},
        {"card_key": "CY04:104", "customer_id": "CY04", "card1": 104, "card4": "visa", "card4_conflict": False, "card6": "debit", "card6_conflict": False, "first_seen_ts": "2016-07-04 10:00:00", "last_seen_ts": "2016-07-04 10:00:00", "transaction_count": 1, "external_ids": ""},
    ]
)


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    TRANSACTIONS.to_csv(tmp_path / "transactions.csv", index=False)
    IDENTITY.to_csv(tmp_path / "identity.csv", index=False)
    CASE_PACK.to_csv(tmp_path / "case_pack.csv", index=False)
    CLOSED_CASES.to_csv(tmp_path / "closed_cases_history.csv", index=False)
    return tmp_path


def test_apply_hub_flag_uses_stated_threshold():
    df = pd.DataFrame({"n_distinct_customers": [1, 5, 20, 200]})
    flagged = apply_hub_flag(df, n_customers_total=1000)  # 1% of 1000 = 10
    assert list(flagged["hub_flag"]) == [False, False, True, True]


def test_transactions_graph_device_profile_only_present_for_online_with_identity(raw_dir):
    df = build_transactions_graph_table(raw_dir, devices_df=pd.DataFrame())
    by_id = df.set_index("TransactionID")

    assert by_id.loc[5001, "device_profile"] == "SM-G935F Build/NRD90M|Android 7.0|chrome 62.0 for android|1920x1080"
    assert pd.isna(by_id.loc[5002, "device_profile"])  # in_person: never has identity
    assert pd.isna(by_id.loc[5003, "device_profile"])  # online but missing identity row
    assert by_id.loc[5001, "card_key"] == "CY01:101"


def test_transactions_graph_row_count_matches_source(raw_dir):
    df = build_transactions_graph_table(raw_dir, devices_df=pd.DataFrame())
    assert len(df) == len(TRANSACTIONS)
    assert df["TransactionID"].is_unique


def test_customers_table_aggregates_correctly():
    customers = build_customers_table(CARDS_DF)
    cy01 = customers[customers["customer_id"] == "CY01"].iloc[0]
    assert cy01["n_cards"] == 1
    assert cy01["total_transaction_count"] == 1
    assert customers["customer_id"].is_unique


def test_historical_cases_resolves_card_key(raw_dir):
    historical_cases, involves_df, closed_connected_raw = build_historical_cases_table(raw_dir)
    row = historical_cases[historical_cases["case_id"] == "CC-X1"].iloc[0]
    assert row["card_key"] == "CY01:101"
    assert set(involves_df["txn_id"]) == {5001}
    assert historical_cases["case_id"].is_unique


def test_connected_card_uses_primary_resolution_when_available(raw_dir):
    # CY01-K1 is a primary case subject, so a case naming it as "connected"
    # should resolve through cards.csv's external_ids, not the fallback
    closed = CLOSED_CASES.copy()
    closed["connected_card_ids"] = "CY01-K1"
    connected_df, unresolved = build_closed_case_connected_to_card_table(closed[["case_id", "connected_card_ids"]], CARDS_DF)
    assert unresolved == []
    assert connected_df.iloc[0]["connected_card_key"] == "CY01:101"
    assert connected_df.iloc[0]["resolution"] == "primary_external_id"


def test_connected_card_falls_back_for_customer_with_no_case_of_their_own():
    connected_df, unresolved = build_closed_case_connected_to_card_table(
        CLOSED_CASES[["case_id", "connected_card_ids"]], CARDS_DF
    )
    assert unresolved == []
    assert connected_df.iloc[0]["connected_card_key"] == "CY04:104"
    assert connected_df.iloc[0]["resolution"] == "fallback_single_card_customer"


def test_case_pack_resolved_matches_flagged_transaction(raw_dir):
    resolved = build_case_pack_resolved_table(raw_dir)
    row = resolved.iloc[0]
    assert row["card_key"] == "CY01:101"
    assert row["customer_id"] == "CY01"
