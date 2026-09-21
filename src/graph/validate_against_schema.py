"""
Pre-load validation: checks every processed CSV against the concrete GSQL
types proposed in gsql/01_vertices.gsql / 02_edges.gsql, before any of it
is ever sent to TigerGraph. Purely local file checks -- no TigerGraph
connection is used or needed here.

This is a different concern from Phase C's graph_data_model_validation.json
(which checked referential integrity between the CSVs). This checks
type-compatibility with the specific GSQL types chosen for Phase D: does
every value in a column declared DATETIME actually parse as one, does
every column declared UINT actually hold non-negative integers, are the
boolean columns really the exact "True"/"False" strings the loading jobs'
CASE WHEN transforms assume, etc.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

DATETIME_FMT = "%Y-%m-%d %H:%M:%S"


def _check_datetime_column(series: pd.Series) -> dict:
    non_null = series.dropna()
    bad = []
    for v in non_null.unique():
        try:
            datetime.strptime(str(v), DATETIME_FMT)
        except ValueError:
            bad.append(v)
    return {"n_checked": len(non_null), "n_bad_format": len(bad), "examples": bad[:5]}


def _check_nonneg_int_column(series: pd.Series) -> dict:
    non_null = series.dropna()
    bad = []
    for v in non_null.unique():
        try:
            f = float(v)
            if f < 0 or f != int(f):
                bad.append(v)
        except (ValueError, TypeError):
            bad.append(v)
    return {"n_checked": len(non_null), "n_bad_format": len(bad), "examples": [str(b) for b in bad[:5]]}


def _check_float_column(series: pd.Series) -> dict:
    non_null = series.dropna()
    bad = []
    for v in non_null.unique():
        try:
            float(v)
        except (ValueError, TypeError):
            bad.append(v)
    return {"n_checked": len(non_null), "n_bad_format": len(bad), "examples": [str(b) for b in bad[:5]]}


def _check_bool_string_column(series: pd.Series) -> dict:
    non_null = series.dropna().astype(str)
    allowed = {"True", "False"}
    bad = sorted(set(non_null.unique()) - allowed)
    return {"n_checked": len(non_null), "unexpected_values": bad[:10]}


def _check_unique_nonnull(series: pd.Series, name: str) -> dict:
    return {
        "column": name,
        "n_rows": len(series),
        "n_null": int(series.isna().sum()),
        "n_unique": int(series.nunique()),
        "is_unique_and_nonnull": bool(series.notna().all() and series.is_unique),
    }


def validate_cards() -> dict:
    df = pd.read_csv(DATA_DIR / "cards.csv")
    return {
        "primary_key_card_key": _check_unique_nonnull(df["card_key"], "card_key"),
        "card1_uint": _check_nonneg_int_column(df["card1"]),
        "transaction_count_uint": _check_nonneg_int_column(df["transaction_count"]),
        "card4_conflict_bool": _check_bool_string_column(df["card4_conflict"]),
        "card6_conflict_bool": _check_bool_string_column(df["card6_conflict"]),
        "first_seen_ts_datetime": _check_datetime_column(df["first_seen_ts"]),
        "last_seen_ts_datetime": _check_datetime_column(df["last_seen_ts"]),
    }


def validate_customers() -> dict:
    df = pd.read_csv(DATA_DIR / "customers.csv")
    return {
        "primary_key_customer_id": _check_unique_nonnull(df["customer_id"], "customer_id"),
        "n_cards_uint": _check_nonneg_int_column(df["n_cards"]),
        "total_transaction_count_uint": _check_nonneg_int_column(df["total_transaction_count"]),
        "first_seen_ts_datetime": _check_datetime_column(df["first_seen_ts"]),
        "last_seen_ts_datetime": _check_datetime_column(df["last_seen_ts"]),
    }


def validate_devices() -> dict:
    df = pd.read_csv(DATA_DIR / "devices.csv")
    return {
        "primary_key_device_profile": _check_unique_nonnull(df["device_profile"], "device_profile"),
        "n_transactions_uint": _check_nonneg_int_column(df["n_transactions"]),
        "n_distinct_customers_uint": _check_nonneg_int_column(df["n_distinct_customers"]),
        "n_distinct_cards_uint": _check_nonneg_int_column(df["n_distinct_cards"]),
        "hub_flag_bool": _check_bool_string_column(df["hub_flag"]),
    }


def validate_billing_regions() -> dict:
    df = pd.read_csv(DATA_DIR / "billing_regions.csv", dtype={"addr1": str})
    return {
        "primary_key_addr1": _check_unique_nonnull(df["addr1"], "addr1"),
        "n_transactions_uint": _check_nonneg_int_column(df["n_transactions"]),
        "hub_flag_bool": _check_bool_string_column(df["hub_flag"]),
    }


def validate_historical_cases() -> dict:
    df = pd.read_csv(DATA_DIR / "historical_cases.csv")
    return {
        "primary_key_case_id": _check_unique_nonnull(df["case_id"], "case_id"),
        "card_key_nonnull": {"n_null": int(df["card_key"].isna().sum())},
        "exposure_usd_float": _check_float_column(df["exposure_usd"]),
        "n_txns_uint": _check_nonneg_int_column(df["n_txns"]),
        "opened_at_datetime": _check_datetime_column(df["opened_at"]),
        "closed_at_datetime": _check_datetime_column(df["closed_at"]),
        "report_filed_values": sorted(df["report_filed"].dropna().unique().tolist()),
    }


def validate_case_pack_resolved() -> dict:
    df = pd.read_csv(DATA_DIR / "case_pack_resolved.csv")
    return {
        "primary_key_case_id": _check_unique_nonnull(df["case_id"], "case_id"),
        "flagged_txn_id_uint": _check_nonneg_int_column(df["flagged_txn_id"]),
        "card_key_nonnull": {"n_null": int(df["card_key"].isna().sum())},
        "risk_score_float_or_empty": _check_float_column(df["risk_score"]),
        "risk_score_null_count": int(df["risk_score"].isna().sum()),
        "opened_at_datetime": _check_datetime_column(df["opened_at"]),
        "trigger_type_values": sorted(df["trigger_type"].dropna().unique().tolist()),
    }


def validate_relationship_tables() -> dict:
    involves = pd.read_csv(DATA_DIR / "relationships" / "closed_case_involves_transaction.csv")
    connected = pd.read_csv(DATA_DIR / "relationships" / "closed_case_connected_to_card.csv")
    return {
        "involves_txn_id_uint": _check_nonneg_int_column(involves["txn_id"]),
        "involves_no_null_case_id": {"n_null": int(involves["case_id"].isna().sum())},
        "connected_resolution_values": sorted(connected["resolution"].dropna().unique().tolist()),
        "connected_no_null_card_key": {"n_null": int(connected["connected_card_key"].isna().sum())},
    }


def validate_transactions_graph(chunksize: int = 100_000) -> dict:
    datetime_bad = set()
    txn_id_seen = set()
    txn_id_dupes = 0
    txn_amt_bad = set()
    risk_score_bad = set()
    card1_bad = set()
    c_cols_bad = {f"C{i}": set() for i in range(1, 15)}
    total_rows = 0

    for chunk in pd.read_csv(DATA_DIR / "transactions_graph.csv", chunksize=chunksize, low_memory=False):
        total_rows += len(chunk)

        before = len(txn_id_seen)
        txn_id_seen.update(chunk["TransactionID"].tolist())
        txn_id_dupes += len(chunk) - (len(txn_id_seen) - before)

        for v in chunk["ts"].dropna().unique():
            try:
                datetime.strptime(str(v), DATETIME_FMT)
            except ValueError:
                datetime_bad.add(v)

        for v in chunk["TransactionAmt"].dropna().unique():
            try:
                float(v)
            except (ValueError, TypeError):
                txn_amt_bad.add(v)

        for v in chunk["risk_score"].dropna().unique():
            try:
                float(v)
            except (ValueError, TypeError):
                risk_score_bad.add(v)

        for v in chunk["card1"].dropna().unique():
            try:
                f = float(v)
                if f < 0 or f != int(f):
                    card1_bad.add(v)
            except (ValueError, TypeError):
                card1_bad.add(v)

        for col in c_cols_bad:
            for v in chunk[col].dropna().unique():
                try:
                    float(v)
                except (ValueError, TypeError):
                    c_cols_bad[col].add(v)

    return {
        "total_rows": total_rows,
        "transaction_id_duplicates": txn_id_dupes,
        "ts_bad_format_count": len(datetime_bad),
        "ts_bad_examples": list(datetime_bad)[:5],
        "transaction_amt_bad_count": len(txn_amt_bad),
        "risk_score_bad_count": len(risk_score_bad),
        "card1_bad_count": len(card1_bad),
        "c_columns_bad_counts": {k: len(v) for k, v in c_cols_bad.items() if v},
    }


def run() -> dict:
    return {
        "cards": validate_cards(),
        "customers": validate_customers(),
        "devices": validate_devices(),
        "billing_regions": validate_billing_regions(),
        "historical_cases": validate_historical_cases(),
        "case_pack_resolved": validate_case_pack_resolved(),
        "relationship_tables": validate_relationship_tables(),
        "transactions_graph": validate_transactions_graph(),
    }


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2, default=str))
