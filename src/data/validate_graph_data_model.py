"""
Phase C's integrity validation: checks every import table produced by
build_import_tables.py against the rules in graph_data_model.md section 16,
and assembles the measured vertex/edge counts used for the graph size
estimate. This is read-only over the already-generated processed files;
it does not touch the raw organizer files at all.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_tables(out_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "cards": pd.read_csv(out_dir / "cards.csv"),
        "customers": pd.read_csv(out_dir / "customers.csv"),
        "devices": pd.read_csv(out_dir / "devices.csv"),
        "billing_regions": pd.read_csv(out_dir / "billing_regions.csv"),
        "transactions_graph": pd.read_csv(out_dir / "transactions_graph.csv", low_memory=False),
        "historical_cases": pd.read_csv(out_dir / "historical_cases.csv"),
        "case_pack_resolved": pd.read_csv(out_dir / "case_pack_resolved.csv"),
        "closed_case_involves_transaction": pd.read_csv(out_dir / "relationships" / "closed_case_involves_transaction.csv"),
        "closed_case_connected_to_card": pd.read_csv(out_dir / "relationships" / "closed_case_connected_to_card.csv"),
    }


def run_validations(t: dict[str, pd.DataFrame]) -> dict:
    results = {}

    # --- primary key uniqueness ---
    results["pk_uniqueness"] = {
        "transactions_graph.TransactionID": bool(t["transactions_graph"]["TransactionID"].is_unique),
        "cards.card_key": bool(t["cards"]["card_key"].is_unique),
        "customers.customer_id": bool(t["customers"]["customer_id"].is_unique),
        "devices.device_profile": bool(t["devices"]["device_profile"].is_unique),
        "billing_regions.addr1": bool(t["billing_regions"]["addr1"].is_unique),
        "historical_cases.case_id": bool(t["historical_cases"]["case_id"].is_unique),
        "case_pack_resolved.case_id": bool(t["case_pack_resolved"]["case_id"].is_unique),
    }

    # --- foreign key resolution (no invented IDs) ---
    card_keys = set(t["cards"]["card_key"])
    customer_ids = set(t["customers"]["customer_id"])
    txn_ids = set(t["transactions_graph"]["TransactionID"])
    device_profiles = set(t["devices"]["device_profile"])

    results["fk_resolution"] = {
        "every_transaction_card_key_exists_in_cards": bool(t["transactions_graph"]["card_key"].isin(card_keys).all()),
        "every_card_customer_id_exists_in_customers": bool(t["cards"]["customer_id"].isin(customer_ids).all()),
        "every_transaction_device_profile_exists_in_devices_or_is_null": bool(
            t["transactions_graph"]["device_profile"].dropna().isin(device_profiles).all()
        ),
        "every_historical_case_card_key_exists_in_cards": bool(t["historical_cases"]["card_key"].dropna().isin(card_keys).all()),
        "historical_case_card_key_null_count": int(t["historical_cases"]["card_key"].isna().sum()),
        "every_involves_txn_id_exists_in_transactions": bool(t["closed_case_involves_transaction"]["txn_id"].isin(txn_ids).all()),
        "every_connected_card_key_exists_in_cards": bool(t["closed_case_connected_to_card"]["connected_card_key"].isin(card_keys).all()),
        "every_case_pack_card_key_exists_in_cards": bool(t["case_pack_resolved"]["card_key"].dropna().isin(card_keys).all()),
        "every_case_pack_flagged_txn_id_exists_in_transactions": bool(t["case_pack_resolved"]["flagged_txn_id"].isin(txn_ids).all()),
    }

    # --- no accidental self-relationships ---
    merged = t["closed_case_connected_to_card"].merge(
        t["historical_cases"][["case_id", "card_key"]], on="case_id", suffixes=("_connected", "_own")
    )
    self_refs = merged[merged["connected_card_key"] == merged["card_key"]]
    results["no_self_relationships"] = {
        "closed_case_connected_to_own_card_count": int(len(self_refs)),
        "examples": self_refs[["case_id", "connected_card_key"]].head(5).to_dict("records"),
    }

    # --- row counts (feeds the graph size estimate) ---
    results["row_counts"] = {name: len(df) for name, df in t.items()}

    # --- null handling: explicit report, not hidden ---
    null_report = {}
    for name, df in t.items():
        null_report[name] = {col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().any()}
    results["null_counts_by_table"] = null_report

    # --- hub flag sanity (devices / billing_regions) ---
    results["hub_flags"] = {
        "devices_flagged_hub": int(t["devices"]["hub_flag"].sum()),
        "devices_total": len(t["devices"]),
        "billing_regions_flagged_hub": int(t["billing_regions"]["hub_flag"].sum()),
        "billing_regions_total": len(t["billing_regions"]),
    }

    return results


def graph_size_estimate(t: dict[str, pd.DataFrame]) -> dict:
    return {
        "vertices": {
            "Customer": len(t["customers"]),
            "Card": len(t["cards"]),
            "Transaction": len(t["transactions_graph"]),
            "DeviceProfile": len(t["devices"]),
            "BillingRegion": len(t["billing_regions"]),
            "ClosedCase": len(t["historical_cases"]),
            "InvestigationCase": 20,  # 1 per benchmark case, upper bound for this run; grows as the agent runs
        },
        "edges": {
            "Customer_OWNS_Card": len(t["cards"]),  # one row per card, each with exactly one customer_id
            "Card_MADE_Transaction": len(t["transactions_graph"]),
            "Transaction_FROM_DEVICE_DeviceProfile": int(t["transactions_graph"]["device_profile"].notna().sum()),
            "Transaction_BILLED_IN_BillingRegion": int(t["transactions_graph"]["addr1"].notna().sum()),
            "ClosedCase_INVOLVES_Transaction": len(t["closed_case_involves_transaction"]),
            "ClosedCase_ON_CARD_Card": int(t["historical_cases"]["card_key"].notna().sum()),
            "ClosedCase_CONNECTED_TO_Card": len(t["closed_case_connected_to_card"]),
        },
    }


def main():
    out_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
    tables = load_tables(out_dir)
    validations = run_validations(tables)
    size_estimate = graph_size_estimate(tables)

    report = {"validation_results": validations, "graph_size_estimate": size_estimate}
    with open(out_dir / "graph_data_model_validation.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(json.dumps(size_estimate, indent=2))
    print("\nPK uniqueness:", validations["pk_uniqueness"])
    print("FK resolution:", validations["fk_resolution"])
    print("Self-relationships:", validations["no_self_relationships"])


if __name__ == "__main__":
    main()
