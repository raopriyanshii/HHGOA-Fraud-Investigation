"""
Builds the derived, graph-loading-ready tables described in
data/processed/graph_data_model.md. Every decision here (which fields are
curated onto Transaction, which entities get a hub_flag, which candidate
"vertices" were rejected) is backed by the measurements in
graph_model_analysis.py -- this module does not introduce new modeling
judgment calls, it materializes the ones already documented.

Consistent hub-detection rule, applied identically to every shared-entity
table (devices, billing regions): a value is flagged a "hub" when more than
1% of all customers (13,553 * 0.01 ~= 136) touch it. Above that share, the
shared-entity signal has essentially no discriminating power for ring
detection and must not be used as a WCC/traversal connector -- see the
graph_model_analysis measurements (e.g. gmail.com touches 66% of customers;
the single most common device+OS+browser+screen combination touches 52).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.card_resolution import parse_pipe_separated_ids, resolve_transactions
from src.data.graph_model_analysis import build_device_profile_key

HUB_CUSTOMER_SHARE_THRESHOLD = 0.01  # see module docstring

# Curated Transaction columns -- see graph_data_model.md section 3 for the
# MUST/USEFUL/RAW-ONLY/NOT-NEEDED classification and why each field landed
# where it did. Anything not listed here stays in transactions.csv only.
CURATED_TRANSACTION_COLUMNS = [
    "TransactionID", "customer_id", "card1", "ts", "TransactionAmt", "ProductCD", "channel", "risk_score",
    "addr1", "addr2", "dist1", "P_emaildomain", "R_emaildomain", "card4", "card6",
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14",
    "D1", "D2", "D3", "D4", "D10", "D11", "D15",
    "M1", "M2", "M3", "M4", "M6",
]

IDENTITY_JOIN_COLUMNS = ["TransactionID", "DeviceType", "id_12", "id_15", "id_23", "id_34"]


def build_customers_table(cards_df: pd.DataFrame) -> pd.DataFrame:
    grouped = cards_df.groupby("customer_id").agg(
        n_cards=("card_key", "count"),
        first_seen_ts=("first_seen_ts", "min"),
        last_seen_ts=("last_seen_ts", "max"),
        total_transaction_count=("transaction_count", "sum"),
    )
    return grouped.reset_index().sort_values("customer_id").reset_index(drop=True)


def build_devices_table(raw_dir: Path) -> pd.DataFrame:
    identity = pd.read_csv(
        raw_dir / "identity.csv", usecols=["TransactionID", "DeviceInfo", "id_30", "id_31", "id_33"], low_memory=False
    )
    identity["device_profile"] = identity.apply(build_device_profile_key, axis=1)
    identity = identity.dropna(subset=["device_profile"])

    resolved = resolve_transactions(raw_dir / "transactions.csv", set(identity["TransactionID"].tolist()))
    identity["customer_id"] = identity["TransactionID"].map(lambda t: resolved.get(t, (None, None))[0])
    identity["card1"] = identity["TransactionID"].map(lambda t: resolved.get(t, (None, None))[1])
    identity = identity.dropna(subset=["customer_id"])

    n_customers_total = identity["customer_id"].nunique()  # reported separately in the report; threshold uses cards.csv total

    rows = []
    for profile, group in identity.groupby("device_profile"):
        device_info, os_, browser, screen = profile.split("|")
        rows.append(
            {
                "device_profile": profile,
                "device_info_raw": device_info,
                "os": os_,
                "browser": browser,
                "screen": screen,
                "n_transactions": len(group),
                "n_distinct_customers": group["customer_id"].nunique(),
                "n_distinct_cards": len(set(zip(group["customer_id"], group["card1"]))),
            }
        )
    df = pd.DataFrame(rows).sort_values("device_profile").reset_index(drop=True)
    return df


def build_billing_regions_table(transactions_path: Path, chunksize: int = 100_000) -> pd.DataFrame:
    accumulator: dict = {}
    for chunk in pd.read_csv(transactions_path, usecols=["addr1", "customer_id", "card1"], chunksize=chunksize, low_memory=False):
        valid = chunk.dropna(subset=["addr1"])
        for addr1, group in valid.groupby("addr1"):
            acc = accumulator.setdefault(addr1, {"n_transactions": 0, "customers": set(), "cards": set()})
            acc["n_transactions"] += len(group)
            acc["customers"].update(group["customer_id"].tolist())
            acc["cards"].update(zip(group["customer_id"], group["card1"]))

    rows = [
        {"addr1": addr1, "n_transactions": a["n_transactions"], "n_distinct_customers": len(a["customers"]), "n_distinct_cards": len(a["cards"])}
        for addr1, a in accumulator.items()
    ]
    return pd.DataFrame(rows).sort_values("addr1").reset_index(drop=True)


def apply_hub_flag(df: pd.DataFrame, n_customers_total: int) -> pd.DataFrame:
    threshold = n_customers_total * HUB_CUSTOMER_SHARE_THRESHOLD
    df = df.copy()
    df["hub_flag"] = df["n_distinct_customers"] > threshold
    return df


def build_transactions_graph_table(raw_dir: Path, devices_df: pd.DataFrame, chunksize: int = 100_000) -> pd.DataFrame:
    identity = pd.read_csv(raw_dir / "identity.csv", usecols=["TransactionID", "DeviceInfo", "id_30", "id_31", "id_33"] + IDENTITY_JOIN_COLUMNS[1:], low_memory=False)
    identity["device_profile"] = identity.apply(build_device_profile_key, axis=1)
    identity_lookup = identity.set_index("TransactionID")[["device_profile"] + IDENTITY_JOIN_COLUMNS[1:]]

    chunks_out = []
    for chunk in pd.read_csv(raw_dir / "transactions.csv", usecols=CURATED_TRANSACTION_COLUMNS, chunksize=chunksize, low_memory=False):
        chunk = chunk.rename(columns={"card4": "card4_raw", "card6": "card6_raw"})
        chunk["card_key"] = chunk["customer_id"] + ":" + chunk["card1"].astype(str)
        joined = chunk.join(identity_lookup, on="TransactionID")
        chunks_out.append(joined)

    df = pd.concat(chunks_out, ignore_index=True)
    return df


def build_historical_cases_table(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (historical_cases, closed_case_involves_transaction, closed_case_connected_to_card)."""
    closed = pd.read_csv(raw_dir / "closed_cases_history.csv", low_memory=False)
    closed["txn_id_list"] = closed["txn_ids"].apply(parse_pipe_separated_ids)

    all_txn_ids: set[int] = set()
    for ids in closed["txn_id_list"]:
        all_txn_ids.update(ids)
    resolved = resolve_transactions(raw_dir / "transactions.csv", all_txn_ids)

    def resolve_card_key(txn_ids: list[int]) -> str | None:
        pairs = {resolved[t] for t in txn_ids if t in resolved}
        if len(pairs) != 1:
            return None
        (customer_id, card1), = pairs
        return f"{customer_id}:{card1}"

    closed["card_key"] = closed["txn_id_list"].apply(resolve_card_key)

    involves_rows = [
        {"case_id": row.case_id, "txn_id": txn_id} for row in closed.itertuples() for txn_id in row.txn_id_list
    ]
    involves_df = pd.DataFrame(involves_rows)

    historical_cases = closed[
        [
            "case_id", "customer_id", "card_key", "outcome", "pattern", "opened_at", "closed_at",
            "exposure_usd", "n_txns", "report_filed", "actions_taken", "analyst_notes",
        ]
    ].copy()

    return historical_cases, involves_df, closed[["case_id", "connected_card_ids"]]


def build_closed_case_connected_to_card_table(closed_connected_raw: pd.DataFrame, cards_df: pd.DataFrame) -> pd.DataFrame:
    """connected_card_ids holds the ORIGINAL card_id label strings (e.g.
    "C00255-K1"), never a card1 value.

    Primary resolution: through cards.csv's own external_ids column (built
    in Phase B from case_pack.csv / closed_cases_history.csv's own card_id
    field), exactly like every other card_id string in this project.

    Fallback: some connected cards (measured: 40 of 52 references here)
    belong to a customer who never has a case of their own -- they are
    mentioned ONLY inside another case's connected_card_ids, so no
    transaction reference ever put their label into external_ids. Phase B
    independently proved every customer in the ENTIRE dataset has exactly
    one card1 (13,553 customers, 13,553 cards, zero exceptions), so for
    these we can safely take just the customer_id prefix off the label
    (never the -K suffix) and look up that customer's one real card in
    cards.csv directly. This is explicitly NOT "parsing -K1/-K2 as separate
    cards" -- it never inspects the suffix -- it only works because
    multi-card customers are proven not to exist here. Every fallback
    resolution is tagged so it is never confused with the stronger,
    directly-evidenced primary path.
    """
    label_to_key: dict[str, str] = {}
    for row in cards_df.itertuples():
        if isinstance(row.external_ids, str) and row.external_ids:
            for label in row.external_ids.split("|"):
                label_to_key[label] = row.card_key

    cards_per_customer = cards_df.groupby("customer_id")["card_key"].apply(list).to_dict()

    rows = []
    unresolved = []
    for row in closed_connected_raw.itertuples():
        if not isinstance(row.connected_card_ids, str) or not row.connected_card_ids:
            continue
        for label in row.connected_card_ids.split("|"):
            key = label_to_key.get(label)
            if key is not None:
                rows.append({"case_id": row.case_id, "connected_card_key": key, "resolution": "primary_external_id"})
                continue

            customer_prefix = label.split("-")[0]
            candidate_keys = cards_per_customer.get(customer_prefix)
            if candidate_keys and len(candidate_keys) == 1:
                rows.append({"case_id": row.case_id, "connected_card_key": candidate_keys[0], "resolution": "fallback_single_card_customer"})
            else:
                unresolved.append({"case_id": row.case_id, "label": label, "reason": "no_case_and_customer_not_single_card" if candidate_keys else "customer_not_found"})

    return pd.DataFrame(rows), unresolved


def build_case_pack_resolved_table(raw_dir: Path) -> pd.DataFrame:
    case_pack = pd.read_csv(raw_dir / "case_pack.csv")
    resolved = resolve_transactions(raw_dir / "transactions.csv", set(case_pack["flagged_txn_id"].tolist()))

    def card_key_for(txn_id: int) -> str | None:
        hit = resolved.get(txn_id)
        return None if hit is None else f"{hit[0]}:{hit[1]}"

    case_pack = case_pack.copy()
    case_pack["card_key"] = case_pack["flagged_txn_id"].apply(card_key_for)
    return case_pack[["case_id", "opened_at", "trigger_type", "trigger_text", "flagged_txn_id", "customer_id", "card_key", "risk_score"]]


def run(raw_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rel_dir = out_dir / "relationships"
    rel_dir.mkdir(parents=True, exist_ok=True)

    cards_df = pd.read_csv(out_dir / "cards.csv")
    n_customers_total = cards_df["customer_id"].nunique()

    customers_df = build_customers_table(cards_df)
    devices_df = apply_hub_flag(build_devices_table(raw_dir), n_customers_total)
    billing_regions_df = apply_hub_flag(build_billing_regions_table(raw_dir / "transactions.csv"), n_customers_total)
    transactions_graph_df = build_transactions_graph_table(raw_dir, devices_df)
    historical_cases_df, involves_df, closed_connected_raw = build_historical_cases_table(raw_dir)
    connected_to_card_df, unresolved_connected = build_closed_case_connected_to_card_table(closed_connected_raw, cards_df)
    case_pack_resolved_df = build_case_pack_resolved_table(raw_dir)

    customers_df.to_csv(out_dir / "customers.csv", index=False)
    devices_df.to_csv(out_dir / "devices.csv", index=False)
    billing_regions_df.to_csv(out_dir / "billing_regions.csv", index=False)
    transactions_graph_df.to_csv(out_dir / "transactions_graph.csv", index=False)
    historical_cases_df.to_csv(out_dir / "historical_cases.csv", index=False)
    case_pack_resolved_df.to_csv(out_dir / "case_pack_resolved.csv", index=False)
    involves_df.to_csv(rel_dir / "closed_case_involves_transaction.csv", index=False)
    connected_to_card_df.to_csv(rel_dir / "closed_case_connected_to_card.csv", index=False)

    return {
        "customers": customers_df,
        "devices": devices_df,
        "billing_regions": billing_regions_df,
        "transactions_graph": transactions_graph_df,
        "historical_cases": historical_cases_df,
        "case_pack_resolved": case_pack_resolved_df,
        "closed_case_involves_transaction": involves_df,
        "closed_case_connected_to_card": connected_to_card_df,
        "unresolved_connected_card_labels": unresolved_connected,
        "n_customers_total": n_customers_total,
    }


if __name__ == "__main__":
    raw_dir = Path("D:/hackathon/task_5")
    out_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
    result = run(raw_dir, out_dir)
    for name, df in result.items():
        if isinstance(df, pd.DataFrame):
            print(f"{name}: {len(df)} rows")
    print(f"unresolved connected_card labels: {result['unresolved_connected_card_labels']}")
