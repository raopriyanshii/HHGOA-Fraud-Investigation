"""
Read-only measurement functions that inform Phase C's graph data model
decisions (device cardinality, email/region hub risk, temporal spans of
real fraud episodes). Nothing here writes to the raw dataset, and nothing
here decides the schema by itself -- it only produces numbers the design
doc cites.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from src.data.card_resolution import parse_pipe_separated_ids, resolve_transactions

IDENTITY_COLUMNS = ["TransactionID", "DeviceType", "DeviceInfo", "id_12", "id_15", "id_23", "id_30", "id_31", "id_33", "id_34"]
UNKNOWN = "UNKNOWN"


def build_device_profile_key(row) -> str | None:
    """A device_profile requires a real DeviceInfo string -- if that's
    missing there isn't enough to fingerprint a device from, so no key is
    built (None), rather than inventing one from OS/browser/screen alone
    (which would be far less specific and even more hub-prone).
    Present-but-missing OS/browser/screen become an explicit "UNKNOWN"
    token so a genuinely-unknown OS is never silently merged with a
    different genuinely-unknown OS under two different missing-vs-known
    states.
    """
    device_info = row.get("DeviceInfo")
    if pd.isna(device_info) or device_info == "":
        return None
    os_ = row.get("id_30")
    browser = row.get("id_31")
    screen = row.get("id_33")
    os_ = UNKNOWN if pd.isna(os_) else os_
    browser = UNKNOWN if pd.isna(browser) else browser
    screen = UNKNOWN if pd.isna(screen) else screen
    return f"{device_info}|{os_}|{browser}|{screen}"


def analyze_device_cardinality(raw_dir: Path) -> dict:
    identity = pd.read_csv(raw_dir / "identity.csv", usecols=IDENTITY_COLUMNS, low_memory=False)
    identity["device_profile"] = identity.apply(build_device_profile_key, axis=1)

    txn_ids = set(identity["TransactionID"].tolist())
    resolved = resolve_transactions(raw_dir / "transactions.csv", txn_ids)

    identity["customer_id"] = identity["TransactionID"].map(lambda t: resolved.get(t, (None, None))[0])
    identity["card1"] = identity["TransactionID"].map(lambda t: resolved.get(t, (None, None))[1])

    def cardinality_report(group_col: str) -> dict:
        valid = identity.dropna(subset=[group_col])
        g = valid.groupby(group_col).agg(
            n_transactions=("TransactionID", "count"),
            n_distinct_customers=("customer_id", "nunique"),
            n_distinct_cards=("card1", lambda s: len(set(zip(valid.loc[s.index, "customer_id"], s)))),
        )
        top_by_customers = g.sort_values("n_distinct_customers", ascending=False).head(10)
        return {
            "n_distinct_values": len(g),
            "n_rows_with_value": len(valid),
            "n_rows_missing_value": len(identity) - len(valid),
            "customers_per_value_distribution": {
                "min": int(g["n_distinct_customers"].min()),
                "median": float(g["n_distinct_customers"].median()),
                "p90": float(g["n_distinct_customers"].quantile(0.9)),
                "max": int(g["n_distinct_customers"].max()),
            },
            "values_with_1_to_3_customers": int(((g["n_distinct_customers"] >= 1) & (g["n_distinct_customers"] <= 3)).sum()),
            "values_with_over_50_customers": int((g["n_distinct_customers"] > 50).sum()),
            "top_10_by_distinct_customers": [
                {"value": idx, "n_transactions": int(r.n_transactions), "n_distinct_customers": int(r.n_distinct_customers), "n_distinct_cards": int(r.n_distinct_cards)}
                for idx, r in top_by_customers.iterrows()
            ],
        }

    return {
        "device_info_alone": cardinality_report("DeviceInfo"),
        "device_profile_composite": cardinality_report("device_profile"),
        "total_identity_rows": len(identity),
    }


def analyze_field_hub_risk(transactions_path: Path, fields: list[str], chunksize: int = 100_000) -> dict:
    """For each candidate field (email domain, addr1, etc.), measure how
    many distinct customers/cards share each value -- the direct evidence
    for whether a field would make a useful, narrow graph edge or a noisy
    hub vertex that connects a large fraction of the whole graph.
    """
    accumulators = {f: {} for f in fields}
    total_rows = 0

    usecols = list(dict.fromkeys(fields + ["customer_id", "card1"]))
    for chunk in pd.read_csv(transactions_path, usecols=usecols, chunksize=chunksize, low_memory=False):
        total_rows += len(chunk)
        for field in fields:
            valid = chunk.dropna(subset=[field])
            for value, sub in valid.groupby(field):
                acc = accumulators[field].setdefault(value, {"count": 0, "customers": set(), "cards": set()})
                acc["count"] += len(sub)
                acc["customers"].update(sub["customer_id"].tolist())
                acc["cards"].update(zip(sub["customer_id"], sub["card1"]))

    report = {}
    for field in fields:
        acc = accumulators[field]
        rows = [
            {"value": value, "n_transactions": a["count"], "n_distinct_customers": len(a["customers"]), "n_distinct_cards": len(a["cards"])}
            for value, a in acc.items()
        ]
        rows.sort(key=lambda r: -r["n_distinct_customers"])
        n_customers_total = 13553  # from Phase B's cards.csv; used only to express hub share as a percentage
        report[field] = {
            "n_distinct_values": len(rows),
            "top_10_by_distinct_customers": rows[:10],
            "pct_of_all_customers_touched_by_top_value": round(100 * rows[0]["n_distinct_customers"] / n_customers_total, 1) if rows else None,
        }
    return {"total_rows_scanned": total_rows, "fields": report}


def analyze_fraud_episode_spans(raw_dir: Path) -> dict:
    """For historical closed cases with more than one transaction, measure
    the real elapsed time between the first and last transaction in the
    episode, grouped by pattern -- grounds the 'useful time window'
    recommendation in what actually happened, instead of re-deriving it
    from all 590k legitimate transactions.
    """
    closed = pd.read_csv(raw_dir / "closed_cases_history.csv", low_memory=False)
    multi = closed[closed["n_txns"] > 1].copy()
    multi["txn_id_list"] = multi["txn_ids"].apply(parse_pipe_separated_ids)

    all_ids: set[int] = set()
    for ids in multi["txn_id_list"]:
        all_ids.update(ids)

    ts_map = {}
    for chunk in pd.read_csv(raw_dir / "transactions.csv", usecols=["TransactionID", "ts"], chunksize=100_000, low_memory=False):
        hits = chunk[chunk["TransactionID"].isin(all_ids)]
        if len(hits):
            ts_map.update(dict(zip(hits["TransactionID"], hits["ts"])))
        if len(ts_map) == len(all_ids):
            break

    def span_minutes(txn_ids):
        timestamps = [ts_map[t] for t in txn_ids if t in ts_map]
        if len(timestamps) < 2:
            return None
        timestamps = pd.to_datetime(timestamps)
        return (timestamps.max() - timestamps.min()).total_seconds() / 60.0

    multi["span_minutes"] = multi["txn_id_list"].apply(span_minutes)
    multi = multi.dropna(subset=["span_minutes"])

    report = {}
    for pattern, group in multi.groupby("pattern"):
        report[pattern] = {
            "n_cases": len(group),
            "span_minutes_min": round(float(group["span_minutes"].min()), 1),
            "span_minutes_median": round(float(group["span_minutes"].median()), 1),
            "span_minutes_p90": round(float(group["span_minutes"].quantile(0.9)), 1),
            "span_minutes_max": round(float(group["span_minutes"].max()), 1),
        }
    return report
