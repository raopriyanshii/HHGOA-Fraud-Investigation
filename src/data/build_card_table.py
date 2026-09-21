"""
Builds the Card dimension table: one row per unique (customer_id, card1)
observed in transactions.csv, ready for a later TigerGraph loading job.

Design decisions carried over from the Phase 1 validation (zero exceptions
found across all 20 case-pack cases and all 5,565 historical cases):

- The Card identity is (customer_id, card1). card1 is the only field that
  is both zero-null and stable per physical card; card2/card3/card5/card6
  are noisy/missing on a per-transaction basis and must never be used to
  split one card into several.
- Organizer-provided `card_id` strings ("C12382-K1") are resolved through
  each case's own transaction IDs (via card_resolution.resolve_transactions)
  and kept only as traceability labels in `external_ids` -- never parsed
  for meaning.
- card4/card6 are transaction-level fields that are *usually* constant for
  a given card1 but are not guaranteed to be. Where more than one distinct
  non-null value is observed for the same card, that is a real conflict in
  the source data, not a bug in this script. The documented, deterministic
  policy is: take the most frequent value, break ties by the lexicographically
  smallest value, and record every such conflict (both in the output table's
  `card4_conflict`/`card6_conflict` flags and in the validation report) so it
  is never silently hidden.

Everything here streams transactions.csv in chunks; the full file is never
held in memory at once.
"""
from __future__ import annotations

import hashlib
import time
import tracemalloc
from collections import Counter
from pathlib import Path

import pandas as pd

from src.data.card_resolution import parse_pipe_separated_ids, resolve_transactions

AGGREGATE_COLUMNS = ["customer_id", "card1", "card4", "card6", "ts"]
RECOUNT_COLUMNS = ["customer_id", "card1"]
RAW_FILES = ["README.md", "transactions.csv", "identity.csv", "closed_cases_history.csv", "case_pack.csv"]


# ---------------------------------------------------------------------------
# Pass 1: main aggregation (the values that become cards.csv)
# ---------------------------------------------------------------------------

def compute_transaction_aggregates(transactions_path: Path, chunksize: int = 100_000):
    """Stream transactions.csv once, grouped by (customer_id, card1).

    Returns (aggregates, total_rows) where aggregates maps
    (customer_id, card1) -> {"count", "first_seen_ts", "last_seen_ts",
    "card4_counts": Counter, "card6_counts": Counter}.
    """
    aggregates: dict[tuple[str, int], dict] = {}
    total_rows = 0

    for chunk in pd.read_csv(transactions_path, usecols=AGGREGATE_COLUMNS, chunksize=chunksize, low_memory=False):
        total_rows += len(chunk)

        grouped = chunk.groupby(["customer_id", "card1"]).agg(
            count=("ts", "size"), first_seen_ts=("ts", "min"), last_seen_ts=("ts", "max")
        )
        for (customer_id, card1), row in grouped.iterrows():
            entry = aggregates.setdefault(
                (customer_id, int(card1)),
                {"count": 0, "first_seen_ts": None, "last_seen_ts": None, "card4_counts": Counter(), "card6_counts": Counter()},
            )
            entry["count"] += int(row["count"])
            entry["first_seen_ts"] = row["first_seen_ts"] if entry["first_seen_ts"] is None else min(entry["first_seen_ts"], row["first_seen_ts"])
            entry["last_seen_ts"] = row["last_seen_ts"] if entry["last_seen_ts"] is None else max(entry["last_seen_ts"], row["last_seen_ts"])

        for col, counter_key in (("card4", "card4_counts"), ("card6", "card6_counts")):
            value_counts = chunk.dropna(subset=[col]).groupby(["customer_id", "card1", col]).size()
            for (customer_id, card1, value), n in value_counts.items():
                key = (customer_id, int(card1))
                aggregates[key][counter_key][value] += int(n)

    return aggregates, total_rows


def resolve_conflicted_value(counter: Counter) -> tuple[str | None, bool]:
    """Documented, deterministic policy for a card4/card6 value observed
    more than one way for the same (customer_id, card1): take the most
    frequent value; break ties by the lexicographically smallest value.
    `conflict=True` whenever more than one distinct non-null value was
    observed at all, even if one clearly dominates -- so a caller can
    always see this wasn't a free choice.
    """
    if not counter:
        return None, False
    max_count = max(counter.values())
    candidates = sorted(value for value, count in counter.items() if count == max_count)
    return candidates[0], len(counter) > 1


# ---------------------------------------------------------------------------
# External IDs: resolve case_pack.csv / closed_cases_history.csv card_id
# labels through their transactions, exactly as validated in Phase 1.
# ---------------------------------------------------------------------------

def build_external_id_map(raw_dir: Path):
    """Returns (external_ids, issues).

    external_ids maps (customer_id, card1) -> set of original card_id
    strings that resolved to it. issues lists any case that failed to
    resolve cleanly (expected to be empty, per the Phase 1 result).
    """
    case_pack = pd.read_csv(raw_dir / "case_pack.csv")
    closed = pd.read_csv(raw_dir / "closed_cases_history.csv", low_memory=False)

    closed_txn_lists = {row.case_id: parse_pipe_separated_ids(row.txn_ids) for row in closed.itertuples()}
    all_txn_ids: set[int] = set(case_pack["flagged_txn_id"].tolist())
    for ids in closed_txn_lists.values():
        all_txn_ids.update(ids)

    resolved = resolve_transactions(raw_dir / "transactions.csv", all_txn_ids)

    external_ids: dict[tuple[str, int], set[str]] = {}
    issues: list[dict] = []

    for row in case_pack.itertuples():
        hit = resolved.get(row.flagged_txn_id)
        if hit is None:
            issues.append({"source": "case_pack", "case_id": row.case_id, "reason": "UNRESOLVED_TXN"})
            continue
        resolved_customer, resolved_card1 = hit
        if resolved_customer != row.customer_id:
            issues.append({"source": "case_pack", "case_id": row.case_id, "reason": "CUSTOMER_MISMATCH"})
            continue
        external_ids.setdefault((resolved_customer, resolved_card1), set()).add(row.card_id)

    for row in closed.itertuples():
        txn_ids = closed_txn_lists[row.case_id]
        pairs = {resolved[t] for t in txn_ids if t in resolved}
        missing = [t for t in txn_ids if t not in resolved]
        if missing:
            issues.append({"source": "closed_cases", "case_id": row.case_id, "reason": "UNRESOLVED_TXN", "missing": missing})
            continue
        if len(pairs) > 1:
            issues.append({"source": "closed_cases", "case_id": row.case_id, "reason": "INTERNAL_INCONSISTENCY", "pairs": list(pairs)})
            continue
        (resolved_customer, resolved_card1), = pairs
        if resolved_customer != row.customer_id:
            issues.append({"source": "closed_cases", "case_id": row.case_id, "reason": "CUSTOMER_MISMATCH"})
            continue
        external_ids.setdefault((resolved_customer, resolved_card1), set()).add(row.card_id)

    return external_ids, issues


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def assemble_card_table(aggregates: dict, external_ids: dict) -> tuple[pd.DataFrame, dict]:
    rows = []
    conflicts = {"card4": [], "card6": []}

    for (customer_id, card1), agg in aggregates.items():
        card4_value, card4_conflict = resolve_conflicted_value(agg["card4_counts"])
        card6_value, card6_conflict = resolve_conflicted_value(agg["card6_counts"])

        if card4_conflict:
            conflicts["card4"].append(
                {"customer_id": customer_id, "card1": card1, "observed": dict(agg["card4_counts"]), "chosen": card4_value}
            )
        if card6_conflict:
            conflicts["card6"].append(
                {"customer_id": customer_id, "card1": card1, "observed": dict(agg["card6_counts"]), "chosen": card6_value}
            )

        labels = sorted(external_ids.get((customer_id, card1), set()))
        rows.append(
            {
                "card_key": f"{customer_id}:{card1}",
                "customer_id": customer_id,
                "card1": card1,
                "card4": card4_value or "",
                "card4_conflict": card4_conflict,
                "card6": card6_value or "",
                "card6_conflict": card6_conflict,
                "first_seen_ts": agg["first_seen_ts"],
                "last_seen_ts": agg["last_seen_ts"],
                "transaction_count": agg["count"],
                "external_ids": "|".join(labels),
            }
        )

    # deterministic order: never rely on dict/groupby iteration order
    df = pd.DataFrame(rows).sort_values(["customer_id", "card1"]).reset_index(drop=True)
    return df, conflicts


# ---------------------------------------------------------------------------
# Independent cross-checks (deliberately separate code path from the main
# aggregation above, so a bug in one is unlikely to be repeated in the other)
# ---------------------------------------------------------------------------

def independent_recount(transactions_path: Path, chunksize: int = 100_000):
    pairs: set[tuple[str, int]] = set()
    counts: Counter = Counter()
    total_rows = 0

    for chunk in pd.read_csv(transactions_path, usecols=RECOUNT_COLUMNS, chunksize=chunksize, low_memory=False):
        total_rows += len(chunk)
        sizes = chunk.groupby(["customer_id", "card1"]).size()
        for (customer_id, card1), n in sizes.items():
            key = (customer_id, int(card1))
            pairs.add(key)
            counts[key] += int(n)

    return pairs, counts, total_rows


def verify_cases_against_card_table(raw_dir: Path, card_df: pd.DataFrame) -> dict:
    card_keys = set(zip(card_df["customer_id"], card_df["card1"]))
    label_to_keys: dict[str, set[tuple[str, int]]] = {}
    for row in card_df.itertuples():
        if row.external_ids:
            for label in row.external_ids.split("|"):
                label_to_keys.setdefault(label, set()).add((row.customer_id, row.card1))

    case_pack = pd.read_csv(raw_dir / "case_pack.csv")
    closed = pd.read_csv(raw_dir / "closed_cases_history.csv", low_memory=False)
    closed_txn_lists = {row.case_id: parse_pipe_separated_ids(row.txn_ids) for row in closed.itertuples()}

    all_txn_ids: set[int] = set(case_pack["flagged_txn_id"].tolist())
    for ids in closed_txn_lists.values():
        all_txn_ids.update(ids)
    resolved = resolve_transactions(raw_dir / "transactions.csv", all_txn_ids)

    def check(case_id, customer_id, card_id, txn_ids):
        pairs = {resolved[t] for t in txn_ids if t in resolved}
        if len(pairs) != 1:
            return {"case_id": case_id, "status": "UNRESOLVED_OR_INCONSISTENT"}
        (resolved_customer, resolved_card1), = pairs
        key = (resolved_customer, resolved_card1)
        in_table = key in card_keys
        label_recorded = key in label_to_keys.get(card_id, set())
        status = "OK" if (in_table and label_recorded and resolved_customer == customer_id) else "MISMATCH"
        return {"case_id": case_id, "status": status, "resolved_key": key}

    case_pack_results = [
        check(row.case_id, row.customer_id, row.card_id, [row.flagged_txn_id]) for row in case_pack.itertuples()
    ]
    closed_results = [
        check(row.case_id, row.customer_id, row.card_id, closed_txn_lists[row.case_id]) for row in closed.itertuples()
    ]

    return {
        "case_pack": {
            "total": len(case_pack_results),
            "ok": sum(1 for r in case_pack_results if r["status"] == "OK"),
            "failures": [r for r in case_pack_results if r["status"] != "OK"],
        },
        "closed_cases": {
            "total": len(closed_results),
            "ok": sum(1 for r in closed_results if r["status"] == "OK"),
            "failures": [r for r in closed_results if r["status"] != "OK"],
        },
        # check 9: no external_id label maps to more than one card key
        "labels_mapping_to_multiple_cards": {k: list(v) for k, v in label_to_keys.items() if len(v) > 1},
    }


def hash_raw_files(raw_dir: Path) -> dict[str, str]:
    hashes = {}
    for name in RAW_FILES:
        path = raw_dir / name
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                digest.update(block)
        hashes[name] = digest.hexdigest()
    return hashes


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(raw_dir: Path, out_dir: Path) -> dict:
    timings: dict[str, float] = {}
    t_total_start = time.perf_counter()

    t0 = time.perf_counter()
    hashes_before = hash_raw_files(raw_dir)
    timings["hash_before_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    aggregates, total_rows = compute_transaction_aggregates(raw_dir / "transactions.csv")
    timings["main_aggregation_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    external_ids, external_id_issues = build_external_id_map(raw_dir)
    timings["external_id_resolution_s"] = time.perf_counter() - t0

    df, conflicts = assemble_card_table(aggregates, external_ids)

    t0 = time.perf_counter()
    independent_pairs, independent_counts, independent_total_rows = independent_recount(raw_dir / "transactions.csv")
    timings["independent_recount_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    case_check = verify_cases_against_card_table(raw_dir, df)
    timings["case_verification_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    hashes_after = hash_raw_files(raw_dir)
    timings["hash_after_s"] = time.perf_counter() - t0

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "cards.csv", index=False)

    # --- the 10 requested validation checks ---
    table_keys = set(zip(df["customer_id"], df["card1"]))
    table_counts = {(row.customer_id, row.card1): row.transaction_count for row in df.itertuples()}
    count_mismatches = [
        key for key, count in table_counts.items() if count != independent_counts.get(key, -1)
    ]
    validations = {
        "1_unique_pairs": {"card_table": len(df), "independent_recount": len(independent_pairs), "match": len(df) == len(independent_pairs)},
        "2_no_duplicate_card_keys": {"duplicate_count": int(df["card_key"].duplicated().sum())},
        "3_every_transaction_pair_has_one_card_record": {"match": table_keys == independent_pairs, "table_only": len(table_keys - independent_pairs), "recount_only": len(independent_pairs - table_keys)},
        "4_every_card_has_at_least_one_transaction": {"min_transaction_count": int(df["transaction_count"].min())},
        "5_first_seen_le_last_seen": {"violations": int((df["first_seen_ts"] > df["last_seen_ts"]).sum())},
        "6_transaction_count_matches_independent_recount": {"mismatches": len(count_mismatches), "examples": count_mismatches[:5]},
        "7_benchmark_cases_resolve": case_check["case_pack"],
        "8_historical_cases_resolve": case_check["closed_cases"],
        "9_no_label_maps_to_multiple_cards": {"violations": case_check["labels_mapping_to_multiple_cards"]},
        "10_raw_files_unmodified": {"match": hashes_before == hashes_after, "hashes_before": hashes_before, "hashes_after": hashes_after},
        "row_count_consistency": {
            "main_pass_total_rows": total_rows,
            "independent_pass_total_rows": independent_total_rows,
            "match": total_rows == independent_total_rows,
        },
        "external_id_resolution_issues": external_id_issues,
    }

    timings["total_s"] = time.perf_counter() - t_total_start

    report = {
        "input_file": str(raw_dir / "transactions.csv"),
        "processing_method": "chunked pandas.read_csv (chunksize=100000); full file never held in memory at once",
        "row_count": total_rows,
        "unique_card_count": len(df),
        "duplicate_count": int(df["card_key"].duplicated().sum()),
        "unresolved_references": external_id_issues,
        "conflicting_attributes": conflicts,
        "benchmark_resolution_status": case_check["case_pack"],
        "historical_resolution_status": case_check["closed_cases"],
        "validation_results": validations,
        "processing_time_seconds": timings,
    }
    return report


def main():
    raw_dir = Path("D:/hackathon/task_5")
    out_dir = Path(__file__).resolve().parents[2] / "data" / "processed"

    tracemalloc.start()
    report = run(raw_dir, out_dir)
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    report["memory"] = {
        "tracemalloc_peak_mb": round(peak_bytes / (1024 * 1024), 1),
        "note": "Python-object peak via stdlib tracemalloc, not full process RSS -- psutil was not added as a dependency for this one measurement.",
    }

    import json

    with open(out_dir / "card_table_validation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"cards.csv rows: {report['unique_card_count']}")
    print(f"total transaction rows processed: {report['row_count']}")
    print(f"duplicate card keys: {report['duplicate_count']}")
    print(f"conflicting card4 cards: {len(report['conflicting_attributes']['card4'])}")
    print(f"conflicting card6 cards: {len(report['conflicting_attributes']['card6'])}")
    print(f"benchmark cases OK: {report['benchmark_resolution_status']['ok']}/{report['benchmark_resolution_status']['total']}")
    print(f"historical cases OK: {report['historical_resolution_status']['ok']}/{report['historical_resolution_status']['total']}")
    print(f"raw files unmodified: {report['validation_results']['10_raw_files_unmodified']['match']}")
    print(f"total time: {report['processing_time_seconds']['total_s']:.1f}s")
    print(f"peak memory (tracemalloc): {report['memory']['tracemalloc_peak_mb']} MB")


if __name__ == "__main__":
    main()
