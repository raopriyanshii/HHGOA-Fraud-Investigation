"""
Card-identity validation experiment (implementation-plan Phase 1, item 3).

Question this answers: can we safely resolve every card_id referenced in
case_pack.csv and closed_cases_history.csv to a single, consistent
(customer_id, card1) pair via the case's own transaction IDs -- and if not,
exactly where does it break?

This does not decide the graph schema; it only produces evidence. Any
exception found here must be reported, not silently patched over.
"""
from pathlib import Path

import pandas as pd

from src.data.card_resolution import parse_pipe_separated_ids, resolve_transactions


def validate(raw_dir: Path) -> dict:
    case_pack = pd.read_csv(raw_dir / "case_pack.csv")
    closed = pd.read_csv(raw_dir / "closed_cases_history.csv", low_memory=False)
    transactions_path = raw_dir / "transactions.csv"

    closed_txn_lists = {row.case_id: parse_pipe_separated_ids(row.txn_ids) for row in closed.itertuples()}

    all_txn_ids: set[int] = set(case_pack["flagged_txn_id"].tolist())
    for ids in closed_txn_lists.values():
        all_txn_ids.update(ids)

    resolved = resolve_transactions(transactions_path, all_txn_ids)

    report: dict = {"case_pack": [], "closed_cases": [], "exceptions": []}

    # ---- case_pack.card_id -> flagged_txn_id -> (customer_id, card1) ----
    for row in case_pack.itertuples():
        entry = {
            "case_id": row.case_id,
            "card_id": row.card_id,
            "flagged_txn_id": row.flagged_txn_id,
            "case_pack_customer_id": row.customer_id,
        }
        hit = resolved.get(row.flagged_txn_id)
        if hit is None:
            entry["status"] = "UNRESOLVED_TXN"
            report["exceptions"].append(entry)
        else:
            resolved_customer, resolved_card1 = hit
            card_id_prefix = row.card_id.split("-")[0]
            entry["resolved_customer_id"] = resolved_customer
            entry["resolved_card1"] = resolved_card1
            entry["status"] = (
                "OK" if resolved_customer == row.customer_id == card_id_prefix else "CUSTOMER_MISMATCH"
            )
            if entry["status"] != "OK":
                report["exceptions"].append(entry)
        report["case_pack"].append(entry)

    # ---- closed_cases_history.txn_ids -> (customer_id, card1) ----
    # Per-customer bookkeeping to separate two very different situations:
    #   (a) a customer has 2+ card_id LABELS that resolve to the SAME card1
    #       (the known missing-card2-6 artifact from the inspection phase)
    #   (b) a customer genuinely has 2+ DIFFERENT card1 values
    customer_labels: dict[str, set[str]] = {}
    customer_card1s: dict[str, set[int]] = {}
    label_to_card1s: dict[tuple[str, str], set[int]] = {}

    for row in closed.itertuples():
        txn_ids = closed_txn_lists[row.case_id]
        pairs = {resolved[t] for t in txn_ids if t in resolved}
        missing = [t for t in txn_ids if t not in resolved]

        entry = {
            "case_id": row.case_id,
            "card_id": row.card_id,
            "customer_id": row.customer_id,
            "n_txn_ids": len(txn_ids),
        }

        if missing:
            entry["status"] = "UNRESOLVED_TXN"
            entry["missing_txn_ids"] = missing
            report["exceptions"].append(entry)
        elif len(pairs) > 1:
            # every txn_id inside ONE case must point at the same card
            entry["status"] = "INTERNAL_INCONSISTENCY"
            entry["resolved_pairs"] = list(pairs)
            report["exceptions"].append(entry)
        else:
            (resolved_customer, resolved_card1), = pairs
            entry["resolved_customer_id"] = resolved_customer
            entry["resolved_card1"] = resolved_card1
            if resolved_customer != row.customer_id:
                entry["status"] = "CUSTOMER_MISMATCH"
                report["exceptions"].append(entry)
            else:
                entry["status"] = "OK"
                customer_labels.setdefault(row.customer_id, set()).add(row.card_id)
                customer_card1s.setdefault(row.customer_id, set()).add(resolved_card1)
                label_to_card1s.setdefault((row.customer_id, row.card_id), set()).add(resolved_card1)

        report["closed_cases"].append(entry)

    # a single card_id label should never resolve to two different card1
    # values across different case rows -- if it does, the label itself is
    # unreliable, not just multi-labeled customers
    inconsistent_labels = {k: v for k, v in label_to_card1s.items() if len(v) > 1}
    for (customer_id, card_id), card1s in inconsistent_labels.items():
        report["exceptions"].append(
            {
                "type": "LABEL_RESOLVES_TO_MULTIPLE_CARD1",
                "customer_id": customer_id,
                "card_id": card_id,
                "card1_values": list(card1s),
            }
        )

    multi_label_customers = {c for c, labels in customer_labels.items() if len(labels) > 1}
    multi_label_same_card1 = {c for c in multi_label_customers if len(customer_card1s[c]) == 1}
    multi_label_genuinely_multi_card1 = {c for c in multi_label_customers if len(customer_card1s[c]) > 1}

    report["summary"] = {
        "n_case_pack": len(case_pack),
        "n_case_pack_ok": sum(1 for e in report["case_pack"] if e["status"] == "OK"),
        "n_closed_cases": len(closed),
        "n_closed_cases_ok": sum(1 for e in report["closed_cases"] if e["status"] == "OK"),
        "n_exceptions": len(report["exceptions"]),
        "customers_with_multiple_card_id_labels": len(multi_label_customers),
        "  of_which_same_card1_data_gap_artifact": len(multi_label_same_card1),
        "  of_which_genuinely_multiple_card1_values": len(multi_label_genuinely_multi_card1),
        "card_id_labels_inconsistent_across_rows": len(inconsistent_labels),
    }
    report["genuinely_multi_card1_customers"] = sorted(multi_label_genuinely_multi_card1)
    report["same_card1_multi_label_customers"] = sorted(multi_label_same_card1)
    return report


if __name__ == "__main__":
    import json

    raw_dir = Path("D:/hackathon/task_5")
    result = validate(raw_dir)

    out_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "card_identity_validation_report.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(json.dumps(result["summary"], indent=2))
    print(f"\nTotal exceptions: {len(result['exceptions'])}")
    print(f"Full report written to: {out_path}")
