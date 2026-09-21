"""
Resolves the organizer-provided `card_id` strings (e.g. "C12382-K1") used in
case_pack.csv and closed_cases_history.csv to the identity that actually
exists in transactions.csv: (customer_id, card1).

Why this indirection is necessary: transactions.csv has no card_id column,
and the K-suffix in card_id does not reliably track a distinct card1 value
-- dataset inspection found customers whose "K1" and "K2" cards share the
identical card1, differing only by missing card2-card6 on a subset of rows
(a real data gap concentrated 2016-09-27 to 2016-10-03). See
data/processed/card_identity_validation_report.json for the full evidence.

The only reliable way to attach a case to a specific card is therefore to
resolve the transaction IDs the case itself references and read their real
customer_id/card1 -- never to parse or trust the card_id string directly.
"""
from pathlib import Path

import pandas as pd

TRANSACTIONS_COLUMNS = ["TransactionID", "customer_id", "card1"]


def resolve_transactions(
    transactions_path: Path,
    txn_ids: set[int],
    chunksize: int = 100_000,
) -> dict[int, tuple[str, int]]:
    """Stream transactions.csv once and return {txn_id: (customer_id, card1)}
    for exactly the requested ids. Stops early once every id is found so a
    small lookup doesn't pay for a full 590k-row scan.
    """
    remaining = set(txn_ids)
    resolved: dict[int, tuple[str, int]] = {}
    if not remaining:
        return resolved

    for chunk in pd.read_csv(
        transactions_path, usecols=TRANSACTIONS_COLUMNS, chunksize=chunksize, low_memory=False
    ):
        hits = chunk[chunk["TransactionID"].isin(remaining)]
        if len(hits):
            for row in hits.itertuples(index=False):
                resolved[row.TransactionID] = (row.customer_id, int(row.card1))
            remaining -= set(hits["TransactionID"].tolist())
        if not remaining:
            break

    return resolved


def parse_pipe_separated_ids(value: str) -> list[int]:
    """closed_cases_history.csv's txn_ids column is pipe-separated (e.g.
    "3000120|3000121"). Shared here because both the validation experiment
    and the card-table builder need to walk the identical reference set.
    """
    return [int(t) for t in str(value).split("|")]
