"""
Focused tests for the Card table builder, using a small hand-built fixture
(not the 675 MB real file) so this suite runs in well under a second.

Fixture design, deliberately covering every case the real dataset showed us:
  CX01 / card1=100  -> 3 txns, card4 always "visa", card6 is 2x "debit" +
                        1x "credit" (a real conflict; majority wins)
  CX02 / card1=200  -> 2 txns, no conflicts at all
  CX03              -> genuinely owns TWO different cards (300 and 400),
                        each with its own label (K1/K2) -- must become two
                        separate rows in the table
  CX04 / card1=500  -> ONE real card referenced by two different labels
                        (K1 and K2) -- the same-card1 artifact from Phase 1
  CX05/600, CX06/700 -> two different customers/cards that (only in this
                        synthetic fixture) get assigned the identical
                        external label "DUPLICATE-K1", to prove check #9
                        (no label may map to more than one card) actually
                        fires when it should.
"""
from pathlib import Path

import pandas as pd
import pytest

from src.data.build_card_table import (
    assemble_card_table,
    build_external_id_map,
    compute_transaction_aggregates,
    independent_recount,
    resolve_conflicted_value,
    run,
    verify_cases_against_card_table,
)

TRANSACTIONS = pd.DataFrame(
    [
        {"TransactionID": 2001, "customer_id": "CX01", "card1": 100, "card4": "visa", "card6": "debit", "ts": "2016-07-01 00:00:00"},
        {"TransactionID": 2002, "customer_id": "CX01", "card1": 100, "card4": "visa", "card6": "debit", "ts": "2016-07-02 00:00:00"},
        {"TransactionID": 2003, "customer_id": "CX01", "card1": 100, "card4": "visa", "card6": "credit", "ts": "2016-07-03 00:00:00"},
        {"TransactionID": 2004, "customer_id": "CX02", "card1": 200, "card4": "mastercard", "card6": "credit", "ts": "2016-08-01 00:00:00"},
        {"TransactionID": 2005, "customer_id": "CX02", "card1": 200, "card4": "mastercard", "card6": "credit", "ts": "2016-08-02 00:00:00"},
        {"TransactionID": 2006, "customer_id": "CX03", "card1": 300, "card4": "visa", "card6": "debit", "ts": "2016-09-01 00:00:00"},
        {"TransactionID": 2007, "customer_id": "CX03", "card1": 400, "card4": "mastercard", "card6": "debit", "ts": "2016-09-05 00:00:00"},
        {"TransactionID": 2008, "customer_id": "CX04", "card1": 500, "card4": "discover", "card6": "credit", "ts": "2016-10-01 00:00:00"},
        {"TransactionID": 2009, "customer_id": "CX04", "card1": 500, "card4": "discover", "card6": "credit", "ts": "2016-10-02 00:00:00"},
        {"TransactionID": 2010, "customer_id": "CX05", "card1": 600, "card4": "amex", "card6": "credit", "ts": "2016-11-01 00:00:00"},
        {"TransactionID": 2011, "customer_id": "CX06", "card1": 700, "card4": "amex", "card6": "credit", "ts": "2016-11-02 00:00:00"},
    ]
)

CASE_PACK = pd.DataFrame(
    [
        {"case_id": "HHG-T1", "flagged_txn_id": 2001, "card_id": "CX01-K1", "customer_id": "CX01"},
        {"case_id": "HHG-T2", "flagged_txn_id": 2004, "card_id": "CX02-K1", "customer_id": "CX02"},
    ]
)

CLOSED_CASES = pd.DataFrame(
    [
        {"case_id": "CC-A", "customer_id": "CX01", "card_id": "CX01-K1", "txn_ids": "2001|2002"},
        {"case_id": "CC-B", "customer_id": "CX03", "card_id": "CX03-K1", "txn_ids": "2006"},
        {"case_id": "CC-C", "customer_id": "CX03", "card_id": "CX03-K2", "txn_ids": "2007"},
        {"case_id": "CC-D", "customer_id": "CX04", "card_id": "CX04-K1", "txn_ids": "2008"},
        {"case_id": "CC-E", "customer_id": "CX04", "card_id": "CX04-K2", "txn_ids": "2009"},
        {"case_id": "CC-F", "customer_id": "CX05", "card_id": "DUPLICATE-K1", "txn_ids": "2010"},
        {"case_id": "CC-G", "customer_id": "CX06", "card_id": "DUPLICATE-K1", "txn_ids": "2011"},
    ]
)


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    TRANSACTIONS.to_csv(tmp_path / "transactions.csv", index=False)
    CASE_PACK.to_csv(tmp_path / "case_pack.csv", index=False)
    CLOSED_CASES.to_csv(tmp_path / "closed_cases_history.csv", index=False)
    (tmp_path / "README.md").write_text("test fixture\n")
    pd.DataFrame({"TransactionID": []}).to_csv(tmp_path / "identity.csv", index=False)
    return tmp_path


def test_resolve_conflicted_value_no_conflict():
    from collections import Counter

    value, conflict = resolve_conflicted_value(Counter({"visa": 3}))
    assert value == "visa"
    assert conflict is False


def test_resolve_conflicted_value_majority_wins():
    from collections import Counter

    value, conflict = resolve_conflicted_value(Counter({"debit": 2, "credit": 1}))
    assert value == "debit"
    assert conflict is True


def test_resolve_conflicted_value_tie_breaks_lexicographically():
    from collections import Counter

    value, conflict = resolve_conflicted_value(Counter({"b": 2, "a": 2}))
    assert value == "a"
    assert conflict is True


def test_resolve_conflicted_value_empty():
    from collections import Counter

    value, conflict = resolve_conflicted_value(Counter())
    assert value is None
    assert conflict is False


def test_compute_transaction_aggregates(raw_dir):
    aggregates, total_rows = compute_transaction_aggregates(raw_dir / "transactions.csv")
    assert total_rows == 11
    cx01 = aggregates[("CX01", 100)]
    assert cx01["count"] == 3
    assert cx01["first_seen_ts"] == "2016-07-01 00:00:00"
    assert cx01["last_seen_ts"] == "2016-07-03 00:00:00"
    assert dict(cx01["card4_counts"]) == {"visa": 3}
    assert dict(cx01["card6_counts"]) == {"debit": 2, "credit": 1}


def test_build_external_id_map_dedupes_same_source(raw_dir):
    external_ids, issues = build_external_id_map(raw_dir)
    assert issues == []
    # CX01-K1 appears in BOTH case_pack and closed_cases_history for the
    # same underlying card -- must collapse to one label, not duplicate
    assert external_ids[("CX01", 100)] == {"CX01-K1"}


def test_assemble_card_table_flags_card6_conflict_and_takes_majority(raw_dir):
    aggregates, _ = compute_transaction_aggregates(raw_dir / "transactions.csv")
    external_ids, _ = build_external_id_map(raw_dir)
    df, conflicts = assemble_card_table(aggregates, external_ids)

    cx01 = df[df["card_key"] == "CX01:100"].iloc[0]
    assert cx01["card6"] == "debit"
    assert bool(cx01["card6_conflict"]) is True
    assert bool(cx01["card4_conflict"]) is False
    assert len(conflicts["card6"]) == 1
    assert conflicts["card6"][0]["customer_id"] == "CX01"


def test_assemble_card_table_keeps_genuine_multicard_customer_as_two_rows(raw_dir):
    aggregates, _ = compute_transaction_aggregates(raw_dir / "transactions.csv")
    external_ids, _ = build_external_id_map(raw_dir)
    df, _ = assemble_card_table(aggregates, external_ids)

    cx03_rows = df[df["customer_id"] == "CX03"]
    assert len(cx03_rows) == 2
    assert set(cx03_rows["card1"]) == {300, 400}
    assert set(cx03_rows["external_ids"]) == {"CX03-K1", "CX03-K2"}


def test_assemble_card_table_merges_same_card1_multilabel_into_one_row(raw_dir):
    aggregates, _ = compute_transaction_aggregates(raw_dir / "transactions.csv")
    external_ids, _ = build_external_id_map(raw_dir)
    df, _ = assemble_card_table(aggregates, external_ids)

    cx04_rows = df[df["customer_id"] == "CX04"]
    assert len(cx04_rows) == 1
    assert cx04_rows.iloc[0]["external_ids"] == "CX04-K1|CX04-K2"


def test_independent_recount_matches_main_aggregation(raw_dir):
    aggregates, total_rows = compute_transaction_aggregates(raw_dir / "transactions.csv")
    pairs, counts, recount_total_rows = independent_recount(raw_dir / "transactions.csv")

    assert recount_total_rows == total_rows
    assert pairs == set(aggregates.keys())
    for key, agg in aggregates.items():
        assert counts[key] == agg["count"]


def test_verify_cases_detects_duplicate_label_across_two_cards(raw_dir):
    aggregates, _ = compute_transaction_aggregates(raw_dir / "transactions.csv")
    external_ids, _ = build_external_id_map(raw_dir)
    df, _ = assemble_card_table(aggregates, external_ids)

    result = verify_cases_against_card_table(raw_dir, df)
    assert result["case_pack"]["ok"] == 2
    assert result["closed_cases"]["ok"] == 7  # every case is internally consistent on its own
    # but "DUPLICATE-K1" was (in this synthetic fixture only) attached to
    # two different real cards -- that must be caught, not silently merged
    assert "DUPLICATE-K1" in result["labels_mapping_to_multiple_cards"]
    assert len(result["labels_mapping_to_multiple_cards"]["DUPLICATE-K1"]) == 2


def test_full_run_is_deterministic_across_two_invocations(raw_dir, tmp_path):
    out_dir_1 = tmp_path / "out1"
    out_dir_2 = tmp_path / "out2"

    run(raw_dir, out_dir_1)
    run(raw_dir, out_dir_2)

    csv_1 = (out_dir_1 / "cards.csv").read_text()
    csv_2 = (out_dir_2 / "cards.csv").read_text()
    assert csv_1 == csv_2


def test_full_run_reports_zero_exceptions_on_clean_fixture_subset(raw_dir, tmp_path):
    # CC-F/CC-G's shared label is the one deliberate anomaly in this fixture;
    # everything else must validate cleanly
    report = run(raw_dir, tmp_path / "out")
    v = report["validation_results"]
    assert v["2_no_duplicate_card_keys"]["duplicate_count"] == 0
    assert v["3_every_transaction_pair_has_one_card_record"]["match"] is True
    assert v["4_every_card_has_at_least_one_transaction"]["min_transaction_count"] >= 1
    assert v["5_first_seen_le_last_seen"]["violations"] == 0
    assert v["6_transaction_count_matches_independent_recount"]["mismatches"] == 0
    assert v["7_benchmark_cases_resolve"]["ok"] == 2
    assert v["8_historical_cases_resolve"]["ok"] == 7
    assert v["10_raw_files_unmodified"]["match"] is True
    assert len(v["9_no_label_maps_to_multiple_cards"]["violations"]) == 1
