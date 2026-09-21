"""
Small, fast tests for the card-identity resolution and validation logic,
using a hand-built fixture instead of the 675 MB real transactions.csv.

These exist to prove the classification logic itself is correct (OK /
CUSTOMER_MISMATCH / INTERNAL_INCONSISTENCY / UNRESOLVED_TXN / label
inconsistency / genuine multi-card detection) before trusting its output on
the real dataset.
"""
from pathlib import Path

import pandas as pd
import pytest

from src.data.card_resolution import resolve_transactions
from src.data.validate_card_identity import validate

# Fixture transaction pool:
#   1001, 1002 -> customer C0001, card1 111   (one real card)
#   1006       -> customer C0001, card1 444   (C0001's SECOND, genuinely different card)
#   1003, 1004 -> customer C0002, card1 222
#   1005       -> customer C0003, card1 333
#   1007, 1008 -> customer C0004, card1 555   (one real card, referenced under two different labels)
#   1009       -> customer C0005, card1 666   (C0005's first real card, label K1)
#   1010       -> customer C0005, card1 777   (C0005's second real card, label K2)
TRANSACTIONS = pd.DataFrame(
    [
        {"TransactionID": 1001, "customer_id": "C0001", "card1": 111},
        {"TransactionID": 1002, "customer_id": "C0001", "card1": 111},
        {"TransactionID": 1006, "customer_id": "C0001", "card1": 444},
        {"TransactionID": 1003, "customer_id": "C0002", "card1": 222},
        {"TransactionID": 1004, "customer_id": "C0002", "card1": 222},
        {"TransactionID": 1005, "customer_id": "C0003", "card1": 333},
        {"TransactionID": 1007, "customer_id": "C0004", "card1": 555},
        {"TransactionID": 1008, "customer_id": "C0004", "card1": 555},
        {"TransactionID": 1009, "customer_id": "C0005", "card1": 666},
        {"TransactionID": 1010, "customer_id": "C0005", "card1": 777},
    ]
)

CASE_PACK = pd.DataFrame(
    [
        # normal, resolvable case
        {"case_id": "HHG-T1", "flagged_txn_id": 1001, "card_id": "C0001-K1", "customer_id": "C0001"},
        # flagged txn id that doesn't exist anywhere
        {"case_id": "HHG-T2", "flagged_txn_id": 9999, "card_id": "C0002-K1", "customer_id": "C0002"},
    ]
)

CLOSED_CASES = pd.DataFrame(
    [
        # OK: both txns belong to the same (customer, card1)
        {"case_id": "CC-A", "customer_id": "C0001", "card_id": "C0001-K1", "txn_ids": "1001|1002"},
        # INTERNAL_INCONSISTENCY: txn_ids span two different (customer, card1) pairs
        {"case_id": "CC-B", "customer_id": "C0002", "card_id": "C0002-K1", "txn_ids": "1003|1005"},
        # CUSTOMER_MISMATCH: the case says C9999 but the transaction really belongs to C0002
        {"case_id": "CC-C", "customer_id": "C9999", "card_id": "C9999-K1", "txn_ids": "1004"},
        # multi-label / same-card1 artifact: two DIFFERENT labels for C0004,
        # both resolving to the SAME real card1 (555) -- the known
        # missing-card2-6 style artifact, not a real second card
        {"case_id": "CC-D", "customer_id": "C0004", "card_id": "C0004-K1", "txn_ids": "1007"},
        # UNRESOLVED_TXN: references a transaction id that doesn't exist
        {"case_id": "CC-E", "customer_id": "C0003", "card_id": "C0003-K1", "txn_ids": "8888"},
        # genuinely multiple card1 values under the SAME label as CC-A
        # (C0001-K1 here resolves to 444, but CC-A's C0001-K1 resolved to 111)
        {"case_id": "CC-F", "customer_id": "C0001", "card_id": "C0001-K1", "txn_ids": "1006"},
        # second label for C0004, also resolving to card1=555
        {"case_id": "CC-G", "customer_id": "C0004", "card_id": "C0004-K2", "txn_ids": "1008"},
        # C0005: two DIFFERENT labels resolving to two DIFFERENT real cards
        # -- the clean, correctly-labeled multi-card scenario
        {"case_id": "CC-H", "customer_id": "C0005", "card_id": "C0005-K1", "txn_ids": "1009"},
        {"case_id": "CC-I", "customer_id": "C0005", "card_id": "C0005-K2", "txn_ids": "1010"},
    ]
)


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    TRANSACTIONS.to_csv(tmp_path / "transactions.csv", index=False)
    CASE_PACK.to_csv(tmp_path / "case_pack.csv", index=False)
    CLOSED_CASES.to_csv(tmp_path / "closed_cases_history.csv", index=False)
    return tmp_path


def test_resolve_transactions_finds_requested_ids_only(raw_dir):
    resolved = resolve_transactions(raw_dir / "transactions.csv", {1001, 1003})
    assert resolved == {1001: ("C0001", 111), 1003: ("C0002", 222)}


def test_resolve_transactions_missing_id_is_simply_absent(raw_dir):
    resolved = resolve_transactions(raw_dir / "transactions.csv", {1001, 9999})
    assert 1001 in resolved
    assert 9999 not in resolved


def test_case_pack_ok_and_unresolved(raw_dir):
    report = validate(raw_dir)
    by_case = {e["case_id"]: e for e in report["case_pack"]}
    assert by_case["HHG-T1"]["status"] == "OK"
    assert by_case["HHG-T1"]["resolved_card1"] == 111
    assert by_case["HHG-T2"]["status"] == "UNRESOLVED_TXN"


def test_closed_case_ok(raw_dir):
    report = validate(raw_dir)
    by_case = {e["case_id"]: e for e in report["closed_cases"]}
    assert by_case["CC-A"]["status"] == "OK"
    assert by_case["CC-A"]["resolved_card1"] == 111


def test_closed_case_internal_inconsistency_detected(raw_dir):
    report = validate(raw_dir)
    by_case = {e["case_id"]: e for e in report["closed_cases"]}
    assert by_case["CC-B"]["status"] == "INTERNAL_INCONSISTENCY"
    assert any(e["case_id"] == "CC-B" for e in report["exceptions"])


def test_closed_case_customer_mismatch_detected(raw_dir):
    report = validate(raw_dir)
    by_case = {e["case_id"]: e for e in report["closed_cases"]}
    assert by_case["CC-C"]["status"] == "CUSTOMER_MISMATCH"
    assert any(e["case_id"] == "CC-C" for e in report["exceptions"])


def test_closed_case_unresolved_txn_detected(raw_dir):
    report = validate(raw_dir)
    by_case = {e["case_id"]: e for e in report["closed_cases"]}
    assert by_case["CC-E"]["status"] == "UNRESOLVED_TXN"
    assert any(e["case_id"] == "CC-E" for e in report["exceptions"])


def test_multi_label_same_card1_is_flagged_as_data_gap_not_real_multicard(raw_dir):
    report = validate(raw_dir)
    # C0004 has two labels (K1 from CC-D, K2 from CC-G) both resolving to card1=555
    assert "C0004" in report["same_card1_multi_label_customers"]
    assert "C0004" not in report["genuinely_multi_card1_customers"]
    assert report["summary"]["  of_which_same_card1_data_gap_artifact"] >= 1


def test_genuine_multi_card1_customer_is_detected(raw_dir):
    report = validate(raw_dir)
    # C0005-K1 -> card1=666, C0005-K2 -> card1=777: two distinct labels,
    # two distinct real cards -- a correctly-labeled multi-card customer
    assert "C0005" in report["genuinely_multi_card1_customers"]
    assert "C0005" not in report["same_card1_multi_label_customers"]


def test_inconsistent_label_is_flagged(raw_dir):
    report = validate(raw_dir)
    # card_id "C0001-K1" resolves to 111 in CC-A but 444 in CC-F -> same
    # label, two different card1 values -> must be reported as an exception
    label_exceptions = [e for e in report["exceptions"] if e.get("type") == "LABEL_RESOLVES_TO_MULTIPLE_CARD1"]
    assert len(label_exceptions) == 1
    assert label_exceptions[0]["card_id"] == "C0001-K1"
    assert set(label_exceptions[0]["card1_values"]) == {111, 444}
