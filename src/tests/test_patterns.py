"""
Tests for src.investigation.patterns.classify_card_testing (G5-A).

Pure-Python detector, no TigerGraph/MCP dependency -- every test here is
a plain synthetic or real-data fixture, no live connection needed, same
offline-testing style already used for _extract_cross_card_results in
test_cross_card_fraud_verification.py.

None of these tests assert an invented dollar threshold for "small" --
every fixture is designed so the match/no-match outcome follows from
count (>=3), the 60-minute window, the online filter, and whether a
strictly larger later transaction exists, never from a specific amount
being "below $X". None of them assert a "strongest" or "best" candidate
either -- the detector reports all qualifying candidates, unranked.
"""
from src.investigation.patterns import classify_card_testing


def _txn(txn_id, ts, amt, channel="online"):
    return {"TransactionID": txn_id, "ts": ts, "TransactionAmt": amt, "channel": channel}


# --- 1. empty temporal_activity ---

def test_empty_temporal_activity_does_not_match():
    result = classify_card_testing([], flagged_txn_id=1)
    assert result == {
        "matched": False,
        "pattern": "card_testing",
        "candidates": [],
        "uncertainty": [],
    }


# --- 2. fewer than 3 online transactions ---

def test_fewer_than_three_online_transactions_does_not_match():
    temporal = [
        _txn(1, "2016-01-01 10:00:00", 5.0),
        _txn(2, "2016-01-01 10:05:00", 6.0),
        _txn(3, "2016-01-01 10:10:00", 50.0, channel="in_person"),  # doesn't count -- not online
    ]
    result = classify_card_testing(temporal, flagged_txn_id=1)
    assert result["matched"] is False
    assert result["candidates"] == []


# --- 3. exactly 3 qualifying online transactions ---

def test_exactly_three_qualifying_online_transactions_matches():
    temporal = [
        _txn(1, "2016-01-01 10:00:00", 5.0),
        _txn(2, "2016-01-01 10:05:00", 6.0),
        _txn(3, "2016-01-01 10:10:00", 7.0),
        _txn(4, "2016-01-01 10:30:00", 50.0),
    ]
    result = classify_card_testing(temporal, flagged_txn_id=1)
    assert result["matched"] is True
    assert result["pattern"] == "card_testing"
    assert len(result["candidates"]) == 1
    c = result["candidates"][0]
    assert c["authorization_txn_ids"] == [1, 2, 3]
    assert c["authorization_count"] == 3
    assert c["authorization_span_minutes"] == 10.0
    assert c["larger_purchase_txn_id"] == 4
    assert c["larger_purchase_amount"] == 50.0
    assert len(result["uncertainty"]) == 1  # states no dollar threshold was used, and no ranking


# --- 4. 4+ qualifying transactions -- multiple overlapping candidate windows expected ---

def test_four_or_more_qualifying_transactions_yields_multiple_candidates():
    temporal = [
        _txn(1, "2016-01-01 10:00:00", 5.0),
        _txn(2, "2016-01-01 10:10:00", 5.0),
        _txn(3, "2016-01-01 10:20:00", 5.0),
        _txn(4, "2016-01-01 10:30:00", 5.0),  # same amount as the cluster -- must NOT count as "larger"
        _txn(5, "2016-01-01 11:00:00", 50.0),
    ]
    result = classify_card_testing(temporal, flagged_txn_id=1)
    assert result["matched"] is True
    # sliding windows [1,2,3] and [2,3,4] are both valid distinct clusters,
    # each followed by the same qualifying larger purchase (txn 5) --
    # neither is a duplicate of the other (different authorization_txn_ids),
    # so both must be reported, unranked.
    ids = [c["authorization_txn_ids"] for c in result["candidates"]]
    assert [1, 2, 3] in ids
    assert [2, 3, 4] in ids
    assert all(c["larger_purchase_txn_id"] == 5 for c in result["candidates"])


# --- 5. exactly 60-minute boundary (inclusive) ---

def test_span_exactly_sixty_minutes_matches():
    temporal = [
        _txn(1, "2016-01-01 10:00:00", 5.0),
        _txn(2, "2016-01-01 10:30:00", 6.0),
        _txn(3, "2016-01-01 11:00:00", 7.0),  # exactly 60.0 minutes after txn 1
        _txn(4, "2016-01-01 11:15:00", 50.0),
    ]
    result = classify_card_testing(temporal, flagged_txn_id=1)
    assert result["matched"] is True
    assert result["candidates"][0]["authorization_span_minutes"] == 60.0


# --- 6. more than 60 minutes ---

def test_span_over_sixty_minutes_does_not_match():
    temporal = [
        _txn(1, "2016-01-01 10:00:00", 5.0),
        _txn(2, "2016-01-01 10:30:00", 6.0),
        _txn(3, "2016-01-01 11:01:00", 7.0),  # 61 minutes after txn 1
    ]
    result = classify_card_testing(temporal, flagged_txn_id=1)
    assert result["matched"] is False
    assert result["candidates"] == []


# --- 7. larger transaction occurring BEFORE the cluster ---

def test_larger_transaction_before_cluster_does_not_count():
    temporal = [
        _txn(1, "2016-01-01 09:00:00", 100.0),  # large, but earlier than the cluster
        _txn(2, "2016-01-01 10:00:00", 5.0),
        _txn(3, "2016-01-01 10:05:00", 6.0),
        _txn(4, "2016-01-01 10:10:00", 7.0),
    ]
    result = classify_card_testing(temporal, flagged_txn_id=1)
    assert result["matched"] is False
    assert result["candidates"] == []


# --- 8. no larger transaction at all ---

def test_no_subsequent_larger_transaction_does_not_match():
    temporal = [
        _txn(1, "2016-01-01 10:00:00", 5.0),
        _txn(2, "2016-01-01 10:05:00", 6.0),
        _txn(3, "2016-01-01 10:10:00", 7.0),
    ]
    result = classify_card_testing(temporal, flagged_txn_id=1)
    assert result["matched"] is False
    assert result["candidates"] == []


# --- 9. non-online transactions excluded ---

def test_non_online_transactions_are_excluded_entirely():
    temporal = [
        _txn(1, "2016-01-01 10:00:00", 5.0),
        _txn(9, "2016-01-01 10:03:00", 1000.0, channel="in_person"),  # ignored: not online
        _txn(2, "2016-01-01 10:05:00", 6.0),
        _txn(3, "2016-01-01 10:10:00", 7.0),
        _txn(4, "2016-01-01 10:30:00", 50.0),
    ]
    result = classify_card_testing(temporal, flagged_txn_id=1)
    assert result["matched"] is True
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["authorization_txn_ids"] == [1, 2, 3]  # the in_person $1000 never appears
    assert result["candidates"][0]["larger_purchase_txn_id"] == 4


# --- 10. duplicate timestamps ---

def test_duplicate_timestamps_do_not_crash_and_still_classify():
    temporal = [
        _txn(1, "2016-01-01 10:00:00", 5.0),
        _txn(2, "2016-01-01 10:00:00", 6.0),  # same ts as txn 1
        _txn(3, "2016-01-01 10:05:00", 7.0),
        _txn(4, "2016-01-01 10:30:00", 50.0),
    ]
    result = classify_card_testing(temporal, flagged_txn_id=1)
    assert result["matched"] is True
    c = result["candidates"][0]
    assert c["authorization_count"] == 3
    assert c["authorization_span_minutes"] == 5.0


# --- 11. HHG-011 real benchmark fixture: multiple candidates must be preserved ---

def test_hhg_011_real_data_preserves_multiple_candidates():
    # Real temporal_activity for HHG-011 (card C11923:21363), pulled live
    # via q2_temporal_neighborhood during the G5 boundary analysis. This
    # card has enough online activity (part of the same C11923 history
    # behind Phase E's original N+1 discovery) to contain many
    # structurally-qualifying, overlapping R5 windows -- this test proves
    # the detector reports more than just the first one it finds, and
    # that both the weaker, earlier-discovered example and the more
    # dramatic one spotted during the original manual boundary analysis
    # are present among the reported candidates.
    temporal = [
        _txn(3580907, "2016-12-28 03:33:38", 16.17),
        _txn(3580933, "2016-12-28 03:57:10", 23.78),
        _txn(3580951, "2016-12-28 04:21:00", 52.17),
        _txn(3580992, "2016-12-28 05:30:26", 39.78),
        _txn(3581048, "2016-12-28 09:02:04", 37.08),
        _txn(3582082, "2016-12-28 18:54:14", 62.11),
        _txn(3582107, "2016-12-28 19:03:07", 62.06),
        _txn(3582142, "2016-12-28 19:13:13", 4.15),
        _txn(3583406, "2016-12-29 04:10:46", 19.55),
        _txn(3583411, "2016-12-29 04:13:36", 6.33),
        _txn(3583415, "2016-12-29 04:16:10", 6.39),
        _txn(3583417, "2016-12-29 04:22:35", 6.35),
        _txn(3583435, "2016-12-29 05:04:18", 22.93),
        _txn(3583670, "2016-12-29 13:23:32", 227.32),
    ]
    result = classify_card_testing(temporal, flagged_txn_id=3583368)
    assert result["matched"] is True
    assert len(result["candidates"]) > 1  # not just the first one found

    all_ids = [c["authorization_txn_ids"] for c in result["candidates"]]
    # the weaker, earliest-discoverable candidate is still present:
    assert [3580907, 3580933, 3580951] in all_ids
    # the tight, dramatic candidate the original manual analysis spotted
    # is ALSO present, not discarded in favor of only the first:
    assert [3583411, 3583415, 3583417] in all_ids
    dramatic = next(c for c in result["candidates"] if c["authorization_txn_ids"] == [3583411, 3583415, 3583417])
    assert dramatic["authorization_span_minutes"] == 8.98
    # note: this candidate's own larger_purchase is the next qualifying
    # transaction after it (3583435, $22.93) -- the $227.32 transaction
    # (3583670) appears as the larger_purchase for OTHER, later-starting
    # candidates in the full result, not this one; nothing here re-ranks
    # candidates to attach the most dramatic later transaction to every
    # cluster that could theoretically reach it.
    assert dramatic["larger_purchase_txn_id"] == 3583435
    assert dramatic["larger_purchase_amount"] == 22.93
    assert any(c["larger_purchase_amount"] == 227.32 for c in result["candidates"])
