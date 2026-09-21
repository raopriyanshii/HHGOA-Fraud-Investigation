"""
Phase E deterministic investigation query layer.

Every function here is READ-ONLY and returns plain, JSON-serializable
Python structures (dicts/lists) -- no LLM, no verdict, no pattern
classification. All graph access goes through INTERPRET QUERY (ephemeral,
never installed as a permanent GSQL object) via the existing
src.graph.tigergraph_connection.get_connection(), so nothing here touches
the schema, loading jobs, or .env.

Field curation: every *_FIELDS tuple below is the deliberately small,
investigation-relevant subset documented in docs/INVESTIGATION_QUERY_SPEC.md
-- vertices are fetched in full from the graph (already curated at schema
level to <= 48 attributes, never the original 397), then filtered down to
these fields in Python. This keeps the GSQL simple (no per-attribute
projection syntax) while still honoring "don't return everything."

Chronology: every window/order uses `ts` (lexicographically sortable
"YYYY-MM-DD HH:MM:SS" strings), never TransactionID/TransactionDT.
"""
from __future__ import annotations

from datetime import datetime, timedelta

TRANSACTION_FIELDS = (
    "TransactionID", "ts", "TransactionAmt", "ProductCD", "channel", "risk_score",
    "addr1", "addr2", "dist1", "P_emaildomain", "R_emaildomain",
    "card4_raw", "card6_raw", "D1", "M1", "M2", "M3", "M4", "M6",
    "DeviceType", "id_12", "id_15", "id_23", "id_34",
)
TRANSACTION_TEMPORAL_FIELDS = ("TransactionID", "ts", "TransactionAmt", "ProductCD", "channel", "risk_score")
CARD_FIELDS = ("card_key", "customer_id", "card1", "card4", "card4_conflict", "card6", "card6_conflict", "transaction_count", "external_ids")
CUSTOMER_FIELDS = ("customer_id", "n_cards", "total_transaction_count", "first_seen_ts", "last_seen_ts")
DEVICE_FIELDS = ("device_profile", "os", "browser", "screen", "n_distinct_customers", "n_distinct_cards", "n_transactions", "hub_flag")
REGION_FIELDS = ("addr1", "n_distinct_customers", "n_distinct_cards", "n_transactions", "hub_flag")
CASE_FIELDS = ("case_id", "outcome", "pattern", "opened_at", "closed_at", "exposure_usd", "n_txns", "report_filed", "actions_taken")


def _pick(attrs: dict, fields: tuple) -> dict:
    return {f: attrs.get(f) for f in fields}


def _vset(result: list, key: str) -> list:
    """Extract a named vertex-set from an INTERPRET QUERY PRINT result."""
    for row in result:
        if key in row:
            return row[key]
    return []


_INVALID_VERTEX_ID_MARKERS = (
    "GSQL-2500",  # "Failed to convert ... vertex id" -- a well-formed but nonexistent id
    "SYS-0005",  # "engine encountered invalid vertex id" -- a malformed/empty id (found via test_q7_malformed_case_id)
    "Failed to convert",
    "invalid vertex id",
)


def _run(conn, gsql: str, params: dict) -> list:
    """Runs an INTERPRET QUERY and returns [] for a bad input vertex id
    instead of raising -- covers two distinct GSQL failure shapes, both
    confirmed by testing, not assumed:
      - GSQL-2500 ("Failed to convert ... vertex id"): a well-typed but
        nonexistent id (e.g. a real-looking TransactionID that isn't in
        the graph).
      - SYS-0005 ("engine encountered invalid vertex id"): a malformed
        id, e.g. an empty string -- this one was missed by the original
        GSQL-2500-only check and only caught because
        test_q7_malformed_case_id_returns_none_not_error exists.
    Every query function here takes a VERTEX<type> parameter or the
    equivalent risk, so this is the one place both failure modes are
    handled, rather than duplicating a try/except in every function. Any
    other exception is re-raised.
    """
    try:
        return conn.runInterpretedQuery(gsql, params=params)
    except Exception as e:
        if any(marker in str(e) for marker in _INVALID_VERTEX_ID_MARKERS):
            return []
        raise


# ---------------------------------------------------------------------------
# Q1 -- Transaction lookup
# ---------------------------------------------------------------------------

_Q1_GSQL = """
INTERPRET QUERY (VERTEX<HHG_Transaction> input_txn) FOR GRAPH HHGOA_FraudInvestigation {
  Start = {input_txn};
  Cards = SELECT c FROM Start:s -(HHG_MADE_reverse:e)-> HHG_Card:c;
  Custs = SELECT cu FROM Cards:c -(HHG_OWNS_reverse:e)-> HHG_Customer:cu;
  Devices = SELECT d FROM Start:s -(HHG_FROM_DEVICE:e)-> HHG_DeviceProfile:d;
  Regions = SELECT r FROM Start:s -(HHG_BILLED_IN:e)-> HHG_BillingRegion:r;
  PRINT Start, Cards, Custs, Devices, Regions;
}
"""


def q1_transaction_lookup(conn, flagged_txn_id: int) -> dict | None:
    result = _run(conn, _Q1_GSQL, {"input_txn": flagged_txn_id})
    start = _vset(result, "Start")
    if not start:
        return None
    txn = _pick(start[0]["attributes"], TRANSACTION_FIELDS)
    cards = _vset(result, "Cards")
    custs = _vset(result, "Custs")
    devices = _vset(result, "Devices")
    regions = _vset(result, "Regions")
    return {
        "transaction": txn,
        "card": _pick(cards[0]["attributes"], CARD_FIELDS) if cards else None,
        "customer": _pick(custs[0]["attributes"], CUSTOMER_FIELDS) if custs else None,
        "device": _pick(devices[0]["attributes"], DEVICE_FIELDS) if devices else None,
        "billing_region": _pick(regions[0]["attributes"], REGION_FIELDS) if regions else None,
    }


# ---------------------------------------------------------------------------
# Q2 -- Transaction temporal neighborhood
# ---------------------------------------------------------------------------

_Q2_CARD_TXNS_GSQL = """
INTERPRET QUERY (VERTEX<HHG_Card> input_card) FOR GRAPH HHGOA_FraudInvestigation {
  Start = {input_card};
  Txns = SELECT t FROM Start:s -(HHG_MADE:e)-> HHG_Transaction:t;
  PRINT Txns;
}
"""

_TS_FMT = "%Y-%m-%d %H:%M:%S"


def q2_temporal_neighborhood(conn, flagged_txn_id: int, window_hours: float) -> dict | None:
    q1 = q1_transaction_lookup(conn, flagged_txn_id)
    if q1 is None or q1["card"] is None:
        return None
    card_key = q1["card"]["card_key"]
    center_ts = q1["transaction"]["ts"]
    center_dt = datetime.strptime(center_ts, _TS_FMT)
    lo = center_dt - timedelta(hours=window_hours)
    hi = center_dt + timedelta(hours=window_hours)

    result = _run(conn, _Q2_CARD_TXNS_GSQL, {"input_card": card_key})
    all_txns = _vset(result, "Txns")

    windowed = []
    for v in all_txns:
        attrs = v["attributes"]
        ts = attrs.get("ts")
        if ts is None:
            continue
        dt = datetime.strptime(ts, _TS_FMT)
        if lo <= dt <= hi:
            windowed.append(_pick(attrs, TRANSACTION_TEMPORAL_FIELDS))
    windowed.sort(key=lambda r: r["ts"])

    amounts = [w["TransactionAmt"] for w in windowed if w["TransactionAmt"] is not None]
    amount_stats = (
        {"min": min(amounts), "max": max(amounts), "mean": sum(amounts) / len(amounts), "sum": sum(amounts)}
        if amounts
        else {"min": None, "max": None, "mean": None, "sum": None}
    )

    return {
        "window_hours": window_hours,
        "center_ts": center_ts,
        "transactions": windowed,
        "transaction_count": len(windowed),
        "amount_stats": amount_stats,
    }


# ---------------------------------------------------------------------------
# Q3 -- Device sharing
# ---------------------------------------------------------------------------

_Q3_GSQL = """
INTERPRET QUERY (STRING input_device) FOR GRAPH HHGOA_FraudInvestigation {
  SetAccum<VERTEX> @@customers;
  Start = {HHG_DeviceProfile.*};
  Filtered = SELECT s FROM Start:s WHERE s.device_profile == input_device;
  Txns = SELECT t FROM Filtered:s -(HHG_FROM_DEVICE_reverse:e)-> HHG_Transaction:t;
  Cards = SELECT c FROM Txns:t -(HHG_MADE_reverse:e)-> HHG_Card:c;
  Custs = SELECT cu FROM Cards:c -(HHG_OWNS_reverse:e)-> HHG_Customer:cu
          ACCUM @@customers += cu;
  PRINT Filtered, Cards, Txns;
}
"""


def q3_device_sharing(conn, device_profile: str) -> dict | None:
    result = _run(conn, _Q3_GSQL, {"input_device": device_profile})
    filtered = _vset(result, "Filtered")
    if not filtered:
        return None
    dev_attrs = filtered[0]["attributes"]
    cards = _vset(result, "Cards")
    txns = _vset(result, "Txns")

    ts_values = sorted(t["attributes"]["ts"] for t in txns if t["attributes"].get("ts"))
    out = _pick(dev_attrs, DEVICE_FIELDS)
    out["cards"] = [{"card_key": c["attributes"]["card_key"], "customer_id": c["attributes"]["customer_id"]} for c in cards]
    out["first_seen_ts"] = ts_values[0] if ts_values else None
    out["last_seen_ts"] = ts_values[-1] if ts_values else None
    return out


# ---------------------------------------------------------------------------
# Q4 -- Historical case evidence
# ---------------------------------------------------------------------------

_Q4_GSQL = """
INTERPRET QUERY (VERTEX<HHG_Card> input_card) FOR GRAPH HHGOA_FraudInvestigation {
  Start = {input_card};
  Direct = SELECT cc FROM Start:s -(HHG_ON_CARD_reverse:e)-> HHG_ClosedCase:cc;
  Connected = SELECT cc FROM Start:s -(HHG_CONNECTED_TO_reverse:e)-> HHG_ClosedCase:cc;
  PRINT Direct, Connected;
}
"""


def q4_historical_case_evidence(conn, card_key: str) -> dict:
    result = _run(conn, _Q4_GSQL, {"input_card": card_key})
    direct = _vset(result, "Direct")
    connected = _vset(result, "Connected")
    return {
        "direct_cases": [_pick(c["attributes"], CASE_FIELDS) for c in direct],
        "connected_cases": [dict(_pick(c["attributes"], CASE_FIELDS), relationship="connected_to") for c in connected],
    }


# ---------------------------------------------------------------------------
# Q5 -- Connected-card investigation
# ---------------------------------------------------------------------------

_Q5_GSQL = """
INTERPRET QUERY (VERTEX<HHG_Card> input_card) FOR GRAPH HHGOA_FraudInvestigation {
  Start = {input_card};
  Cases = SELECT cc FROM Start:s -(HHG_CONNECTED_TO_reverse:e)-> HHG_ClosedCase:cc;
  Siblings = SELECT c FROM Cases:cc -(HHG_CONNECTED_TO:e)-> HHG_Card:c
             WHERE c != input_card;
  PRINT Cases, Siblings;
}
"""

_Q5_VIA_DEVICE_GSQL = """
INTERPRET QUERY (VERTEX<HHG_Card> input_card) FOR GRAPH HHGOA_FraudInvestigation {
  Start = {input_card};
  Txns = SELECT t FROM Start:s -(HHG_MADE:e)-> HHG_Transaction:t;
  Devices = SELECT d FROM Txns:t -(HHG_FROM_DEVICE:e)-> HHG_DeviceProfile:d
            WHERE d.hub_flag == FALSE;
  OtherTxns = SELECT t2 FROM Devices:d -(HHG_FROM_DEVICE_reverse:e)-> HHG_Transaction:t2;
  OtherCards = SELECT c FROM OtherTxns:t2 -(HHG_MADE_reverse:e)-> HHG_Card:c
               WHERE c != input_card;
  PRINT OtherCards;
}
"""


def q5_connected_cards(conn, card_key: str) -> dict:
    result = _run(conn, _Q5_GSQL, {"input_card": card_key})
    cases = _vset(result, "Cases")
    siblings = _vset(result, "Siblings")

    case_ids = [c["attributes"]["case_id"] for c in cases]
    via_case = [
        {"case_ids": case_ids, "connected_card_key": s["attributes"]["card_key"], "connected_customer_id": s["attributes"]["customer_id"]}
        for s in siblings
    ]

    # Device-sharing evidence, in ONE batched server-side traversal, not one
    # round trip per distinct device profile. An earlier per-device-loop
    # implementation (calling q3_device_sharing once per distinct device)
    # was found, during Phase E's own benchmark run, to take 661s for a real
    # card (C11923's, case HHG-011) that has used 1,439 distinct device
    # profiles across its 10,361 transactions -- a genuine N+1 query
    # problem, not a hypothetical one. This single query does the same
    # traversal (card -> its non-hub devices -> their other transactions ->
    # their cards) entirely server-side in one call.
    #
    # Hub-flagged devices are excluded (WHERE d.hub_flag == FALSE) for the
    # same reason established earlier: a real card in this data (C03528,
    # part of the confirmed 22-card ring) used a hub-flagged desktop device
    # (208 distinct customers) alongside its real fraud device; including it
    # inflated connected_via_device from ~106 genuinely narrow matches to
    # 290, burying the real signal in noise. This directly implements the
    # hub-exclusion principle documented in
    # data/processed/graph_data_model.md section 6/13.
    device_result = _run(conn, _Q5_VIA_DEVICE_GSQL, {"input_card": card_key})
    other_cards = _vset(device_result, "OtherCards")
    seen = set()
    via_device = []
    for c in other_cards:
        ck = c["attributes"]["card_key"]
        if ck not in seen:
            seen.add(ck)
            via_device.append({"card_key": ck, "customer_id": c["attributes"]["customer_id"]})

    return {"connected_via_case": via_case, "connected_via_device": via_device}


# ---------------------------------------------------------------------------
# Q6 -- Billing-region evidence
# ---------------------------------------------------------------------------

_Q6_GSQL = """
INTERPRET QUERY (VERTEX<HHG_Transaction> input_txn) FOR GRAPH HHGOA_FraudInvestigation {
  Start = {input_txn};
  Regions = SELECT r FROM Start:s -(HHG_BILLED_IN:e)-> HHG_BillingRegion:r;
  PRINT Regions;
}
"""


def q6_billing_region_evidence(conn, flagged_txn_id: int) -> dict | None:
    result = _run(conn, _Q6_GSQL, {"input_txn": flagged_txn_id})
    regions = _vset(result, "Regions")
    if not regions:
        return None
    return _pick(regions[0]["attributes"], REGION_FIELDS)


# ---------------------------------------------------------------------------
# Q7 -- Benchmark case resolution
# ---------------------------------------------------------------------------

_Q7_GSQL = """
INTERPRET QUERY (VERTEX<HHG_InvestigationCase> input_case) FOR GRAPH HHGOA_FraudInvestigation {
  Start = {input_case};
  PRINT Start;
}
"""

INVESTIGATION_CASE_FIELDS = ("case_id", "customer_id", "card_key", "trigger_type", "trigger_text", "flagged_txn_id", "opened_at", "risk_score", "status")


def q7_benchmark_case_resolution(conn, case_id: str) -> dict | None:
    result = _run(conn, _Q7_GSQL, {"input_case": case_id})
    start = _vset(result, "Start")
    if not start:
        return None
    return _pick(start[0]["attributes"], INVESTIGATION_CASE_FIELDS)


# ---------------------------------------------------------------------------
# Q8 -- Combined investigation evidence
# ---------------------------------------------------------------------------

def q8_combined_evidence(conn, flagged_txn_id: int, window_hours: float = 24.0) -> dict | None:
    q1 = q1_transaction_lookup(conn, flagged_txn_id)
    if q1 is None:
        return None

    temporal = q2_temporal_neighborhood(conn, flagged_txn_id, window_hours) if q1["card"] else None
    device_evidence = q3_device_sharing(conn, q1["device"]["device_profile"]) if q1["device"] else None
    historical = q4_historical_case_evidence(conn, q1["card"]["card_key"]) if q1["card"] else {"direct_cases": [], "connected_cases": []}
    connected = q5_connected_cards(conn, q1["card"]["card_key"]) if q1["card"] else {"connected_via_case": [], "connected_via_device": []}
    billing = q6_billing_region_evidence(conn, flagged_txn_id)

    return {
        "transaction": q1["transaction"],
        "card": q1["card"],
        "customer": q1["customer"],
        "temporal_activity": temporal["transactions"] if temporal else [],
        "device_evidence": device_evidence,
        "historical_cases": historical,
        "connected_cards": connected,
        "billing_region": billing,
    }
