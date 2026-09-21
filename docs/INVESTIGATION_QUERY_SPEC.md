# Investigation Query Specification — Phase E

Deterministic, read-only graph queries. **No LLM anywhere in this layer** — every query below returns structured facts the graph can prove; interpreting or weighing that evidence is explicitly out of scope here (that's the future agent's job, reasoning over what this layer returns).

Source of truth for schema: `docs/TIGERGRAPH_SCHEMA.md`, `docs/TIGERGRAPH_LOAD.md`, `data/processed/graph_data_model.md` (path correction from the Phase E brief, which said `docs/graph_data_model.md` — the actual file is under `data/processed/`), cross-checked against the live server throughout Phase D.

All queries implemented as `INTERPRET QUERY` (ephemeral, never installed/persisted) in `src/investigation/queries.py`, called through the existing `src.graph.tigergraph_connection.get_connection()` — no new connection logic, no schema/job changes.

## Chronology rule

Every query that orders or windows transactions uses **`ts`** (the `YYYY-MM-DD HH:MM:SS` string, lexicographically sortable — already established and validated in earlier phases). `TransactionID`/`TransactionDT` are never used for ordering (confirmed in Phase C: correlation with `ts` is 0.998, not 1.0).

## Q1 — Transaction lookup

**Input:** `flagged_txn_id` (int, `HHG_Transaction` primary id)

**Graph path:** `Transaction --MADE_reverse--> Card --OWNS_reverse--> Customer`; `Transaction --FROM_DEVICE--> DeviceProfile` (optional, ~20.2% coverage); `Transaction --BILLED_IN--> BillingRegion` (optional, ~88.9% coverage)

**Returns** (curated fields, not all 48 `HHG_Transaction` attributes):
```
transaction: TransactionID, ts, TransactionAmt, ProductCD, channel, risk_score,
             addr1, addr2, dist1, P_emaildomain, R_emaildomain,
             card4_raw, card6_raw, D1, M1, M2, M3, M4, M6,
             DeviceType, id_12, id_15, id_23, id_34
card: card_key, customer_id, card1, card4, card4_conflict, card6, card6_conflict,
      transaction_count, external_ids
customer: customer_id, n_cards, total_transaction_count, first_seen_ts, last_seen_ts
device: device_profile, os, browser, screen, n_distinct_customers, hub_flag (null if no device edge)
billing_region: addr1, n_distinct_customers, hub_flag (null if no billing edge)
```
**Empty-result behavior:** if `flagged_txn_id` doesn't exist, returns `None` for `transaction` (and all dependent fields) rather than raising — callers check for this explicitly.

## Q2 — Transaction temporal neighborhood

**Input:** `flagged_txn_id`, `window_hours` (float, **caller-supplied, no default fraud meaning** — see Step 4 rule: this is a query parameter, not a threshold)

**Approach:** resolve the flagged transaction's `card_key` and `ts` via Q1's graph path; fetch *all* transactions for that `card_key` via `Card --MADE--> Transaction` (curated fields: `TransactionID, ts, TransactionAmt, ProductCD, channel, risk_score`); filter to `[ts - window_hours, ts + window_hours]` and sort **in Python** using `ts` string comparison (already validated as chronologically correct for this format) rather than uncertain GSQL datetime-function syntax under time pressure — a deliberate, documented implementation choice.

**Returns:**
```
window_hours, center_ts, transactions: [ {TransactionID, ts, TransactionAmt, ProductCD, channel, risk_score}, ... ] (ts-ordered),
transaction_count, amount_stats: {min, max, mean, sum}
```

## Q3 — Device sharing

**Input:** `device_profile` (string) — or resolved from a `txn_id`/`card_key` via Q1 first

**Graph path:** `DeviceProfile --FROM_DEVICE_reverse--> Transaction --MADE_reverse--> Card --OWNS_reverse--> Customer`

**Returns:**
```
device_profile, os, browser, screen, hub_flag,
n_distinct_customers, n_distinct_cards, n_transactions,
cards: [ {card_key, customer_id} ... ], first_seen_ts, last_seen_ts
```
**Explicitly not a verdict:** `hub_flag` is always surfaced prominently; callers must not treat shared-device membership as fraud by itself (per the architecture's established evidence-vs-verdict principle, re-confirmed by Phase D's own traversal of the 22-card ring's device profile touching 52 customers, not just the ~22 confirmed-fraud ones).

## Q4 — Historical case evidence

**Input:** `card_key`

**Graph path:** `Card --ON_CARD_reverse--> ClosedCase` (direct precedent on this card); `Card --CONNECTED_TO_reverse--> ClosedCase` (this card named as connected in another case)

**Returns:**
```
direct_cases: [ {case_id, outcome, pattern, opened_at, closed_at, exposure_usd, n_txns, report_filed, actions_taken} ... ]
connected_cases: [ same shape, plus relationship: "connected_to" ] ... ]
```
`analyst_notes` intentionally excluded from the default return (free text, can be long) — available via a separate lookup if a caller needs the narrative.

## Q5 — Connected-card investigation

**Input:** `card_key`

**Graph path:** `Card --CONNECTED_TO_reverse--> ClosedCase --CONNECTED_TO--> Card` (siblings sharing a ring membership) union with Q3's device-sharing result for the same card's device(s)

**Returns:**
```
connected_via_case: [ {case_id, connected_card_key, connected_customer_id} ... ]
connected_via_device: [ same shape as Q3's cards list, excluding the input card itself ]
```

## Q6 — Billing-region evidence

**Input:** `flagged_txn_id`

**Graph path:** `Transaction --BILLED_IN--> BillingRegion`

**Returns:**
```
addr1, n_distinct_customers, n_distinct_cards, n_transactions, hub_flag
```
Per Phase C's documented finding: `BillingRegion` sharing is only meaningful time-windowed (unlike the flat aggregate returned here); this query returns the region's static facts only — any ring-style "who else was billed here recently" reasoning belongs to a future, explicitly time-windowed query, not this one.

## Q7 — Benchmark case resolution

**Input:** `case_id` (e.g. `"HHG-001"`, an `HHG_InvestigationCase` primary id)

**Graph path:** direct vertex lookup, no traversal (all fields were populated at Phase D seed time)

**Returns:**
```
case_id, customer_id, card_key, trigger_type, trigger_text, flagged_txn_id,
opened_at, risk_score, status
```
**Empty-result behavior:** `None` if the case doesn't exist.

## Q8 — Combined investigation evidence

**Input:** `flagged_txn_id`

Calls Q1 → (if resolved) Q2, Q3 (using Q1's `device_profile` if present), Q4 and Q5 (using Q1's `card_key`), Q6 (using `flagged_txn_id` directly), and assembles:

```json
{
  "transaction": {...},
  "card": {...},
  "customer": {...},
  "temporal_activity": [...],
  "device_evidence": {...} | null,
  "historical_cases": {"direct": [...], "connected": [...]},
  "connected_cards": {"via_case": [...], "via_device": [...]},
  "billing_region": {...} | null
}
```

## What's deliberately out of scope for Phase E

No verdict, no pattern classification, no probability, no policy/action recommendation — those require the agent (a later phase) reasoning over this evidence. No permanent `INSTALL QUERY` — every query here is `INTERPRET QUERY`, matching the instruction to validate correctness before committing to production endpoints (Step 10 names which ones are recommended for that conversion).
