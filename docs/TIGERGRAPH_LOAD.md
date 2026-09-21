# TigerGraph Load Plan — Phase D (APPROVED, NOT YET EXECUTED)

GSQL source: `gsql/04_loading_jobs.gsql`. Pre-load validation: `src/graph/validate_against_schema.py`, results in `data/processed/tigergraph_validation_report.json` (`pre_load_validation` section). Nothing below has been run yet.

## Source files → targets

| Source file | Rows | Loads vertex | Loads edge(s) |
|---|---|---|---|
| `customers.csv` | 13,553 | `HHG_Customer` | — |
| `cards.csv` | 13,553 | `HHG_Card` | `HHG_OWNS` (Customer→Card) |
| `devices.csv` | 9,436 | `HHG_DeviceProfile` | — |
| `billing_regions.csv` | 332 | `HHG_BillingRegion` | — |
| `transactions_graph.csv` | 590,742 | `HHG_Transaction` | `HHG_MADE` (Card→Transaction), `HHG_FROM_DEVICE` (conditional), `HHG_BILLED_IN` (conditional) |
| `historical_cases.csv` | 5,565 | `HHG_ClosedCase` | `HHG_ON_CARD` (ClosedCase→Card) |
| `relationships/closed_case_involves_transaction.csv` | 14,955 | — | `HHG_INVOLVES` (ClosedCase→Transaction) |
| `relationships/closed_case_connected_to_card.csv` | 92 | — | `HHG_CONNECTED_TO` (ClosedCase→Card, with `resolution`) |
| `case_pack_resolved.csv` | 20 | `HHG_InvestigationCase` (seed, `READY` status) | `HHG_ON_CARD` (InvestigationCase→Card) |

`cards.csv` itself is Phase B's unchanged output — not regenerated for this phase, per instruction 4.

## Required run order (dependency: an edge's endpoints must exist before the edge loads)

```
1. load_customers
2. load_cards                (needs HHG_Customer for HHG_OWNS)
3. load_devices
4. load_billing_regions
5. load_transactions          (needs HHG_Card for HHG_MADE; HHG_DeviceProfile/HHG_BillingRegion for the conditional edges)
6. load_historical_cases      (needs HHG_Card for HHG_ON_CARD)
7. load_closed_case_involves_transaction   (needs HHG_ClosedCase + HHG_Transaction)
8. load_closed_case_connected_to_card      (needs HHG_ClosedCase + HHG_Card)
9. load_case_pack_seed        (needs HHG_Card for HHG_ON_CARD)
```

`HHG_NEXT` and `HHG_CITES` have no loading job — deliberately not populated this phase.

## Pre-load validation performed (read-only, local files only — see the JSON report for full detail)

Every column mapped to a non-STRING GSQL type was checked against every row of the actual processed file (chunked for `transactions_graph.csv`'s 590,742 rows):

- **Primary keys**: `card_key`, `customer_id`, `device_profile`, `addr1`, `case_id` (both case tables), `TransactionID` — all unique, zero nulls.
- **DATETIME columns**: `first_seen_ts`, `last_seen_ts`, `opened_at`, `closed_at`, `ts` — 100% parse in the exact format GSQL expects (`YYYY-MM-DD HH:MM:SS`), zero bad values across all files including the full 590,742-row transaction table.
- **UINT columns**: `card1`, `transaction_count`, `n_cards`, `n_transactions`, `n_txns`, `flagged_txn_id`, `txn_id`, `TransactionID` — all non-negative integers, zero exceptions.
- **DOUBLE columns**: `TransactionAmt`, `risk_score`, `exposure_usd`, `C1`–`C14` — all parse as floats, zero exceptions.
- **BOOL-intended columns**: `card4_conflict`, `card6_conflict`, `hub_flag` (both tables) — confirmed to contain *only* the literal strings `"True"`/`"False"`, validating the loading jobs' CASE WHEN normalization.
- **Controlled-vocabulary columns**: `report_filed` → exactly `{"Yes", "No"}`; `trigger_type` → exactly `{"risk_score", "customer_report", "analyst_request"}`; `resolution` → exactly `{"primary_external_id", "fallback_single_card_customer"}`.
- **`case_pack_resolved.risk_score`**: 11 non-null values, all parse as floats; 9 nulls — exactly matching the known 8 `customer_report` + 1 `analyst_request` rows, confirming the `-1` sentinel transform in `load_case_pack_seed` is necessary and sufficient.

**Result: zero validation failures across every check, every file.** Full machine-readable detail in `data/processed/tigergraph_validation_report.json`.

## Execution plan (for the NEXT approval, not this one)

1. Run `gsql/01_vertices.gsql`, then `02_edges.gsql`, then `03_graph.gsql` against the live connection (`USE GLOBAL` scope, since these create global types + a new graph — never `Transaction_Fraud`).
2. Switch `.env`'s `TG_GRAPH_NAME` to `HHGOA_FraudInvestigation`.
3. Upload each CSV and run its loading job (`RUN LOADING JOB ... USING FILENAME=...`), in the order above.
4. Run the post-load validation (vertex/edge counts *from TigerGraph itself*, not the CSVs — per your explicit instruction not to just trust CSV row counts) and append a `post_load_validation` section to `tigergraph_validation_report.json`.
5. Only then move to the investigation queries (Phase D items 11–13) and WCC (item 14).

None of step 1–4 has been executed. This document and the GSQL files are the plan for your review.
