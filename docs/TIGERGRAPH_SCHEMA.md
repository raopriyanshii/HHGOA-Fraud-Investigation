# TigerGraph Schema — Phase D (APPROVED, NOT YET EXECUTED)

Environment: TigerGraph Savanna, database `MyDatabase`, engine version **4.2.5** (confirmed via `conn.getVer()`, not assumed). pyTigerGraph client 2.0.4 (confirmed current on PyPI/GitHub). Nothing in this document has been run against the server yet — see `docs/TIGERGRAPH_LOAD.md` for the execution plan and STOP condition.

GSQL source files: `gsql/01_vertices.gsql`, `gsql/02_edges.gsql`, `gsql/03_graph.gsql`.

## Naming: why every type is prefixed `HHG_`

Before writing any schema, I ran a read-only `gsql("ls")` to check for collisions (GSQL vertex/edge type names are **global across the whole server**, not scoped to one graph). The existing `Transaction_Fraud` graph already defines a vertex type literally named **`Card`** (plus `Party`, `Account`, `Device`, `Merchant`, `Cases`, and more) — a bare `CREATE VERTEX Card` would collide directly with it. Per your instruction to stop and report before modifying the approved model, here is that report: every one of our 7 vertex types and 9 edge types is prefixed `HHG_` — not just the one that collided, applied consistently so there's never ambiguity in a future `ls` listing and zero chance of ever touching `Transaction_Fraud`. The properties, relationships, and semantics are otherwise unchanged from `data/processed/graph_data_model.md`.

## Vertices (7, as approved)

| Vertex | Primary ID | Notable properties | Source |
|---|---|---|---|
| `HHG_Customer` | `customer_id` STRING | `n_cards`, `first_seen_ts`/`last_seen_ts` DATETIME, `total_transaction_count` UINT | `customers.csv` |
| `HHG_Card` | `card_key` STRING (`customer_id:card1`) | `card4`/`card6` (resolved, majority-vote) + `card4_conflict`/`card6_conflict` BOOL, `external_ids` | `cards.csv` |
| `HHG_Transaction` | `TransactionID` UINT | See "Field types" below for the full 41-property breakdown | `transactions_graph.csv` |
| `HHG_DeviceProfile` | `device_profile` STRING (composite key) | `n_distinct_customers`, `hub_flag` BOOL | `devices.csv` |
| `HHG_BillingRegion` | `addr1` STRING | `n_distinct_customers`, `hub_flag` BOOL | `billing_regions.csv` |
| `HHG_ClosedCase` | `case_id` STRING | `outcome`, `pattern`, `exposure_usd`, `analyst_notes` | `historical_cases.csv` |
| `HHG_InvestigationCase` | `case_id` STRING | Trigger fields + a large set of DEFAULT-valued live-investigation fields (see below); `status` defaults to `READY` | `case_pack_resolved.csv` (seed only) |

## Field type decisions (concretizing what Phase C left abstract)

Phase C's data model specified *which* fields matter; Phase D has to pick an exact GSQL type for each. Every non-obvious choice below is a real decision, not an assumption — flagging for your review:

1. **card4/card6/hub-flag booleans stored as BOOL, not STRING** — straightforward; `src/graph/validate_against_schema.py` confirmed all three columns contain only the exact strings `"True"`/`"False"` across every row of every file (zero exceptions), so the CASE WHEN normalization in the loading jobs is safe.
2. **Sparse/engineered numeric fields (`D2`, `D3`, `D4`, `D10`, `D11`, `D15`, `dist1`) are typed STRING, not DOUBLE.** These have real null rates (12.9%–59.7%). If loaded as DOUBLE, an empty CSV cell would silently become `0.0` — a real, different value (e.g. "zero days since previous transaction") from "unknown." Keeping them as STRING (empty string = genuinely missing) trades query-time convenience (a caller must cast) for the correctness the project has repeatedly prioritized. `D1` (0.2% null — the most complete field in the raw dataset) got the same treatment for consistency, even though its practical risk is much smaller.
3. **`C1`–`C14` typed DOUBLE.** These are 0%-null (confirmed again this phase) but stored as floats in the source (`"1.0"`, not `"1"`), so DOUBLE avoids any int-truncation risk.
4. **`HHG_BillingRegion`'s primary id (`addr1`) is STRING, holding the literal value as it appears in the CSV (e.g. `"100.0"`)**, not DOUBLE. `addr1` is a region *code*, not a quantity — nothing does arithmetic on it — and a STRING primary id sidesteps any float-equality risk when the same value is referenced from `HHG_Transaction.addr1` for the `BILLED_IN` edge. The visible `.0` suffix is a cosmetic artifact of how pandas wrote it, not a schema defect; happy to normalize it in the CSV instead if you'd prefer clean integers over avoiding a second edit to an already-approved processed file — your call.
5. **`report_filed` stays STRING (`"Yes"`/`"No"`), not BOOL.** Confirmed via validation that it's always exactly one of those two literal values; kept as-is (no transform) since a transform here wasn't necessary for correctness, only for a marginal query convenience.
6. **`HHG_InvestigationCase.risk_score` and `fraud_probability` default to `-1`, not `0`.** Both are legitimately absent for the 9 case_pack rows with no risk-score trigger (confirmed: exactly 9 nulls) and for every case before an agent runs. `0` is a real, meaningful value (near-certainly-legitimate / lowest risk) that must never be confused with "not yet computed."
7. **`HHG_InvestigationCase`'s complex nested fields (evidence list, `next_best_actions`, `sar`, `evidence_requests`) are stored as JSON-serialized STRING properties** (`evidence_json`, etc.), not decomposed into new vertex/edge types. Phase C approved 7 vertices and 9 edges — inventing an `Evidence` vertex now would silently expand that model without your sign-off. This is a pragmatic Phase D concretization of "same shape as ClosedCase's edges, full shape defined in the architecture phase's case model" — flagging it explicitly since it's the one place a genuinely complex nested structure got flattened to a string rather than fully modeled in the graph. If a later phase needs to query *inside* the evidence list from GSQL (not just read it back whole), this would need revisiting.
8. **`HHG_InvestigationCase.status` defaults to `READY`, not `TRIGGERED`.** The 20 seeded vertices exist so the case_pack data has a graph anchor, but nothing has "happened" to them yet — a future agent must explicitly transition status away from `READY` (e.g. to an `INVESTIGATING` state) as its own auditable action, rather than the load job itself implying an investigation already started.

## Edges (9, as approved, one concretization)

| Edge | From → To | Loaded in Phase D? |
|---|---|---|
| `HHG_OWNS` | Customer → Card | Yes |
| `HHG_MADE` | Card → Transaction | Yes |
| `HHG_FROM_DEVICE` | Transaction → DeviceProfile | Yes (only where `device_profile` is non-empty, ~20.2% of transactions) |
| `HHG_BILLED_IN` | Transaction → BillingRegion | Yes (only where `addr1` is non-empty, ~88.9%) |
| `HHG_NEXT` | Transaction → Transaction | **No — declared only.** Fixed Phase D decision: no universal NEXT edge yet; temporal queries use `ts`-based windows. |
| `HHG_INVOLVES` | (ClosedCase \| InvestigationCase) → Transaction | Yes, for ClosedCase now; InvestigationCase side populates once an agent runs (future phase) |
| `HHG_ON_CARD` | (ClosedCase \| InvestigationCase) → Card | Yes, both sides (ClosedCase from `historical_cases.csv`, InvestigationCase from the case_pack seed) |
| `HHG_CONNECTED_TO` | (ClosedCase \| InvestigationCase) → Card | Yes, for ClosedCase now (the one real 92-row ring); carries the `resolution` provenance tag (`primary_external_id` \| `fallback_single_card_customer`) as an edge property so it's never lost |
| `HHG_CITES` | InvestigationCase → ClosedCase | **No — not loaded.** No investigation has cited any precedent yet; populated once an agent runs. |

**Concretization**: `INVOLVES`/`ON_CARD`/`CONNECTED_TO` are each declared as a single edge type with two FROM/TO pairs (GSQL supports this natively), shared between `HHG_ClosedCase` and `HHG_InvestigationCase`, rather than four near-duplicate edge types — directly matching Phase C's own description of `InvestigationCase` as "same shape as `ClosedCase`'s edges."

**Reverse edges** are declared for every loaded edge (`HHG_OWNS_reverse`, etc.) because the already-designed investigation queries need to traverse in both directions (e.g. "which cards used this device" is `DeviceProfile → Transaction → Card`, the reverse of how the edge was loaded).

## What's deliberately not in this graph

Per Phase C: no `EmailDomain` vertex (measured hub risk too high — see `graph_data_model.md` §7), no per-`id_*`-column vertices, no `Pattern` vertex (patterns are agent-identified findings, not a lookup table), no `Merchant` vertex (no such field exists in the data). `TransactionDT`, `V1`–`V339`, and the >50%-null `D`/`M` columns stay out of the graph entirely, retained only in the raw/processed CSVs — see `graph_data_model.md` §17 for the full excluded-field table, unchanged by this phase.

## Decisions confirmed

1. `HHG_` prefix on all 7 vertices and 9 edges — **confirmed**.
2. `addr1` as STRING with its existing `.0` formatting — **confirmed, no CSV changes**.
3. `HHG_InvestigationCase`'s evidence/actions/sar as JSON-string properties for Phase D — **confirmed**; revisit if a later phase needs to query inside them.
4. Case-pack seeding: 20 `HHG_InvestigationCase` vertices, **`status` DEFAULT changed to `READY`** (not `TRIGGERED`) so an agent must explicitly start the investigation as its own auditable step.
5. `HHG_NEXT` stays declared, not loaded.
6. `HHG_CITES` stays declared, not loaded.
7. `Transaction_Fraud` is never touched — confirmed by construction (every type/graph name here is `HHG_`/`HHGOA_`-prefixed and distinct).
8. Nothing in `gsql/` has been executed against the live connection.
