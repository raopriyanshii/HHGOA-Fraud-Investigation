# Phase E Validation Report — Deterministic Investigation Query Layer

Scope: read-only graph queries only. No LLM, no MCP, no agent, no policy execution, no UI — none of that exists yet. Every number below is measured against the live `HHGOA_FraudInvestigation` graph, not estimated.

## 1. Query inventory

| Query | Purpose | File |
|---|---|---|
| Q1 `q1_transaction_lookup` | Transaction → card → customer → device → billing region | `src/investigation/queries.py` |
| Q2 `q2_temporal_neighborhood` | Same-card activity within a caller-supplied `ts` window | same |
| Q3 `q3_device_sharing` | Cards/customers sharing a `DeviceProfile` | same |
| Q4 `q4_historical_case_evidence` | `ClosedCase` precedent for a card (direct + connected) | same |
| Q5 `q5_connected_cards` | Ring-mates via shared `ClosedCase` and via shared non-hub device | same |
| Q6 `q6_billing_region_evidence` | Static `BillingRegion` facts for a transaction | same |
| Q7 `q7_benchmark_case_resolution` | Direct `InvestigationCase` lookup by id | same |
| Q8 `q8_combined_evidence` | Q1–Q6 composed into one evidence object | same |

All implemented as `INTERPRET QUERY` — ephemeral, never `INSTALL QUERY`'d, so nothing here is a persistent schema object.

## 2. Input/output contract

Full field-level contract for every query is in `docs/INVESTIGATION_QUERY_SPEC.md`, written before implementation and not altered to match results after the fact. Two implementation-time deviations from that spec are worth calling out explicitly, both are real fixes made during this phase, not planned in advance:

- **Q5's device-sharing evidence excludes hub-flagged devices** (`hub_flag == FALSE` filter, server-side). Not in the original one-line spec description — added after finding that a real ring card (`C03528`) also used a hub-flagged desktop device (208 distinct customers), which inflated its `connected_via_device` count from ~106 to 290 and would have buried the real signal in noise.
- **Q5's device-sharing evidence is now one batched query, not one query per distinct device.** The original per-device-loop design is what caused the performance bug in §3 below.

## 3. Benchmark results — all 20 cases

Script: `src/investigation/run_benchmark_validation.py`. Raw output: `data/processed/phase_e_benchmark_results.json`.

**20/20 cases passed all 8 required checks** (flagged transaction resolves; card resolves correctly; customer resolves correctly; transaction chronology works; historical cases retrievable; device relationships retrievable; connected-card relationships retrievable; no incorrect customer/card mapping) — zero exceptions, zero incorrect mappings.

| Case | Result | Elapsed | Direct historical cases found | Had device evidence |
|---|---|---|---|---|
| HHG-001 | PASS | 5.6s | 4 | No |
| HHG-002 | PASS | 2.5s | 1 | No |
| HHG-003 | PASS | 3.8s | 6 | No |
| HHG-004 | PASS | 2.5s | 4 | No |
| HHG-005 | PASS | 3.1s | 3 | Yes |
| HHG-006 | PASS | 4.2s | 0 | Yes |
| HHG-007 | PASS | 4.7s | 19 | No |
| HHG-008 | PASS | 3.0s | 19 | No |
| HHG-009 | PASS | 2.5s | 0 | No |
| HHG-010 | PASS | 3.6s | 1 | Yes |
| HHG-011 | PASS | 7.4s | 11 | Yes |
| HHG-012 | PASS | 3.1s | 2 | No |
| HHG-013 | PASS | 4.0s | 4 | Yes |
| HHG-014 | PASS | 3.0s | 0 | Yes |
| HHG-015 | PASS | 3.2s | 3 | Yes |
| HHG-016 | PASS | 3.5s | 0 | Yes |
| HHG-017 | PASS | 3.1s | 1 | Yes |
| HHG-018 | PASS | 5.0s | 20 | No |
| HHG-019 | PASS | 3.2s | 4 | Yes |
| HHG-020 | PASS | 3.2s | 2 | Yes |

No verdict is drawn from this table (per the phase's own instruction) — "direct historical cases found" and "had device evidence" are reported purely as facts the layer retrieved, not as fraud indicators. Worth noting factually: several cases (e.g. HHG-007, HHG-008, HHG-018) already have a substantial number of prior confirmed-fraud closed cases on record for that exact card, which is exactly the kind of structured evidence this layer exists to surface for a future agent to weigh — not to weigh itself.

## 4. 22-card ring validation

Recovered **through actual graph traversal** (`q3_device_sharing`, called with the ring's real device profile string as a plain input — no hardcoded customer/card list, no shortcut):

- **52 distinct customers** (matches Phase D's independent graph-traversal validation and Phase C's original pandas-based measurement)
- **114 transactions** (same match)

Also re-validated via `q5_connected_cards(conn, "C03528:14407")` (one of the 4 formally-cased ring members): 23 `connected_via_case` results (the ring, self excluded) and 106 `connected_via_device` results (post hub-exclusion fix) — consistent with, and slightly broader than, the case-based ring membership, exactly as expected (device-sharing is a superset signal, not identical to the formally-opened cases).

## 5. Latency results

Full methodology and numbers in `docs/INVESTIGATION_QUERY_BENCHMARK.md`. Summary (P50 / P95 / P99, 20 reps each):

| Query | P50 | P95 | P99 |
|---|---|---|---|
| Q1 | 343.5ms | 2771.3ms | 2771.3ms |
| Q2 | 699.3ms | 1846.4ms | 1846.4ms |
| Q3 | 376.1ms | 399.4ms | 399.4ms |
| Q4 | 294.9ms | 401.5ms | 401.5ms |
| Q5 | 645.2ms | 944.4ms | 944.4ms |
| Q6 | 289.6ms | 404.6ms | 404.6ms |
| Q7 | 289.2ms | 516.8ms | 516.8ms |
| Q8 | 2267.0ms | 2652.5ms | 2652.5ms |

P95≈P99 throughout is a 20-sample-size artifact (both land on the same sorted index), not a claim of unusually tight tail latency — stated explicitly rather than presented as more precise than it is.

## 6. Test results

**77/77 passing** — 55 from the pre-existing offline suite (Phases 1–D, unaffected by this phase) + **22/22** new in `src/tests/test_investigation_queries.py`, covering: valid/nonexistent transaction, valid/nonexistent benchmark case, customer/card resolution, temporal ordering, device-sharing retrieval (including the known ring), historical-case retrieval, connected-card retrieval (including the hub-exclusion regression test), empty-result handling, and malformed input handling.

One test failure occurred and was fixed during development, not hidden: `test_q7_malformed_case_id_returns_none_not_error` (passing `""` as a case id) initially raised `TigerGraphException: SYS-0005` instead of returning `None`, because the original error-handling in `_run()` only caught the `GSQL-2500` failure shape (a well-formed-but-absent id), not `SYS-0005` (a malformed/empty id) — a genuinely different GSQL error path found by the test itself. Fixed by widening `_run()`'s exception matching to cover both.

## 7. Known limitations

- **`Q2`'s temporal window is computed by fetching the full transaction list for a card, then filtering in Python** by `ts`, rather than filtering server-side. This was a deliberate simplicity choice (avoids uncertain GSQL datetime-function syntax) and works correctly, but costs more than necessary for extreme-cardinality cards (the same `C11923` card with 10,361 transactions took 9.3s for `Q2` alone, vs. sub-second for a typical card). Not fixed in this phase because it didn't fail any check — flagged for the same batched-query treatment `Q5` got, if `Q2` latency matters later.
- **`Q1`'s occasional multi-second outlier** (2771ms once out of 20 reps) wasn't root-caused — plausibly a single slow network round-trip to the remote Savanna endpoint, but not confirmed with certainty.
- **`Q8` is strictly sequential** (6 queries composed one after another) — no batching/parallelism attempted, consistent with "don't optimize prematurely," but a real cost if `Q8` becomes the primary agent-facing call.
- **No caching anywhere** — every call re-queries the graph. Fine for a 20-case benchmark; would need reconsidering for higher call volume.
- **`Q5`'s `connected_via_device` can still be large for cards with genuinely unusual device-churn behavior** (`C11923`: 795 results, after hub-exclusion) — this isn't a bug, it's a real, correctly-computed fact about that card's history, but a future agent consuming this evidence will need its own judgment about when a large `connected_via_device` list is itself the signal (extreme device churn) versus noise to filter further.

## 8. Queries recommended for conversion to permanent TigerGraph endpoints

All 8 behaved deterministically and correctly across the full benchmark; recommending conversion from `INTERPRET QUERY` to `INSTALL QUERY` (compiled, faster, permanent) for the ones a future agent will call most frequently and repeatedly:

- **Q1, Q7** — the two simplest, most frequently-needed lookups (every investigation starts with one of these); installing removes interpretation overhead for the highest call volume.
- **Q2, Q5** — the two queries whose current implementation has known scaling costs (§7); installing alone won't fix the underlying batching issue for `Q2`, but is worth doing together with that fix.
- **Q3, Q4, Q6** — straightforward, already fast (P50 under 400ms); good candidates but lower priority than Q1/Q7.
- **Q8** — recommend leaving as an *orchestration* function (calling the installed Q1–Q6) rather than installing as its own compiled query, since it's simple composition, not a distinct traversal.

This conversion is explicitly **not done in this phase** (Step 3's own instruction: validate correctness first) — noted here only as the phase's own recommendation for what comes next, per the Step 10 deliverable.

---

## Deliverable summary

- **Files created:** `docs/INVESTIGATION_QUERY_SPEC.md`, `docs/INVESTIGATION_QUERY_BENCHMARK.md`, `docs/PHASE_E_VALIDATION.md`, `src/investigation/__init__.py`, `src/investigation/queries.py`, `src/investigation/run_benchmark_validation.py`, `src/investigation/run_latency_benchmark.py`, `src/tests/test_investigation_queries.py`, `data/processed/phase_e_benchmark_results.json`, `data/processed/phase_e_latency_benchmark.json`.
- **Queries implemented:** 8 (Q1–Q8), all read-only `INTERPRET QUERY`.
- **20/20 benchmark cases:** PASS, zero exceptions, zero incorrect mappings.
- **22-card ring:** recovered exactly (52 customers, 114 transactions) via real traversal, no hardcoding.
- **Latency:** see §5; full data in `docs/INVESTIGATION_QUERY_BENCHMARK.md`.
- **Tests:** 77/77 passing (55 pre-existing + 22 new).
- **Known limitations:** §7.
- **Recommended for permanent endpoints:** §8.

Nothing in Phase D (schema, loading jobs, processed datasets) was modified. No MCP, LLM, agent, or UI work was done in this phase.
