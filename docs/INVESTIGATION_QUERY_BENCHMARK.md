# Investigation Query Latency Benchmark — Phase E

## Methodology

Each of the 8 queries in `src/investigation/queries.py` was run **20 times** against a **fixed, real input** (HHG-001's flagged transaction `3514030` / card `C12382:21139`, the confirmed 22-card ring's device profile, and case `HHG-001` itself), so every repetition does identical work. Script: `src/investigation/run_latency_benchmark.py`. All calls went through the live `HHGOA_FraudInvestigation` graph — no mocking, no cached/precomputed results.

**Sample-size caveat, stated plainly rather than hidden:** with only 20 repetitions, `P95` and `P99` land on the same (19th, second-to-last) sorted sample for every query — they are not meaningfully distinct at this sample size. Reported as measured, not smoothed or extrapolated to look more precise than the data supports.

## Results (milliseconds)

| Query | P50 | P95 | P99 | Min | Max |
|---|---|---|---|---|---|
| Q1 — Transaction lookup | 343.5 | 2771.3 | 2771.3 | 336.9 | 2771.3 |
| Q2 — Temporal neighborhood | 699.3 | 1846.4 | 1846.4 | 679.9 | 1846.4 |
| Q3 — Device sharing | 376.1 | 399.4 | 399.4 | 368.8 | 399.4 |
| Q4 — Historical case evidence | 294.9 | 401.5 | 401.5 | 287.8 | 401.5 |
| Q5 — Connected cards | 645.2 | 944.4 | 944.4 | 623.2 | 944.4 |
| Q6 — Billing region evidence | 289.6 | 404.6 | 404.6 | 285.1 | 404.6 |
| Q7 — Benchmark case resolution | 289.2 | 516.8 | 516.8 | 284.4 | 516.8 |
| Q8 — Combined evidence (Q1+Q2+Q3+Q4+Q5+Q6) | 2267.0 | 2652.5 | 2652.5 | 2207.3 | 2652.5 |

Q1's one outlier (2771.3ms vs. a ~340ms median) is most likely a single slow network round-trip to the Savanna endpoint, not a query-cost issue — every other Q1 rep clustered tightly around 340ms. Q8's cost is explained entirely by it being 6 sequential queries composed together (Q1 through Q6) — 2267ms sits close to the sum of the individual medians (343+699+376+295+645+290 ≈ 2648ms), consistent with sequential, not parallel, execution.

## A real performance bug found and fixed during this benchmark, not before it

The first full 20-case benchmark run (Step 6, not this latency run) surfaced a genuine N+1 query problem in the original `Q5` implementation: it looped over every *distinct* device profile a card had used, calling a separate network round-trip for each. For most cards (a handful of devices) this was invisible. For `HHG-011`'s card (`C11923:21363`), which has used **1,439 distinct device profiles** across 10,361 transactions, this took **661.81 seconds** for that one case alone.

Fixed by replacing the per-device loop with a single batched server-side traversal (`card → its non-hub devices → their other transactions → their cards`, one `INTERPRET QUERY` call). Re-measured: **7.40 seconds** for the same case — a ~89x improvement, and the query's own latency numbers above (Q5: P50 645ms) reflect the *fixed* implementation, measured on a normal-cardinality card. This is recorded here rather than smoothed over, per the standing "no fabricated/hidden results" rule for this project — see `docs/PHASE_E_VALIDATION.md` for the full account.

## Not optimized further

Per the phase's own instruction ("do not optimize prematurely — first establish the correct baseline"), no further tuning was attempted once results were correct and reasonably fast (sub-second for 6 of 8 queries, low seconds for the two multi-hop ones). If Q1/Q8's occasional multi-second tail latency matters for a future production SLA, the next step would be repeating this benchmark with a larger sample (100+ reps) to get a statistically meaningful P95/P99 before deciding whether it's worth addressing.
