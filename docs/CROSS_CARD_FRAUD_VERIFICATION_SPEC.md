# Cross-Card Fraud Verification (Q9) — Implementation Note

Status: implemented, **not yet live-validated** (TigerGraph has been
unreachable throughout this implementation turn — see "Live validation
status" below). This is a pointer to the code and an honest record of
what was and wasn't verified, not a restatement of the approved G3
specification from the prior conversation turn.

## What this is

An additive, read-only extension to the Phase E query layer
(`src/investigation/queries.py`) and the Phase F MCP layer
(`src/mcp_server/server.py`): one new function, `q9_cross_card_fraud_verification(conn, card_key)`,
and one new MCP tool, `cross_card_fraud_verification`. Q1–Q8 and the
existing 8 MCP tools are byte-for-byte unchanged — confirmed via
`git diff`, which shows only additions (107 lines in `queries.py`, 6 in
`server.py`, 0 deletions in either).

It answers the question G1/G2 previously couldn't: for cards sharing a
non-hub device profile or billing region with a given card, does that
*other* card have its own independently confirmed-fraud history, and
does it belong to the same customer or a different one? This is the
evidence R6 needs. It deliberately never touches `HHG_CONNECTED_TO` —
case-based ring evidence (`q5_connected_cards`'s `connected_via_case`)
and shared-element evidence stay in separate tools.

## Output shape

```json
{
  "via_device": {"other_cards": [{"card_key": "...", "customer_id": "...", "same_customer": false, "has_confirmed_fraud": true, "confirmed_fraud_case_ids": ["CC-..."]}]} | null,
  "via_region": { ... same shape ... } | null
}
```
`null` covers both "no such evidence exists at all" and "evidence exists
but nobody else shares it" — both collapse to the same caller-facing
null, matching Q1–Q8's existing convention rather than inventing a new
three-state signal.

## Design decisions worth recording explicitly

- **Two independent GSQL blocks** (`_Q9_VIA_DEVICE_GSQL`, `_Q9_VIA_REGION_GSQL`),
  each resolving the input card's own `customer_id` *and* enriching every
  other card with its `customer_id` and confirmed-fraud case ids in one
  server-side traversal — two total calls per invocation, never one per
  discovered card. This reuses the exact batched-traversal pattern Phase
  E's Q5 fix established (the same principle that turned a 661.81s N+1
  bug into 7.4s for `C11923`), not a new technique.
- **GSQL enrichment without narrowing the vertex set**: the per-card
  customer/case lookups are assigned to throwaway variable names
  (`OtherCardsWithCustomer`, `OtherCardsWithCases`) rather than
  reassigning `OtherCards` itself — reassigning would have silently
  dropped any card with zero closed cases from the result set entirely,
  which would be wrong (a card with no fraud history should still appear
  with `has_confirmed_fraud: false`, not vanish).
- **`same_customer` only covers cards reachable via a shared device/region.**
  It does **not**, and structurally cannot, verify R10's "two of the
  customer's own cards" condition — that needs a different traversal
  starting from the *customer* (enumerating all their cards, regardless
  of any shared element), which this capability doesn't provide. G2's
  `BLOCK_ALL_CARDS` determination is unaffected by this addition and
  remains `PROHIBITED`, unchanged, per the approved spec's explicit
  scope boundary.
- **Output fields kept to exactly what was specified**: `via_device`/`via_region`
  each contain only `other_cards`; no `device_profile`/`hub_flag`
  descriptive fields were added, since the exact restated output
  requirements for this implementation turn only mandated the outer
  null-ability and the five per-card fields. Not adding extra fields
  beyond what was asked, even though a device-identifying field seems
  plausibly useful, is a deliberate choice to stay inside scope rather
  than improvise.

## Live validation status

**TigerGraph remains unreachable** (same `500` on `/restpp/version` as
every prior turn this session). Consequences, stated plainly:

- All 9 live-connection tests in `src/tests/test_cross_card_fraud_verification.py`
  failed at the connection-establishment step, before any query executed —
  confirmed to be the same pre-existing outage, not a new regression, by
  running the already-frozen `test_investigation_queries.py` (Q1–Q8) in
  the same session and observing the identical 22/22 failure pattern.
- **This means Q9's GSQL has not been executed against the live engine at
  all.** Unlike Q1–Q8, which were iteratively validated during Phase E's
  actual development (including catching a real 0-indexed-column loading
  bug that only surfaced under execution), Q9's GSQL syntax is new and
  unexercised. It was written carefully against confirmed schema facts
  (e.g. `HHG_BILLED_IN_reverse`'s exact name was read directly from
  `gsql/02_edges.gsql` rather than assumed from the `_reverse` naming
  convention seen elsewhere), but "carefully written" is not the same
  claim as "verified correct." This should be the first thing re-run once
  connectivity is restored, before this capability is relied on for
  anything.
- **5 offline tests pass** — `test_extract_cross_card_results_*` — these
  verify the `same_customer`/dedup/case-id-sort logic directly and
  precisely, independent of the live outage, including the "same
  customer's own cards share a device" scenario (test matrix item 6),
  tested via a synthetic fixture rather than a real example, since no
  real qualifying card pair was searched for or confirmed this turn.
