# Phase G — Agent Investigation Workflow Spec

Status: **finalized specification, not yet implemented.** No LLM has been
called, no agent loop exists, no policy logic exists, no UI exists, no
Python code has been written for this phase. This document is the design
to be built against once implementation starts.

## 1. What this phase is, and isn't

**Is:** the design of an orchestration layer that calls the 8 existing MCP
tools (Phase F, wrapping the Phase E deterministic queries) in a bounded,
evidence-driven sequence for a given flagged transaction, and turns the
results into a structured investigation report with a next-step
recommendation.

**Isn't:**
- Not a new source of graph analysis. Every fact in the report must trace
  back to a Q1–Q8 MCP tool call. The agent orchestrates and narrates
  already-computed evidence; it never reasons about the graph independently
  of what a tool actually returned (§9).
- Not a decision-maker. The output is a recommendation for a human
  reviewer, never an automatic action.
- Not tied to a specific LLM/provider yet. This spec defines the
  tool-calling contract and the output contract; wiring an actual model in
  is a later, separate step.

## 2. Entrypoint

One canonical entrypoint: **a flagged transaction ID** (matches Q1/Q8's
`flagged_txn_id`). A benchmark case ID (Q7's `case_id`) is a secondary
entrypoint — it resolves to a `flagged_txn_id` via one `benchmark_case_resolution`
call and then follows the same path. No other trigger shape is in scope.

## 3. Workflow model — bounded adaptive investigation (finalized)

```
Trigger
  |
  v
Q8 combined_evidence
  |
  v
Assess evidence
  |
  v
connected_via_case present?
  |-- No  -----------------------------> recommendation
  |
  |-- Yes
       |
       v
  deterministic selection of <=3 cards (Section 6)
       |
       v
  Q4 historical_case_evidence (once per selected card)
       |
       v
  reassess
       |
       v
  recommendation
       |
       v
  evidence ledger (Section 8)
```

**Round 1 (mandatory):** resolve the trigger to a `flagged_txn_id` (one
`benchmark_case_resolution` call, only if the trigger was a case ID), then
call `combined_evidence` (Q8) exactly once. This is the entire evidence
base for every investigation — nothing bypasses it.

**Round 2 (conditional, at most once):** if and only if Q8's
`connected_cards.connected_via_case` is non-empty, select up to 3 of those
connected cards (Section 5's deterministic rule) and call
`historical_case_evidence` (Q4) once per selected card. If
`connected_via_case` is empty, Round 2 does not happen and the workflow
goes straight to the recommendation.

**No Round 3.** Findings from Round 2 (e.g. a connected card turning out to
have a `confirmed_fraud` outcome) are surfaced in the report; they never
trigger a further round of tool calls. This is what "no recursive or
unbounded tool calling" means concretely: the reassessment step reads
Round 2's results, it does not requery based on them.

## 4. Hard limits (finalized)

| Limit | Value |
|---|---|
| Maximum investigation rounds | **2** (trigger-resolution + Q8 = Round 1; capped Q4 follow-ups = Round 2) |
| Maximum follow-up cards (Round 2) | **3** |
| Recursive/chained tool calls | **Not permitted** — Round 2's output is never fed back into another tool call |
| Arbitrary/ad-hoc GSQL | **Not permitted** — only the 8 existing MCP tools may be called |
| Tool calls that write | **None exist in the agent's tool registry** — all 8 registered tools are read-only by construction (Phase E/F) |
| Autonomous external action | **Not permitted** — output is a recommendation string, no tool exists to act on it |

### Exact MCP call count, by actual tool sequence (not estimated)

The agent only ever sees calls at the MCP boundary — `combined_evidence`
internally composing Q1–Q6 in Python (Phase E's own implementation) is
opaque to the agent and does not count as a separate agent-issued call.
Counting only calls the agent itself dispatches via `mcp.call_tool`:

| Trigger type | `connected_via_case` empty? | Calls | Sequence |
|---|---|---|---|
| `flagged_txn_id` | Yes | **1** | `combined_evidence` |
| `flagged_txn_id` | No (K <= 3 selected) | **1 + K** | `combined_evidence`, then K x `historical_case_evidence` |
| `case_id` | Yes | **2** | `benchmark_case_resolution`, `combined_evidence` |
| `case_id` | No (K <= 3 selected) | **2 + K** | `benchmark_case_resolution`, `combined_evidence`, then K x `historical_case_evidence` |

Minimum possible: **1** call. Maximum possible: **5** calls (`case_id`
trigger, 3 connected cards selected). This range is derived directly from
the workflow in Section 3, not assumed.

### 4a. G4 addition — the Q9 gate, and its accepted blind spot (finalized)

G4 added one more possible call, `cross_card_fraud_verification` (Q9,
Phase G3), inside Round 1, gated on evidence `combined_evidence` already
returned — no new query is needed to evaluate the gate itself:

```
q9_gate = card resolved AND (
    (device_evidence exists AND device_evidence.hub_flag == False)
    OR (billing_region exists AND billing_region.hub_flag == False)
)
```

The `card resolved` prerequisite is structural, not part of the hub-flag
condition itself: `cross_card_fraud_verification` takes `card_key` as its
only input, so the gate cannot fire without a resolved card to call it
with, regardless of what `device_evidence`/`billing_region` show.

When `q9_gate` is true, Q9 is called exactly once, and Round 2's
effective follow-up cap drops from 3 to 2 **for that investigation only**
— `MAX_FOLLOW_UP_CARDS` itself stays 3. This keeps the maximum possible
call count at **5 in every trigger/gate/ring combination**, not 6:

| Trigger | Q9 gate | Calls | Sequence |
|---|---|---|---|
| `case_id` | true | **5** (at most) | resolution, `combined_evidence`, Q9, up to 2× `historical_case_evidence` |
| `case_id` | false | **5** (at most) | resolution, `combined_evidence`, up to 3× `historical_case_evidence` |
| `flagged_txn_id` | true | **4** (at most) | `combined_evidence`, Q9, up to 2× `historical_case_evidence` |
| `flagged_txn_id` | false | **4** (at most) | `combined_evidence`, up to 3× `historical_case_evidence` |

**Accepted limitation, not a defect:** the gate reads `device_evidence`/
`billing_region` as `combined_evidence` returns them — both scoped to the
*flagged transaction specifically* (Q3/Q6 called on that one transaction's
own device/region). Q9 itself, once invoked, checks the *card's full
transaction history* (all non-hub devices/regions the card has ever used,
via its own batched traversal). This creates a real, specific blind spot:
if the flagged transaction's own device/region happens to be hub-flagged
or absent, but a *different* transaction on the same card used a non-hub
device/region that Q9 would otherwise have found cross-card fraud
evidence through, the gate stays closed and Q9 is never called for that
investigation — even though Q9 itself would have found something. This
was identified and accepted explicitly when Option B was chosen over the
alternatives (Options A/C/D, see the G4 design review): given the frozen
5-call ceiling, some gate imprecision is unavoidable in any design that
doesn't touch the ceiling itself, and this specific trade-off was chosen
because it costs nothing extra to evaluate and preserves Round 2's full
strength whenever Q9 isn't relevant, rather than weakening Round 2
unconditionally (Option A) or introducing an unprincipled exclusion
(Option C).

### 4b. G5-B addition — R5 `pattern_evidence` (local, zero-cost) (finalized)

G5-A/G5-B added one more piece of evidence to the ledger,
`pattern_evidence`, produced by `src.investigation.patterns.classify_card_testing`
over `temporal_activity` — the same list `combined_evidence` already
returns. This is **not** an MCP/tool call: `classify_card_testing` makes
no network call of any kind, so it costs nothing against the 5-call
ceiling documented above, and the call-count table in this section is
completely unaffected — `_ALLOWED_TOOLS`, `MAX_FOLLOW_UP_CARDS`, and the
Q9-gate arithmetic in §4a are all untouched.

It is computed *after* `_assess()` (§6a) has already run and returned,
specifically so it structurally cannot influence G1's own
escalate/monitor/insufficient_evidence tier — it is separate evidence for
G2's policy layer to consume, not part of G1's own recommendation.
`classify_card_testing` reports every structurally-qualifying candidate
sequence it finds, unranked (no "best" or "strongest" selection), never
asserting a numeric dollar value for "small" — see
`src/investigation/patterns.py`'s own docstring for the full detection
rule.

G2 consumes `pattern_evidence["matched"]` to ground exactly the two R5
actions the organizer wording supports without any missing information:
`DECLINE_TRANSACTION` (route `L1`) and `STEP_UP_AUTH` (route `auto`).
`BLOCK_CARD` remains `DEFERRED` — R5's own ">$100, already cleared"
escalation clause is checkable from `pattern_evidence`, but `BLOCK_CARD`'s
required route (`L1` vs `L2`) depends on `exposure_usd`, which does not
exist anywhere in this system and is not invented to work around the gap.
`BLOCK_ALL_CARDS`/R10 is entirely unaffected by any of this.

## 5. Follow-up card selection — deterministic rule (finalized)

The agent must not choose which connected cards to look up based on any
free-text/prose judgment. The selection rule uses only fields that
`connected_cards.connected_via_case` (produced by Q5, embedded in Q8's
output) actually returns for each entry — confirmed from
`src/investigation/queries.py`'s `q5_connected_cards`:

```python
{"case_ids": [...], "connected_card_key": "...", "connected_customer_id": "..."}
```

**There is no severity, exposure, date, or outcome field at this level**
(those only exist inside Q4's own output, which is exactly why Round 2
calls Q4 in the first place — to obtain them). Since the evidence available
*before* Round 2 contains no ranking-relevant field, no ranking heuristic
is invented. The rule is:

> Sort `connected_via_case` entries by `connected_card_key` ascending
> (plain string sort, deterministic, ties impossible since card keys are
> unique), then take the first 3.

This is canonical ordering, not a fabricated risk score. If a future phase
adds a genuinely rankable field to Q5's output (e.g. a precomputed
exposure or confirmed-case count per connected card), this rule should be
revisited then — not anticipated now with a field that does not exist.

## 6. Recommendations (finalized)

Exactly three values, and nothing else:

- **`escalate`** — the evidence (direct historical cases, and/or Round 2
  findings on connected cards) contains enough signal that a human
  reviewer should prioritize this transaction for manual review.
- **`monitor`** — some evidence exists (e.g. a device or connected-card
  signal) but not enough to warrant escalation; a human reviewer may want
  to keep watching this card/customer.
- **`insufficient_evidence`** — the graph evidence gathered does not
  support a meaningful judgment either way.

**These are investigation/next-step recommendations, not fraud verdicts.**
None of them mean "this is fraud" or "this is not fraud" — they mean "here
is how much investigative attention this deserves next." The agent must
not invent its own fraud-detection policy, scoring thresholds, or dollar
cutoffs to decide between these three; the specific decision logic mapping
evidence to one of these three labels is implementation detail for the
build step, not this spec, but it must be a fixed, documented, auditable
rule — never a threshold the model chooses ad hoc per-investigation. That
rule is no longer open — see Section 6a.

## 6a. V1 recommendation rule — deterministic evidence-to-label mapping (finalized)

This resolves the one item Section 12 (prior draft) left open: the exact
fixed rule mapping gathered evidence to one of `escalate` / `monitor` /
`insufficient_evidence`. It uses only fields that Q1–Q8 actually return
today (verified against `src/investigation/queries.py` and the real
`outcome` value domain in `data/processed/historical_cases.csv`, which is
exactly `{"cleared", "confirmed_fraud"}` — no third value exists). No new
field, query, or threshold is invented.

### Evidence observed vs. recommendation vs. fraud determination

These are three distinct things, and the rule keeps them distinct:

- **Evidence observed** — the raw curated facts a tool call returned (e.g.
  "this card has 4 direct historical cases, all `outcome=confirmed_fraud`").
  This is what goes in the evidence ledger (Section 8), always, regardless
  of what it leads to.
- **Recommendation** — one of the 3 fixed labels below, derived from
  evidence observed via the fixed rule in this section. It says how much
  investigative attention this deserves next.
- **Fraud determination** — whether this transaction/card is actually
  fraudulent. **This system never produces one.** That determination is a
  human reviewer's job (or, for already-closed historical cases, it's the
  `outcome` field itself — a past human/process determination this system
  only reads, never makes).

### Evidence fields used (all confirmed present in Q1–Q8's current output)

| Signal | Source field(s) | From | Used as a trigger? |
|---|---|---|---|
| Direct confirmed-fraud history | `historical_cases.direct_cases[].outcome == "confirmed_fraud"` | Q8 (embeds Q4 on the flagged card) | Yes — branch 1 |
| Direct cleared history | `historical_cases.direct_cases[].outcome == "cleared"` | Q8 (embeds Q4 on the flagged card) | No — recorded only (§ below) |
| Connected confirmed-fraud history | `historical_cases.connected_cases[].outcome == "confirmed_fraud"` | Q8 (embeds Q4 on the flagged card) | Yes — branch 1 |
| Case-based ring existence | `connected_cards.connected_via_case` non-empty | Q8 (embeds Q5 on the flagged card) | No — see "Why case-ring alone is not a monitor trigger" below |
| Sibling confirmed-fraud history | any Round-2 `historical_case_evidence` call (on a selected sibling card) returning `outcome == "confirmed_fraud"` in its own `direct_cases` or `connected_cases` | Round 2 (separate Q4 calls, Section 5's selection) | Yes — branch 1 |
| Narrow (non-hub) device evidence | `device_evidence is not None and device_evidence.hub_flag == False` | Q8 (embeds Q3 on the flagged transaction's device, if any) | **Yes — branch 2 (the only branch-2 trigger)** |
| Notable (non-hub) billing region | `billing_region is not None and billing_region.hub_flag == False` | Q8 (embeds Q6) | No — see "Why billing-region hub_flag alone is not a monitor trigger" below |
| Temporal activity beyond the flagged transaction itself | `len(temporal_activity) > 1` | Q8 (embeds Q2) | No — see "Why bare temporal activity is not a monitor trigger" below |

**Deliberately excluded from V1, and why:** `risk_score` (already the
reason this transaction was flagged in the first place — per Q7's own
`trigger_type`/`trigger_text`, e.g. *"Real-time model scored transaction
... at 0.61"* — reusing it here would silently make risk_score the real
final answer under a different name, which is explicitly disallowed).
`exposure_usd`, `n_txns`, `report_filed`, `actions_taken`, `pattern` (all
real CASE_FIELDS, but using them would require picking a dollar or count
threshold with no principled place to draw it — exactly the "invented
weights to look sophisticated" this rule avoids). These fields still
belong in the evidence ledger for human review; they're just not inputs to
the V1 label decision.

### Distinguishing meaningful corroboration from merely observed activity

Every candidate branch-2 signal was re-examined against one question: does
this establish something distinctive about *this* card/transaction, or
does it just describe activity that would exist for a large share of
ordinary, non-fraudulent cards too? Three of the four original candidates
fail that test on inspection of what the field actually encodes; only one
survives.

**Why case-ring alone is not a monitor trigger.** Q4 and Q5 both traverse
the *same* `HHG_CONNECTED_TO_reverse` edge from the *same* input card — Q4
to read the connected cases' own records, Q5 to find the sibling cards on
the other side of those same cases. That means: if any case connecting
this card to a ring had `outcome == "confirmed_fraud"`, it would already
appear in `historical_cases.connected_cases` and branch 1 would already
have fired. So by the time evaluation reaches branch 2 with
`connected_via_case` non-empty, every connecting case for *this card* is
necessarily `outcome == "cleared"` — there is no other possibility left.
A cleared-case-only ring is, in substance, the same category of
information as direct cleared history, which principle 6 already says
must not count as corroborating evidence. Counting it here would smuggle
the same thing back in under a different field name. **Not a trigger.**

**Why billing-region hub_flag alone is not a monitor trigger.** A
`BillingRegion` (`addr1`) is a broad geographic/administrative area, not a
specific fingerprint. `hub_flag == False` only says fewer than 1% of
customers share that exact region — but with many distinct regions in the
dataset, being below that threshold could easily be the *ordinary* state
for most regions, not a distinguishing one, and nothing in Q6's fields
says how common "non-hub" actually is across regions overall. Deciding a
count/percentage/rank *within* non-hub regions that would count as
"notable" would require inventing a threshold this phase is explicitly
not allowed to invent, and TigerGraph is unavailable this turn to check
the real distribution empirically. Per the instruction to prefer stating
this limitation over manufacturing a cutoff: **not a trigger in V1.**

**Why bare temporal activity is not a monitor trigger.** `temporal_activity`
always includes the flagged transaction itself, so `len(...) > 1` only
means "at least one other transaction happened on this card within the
window" — true for a large share of ordinary active cards, fraudulent or
not. Nothing in `TRANSACTION_TEMPORAL_FIELDS` distinguishes "one more
routine transaction" from something materially unusual without picking an
arbitrary count, dollar-sum, or time-gap cutoff — exactly what this
tightening forbids inventing. **Not a trigger in V1**, per the same
principle: state the limitation rather than manufacture a threshold.

**Why narrow device evidence is different, and survives.** A `DeviceProfile`
is a specific fingerprint (device model + OS + browser + screen
resolution together), not a broad category. `hub_flag == False` there
means fewer than 1% of *all customers* were ever seen on that *exact*
fingerprint — and Q3's own output (`device_evidence.cards`) shows the
actual other cards tied to it. This is a concrete link to specific other
entities via a narrow, already-graph-computed signal, the same kind of
fact the project's own Phase C/D/E "hub-exclusion principle" already
treats as the genuine signal (as opposed to hub-flagged devices/regions,
which are treated as noise). It is qualitatively different from "another
transaction happened" or "connected only via a cleared case" — it is
evidence of an actual narrow connection to other cards. **Kept as the one
branch-2 trigger.**

### The deterministic mapping (tightened)

Evaluated in this fixed order; the first branch that matches decides the
label. Every condition is a plain presence/absence check on an
already-computed graph field (`outcome`, `hub_flag`) — no invented
magnitude threshold.

```
1. IF has_direct_confirmed_fraud
   OR has_connected_confirmed_fraud
   OR has_sibling_confirmed_fraud (from Round 2, if it ran)
   THEN escalate

2. ELSE IF has_narrow_device_evidence
   (device_evidence is not None AND device_evidence.hub_flag == False)
   THEN monitor

3. ELSE insufficient_evidence
```

This is deliberately narrower than the prior draft: case-ring existence,
billing-region hub status, and bare temporal-activity presence no longer
trigger `monitor` on their own (see the analysis above for each). All
three remain visible in the evidence ledger — they are still real,
recorded evidence — they simply no longer decide the label by themselves.
`insufficient_evidence` is now the default whenever the only observations
are ordinary activity, a cleared-case-only connection, or a common billing
region, which is the intended effect: `monitor` should mean something
specific was found, not that *anything at all* was found.

Direct **cleared** history (`outcome == "cleared"`) remains recorded in
the evidence ledger but is **not** a trigger condition in either branch
above — on its own it neither escalates nor downgrades, exactly as before.

### Known coverage limitation (not a missing-field problem — a stated consequence of an already-approved limit)

`has_sibling_confirmed_fraud` only reflects the ≤3 connected cards
Section 5's deterministic rule selects. If a ring has more than 3
case-connected members, the rest are not sampled by V1's Round 2, so a
confirmed-fraud record on an unsampled sibling won't surface as
`escalate` — the ring's mere existence would still correctly produce
`monitor` via branch 2, just not the stronger `escalate` label. This is a
direct consequence of the Section 4 hard limit (max 3 follow-up cards),
not evidence this system lacks — no new field or query is needed to fix
it; only raising the already-debated cap would change it, which is a
Section 4 decision, not this one.

### This is an investigation-attention policy, not a fraud classifier

Nothing in this mapping computes a probability of fraud, a risk score, or
a verdict. It computes how much a human reviewer's attention this
transaction warrants next, from evidence that already exists elsewhere in
the graph. Two transactions with identical `escalate` labels can have
completely different actual outcomes once a human reviews them — the
label is a triage signal, not a prediction.

### Three concrete examples

**1. `escalate` — real, live-verified data** (this exact evidence was
pulled from the live graph earlier in this working session, via Q8 on
`HHG-001`'s flagged transaction `3514030`, card `C12382:21139`):
- `historical_cases.direct_cases`: 4 entries (`CC-2964`, `CC-1066`,
  `CC-3587`, `CC-1673`), **all four** with `outcome: "confirmed_fraud"`.
- `historical_cases.connected_cases`: `[]`
- `connected_cards.connected_via_case`: `[]` (so Round 2 does not run)
- `device_evidence`: `null` (this transaction has no device fields populated)
- `billing_region`: `hub_flag: true`
- `temporal_activity`: 6 entries
- **Causing field:** `has_direct_confirmed_fraud = True` (branch 1) →
  **`escalate`**. Nothing else needs to be checked; branch 1 already matched.

**2. `monitor` — illustrative, not live-verified** (constructed from real
field names/domains to show the rule's logic; TigerGraph was unreachable
during this and the prior turn, so this is explicitly **not** claimed as a
live-pulled case):
- `historical_cases.direct_cases`: `[]`, `connected_cases`: `[]`
- `connected_cards.connected_via_case`: `[]`
- `device_evidence`: `{"hub_flag": false, "n_distinct_customers": 3, "cards": [...2 other cards...], ...}`
- `billing_region`: `{"hub_flag": true, ...}`
- `temporal_activity`: 1 entry (just the flagged transaction)
- **Causing field:** no branch-1 condition holds. `has_narrow_device_evidence
  = True` (`device_evidence` present, `hub_flag == False`, and it names 2
  other specific cards sharing that exact device fingerprint) → branch 2 →
  **`monitor`**. This is unchanged by the tightening — narrow device
  evidence was already, and remains, the one signal treated as meaningful
  corroboration rather than mere activity.

**3. `insufficient_evidence` — illustrative, not live-verified** (same
caveat as above):
- `historical_cases.direct_cases`: one entry, `outcome: "cleared"`;
  `connected_cases`: `[]`
- `connected_cards.connected_via_case`: `[]`
- `device_evidence`: `null`
- `billing_region`: `{"hub_flag": true, ...}`
- `temporal_activity`: 1 entry (just the flagged transaction)
- **Causing field:** no branch-1 condition holds (`cleared` != `confirmed_fraud`);
  no branch-2 condition holds (`device_evidence` is `null`, so the one
  remaining trigger cannot fire) → falls through to
  **`insufficient_evidence`**. The cleared case is still recorded in the
  evidence ledger, per Section 8 — it just isn't a trigger.

**Contrast — what the tightening actually changed:** take example 3 above,
but suppose `connected_cards.connected_via_case` were non-empty (a
cleared-case-only ring) *and* `temporal_activity` had 4 entries instead of
1, with `device_evidence` still `null`. Under the **prior** draft of this
rule, either of those alone would have triggered branch 2 → `monitor`.
Under the **tightened** rule, neither counts (see the three "why ... is
not a monitor trigger" analyses above), so this case still falls through
to `insufficient_evidence`. That is the intended effect of this tightening:
`monitor` no longer fires just because *something* was observed.

## 7. Architecture boundaries (finalized)

| Directory | Responsibility |
|---|---|
| `src/investigation/` | Deterministic evidence queries (Q1–Q8). Unchanged by this phase. |
| `src/mcp_server/` | MCP tool interface over those queries. Unchanged by this phase. |
| `src/agent/` | **This phase.** Investigation reasoning/workflow: trigger resolution, the Round 1/Round 2 sequence, evidence assessment, recommendation generation, the evidence ledger. |
| `src/policy/` | Reserved for a future phase: turning a recommendation into an actual permission/action decision. Not touched here. |
| `src/orchestrator/` | Reserved for a future phase: broader workflow orchestration beyond a single investigation (e.g. batch processing, scheduling). Not touched here. |

## 8. Evidence ledger (finalized)

Every investigation produces one ledger record. Minimum required fields:

```json
{
  "investigation_id": "uuid or similar, generated per run",
  "trigger": {"type": "flagged_txn_id | case_id", "value": "..."},
  "tool_calls": [
    {
      "order": 1,
      "tool": "combined_evidence",
      "arguments": {"flagged_txn_id": 3514030, "window_hours": 24.0},
      "success": true,
      "curated_result_summary": "small, curated fields only -- see below"
    }
  ],
  "uncertainties": [
    "e.g. Q5's device-churn evidence was large / device evidence absent / connected-card follow-up hit a tool error"
  ],
  "recommendation": "escalate | monitor | insufficient_evidence",
  "recommendation_basis": "which specific tool_calls entries (by order) the recommendation was derived from"
}
```

Requirements on this record:
- **`tool_calls` is in execution order**, with `order` starting at 1, and
  records every call the agent made (including Round 2's follow-ups) —
  this is the audit trail: a human reviewer must be able to see exactly
  what evidence produced the recommendation, not just read the narrative.
- **`success`/failure per call** must be recorded (a tool call that errors
  — e.g. a malformed id — is still a ledger entry, not silently dropped).
- **Curated, not raw.** The ledger stores curated evidence (the same
  already-curated field sets Q1–Q8 return — Phase E never returns full
  397-column payloads, and this phase must not start doing so either), not
  a dump of every byte a tool returned. No requirement to persist
  unnecessarily large raw graph responses (e.g. a `connected_via_device`
  list running into the hundreds does not need to be stored in full if a
  count plus a small sample is sufficient for the audit trail's purpose).
- **`uncertainties`** is a plain list of anything that limits confidence in
  the recommendation (missing device evidence, a failed tool call, an
  empty ring, etc.) — stated, not hidden.

### 8b. G7 addition — `full_evidence` (untrimmed, reasoning-layer support) (finalized)

The ledger's top-level record gained one additive key, `full_evidence`:

```json
{
  "full_evidence": {
    "combined_evidence": "the untrimmed dict returned by the single combined_evidence call, never passed through _trim()",
    "cross_card_fraud_verification": "the untrimmed cross_card_fraud_verification result if the Q9 gate fired, else null"
  }
}
```

Reason: Section 8's `tool_calls[].curated_result_summary` entries are
intentionally `_trim()`-collapsed for ledger compactness (any list past 5
entries becomes `{count, sample}`). That is correct for an audit log, but
it is not enough for a reasoning layer (Section 13) to ground its
transaction-ID citations against — a real case (HHG-011) has more than 20
distinct online transactions in one 24h window, and a trimmed sample would
hide most of them. `full_evidence` is purely additive: no existing key
changed shape or meaning, no existing test asserts an exhaustive key-set on
the ledger, and it costs zero additional tool calls — it carries results
already computed during the same investigation.

## 9. Separation of concerns (finalized, non-negotiable)

The LLM/agent reasons over evidence already returned by MCP tool calls. It
must not perform graph analysis itself (no inferring relationships,
counting connections, or computing anything a Q1–Q8 tool is responsible
for), and it must not replace a deterministic TigerGraph query with LLM
reasoning as a shortcut. If a future need arises for evidence this
workflow doesn't already gather, the answer is a new or extended
deterministic query (a Phase E/F-style change, reviewed as such) — never
the agent "figuring it out" from partial data.

## 10. Security (finalized, unchanged from Phase E/F, restated for this layer)

The agent must never receive or expose:
- `.env` contents
- `TG_GSQL_SECRET`
- JWTs
- Authorization headers
- cookies
- raw credential-bearing responses

The agent only ever talks to the existing MCP tool interface (never
`tigergraph_connection.py` directly), so it has no code path that could
originate a credential-bearing value in the first place. Any error that
does surface must continue to pass through the existing `scrub_secret()`
mechanism at the MCP layer (Phase F) before it could ever reach agent-level
logging or the evidence ledger.

## 11. Current phase boundary (finalized)

**In scope for Phase G's eventual implementation:**
- The bounded investigation workflow (Section 3)
- Tool selection/execution against the existing 8 MCP tools
- Evidence assessment and the deterministic follow-up selection rule (Section 5)
- Recommendation generation (Section 6)
- The audit/evidence ledger (Section 8)

**Still explicitly out of scope:**
- Autonomous actions of any kind
- Payment/card blocking
- Case updates written back into TigerGraph
- Policy engine implementation (`src/policy/`)
- Approval workflows
- Any UI
- New GSQL of any kind
- New MCP tools beyond the existing 8
- Permanent GSQL installation (`INSTALL QUERY`)

## 12. Open questions carried forward

None. Sections 3–8 resolved the four open questions from the original
draft (workflow shape, follow-up cap, recommendation vocabulary, directory
placement). Section 6a resolves the one item that draft explicitly left
open afterward: the fixed rule mapping evidence to a recommendation label.
No design dependency remains before Phase G implementation can start.

## 13. G7 addition — LLM reasoning layer (finalized)

Three new, isolated modules, none of which modify G1/G2/G5-A's own logic:

- `src/agent/reasoning.py` — builds the curated evidence package (FACTS /
  DETERMINISTIC FINDINGS / REASONING-ALLOWED / UNKNOWN, per this section's
  own boundary) and the prompt from it; invokes an injectable `call_llm`
  (same pattern as `workflow.py`'s `call_tool` injection — no default
  provider is wired in, and tests never depend on a live LLM); parses the
  response as JSON. Performs no semantic validation.
- `src/agent/reasoning_validator.py` — the deterministic acceptance gate.
  The LLM's raw JSON is never trusted directly: every entity ID it returns
  is checked against the evidence package's `known_ids` (built from
  `full_evidence`, Section 8b) and stripped if not grounded, never
  re-verified against TigerGraph and never silently swapped for another
  ID. Enforces the two deterministic overrides Section 9 requires: a
  positive R5 match (`pattern_evidence.matched`) forces `pattern =
  "card_testing"` regardless of the LLM's opinion, and the LLM asserting
  `card_testing` when R5 did *not* match is rejected outright (the same
  violation in the other direction). `exposure_usd` is always computed
  here, as a pure sum of absolute amounts over the final grounded ID list
  — the LLM never determines it directly.
- `src/agent/case_output.py` — the output adapter. Combines the ledger,
  the G2 policy result, and the validated reasoning output into the
  `case` and `next_best_actions` objects. `next_best_actions` is built
  entirely from G2's own `recommended` bucket — `prohibited`/`deferred`
  are never read here, which is what structurally guarantees a
  `PROHIBITED` action (e.g. `BLOCK_ALL_CARDS`) can never reach this
  output, let alone through the LLM. `similar_prior_cases` is likewise
  fully deterministic (confirmed-fraud entries from Q4's already-retrieved
  historical cases) — the LLM is never asked to select or invent case IDs.

Call-budget: this layer makes exactly one external call (`call_llm`) and
zero MCP/TigerGraph calls, by construction — nothing in `reasoning.py`,
`reasoning_validator.py`, or `case_output.py` imports
`src.graph.tigergraph_connection.get_connection` or
`src.mcp_server.server.mcp`. Verified by an active monkeypatch-based test
(`test_zero_mcp_tigergraph_calls_is_an_active_guard_not_an_inference`), not
just inferred from call counts. Section 4's ≤5-call ceiling governs
graph/tool calls only and is unaffected. The one `call_llm` call is bounded
by an explicit, configurable timeout (`reasoning.DEFAULT_LLM_TIMEOUT_S`,
60s, via `asyncio.wait_for`) — a non-responding provider cannot hang an
investigation indefinitely; a timed-out call fails exactly like any other
LLM failure (`failure_reason: "llm_timeout"`) and never reaches
`assemble_case`.

Failure handling: any LLM failure (call error, timeout, non-JSON response,
non-object JSON, missing keys, invalid enum, out-of-range probability, or
an entity ID that fails grounding entirely) produces an explicit `{"ok":
false, "failure_reason": ..., "stop_reason": ...}` result from
`case_output.investigate_and_assemble` — never a fabricated `case` object.
`stop_reason` is present on the success path too, built only from
vocabulary this project already established (G1's own `recommendation`
value and the reasoning layer's own validated `verdict`) — no new semantic
meaning invented for the field.

Deliberately out of scope this phase (not invented here): `sar`,
`evidence_requests`, `tool_calls`/`tokens`/`latency_s`, `connected_card_ids`,
`connected_device_profiles`, `written_to_graph`, `graph_case_id`, `case_id`
at the top level, and any `next_best_actions` evolution beyond `initial ==
final` (no evidence-request/response loop exists yet).

## 14. G8 addition — real LLM providers + prompt-serialization compaction (finalized)

Two provider adapters, `src/agent/llm_providers/gemini_provider.py`
(`make_gemini_call_llm`, default model `gemini-3.6-flash`) and
`groq_provider.py` (`make_groq_call_llm`, `openai/gpt-oss-20b`), each
producing nothing but a `CallLLM` closure — Section 13's isolation
guarantees are unchanged and independently re-verified for both (zero
MCP/TigerGraph imports, confirmed by AST inspection tests; no retries; no
timeout of their own, relying entirely on Section 13's existing 60s
`asyncio.wait_for`).

**A real defect was found and fixed via live HHG-011 data**, before any
LLM call was involved: `g2_policy._r6_evidence_present` read
`tool_calls[].curated_result_summary`, which `_trim()` collapses past 5
entries — for a real, highly-connected card (795 device-linked cards),
`other_cards` becomes `{"count": 795, "sample": [...]}`, and iterating a
dict yields its *keys* as bare strings, crashing on `entry.get(...)`. Fixed
by reading `full_evidence` (untrimmed) when present, falling back to the
old path for pre-G7 ledgers, with an `isinstance` guard against a repeat.

**Prompt-serialization-only compaction** (the actual fix for a second real
finding — a genuine HHG-011 prompt reached 441,741 characters and was
rejected outright by a provider): `evidence_package["facts"]` and
`["known_ids"]` (built by `_collect_known_ids`/`_compact_cross_card_evidence`)
remain **complete** — every confirmed-fraud connected card, in full, no
first-N sampling, plus all six known-id categories (txn/card/customer/
device/case/region). `build_prompt()` alone renders a narrower view:
`_compact_cross_card_for_prompt` re-shapes confirmed-fraud records into
`[card_key, same_customer, confirmed_fraud_case_ids]` arrays (customer_id
dropped from the *rendered text only* — always derivable from `card_key`
by this project's own `card_key == f"{customer_id}:{card1}"` convention,
and never an output field), and the KNOWN IDS block renders only `txn`
(`REASONING_OUTPUT_SCHEMA` has no card/customer/device/case/region output
field). Net result on real HHG-011 data: 441,741 → 140,570 characters
(68.2% reduction), 58,038 exact tokens against `gemini-3.6-flash`'s
1,048,576-token window (measured via Gemini's own `count_tokens`, not
estimated).

`scrub_secret()` (`src/graph/tigergraph_connection.py`) was widened to
cover `GROQ_API_KEY`/`GEMINI_API_KEY` fragment-matching, the same
mechanism already used for `TG_GSQL_SECRET` — confirmed empirically that
neither key format would have been caught by the pre-existing JWT/Bearer/
named-field rules.

## 15. G9 addition — end-to-end benchmark runner (finalized)

`src/benchmark/g9_runner.py` adds **no new investigation, reasoning,
validation, or policy logic** — it only iterates `workflow.investigate()`
→ `g2_policy.evaluate()` → `case_output.investigate_and_assemble()`
(unmodified) across the 20 `case_pack_resolved.csv` rows, each as a
`{"type": "case_id", "value": ...}` trigger (the same trigger form used
throughout this project's real-data testing). Never calls TigerGraph
directly and never bypasses MCP: the only path to graph data remains
`workflow.investigate()`'s own `call_tool`, defaulting to the real MCP
dispatch exactly as G1 already established.

**Call budget**: exactly one LLM call per case, unchanged from Section
13/14 — the runner's only addition is a non-invasive recording wrapper
around the injected `call_llm` (captures the raw response/error for the
audit artifact, forwards to the real call unchanged, adds no call, no
retry, no timeout of its own).

**Per-case isolation**: a case whose `workflow.investigate()` call itself
raises (e.g. a transient TigerGraph error) is caught at the loop level,
recorded as `{"success": false, "stage": "runner_exception", "error":
scrub_secret(...)}`, and the run continues — one case's failure never
stops the remaining 19.

**Command**:
```bash
python -m src.benchmark.g9_runner --provider gemini
python -m src.benchmark.g9_runner --provider gemini --case HHG-001   # single case
python -m src.benchmark.g9_runner --provider groq --output-dir outputs_groq
```

**Output** (gitignored — see `.gitignore`, same reasoning as
`data/processed/`): `outputs/case_results/<CASE_ID>.json` per case
(`case_id`, `investigation_id`, `trigger`, `flagged_transaction`, `g1`
[recommendation/basis/tool_calls/uncertainties/pattern_evidence],
`g2_policy` [full `evaluate()` output], `llm` [raw_response/call_error],
`outcome` [the real `case_output.investigate_and_assemble()` return —
`case`/`next_best_actions`/`stripped_ids`/`stop_reason` on success, or
`failure_reason`/`detail`/`stop_reason` on failure], `success`,
`timing_s`), plus one `outputs/evaluation_summary.json` (total/successful/
failed case counts, LLM successes/failures, validator failures, pipeline
exceptions, per-case policy results, per-case and total execution time,
and a `failure_reasons` tally). **No accuracy or correctness score is
computed or invented** — no organizer answer key exists in this
repository to compare against.

**Live run result (this phase, real data, honestly recorded — not all 20
succeeded)**: 20/20 cases attempted, 1 succeeded end-to-end (HHG-003 —
real verdict `uncertain`, `fraud_probability` 0.45, grounded
`affected_txn_ids`, deterministic `exposure_usd`), 19 failed at the LLM
call stage: 13 with `503 UNAVAILABLE` (Gemini `gemini-3.6-flash`
reporting genuine, current high-demand overload) and 6 with `429
RESOURCE_EXHAUSTED` (this account's request quota, exhausted by this
session's cumulative real testing volume). Zero fabricated results; every
failure recorded exactly as returned; zero retries attempted anywhere in
this run. `DEFAULT_MAX_OUTPUT_TOKENS` was raised from 1024 to 4096 during
this phase after a real HHG-001 attempt produced a JSON-schema-violating
truncated response at the old value — a disclosed configuration change,
not an architecture change.

## 16. G10 addition — case memory / graph write-back (finalized)

One schema addition, executed as a global schema change job (confirmed
necessary: `HHG_InvestigationCase` is registered in TigerGraph's
server-wide type catalog, so a graph-scoped `ALTER VERTEX` is rejected):
`first_suspicious_txn_id UINT DEFAULT 0` on `HHG_InvestigationCase`,
mirroring the existing `flagged_txn_id UINT` column exactly. No new
vertex or edge type — `HHG_INVOLVES` (`InvestigationCase → Transaction`,
declared Phase D, never loaded for InvestigationCase) and `HHG_CITES`
(`InvestigationCase → ClosedCase`, declared Phase D, never loaded at
all) are reused exactly as already declared, for `affected_txn_ids` and
`similar_prior_cases` respectively.

**Write path** (`src/investigation/case_memory.py`, the *only*
write-capable TigerGraph function in this project — deliberately kept
out of `queries.py`, whose own docstring promises every function there
is read-only): four sequential, non-atomic steps via pyTigerGraph
2.0.4's native `upsertVertex`/`upsertEdges` (signatures verified via
`inspect.signature` against the installed package, not guessed) —
vertex content attributes (`written_to_graph` explicitly `False`) →
`HHG_INVOLVES` edges → `HHG_CITES` edges → a final, minimal vertex
upsert flipping `written_to_graph` to `True`. Any failure at any step
stops immediately and leaves `written_to_graph` `False`; there is no
code path that reports success after a partial sequence. Both edge
upserts use `vertexMustExist=True` — an id that doesn't already exist in
the graph is rejected, never silently creating a phantom vertex.

**Orchestration** (`src/agent/case_writer.py`): consumes only an already
`ok: true` `case_output.investigate_and_assemble()` result, calls the
new `write_case_memory` MCP tool (never TigerGraph directly), and never
reads `g2_result` at all — locked decision: every reasoning-successful
investigation is persisted regardless of what G2 recommends; case-memory
persistence and action execution are unconditionally separate, and a
successful write never triggers any of the 14 fraud-policy actions. The
LLM has no path to this module — confirmed by structural (AST-based)
tests that `reasoning.py`/`reasoning_validator.py` never import it.

**Idempotency**: vertices update by primary ID (TigerGraph's own
upsert-by-primary-id semantics, already relied on by the Phase D seed
job); edges update by `(from, to)` pair (neither `HHG_INVOLVES` nor
`HHG_CITES` declares a discriminator attribute). **Locked decision:
edges are monotonic — a rerun with a smaller `affected_txn_ids`/
`similar_prior_cases` set never deletes previously-written edges.**
Graph lifecycle uses one new explicit value, `INVESTIGATED`, deliberately
kept separate from the organizer's own `case.status` answer-file enum
(`open`/`closed_fraud`/`closed_legitimate`/`escalated`) — no larger state
machine invented. `graph_case_id` is `""` whenever `written_to_graph` is
`False`.

**Live-verified this phase** (real, previously-obtained HHG-003 Gemini
result reused — zero new Gemini calls): a full write succeeded end to
end (vertex attributes, 1 `HHG_INVOLVES` edge, 5 `HHG_CITES` edges, all
read back and confirmed correct), then the identical write was repeated
and confirmed idempotent — `HHG_InvestigationCase` vertex count stayed
at 20, `HHG_INVOLVES` stayed at 1, `HHG_CITES` stayed at 5. `Transaction_
Fraud` confirmed untouched throughout (18 vertex types, none `HHG_`
-prefixed) both before and after the schema change and the write.
