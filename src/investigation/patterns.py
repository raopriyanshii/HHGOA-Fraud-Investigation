"""
Deterministic R5 "card testing" pattern evidence detector (G5-A).

This module performs deterministic EVIDENCE ANALYSIS only. A `matched`
result is NOT a fraud verdict, NOT a policy decision, and NOT an
instruction to take any action -- it reports whether the specific,
literal transaction-sequence shape the organizer's R5 rule describes is
present in already-gathered evidence:

  "Three or more small online authorizations on one card within an hour,
  followed by a larger purchase" (organizer Fraud Policy, R5).

Wiring a match into an actual recommendation (DECLINE_TRANSACTION /
STEP_UP_AUTH / BLOCK_CARD) is a separate, later step -- not part of this
module.

No TigerGraph access, no MCP call, no LLM: this is pure Python over the
`temporal_activity` list G1's existing `combined_evidence` call already
returns. Nothing here invents a numeric dollar threshold for "small" --
the organizer material never defines one (the Known Fraud Patterns
section's "often under $5" is an illustrative example, hedged with
"often," not a rule; R5's own policy text drops the number entirely).
"Small" is therefore established relationally, never absolutely: an
authorization only ever counts as part of a candidate because a specific,
identified later transaction is strictly larger than it -- never because
its amount is below some invented cutoff.

The two numeric constants below (3, 60 minutes) ARE official, literally
quoted from R5 itself ("three or more", "within an hour") -- they are not
in the same category as the forbidden dollar threshold.

Multiple candidates: a single card's temporal_activity can contain more
than one structurally-qualifying R5 sequence (a real example: HHG-011).
This detector reports every one it finds, in chronological discovery
order, rather than picking a single "best" one -- doing the latter would
require a scoring/ratio/strength judgment this project has deliberately
chosen not to invent. Consumers of this evidence decide what to do with
multiple candidates; this module only reports what's structurally there.
"""
from __future__ import annotations

from datetime import datetime

_TS_FMT = "%Y-%m-%d %H:%M:%S"
_MIN_AUTHORIZATIONS = 3  # R5: "three or more"
_MAX_SPAN_MINUTES = 60.0  # R5: "within an hour" (inclusive)


def classify_card_testing(temporal_activity: list[dict], flagged_txn_id: int) -> dict:
    """Scans `temporal_activity` (as G1 already gathers it) for every R5
    card-testing-shaped sequence present. `flagged_txn_id` is accepted for
    interface consistency with how this evidence would be consumed
    downstream, but is not otherwise used by the detection logic below:
    the flagged transaction is just one more entry in `temporal_activity`,
    eligible to participate in a candidate like any other. Does not
    reconstruct or infer card identity -- `temporal_activity` already
    represents a single card's context, exactly as G1's own
    combined_evidence call already scopes it.
    """
    online = sorted(
        (t for t in temporal_activity if t.get("channel") == "online"),
        key=lambda t: t["ts"],
    )

    candidates: list[dict] = []
    seen_clusters: set[tuple] = set()

    for start in range(len(online) - _MIN_AUTHORIZATIONS + 1):
        cluster = online[start:start + _MIN_AUTHORIZATIONS]
        first_ts = datetime.strptime(cluster[0]["ts"], _TS_FMT)
        last_ts = datetime.strptime(cluster[-1]["ts"], _TS_FMT)
        span_minutes = (last_ts - first_ts).total_seconds() / 60.0
        if span_minutes > _MAX_SPAN_MINUTES:
            continue

        cluster_ids = tuple(t["TransactionID"] for t in cluster)
        if cluster_ids in seen_clusters:
            continue  # only guards against an exact duplicate window (e.g. repeated input rows)

        cluster_max = max(t["TransactionAmt"] for t in cluster)
        for later in online[start + _MIN_AUTHORIZATIONS:]:
            if later["TransactionAmt"] > cluster_max:
                seen_clusters.add(cluster_ids)
                candidates.append({
                    "authorization_txn_ids": list(cluster_ids),
                    "authorization_count": len(cluster),
                    "authorization_span_minutes": round(span_minutes, 2),
                    "larger_purchase_txn_id": later["TransactionID"],
                    "larger_purchase_amount": later["TransactionAmt"],
                })
                break  # first qualifying later transaction for THIS cluster; move to the next start

    matched = len(candidates) > 0
    return {
        "matched": matched,
        "pattern": "card_testing",
        "candidates": candidates,
        "uncertainty": (
            [
                "No numeric definition of 'small' is asserted -- the organizer material does not "
                "provide one. Authorizations are 'small' only relative to each candidate's "
                "identified larger purchase, not against any absolute dollar figure. Multiple "
                "candidates, when present, are not ranked or scored against each other.",
            ]
            if matched
            else []
        ),
    }
