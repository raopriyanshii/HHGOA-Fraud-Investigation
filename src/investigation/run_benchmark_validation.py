"""
Phase E Step 6: run the deterministic investigation layer against all 20
benchmark cases and record the 8 required checks per case. Read-only.
"""
import csv
import json
import time

from src.graph.tigergraph_connection import get_connection
from src.investigation import queries as q

conn = get_connection(graphname="HHGOA_FraudInvestigation")

with open("data/processed/case_pack_resolved.csv", newline="") as f:
    rows = list(csv.DictReader(f))

results = []
for row in rows:
    case_id = row["case_id"]
    expected_customer = row["customer_id"]
    expected_card_key = row["card_key"]
    expected_txn = int(row["flagged_txn_id"])

    checks = {}
    t0 = time.perf_counter()

    q7 = q.q7_benchmark_case_resolution(conn, case_id)
    checks["1_flagged_transaction_resolves"] = q7 is not None and q7["flagged_txn_id"] == expected_txn

    q1 = q.q1_transaction_lookup(conn, expected_txn)
    checks["2_card_resolves_correctly"] = q1 is not None and q1["card"] is not None and q1["card"]["card_key"] == expected_card_key
    checks["3_customer_resolves_correctly"] = q1 is not None and q1["customer"] is not None and q1["customer"]["customer_id"] == expected_customer
    checks["8_no_incorrect_mapping"] = (
        q7 is not None and q7["customer_id"] == expected_customer and q7["card_key"] == expected_card_key
    )

    try:
        q2 = q.q2_temporal_neighborhood(conn, expected_txn, 24.0)
        checks["4_transaction_chronology_works"] = q2 is not None and len(q2["transactions"]) >= 1
    except Exception as e:
        checks["4_transaction_chronology_works"] = False
        checks["4_error"] = str(e)

    try:
        q4 = q.q4_historical_case_evidence(conn, expected_card_key)
        checks["5_historical_cases_retrievable"] = isinstance(q4["direct_cases"], list) and isinstance(q4["connected_cases"], list)
        checks["5_direct_case_count"] = len(q4["direct_cases"])
    except Exception as e:
        checks["5_historical_cases_retrievable"] = False
        checks["5_error"] = str(e)

    try:
        device_evidence = None
        if q1 and q1["device"]:
            device_evidence = q.q3_device_sharing(conn, q1["device"]["device_profile"])
        checks["6_device_relationships_retrievable"] = True  # no exception, whether or not a device existed
        checks["6_had_device"] = device_evidence is not None
    except Exception as e:
        checks["6_device_relationships_retrievable"] = False
        checks["6_error"] = str(e)

    try:
        q5 = q.q5_connected_cards(conn, expected_card_key)
        checks["7_connected_card_relationships_retrievable"] = isinstance(q5["connected_via_case"], list) and isinstance(q5["connected_via_device"], list)
    except Exception as e:
        checks["7_connected_card_relationships_retrievable"] = False
        checks["7_error"] = str(e)

    elapsed = time.perf_counter() - t0
    all_pass = all(v for k, v in checks.items() if not k.endswith(("_error", "_count", "_had_device")))
    results.append({"case_id": case_id, "elapsed_s": round(elapsed, 3), "all_pass": all_pass, "checks": checks})
    print(f"{case_id}: all_pass={all_pass} elapsed={elapsed:.2f}s", flush=True)

    # write incrementally so a later failure doesn't lose earlier results
    with open("data/processed/phase_e_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

n_pass = sum(1 for r in results if r["all_pass"])
print(f"\n{n_pass}/20 cases fully passed all 8 checks")

with open("data/processed/phase_e_benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("written data/processed/phase_e_benchmark_results.json")
