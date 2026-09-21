"""
Phase E Step 8: measure query latency via repeated executions. Read-only.
Uses a fixed, real input per query (HHG-001's flagged transaction and its
resolved card) so every repetition does identical work.
"""
import json
import statistics
import time

from src.graph.tigergraph_connection import get_connection
from src.investigation import queries as q

conn = get_connection(graphname="HHGOA_FraudInvestigation")

TXN_ID = 3514030  # HHG-001
CARD_KEY = "C12382:21139"
DEVICE_PROFILE = "SM-G935F Build/NRD90M|Android 7.0|chrome 62.0 for android|1920x1080"
CASE_ID = "HHG-001"

REPS = 20


def timed(fn, *args, **kwargs):
    times = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    times.sort()
    n = len(times)
    return {
        "reps": n,
        "p50_ms": round(times[int(n * 0.50) - 1] * 1000, 1),
        "p95_ms": round(times[min(int(n * 0.95), n - 1)] * 1000, 1),
        "p99_ms": round(times[min(int(n * 0.99), n - 1)] * 1000, 1),
        "min_ms": round(times[0] * 1000, 1),
        "max_ms": round(times[-1] * 1000, 1),
    }


results = {}
results["Q1_transaction_lookup"] = timed(q.q1_transaction_lookup, conn, TXN_ID)
print("Q1 done")
results["Q2_temporal_neighborhood"] = timed(q.q2_temporal_neighborhood, conn, TXN_ID, 24.0)
print("Q2 done")
results["Q3_device_sharing"] = timed(q.q3_device_sharing, conn, DEVICE_PROFILE)
print("Q3 done")
results["Q4_historical_case_evidence"] = timed(q.q4_historical_case_evidence, conn, CARD_KEY)
print("Q4 done")
results["Q5_connected_cards"] = timed(q.q5_connected_cards, conn, CARD_KEY)
print("Q5 done")
results["Q6_billing_region_evidence"] = timed(q.q6_billing_region_evidence, conn, TXN_ID)
print("Q6 done")
results["Q7_benchmark_case_resolution"] = timed(q.q7_benchmark_case_resolution, conn, CASE_ID)
print("Q7 done")
results["Q8_combined_evidence"] = timed(q.q8_combined_evidence, conn, TXN_ID, 24.0)
print("Q8 done")

print(json.dumps(results, indent=2))
with open("data/processed/phase_e_latency_benchmark.json", "w") as f:
    json.dump({"reps_per_query": REPS, "fixed_input": {"txn_id": TXN_ID, "card_key": CARD_KEY, "device_profile": DEVICE_PROFILE, "case_id": CASE_ID}, "results": results}, f, indent=2)
print("written data/processed/phase_e_latency_benchmark.json")
