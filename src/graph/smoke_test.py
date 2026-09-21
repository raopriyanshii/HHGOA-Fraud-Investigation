"""
Read-only connection/authentication smoke test. Does not create, modify,
or load anything. Never prints the secret/token; only pass/fail status
and non-sensitive diagnostic text (version strings, graph names already
visible to the account, error messages).

graphname is a parameter, not hardcoded: an empty string tests global-scope
auth; passing an existing graph's name (e.g. "Transaction_Fraud") tests
auth scoped to that graph WITHOUT reading or writing any of its data --
every call below (echo, getVer, "ls") is metadata-only.
"""
from src.graph.tigergraph_connection import get_connection, scrub_secret


def run(graphname: str = ""):
    results = {}
    conn = get_connection(graphname=graphname)

    try:
        echo_result = conn.echo()
        results["echo"] = {"ok": True, "response": echo_result}
    except Exception as e:
        results["echo"] = {"ok": False, "error": scrub_secret(f"{type(e).__name__}: {e}")}

    try:
        version = conn.getVer()
        results["getVer"] = {"ok": True, "response": version}
    except Exception as e:
        results["getVer"] = {"ok": False, "error": scrub_secret(f"{type(e).__name__}: {e}")}

    try:
        # "ls" is a read-only GSQL catalog listing -- it does not modify
        # or query any graph's data, only shows metadata already visible
        # to this account (confirms GSQL-level auth works too).
        catalog = conn.gsql("ls")
        results["gsql_ls"] = {"ok": True, "response": catalog}
    except Exception as e:
        results["gsql_ls"] = {"ok": False, "error": scrub_secret(f"{type(e).__name__}: {e}")}

    return results


if __name__ == "__main__":
    import sys

    graphname = sys.argv[1] if len(sys.argv) > 1 else ""
    results = run(graphname=graphname)
    for check, outcome in results.items():
        status = "PASS" if outcome["ok"] else "FAIL"
        print(f"\n=== {check}: {status} ===")
        if outcome["ok"]:
            print(outcome["response"])
        else:
            print(outcome["error"])
