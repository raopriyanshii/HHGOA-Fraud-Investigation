# Phase G1 Implementation Note

Implements the workflow skeleton defined in `docs/AGENT_WORKFLOW_SPEC.md`
(frozen at commit `123473e`). This note is a pointer to the code, not a
restatement of the spec -- read the spec for the design rationale.

## Files

- `src/agent/workflow.py` -- the workflow itself. `investigate(trigger, *,
  window_hours=24.0, call_tool=None)` is the one public entrypoint.
  `call_tool` defaults to real MCP dispatch (`_default_call_tool`, which
  calls `src.mcp_server.server.mcp.call_tool` -- never a Phase E function
  directly) and is injectable so tests can supply a fake without touching
  TigerGraph.
- `src/tests/test_agent_workflow.py` -- 15 unit tests, all using a mocked
  `call_tool`, no live dependency.
- `src/agent/validate_workflow.py` -- integration script, real MCP
  dispatch, no mocking. Run with `python -m src.agent.validate_workflow`.

## Known limitation from this session, not from the code

TigerGraph has been unreachable (`500` on the base `/restpp/version`
endpoint) since partway through the Phase G specification work, and
remained down through this implementation turn. Consequences, stated
plainly:

- `src/tests/test_investigation_queries.py` (Phase E's own 22 live tests)
  currently fail for this reason -- confirmed unrelated to this phase's
  changes by re-running them unmodified.
- The integration script (`validate_workflow.py`) was run against the real
  MCP/TigerGraph layer and did exercise real, non-mocked failure handling
  end-to-end (the `benchmark_case_resolution` call failed with the live
  connectivity error, and the workflow correctly produced a controlled
  `insufficient_evidence` result instead of crashing). It has **not** yet
  exercised the happy path -- real evidence flowing into a real `escalate`
  determination -- because that requires TigerGraph to actually respond.
  That check is still outstanding and should be re-run once connectivity
  is restored; it isn't fabricated here.

## Where each hard limit is enforced (not just documented)

- Max 3 follow-up cards: `[:MAX_FOLLOW_UP_CARDS]` slice at selection time,
  plus an `assert` after Round 2 as a second, independent check.
- No Round 3 / no recursion: structural -- `investigate()` has no loop or
  recursive call that could feed Round 2's results back into another tool
  call; the function body is a fixed sequence, not a loop over rounds.
- Only 3 named tools ever dispatched: `_ALLOWED_TOOLS` allowlist checked
  in `_default_call_tool`; no tool name is ever built from trigger or
  evidence data, only literal string constants appear at call sites.
- No large raw payloads in the ledger: `_trim()` bounds any list over 5
  entries to `{count, sample}` when writing `curated_result_summary`;
  the recommendation rule itself still evaluates the untrimmed evidence.
