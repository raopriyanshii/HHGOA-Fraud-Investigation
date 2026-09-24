"use client";

import { useState } from "react";

const CASES = Array.from({ length: 20 }, (_, i) =>
  `HHG-${String(i + 1).padStart(3, "0")}`
);

const API_URL = "http://localhost:8000/investigate";

// --- Response types, matching the REAL organizer submission shape the
// backend's /investigate endpoint returns (verified against actual
// generated case output, not assumed) -- every field is optional/
// nullable here on purpose, since the backend response can vary by case
// (e.g. a failed investigation returns an honest, mostly-empty shape). ---

type ActionEntry = {
  action: string;
  route?: string | null;
  reason?: string;
};

type NextBestActions = {
  initial?: ActionEntry[];
  final?: ActionEntry[];
  what_changed?: string;
};

type EvidenceEntry = {
  claim?: string;
  source?: string;
  ref?: string;
  entity_ids?: string[];
};

type EvidenceRequestEntry = {
  type?: string;
  asked_after_step?: number;
  assumed_response?: string;
};

type SarInfo = {
  file?: boolean;
  reason?: string;
  narrative?: string;
  subjects?: string[];
  total_amount_usd?: number;
  activity_dates?: string[];
};

type CaseInfo = {
  status?: string;
  verdict?: string;
  fraud_probability?: number;
  pattern?: string;
  pattern_description?: string;
  affected_txn_ids?: string[];
  first_suspicious_txn_id?: string;
  connected_card_ids?: string[];
  connected_device_profiles?: string[];
  exposure_usd?: number;
  evidence?: EvidenceEntry[];
  similar_prior_cases?: string[];
  summary?: string;
  written_to_graph?: boolean;
  graph_case_id?: string;
};

type InvestigationResult = {
  case_id?: string;
  case?: CaseInfo;
  evidence_requests?: EvidenceRequestEntry[];
  next_best_actions?: NextBestActions;
  sar?: SarInfo;
  stop_reason?: string;
  // NOTE: the real backend response carries `tool_calls` as a plain
  // integer COUNT and `tokens` as a plain integer (always 0 today) --
  // not arrays/objects. Typed to match what the API actually returns.
  tool_calls?: number;
  tokens?: number;
  latency_s?: number;
};

const STATUS_BADGE_STYLE: Record<string, string> = {
  closed_fraud: "border-red-500/30 bg-red-500/10 text-red-300",
  closed_legitimate: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300",
  escalated: "border-amber-500/30 bg-amber-500/10 text-amber-300",
  open: "border-zinc-600 bg-zinc-800/60 text-zinc-400",
};

export default function Home() {
  const [selectedCase, setSelectedCase] = useState("HHG-001");
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<InvestigationResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleCaseChange = (caseId: string) => {
    setSelectedCase(caseId);
    // A stale result for a different case would be misleading -- clear
    // it whenever the selection changes rather than leaving it on screen.
    setResult(null);
    setError(null);
  };

  const runInvestigation = async () => {
    setRunning(true);
    setError(null);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ case_id: selectedCase }),
      });

      if (!response.ok) {
        // Only ever surface the backend's own fixed, generic message
        // (e.g. {"detail": "Investigation failed"}) -- never raw
        // response text, which could echo something unexpected.
        let message = "Investigation failed.";
        try {
          const body = await response.json();
          if (body && typeof body.detail === "string") {
            message = body.detail;
          }
        } catch {
          // Non-JSON error body -- keep the generic message.
        }
        setError(message);
        setResult(null);
        return;
      }

      const data: InvestigationResult = await response.json();
      setResult(data);
    } catch {
      setError(
        "Could not reach the investigation service. Confirm the API is running at " +
          API_URL +
          "."
      );
      setResult(null);
    } finally {
      setRunning(false);
    }
  };

  const caseInfo = result?.case;
  const sar = result?.sar;
  const nba = result?.next_best_actions;
  const evidenceRequests = result?.evidence_requests ?? [];
  const evidenceEntries = caseInfo?.evidence ?? [];

  // Grounded only in what the response actually says: the backend's own
  // stop_reason text always starts with "G1 investigation completed"
  // exactly when the full G1 -> G2 -> reasoning pipeline genuinely
  // succeeded (see src/agent/case_output.py) -- never inferred from
  // anything else.
  const pipelineSucceeded =
    typeof result?.stop_reason === "string" &&
    result.stop_reason.startsWith("G1 investigation completed");

  const transactionId =
    caseInfo?.first_suspicious_txn_id ||
    (caseInfo?.affected_txn_ids && caseInfo.affected_txn_ids.length > 0
      ? caseInfo.affected_txn_ids[0]
      : "");

  // The organizer case schema has no dedicated customer_id field. When a
  // SAR was filed, sar.subjects[0] is reliably the primary customer_id
  // (see src/agent/case_output.py::_sar_subjects, which always inserts
  // it first) -- used here only when it's actually present, never guessed.
  const customerId =
    sar?.file && sar.subjects && sar.subjects.length > 0 ? sar.subjects[0] : "";

  return (
    <main className="min-h-screen bg-[#07090d] text-zinc-100">
      <header className="border-b border-white/8 bg-[#090b10]">
        <div className="mx-auto flex h-16 max-w-[1500px] items-center justify-between px-6">
          <div className="flex items-center gap-4">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-red-500/30 bg-red-500/10 text-sm font-bold text-red-400">
              FI
            </div>

            <div>
              <div className="text-sm font-semibold tracking-wide">
                FRAUD INVESTIGATION
              </div>
              <div className="text-[11px] text-zinc-500">
                Agentic investigation console
              </div>
            </div>
          </div>

          <div className="flex items-center gap-5 text-xs">
            <div className="hidden items-center gap-2 text-zinc-400 sm:flex">
              <span className="h-2 w-2 rounded-full bg-emerald-400" />
              TigerGraph connected
            </div>

            <div className="hidden items-center gap-2 text-zinc-400 sm:flex">
              <span className="h-2 w-2 rounded-full bg-emerald-400" />
              Agent ready
            </div>

            <div className="rounded-md border border-white/10 bg-white/5 px-3 py-1.5 text-zinc-400">
              HHGOA 2026
            </div>
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-[1500px] px-6 py-6">
        <div className="mb-6 flex flex-col justify-between gap-4 lg:flex-row lg:items-end">
          <div>
            <p className="mb-2 text-xs font-medium uppercase tracking-[0.2em] text-red-400">
              Investigation workspace
            </p>

            <h1 className="text-2xl font-semibold tracking-tight">
              Case Investigation
            </h1>

            <p className="mt-1 text-sm text-zinc-500">
              Investigate suspicious transactions using graph evidence,
              policy rules and agentic reasoning.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <select
              value={selectedCase}
              onChange={(e) => handleCaseChange(e.target.value)}
              disabled={running}
              className="rounded-lg border border-white/10 bg-[#0d1016] px-4 py-2.5 text-sm text-zinc-200 outline-none transition focus:border-red-500/50 disabled:opacity-60"
            >
              {CASES.map((caseId) => (
                <option key={caseId} value={caseId}>
                  {caseId}
                </option>
              ))}
            </select>

            <button
              onClick={runInvestigation}
              disabled={running}
              className="rounded-lg bg-red-500 px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-red-400 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {running ? "Investigating..." : "Investigate Case"}
            </button>
          </div>
        </div>

        {error && (
          <div className="mb-5 rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-300">
            {error}
          </div>
        )}

        <section className="mb-5 grid grid-cols-2 gap-3 lg:grid-cols-4">
          <StatCard label="Selected Case" value={selectedCase} />
          <StatCard label="Workflow" value="G1 → G2 → Reasoning" />
          <StatCard label="Evidence" value="Graph + Case Memory" />
          <StatCard
            label="Status"
            value={running ? "RUNNING" : result ? "COMPLETE" : "READY"}
          />
        </section>

        <div className="grid gap-5 lg:grid-cols-[1.6fr_1fr]">
          <div className="space-y-5">
            <Panel
              title="Case Overview"
              subtitle="Investigation trigger and transaction context"
            >
              <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
                <Info label="Case ID" value={result?.case_id || selectedCase} />
                <Info
                  label="Investigation"
                  value={caseInfo?.status ?? "Pending"}
                />
                <Info
                  label="Transaction"
                  value={transactionId || "Awaiting investigation"}
                />
                <Info
                  label="Customer"
                  value={customerId || "Not available"}
                />
              </div>

              <div className="mt-5 rounded-lg border border-dashed border-white/10 bg-black/10 p-5">
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-xs font-medium uppercase tracking-wider text-zinc-500">
                    Investigation trigger
                  </span>

                  {caseInfo?.status ? (
                    <span
                      className={`rounded-full border px-2.5 py-1 text-[10px] font-medium ${
                        STATUS_BADGE_STYLE[caseInfo.status] ??
                        "border-amber-500/20 bg-amber-500/10 text-amber-300"
                      }`}
                    >
                      {caseInfo.status.toUpperCase()}
                    </span>
                  ) : (
                    <span className="rounded-full border border-amber-500/20 bg-amber-500/10 px-2.5 py-1 text-[10px] font-medium text-amber-300">
                      PENDING
                    </span>
                  )}
                </div>

                <p className="text-sm leading-6 text-zinc-400">
                  {result?.stop_reason ||
                    "Select a benchmark case and run the investigation to populate transaction, customer, graph and policy evidence."}
                </p>
              </div>
            </Panel>

            <div className="grid gap-5 md:grid-cols-2">
              <Panel
                title="Risk Assessment"
                subtitle="Input signal — not the final decision"
              >
                <div className="flex items-center gap-5">
                  <div className="flex h-24 w-24 items-center justify-center rounded-full border-4 border-zinc-800 bg-zinc-900">
                    <div className="text-center">
                      <div className="text-2xl font-semibold text-zinc-500">
                        {typeof caseInfo?.fraud_probability === "number"
                          ? `${Math.round(caseInfo.fraud_probability * 100)}%`
                          : "—"}
                      </div>
                      <div className="text-[9px] uppercase tracking-wider text-zinc-600">
                        score
                      </div>
                    </div>
                  </div>

                  <div>
                    <div className="text-sm font-medium text-zinc-300">
                      {caseInfo?.verdict
                        ? `Verdict: ${caseInfo.verdict}`
                        : "Awaiting investigation"}
                    </div>
                    <p className="mt-1 text-xs leading-5 text-zinc-500">
                      This is the agent&apos;s calibrated fraud probability —
                      an input signal. The policy engine, not this score,
                      determines the final recommended actions.
                    </p>
                  </div>
                </div>
              </Panel>

              <Panel
                title="Evidence"
                subtitle="Graph-derived investigation signals"
              >
                {!result ? (
                  <div className="space-y-3">
                    <EvidenceRow label="Transaction history" />
                    <EvidenceRow label="Customer profile" />
                    <EvidenceRow label="Device connections" />
                    <EvidenceRow label="Connected cards" />
                    <EvidenceRow label="Historical cases" />
                  </div>
                ) : evidenceEntries.length === 0 ? (
                  <EmptyState text="No evidence recorded for this case." />
                ) : (
                  <div className="space-y-3">
                    {evidenceEntries.map((ev, i) => (
                      <EvidenceClaim key={i} evidence={ev} />
                    ))}
                  </div>
                )}
              </Panel>
            </div>

            <Panel
              title="Recommended Actions"
              subtitle="Policy-constrained next-best actions"
            >
              {!result ? (
                <>
                  <div className="rounded-lg border border-white/8 bg-[#0b0e13] p-4">
                    <div className="flex items-center justify-between">
                      <span className="text-xs uppercase tracking-wider text-zinc-500">
                        Initial recommendation
                      </span>

                      <span className="rounded-md bg-zinc-800 px-2 py-1 text-[10px] text-zinc-500">
                        PENDING
                      </span>
                    </div>

                    <div className="mt-3 text-lg font-semibold text-zinc-600">
                      No action generated
                    </div>

                    <p className="mt-1 text-xs leading-5 text-zinc-600">
                      Run the investigation to generate policy-compliant
                      actions.
                    </p>
                  </div>

                  <div className="mt-4 grid gap-2 sm:grid-cols-2">
                    {[
                      "VERIFY_WITH_CUSTOMER",
                      "STEP_UP_AUTH",
                      "BLOCK_CARD",
                      "CREATE_CASE",
                      "FILE_REPORT",
                      "ESCALATE_TO_ANALYST",
                    ].map((action) => (
                      <div
                        key={action}
                        className="rounded-md border border-white/6 bg-white/[0.02] px-3 py-2 text-xs text-zinc-600"
                      >
                        {action}
                      </div>
                    ))}
                  </div>
                </>
              ) : (
                <div className="space-y-4">
                  <ActionList
                    label="Initial recommendation"
                    actions={nba?.initial ?? []}
                  />
                  <ActionList
                    label="Final recommendation"
                    actions={nba?.final ?? []}
                    highlight
                  />

                  <div className="rounded-lg border border-white/8 bg-[#0b0e13] p-4">
                    <div className="text-xs uppercase tracking-wider text-zinc-500">
                      What changed
                    </div>
                    <p className="mt-2 text-sm leading-6 text-zinc-400">
                      {nba?.what_changed || "nothing"}
                    </p>
                  </div>
                </div>
              )}
            </Panel>
          </div>

          <div className="space-y-5">
            <Panel title="Agent Activity" subtitle="Investigation workflow">
              <div className="space-y-4">
                <TimelineItem
                  step="01"
                  title="Trigger"
                  description="Load investigation case"
                  state={result ? "ready" : running ? "ready" : "waiting"}
                />

                <TimelineItem
                  step="02"
                  title="Graph Investigation"
                  description="Gather transaction and relationship evidence"
                  state={!result ? "waiting" : pipelineSucceeded ? "ready" : "failed"}
                />

                <TimelineItem
                  step="03"
                  title="Policy Evaluation"
                  description="Apply G2 investigation rules"
                  state={!result ? "waiting" : pipelineSucceeded ? "ready" : "failed"}
                />

                <TimelineItem
                  step="04"
                  title="Agent Reasoning"
                  description="Assess evidence and uncertainty"
                  state={!result ? "waiting" : pipelineSucceeded ? "ready" : "failed"}
                />

                <TimelineItem
                  step="05"
                  title="Case Memory"
                  description="Write investigation result to graph"
                  state={
                    !result
                      ? "waiting"
                      : caseInfo?.written_to_graph
                      ? "ready"
                      : "failed"
                  }
                />
              </div>
            </Panel>

            <Panel
              title="Evidence Requests"
              subtitle="Additional information requested from the investigation loop"
            >
              {!result ? (
                <EmptyState text="No evidence requests yet." />
              ) : evidenceRequests.length === 0 ? (
                <EmptyState text="No evidence requests." />
              ) : (
                <div className="space-y-3">
                  {evidenceRequests.map((req, i) => (
                    <div
                      key={i}
                      className="rounded-lg border border-white/8 bg-[#0b0e13] p-3"
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-medium text-zinc-300">
                          {req.type ?? "unknown"}
                        </span>
                        <span className="rounded-full bg-zinc-800 px-2 py-0.5 text-[9px] text-zinc-500">
                          step {req.asked_after_step ?? "?"}
                        </span>
                      </div>
                      <p className="mt-2 text-[11px] leading-5 text-zinc-500">
                        {req.assumed_response ?? ""}
                      </p>
                    </div>
                  ))}
                </div>
              )}
            </Panel>

            <Panel title="Suspicious Activity Report" subtitle="Deterministic SAR gate">
              {!result ? (
                <div className="flex items-center justify-between rounded-lg border border-white/8 bg-[#0b0e13] p-4">
                  <div>
                    <div className="text-sm font-medium text-zinc-300">
                      SAR submission
                    </div>
                    <div className="mt-1 text-xs text-zinc-600">
                      Awaiting investigation verdict
                    </div>
                  </div>

                  <div className="rounded-full border border-zinc-700 bg-zinc-800/60 px-3 py-1 text-[10px] font-semibold text-zinc-500">
                    PENDING
                  </div>
                </div>
              ) : !sar?.file ? (
                <div className="flex items-center justify-between rounded-lg border border-white/8 bg-[#0b0e13] p-4">
                  <div>
                    <div className="text-sm font-medium text-zinc-300">
                      SAR not filed for this case
                    </div>
                    <div className="mt-1 text-xs text-zinc-600">
                      The policy engine determined a report is not required.
                    </div>
                  </div>

                  <div className="rounded-full border border-zinc-700 bg-zinc-800/60 px-3 py-1 text-[10px] font-semibold text-zinc-500">
                    NOT FILED
                  </div>
                </div>
              ) : (
                <div className="space-y-3 rounded-lg border border-red-500/20 bg-red-500/5 p-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-red-300">
                      SAR filed
                    </span>
                    <span className="rounded-full border border-red-500/30 bg-red-500/10 px-3 py-1 text-[10px] font-semibold text-red-300">
                      FILED
                    </span>
                  </div>

                  {sar.reason && (
                    <p className="text-xs leading-5 text-zinc-400">
                      <span className="text-zinc-500">Reason: </span>
                      {sar.reason}
                    </p>
                  )}

                  {sar.narrative && (
                    <p className="max-h-40 overflow-y-auto text-[11px] leading-5 text-zinc-500">
                      {sar.narrative}
                    </p>
                  )}

                  <div className="grid grid-cols-2 gap-3 pt-1 text-[11px]">
                    <div>
                      <div className="text-zinc-600 uppercase tracking-wider text-[9px]">
                        Total amount
                      </div>
                      <div className="mt-1 text-zinc-300">
                        {typeof sar.total_amount_usd === "number"
                          ? `$${sar.total_amount_usd}`
                          : "—"}
                      </div>
                    </div>
                    <div>
                      <div className="text-zinc-600 uppercase tracking-wider text-[9px]">
                        Activity dates
                      </div>
                      <div className="mt-1 text-zinc-300">
                        {sar.activity_dates && sar.activity_dates.length > 0
                          ? sar.activity_dates.join(" → ")
                          : "—"}
                      </div>
                    </div>
                  </div>

                  {sar.subjects && sar.subjects.length > 0 && (
                    <div className="pt-1">
                      <div className="text-zinc-600 uppercase tracking-wider text-[9px]">
                        Subjects
                      </div>
                      <div className="mt-1 flex flex-wrap gap-1.5">
                        {sar.subjects.map((s, i) => (
                          <span
                            key={i}
                            className="rounded bg-black/30 px-2 py-0.5 text-[10px] text-zinc-400"
                          >
                            {s}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </Panel>

            <Panel title="Tool Calls" subtitle="Graph tools invoked by the agent">
              {!result ? (
                <div className="rounded-lg border border-dashed border-white/8 p-5 text-center">
                  <div className="text-xs font-medium text-zinc-500">
                    No tool calls yet
                  </div>
                  <div className="mt-1 text-[11px] text-zinc-700">
                    Investigation activity will appear here.
                  </div>
                </div>
              ) : (
                <div className="rounded-lg border border-white/8 bg-[#0b0e13] p-5 text-center">
                  <div className="text-2xl font-semibold text-zinc-200">
                    {result.tool_calls ?? 0}
                  </div>
                  <div className="mt-1 text-[11px] text-zinc-600">
                    graph/tool call{result.tool_calls === 1 ? "" : "s"} made
                    during this investigation
                  </div>
                </div>
              )}
            </Panel>
          </div>
        </div>

        <footer className="mt-8 flex flex-col justify-between gap-2 border-t border-white/6 pt-4 text-[11px] text-zinc-600 sm:flex-row">
          <span>HHGOA Fraud Investigation • Agentic GraphRAG</span>
          <span>Policy-controlled investigation workflow</span>
        </footer>
      </div>
    </main>
  );
}

function Panel({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle: string;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-xl border border-white/8 bg-[#0b0e13] p-5 shadow-[0_10px_40px_rgba(0,0,0,0.18)]">
      <div className="mb-5">
        <h2 className="text-sm font-semibold text-zinc-200">{title}</h2>
        <p className="mt-1 text-[11px] text-zinc-600">{subtitle}</p>
      </div>

      {children}
    </section>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-white/8 bg-[#0b0e13] px-4 py-4">
      <div className="text-[10px] uppercase tracking-wider text-zinc-600">
        {label}
      </div>

      <div className="mt-2 truncate text-sm font-medium text-zinc-300">
        {value}
      </div>
    </div>
  );
}

function Info({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-white/6 bg-black/10 p-3">
      <div className="text-[10px] uppercase tracking-wider text-zinc-600">
        {label}
      </div>

      <div className="mt-2 truncate text-sm text-zinc-400">{value}</div>
    </div>
  );
}

function EvidenceRow({ label }: { label: string }) {
  return (
    <div className="flex items-center justify-between border-b border-white/5 pb-2 last:border-0 last:pb-0">
      <span className="text-xs text-zinc-500">{label}</span>

      <span className="rounded-full bg-zinc-900 px-2 py-0.5 text-[9px] text-zinc-700">
        PENDING
      </span>
    </div>
  );
}

function EvidenceClaim({ evidence }: { evidence: EvidenceEntry }) {
  const count = evidence.entity_ids?.length ?? 0;
  return (
    <div className="border-b border-white/5 pb-3 last:border-0 last:pb-0">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs text-zinc-400">
          {evidence.claim ?? "Evidence claim"}
        </span>
        <span className="shrink-0 rounded-full bg-zinc-900 px-2 py-0.5 text-[9px] text-zinc-500">
          {evidence.source ?? "graph"}
        </span>
      </div>
      <div className="mt-1 text-[10px] text-zinc-600">
        {evidence.ref ? `${evidence.ref} · ` : ""}
        {count} entity id{count === 1 ? "" : "s"}
      </div>
    </div>
  );
}

function ActionList({
  label,
  actions,
  highlight = false,
}: {
  label: string;
  actions: ActionEntry[];
  highlight?: boolean;
}) {
  return (
    <div
      className={`rounded-lg border p-4 ${
        highlight
          ? "border-red-500/20 bg-red-500/5"
          : "border-white/8 bg-[#0b0e13]"
      }`}
    >
      <div className="flex items-center justify-between">
        <span className="text-xs uppercase tracking-wider text-zinc-500">
          {label}
        </span>
        <span className="rounded-md bg-zinc-800 px-2 py-1 text-[10px] text-zinc-500">
          {actions.length} action{actions.length === 1 ? "" : "s"}
        </span>
      </div>

      {actions.length === 0 ? (
        <div className="mt-3 text-sm text-zinc-600">No action generated.</div>
      ) : (
        <div className="mt-3 space-y-2">
          {actions.map((a, i) => (
            <div key={i} className="rounded-md border border-white/6 bg-black/10 p-2.5">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium text-zinc-200">
                  {a.action}
                </span>
                {a.route && (
                  <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-[9px] uppercase text-zinc-400">
                    {a.route}
                  </span>
                )}
              </div>
              {a.reason && (
                <p className="mt-1 text-[11px] leading-5 text-zinc-500">
                  {a.reason}
                </p>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function TimelineItem({
  step,
  title,
  description,
  state,
}: {
  step: string;
  title: string;
  description: string;
  state: "ready" | "waiting" | "failed";
}) {
  const badgeClass =
    state === "ready"
      ? "bg-red-500/15 text-red-400"
      : state === "failed"
      ? "bg-amber-500/15 text-amber-400"
      : "bg-zinc-900 text-zinc-700";

  const titleClass =
    state === "ready"
      ? "text-zinc-300"
      : state === "failed"
      ? "text-amber-400"
      : "text-zinc-600";

  return (
    <div className="flex gap-3">
      <div
        className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-[10px] font-semibold ${badgeClass}`}
      >
        {step}
      </div>

      <div className="min-w-0">
        <div className={`text-xs font-medium ${titleClass}`}>{title}</div>

        <div className="mt-0.5 text-[11px] leading-5 text-zinc-700">
          {description}
        </div>
      </div>
    </div>
  );
}

function EmptyState({ text }: { text: string }) {
  return (
    <div className="rounded-lg border border-dashed border-white/8 p-6 text-center">
      <div className="text-xs text-zinc-600">{text}</div>
    </div>
  );
}
