"use client";

import { estCostUsd, fmtUsd } from "@/lib/pricing";

type Run = {
  id: string; agentType: string | null; runtime: string | null; model: string; provider: string | null;
  iterations: number; toolCalls: { tool: string; count: number }[]; toolCallCount: number;
  totalTokens: number; latencyMs: number; outcome: string; ts: string | null;
};

const fmt = (n: number) => n.toLocaleString();
const OUTCOME_COLOR: Record<string, string> = { success: "#3b9e5f", max_iters: "#f59e0b", error: "#e5484d" };

export default function AgentAnalytics({ runs }: { runs: Run[] }) {
  const total = runs.length;
  const success = runs.filter((r) => r.outcome === "success").length;
  const successRate = total ? Math.round((success / total) * 100) : 0;
  const avgIters = total ? (runs.reduce((a, r) => a + r.iterations, 0) / total) : 0;
  const avgMs = total ? Math.round(runs.reduce((a, r) => a + r.latencyMs, 0) / total) : 0;
  const totalTokens = runs.reduce((a, r) => a + r.totalTokens, 0);
  const totalCost = runs.reduce((a, r) => a + estCostUsd(r.model, r.totalTokens), 0);

  const byOutcome = runs.reduce<Record<string, number>>((m, r) => { m[r.outcome] = (m[r.outcome] || 0) + 1; return m; }, {});
  const byModel = Object.entries(runs.reduce<Record<string, { runs: number; tokens: number }>>((m, r) => {
    const k = r.model; (m[k] ||= { runs: 0, tokens: 0 }); m[k].runs++; m[k].tokens += r.totalTokens; return m;
  }, {})).sort((a, b) => b[1].runs - a[1].runs).slice(0, 8);
  const byTool = Object.entries(runs.reduce<Record<string, number>>((m, r) => {
    for (const t of r.toolCalls) m[t.tool] = (m[t.tool] || 0) + (t.count || 0); return m;
  }, {})).sort((a, b) => b[1] - a[1]).slice(0, 10);
  const maxTool = Math.max(1, ...byTool.map(([, c]) => c));

  const when = (ts: string | null) => ts ? new Date(ts).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "—";

  if (total === 0) return (
    <div className="card"><div className="card-b" style={{ textAlign: "center", padding: 40 }}>
      <div style={{ fontSize: 30, opacity: 0.4 }}>◇</div>
      <p className="page-sub" style={{ margin: "10px 0 4px" }}>No agent runs yet.</p>
      <div className="note">Run an agent in the Agent Lab and its metrics show up here.</div>
    </div></div>
  );

  return (
    <>
      <div className="cv-summary" style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 16 }}>
        <div className="metric"><span className="v">{fmt(total)}</span><span className="k">Runs</span></div>
        <div className="metric"><span className="v">{successRate}%</span><span className="k">Success rate</span></div>
        <div className="metric"><span className="v">{avgIters.toFixed(1)}</span><span className="k">Avg iterations</span></div>
        <div className="metric"><span className="v">{(avgMs / 1000).toFixed(1)}s</span><span className="k">Avg latency</span></div>
        <div className="metric"><span className="v">{fmt(totalTokens)}</span><span className="k">Tokens</span></div>
        <div className="metric"><span className="v">{fmtUsd(totalCost)}</span><span className="k">Est. cost</span></div>
      </div>

      <div className="split" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
        <div className="card">
          <div className="card-h"><span className="t">Outcomes</span></div>
          <div className="card-b">
            {Object.entries(byOutcome).sort((a, b) => b[1] - a[1]).map(([o, c]) => (
              <div key={o} style={{ marginBottom: 8 }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12.5 }}><span style={{ color: OUTCOME_COLOR[o] || "var(--muted)" }}>{o}</span><span className="mono" style={{ color: "var(--muted)" }}>{c} · {Math.round((c / total) * 100)}%</span></div>
                <div style={{ height: 5, background: "var(--panel-2)", borderRadius: 3, overflow: "hidden", marginTop: 3 }}><div style={{ height: "100%", width: `${(c / total) * 100}%`, background: OUTCOME_COLOR[o] || "var(--accent)" }} /></div>
              </div>
            ))}
          </div>
        </div>
        <div className="card">
          <div className="card-h"><span className="t">Most-used tools</span></div>
          <div className="card-b">
            {byTool.length === 0 ? <div className="note">No tool calls recorded yet.</div> : byTool.map(([t, c]) => (
              <div key={t} style={{ marginBottom: 8 }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12.5 }}><span>{t}</span><span className="mono" style={{ color: "var(--muted)" }}>{c}</span></div>
                <div style={{ height: 5, background: "var(--panel-2)", borderRadius: 3, overflow: "hidden", marginTop: 3 }}><div style={{ height: "100%", width: `${(c / maxTool) * 100}%`, background: "var(--accent)" }} /></div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="card" style={{ marginBottom: 16 }}>
        <div className="card-h"><span className="t">By model</span></div>
        <div className="card-b" style={{ overflowX: "auto" }}>
          <table className="tbl"><thead><tr><th>Model</th><th style={{ textAlign: "right" }}>Runs</th><th style={{ textAlign: "right" }}>Tokens</th><th style={{ textAlign: "right" }}>Est. cost</th></tr></thead>
            <tbody>{byModel.map(([m, v]) => <tr key={m}><td style={{ fontSize: 12.5 }}>{m}</td><td className="mono" style={{ textAlign: "right" }}>{fmt(v.runs)}</td><td className="mono" style={{ textAlign: "right", color: "var(--muted)" }}>{fmt(v.tokens)}</td><td className="mono" style={{ textAlign: "right" }}>{fmtUsd(estCostUsd(m, v.tokens))}</td></tr>)}</tbody>
          </table>
        </div>
      </div>

      <div className="card">
        <div className="card-h"><span className="t">Recent runs</span><span className="mono r">latest {Math.min(runs.length, 20)}</span></div>
        <div className="card-b" style={{ overflowX: "auto" }}>
          <table className="tbl"><thead><tr><th>When</th><th>Runtime</th><th>Model</th><th style={{ textAlign: "right" }}>Steps</th><th style={{ textAlign: "right" }}>Tokens</th><th style={{ textAlign: "right" }}>Latency</th><th>Outcome</th></tr></thead>
            <tbody>{runs.slice(0, 20).map((r) => (
              <tr key={r.id}>
                <td className="mono" style={{ whiteSpace: "nowrap", fontSize: 11, color: "var(--muted)" }}>{when(r.ts)}</td>
                <td><span className="badge">{r.runtime || r.agentType}</span></td>
                <td style={{ fontSize: 12 }}>{r.model}</td>
                <td className="mono" style={{ textAlign: "right" }}>{r.iterations}</td>
                <td className="mono" style={{ textAlign: "right", color: "var(--muted)" }}>{fmt(r.totalTokens)}</td>
                <td className="mono" style={{ textAlign: "right", color: "var(--muted)" }}>{(r.latencyMs / 1000).toFixed(1)}s</td>
                <td><span style={{ fontSize: 11.5, color: OUTCOME_COLOR[r.outcome] || "var(--muted)" }}>{r.outcome}</span></td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </div>
    </>
  );
}
