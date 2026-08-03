"use client";

import { useEffect, useRef, useState } from "react";
import { estCostUsd, fmtUsd } from "@/lib/pricing";

type Run = {
  id: string; agentType: string | null; runtime: string | null; model: string; provider: string | null;
  iterations: number; toolCalls: { tool: string; count: number }[]; toolCallCount: number;
  totalTokens: number; latencyMs: number; outcome: string; ts: string | null;
};

const fmt = (n: number) => n.toLocaleString();
const OUTCOME_COLOR: Record<string, string> = { success: "#3ecf7f", max_iters: "#f59e0b", error: "#f0616d", stopped: "#6a7280" };
const TOOL_COLORS = ["#5b7cff", "#22b8cf", "#a855f7", "#3ecf7f", "#f59e0b", "#f0616d"];

const pnl: React.CSSProperties = { border: "1px solid var(--border)", borderRadius: 13, background: "var(--surface)", overflow: "hidden", display: "flex", flexDirection: "column" };
const head = (dot: string, title: string, right?: React.ReactNode) => <div className="row" style={{ alignItems: "center", justifyContent: "space-between", padding: "11px 14px", borderBottom: "1px solid var(--border)", background: "var(--panel)" }}><div className="row" style={{ gap: 8, alignItems: "center" }}><span style={{ width: 7, height: 7, borderRadius: "50%", background: dot }} /><span style={{ fontWeight: 600, fontSize: 10.5, textTransform: "uppercase", letterSpacing: ".06em", color: "var(--muted)" }}>{title}</span></div>{right}</div>;

function Spark({ vals, color }: { vals: number[]; color: string }) {
  const mx = Math.max(1, ...vals);
  return <div style={{ display: "flex", alignItems: "flex-end", gap: 2, height: 20, marginTop: 8 }}>{vals.map((v, i) => <span key={i} style={{ flex: 1, height: Math.max(2, (v / mx) * 20), background: color, borderRadius: 1, opacity: 0.55, transformOrigin: "bottom", animation: "grow .5s ease both", animationDelay: `${i * 0.03}s` }} />)}</div>;
}

export default function AgentAnalytics({ runs }: { runs: Run[] }) {
  const [on, setOn] = useState(false);
  const lineRef = useRef<SVGPathElement>(null);
  useEffect(() => { const t = setTimeout(() => setOn(true), 90); return () => clearTimeout(t); }, []);
  useEffect(() => { const p = lineRef.current; if (!p) return; const L = p.getTotalLength(); p.style.strokeDasharray = String(L); p.style.strokeDashoffset = on ? "0" : String(L); }, [on]);

  const total = runs.length;
  const success = runs.filter((r) => r.outcome === "success").length;
  const successRate = total ? Math.round((success / total) * 100) : 0;
  const avgIters = total ? (runs.reduce((a, r) => a + r.iterations, 0) / total) : 0;
  const avgMs = total ? Math.round(runs.reduce((a, r) => a + r.latencyMs, 0) / total) : 0;
  const totalTokens = runs.reduce((a, r) => a + r.totalTokens, 0);
  const totalCost = runs.reduce((a, r) => a + estCostUsd(r.model, r.totalTokens), 0);

  const byOutcome = Object.entries(runs.reduce<Record<string, number>>((m, r) => { m[r.outcome] = (m[r.outcome] || 0) + 1; return m; }, {})).sort((a, b) => b[1] - a[1]);
  const byModel = Object.entries(runs.reduce<Record<string, { runs: number; tokens: number }>>((m, r) => { const k = r.model; (m[k] ||= { runs: 0, tokens: 0 }); m[k].runs++; m[k].tokens += r.totalTokens; return m; }, {})).sort((a, b) => b[1].runs - a[1].runs).slice(0, 8);
  const byTool = Object.entries(runs.reduce<Record<string, number>>((m, r) => { for (const t of r.toolCalls || []) m[t.tool] = (m[t.tool] || 0) + (t.count || 0); return m; }, {})).sort((a, b) => b[1] - a[1]).slice(0, 6);
  const maxTool = Math.max(1, ...byTool.map(([, c]) => c));

  // last-14-day series (for sparklines + trend)
  const days: string[] = []; for (let i = 13; i >= 0; i--) days.push(new Date(Date.now() - i * 86400000).toISOString().slice(0, 10));
  const dm: Record<string, { runs: number; ok: number; tokens: number; lat: number; iters: number; cost: number }> = {};
  runs.forEach((r) => { const k = r.ts ? new Date(r.ts).toISOString().slice(0, 10) : null; if (!k) return; (dm[k] ||= { runs: 0, ok: 0, tokens: 0, lat: 0, iters: 0, cost: 0 }); const d = dm[k]; d.runs++; if (r.outcome === "success") d.ok++; d.tokens += r.totalTokens; d.lat += r.latencyMs; d.iters += r.iterations; d.cost += estCostUsd(r.model, r.totalTokens); });
  const ser = days.map((k) => dm[k] || { runs: 0, ok: 0, tokens: 0, lat: 0, iters: 0, cost: 0 });
  const sp = { runs: ser.map((d) => d.runs), sr: ser.map((d) => (d.runs ? d.ok / d.runs * 100 : 0)), iters: ser.map((d) => (d.runs ? d.iters / d.runs : 0)), lat: ser.map((d) => (d.runs ? d.lat / d.runs : 0)), tokens: ser.map((d) => d.tokens), cost: ser.map((d) => d.cost) };

  const when = (ts: string | null) => ts ? new Date(ts).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "—";

  if (total === 0) return (
    <div style={{ ...pnl, textAlign: "center", padding: 44 }}>
      <div style={{ fontSize: 30, opacity: 0.4 }}>◇</div>
      <p className="page-sub" style={{ margin: "10px 0 4px" }}>No agent runs yet.</p>
      <div className="note">Run an agent in the Agent Lab and its metrics show up here.</div>
    </div>
  );

  const kpis: [React.ReactNode, string, number[], string][] = [
    [fmt(total), "runs", sp.runs, "#5b7cff"],
    [<span key="s" style={{ color: "var(--good)" }}>{successRate}%</span>, "success rate", sp.sr, "#3ecf7f"],
    [avgIters.toFixed(1), "avg iterations", sp.iters, "#a855f7"],
    [`${(avgMs / 1000).toFixed(1)}s`, "avg latency", sp.lat, "#22b8cf"],
    [fmt(totalTokens), "tokens", sp.tokens, "#5b7cff"],
    [<span key="c" style={{ color: "var(--accent-strong)" }}>{fmtUsd(totalCost)}</span>, "est. cost", sp.cost, "#f59e0b"],
  ];

  // outcomes donut geometry
  const outTot = byOutcome.reduce((a, [, c]) => a + c, 0) || 1; let acc = 0;
  const donut = byOutcome.map(([o, c]) => { const frac = c / outTot; const seg = { o, c, frac, off: acc, color: OUTCOME_COLOR[o] || "#6a7280" }; acc += frac; return seg; });

  // trend area path
  const W = 320, H = 132, pad = 6, mxRuns = Math.max(1, ...sp.runs);
  const pts = sp.runs.map((v, i) => [pad + i / (sp.runs.length - 1) * (W - 2 * pad), H - pad - (v / mxRuns) * (H - 2 * pad)] as const);
  const linePath = pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(" ");
  const areaPath = `${linePath} L${W - pad} ${H - pad} L${pad} ${H - pad} Z`;
  const C = 2 * Math.PI * 60;

  return (
    <>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(6, minmax(0,1fr))", gap: 12, marginBottom: 16 }}>
        {kpis.map(([v, k, s, col], i) => <div key={i} style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 12, padding: "13px 15px" }}>
          <div style={{ fontFamily: "var(--mono)", fontSize: 22, fontWeight: 600, letterSpacing: "-.02em" }}>{v}</div>
          <div style={{ fontSize: 9, textTransform: "uppercase", letterSpacing: ".05em", color: "var(--faint)", marginTop: 3 }}>{k}</div>
          <Spark vals={s} color={col} />
        </div>)}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 16 }}>
        <div style={pnl}>{head("#3ecf7f", "Success rate")}
          <div style={{ padding: 15, flex: 1, display: "grid", placeItems: "center" }}>
            <svg width="152" height="152" viewBox="0 0 150 150">
              <circle cx="75" cy="75" r="60" fill="none" stroke="var(--panel-2)" strokeWidth="13" />
              <circle cx="75" cy="75" r="60" fill="none" stroke="#3ecf7f" strokeWidth="13" strokeLinecap="round" strokeDasharray={C} strokeDashoffset={on ? C * (1 - successRate / 100) : C} transform="rotate(-90 75 75)" style={{ transition: "stroke-dashoffset 1s ease" }} />
              <text x="75" y="72" textAnchor="middle" fontSize="30" fontFamily="var(--mono)" fontWeight="700" fill="var(--text)">{successRate}%</text>
              <text x="75" y="92" textAnchor="middle" fontSize="10" fill="var(--faint)">{fmt(success)} / {fmt(total)} ok</text>
            </svg>
          </div>
        </div>

        <div style={pnl}>{head("#a855f7", "Outcomes")}
          <div style={{ padding: 15, flex: 1, display: "flex", alignItems: "center", gap: 18 }}>
            <svg width="118" height="118" viewBox="0 0 42 42" style={{ flex: "0 0 auto" }}>
              <circle cx="21" cy="21" r="15.9" fill="none" stroke="var(--panel-2)" strokeWidth="6" />
              {donut.map((s) => <circle key={s.o} cx="21" cy="21" r="15.9" fill="none" stroke={s.color} strokeWidth="6" strokeDasharray={on ? `${s.frac * 100} ${100 - s.frac * 100}` : "0 100"} strokeDashoffset={-s.off * 100} transform="rotate(-90 21 21)" style={{ transition: "stroke-dasharray .8s ease", transitionDelay: `${donut.indexOf(s) * 0.1}s` }} />)}
            </svg>
            <div style={{ display: "flex", flexDirection: "column", gap: 8, flex: 1, fontSize: 11.5 }}>
              {donut.map((s) => <div key={s.o} className="row" style={{ gap: 7, alignItems: "center", color: "var(--muted)" }}><span style={{ width: 8, height: 8, borderRadius: "50%", background: s.color }} />{s.o}<b className="mono" style={{ marginLeft: "auto", color: "var(--text)" }}>{s.c}</b></div>)}
            </div>
          </div>
        </div>

        <div style={pnl}>{head("#5b7cff", "Most-used tools")}
          <div style={{ padding: 15, flex: 1, display: "flex", flexDirection: "column", justifyContent: "center" }}>
            {byTool.length === 0 ? <div className="note">No tool calls recorded yet.</div> : <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>{byTool.map(([t, c], i) => <div key={t} className="row" style={{ gap: 10, alignItems: "center", fontSize: 11.5 }}>
              <span style={{ flex: "0 0 92px", color: "var(--muted)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{t}</span>
              <div style={{ flex: 1, height: 15, background: "var(--panel-2)", borderRadius: 5, overflow: "hidden" }}><div style={{ height: "100%", width: `${(c / maxTool) * 100}%`, background: TOOL_COLORS[i % TOOL_COLORS.length], borderRadius: 5, transformOrigin: "left", transform: on ? "scaleX(1)" : "scaleX(0)", transition: "transform .8s cubic-bezier(.2,.8,.2,1)", transitionDelay: `${i * 0.07}s` }} /></div>
              <span className="mono" style={{ flex: "0 0 30px", textAlign: "right", color: "var(--faint)" }}>{c}</span>
            </div>)}</div>}
          </div>
        </div>

        <div style={pnl}>{head("#22b8cf", "Runs over time", <span className="note" style={{ fontSize: 10 }}>14 days</span>)}
          <div style={{ padding: "15px 15px 10px", flex: 1, display: "flex", alignItems: "flex-end" }}>
            <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} preserveAspectRatio="none">
              <path d={areaPath} fill="rgba(34,184,207,.14)" />
              <path ref={lineRef} d={linePath} fill="none" stroke="#22b8cf" strokeWidth="2" vectorEffect="non-scaling-stroke" style={{ transition: "stroke-dashoffset 1.3s ease" }} />
            </svg>
          </div>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 16, alignItems: "start" }}>
        <div style={pnl}>{head("var(--muted)", "By model")}
          <div style={{ overflowX: "auto" }}><table className="tbl"><thead><tr><th style={{ paddingLeft: 14 }}>Model</th><th style={{ textAlign: "right" }}>Runs</th><th style={{ textAlign: "right" }}>Tokens</th><th style={{ textAlign: "right", paddingRight: 14 }}>Est. cost</th></tr></thead>
            <tbody>{byModel.map(([m, v]) => <tr key={m}><td style={{ fontSize: 12.5, paddingLeft: 14 }}>{m}</td><td className="mono" style={{ textAlign: "right" }}>{fmt(v.runs)}</td><td className="mono" style={{ textAlign: "right", color: "var(--muted)" }}>{fmt(v.tokens)}</td><td className="mono" style={{ textAlign: "right", paddingRight: 14 }}>{fmtUsd(estCostUsd(m, v.tokens))}</td></tr>)}</tbody>
          </table></div>
        </div>
        <div style={pnl}>{head("#3ecf7f", "Outcome mix")}
          <div style={{ padding: 15, display: "flex", flexDirection: "column", gap: 11, justifyContent: "center", flex: 1 }}>
            {byOutcome.map(([o, c]) => <div key={o}><div className="row" style={{ justifyContent: "space-between", fontSize: 12 }}><span style={{ color: OUTCOME_COLOR[o] || "var(--muted)" }}>{o}</span><span className="mono" style={{ color: "var(--muted)" }}>{c} · {Math.round((c / total) * 100)}%</span></div>
              <div style={{ height: 6, background: "var(--panel-2)", borderRadius: 3, overflow: "hidden", marginTop: 4 }}><div style={{ height: "100%", width: `${(c / total) * 100}%`, background: OUTCOME_COLOR[o] || "var(--accent)", borderRadius: 3, transformOrigin: "left", transform: on ? "scaleX(1)" : "scaleX(0)", transition: "transform .7s ease" }} /></div>
            </div>)}
          </div>
        </div>
      </div>

      <div style={pnl}>{head("var(--muted)", "Recent runs", <span className="note" style={{ fontSize: 10 }}>latest {Math.min(runs.length, 20)}</span>)}
        <div style={{ overflowX: "auto" }}><table className="tbl"><thead><tr><th style={{ paddingLeft: 14 }}>When</th><th>Runtime</th><th>Model</th><th style={{ textAlign: "right" }}>Steps</th><th style={{ textAlign: "right" }}>Tokens</th><th style={{ textAlign: "right" }}>Latency</th><th style={{ paddingRight: 14 }}>Outcome</th></tr></thead>
          <tbody>{runs.slice(0, 20).map((r) => <tr key={r.id}>
            <td className="mono" style={{ whiteSpace: "nowrap", fontSize: 11, color: "var(--muted)", paddingLeft: 14 }}>{when(r.ts)}</td>
            <td><span className="badge">{r.runtime || r.agentType}</span></td>
            <td style={{ fontSize: 12 }}>{r.model}</td>
            <td className="mono" style={{ textAlign: "right" }}>{r.iterations}</td>
            <td className="mono" style={{ textAlign: "right", color: "var(--muted)" }}>{fmt(r.totalTokens)}</td>
            <td className="mono" style={{ textAlign: "right", color: "var(--muted)" }}>{(r.latencyMs / 1000).toFixed(1)}s</td>
            <td style={{ paddingRight: 14 }}><span style={{ fontSize: 9.5, fontWeight: 600, padding: "2px 8px", borderRadius: 20, color: OUTCOME_COLOR[r.outcome] || "var(--muted)", background: `color-mix(in srgb, ${OUTCOME_COLOR[r.outcome] || "#6a7280"} 14%, transparent)` }}>{r.outcome}</span></td>
          </tr>)}</tbody>
        </table></div>
      </div>
    </>
  );
}
