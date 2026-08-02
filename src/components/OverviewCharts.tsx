"use client";

import { useEffect, useRef, useState } from "react";

type Donut = { label: string; count: number; color: string };
const PAL = ["#5b7cff", "#22b8cf", "#a855f7", "#3ecf7f", "#f59e0b", "#f0616d", "#ec4899", "#84cc16"];
const kfmt = (n: number) => n >= 1000 ? `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k` : String(n);

const pnl: React.CSSProperties = { border: "1px solid var(--border)", borderRadius: 13, background: "var(--surface)", overflow: "hidden", display: "flex", flexDirection: "column" };
const head = (dot: string, title: string, right?: React.ReactNode) => <div className="row" style={{ alignItems: "center", justifyContent: "space-between", padding: "11px 15px", borderBottom: "1px solid var(--border)", background: "var(--panel)" }}><div className="row" style={{ gap: 8, alignItems: "center" }}><span style={{ width: 7, height: 7, borderRadius: "50%", background: dot }} /><span style={{ fontWeight: 600, fontSize: 10.5, textTransform: "uppercase", letterSpacing: ".06em", color: "var(--muted)" }}>{title}</span></div>{right}</div>;

function DonutCard({ dot, title, segs }: { dot: string; title: string; segs: Donut[] }) {
  const [on, setOn] = useState(false);
  useEffect(() => { const t = setTimeout(() => setOn(true), 90); return () => clearTimeout(t); }, []);
  const tot = Math.max(1, segs.reduce((a, s) => a + s.count, 0)); let acc = 0;
  const arcs = segs.map((s) => { const frac = s.count / tot; const a = { ...s, frac, off: acc }; acc += frac; return a; });
  return (
    <div style={pnl}>{head(dot, title)}
      <div style={{ padding: 15, flex: 1, display: "flex", alignItems: "center", gap: 18 }}>
        {segs.every((s) => s.count === 0) ? <div className="note">No data yet.</div> : <>
          <svg width="118" height="118" viewBox="0 0 42 42" style={{ flex: "0 0 auto" }}>
            <circle cx="21" cy="21" r="15.9" fill="none" stroke="var(--panel-2)" strokeWidth="6" />
            {arcs.map((s, i) => <circle key={s.label} cx="21" cy="21" r="15.9" fill="none" stroke={s.color} strokeWidth="6" strokeDasharray={on ? `${s.frac * 100} ${100 - s.frac * 100}` : "0 100"} strokeDashoffset={-s.off * 100} transform="rotate(-90 21 21)" style={{ transition: "stroke-dasharray .8s ease", transitionDelay: `${i * 0.1}s` }} />)}
          </svg>
          <div style={{ display: "flex", flexDirection: "column", gap: 8, flex: 1, fontSize: 11.5 }}>
            {arcs.map((s) => <div key={s.label} className="row" style={{ gap: 7, alignItems: "center", color: "var(--muted)" }}><span style={{ width: 8, height: 8, borderRadius: "50%", background: s.color }} />{s.label}<b className="mono" style={{ marginLeft: "auto", color: "var(--text)" }}>{kfmt(s.count)}</b></div>)}
          </div>
        </>}
      </div>
    </div>
  );
}

export default function OverviewCharts({ days, tokens, runs, usersByStatus, tokensByLab }: {
  days: string[]; tokens: number[]; runs: number[];
  usersByStatus: { status: string; count: number }[];
  tokensByLab: { lab: string; tokens: number }[];
}) {
  const [on, setOn] = useState(false);
  const tokRef = useRef<SVGPathElement>(null); const runRef = useRef<SVGPathElement>(null);
  useEffect(() => { const t = setTimeout(() => setOn(true), 90); return () => clearTimeout(t); }, []);
  useEffect(() => { [tokRef.current, runRef.current].forEach((p) => { if (!p) return; const L = p.getTotalLength(); p.style.strokeDasharray = String(L); p.style.strokeDashoffset = on ? "0" : String(L); }); }, [on]);

  const W = 660, H = 150, pad = 6;
  const path = (data: number[]) => { const mx = Math.max(1, ...data); const pts = data.map((v, i) => [pad + i / Math.max(1, data.length - 1) * (W - 2 * pad), H - pad - (v / mx) * (H - 2 * pad)] as const); return { line: pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(" "), pts }; };
  const tok = path(tokens); const run = path(runs);
  const tokArea = `${tok.line} L${W - pad} ${H - pad} L${pad} ${H - pad} Z`;
  const hasSeries = tokens.some((v) => v > 0) || runs.some((v) => v > 0);

  const STATUS_COLOR: Record<string, string> = { active: "#3ecf7f", pending: "#f59e0b", suspended: "#f0616d" };
  const uSegs: Donut[] = usersByStatus.map((u) => ({ label: u.status, count: u.count, color: STATUS_COLOR[u.status] || "#6a7280" }));
  const lSegs: Donut[] = tokensByLab.slice(0, 8).map((l, i) => ({ label: l.lab, count: l.tokens, color: PAL[i % PAL.length] }));

  return (
    <>
      <div style={{ ...pnl, marginBottom: 16 }}>{head("#5b7cff", "Platform activity · last 14 days", <div className="row" style={{ gap: 14, fontSize: 10.5, color: "var(--muted)" }}><span className="row" style={{ gap: 5, alignItems: "center" }}><span style={{ width: 14, height: 2, borderRadius: 2, background: "#5b7cff" }} />tokens</span><span className="row" style={{ gap: 5, alignItems: "center" }}><span style={{ width: 14, height: 2, borderRadius: 2, background: "#22b8cf" }} />agent runs</span></div>)}
        <div style={{ padding: "15px 15px 10px" }}>
          {!hasSeries ? <div className="note" style={{ padding: "30px 0", textAlign: "center" }}>No activity in the last 14 days yet.</div> : <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} preserveAspectRatio="none">
            <path d={tokArea} fill="rgba(91,124,255,.13)" />
            <path ref={tokRef} d={tok.line} fill="none" stroke="#5b7cff" strokeWidth="2" vectorEffect="non-scaling-stroke" style={{ transition: "stroke-dashoffset 1.4s ease" }} />
            <path ref={runRef} d={run.line} fill="none" stroke="#22b8cf" strokeWidth="2" vectorEffect="non-scaling-stroke" style={{ transition: "stroke-dashoffset 1.4s ease" }} />
          </svg>}
          <div className="row" style={{ justifyContent: "space-between", marginTop: 4 }}><span className="note" style={{ fontSize: 9 }}>{days[0]?.slice(5)}</span><span className="note" style={{ fontSize: 9 }}>{days[days.length - 1]?.slice(5)}</span></div>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 18 }}>
        <DonutCard dot="#a855f7" title="Users by status" segs={uSegs} />
        <DonutCard dot="#22b8cf" title="Tokens by lab · this month" segs={lSegs} />
      </div>
    </>
  );
}
