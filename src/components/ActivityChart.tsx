"use client";

import { useEffect, useRef, useState } from "react";

// Animated dual-line area (tokens + runs) used on the Dashboard & Overview.
export default function ActivityChart({ title, days, tokens, runs }: { title: string; days: string[]; tokens: number[]; runs: number[] }) {
  const [on, setOn] = useState(false);
  const tokRef = useRef<SVGPathElement>(null); const runRef = useRef<SVGPathElement>(null);
  useEffect(() => { const t = setTimeout(() => setOn(true), 90); return () => clearTimeout(t); }, []);
  useEffect(() => { [tokRef.current, runRef.current].forEach((p) => { if (!p) return; const L = p.getTotalLength(); p.style.strokeDasharray = String(L); p.style.strokeDashoffset = on ? "0" : String(L); }); }, [on]);

  const W = 660, H = 150, pad = 6;
  const path = (data: number[]) => { const mx = Math.max(1, ...data); const pts = data.map((v, i) => [pad + i / Math.max(1, data.length - 1) * (W - 2 * pad), H - pad - (v / mx) * (H - 2 * pad)] as const); return pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(" "); };
  const tokLine = path(tokens), runLine = path(runs);
  const tokArea = `${tokLine} L${W - pad} ${H - pad} L${pad} ${H - pad} Z`;
  const has = tokens.some((v) => v > 0) || runs.some((v) => v > 0);

  return (
    <div style={{ border: "1px solid var(--border)", borderRadius: 13, background: "var(--surface)", overflow: "hidden" }}>
      <div className="row" style={{ alignItems: "center", justifyContent: "space-between", padding: "11px 15px", borderBottom: "1px solid var(--border)", background: "var(--panel)" }}>
        <div className="row" style={{ gap: 8, alignItems: "center" }}><span style={{ width: 7, height: 7, borderRadius: "50%", background: "#5b7cff" }} /><span style={{ fontWeight: 600, fontSize: 10.5, textTransform: "uppercase", letterSpacing: ".06em", color: "var(--muted)" }}>{title}</span></div>
        <div className="row" style={{ gap: 14, fontSize: 10.5, color: "var(--muted)" }}><span className="row" style={{ gap: 5, alignItems: "center" }}><span style={{ width: 14, height: 2, borderRadius: 2, background: "#5b7cff" }} />tokens</span><span className="row" style={{ gap: 5, alignItems: "center" }}><span style={{ width: 14, height: 2, borderRadius: 2, background: "#22b8cf" }} />agent runs</span></div>
      </div>
      <div style={{ padding: "15px 15px 10px" }}>
        {!has ? <div className="note" style={{ padding: "30px 0", textAlign: "center" }}>No activity in the last 14 days yet — build something in a Lab.</div> : <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} preserveAspectRatio="none">
          <path d={tokArea} fill="rgba(91,124,255,.13)" />
          <path ref={tokRef} d={tokLine} fill="none" stroke="#5b7cff" strokeWidth="2" vectorEffect="non-scaling-stroke" style={{ transition: "stroke-dashoffset 1.4s ease" }} />
          <path ref={runRef} d={runLine} fill="none" stroke="#22b8cf" strokeWidth="2" vectorEffect="non-scaling-stroke" style={{ transition: "stroke-dashoffset 1.4s ease" }} />
        </svg>}
        <div className="row" style={{ justifyContent: "space-between", marginTop: 4 }}><span className="note" style={{ fontSize: 9 }}>{days[0]?.slice(5)}</span><span className="note" style={{ fontSize: 9 }}>{days[days.length - 1]?.slice(5)}</span></div>
      </div>
    </div>
  );
}
