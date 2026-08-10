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
    <div style={{ overflow: "hidden" }}>
      <div className="row" style={{
        alignItems: "center", justifyContent: "space-between",
        padding: "13px 18px",
        borderBottom: "1px solid rgba(108,71,255,0.18)",
        background: "rgba(255,255,255,0.03)"
      }}>
        <div className="row" style={{ gap: 9, alignItems: "center" }}>
          <span style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--accent)", boxShadow: "0 0 8px rgba(108,71,255,0.6)", flexShrink: 0 }} />
          <span style={{ fontWeight: 700, fontSize: 10.5, textTransform: "uppercase", letterSpacing: ".09em", color: "var(--muted)" }}>{title}</span>
        </div>
        <div className="row" style={{ gap: 16, fontSize: 10.5, color: "var(--muted)" }}>
          <span className="row" style={{ gap: 6, alignItems: "center" }}>
            <span style={{ width: 16, height: 2.5, borderRadius: 2, background: "#6c47ff" }} />tokens
          </span>
          <span className="row" style={{ gap: 6, alignItems: "center" }}>
            <span style={{ width: 16, height: 2.5, borderRadius: 2, background: "#38bdf8" }} />agent runs
          </span>
        </div>
      </div>
      <div style={{ padding: "16px 18px 12px" }}>
        {!has
          ? <div className="note" style={{ padding: "32px 0", textAlign: "center", color: "var(--faint)" }}>No activity in the last 14 days yet — build something in a Lab.</div>
          : <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} preserveAspectRatio="none">
              <defs>
                <linearGradient id="tokGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="rgba(108,71,255,0.30)" />
                  <stop offset="100%" stopColor="rgba(108,71,255,0.00)" />
                </linearGradient>
              </defs>
              <path d={tokArea} fill="url(#tokGrad)" />
              <path ref={tokRef} d={tokLine} fill="none" stroke="#6c47ff" strokeWidth="2.5" vectorEffect="non-scaling-stroke" style={{ transition: "stroke-dashoffset 1.4s ease", filter: "drop-shadow(0 0 4px rgba(108,71,255,0.55))" }} />
              <path ref={runRef} d={runLine} fill="none" stroke="#38bdf8" strokeWidth="2" vectorEffect="non-scaling-stroke" style={{ transition: "stroke-dashoffset 1.4s ease", filter: "drop-shadow(0 0 4px rgba(56,189,248,0.50))" }} />
            </svg>
        }
        <div className="row" style={{ justifyContent: "space-between", marginTop: 5 }}>
          <span className="note" style={{ fontSize: 9, color: "var(--faint)" }}>{days[0]?.slice(5)}</span>
          <span className="note" style={{ fontSize: 9, color: "var(--faint)" }}>{days[days.length - 1]?.slice(5)}</span>
        </div>
      </div>
    </div>
  );
}
