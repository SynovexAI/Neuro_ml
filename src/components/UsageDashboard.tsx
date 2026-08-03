"use client";

import { useEffect, useRef, useState } from "react";

type ByUser = { email: string; name: string | null; role: string; limit: number | null; tokens: number; calls: number };
type Data = {
  total: { tokens: number; calls: number; estimated: number };
  byUser: ByUser[];
  byLab: { lab: string; tokens: number }[];
  byModel: { model: string; tokens: number; calls: number }[];
  byDay: { day: string; tokens: number }[];
  audit: { id: string; ts: string | null; event: string; userId: string | null; detail: Record<string, unknown> | null }[];
  defaultLimit: number;
};

const fmt = (n: number) => n.toLocaleString();
const kfmt = (n: number) => n >= 1000 ? `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k` : String(n);
const pct = (a: number, b: number) => (b > 0 ? Math.round((a / b) * 100) : 0);
const PAL = ["#5b7cff", "#22b8cf", "#a855f7", "#3ecf7f", "#f59e0b", "#f0616d", "#ec4899", "#84cc16"];

const pnl: React.CSSProperties = { border: "1px solid var(--border)", borderRadius: 13, background: "var(--surface)", overflow: "hidden", display: "flex", flexDirection: "column" };
const head = (dot: string, title: string, right?: React.ReactNode) => <div className="row" style={{ alignItems: "center", justifyContent: "space-between", padding: "11px 14px", borderBottom: "1px solid var(--border)", background: "var(--panel)" }}><div className="row" style={{ gap: 8, alignItems: "center" }}><span style={{ width: 7, height: 7, borderRadius: "50%", background: dot }} /><span style={{ fontWeight: 600, fontSize: 10.5, textTransform: "uppercase", letterSpacing: ".06em", color: "var(--muted)" }}>{title}</span></div>{right}</div>;

function eventBadge(ev: string): { color: string; label: string } {
  if (ev === "login_failed") return { color: "#f59e0b", label: "login failed" };
  if (ev === "quota_exceeded") return { color: "#f0616d", label: "quota hit" };
  if (ev === "login") return { color: "#3ecf7f", label: "login" };
  return { color: "#6a7280", label: ev };
}

function Health() {
  const [s, setS] = useState<{ ok: boolean; db: string; ms?: number } | null>(null);
  const [err, setErr] = useState(false);
  useEffect(() => { fetch("/api/health").then((r) => r.json()).then(setS).catch(() => setErr(true)); }, []);
  const up = s?.ok && !err; const color = err ? "#f59e0b" : up ? "#3ecf7f" : "#f59e0b";
  return <span style={{ fontSize: 10.5, fontWeight: 600, padding: "5px 11px", borderRadius: 20, color, background: `color-mix(in srgb, ${color} 14%, transparent)`, display: "inline-flex", alignItems: "center", gap: 6 }}><span style={{ width: 7, height: 7, borderRadius: "50%", background: color }} />{err ? "health: unreachable" : s ? (up ? `healthy · db ${s.ms}ms` : "degraded") : "checking…"}</span>;
}

export default function UsageDashboard({ data }: { data: Data }) {
  const { total, byUser, byLab, byModel, byDay, audit } = data;
  const [on, setOn] = useState(false);
  const lineRef = useRef<SVGPathElement>(null);
  useEffect(() => { const t = setTimeout(() => setOn(true), 90); return () => clearTimeout(t); }, []);
  useEffect(() => { const p = lineRef.current; if (!p) return; const L = p.getTotalLength(); p.style.strokeDasharray = String(L); p.style.strokeDashoffset = on ? "0" : String(L); }, [on]);

  const meteredShare = 100 - pct(total.estimated, total.tokens);
  const maxModel = Math.max(1, ...byModel.map((m) => m.tokens));

  // daily area geometry
  const W = 320, H = 140, pad = 6;
  const dv = byDay.map((d) => d.tokens); const mxDay = Math.max(1, ...dv);
  const pts = (dv.length > 1 ? dv : [0, ...dv]).map((v, i, a) => [pad + i / Math.max(1, a.length - 1) * (W - 2 * pad), H - pad - (v / mxDay) * (H - 2 * pad)] as const);
  const linePath = pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(" ");
  const areaPath = `${linePath} L${W - pad} ${H - pad} L${pad} ${H - pad} Z`;

  // by-lab donut geometry
  const labTot = Math.max(1, byLab.reduce((a, l) => a + l.tokens, 0)); let acc = 0;
  const donut = byLab.slice(0, 8).map((l, i) => { const frac = l.tokens / labTot; const seg = { ...l, frac, off: acc, color: PAL[i % PAL.length] }; acc += frac; return seg; });

  const C = 2 * Math.PI * 60;

  return (
    <>
      <div className="row" style={{ justifyContent: "flex-end", marginBottom: 12 }}><Health /></div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, minmax(0,1fr))", gap: 12, marginBottom: 16 }}>
        {([[fmt(total.tokens), "tokens this month", undefined], [fmt(total.calls), "LLM calls", undefined], [String(byUser.length), "active users", undefined], [`${meteredShare}%`, "metered (vs estimated)", "var(--good)"]] as [string, string, string | undefined][]).map(([v, k, col]) => <div key={k} style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 12, padding: "14px 16px" }}>
          <div style={{ fontFamily: "var(--mono)", fontSize: 24, fontWeight: 600, letterSpacing: "-.02em", color: col }}>{v}</div>
          <div style={{ fontSize: 9.5, textTransform: "uppercase", letterSpacing: ".05em", color: "var(--faint)", marginTop: 4 }}>{k}</div>
        </div>)}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 16 }}>
        <div style={pnl}>{head("#5b7cff", "Daily token usage", <span className="note" style={{ fontSize: 10 }}>14 days</span>)}
          <div style={{ padding: "15px 15px 10px", flex: 1, display: "flex", alignItems: "flex-end" }}>
            {byDay.length === 0 ? <div className="note">No usage recorded yet.</div> : <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} preserveAspectRatio="none"><path d={areaPath} fill="rgba(91,124,255,.14)" /><path ref={lineRef} d={linePath} fill="none" stroke="#5b7cff" strokeWidth="2" vectorEffect="non-scaling-stroke" style={{ transition: "stroke-dashoffset 1.3s ease" }} /></svg>}
          </div>
        </div>

        <div style={pnl}>{head("#a855f7", "Usage by lab")}
          <div style={{ padding: 15, flex: 1, display: "flex", alignItems: "center", gap: 18 }}>
            {byLab.length === 0 ? <div className="note">No usage yet.</div> : <>
              <svg width="120" height="120" viewBox="0 0 42 42" style={{ flex: "0 0 auto" }}>
                <circle cx="21" cy="21" r="15.9" fill="none" stroke="var(--panel-2)" strokeWidth="6" />
                {donut.map((s, i) => <circle key={s.lab} cx="21" cy="21" r="15.9" fill="none" stroke={s.color} strokeWidth="6" strokeDasharray={on ? `${s.frac * 100} ${100 - s.frac * 100}` : "0 100"} strokeDashoffset={-s.off * 100} transform="rotate(-90 21 21)" style={{ transition: "stroke-dasharray .8s ease", transitionDelay: `${i * 0.1}s` }} />)}
              </svg>
              <div style={{ display: "flex", flexDirection: "column", gap: 8, flex: 1, fontSize: 11.5 }}>
                {donut.map((s) => <div key={s.lab} className="row" style={{ gap: 7, alignItems: "center", color: "var(--muted)" }}><span style={{ width: 8, height: 8, borderRadius: "50%", background: s.color }} />{s.lab}<b className="mono" style={{ marginLeft: "auto", color: "var(--text)" }}>{kfmt(s.tokens)}</b></div>)}
              </div>
            </>}
          </div>
        </div>

        <div style={pnl}>{head("#22b8cf", "Tokens by model")}
          <div style={{ padding: 15, flex: 1, display: "flex", flexDirection: "column", justifyContent: "center" }}>
            {byModel.length === 0 ? <div className="note">No usage yet.</div> : <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>{byModel.slice(0, 6).map((m, i) => <div key={m.model} className="row" style={{ gap: 10, alignItems: "center", fontSize: 11.5 }}>
              <span style={{ flex: "0 0 130px", color: "var(--muted)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{m.model}</span>
              <div style={{ flex: 1, height: 14, background: "var(--panel-2)", borderRadius: 5, overflow: "hidden" }}><div style={{ height: "100%", width: `${(m.tokens / maxModel) * 100}%`, background: PAL[i % PAL.length], borderRadius: 5, transformOrigin: "left", transform: on ? "scaleX(1)" : "scaleX(0)", transition: "transform .8s cubic-bezier(.2,.8,.2,1)", transitionDelay: `${i * 0.07}s` }} /></div>
              <span className="mono" style={{ flex: "0 0 46px", textAlign: "right", color: "var(--faint)" }}>{kfmt(m.tokens)}</span>
            </div>)}</div>}
          </div>
        </div>

        <div style={pnl}>{head("#3ecf7f", "Metered vs estimated")}
          <div style={{ padding: 15, flex: 1, display: "grid", placeItems: "center" }}>
            <svg width="140" height="140" viewBox="0 0 150 150">
              <circle cx="75" cy="75" r="60" fill="none" stroke="var(--panel-2)" strokeWidth="13" />
              <circle cx="75" cy="75" r="60" fill="none" stroke="#3ecf7f" strokeWidth="13" strokeLinecap="round" strokeDasharray={C} strokeDashoffset={on ? C * (1 - meteredShare / 100) : C} transform="rotate(-90 75 75)" style={{ transition: "stroke-dashoffset 1s ease" }} />
              <text x="75" y="70" textAnchor="middle" fontSize="28" fontFamily="var(--mono)" fontWeight="700" fill="var(--text)">{meteredShare}%</text>
              <text x="75" y="90" textAnchor="middle" fontSize="9.5" fill="var(--faint)">real usage metered</text>
            </svg>
          </div>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1.5fr 1fr", gap: 14, alignItems: "start" }}>
        <div style={pnl}>{head("var(--muted)", "Top users this month")}
          <div style={{ overflowX: "auto" }}>{byUser.length === 0 ? <div className="note" style={{ padding: 15 }}>No usage yet.</div> : (
            <table className="tbl"><thead><tr><th style={{ paddingLeft: 14 }}>User</th><th style={{ textAlign: "right" }}>Tokens</th><th style={{ textAlign: "right" }}>Calls</th><th style={{ paddingRight: 14 }}>Budget</th></tr></thead>
              <tbody>{byUser.map((u) => { const lim = u.role === "admin" ? null : (u.limit ?? data.defaultLimit); const p = lim ? pct(u.tokens, lim) : 0; const col = p >= 100 ? "#f0616d" : p >= 80 ? "#f59e0b" : "#5b7cff"; return (
                <tr key={u.email}>
                  <td style={{ paddingLeft: 14 }}><b style={{ fontSize: 12.5 }}>{u.name || u.email}</b><div className="note">{u.email}</div></td>
                  <td className="mono" style={{ textAlign: "right" }}>{fmt(u.tokens)}</td>
                  <td className="mono" style={{ textAlign: "right", color: "var(--muted)" }}>{fmt(u.calls)}</td>
                  <td style={{ minWidth: 120, paddingRight: 14 }}>{lim == null ? <span className="note">unlimited</span> : <><div style={{ height: 5, background: "var(--panel-2)", borderRadius: 3, overflow: "hidden" }}><div style={{ height: "100%", width: `${Math.min(100, p)}%`, background: col, transformOrigin: "left", transform: on ? "scaleX(1)" : "scaleX(0)", transition: "transform .8s ease" }} /></div><div className="note" style={{ marginTop: 2 }}>{p}% of {fmt(lim)}</div></>}</td>
                </tr>); })}</tbody>
            </table>)}</div>
        </div>

        <div style={pnl}>{head("#f59e0b", "Recent activity", <span className="note" style={{ fontSize: 10 }}>audit · {Math.min(audit.length, 40)}</span>)}
          <div style={{ overflowX: "auto", maxHeight: 320 }}>{audit.length === 0 ? <div className="note" style={{ padding: 15 }}>No events logged yet.</div> : (
            <table className="tbl"><thead><tr><th style={{ paddingLeft: 14 }}>When</th><th style={{ paddingRight: 14 }}>Event</th></tr></thead>
              <tbody>{audit.map((a) => { const b = eventBadge(a.event); return (
                <tr key={a.id}>
                  <td className="mono" style={{ whiteSpace: "nowrap", fontSize: 10.5, color: "var(--muted)", paddingLeft: 14 }}>{a.ts ? new Date(a.ts).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "—"}</td>
                  <td style={{ paddingRight: 14 }}><span style={{ fontSize: 9.5, fontWeight: 600, padding: "2px 8px", borderRadius: 20, color: b.color, background: `color-mix(in srgb, ${b.color} 14%, transparent)` }}>{b.label}</span></td>
                </tr>); })}</tbody>
            </table>)}</div>
        </div>
      </div>
    </>
  );
}
