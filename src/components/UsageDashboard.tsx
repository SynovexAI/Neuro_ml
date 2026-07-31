"use client";

import { useEffect, useState } from "react";

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
const pct = (a: number, b: number) => (b > 0 ? Math.round((a / b) * 100) : 0);

function eventBadge(ev: string): { cls: string; label: string } {
  if (ev === "login_failed") return { cls: "badge warn", label: "login failed" };
  if (ev === "quota_exceeded") return { cls: "badge", label: "quota hit" };
  if (ev === "login") return { cls: "badge good", label: "login" };
  return { cls: "badge", label: ev };
}

function Health() {
  const [s, setS] = useState<{ ok: boolean; db: string; ms?: number } | null>(null);
  const [err, setErr] = useState(false);
  useEffect(() => {
    fetch("/api/health").then((r) => r.json()).then(setS).catch(() => setErr(true));
  }, []);
  const up = s?.ok && !err;
  return (
    <span className={`badge ${up ? "good" : "warn"}`} title={s ? `db ${s.db}${s.ms != null ? ` · ${s.ms}ms` : ""}` : ""}>
      {err ? "health: unreachable" : s ? (up ? `healthy · db ${s.ms}ms` : "degraded") : "checking…"}
    </span>
  );
}

export default function UsageDashboard({ data }: { data: Data }) {
  const { total, byUser, byLab, byModel, byDay, audit } = data;
  const meteredShare = 100 - pct(total.estimated, total.tokens);
  const maxDay = Math.max(1, ...byDay.map((d) => d.tokens));
  const maxLab = Math.max(1, ...byLab.map((l) => l.tokens));

  return (
    <>
      <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 10 }}><Health /></div>

      {/* KPI tiles */}
      <div className="cv-summary" style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 16 }}>
        <div className="metric"><span className="v">{fmt(total.tokens)}</span><span className="k">Tokens this month</span></div>
        <div className="metric"><span className="v">{fmt(total.calls)}</span><span className="k">LLM calls</span></div>
        <div className="metric"><span className="v">{byUser.length}</span><span className="k">Active users</span></div>
        <div className="metric"><span className="v">{meteredShare}%</span><span className="k">Metered (vs estimated)</span></div>
      </div>

      {/* Daily trend */}
      <div className="card" style={{ marginBottom: 16 }}>
        <div className="card-h"><span className="t">Daily token usage</span><span className="mono r">last 14 days</span></div>
        <div className="card-b">
          {byDay.length === 0 ? <div className="note">No usage recorded yet.</div> : (
            <div style={{ display: "flex", alignItems: "flex-end", gap: 4, height: 120 }}>
              {byDay.map((d) => (
                <div key={d.day} title={`${d.day}: ${fmt(d.tokens)} tokens`} style={{ flex: 1, display: "flex", flexDirection: "column", justifyContent: "flex-end", height: "100%" }}>
                  <div style={{ height: `${Math.max(2, (d.tokens / maxDay) * 100)}%`, background: "var(--accent)", borderRadius: "3px 3px 0 0", opacity: 0.85 }} />
                  <div className="note" style={{ textAlign: "center", marginTop: 3, fontSize: 9 }}>{d.day.slice(5)}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="split" style={{ display: "grid", gridTemplateColumns: "1.4fr 1fr", gap: 16, marginBottom: 16 }}>
        {/* Top users */}
        <div className="card">
          <div className="card-h"><span className="t">Top users this month</span></div>
          <div className="card-b" style={{ overflowX: "auto" }}>
            {byUser.length === 0 ? <div className="note">No usage yet.</div> : (
              <table className="tbl">
                <thead><tr><th>User</th><th style={{ textAlign: "right" }}>Tokens</th><th style={{ textAlign: "right" }}>Calls</th><th>Budget</th></tr></thead>
                <tbody>
                  {byUser.map((u) => {
                    const lim = u.role === "admin" ? null : (u.limit ?? data.defaultLimit);
                    const p = lim ? pct(u.tokens, lim) : 0;
                    return (
                      <tr key={u.email}>
                        <td><b style={{ fontSize: 12.5 }}>{u.name || u.email}</b><div className="note">{u.email}</div></td>
                        <td className="mono" style={{ textAlign: "right" }}>{fmt(u.tokens)}</td>
                        <td className="mono" style={{ textAlign: "right", color: "var(--muted)" }}>{fmt(u.calls)}</td>
                        <td style={{ minWidth: 110 }}>
                          {lim == null ? <span className="note">unlimited</span> : (
                            <div>
                              <div style={{ height: 5, background: "var(--panel-2)", borderRadius: 3, overflow: "hidden" }}>
                                <div style={{ height: "100%", width: `${Math.min(100, p)}%`, background: p >= 100 ? "#e5484d" : p >= 80 ? "#f59e0b" : "var(--accent)" }} />
                              </div>
                              <div className="note" style={{ marginTop: 2 }}>{p}% of {fmt(lim)}</div>
                            </div>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </div>
        </div>

        {/* By lab + by model */}
        <div className="card">
          <div className="card-h"><span className="t">By lab &amp; model</span></div>
          <div className="card-b">
            {byLab.length === 0 ? <div className="note">No usage yet.</div> : (
              <>
                {byLab.map((l) => (
                  <div key={l.lab} style={{ marginBottom: 8 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }}><span>{l.lab}</span><span className="mono" style={{ color: "var(--muted)" }}>{fmt(l.tokens)}</span></div>
                    <div style={{ height: 5, background: "var(--panel-2)", borderRadius: 3, overflow: "hidden", marginTop: 3 }}>
                      <div style={{ height: "100%", width: `${(l.tokens / maxLab) * 100}%`, background: "var(--accent)" }} />
                    </div>
                  </div>
                ))}
                <div className="note" style={{ margin: "12px 0 6px", textTransform: "uppercase", letterSpacing: ".04em" }}>Models</div>
                <table className="tbl">
                  <tbody>
                    {byModel.map((m) => (
                      <tr key={m.model}><td style={{ fontSize: 12 }}>{m.model}</td><td className="mono" style={{ textAlign: "right", color: "var(--muted)" }}>{fmt(m.tokens)}</td></tr>
                    ))}
                  </tbody>
                </table>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Audit / event log */}
      <div className="card">
        <div className="card-h"><span className="t">Recent activity</span><span className="mono r">audit log · latest 40</span></div>
        <div className="card-b" style={{ overflowX: "auto" }}>
          {audit.length === 0 ? <div className="note">No events logged yet.</div> : (
            <table className="tbl">
              <thead><tr><th>When</th><th>Event</th><th>Detail</th></tr></thead>
              <tbody>
                {audit.map((a) => {
                  const b = eventBadge(a.event);
                  const detail = a.detail ? Object.entries(a.detail).map(([k, v]) => `${k}=${v}`).join("  ") : "";
                  return (
                    <tr key={a.id}>
                      <td className="mono" style={{ whiteSpace: "nowrap", fontSize: 11, color: "var(--muted)" }}>{a.ts ? new Date(a.ts).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "—"}</td>
                      <td><span className={b.cls}>{b.label}</span></td>
                      <td className="mono" style={{ fontSize: 11, color: "var(--muted)" }}>{detail}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </>
  );
}
