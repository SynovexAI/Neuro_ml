"use client";
import { useEffect, useState } from "react";

type Data = { configured: boolean; views7d: number; activeUsers24h: number; activeUsers7d: number; perDay: { day: string; n: number }[]; topPaths: { path: string; n: number }[]; totals?: { users: number; projects: number; kbs: number; runs7d: number }; storage?: { configured: boolean; bytes: number; files: number; pct: number } };
const fmtMB = (b: number) => (b < 1024 * 1024 ? `${(b / 1024).toFixed(0)} KB` : `${(b / 1024 / 1024).toFixed(1)} MB`);

export default function AnalyticsDashboard() {
  const [d, setD] = useState<Data | null>(null);
  const [loading, setLoading] = useState(true);
  async function load() { setLoading(true); try { const r = await fetch("/api/admin/analytics"); if (r.ok) setD(await r.json()); } catch { /* ignore */ } finally { setLoading(false); } }
  useEffect(() => { load(); }, []);

  if (loading) return <div className="note"><span className="busy-dot" /> loading…</div>;
  if (!d) return <div className="warnbar">Could not load analytics.</div>;
  if (!d.configured) return <div className="warnbar">Analytics table not found yet — it records automatically once the <code>page_views</code> table exists.</div>;

  const maxDay = Math.max(1, ...d.perDay.map((x) => x.n));
  const maxPath = Math.max(1, ...d.topPaths.map((x) => x.n));
  const tile = (label: string, val: number | string) => (
    <div style={{ background: "var(--panel)", border: "1px solid var(--border)", borderRadius: 12, padding: "12px 16px", flex: 1, minWidth: 140 }}>
      <div style={{ fontSize: 22, fontWeight: 600, fontFamily: "var(--mono)" }}>{val}</div>
      <div className="note" style={{ marginTop: 2 }}>{label}</div>
    </div>
  );

  return (
    <div>
      <div className="row" style={{ gap: 12, marginBottom: 16, flexWrap: "wrap" }}>
        {tile("page views · 7d", d.views7d.toLocaleString())}
        {tile("active users · 24h", d.activeUsers24h)}
        {tile("active users · 7d", d.activeUsers7d)}
        <button className="btn ghost sm" style={{ alignSelf: "center" }} onClick={load}>↻ Refresh</button>
      </div>

      {d.totals && (<>
        <div className="prep-col-h">server overview</div>
        <div className="row" style={{ gap: 12, margin: "0 0 16px", flexWrap: "wrap" }}>
          {tile("total users", d.totals.users)}
          {tile("saved projects", d.totals.projects)}
          {tile("knowledge bases", d.totals.kbs)}
          {tile("agent runs · 7d", d.totals.runs7d)}
        </div>
      </>)}

      {d.storage && (
        <div style={{ marginBottom: 18 }}>
          <div className="prep-col-h">object storage</div>
          {d.storage.configured ? (<>
            <div className="row" style={{ gap: 10, alignItems: "center", marginBottom: 6 }}>
              <span className="note"><b style={{ color: "var(--text)" }}>{fmtMB(d.storage.bytes)}</b> used · {d.storage.files} files · {d.storage.pct.toFixed(0)}% of ~1 GB</span>
            </div>
            <div style={{ height: 8, borderRadius: 4, background: "var(--panel)", border: "1px solid var(--border)", overflow: "hidden" }}>
              <div style={{ height: "100%", width: `${Math.min(100, d.storage.pct)}%`, background: d.storage.pct > 80 ? "var(--crit)" : "var(--accent)", transition: "width .3s" }} />
            </div>
            {d.storage.pct > 80 && <div className="note" style={{ marginTop: 6, color: "var(--crit)" }}>⚠ Over 80% — free space in the Storage page.</div>}
          </>) : <div className="note">Storage not configured (set <code>BLOB_READ_WRITE_TOKEN</code> or R2 vars).</div>}
        </div>
      )}

      <div className="prep-col-h">page views · last 7 days</div>
      <div style={{ display: "flex", alignItems: "flex-end", gap: 8, height: 140, padding: "8px 4px", border: "1px solid var(--border)", borderRadius: 10, marginBottom: 18 }}>
        {d.perDay.length === 0 ? <div className="note">No views recorded yet — browse the app and refresh.</div>
          : d.perDay.map((x) => (
            <div key={x.day} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "flex-end", height: "100%" }} title={`${x.day}: ${x.n} views`}>
              <div style={{ fontSize: 10, color: "var(--muted)", fontFamily: "var(--mono)" }}>{x.n}</div>
              <div style={{ width: "70%", height: `${Math.round((x.n / maxDay) * 100)}%`, background: "var(--accent)", borderRadius: "4px 4px 0 0", minHeight: 2 }} />
              <div style={{ fontSize: 9, color: "var(--faint)", marginTop: 3 }}>{x.day.slice(5)}</div>
            </div>
          ))}
      </div>

      <div className="prep-col-h">top pages · 7d</div>
      <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
        {d.topPaths.length === 0 ? <div className="note">No data yet.</div>
          : d.topPaths.map((x) => (
            <div key={x.path} className="row" style={{ gap: 10, alignItems: "center" }}>
              <span style={{ width: 220, flex: "0 0 auto", fontFamily: "var(--mono)", fontSize: 12, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{x.path}</span>
              <div style={{ flex: 1, height: 14, background: "var(--panel)", borderRadius: 4, overflow: "hidden" }}><div style={{ height: "100%", width: `${Math.round((x.n / maxPath) * 100)}%`, background: "var(--sky, #22b8cf)" }} /></div>
              <span className="note" style={{ width: 44, textAlign: "right" }}>{x.n}</span>
            </div>
          ))}
      </div>
    </div>
  );
}
