"use client";
import { useEffect, useState } from "react";

type Row = { id: string; event: string; userId: string | null; detail: unknown; ts: string | null; email: string | null };

const COLOR: Record<string, string> = {
  login: "var(--good)", login_failed: "var(--crit)", quota_exceeded: "var(--warn)",
  user_deleted: "var(--crit)", provider_added: "var(--accent)", provider_deleted: "var(--crit)",
  mcp_server_added: "var(--accent)", signup: "var(--good)",
};

export default function AuditLogViewer() {
  const [rows, setRows] = useState<Row[]>([]);
  const [events, setEvents] = useState<string[]>([]);
  const [filter, setFilter] = useState("");
  const [loading, setLoading] = useState(true);

  async function load(ev = filter) {
    setLoading(true);
    try {
      const r = await fetch(`/api/admin/audit${ev ? `?event=${encodeURIComponent(ev)}` : ""}`);
      const j = await r.json();
      if (r.ok) { setRows(j.rows || []); setEvents(j.events || []); }
    } catch { /* ignore */ } finally { setLoading(false); }
  }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { load(""); }, []);

  const fmtTs = (t: string | null) => (t ? new Date(t).toLocaleString() : "—");
  const fmtDetail = (d: unknown) => { if (d == null) return ""; try { return typeof d === "string" ? d : JSON.stringify(d); } catch { return String(d); } };

  return (
    <div>
      <div className="row" style={{ gap: 8, alignItems: "center", marginBottom: 12, flexWrap: "wrap" }}>
        <label className="note">Event</label>
        <select value={filter} onChange={(e) => { setFilter(e.target.value); load(e.target.value); }} style={{ maxWidth: 220 }}>
          <option value="">all events</option>
          {events.map((e) => <option key={e} value={e}>{e}</option>)}
        </select>
        <span className="note">{rows.length} shown</span>
        <button className="btn ghost sm" style={{ marginLeft: "auto" }} onClick={() => load()}>↻ Refresh</button>
      </div>
      {loading ? <div className="note"><span className="busy-dot" /> loading…</div>
        : rows.length === 0 ? <div className="note">No audit events{filter ? ` for "${filter}"` : ""} yet.</div>
        : (
          <div style={{ overflowX: "auto", border: "1px solid var(--border)", borderRadius: 10 }}>
            <table className="dtable" style={{ width: "100%" }}><tbody>
              <tr><th style={{ textAlign: "left" }}>when</th><th style={{ textAlign: "left" }}>event</th><th style={{ textAlign: "left" }}>user</th><th style={{ textAlign: "left" }}>detail</th></tr>
              {rows.map((r) => (
                <tr key={r.id}>
                  <td style={{ whiteSpace: "nowrap", color: "var(--muted)", fontFamily: "var(--mono)", fontSize: 11 }}>{fmtTs(r.ts)}</td>
                  <td><span style={{ fontFamily: "var(--mono)", fontSize: 11.5, fontWeight: 600, color: COLOR[r.event] || "var(--text)" }}>{r.event}</span></td>
                  <td style={{ fontSize: 12 }}>{r.email || <span className="note">{r.userId ? r.userId.slice(0, 8) : "system"}</span>}</td>
                  <td style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--muted)", maxWidth: 520, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={fmtDetail(r.detail)}>{fmtDetail(r.detail)}</td>
                </tr>
              ))}
            </tbody></table>
          </div>
        )}
    </div>
  );
}
