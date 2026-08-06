"use client";
import { useEffect, useState } from "react";

type File = { key: string; url: string; size: number; uploadedAt?: string };
const fmt = (b: number) => (b < 1024 ? `${b} B` : b < 1024 * 1024 ? `${(b / 1024).toFixed(1)} KB` : `${(b / 1024 / 1024).toFixed(2)} MB`);

export default function StorageManager() {
  const [files, setFiles] = useState<File[]>([]);
  const [backend, setBackend] = useState<string | null>(null);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState("");
  const CAP = 1024 * 1024 * 1024; // ~1 GB reference (Blob free tier) for the usage bar

  async function load() {
    setLoading(true);
    try { const r = await fetch("/api/admin/storage"); const j = await r.json(); if (r.ok) { setFiles(j.files || []); setBackend(j.backend); setTotal(j.totalBytes || 0); } }
    catch { /* ignore */ } finally { setLoading(false); }
  }
  useEffect(() => { load(); }, []);

  async function del(f: File) {
    if (!confirm(`Delete ${f.key}? This frees space so users can upload again.`)) return;
    setBusy(f.url);
    try { await fetch(`/api/admin/storage?url=${encodeURIComponent(f.url)}`, { method: "DELETE" }); await load(); }
    finally { setBusy(""); }
  }

  if (loading) return <div className="note"><span className="busy-dot" /> loading storage…</div>;
  if (!backend) return <div className="warnbar">No storage backend configured. Set <code>BLOB_READ_WRITE_TOKEN</code> (Vercel Blob) or the <code>R2_*</code> vars to enable file storage.</div>;

  const pct = Math.min(100, (total / CAP) * 100);
  return (
    <div>
      <div className="row" style={{ alignItems: "center", gap: 10, marginBottom: 6 }}>
        <b>{files.length}</b> files · <b>{fmt(total)}</b> used <span className="badge">{backend}</span>
        <button className="btn ghost sm" style={{ marginLeft: "auto" }} onClick={load}>↻ Refresh</button>
      </div>
      <div style={{ height: 8, borderRadius: 4, background: "var(--panel)", border: "1px solid var(--border)", overflow: "hidden", marginBottom: 14 }}>
        <div style={{ height: "100%", width: `${pct}%`, background: pct > 85 ? "var(--crit)" : "var(--accent)", transition: "width .3s" }} />
      </div>
      {backend === "blob" && <div className="note" style={{ marginBottom: 10 }}>Bar is relative to ~1 GB (Vercel Blob free tier). Delete files here to free space when it fills up.</div>}
      {files.length === 0 ? <div className="note">No files stored yet.</div> : (
        <div style={{ display: "flex", flexDirection: "column", gap: 5, maxHeight: 480, overflowY: "auto" }}>
          {files.map((f) => (
            <div key={f.url} className="row" style={{ gap: 10, alignItems: "center", border: "1px solid var(--border)", borderRadius: 8, padding: "8px 11px", background: "var(--surface)" }}>
              <a href={f.url} target="_blank" rel="noreferrer" style={{ flex: 1, minWidth: 0, fontFamily: "var(--mono)", fontSize: 12, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", color: "var(--text)" }}>{f.key}</a>
              <span className="note" style={{ flex: "0 0 auto" }}>{fmt(f.size)}</span>
              <button className="btn ghost sm" onClick={() => del(f)} disabled={busy === f.url}>{busy === f.url ? "…" : "Delete"}</button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
