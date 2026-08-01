"use client";

import { useEffect, useRef, useState } from "react";
import { toast, confirmDialog } from "@/lib/toast";

type Kb = { id: string; name: string; status: string; docCount: number; chunkCount: number; embModel: string | null; updatedAt?: string | null };
type Staged = { name: string; text: string };

const CONNECTORS = [
  { id: "file", name: "File Upload", desc: "PDF, DOCX, XLSX, CSV, TXT", icon: "📄", ok: true },
  { id: "web", name: "Web page", desc: "Fetch & index a URL", icon: "🌐", ok: true },
  { id: "confluence", name: "Confluence", desc: "Wiki pages & spaces", icon: "🟦", ok: false },
  { id: "notion", name: "Notion", desc: "Pages & databases", icon: "⬛", ok: false },
  { id: "sharepoint", name: "SharePoint", desc: "Microsoft 365 docs", icon: "🟩", ok: false },
  { id: "gdrive", name: "Google Drive", desc: "Docs, sheets, slides", icon: "🟨", ok: false },
  { id: "github", name: "GitHub", desc: "Repos, issues, wikis", icon: "🐙", ok: false },
  { id: "db", name: "Database", desc: "SQL / warehouse table", icon: "🗄", ok: false },
];

function fmtWhen(v?: string | null): string {
  if (!v) return "—";
  const d = new Date(v); if (isNaN(d.getTime())) return "—";
  const mins = Math.round((Date.now() - d.getTime()) / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const h = Math.round(mins / 60); if (h < 24) return `${h}h ago`;
  return `${Math.round(h / 24)}d ago`;
}

export default function KnowledgeBases() {
  const [kbs, setKbs] = useState<Kb[]>([]);
  const [selId, setSelId] = useState<string | null>(null);
  const [staged, setStaged] = useState<Staged[]>([]);
  const [newName, setNewName] = useState("");
  const [url, setUrl] = useState("https://en.wikipedia.org/wiki/Retrieval-augmented_generation");
  const [busy, setBusy] = useState("");
  const [msg, setMsg] = useState("");
  const [tab, setTab] = useState<"add" | "docs">("add");
  const [syncedDocs, setSyncedDocs] = useState<{ docName: string | null; chunks: number }[]>([]);
  const [search, setSearch] = useState("");
  const [connect, setConnect] = useState(false);
  const [pick, setPick] = useState<string | null>(null);
  const [chunkSize, setChunkSize] = useState(60);
  const [chunkOverlap, setChunkOverlap] = useState(12);
  const fileRef = useRef<HTMLInputElement>(null);

  async function loadKbs() { try { const j = await fetch("/api/kb").then((r) => r.json()); setKbs(j.kbs || []); } catch { setKbs([]); } }
  async function loadDetail(id: string) { try { const j = await fetch(`/api/kb/${id}`).then((r) => r.json()); setSyncedDocs(j.docs || []); } catch { setSyncedDocs([]); } }
  function selectKb(id: string, hasDocs: boolean) { setSelId(id); setStaged([]); setMsg(""); setSyncedDocs([]); setTab(hasDocs ? "docs" : "add"); loadDetail(id); }
  useEffect(() => { loadKbs(); }, []);

  const sel = kbs.find((k) => k.id === selId) || null;
  const shown = kbs.filter((k) => k.name.toLowerCase().includes(search.toLowerCase()));
  const totals = { docs: kbs.reduce((a, k) => a + k.docCount, 0), chunks: kbs.reduce((a, k) => a + k.chunkCount, 0), syncing: kbs.filter((k) => k.status === "syncing").length, failed: kbs.filter((k) => k.status === "error").length };

  async function createKb() {
    if (!newName.trim()) return;
    const name = newName.trim();
    setBusy("create");
    const r = await fetch("/api/kb", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ name }) });
    const j = await r.json().catch(() => ({}));
    setNewName(""); await loadKbs();
    if (j.id) { setSelId(j.id); setStaged([]); setSyncedDocs([]); setTab("add"); setConnect(false); setPick(null); toast(`Created “${name}” — add documents to sync`, "success"); }
    else toast("Couldn't create the knowledge base", "error");
    setBusy("");
  }
  async function removeKb(k: Kb) {
    if (!(await confirmDialog(`Delete knowledge base “${k.name}”? This can't be undone.`, { confirmLabel: "Delete", danger: true }))) return;
    await fetch(`/api/kb/${k.id}`, { method: "DELETE" }).catch(() => {});
    if (selId === k.id) { setSelId(null); setStaged([]); }
    loadKbs();
    toast("Knowledge base deleted", "success");
  }

  async function addFiles(files: FileList | null) {
    if (!files?.length) return;
    setBusy("extract"); setMsg("");
    try {
      for (const f of Array.from(files)) {
        const ext = (f.name.split(".").pop() || "").toLowerCase();
        let text = "";
        if (["pdf", "docx", "doc", "xlsx", "xls", "xlsm"].includes(ext)) {
          const fd = new FormData(); fd.append("file", f);
          const r = await fetch("/api/rag/extract", { method: "POST", body: fd });
          const j = await r.json(); if (!r.ok) throw new Error(j.error || `couldn't read ${f.name}`);
          text = j.text || "";
        } else { text = await f.text(); }
        if (text.trim()) setStaged((s) => [...s, { name: f.name, text }]);
      }
    } catch (e) { setMsg((e as Error).message); }
    setBusy(""); if (fileRef.current) fileRef.current.value = "";
  }
  async function addUrl() {
    if (!url.trim()) return;
    setBusy("url"); setMsg("");
    try {
      const r = await fetch("/api/rag/fetch-url", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ url }) });
      const j = await r.json(); if (!r.ok) throw new Error(j.error || "couldn't fetch URL");
      if ((j.text || "").trim()) setStaged((s) => [...s, { name: url, text: j.text }]);
    } catch (e) { setMsg((e as Error).message); }
    setBusy("");
  }
  async function sync() {
    if (!selId || !staged.length) { setMsg("Add files or URLs first."); return; }
    setBusy("sync"); setMsg("");
    try {
      const r = await fetch(`/api/kb/${selId}/sync`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ docs: staged, chunkSize, chunkOverlap: Math.min(chunkOverlap, chunkSize - 1) }) });
      const j = await r.json().catch(() => ({}));
      if (!r.ok) { setMsg(j.error || "Sync failed."); toast(j.error || "Sync failed", "error"); }
      else { setStaged([]); await loadKbs(); await loadDetail(selId); setTab("docs"); toast(`Synced ✓ ${j.chunkCount} chunks · ${j.embModel === "tfidf" ? "TF-IDF" : "embeddings (" + j.embModel + ")"}`, "success"); }
    } catch (e) { setMsg((e as Error).message); }
    setBusy("");
  }

  const badge = (s: string) => s === "ready" ? "badge good" : s === "syncing" ? "badge warn" : s === "error" ? "badge" : "badge";

  return (
    <>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 16 }}>
        <span className={`badge ${totals.failed ? "warn" : "good"}`}>{totals.failed ? `${totals.failed} failed` : "all synced"}</span>
        <span className="badge">{kbs.length} sources</span>
        <span className="badge">{totals.docs} docs</span>
        <span style={{ flex: 1 }} />
        <button className="btn" onClick={() => { setConnect(true); setPick(null); }}>+ Connect source</button>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 12, marginBottom: 16 }}>
        <div className="metric"><span className="k">Total sources</span><span className="v">{kbs.length}</span></div>
        <div className="metric"><span className="k">Active syncs</span><span className="v" style={{ color: totals.syncing ? "#f59e0b" : undefined }}>{totals.syncing}</span></div>
        <div className="metric"><span className="k">Failed</span><span className="v" style={{ color: totals.failed ? "#e5484d" : undefined }}>{totals.failed}</span></div>
        <div className="metric"><span className="k">Documents</span><span className="v">{totals.docs.toLocaleString()}</span></div>
        <div className="metric"><span className="k">Chunks</span><span className="v">{totals.chunks.toLocaleString()}</span></div>
      </div>

      <input type="text" placeholder="Search sources…" value={search} onChange={(e) => setSearch(e.target.value)} style={{ marginBottom: 14 }} />

      <div className="card" style={{ marginBottom: sel ? 18 : 0 }}>
        <div className="card-h"><span className="t">{shown.length} source{shown.length === 1 ? "" : "s"}</span></div>
        <div className="card-b" style={{ padding: 0, overflowX: "auto" }}>
          {kbs.length === 0 ? <div className="note" style={{ padding: 20, textAlign: "center" }}>No sources yet — click <b>+ Connect source</b> to add one.</div> : (
            <table className="tbl">
              <thead><tr><th style={{ paddingLeft: 16 }}>Source</th><th style={{ textAlign: "right" }}>Documents</th><th style={{ textAlign: "right" }}>Chunks</th><th>Status</th><th>Last sync</th><th></th></tr></thead>
              <tbody>
                {shown.map((k) => (
                  <tr key={k.id} onClick={() => selectKb(k.id, k.docCount > 0)} style={{ cursor: "pointer", background: selId === k.id ? "var(--panel)" : undefined }}>
                    <td style={{ paddingLeft: 16 }}><b style={{ fontSize: 13 }}>📄 {k.name}</b><div className="note">{k.embModel === "tfidf" ? "TF-IDF vectors" : k.embModel ? `embeddings · ${k.embModel}` : "not synced"}</div></td>
                    <td className="mono" style={{ textAlign: "right" }}>{k.docCount}</td>
                    <td className="mono" style={{ textAlign: "right", color: "var(--muted)" }}>{k.chunkCount}</td>
                    <td><span className={badge(k.status)}>{k.status}</span></td>
                    <td className="note" style={{ whiteSpace: "nowrap" }}>{fmtWhen(k.updatedAt)}</td>
                    <td style={{ textAlign: "right", whiteSpace: "nowrap", paddingRight: 12 }}>
                      <button className="btn ghost sm" title="Open" onClick={(e) => { e.stopPropagation(); selectKb(k.id, k.docCount > 0); }}>Open</button>
                      <button className="btn ghost sm danger" title="Delete" style={{ marginLeft: 6 }} onClick={(e) => { e.stopPropagation(); removeKb(k); }}>🗑</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {sel && (
        <div className="card">
          <div className="card-h"><span className="t">{sel.name}</span><span className={badge(sel.status)} style={{ marginLeft: 8 }}>{sel.status}</span><span style={{ flex: 1 }} /><button className="btn ghost sm" onClick={() => setSelId(null)}>Close</button><button className="btn ghost sm danger" style={{ marginLeft: 6 }} onClick={() => removeKb(sel)}>Delete</button></div>
          <div className="card-b">
            <div className="seg" style={{ maxWidth: 300, marginBottom: 14 }}>
              <button className={tab === "add" ? "on" : ""} onClick={() => setTab("add")}>Add documents</button>
              <button className={tab === "docs" ? "on" : ""} onClick={() => setTab("docs")}>Documents{sel.docCount ? ` · ${sel.docCount}` : ""}</button>
            </div>
            {tab === "add" ? (
              <>
                <div className="kb-drop">
                  <div style={{ fontSize: 26, opacity: 0.45, lineHeight: 1 }}>⬆</div>
                  <div style={{ fontWeight: 600, fontSize: 13.5, marginTop: 6 }}>Add documents</div>
                  <div className="note" style={{ marginTop: 2 }}>PDF · DOCX · XLSX · CSV · TXT — or a web page</div>
                  <div className="row" style={{ gap: 8, marginTop: 12, justifyContent: "center" }}>
                    <button className="btn sm" onClick={() => fileRef.current?.click()} disabled={busy === "extract"}>{busy === "extract" ? "Reading…" : "Choose files"}</button>
                    <input ref={fileRef} type="file" multiple accept=".txt,.md,.csv,.json,.pdf,.docx,.doc,.xlsx,.xls" onChange={(e) => addFiles(e.target.files)} style={{ display: "none" }} />
                  </div>
                  <div className="row" style={{ gap: 8, marginTop: 10, maxWidth: 460, marginLeft: "auto", marginRight: "auto" }}>
                    <input type="text" placeholder="https://…" value={url} onChange={(e) => setUrl(e.target.value)} />
                    <button className="btn ghost sm" onClick={addUrl} disabled={busy === "url"}>{busy === "url" ? "Fetching…" : "Add URL"}</button>
                  </div>
                </div>
                {staged.length > 0 && (
                  <div style={{ marginTop: 12 }}>
                    <label className="fld">Staged for sync ({staged.length})</label>
                    {staged.map((d, i) => (
                      <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, padding: "5px 0", borderBottom: "1px solid var(--border)" }}>
                        <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{d.name}</span>
                        <span className="note">{Math.round(d.text.length / 1000)}k chars</span>
                        <button className="btn ghost sm" onClick={() => setStaged((s) => s.filter((_, j) => j !== i))}>×</button>
                      </div>
                    ))}
                    <div className="row" style={{ gap: 14, flexWrap: "wrap", marginTop: 12, alignItems: "center" }}>
                      <div className="knob" style={{ margin: 0, minWidth: 150 }}><div className="kr"><span>Chunk size (words)</span><b>{chunkSize}</b></div><input type="range" min={20} max={200} step={10} value={chunkSize} onChange={(e) => { const v = +e.target.value; setChunkSize(v); if (chunkOverlap > v - 1) setChunkOverlap(Math.max(0, v - 1)); }} /></div>
                      <div className="knob" style={{ margin: 0, minWidth: 150 }}><div className="kr"><span>Overlap (words)</span><b>{chunkOverlap}</b></div><input type="range" min={0} max={Math.max(0, chunkSize - 1)} step={2} value={chunkOverlap} onChange={(e) => setChunkOverlap(+e.target.value)} /></div>
                      <span className="note">smaller chunks = more precise retrieval; larger = more context per chunk</span>
                    </div>
                    <button className="btn block" style={{ marginTop: 12 }} onClick={sync} disabled={busy === "sync"}>{busy === "sync" ? "Syncing → embedding + storing vectors…" : `⟳ Sync ${staged.length} doc${staged.length === 1 ? "" : "s"} to vector store`}</button>
                  </div>
                )}
                {sel.docCount > 0 && staged.length === 0 && <div className="note" style={{ marginTop: 12 }}>Re-syncing <b>replaces</b> the current contents ({sel.docCount} docs). Add files/URLs above, then sync.</div>}
                {msg && <div className={msg.startsWith("Synced") ? "note" : "err"} style={{ marginTop: 10 }}>{msg}</div>}
              </>
            ) : (
              <>
                {sel.status === "syncing" ? <div className="note">Syncing…</div>
                  : syncedDocs.length === 0 ? <div className="note">Nothing synced yet — switch to <b>Add documents</b> to ingest files or URLs.</div>
                  : (<>
                      <div className="stat-row" style={{ marginBottom: 12 }}>
                        <div className="stat"><b>{sel.docCount}</b>documents</div>
                        <div className="stat"><b>{sel.chunkCount}</b>chunks</div>
                        <div className="stat"><b>{sel.embModel === "tfidf" ? "TF-IDF" : "embeddings"}</b>vector store</div>
                      </div>
                      {syncedDocs.map((d, i) => (
                        <div key={i} style={{ display: "flex", alignItems: "center", gap: 10, fontSize: 12.5, padding: "8px 0", borderBottom: "1px solid var(--border)" }}>
                          <span style={{ fontSize: 15, opacity: 0.5 }}>📄</span>
                          <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{d.docName || "document"}</span>
                          <span className="note">{d.chunks} chunks</span>
                        </div>
                      ))}
                    </>)}
              </>
            )}
          </div>
        </div>
      )}

      {connect && (
        <div className="modal-wrap show" onClick={(e) => { if (e.target === e.currentTarget) setConnect(false); }}>
          <div className="modal" style={{ maxWidth: 640 }}>
            <div className="mh"><b>Connect a knowledge source</b><button className="x" onClick={() => setConnect(false)}>×</button></div>
            <div className="mb">
              {!pick ? (
                <>
                  <div className="note" style={{ marginBottom: 10 }}>Pick a source. File upload and web pages work today — the rest are on the roadmap.</div>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))", gap: 10 }}>
                    {CONNECTORS.map((c) => (
                      <button key={c.id} type="button" disabled={!c.ok} onClick={() => c.ok && setPick(c.id)}
                        style={{ textAlign: "left", border: "1px solid var(--border)", borderRadius: 9, padding: "11px 12px", background: "var(--surface)", cursor: c.ok ? "pointer" : "not-allowed", opacity: c.ok ? 1 : 0.5, fontFamily: "inherit" }}>
                        <div style={{ fontSize: 18 }}>{c.icon}</div>
                        <div style={{ fontSize: 13, fontWeight: 600, marginTop: 4 }}>{c.name}{!c.ok && <span className="badge" style={{ marginLeft: 6 }}>soon</span>}</div>
                        <div className="note" style={{ marginTop: 1 }}>{c.desc}</div>
                      </button>
                    ))}
                  </div>
                </>
              ) : (
                <>
                  <div className="note" style={{ marginBottom: 10 }}>New source · {CONNECTORS.find((c) => c.id === pick)?.name}</div>
                  <label className="fld">Name</label>
                  <input type="text" autoFocus placeholder="e.g. Product docs" value={newName} onChange={(e) => setNewName(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") createKb(); }} />
                  <div className="row" style={{ gap: 8, marginTop: 12 }}>
                    <button className="btn" onClick={createKb} disabled={busy === "create" || !newName.trim()}>{busy === "create" ? "Creating…" : "Create & add documents"}</button>
                    <button className="btn ghost" onClick={() => setPick(null)}>← Back</button>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
