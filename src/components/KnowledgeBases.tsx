"use client";

import { useEffect, useRef, useState } from "react";
import { toast, confirmDialog } from "@/lib/toast";

type Kb = { id: string; name: string; status: string; docCount: number; chunkCount: number; embModel: string | null; updatedAt?: string | null };
type Staged = { name: string; text: string };

const CONNECTORS = [
  { id: "file", name: "File Upload", desc: "PDF, DOCX, XLSX, CSV, TXT", icon: "📄", ok: true },
  { id: "web", name: "Web page", desc: "Fetch & index a URL", icon: "🌐", ok: true },
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
        // Best-effort: archive the original file to object storage (no-ops if storage isn't configured).
        try { const afd = new FormData(); afd.append("file", f); void fetch("/api/storage/upload", { method: "POST", body: afd }); } catch { /* ignore */ }
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

  const stColor = (s: string) => s === "ready" ? "var(--good)" : s === "syncing" ? "var(--warn)" : s === "error" ? "var(--crit)" : "var(--faint)";
  const stPill = (s: string): React.CSSProperties => ({ fontSize: 9.5, fontWeight: 600, textTransform: "uppercase", letterSpacing: ".04em", padding: "2px 8px", borderRadius: 20, color: stColor(s), background: `color-mix(in srgb, ${stColor(s)} 14%, transparent)` });
  const pnl: React.CSSProperties = { border: "1px solid var(--border)", borderRadius: 13, background: "var(--surface)", overflow: "hidden" };
  const docIcon = (name: string): [string, string] => { const k = (name.split(".").pop() || "").toLowerCase(); if (/^https?:/.test(name)) return ["🌐", "34,184,207"]; if (k === "pdf") return ["📕", "240,97,109"]; if (["docx", "doc"].includes(k)) return ["📘", "91,124,255"]; if (["xlsx", "xls", "csv"].includes(k)) return ["📊", "62,207,127"]; return ["📄", "91,124,255"]; };

  return (
    <>
      <div className="row" style={{ alignItems: "center", justifyContent: "space-between", gap: 10, marginBottom: 14, flexWrap: "wrap" }}>
        <div className="sec-title" style={{ margin: 0 }}>Your knowledge bases · {kbs.length}{totals.docs ? ` · ${totals.docs} docs` : ""}</div>
        <div className="row" style={{ gap: 10, alignItems: "center" }}>
          <input type="text" placeholder="Search…" value={search} onChange={(e) => setSearch(e.target.value)} style={{ width: 180 }} />
          <button className="btn" onClick={() => { setConnect(true); setPick(null); }}>+ Connect source</button>
        </div>
      </div>

      {kbs.length === 0 ? <div style={{ ...pnl, padding: 26, textAlign: "center" }} className="note">No knowledge bases yet — click <b>+ Connect source</b> to create one.</div> : (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(230px, 1fr))", gap: 14, marginBottom: sel ? 20 : 0 }}>
          {shown.map((k) => (
            <div key={k.id} onClick={() => selectKb(k.id, k.docCount > 0)} style={{ ...pnl, padding: 15, cursor: "pointer", borderColor: selId === k.id ? "var(--accent)" : "var(--border)", boxShadow: selId === k.id ? "0 0 0 1px var(--accent)" : undefined }}>
              <div className="row" style={{ gap: 11, alignItems: "center" }}>
                <span style={{ width: 36, height: 36, borderRadius: 10, display: "grid", placeItems: "center", fontSize: 16, flex: "0 0 auto", background: "var(--accent-weak)", color: "var(--accent-strong)" }}>🗄</span>
                <div style={{ flex: 1, minWidth: 0 }}><div style={{ fontWeight: 600, fontSize: 14, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{k.name}</div></div>
                <span style={stPill(k.status)}>{k.status}</span>
              </div>
              <div className="row" style={{ gap: 16, marginTop: 12, fontFamily: "var(--mono)", fontSize: 11, color: "var(--muted)" }}>
                <span><b style={{ color: "var(--text)" }}>{k.docCount}</b> docs</span>
                <span><b style={{ color: "var(--text)" }}>{k.chunkCount}</b> chunks</span>
                <span>{k.embModel === "tfidf" ? "TF-IDF" : k.embModel ? "embeddings" : "not built"}</span>
              </div>
              <div className="row" style={{ justifyContent: "space-between", alignItems: "center", marginTop: 10 }}>
                <span className="note">{fmtWhen(k.updatedAt)}</span>
                <button className="btn ghost sm danger" onClick={(e) => { e.stopPropagation(); removeKb(k); }}>Delete</button>
              </div>
            </div>
          ))}
        </div>
      )}

      {sel && (
        <div style={pnl}>
          <div className="row" style={{ alignItems: "center", gap: 10, padding: "12px 16px", borderBottom: "1px solid var(--border)", background: "var(--panel)" }}>
            <span style={{ width: 28, height: 28, borderRadius: 8, display: "grid", placeItems: "center", fontSize: 14, background: "var(--accent-weak)", color: "var(--accent-strong)" }}>🗄</span>
            <b style={{ fontSize: 14 }}>{sel.name}</b><span style={stPill(sel.status)}>{sel.status}</span>
            <span style={{ flex: 1 }} /><button className="btn ghost sm" onClick={() => setSelId(null)}>Close</button><button className="btn ghost sm danger" onClick={() => removeKb(sel)}>Delete</button>
          </div>
          <div style={{ display: "flex", gap: 18, padding: "0 16px", borderBottom: "1px solid var(--border)", background: "var(--panel)" }}>
            {(["add", "docs"] as const).map((tb) => <button key={tb} onClick={() => setTab(tb)} style={{ padding: "11px 0", background: "none", border: "none", borderBottom: `2px solid ${tab === tb ? "var(--accent)" : "transparent"}`, color: tab === tb ? "var(--accent-strong)" : "var(--muted)", fontSize: 12.5, fontWeight: tab === tb ? 600 : 500, cursor: "pointer", fontFamily: "inherit" }}>{tb === "add" ? "Add documents" : `Documents${sel.docCount ? ` · ${sel.docCount}` : ""}`}</button>)}
          </div>
          <div style={{ padding: 16 }}>
            {tab === "add" ? (
              <>
                <div onClick={() => fileRef.current?.click()} onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); addFiles(e.dataTransfer.files); }} style={{ border: "1.5px dashed var(--border-strong)", borderRadius: 12, padding: 22, textAlign: "center", cursor: "pointer", background: "var(--panel)" }}>
                  <div style={{ fontSize: 24, marginBottom: 5 }}>{busy === "extract" ? <span className="busy-dot" /> : "⬆"}</div>
                  <b style={{ fontSize: 13.5 }}>{busy === "extract" ? "Reading…" : "Drop files here or click to upload"}</b>
                  <div className="note" style={{ marginTop: 5 }}>PDF · DOCX · XLSX · CSV · TXT — multiple at once, they accumulate in this KB</div>
                  <input ref={fileRef} type="file" multiple accept=".txt,.md,.csv,.json,.pdf,.docx,.doc,.xlsx,.xls" onChange={(e) => addFiles(e.target.files)} style={{ display: "none" }} />
                </div>
                <div className="row" style={{ gap: 8, marginTop: 12 }}><input type="text" placeholder="https://… (web page)" value={url} onChange={(e) => setUrl(e.target.value)} onClick={(e) => e.stopPropagation()} /><button className="btn ghost sm" onClick={addUrl} disabled={busy === "url"} style={{ whiteSpace: "nowrap" }}>{busy === "url" ? "Fetching…" : "Add URL"}</button></div>
                {staged.length > 0 && (
                  <div style={{ marginTop: 14 }}>
                    <label className="fld">Staged · {staged.length} ready to add</label>
                    <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>{staged.map((d, i) => { const [ic, rgb] = docIcon(d.name); return (
                      <div key={i} className="row" style={{ alignItems: "center", gap: 10, border: "1px solid var(--border)", borderRadius: 9, padding: "9px 12px", background: "var(--panel)" }}>
                        <span style={{ width: 26, height: 26, borderRadius: 7, display: "grid", placeItems: "center", fontSize: 13, background: `rgba(${rgb},.14)`, flex: "0 0 auto" }}>{ic}</span>
                        <span style={{ flex: 1, minWidth: 0, fontSize: 12.5, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{d.name}</span>
                        <span className="note">{Math.round(d.text.length / 1000)}k chars</span>
                        <button className="btn ghost sm" onClick={() => setStaged((s) => s.filter((_, j) => j !== i))}>×</button>
                      </div>); })}</div>
                    <div className="row" style={{ gap: 14, flexWrap: "wrap", marginTop: 14, alignItems: "center" }}>
                      <div className="knob" style={{ margin: 0, minWidth: 150 }}><div className="kr"><span>Chunk size (words)</span><b>{chunkSize}</b></div><input type="range" min={20} max={200} step={10} value={chunkSize} onChange={(e) => { const v = +e.target.value; setChunkSize(v); if (chunkOverlap > v - 1) setChunkOverlap(Math.max(0, v - 1)); }} /></div>
                      <div className="knob" style={{ margin: 0, minWidth: 150 }}><div className="kr"><span>Overlap (words)</span><b>{chunkOverlap}</b></div><input type="range" min={0} max={Math.max(0, chunkSize - 1)} step={2} value={chunkOverlap} onChange={(e) => setChunkOverlap(+e.target.value)} /></div>
                    </div>
                    <button className="btn block" style={{ marginTop: 14 }} onClick={sync} disabled={busy === "sync"}>{busy === "sync" ? "Syncing → embedding + storing vectors…" : `⟳ Add ${staged.length} doc${staged.length === 1 ? "" : "s"} to this knowledge base`}</button>
                  </div>
                )}
                {sel.docCount > 0 && staged.length === 0 && <div className="note" style={{ marginTop: 12 }}>This KB already holds <b>{sel.docCount} doc{sel.docCount === 1 ? "" : "s"}</b>. New files/URLs you sync are <b>added</b> to it — nothing is replaced.</div>}
                {msg && <div className={msg.startsWith("Synced") ? "note" : "err"} style={{ marginTop: 10 }}>{msg}</div>}
              </>
            ) : (
              <>
                {sel.status === "syncing" ? <div className="note">Syncing…</div>
                  : syncedDocs.length === 0 ? <div className="note">Nothing synced yet — switch to <b>Add documents</b> to ingest files or URLs.</div>
                  : (<>
                      <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 12, marginBottom: 14 }}>
                        {([[String(sel.docCount), "documents"], [String(sel.chunkCount), "chunks"], [sel.embModel === "tfidf" ? "TF-IDF" : "embeddings", "vector store"]] as [string, string][]).map(([v, kk]) => <div key={kk} style={{ background: "var(--panel)", border: "1px solid var(--border)", borderRadius: 11, padding: "11px 14px" }}><div style={{ fontFamily: "var(--mono)", fontSize: 20, fontWeight: 600 }}>{v}</div><div style={{ fontSize: 9, textTransform: "uppercase", letterSpacing: ".05em", color: "var(--faint)", marginTop: 2 }}>{kk}</div></div>)}
                      </div>
                      {syncedDocs.map((d, i) => { const [ic, rgb] = docIcon(d.docName || ""); return (
                        <div key={i} className="row" style={{ alignItems: "center", gap: 10, fontSize: 12.5, padding: "9px 0", borderBottom: "1px solid var(--border)" }}>
                          <span style={{ width: 26, height: 26, borderRadius: 7, display: "grid", placeItems: "center", fontSize: 13, background: `rgba(${rgb},.14)`, flex: "0 0 auto" }}>{ic}</span>
                          <span style={{ flex: 1, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{d.docName || "document"}</span>
                          <span className="note">{d.chunks} chunks</span>
                        </div>); })}
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
                  <div className="note" style={{ marginBottom: 10 }}>Pick a source, then upload files or add a web page — documents accumulate in the knowledge base.</div>
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
