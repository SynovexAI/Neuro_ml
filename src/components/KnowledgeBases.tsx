"use client";

import { useEffect, useRef, useState } from "react";

type Kb = { id: string; name: string; status: string; docCount: number; chunkCount: number; embModel: string | null; updatedAt?: string | null };
type Staged = { name: string; text: string };

export default function KnowledgeBases() {
  const [kbs, setKbs] = useState<Kb[]>([]);
  const [selId, setSelId] = useState<string | null>(null);
  const [staged, setStaged] = useState<Staged[]>([]);
  const [newName, setNewName] = useState("");
  const [url, setUrl] = useState("https://en.wikipedia.org/wiki/Retrieval-augmented_generation");
  const [busy, setBusy] = useState("");
  const [msg, setMsg] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  async function loadKbs() {
    try { const j = await fetch("/api/kb").then((r) => r.json()); setKbs(j.kbs || []); } catch { setKbs([]); }
  }
  useEffect(() => { loadKbs(); }, []);

  const sel = kbs.find((k) => k.id === selId) || null;

  async function createKb() {
    if (!newName.trim()) return;
    setBusy("create");
    const r = await fetch("/api/kb", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ name: newName }) });
    const j = await r.json().catch(() => ({}));
    setNewName(""); await loadKbs();
    if (j.id) { setSelId(j.id); setStaged([]); }
    setBusy("");
  }
  async function removeKb(k: Kb) {
    if (!window.confirm(`Delete knowledge base "${k.name}"?`)) return;
    await fetch(`/api/kb/${k.id}`, { method: "DELETE" }).catch(() => {});
    if (selId === k.id) { setSelId(null); setStaged([]); }
    loadKbs();
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
      const r = await fetch(`/api/kb/${selId}/sync`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ docs: staged }) });
      const j = await r.json().catch(() => ({}));
      if (!r.ok) { setMsg(j.error || "Sync failed."); }
      else { setMsg(`Synced ✓ ${j.chunkCount} chunks · ${j.embModel === "tfidf" ? "TF-IDF (provider has no embeddings)" : "embeddings: " + j.embModel}`); setStaged([]); await loadKbs(); }
    } catch (e) { setMsg((e as Error).message); }
    setBusy("");
  }

  const badge = (s: string) => s === "ready" ? "badge good" : s === "syncing" ? "badge warn" : s === "error" ? "badge" : "badge";

  return (
    <div className="split" style={{ gridTemplateColumns: "300px 1fr", gap: 16 }}>
      <div className="card">
        <div className="card-h"><span className="t">Your knowledge bases</span><span className="mono r">{kbs.length}</span></div>
        <div className="card-b">
          <div className="row" style={{ gap: 8 }}>
            <input type="text" placeholder="new KB name" value={newName} onChange={(e) => setNewName(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") createKb(); }} />
            <button className="btn sm" onClick={createKb} disabled={busy === "create"}>+ New</button>
          </div>
          <div style={{ marginTop: 12 }}>
            {kbs.length === 0 ? <div className="note">No knowledge bases yet. Create one, add files or URLs, and sync.</div>
              : kbs.map((k) => (
                <div key={k.id} onClick={() => { setSelId(k.id); setStaged([]); setMsg(""); }}
                  style={{ padding: "9px 10px", borderRadius: 7, cursor: "pointer", marginBottom: 6, border: "1px solid var(--border)", background: selId === k.id ? "var(--panel)" : "transparent" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <b style={{ fontSize: 13 }}>{k.name}</b><span className={badge(k.status)} style={{ marginLeft: "auto" }}>{k.status}</span>
                  </div>
                  <div className="note" style={{ marginTop: 3 }}>{k.docCount} docs · {k.chunkCount} chunks{k.embModel ? ` · ${k.embModel === "tfidf" ? "TF-IDF" : k.embModel}` : ""}</div>
                </div>
              ))}
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-h"><span className="t">{sel ? sel.name : "Select or create a knowledge base"}</span>{sel && <button className="btn ghost sm danger r" onClick={() => removeKb(sel)}>Delete</button>}</div>
        <div className="card-b">
          {!sel ? <div className="note">Pick a knowledge base on the left, or create one, to add documents.</div> : (
            <>
              <label className="fld">Add documents</label>
              <div className="row" style={{ gap: 8, flexWrap: "wrap" }}>
                <button className="btn ghost sm" onClick={() => fileRef.current?.click()} disabled={busy === "extract"}>{busy === "extract" ? "Reading…" : "Upload files (pdf/docx/xlsx/csv/txt)"}</button>
                <input ref={fileRef} type="file" multiple accept=".txt,.md,.csv,.json,.pdf,.docx,.doc,.xlsx,.xls" onChange={(e) => addFiles(e.target.files)} style={{ display: "none" }} />
              </div>
              <div className="row" style={{ gap: 8, marginTop: 8 }}>
                <input type="text" placeholder="https://…" value={url} onChange={(e) => setUrl(e.target.value)} />
                <button className="btn ghost sm" onClick={addUrl} disabled={busy === "url"}>{busy === "url" ? "Fetching…" : "Add URL"}</button>
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
                  <button className="btn block" style={{ marginTop: 12 }} onClick={sync} disabled={busy === "sync"}>{busy === "sync" ? "Syncing → embedding + storing vectors…" : "⟳ Sync to vector store"}</button>
                </div>
              )}

              {sel.status === "ready" && staged.length === 0 && <div className="note" style={{ marginTop: 12 }}>Synced: {sel.docCount} docs · {sel.chunkCount} chunks · {sel.embModel === "tfidf" ? "TF-IDF vectors" : `embeddings (${sel.embModel})`}. Re-syncing replaces the contents.</div>}
              {msg && <div className={msg.startsWith("Synced") ? "note" : "err"} style={{ marginTop: 10 }}>{msg}</div>}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
