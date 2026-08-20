"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { PROVIDER_CATALOG } from "@/lib/providerCatalog";
import ModelPicker from "@/components/ModelPicker";

const CATALOG = PROVIDER_CATALOG;

type Row = { id: string; provider: string; label: string | null; baseUrl: string; defaultModel: string | null; enabled: boolean; maskedKey: string };
type Msg = { type: "ok" | "err"; text: string } | null;

// per-row editor state
type EditState = { baseUrl: string; apiKey: string; defaultModel: string; models: string[]; busy: boolean; msg: Msg };

function Badges({ p }: { p: { free: boolean; embeddings: boolean } }) {
  return (
    <span style={{ display: "inline-flex", gap: 5, marginLeft: 6 }}>
      {p.free && <span className="badge good" style={{ fontSize: 9 }}>free key</span>}
      <span className="badge" style={{ fontSize: 9, opacity: p.embeddings ? 1 : 0.5 }}>{p.embeddings ? "embeddings ✓" : "no embeddings"}</span>
    </span>
  );
}

// basePath defaults to the admin (global) endpoints; pass "/api/me/providers" for a user's own keys.
export default function ProvidersManager({ initial, basePath = "/api/admin/providers", modelsPath }: { initial: Row[]; basePath?: string; modelsPath?: string }) {
  const router = useRouter();
  const MODELS_PATH = modelsPath || `${basePath}/models`;
  const [provider, setProvider] = useState("groq");
  const [baseUrl, setBaseUrl] = useState(CATALOG.groq.baseUrl);
  const [apiKey, setApiKey] = useState("");
  const [models, setModels] = useState<string[]>([]);
  const [defaultModel, setDefaultModel] = useState(CATALOG.groq?.defaultModels?.[0] || "llama-3.3-70b-versatile");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<Msg>(null);

  const [editId, setEditId] = useState<string | null>(null);
  const [edit, setEdit] = useState<EditState | null>(null);
  const [addOpenKey, setAddOpenKey] = useState(0);   // auto-open add-form picker after Load models
  const [editOpenKey, setEditOpenKey] = useState(0); // auto-open edit picker after Load models

  const meta = (key: string) => CATALOG[key] || { label: key, baseUrl: "", free: false, embeddings: false, keyHint: undefined, defaultModels: [] };

  function pick(p: string) { setProvider(p); setBaseUrl(CATALOG[p].baseUrl); setModels([]); setDefaultModel(CATALOG[p]?.defaultModels?.[0] || ""); setMsg(null); }

  async function loadModels() {
    setBusy(true); setMsg(null);
    try {
      const r = await fetch(MODELS_PATH, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ baseUrl, apiKey, provider }) });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || "failed to load models");
      setModels(j.models); setDefaultModel(j.models[0] || "");
      setAddOpenKey((k) => k + 1);
      setMsg({ type: "ok", text: `Loaded ${j.models.length} models from ${meta(provider).label}` });
    } catch (e) { setMsg({ type: "err", text: (e as Error).message }); }
    finally { setBusy(false); }
  }

  async function save() {
    if (!defaultModel) { setMsg({ type: "err", text: "Pick or type a default model first." }); return; }
    setBusy(true); setMsg(null);
    try {
      const r = await fetch(basePath, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ provider, baseUrl, apiKey, defaultModel, label: meta(provider).label }) });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || "failed to save");
      setApiKey(""); setModels([]); setDefaultModel(""); setMsg({ type: "ok", text: "Provider saved." });
      router.refresh();
    } catch (e) { setMsg({ type: "err", text: (e as Error).message }); }
    finally { setBusy(false); }
  }

  async function del(id: string) {
    if (!confirm("Delete this provider? Any lab using it will fall back to another enabled provider (or TF-IDF).")) return;
    await fetch(`${basePath}/${id}`, { method: "DELETE" });
    if (editId === id) { setEditId(null); setEdit(null); }
    router.refresh();
  }
  async function toggleEnabled(p: Row) { await fetch(`${basePath}/${p.id}`, { method: "PATCH", headers: { "content-type": "application/json" }, body: JSON.stringify({ enabled: !p.enabled }) }); router.refresh(); }

  function openEdit(p: Row) {
    if (editId === p.id) { setEditId(null); setEdit(null); return; }
    setEditId(p.id);
    setEdit({ baseUrl: p.baseUrl, apiKey: "", defaultModel: p.defaultModel || "", models: [], busy: false, msg: null });
  }
  const patchEdit = (v: Partial<EditState>) => setEdit((e) => (e ? { ...e, ...v } : e));

  async function editLoadModels(p: Row) {
    if (!edit) return;
    patchEdit({ busy: true, msg: null });
    try {
      // use the typed key/url if provided, else the stored key via id
      const body = edit.apiKey.trim() ? { baseUrl: edit.baseUrl, apiKey: edit.apiKey.trim(), provider: p.provider } : { id: p.id, baseUrl: edit.baseUrl };
      const r = await fetch(MODELS_PATH, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || "failed to load models");
      patchEdit({ models: j.models, msg: { type: "ok", text: `Loaded ${j.models.length} models` }, defaultModel: edit.defaultModel && j.models.includes(edit.defaultModel) ? edit.defaultModel : (j.models[0] || edit.defaultModel) });
      setEditOpenKey((k) => k + 1);
    } catch (e) { patchEdit({ msg: { type: "err", text: (e as Error).message } }); }
    finally { patchEdit({ busy: false }); }
  }

  async function saveEdit(p: Row) {
    if (!edit) return;
    patchEdit({ busy: true, msg: null });
    try {
      const body: Record<string, unknown> = { baseUrl: edit.baseUrl, defaultModel: edit.defaultModel };
      if (edit.apiKey.trim()) body.apiKey = edit.apiKey.trim();
      const r = await fetch(`${basePath}/${p.id}`, { method: "PATCH", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || "failed to save");
      setEditId(null); setEdit(null);
      router.refresh();
    } catch (e) { patchEdit({ msg: { type: "err", text: (e as Error).message }, busy: false }); }
  }

  const cur = meta(provider);

  return (
    <div className="split col-2">
      <div className="card">
        <div className="card-h"><span className="t">Add a provider</span></div>
        <div className="card-b">
          {msg && <div className={msg.type === "ok" ? "ok" : "err"}>{msg.text}</div>}
          <div className="field"><label className="fld">Provider</label>
            <select value={provider} onChange={(e) => pick(e.target.value)}>
              {Object.entries(CATALOG).map(([k, v]) => <option key={k} value={k}>{v.label}{v.free ? " · free" : ""}</option>)}
            </select>
          </div>
          <div className="note" style={{ margin: "-4px 0 10px", display: "flex", alignItems: "center", gap: 4, flexWrap: "wrap" }}>
            <Badges p={cur} />{cur.keyHint && <span style={{ marginLeft: 6 }}>key: <code>{cur.keyHint}</code></span>}
          </div>
          <div className="field"><label className="fld">Base URL</label><input type="text" value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} placeholder="https://…/v1" /></div>
          <div className="field"><label className="fld">API key {provider === "ollama" && "(usually blank for local Ollama)"}</label><input type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} placeholder="Paste the API key" /></div>
          <div className="row" style={{ margin: "4px 0 12px" }}><button className="btn ghost sm" onClick={loadModels} disabled={busy}>{busy ? "Loading…" : "Load models"}</button><span className="note">Fetches the real model list from the provider</span></div>
          <div className="field"><label className="fld">Default model {(models.length > 0 || cur.defaultModels.length > 0) && <span className="note">· {models.length || cur.defaultModels.length} available · click or type to search</span>}</label>
            <ModelPicker models={models} presetModels={cur.defaultModels || []} value={defaultModel} onChange={setDefaultModel} placeholder="Select a model, or type a custom model id" openKey={addOpenKey} />
          </div>
          <button className="btn" onClick={save} disabled={busy}>Save provider</button>
        </div>
      </div>

      <div className="card">
        <div className="card-h"><span className="t">Configured providers</span><span className="mono r">{initial.length}</span></div>
        <div className="card-b">
          {initial.length === 0 && <div className="note">No providers yet — add one on the left.</div>}
          {initial.map((p) => {
            const m = meta(p.provider);
            const isEdit = editId === p.id;
            return (
              <div key={p.id} style={{ border: `1px solid ${isEdit ? "var(--accent)" : "var(--border)"}`, borderRadius: "var(--rs)", padding: "12px 14px", marginBottom: 10 }}>
                <div className="row" style={{ justifyContent: "space-between", alignItems: "center" }}>
                  <div><b>{p.label || p.provider}</b> <span className={`badge ${p.enabled ? "good" : ""}`}>{p.enabled ? "enabled" : "disabled"}</span><Badges p={m} /></div>
                  <div className="row" style={{ gap: 6 }}>
                    <button className="btn ghost sm" onClick={() => openEdit(p)}>{isEdit ? "Close" : "View / edit"}</button>
                    <button className="btn ghost sm" onClick={() => toggleEnabled(p)}>{p.enabled ? "Disable" : "Enable"}</button>
                    <button className="btn ghost sm" onClick={() => del(p.id)}>Delete</button>
                  </div>
                </div>
                <div className="note" style={{ marginTop: 6 }}>model: {p.defaultModel || "—"} · key: {p.maskedKey || "none"}</div>
                <div className="note">{p.baseUrl}</div>

                {isEdit && edit && (
                  <div style={{ marginTop: 12, paddingTop: 12, borderTop: "1px solid var(--border)" }}>
                    {edit.msg && <div className={edit.msg.type === "ok" ? "ok" : "err"} style={{ marginBottom: 8 }}>{edit.msg.text}</div>}
                    <div className="field"><label className="fld">Base URL</label><input type="text" value={edit.baseUrl} onChange={(e) => patchEdit({ baseUrl: e.target.value })} placeholder="https://…/v1" /></div>
                    <div className="field"><label className="fld">API key</label><input type="password" value={edit.apiKey} onChange={(e) => patchEdit({ apiKey: e.target.value })} placeholder={p.maskedKey ? `current: ${p.maskedKey} — leave blank to keep` : "no key set — paste one"} /><span className="note">Stored AES-256-GCM encrypted; the real key is never sent back to the browser.</span></div>
                    <div className="row" style={{ margin: "4px 0 12px" }}><button className="btn ghost sm" onClick={() => editLoadModels(p)} disabled={edit.busy}>{edit.busy ? "Loading…" : "↻ Load models"}</button><span className="note">Uses the saved key (or the new one above)</span></div>
                    <div className="field"><label className="fld">Default model {(edit.models.length > 0 || m.defaultModels.length > 0) && <span className="note">· {edit.models.length || m.defaultModels.length} available · click or type to search</span>}</label>
                      <ModelPicker models={edit.models} presetModels={m.defaultModels || []} value={edit.defaultModel} onChange={(v) => patchEdit({ defaultModel: v })} placeholder="Select a model, or type a custom model id" openKey={editOpenKey} />
                    </div>
                    <div className="row" style={{ gap: 8 }}>
                      <button className="btn" onClick={() => saveEdit(p)} disabled={edit.busy}>Save changes</button>
                      <button className="btn ghost sm" onClick={() => { setEditId(null); setEdit(null); }}>Cancel</button>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
