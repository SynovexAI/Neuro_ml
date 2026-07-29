"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

const CATALOG: Record<string, { label: string; baseUrl: string }> = {
  groq: { label: "Groq", baseUrl: "https://api.groq.com/openai/v1" },
  cerebras: { label: "Cerebras", baseUrl: "https://api.cerebras.ai/v1" },
  gemini: { label: "Google Gemini", baseUrl: "https://generativelanguage.googleapis.com/v1beta/openai" },
  ollama: { label: "Ollama (local)", baseUrl: "http://localhost:11434/v1" },
  custom: { label: "Custom (OpenAI-compatible)", baseUrl: "" },
};

type Row = { id: string; provider: string; label: string | null; baseUrl: string; defaultModel: string | null; enabled: boolean; maskedKey: string };
type Msg = { type: "ok" | "err"; text: string } | null;

export default function ProvidersManager({ initial }: { initial: Row[] }) {
  const router = useRouter();
  const [provider, setProvider] = useState("groq");
  const [baseUrl, setBaseUrl] = useState(CATALOG.groq.baseUrl);
  const [apiKey, setApiKey] = useState("");
  const [models, setModels] = useState<string[]>([]);
  const [defaultModel, setDefaultModel] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<Msg>(null);

  function pick(p: string) { setProvider(p); setBaseUrl(CATALOG[p].baseUrl); setModels([]); setDefaultModel(""); setMsg(null); }

  async function loadModels() {
    setBusy(true); setMsg(null);
    try {
      const r = await fetch("/api/admin/providers/models", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ baseUrl, apiKey, provider }) });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || "failed to load models");
      setModels(j.models); setDefaultModel(j.models[0] || "");
      setMsg({ type: "ok", text: `Loaded ${j.models.length} models from ${CATALOG[provider]?.label || provider}` });
    } catch (e) { setMsg({ type: "err", text: (e as Error).message }); }
    finally { setBusy(false); }
  }

  async function save() {
    if (!defaultModel) { setMsg({ type: "err", text: "Pick or type a default model first." }); return; }
    setBusy(true); setMsg(null);
    try {
      const r = await fetch("/api/admin/providers", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ provider, baseUrl, apiKey, defaultModel, label: CATALOG[provider]?.label }) });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || "failed to save");
      setApiKey(""); setModels([]); setDefaultModel(""); setMsg({ type: "ok", text: "Provider saved and available platform-wide." });
      router.refresh();
    } catch (e) { setMsg({ type: "err", text: (e as Error).message }); }
    finally { setBusy(false); }
  }

  async function patch(id: string, body: object) { await fetch(`/api/admin/providers/${id}`, { method: "PATCH", headers: { "content-type": "application/json" }, body: JSON.stringify(body) }); router.refresh(); }
  async function del(id: string) { await fetch(`/api/admin/providers/${id}`, { method: "DELETE" }); router.refresh(); }

  return (
    <div className="split col-2">
      <div className="card">
        <div className="card-h"><span className="t">Add / configure a provider</span></div>
        <div className="card-b">
          {msg && <div className={msg.type === "ok" ? "ok" : "err"}>{msg.text}</div>}
          <div className="field"><label className="fld">Provider</label>
            <select value={provider} onChange={(e) => pick(e.target.value)}>
              {Object.entries(CATALOG).map(([k, v]) => <option key={k} value={k}>{v.label}</option>)}
            </select>
          </div>
          <div className="field"><label className="fld">Base URL</label><input type="text" value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} placeholder="https://…/v1" /></div>
          <div className="field"><label className="fld">API key {provider === "ollama" && "(usually blank for local Ollama)"}</label><input type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} placeholder="Paste the API key" /></div>
          <div className="row" style={{ margin: "4px 0 12px" }}><button className="btn ghost sm" onClick={loadModels} disabled={busy}>{busy ? "Loading…" : "Load models"}</button><span className="note">Fetches the real model list from the provider</span></div>
          <div className="field"><label className="fld">Default model</label>
            {models.length > 0
              ? <select value={defaultModel} onChange={(e) => setDefaultModel(e.target.value)}>{models.map((m) => <option key={m} value={m}>{m}</option>)}</select>
              : <input type="text" value={defaultModel} onChange={(e) => setDefaultModel(e.target.value)} placeholder="Load models, or type a model id" />}
          </div>
          <button className="btn" onClick={save} disabled={busy}>Save provider</button>
        </div>
      </div>

      <div className="card">
        <div className="card-h"><span className="t">Configured providers</span><span className="mono r">{initial.length}</span></div>
        <div className="card-b">
          {initial.length === 0 && <div className="note">No providers yet — add one on the left.</div>}
          {initial.map((p) => (
            <div key={p.id} style={{ border: "1px solid var(--border)", borderRadius: "var(--rs)", padding: "12px 14px", marginBottom: 10 }}>
              <div className="row" style={{ justifyContent: "space-between" }}>
                <div><b>{p.label || p.provider}</b> <span className={`badge ${p.enabled ? "good" : ""}`}>{p.enabled ? "enabled" : "disabled"}</span></div>
                <div className="row">
                  <button className="btn ghost sm" onClick={() => patch(p.id, { enabled: !p.enabled })}>{p.enabled ? "Disable" : "Enable"}</button>
                  <button className="btn ghost sm" onClick={() => del(p.id)}>Delete</button>
                </div>
              </div>
              <div className="note" style={{ marginTop: 6 }}>model: {p.defaultModel || "—"} · key: {p.maskedKey || "none"}</div>
              <div className="note">{p.baseUrl}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
