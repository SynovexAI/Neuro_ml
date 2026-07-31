"use client";

import { useEffect, useState } from "react";

type ProviderOpt = { id: string; provider: string; label: string | null };
type NatResult = { answer: string; latency_ms: number; model: string; tool_names: string[]; unsupported_tools: string[]; profiler?: { total_ms?: number } };

// Tools NAT core supports today; others are wired in later phases.
const SUPPORTED = [
  { id: "calculator", label: "calculator" },
  { id: "current_datetime", label: "current_datetime" },
];
const COMING = ["http", "knowledge (RAG)", "MCP tools"];

export default function NatAgentPanel() {
  const [providers, setProviders] = useState<ProviderOpt[]>([]);
  const [providerId, setProviderId] = useState("");
  const [models, setModels] = useState<string[]>([]);
  const [model, setModel] = useState("");
  const [loadingModels, setLoadingModels] = useState(false);

  const [tools, setTools] = useState<Set<string>>(new Set(["calculator", "current_datetime"]));
  const [systemPrompt, setSystemPrompt] = useState("You are a careful reasoning agent. Use a tool when a calculation or the current date is needed.");
  const [task, setTask] = useState("What is 18% of 2450, and what is today's date?");

  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<NatResult | null>(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    fetch("/api/models").then((r) => r.json()).then((j) => {
      setProviders(j.providers || []);
      const pid = j.providerId || j.providers?.[0]?.id || "";
      setProviderId(pid);
      setModels(j.models || []);
      setModel(j.default || j.models?.[0] || "");
    }).catch(() => {});
  }, []);

  async function onProvider(id: string) {
    setProviderId(id); setModels([]); setModel(""); setLoadingModels(true);
    try {
      const j = await fetch(`/api/models?providerId=${encodeURIComponent(id)}`).then((r) => r.json());
      setModels(j.models || []); setModel(j.default || j.models?.[0] || "");
    } catch { /* leave empty */ }
    setLoadingModels(false);
  }

  function toggleTool(id: string) {
    setTools((s) => { const n = new Set(s); if (n.has(id)) n.delete(id); else n.add(id); return n; });
  }

  async function run() {
    if (!task.trim()) { setErr("Enter a task for the agent."); return; }
    setRunning(true); setErr(""); setResult(null);
    try {
      const res = await fetch("/api/agent/nat-run", {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ task, providerId, model, systemPrompt, tools: [...tools], temperature: 0 }),
      });
      const j = await res.json().catch(() => null);
      if (!res.ok) { setErr(j?.error || `Run failed (${res.status})`); return; }
      setResult(j as NatResult);
    } catch (e) { setErr((e as Error).message); }
    finally { setRunning(false); }
  }

  return (
    <>
      <div className="teach-note"><span className="ic">⚡</span><span><b>NAT runtime.</b> This agent runs server-side through the real <b>NVIDIA NeMo Agent Toolkit</b>, not in your browser — so you get its production profiler. Needs the NAT sidecar deployed (<code>NAT_SERVICE_URL</code>).</span></div>

      <div className="split" style={{ gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <div className="card">
          <div className="card-h"><span className="t">Build</span></div>
          <div className="card-b">
            <label className="fld">Provider</label>
            <select value={providerId} onChange={(e) => onProvider(e.target.value)}>
              {providers.length === 0 && <option value="">No provider configured</option>}
              {providers.map((p) => <option key={p.id} value={p.id}>{p.label || p.provider}</option>)}
            </select>

            <label className="fld" style={{ marginTop: 12 }}>Model</label>
            <select value={model} onChange={(e) => setModel(e.target.value)} disabled={loadingModels}>
              {loadingModels && <option>loading…</option>}
              {!loadingModels && models.length === 0 && <option value="">no models</option>}
              {models.map((m) => <option key={m} value={m}>{m}</option>)}
            </select>

            <label className="fld" style={{ marginTop: 12 }}>Tools</label>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 7 }}>
              {SUPPORTED.map((t) => (
                <button key={t.id} type="button" onClick={() => toggleTool(t.id)}
                  className={`chip ${tools.has(t.id) ? "on" : ""}`}>{tools.has(t.id) ? "✓ " : ""}{t.label}</button>
              ))}
              {COMING.map((c) => <span key={c} className="chip" style={{ opacity: 0.5 }} title="coming in a later phase">{c} · soon</span>)}
            </div>

            <label className="fld" style={{ marginTop: 12 }}>System prompt</label>
            <textarea value={systemPrompt} onChange={(e) => setSystemPrompt(e.target.value)} rows={3} />

            <label className="fld" style={{ marginTop: 12 }}>Task</label>
            <textarea value={task} onChange={(e) => setTask(e.target.value)} rows={2} />

            <button className="btn block" style={{ marginTop: 14 }} onClick={run} disabled={running || !providerId || !model}>
              {running ? "Running via NAT…" : "▶ Run via NAT"}
            </button>
            {err && <div className="err" style={{ marginTop: 10 }}>{err}</div>}
          </div>
        </div>

        <div className="card">
          <div className="card-h"><span className="t">NAT profiler report</span>{result && <span className="mono r" style={{ color: "var(--good, #3b9e5f)" }}>success</span>}</div>
          <div className="card-b">
            {!result && !running && <div className="note">Run the agent to see its answer and profiler.</div>}
            {running && <div className="note">Waiting for the NAT service…</div>}
            {result && (
              <>
                <div className="cv-summary" style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 14 }}>
                  <div className="metric"><span className="v">{((result.profiler?.total_ms ?? result.latency_ms) / 1000).toFixed(2)}s</span><span className="k">Total latency</span></div>
                  <div className="metric"><span className="v">{result.tool_names.length}</span><span className="k">Tools wired</span></div>
                  <div className="metric"><span className="v">{result.model}</span><span className="k">Model</span></div>
                </div>
                <label className="fld">Answer</label>
                <div style={{ background: "var(--panel)", border: "1px solid var(--border)", borderRadius: 7, padding: "10px 12px", fontSize: 13, lineHeight: 1.55, whiteSpace: "pre-wrap" }}>{result.answer}</div>
                <div className="note" style={{ marginTop: 10 }}>tools: {result.tool_names.join(", ") || "none"}{result.unsupported_tools.length ? ` · not yet supported: ${result.unsupported_tools.join(", ")}` : ""}</div>
                <div className="note" style={{ marginTop: 4 }}>per-step latency/token profiler arrives in phase 2.</div>
              </>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
