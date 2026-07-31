"use client";

import { useEffect, useState } from "react";

type ProviderOpt = { id: string; provider: string; label: string | null };
type McpOpt = { id: string; name: string; transport: string };
type KbOpt = { id: string; name: string; status: string; chunkCount: number };
type Step = { name: string; type: string; ms: number | null; tokens: number };
type NatResult = { answer: string; latency_ms: number; model: string; tool_names: string[]; unsupported_tools: string[]; context_used?: boolean; profiler?: { total_ms?: number; steps?: Step[] } };

const SUPPORTED = [{ id: "calculator", label: "calculator" }, { id: "current_datetime", label: "current_datetime" }];

export default function NatAgentPanel() {
  const [providers, setProviders] = useState<ProviderOpt[]>([]);
  const [providerId, setProviderId] = useState("");
  const [models, setModels] = useState<string[]>([]);
  const [model, setModel] = useState("");
  const [loadingModels, setLoadingModels] = useState(false);
  const [mcp, setMcp] = useState<McpOpt[]>([]);

  const [tools, setTools] = useState<Set<string>>(new Set(["calculator", "current_datetime"]));
  const [mcpIds, setMcpIds] = useState<Set<string>>(new Set());
  const [kbs, setKbs] = useState<KbOpt[]>([]);
  const [kbIds, setKbIds] = useState<Set<string>>(new Set());
  const [systemPrompt, setSystemPrompt] = useState("You are a careful reasoning agent. Use a tool when a calculation, lookup, or the current date is needed.");
  const [task, setTask] = useState("What is 18% of 2450, and what is today's date?");

  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<NatResult | null>(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    fetch("/api/models").then((r) => r.json()).then((j) => {
      setProviders(j.providers || []);
      setProviderId(j.providerId || j.providers?.[0]?.id || "");
      setModels(j.models || []); setModel(j.default || j.models?.[0] || "");
    }).catch(() => {});
    fetch("/api/agent/mcp").then((r) => r.json()).then((j) => setMcp(j.servers || [])).catch(() => {});
    fetch("/api/kb").then((r) => r.json()).then((j) => setKbs((j.kbs || []).filter((k: KbOpt) => k.status === "ready"))).catch(() => {});
  }, []);

  async function onProvider(id: string) {
    setProviderId(id); setModels([]); setModel(""); setLoadingModels(true);
    try { const j = await fetch(`/api/models?providerId=${encodeURIComponent(id)}`).then((r) => r.json()); setModels(j.models || []); setModel(j.default || j.models?.[0] || ""); }
    catch { /* leave empty */ }
    setLoadingModels(false);
  }
  const toggle = (set: React.Dispatch<React.SetStateAction<Set<string>>>, id: string) =>
    set((s) => { const n = new Set(s); if (n.has(id)) n.delete(id); else n.add(id); return n; });

  async function run() {
    if (!task.trim()) { setErr("Enter a task for the agent."); return; }
    setRunning(true); setErr(""); setResult(null);
    try {
      const res = await fetch("/api/agent/nat-run", {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ task, providerId, model, systemPrompt, tools: [...tools], temperature: 0, mcpServerIds: [...mcpIds], knowledgeBaseIds: [...kbIds] }),
      });
      const j = await res.json().catch(() => null);
      if (!res.ok) { setErr(j?.error || `Run failed (${res.status})`); return; }
      setResult(j as NatResult);
    } catch (e) { setErr((e as Error).message); }
    finally { setRunning(false); }
  }

  const steps = result?.profiler?.steps || [];
  const maxMs = Math.max(1, ...steps.map((s) => s.ms || 0));

  return (
    <>
      <div className="teach-note"><span className="ic">⚡</span><span><b>NAT runtime.</b> Runs server-side through the real <b>NVIDIA NeMo Agent Toolkit</b> — with MCP tools, document grounding, and a per-step profiler. Needs the NAT sidecar deployed (<code>NAT_SERVICE_URL</code>).</span></div>

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

            <label className="fld" style={{ marginTop: 12 }}>Built-in tools</label>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 7 }}>
              {SUPPORTED.map((t) => <button key={t.id} type="button" onClick={() => toggle(setTools, t.id)} className={`chip ${tools.has(t.id) ? "on" : ""}`}>{tools.has(t.id) ? "✓ " : ""}{t.label}</button>)}
            </div>

            <label className="fld" style={{ marginTop: 12 }}>MCP tools <span className="note">· from Admin → MCP servers</span></label>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 7 }}>
              {mcp.length === 0 ? <span className="note">no enabled MCP servers</span>
                : mcp.map((s) => <button key={s.id} type="button" onClick={() => toggle(setMcpIds, s.id)} className={`chip ${mcpIds.has(s.id) ? "on" : ""}`} title={s.transport}>{mcpIds.has(s.id) ? "✓ " : ""}{s.name} · MCP</button>)}
            </div>

            <label className="fld" style={{ marginTop: 12 }}>Knowledge bases (RAG) <span className="note">· from Studio → Knowledge bases</span></label>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 7 }}>
              {kbs.length === 0 ? <span className="note">no synced knowledge bases — <a href="/kb">create one</a></span>
                : kbs.map((k) => <button key={k.id} type="button" onClick={() => toggle(setKbIds, k.id)} className={`chip ${kbIds.has(k.id) ? "on" : ""}`} title={`${k.chunkCount} chunks`}>{kbIds.has(k.id) ? "✓ " : ""}{k.name}</button>)}
            </div>

            <label className="fld" style={{ marginTop: 12 }}>System prompt</label>
            <textarea value={systemPrompt} onChange={(e) => setSystemPrompt(e.target.value)} rows={2} />
            <label className="fld" style={{ marginTop: 12 }}>Task</label>
            <textarea value={task} onChange={(e) => setTask(e.target.value)} rows={2} />

            <button className="btn block" style={{ marginTop: 14 }} onClick={run} disabled={running || !providerId || !model}>{running ? "Running via NAT…" : "▶ Run via NAT"}</button>
            {err && <div className="err" style={{ marginTop: 10 }}>{err}</div>}
          </div>
        </div>

        <div className="card">
          <div className="card-h"><span className="t">NAT profiler report</span>{result && <span className="mono r" style={{ color: "#3b9e5f" }}>success</span>}</div>
          <div className="card-b">
            {!result && !running && <div className="note">Run the agent to see its answer and profiler.</div>}
            {running && <div className="note">Waiting for the NAT service…</div>}
            {result && (
              <>
                <div className="cv-summary" style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 14 }}>
                  <div className="metric"><span className="v">{((result.profiler?.total_ms ?? result.latency_ms) / 1000).toFixed(2)}s</span><span className="k">Total latency</span></div>
                  <div className="metric"><span className="v">{result.tool_names.length}</span><span className="k">Tools</span></div>
                  {result.context_used && <div className="metric"><span className="v">RAG</span><span className="k">Grounded</span></div>}
                </div>

                {steps.length > 0 && (
                  <>
                    <label className="fld">Step timeline</label>
                    <div style={{ display: "flex", flexDirection: "column", gap: 6, marginBottom: 12 }}>
                      {steps.map((s, i) => {
                        const bottleneck = (s.ms || 0) === maxMs && maxMs > 1;
                        return (
                          <div key={i} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                            <span style={{ width: 110, fontSize: 11, color: "var(--muted)", flex: "0 0 auto", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{s.type || s.name}</span>
                            <div style={{ flex: 1, background: "var(--panel-2)", borderRadius: 4, height: 16, position: "relative" }}>
                              <div style={{ width: `${Math.max(3, ((s.ms || 0) / maxMs) * 100)}%`, height: "100%", background: bottleneck ? "#f59e0b" : "var(--accent)", borderRadius: 4 }} />
                              <span style={{ position: "absolute", right: 6, top: 0, lineHeight: "16px", fontSize: 10, color: "var(--muted)", fontFamily: "var(--mono)" }}>{s.ms != null ? `${s.ms}ms` : ""}{s.tokens ? ` · ${s.tokens}t` : ""}{bottleneck ? " · slow" : ""}</span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </>
                )}

                <label className="fld">Answer</label>
                <div style={{ background: "var(--panel)", border: "1px solid var(--border)", borderRadius: 7, padding: "10px 12px", fontSize: 13, lineHeight: 1.55, whiteSpace: "pre-wrap" }}>{result.answer}</div>
                <div className="note" style={{ marginTop: 10 }}>tools: {result.tool_names.join(", ") || "none"}{result.unsupported_tools.length ? ` · unsupported: ${result.unsupported_tools.join(", ")}` : ""}</div>
                {steps.length === 0 && <div className="note" style={{ marginTop: 4 }}>per-step profiler data wasn&apos;t returned for this run (total latency shown).</div>}
              </>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
