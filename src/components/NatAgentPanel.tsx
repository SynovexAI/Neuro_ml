"use client";

import { useEffect, useState } from "react";

type ProviderOpt = { id: string; provider: string; label: string | null };
type McpOpt = { id: string; name: string; transport: string };
type KbOpt = { id: string; name: string; status: string; chunkCount: number };
type Step = { name: string; type: string; ms: number | null; tokens: number };
type NatResult = { answer: string; latency_ms: number; model: string; tool_names: string[]; unsupported_tools: string[]; context_used?: boolean; profiler?: { total_ms?: number; steps?: Step[] } };

const SUPPORTED = [
  { id: "calculator", label: "Calculator", icon: "🧮", sub: "arithmetic" },
  { id: "current_datetime", label: "Date & time", icon: "🕐", sub: "current date" },
];

export default function NatAgentPanel() {
  const [providers, setProviders] = useState<ProviderOpt[]>([]);
  const [providerId, setProviderId] = useState("");
  const [models, setModels] = useState<string[]>([]);
  const [model, setModel] = useState("");
  const [loadingModels, setLoadingModels] = useState(false);
  const [mcp, setMcp] = useState<McpOpt[]>([]);
  const [kbs, setKbs] = useState<KbOpt[]>([]);

  const [agentType, setAgentType] = useState<"react_agent" | "tool_calling_agent">("react_agent");
  const [tools, setTools] = useState<Set<string>>(new Set(["calculator", "current_datetime"]));
  const [mcpIds, setMcpIds] = useState<Set<string>>(new Set());
  const [kbIds, setKbIds] = useState<Set<string>>(new Set());
  const [temperature, setTemperature] = useState(0.2);
  const [systemPrompt, setSystemPrompt] = useState("You are a helpful agent. Use a tool when a calculation, lookup, or the current date is needed, and answer from the attached knowledge when relevant.");
  const [task, setTask] = useState("What is 18% of 2450, and what is today's date?");

  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<NatResult | null>(null);
  const [err, setErr] = useState("");
  const [picker, setPicker] = useState<"tool" | "kb" | null>(null);

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
        body: JSON.stringify({ task, providerId, model, systemPrompt, agentType, tools: [...tools], temperature, mcpServerIds: [...mcpIds], knowledgeBaseIds: [...kbIds] }),
      });
      const j = await res.json().catch(() => null);
      if (!res.ok) { setErr(j?.error || `Run failed (${res.status})`); return; }
      setResult(j as NatResult);
    } catch (e) { setErr((e as Error).message); }
    finally { setRunning(false); }
  }

  const steps = result?.profiler?.steps || [];
  const maxMs = Math.max(1, ...steps.map((s) => s.ms || 0));

  // attached tool cards (built-in + MCP)
  const toolCards = [
    ...[...tools].map((t) => { const m = SUPPORTED.find((x) => x.id === t); return { id: `tool:${t}`, icon: m?.icon || "🔧", label: m?.label || t, sub: m?.sub || "tool" }; }),
    ...[...mcpIds].map((id) => ({ id: `mcp:${id}`, icon: "🔌", label: mcp.find((m) => m.id === id)?.name || "mcp", sub: "MCP" })),
  ];
  const kbCards = [...kbIds].map((id) => { const k = kbs.find((x) => x.id === id); return { id: `kb:${id}`, label: k?.name || "knowledge", sub: `${k?.chunkCount ?? 0} chunks` }; });
  function detach(id: string) {
    if (id.startsWith("tool:")) toggle(setTools, id.slice(5));
    else if (id.startsWith("mcp:")) toggle(setMcpIds, id.slice(4));
    else if (id.startsWith("kb:")) toggle(setKbIds, id.slice(3));
  }

  return (
    <>
      <div className="teach-note"><span className="ic">⚡</span><span><b>NAT runtime.</b> Runs server-side through the real <b>NVIDIA NeMo Agent Toolkit</b>. Needs the NAT sidecar deployed (<code>NAT_SERVICE_URL</code>).</span></div>

      <div className="flowstrip">
        <span className="fs-node"><span>💬</span>Input</span>
        <span className="fs-arrow">→</span>
        <span className="fs-agent"><span>🤖</span>Agent</span>
        <span className="fs-arrow">→</span>
        <span className="fs-node"><span>✅</span>Answer</span>
        <span style={{ marginLeft: "auto", fontSize: 11, color: "var(--faint)" }}>how it runs</span>
      </div>

      <div className="card" style={{ marginBottom: 16 }}>
        <div className="card-h"><span className="fs-agent" style={{ padding: "3px 9px" }}><span>🤖</span>Agent</span><span className="note">{agentType === "react_agent" ? "ReAct — reasons and calls tools on its own" : "Tool-calling — native function calls"}</span><span style={{ flex: 1 }} /><button className="btn sm" onClick={run} disabled={running || !providerId || !model}>{running ? "Running…" : "▶ Run"}</button></div>
        <div className="card-b">
          <div className="split col-2e">
            <div><label className="fld">Provider</label><select value={providerId} onChange={(e) => onProvider(e.target.value)}>{providers.length === 0 && <option value="">No provider configured</option>}{providers.map((p) => <option key={p.id} value={p.id}>{p.label || p.provider}</option>)}</select></div>
            <div><label className="fld">Model</label><select value={model} onChange={(e) => setModel(e.target.value)} disabled={loadingModels}>{loadingModels && <option>loading…</option>}{!loadingModels && models.length === 0 && <option value="">no models</option>}{models.map((m) => <option key={m} value={m}>{m}</option>)}</select></div>
          </div>
          <div className="split col-2e" style={{ marginTop: 12 }}>
            <div><label className="fld">Agent type</label><div className="seg"><button className={agentType === "react_agent" ? "on" : ""} onClick={() => setAgentType("react_agent")}>ReAct</button><button className={agentType === "tool_calling_agent" ? "on" : ""} onClick={() => setAgentType("tool_calling_agent")}>Tool-calling</button></div></div>
            <div><label className="fld">Temperature · {temperature.toFixed(2)}</label><input type="range" min={0} max={1} step={0.05} value={temperature} onChange={(e) => setTemperature(+e.target.value)} /></div>
          </div>
          <label className="fld" style={{ marginTop: 12 }}>Instructions</label>
          <textarea rows={3} value={systemPrompt} onChange={(e) => setSystemPrompt(e.target.value)} />
        </div>
      </div>

      <div className="sec-head"><b>Tools</b><span className="note">the agent calls these when it decides to</span></div>
      <div className="tool-grid">
        {toolCards.map((t) => (
          <div key={t.id} className="tool-card">
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}><span style={{ fontSize: 15 }}>{t.icon}</span><b>{t.label}</b>{t.sub === "MCP" && <span className="badge" style={{ marginLeft: "auto" }}>MCP</span>}</div>
            {t.sub !== "MCP" && <div className="note" style={{ marginTop: 3 }}>{t.sub}</div>}
            <button className="tc-x" title="Remove" onClick={() => detach(t.id)}>×</button>
          </div>
        ))}
        <button className="tool-card add" onClick={() => setPicker("tool")}>＋ Add tool</button>
      </div>

      <div className="sec-head"><b>Knowledge</b><span className="note">grounds answers on your documents (RAG)</span></div>
      <div className="tool-grid">
        {kbCards.map((k) => (
          <div key={k.id} className="tool-card">
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}><span style={{ fontSize: 15 }}>📚</span><b>{k.label}</b></div>
            <div className="note" style={{ marginTop: 3 }}>{k.sub}</div>
            <button className="tc-x" title="Detach" onClick={() => detach(k.id)}>×</button>
          </div>
        ))}
        <button className="tool-card add" onClick={() => setPicker("kb")}>＋ Attach knowledge</button>
      </div>

      <div className="card" style={{ marginBottom: 16 }}>
        <div className="card-h"><span className="t">Task</span></div>
        <div className="card-b">
          <textarea rows={2} value={task} onChange={(e) => setTask(e.target.value)} />
          <button className="btn block" style={{ marginTop: 12 }} onClick={run} disabled={running || !providerId || !model}>{running ? "Running via NAT…" : "▶ Run via NAT"}</button>
          {err && <div className="err" style={{ marginTop: 10 }}>{err}</div>}
        </div>
      </div>

      <div className="card">
        <div className="card-h"><span className="t">NAT profiler report</span>{result && <span className="mono r" style={{ color: "#3b9e5f" }}>success</span>}</div>
        <div className="card-b">
          {!result && !running && <div className="note">Run the agent to see its answer and profiler.</div>}
          {running && <div className="note">Waiting for the NAT service…</div>}
          {result && (<>
            <div className="cv-summary" style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 14 }}>
              <div className="metric"><span className="v">{((result.profiler?.total_ms ?? result.latency_ms) / 1000).toFixed(2)}s</span><span className="k">Total latency</span></div>
              <div className="metric"><span className="v">{result.tool_names.length}</span><span className="k">Tools</span></div>
              {result.context_used && <div className="metric"><span className="v">RAG</span><span className="k">Grounded</span></div>}
            </div>
            {steps.length > 0 && (<>
              <label className="fld">Step timeline</label>
              <div style={{ display: "flex", flexDirection: "column", gap: 6, marginBottom: 12 }}>
                {steps.map((s, i) => { const bottleneck = (s.ms || 0) === maxMs && maxMs > 1; return (
                  <div key={i} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <span style={{ width: 110, fontSize: 11, color: "var(--muted)", flex: "0 0 auto", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{s.type || s.name}</span>
                    <div style={{ flex: 1, background: "var(--panel-2)", borderRadius: 4, height: 16, position: "relative" }}>
                      <div style={{ width: `${Math.max(3, ((s.ms || 0) / maxMs) * 100)}%`, height: "100%", background: bottleneck ? "#f59e0b" : "var(--accent)", borderRadius: 4 }} />
                      <span style={{ position: "absolute", right: 6, top: 0, lineHeight: "16px", fontSize: 10, color: "var(--muted)", fontFamily: "var(--mono)" }}>{s.ms != null ? `${s.ms}ms` : ""}{s.tokens ? ` · ${s.tokens}t` : ""}{bottleneck ? " · slow" : ""}</span>
                    </div>
                  </div>
                ); })}
              </div>
            </>)}
            <label className="fld">Answer</label>
            <div style={{ background: "var(--panel)", border: "1px solid var(--border)", borderRadius: 7, padding: "10px 12px", fontSize: 13, lineHeight: 1.55, whiteSpace: "pre-wrap" }}>{result.answer}</div>
            <div className="note" style={{ marginTop: 10 }}>tools: {result.tool_names.join(", ") || "none"}{result.unsupported_tools.length ? ` · unsupported: ${result.unsupported_tools.join(", ")}` : ""}</div>
          </>)}
        </div>
      </div>

      {picker && (
        <div className="modal-wrap show" onClick={(e) => { if (e.target === e.currentTarget) setPicker(null); }}>
          <div className="modal" style={{ maxWidth: 460 }}>
            <div className="mh"><b>{picker === "tool" ? "Add a tool" : "Attach knowledge"}</b><button className="x" onClick={() => setPicker(null)}>×</button></div>
            <div className="mb">
              {picker === "tool" ? (
                <>
                  <div className="note" style={{ marginBottom: 8 }}>Built-in</div>
                  {SUPPORTED.map((t) => (
                    <div key={t.id} className="pick-row" onClick={() => toggle(setTools, t.id)}>
                      <span style={{ fontSize: 15 }}>{t.icon}</span><b>{t.label}</b><span className="note">{t.sub}</span>
                      <span className="pick-tick">{tools.has(t.id) ? "✓ added" : "+ add"}</span>
                    </div>
                  ))}
                  <div className="note" style={{ margin: "12px 0 8px" }}>MCP servers <span style={{ color: "var(--faint)" }}>· from Studio → MCP servers</span></div>
                  {mcp.length === 0 ? <div className="note">no MCP servers — <a href="/admin/mcp">connect one</a></div>
                    : mcp.map((s) => (
                      <div key={s.id} className="pick-row" onClick={() => toggle(setMcpIds, s.id)}>
                        <span style={{ fontSize: 15 }}>🔌</span><b>{s.name}</b><span className="badge">{s.transport}</span>
                        <span className="pick-tick">{mcpIds.has(s.id) ? "✓ added" : "+ add"}</span>
                      </div>
                    ))}
                </>
              ) : (
                kbs.length === 0 ? <div className="note">no knowledge bases — <a href="/kb">create one</a></div>
                  : kbs.map((k) => (
                    <div key={k.id} className="pick-row" onClick={() => toggle(setKbIds, k.id)}>
                      <span style={{ fontSize: 15 }}>📚</span><b>{k.name}</b><span className="note">{k.chunkCount} chunks</span>
                      <span className="pick-tick">{kbIds.has(k.id) ? "✓ attached" : "+ attach"}</span>
                    </div>
                  ))
              )}
              <div className="row" style={{ marginTop: 14, justifyContent: "flex-end" }}><button className="btn" onClick={() => setPicker(null)}>Done</button></div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
