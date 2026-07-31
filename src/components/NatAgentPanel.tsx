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

  const [agentType, setAgentType] = useState<"react_agent" | "tool_calling_agent">("react_agent");
  const [tools, setTools] = useState<Set<string>>(new Set(["calculator", "current_datetime"]));
  const [mcpIds, setMcpIds] = useState<Set<string>>(new Set());
  const [kbs, setKbs] = useState<KbOpt[]>([]);
  const [kbIds, setKbIds] = useState<Set<string>>(new Set());
  const [systemPrompt, setSystemPrompt] = useState("You are a careful reasoning agent. Use a tool when a calculation, lookup, or the current date is needed.");
  const [task, setTask] = useState("What is 18% of 2450, and what is today's date?");

  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<NatResult | null>(null);
  const [err, setErr] = useState("");
  const [view, setView] = useState<"form" | "visual">("form");
  const [pos, setPos] = useState<Record<string, { x: number; y: number }>>({});
  const [sel, setSel] = useState<string | null>(null);
  const [zoom, setZoom] = useState(1);

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
        body: JSON.stringify({ task, providerId, model, systemPrompt, agentType, tools: [...tools], temperature: 0, mcpServerIds: [...mcpIds], knowledgeBaseIds: [...kbIds] }),
      });
      const j = await res.json().catch(() => null);
      if (!res.ok) { setErr(j?.error || `Run failed (${res.status})`); return; }
      setResult(j as NatResult);
    } catch (e) { setErr((e as Error).message); }
    finally { setRunning(false); }
  }

  const steps = result?.profiler?.steps || [];
  const maxMs = Math.max(1, ...steps.map((s) => s.ms || 0));

  // ── visual node builder (shares the same config state) ──
  const providerLabel = providers.find((p) => p.id === providerId)?.label || providers.find((p) => p.id === providerId)?.provider || "provider";
  const attached = [
    ...[...tools].map((t) => ({ id: `tool:${t}`, type: "tool", label: t })),
    ...[...mcpIds].map((id) => ({ id: `mcp:${id}`, type: "mcp", label: mcp.find((m) => m.id === id)?.name || "mcp" })),
    ...[...kbIds].map((id) => ({ id: `kb:${id}`, type: "kb", label: kbs.find((k) => k.id === id)?.name || "kb" })),
  ];
  const nodes = [
    { id: "provider", type: "provider", label: providerLabel, sub: model },
    { id: "agent", type: "agent", label: agentType === "react_agent" ? "ReAct agent" : "Tool-calling", sub: `${attached.length} tool${attached.length === 1 ? "" : "s"}` },
    ...attached.map((a) => ({ ...a, sub: undefined as string | undefined })),
    { id: "output", type: "output", label: "Answer", sub: undefined },
  ];
  const NW = 148, NH = 46;
  const defPos = (id: string): { x: number; y: number } => {
    if (id === "provider") return { x: 16, y: 150 };
    if (id === "agent") return { x: 250, y: 150 };
    if (id === "output") return { x: 660, y: 150 };
    const i = attached.findIndex((a) => a.id === id);
    return { x: 470, y: 20 + i * 62 };
  };
  const gp = (id: string) => pos[id] || defPos(id);
  const center = (id: string) => { const p = gp(id); return { x: p.x + NW / 2, y: p.y + NH / 2 }; };
  const wires: [string, string][] = [["provider", "agent"], ["agent", "output"], ...attached.map((a) => [a.id, "agent"] as [string, string])];
  function nodeDown(e: React.PointerEvent, id: string) {
    const start = gp(id); const sx = e.clientX, sy = e.clientY; let moved = false;
    const mv = (ev: PointerEvent) => { const dx = (ev.clientX - sx) / zoom, dy = (ev.clientY - sy) / zoom; if (Math.abs(dx) + Math.abs(dy) > 3) moved = true; setPos((p) => ({ ...p, [id]: { x: Math.max(0, start.x + dx), y: Math.max(0, start.y + dy) } })); };
    const up = () => { window.removeEventListener("pointermove", mv); window.removeEventListener("pointerup", up); if (!moved) setSel(id); };
    window.addEventListener("pointermove", mv); window.addEventListener("pointerup", up);
  }
  function removeNode(id: string) {
    if (id.startsWith("tool:")) toggle(setTools, id.slice(5));
    else if (id.startsWith("mcp:")) toggle(setMcpIds, id.slice(4));
    else if (id.startsWith("kb:")) toggle(setKbIds, id.slice(3));
  }
  const canvasH = Math.max(340, 40 + attached.length * 62);
  const lit = (id: string) => running ? (id === "agent" || id === "provider" ? "lit" : "") : (result ? "done" : "");

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

            <label className="fld" style={{ marginTop: 12 }}>Agent workflow</label>
            <div className="seg">
              <button className={agentType === "react_agent" ? "on" : ""} onClick={() => setAgentType("react_agent")}>ReAct</button>
              <button className={agentType === "tool_calling_agent" ? "on" : ""} onClick={() => setAgentType("tool_calling_agent")}>Tool-calling</button>
            </div>
            <div className="note" style={{ marginTop: 4 }}>{agentType === "react_agent" ? "Reason → act → observe loop (Thought/Action/Observation)." : "Uses the model's native function-calling to invoke tools directly."}</div>

            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginTop: 14, marginBottom: 6 }}>
              <label className="fld" style={{ margin: 0 }}>Tools &amp; knowledge</label>
              <div className="seg" style={{ width: 150 }}>
                <button className={view === "form" ? "on" : ""} onClick={() => setView("form")}>List</button>
                <button className={view === "visual" ? "on" : ""} onClick={() => setView("visual")}>Visual</button>
              </div>
            </div>

            {view === "visual" && (<>
              <div className="nat-canvas" style={{ height: canvasH }} onPointerDown={(e) => { if (e.target === e.currentTarget) setSel(null); }}>
                <div style={{ position: "absolute", inset: 0, transform: `scale(${zoom})`, transformOrigin: "0 0" }}>
                  <svg style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none", overflow: "visible" }}>
                    {wires.map(([a, b], i) => { const c1 = center(a), c2 = center(b); const mx = (c1.x + c2.x) / 2; return <path key={i} d={`M ${c1.x} ${c1.y} C ${mx} ${c1.y}, ${mx} ${c2.y}, ${c2.x} ${c2.y}`} className="nat-wire" />; })}
                  </svg>
                  {nodes.map((n) => { const p = gp(n.id); return (
                    <div key={n.id} className={`nnode nt-${n.type} ${lit(n.id)}${sel === n.id ? " sel" : ""}`} style={{ left: p.x, top: p.y, width: NW }} onPointerDown={(e) => nodeDown(e, n.id)}>
                      <div className="nn-hd">{n.type}</div>
                      <div className="nn-t">{n.label}</div>{n.sub && <div className="nn-s">{n.sub}</div>}
                      {n.id !== "provider" && <span className="nport nport-in" />}
                      {n.id !== "output" && <span className="nport nport-out" />}
                      {(n.type === "tool" || n.type === "mcp" || n.type === "kb") && <button className="nn-x" onPointerDown={(e) => e.stopPropagation()} onClick={(e) => { e.stopPropagation(); removeNode(n.id); if (sel === n.id) setSel(null); }}>×</button>}
                    </div>
                  ); })}
                </div>
                <div className="nat-zoom">
                  <button onClick={() => setZoom((z) => Math.max(0.4, +(z - 0.1).toFixed(2)))}>−</button>
                  <span>{Math.round(zoom * 100)}%</span>
                  <button onClick={() => setZoom((z) => Math.min(1.5, +(z + 0.1).toFixed(2)))}>+</button>
                  <button onClick={() => { setZoom(1); setPos({}); }}>Fit</button>
                </div>
              </div>

              {sel && (() => {
                const n = nodes.find((x) => x.id === sel); if (!n) return null;
                return (
                  <div className="nat-cfg">
                    <div className="nat-cfg-h"><span className="badge">{n.type}</span><b>{n.label}</b><button className="nn-x" style={{ position: "static", marginLeft: "auto" }} onClick={() => setSel(null)}>×</button></div>
                    {n.type === "provider" && (<>
                      <label className="fld">Provider</label><select value={providerId} onChange={(e) => onProvider(e.target.value)}>{providers.map((p) => <option key={p.id} value={p.id}>{p.label || p.provider}</option>)}</select>
                      <label className="fld" style={{ marginTop: 10 }}>Model</label><select value={model} onChange={(e) => setModel(e.target.value)}>{models.map((m) => <option key={m} value={m}>{m}</option>)}</select>
                    </>)}
                    {n.type === "agent" && (<>
                      <label className="fld">Agent workflow</label>
                      <div className="seg"><button className={agentType === "react_agent" ? "on" : ""} onClick={() => setAgentType("react_agent")}>ReAct</button><button className={agentType === "tool_calling_agent" ? "on" : ""} onClick={() => setAgentType("tool_calling_agent")}>Tool-calling</button></div>
                      <label className="fld" style={{ marginTop: 10 }}>System prompt</label><textarea rows={3} value={systemPrompt} onChange={(e) => setSystemPrompt(e.target.value)} />
                    </>)}
                    {n.type === "output" && (<><label className="fld">Task</label><textarea rows={3} value={task} onChange={(e) => setTask(e.target.value)} /></>)}
                    {(n.type === "tool" || n.type === "mcp" || n.type === "kb") && (
                      <div className="note">{n.type === "kb" ? "Knowledge base — the agent retrieves the top passages at run time." : n.type === "mcp" ? "MCP server tool." : "Built-in tool."} <button className="btn ghost sm" style={{ marginLeft: 8 }} onClick={() => { removeNode(n.id); setSel(null); }}>Remove node</button></div>
                    )}
                  </div>
                );
              })()}
              <div className="note" style={{ margin: "8px 0 5px" }}>drag to arrange · click a node to configure · add tools &amp; knowledge below</div>
            </>)}

            {view === "form" && <label className="fld" style={{ marginTop: 4 }}>Built-in tools</label>}
            <div style={{ display: "flex", flexWrap: "wrap", gap: 7, marginTop: view === "visual" ? 0 : undefined }}>
              {SUPPORTED.map((t) => <button key={t.id} type="button" onClick={() => toggle(setTools, t.id)} className={`chip ${tools.has(t.id) ? "on" : ""}`}>{tools.has(t.id) ? "✓ " : "+ "}{t.label}</button>)}
              {view === "visual" && mcp.map((s) => <button key={s.id} type="button" onClick={() => toggle(setMcpIds, s.id)} className={`chip ${mcpIds.has(s.id) ? "on" : ""}`} title={s.transport}>{mcpIds.has(s.id) ? "✓ " : "+ "}{s.name} · MCP</button>)}
              {view === "visual" && kbs.map((k) => <button key={k.id} type="button" onClick={() => toggle(setKbIds, k.id)} className={`chip ${kbIds.has(k.id) ? "on" : ""}`} title={`${k.chunkCount} chunks`}>{kbIds.has(k.id) ? "✓ " : "+ "}{k.name}</button>)}
            </div>

            {view === "form" && <><label className="fld" style={{ marginTop: 12 }}>MCP tools <span className="note">· from Admin → MCP servers</span></label>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 7 }}>
              {mcp.length === 0 ? <span className="note">no enabled MCP servers</span>
                : mcp.map((s) => <button key={s.id} type="button" onClick={() => toggle(setMcpIds, s.id)} className={`chip ${mcpIds.has(s.id) ? "on" : ""}`} title={s.transport}>{mcpIds.has(s.id) ? "✓ " : ""}{s.name} · MCP</button>)}
            </div>

            <label className="fld" style={{ marginTop: 12 }}>Knowledge bases (RAG) <span className="note">· from Studio → Knowledge bases</span></label>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 7 }}>
              {kbs.length === 0 ? <span className="note">no synced knowledge bases — <a href="/kb">create one</a></span>
                : kbs.map((k) => <button key={k.id} type="button" onClick={() => toggle(setKbIds, k.id)} className={`chip ${kbIds.has(k.id) ? "on" : ""}`} title={`${k.chunkCount} chunks`}>{kbIds.has(k.id) ? "✓ " : ""}{k.name}</button>)}
            </div></>}

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
