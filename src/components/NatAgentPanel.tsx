"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { estCostUsd, fmtUsd } from "@/lib/pricing";
import { toast } from "@/lib/toast";
import Markdown from "@/components/Markdown";

type ProviderOpt = { id: string; provider: string; label: string | null };
type McpOpt = { id: string; name: string; transport: string };
type KbOpt = { id: string; name: string; status: string; chunkCount: number };
type Step = { name: string; type: string; ms: number | null; tokens: number };
type NatResult = { answer: string; latency_ms: number; model: string; tool_names: string[]; unsupported_tools: string[]; context_used?: boolean; profiler?: { total_ms?: number; steps?: Step[] }; usage?: { total_tokens?: number } };
/* eslint-disable @typescript-eslint/no-explicit-any */
type Saved = { id: string; name: string; config: any; published?: boolean };

const SUPPORTED = [
  { id: "calculator", label: "Calculator", icon: "🧮", sub: "arithmetic" },
  { id: "current_datetime", label: "Date & time", icon: "🕐", sub: "current date" },
];

const TEMPLATES = [
  { name: "Support agent", icon: "🎧", agentType: "react_agent", tools: ["calculator", "current_datetime"], prompt: "You are a friendly customer-support agent. Be concise, use a tool when a calculation or the date is needed, and answer from the attached knowledge, citing it. If you're unsure, say so.", task: "My order arrived damaged — what are my options and by when?" },
  { name: "Research assistant", icon: "🔎", agentType: "react_agent", tools: ["current_datetime"], prompt: "You are a research assistant. Break the question down, reason step by step, and give a clear, well-structured summary.", task: "Summarize the key ideas of retrieval-augmented generation." },
  { name: "Data helper", icon: "📊", agentType: "tool_calling_agent", tools: ["calculator"], prompt: "You are a precise data assistant. Use the calculator for any arithmetic and show the numbers you used.", task: "If revenue is 2,450 and costs are 1,890, what's the margin percentage?" },
];

export default function NatAgentPanel() {
  const [providers, setProviders] = useState<ProviderOpt[]>([]);
  const [providerId, setProviderId] = useState("");
  const [models, setModels] = useState<string[]>([]);
  const [model, setModel] = useState("");
  const [loadingModels, setLoadingModels] = useState(false);
  const [mcp, setMcp] = useState<McpOpt[]>([]);
  const [kbs, setKbs] = useState<KbOpt[]>([]);

  const [agentName, setAgentName] = useState("Support agent");
  const [agentType, setAgentType] = useState<"react_agent" | "tool_calling_agent">("react_agent");
  const [tools, setTools] = useState<Set<string>>(new Set(["calculator", "current_datetime"]));
  const [mcpIds, setMcpIds] = useState<Set<string>>(new Set());
  const [kbIds, setKbIds] = useState<Set<string>>(new Set());
  const [temperature, setTemperature] = useState(0.2);
  const [systemPrompt, setSystemPrompt] = useState("You are a helpful agent. Use a tool when a calculation, lookup, or the current date is needed, and answer from the attached knowledge when relevant.");
  const [task, setTask] = useState("What is 18% of 2450, and what is today's date?");

  const [mode, setMode] = useState<"run" | "chat" | "test">("run");
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<NatResult | null>(null);
  const [err, setErr] = useState("");
  const [picker, setPicker] = useState<"tool" | "kb" | null>(null);
  const [tplOpen, setTplOpen] = useState(false);
  const [loadOpen, setLoadOpen] = useState(false);
  const [saved, setSaved] = useState<Saved[]>([]);
  const [savedId, setSavedId] = useState("");
  const [publishing, setPublishing] = useState(false);
  const [published, setPublished] = useState(false);

  // chat + test
  const [chat, setChat] = useState<{ role: "user" | "assistant"; text: string }[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [testText, setTestText] = useState("What is 15% of 320?\nWhat day is it today?\nWhat is 100 divided by 7?");
  const [testRows, setTestRows] = useState<{ input: string; answer: string; ms: number; cost: number }[]>([]);

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

  async function runOne(taskText: string): Promise<{ ok: boolean; result?: NatResult; error?: string }> {
    try {
      const res = await fetch("/api/agent/nat-run", {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ task: taskText, providerId, model, systemPrompt, agentType, tools: [...tools], temperature, mcpServerIds: [...mcpIds], knowledgeBaseIds: [...kbIds] }),
      });
      const j = await res.json().catch(() => null);
      if (!res.ok) return { ok: false, error: j?.error || `Run failed (${res.status})` };
      return { ok: true, result: j as NatResult };
    } catch (e) { return { ok: false, error: (e as Error).message }; }
  }

  async function run() {
    if (!task.trim()) { setErr("Enter a task for the agent."); return; }
    setRunning(true); setErr(""); setResult(null);
    const r = await runOne(task);
    if (!r.ok) setErr(r.error || "Run failed"); else setResult(r.result!);
    setRunning(false);
  }
  async function sendChat() {
    const input = chatInput.trim(); if (!input || running) return;
    const history = chat.map((m) => `${m.role === "user" ? "User" : "Assistant"}: ${m.text}`).join("\n");
    setChat((c) => [...c, { role: "user", text: input }]); setChatInput(""); setRunning(true);
    const r = await runOne(history ? `${history}\nUser: ${input}` : input);
    setChat((c) => [...c, { role: "assistant", text: r.ok ? (r.result!.answer || "(no answer)") : `⚠ ${r.error}` }]);
    setRunning(false);
  }
  async function runTests() {
    const inputs = testText.split("\n").map((s) => s.trim()).filter(Boolean).slice(0, 10);
    if (!inputs.length) return;
    setRunning(true); setTestRows([]);
    for (const input of inputs) {
      const r = await runOne(input);
      setTestRows((rows) => [...rows, { input, answer: r.ok ? r.result!.answer : `⚠ ${r.error}`, ms: r.result?.latency_ms || 0, cost: estCostUsd(model, r.result?.usage?.total_tokens || 0) }]);
    }
    setRunning(false);
  }

  function applyTemplate(t: typeof TEMPLATES[number]) {
    setAgentName(t.name); setAgentType(t.agentType as "react_agent" | "tool_calling_agent"); setTools(new Set(t.tools)); setSystemPrompt(t.prompt); setTask(t.task); setTplOpen(false);
    toast(`Loaded the “${t.name}” template`, "success");
  }
  function agentConfig() {
    return { runtime: "nat", agentName, agentType, providerId, model, temperature, systemPrompt, tools: [...tools], mcpIds: [...mcpIds], kbIds: [...kbIds], task };
  }
  // Create-or-update the saved project. Returns its id (or "" on failure) so
  // Publish can flip the flag on the same row instead of duplicating it.
  async function persist(): Promise<string> {
    const cfg = agentConfig();
    if (savedId) {
      const r = await fetch("/api/projects", { method: "PATCH", headers: { "content-type": "application/json" }, body: JSON.stringify({ id: savedId, name: agentName || "NAT agent", config: cfg }) });
      return r.ok ? savedId : "";
    }
    const r = await fetch("/api/projects", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ lab: "agent-nat", name: agentName || "NAT agent", config: cfg }) });
    const j = await r.json().catch(() => null);
    if (r.ok && j?.id) { setSavedId(j.id); return j.id; }
    return "";
  }
  async function save() {
    const id = await persist();
    toast(id ? `Saved “${agentName}”` : "Save failed", id ? "success" : "error");
  }
  async function publish() {
    setPublishing(true);
    try {
      const id = await persist();
      if (!id) { toast("Publish failed", "error"); return; }
      const r = await fetch("/api/projects", { method: "PATCH", headers: { "content-type": "application/json" }, body: JSON.stringify({ id, published: true }) });
      if (r.ok) { setPublished(true); toast(`Published “${agentName}” — open it in the Workroom`, "success"); }
      else toast("Publish failed", "error");
    } catch { toast("Publish failed", "error"); }
    finally { setPublishing(false); }
  }
  async function unpublish() {
    if (!savedId) { setPublished(false); return; }
    setPublishing(true);
    try {
      const r = await fetch("/api/projects", { method: "PATCH", headers: { "content-type": "application/json" }, body: JSON.stringify({ id: savedId, published: false }) });
      if (r.ok) { setPublished(false); toast("Removed from the Workroom", "success"); } else toast("Could not unpublish", "error");
    } catch { toast("Could not unpublish", "error"); }
    finally { setPublishing(false); }
  }
  async function openLoad() {
    if (loadOpen) { setLoadOpen(false); return; }
    try { const j = await fetch("/api/projects?lab=agent-nat").then((r) => r.json()); setSaved(j.projects || []); } catch { setSaved([]); }
    setLoadOpen(true);
  }
  function applyConfig(c: any, id?: string, pub?: boolean) {
    if (!c) return;
    setSavedId(id || "");
    setPublished(!!pub);
    if (c.agentName) setAgentName(c.agentName);
    if (c.agentType) setAgentType(c.agentType);
    if (c.providerId) setProviderId(c.providerId);
    if (c.model) setModel(c.model);
    if (c.temperature != null) setTemperature(c.temperature);
    if (c.systemPrompt) setSystemPrompt(c.systemPrompt);
    if (Array.isArray(c.tools)) setTools(new Set(c.tools));
    if (Array.isArray(c.mcpIds)) setMcpIds(new Set(c.mcpIds));
    if (Array.isArray(c.kbIds)) setKbIds(new Set(c.kbIds));
    if (c.task) setTask(c.task);
    setLoadOpen(false); toast("Agent loaded", "success");
  }

  const steps = result?.profiler?.steps || [];
  const maxMs = Math.max(1, ...steps.map((s) => s.ms || 0));
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

      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 14, position: "relative" }}>
        <div className="flowstrip" style={{ margin: 0, flex: 1 }}>
          <span className="fs-node"><span>💬</span>Input</span><span className="fs-arrow">→</span>
          <span className="fs-agent"><span>🤖</span>{agentName || "Agent"}</span><span className="fs-arrow">→</span>
          <span className="fs-node"><span>✅</span>Answer</span>
        </div>
        <div style={{ position: "relative" }}>
          <button className="btn ghost sm" onClick={() => { setTplOpen((o) => !o); setLoadOpen(false); }}>✦ Templates</button>
          {tplOpen && <div className="menu-pop">{TEMPLATES.map((t) => <div key={t.name} className="menu-item" onClick={() => applyTemplate(t)}><span>{t.icon}</span> {t.name}</div>)}</div>}
        </div>
        <button className="btn ghost sm" onClick={save}>💾 Save</button>
        <div style={{ position: "relative" }}>
          <button className="btn ghost sm" onClick={openLoad}>📂 Load</button>
          {loadOpen && <div className="menu-pop"><div className="menu-hd">Saved agents</div>{saved.length ? saved.map((s) => <div key={s.id} className="menu-item" onClick={() => applyConfig(s.config, s.id, s.published)}>{s.name}{s.published ? " ●" : ""}</div>) : <div className="menu-item" style={{ color: "var(--faint)" }}>none saved</div>}</div>}
        </div>
        {published
          ? <><Link className="btn ghost sm" href="/workroom" style={{ color: "#3b9e5f" }}>● Published</Link><button className="btn ghost sm" onClick={unpublish} disabled={publishing} title="Remove from the Workroom">Unpublish</button></>
          : <button className="btn sm" onClick={publish} disabled={publishing || !providerId || !model} title="Make this agent usable in the Workroom">{publishing ? "Publishing…" : "🚀 Publish"}</button>}
      </div>

      <div className="card" style={{ marginBottom: 16 }}>
        <div className="card-h"><span className="fs-agent" style={{ padding: "3px 9px" }}><span>🤖</span>Agent</span><input type="text" value={agentName} onChange={(e) => setAgentName(e.target.value)} style={{ width: 180, height: 30, padding: "4px 9px" }} /><span style={{ flex: 1 }} /><button className="btn sm" onClick={run} disabled={running || !providerId || !model}>{running && mode === "run" ? "Running…" : "▶ Run"}</button></div>
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

      <div className="seg" style={{ maxWidth: 330, marginBottom: 14 }}>
        <button className={mode === "run" ? "on" : ""} onClick={() => setMode("run")}>Run</button>
        <button className={mode === "chat" ? "on" : ""} onClick={() => setMode("chat")}>Chat</button>
        <button className={mode === "test" ? "on" : ""} onClick={() => setMode("test")}>Test</button>
      </div>

      {mode === "run" && (<>
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
                <div className="metric"><span className="v">{((result.profiler?.total_ms ?? result.latency_ms) / 1000).toFixed(2)}s</span><span className="k">Latency</span></div>
                <div className="metric"><span className="v">{result.tool_names.length}</span><span className="k">Tools</span></div>
                <div className="metric"><span className="v">{fmtUsd(estCostUsd(model, result.usage?.total_tokens || 0))}</span><span className="k">Est. cost</span></div>
                {result.context_used && <div className="metric"><span className="v">RAG</span><span className="k">Grounded</span></div>}
              </div>
              {steps.length > 0 && (<>
                <label className="fld">Step timeline</label>
                <div style={{ display: "flex", flexDirection: "column", gap: 6, marginBottom: 12 }}>
                  {steps.map((s, i) => { const b = (s.ms || 0) === maxMs && maxMs > 1; return (
                    <div key={i} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <span style={{ width: 110, fontSize: 11, color: "var(--muted)", flex: "0 0 auto", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{s.type || s.name}</span>
                      <div style={{ flex: 1, background: "var(--panel-2)", borderRadius: 4, height: 16, position: "relative" }}><div style={{ width: `${Math.max(3, ((s.ms || 0) / maxMs) * 100)}%`, height: "100%", background: b ? "#f59e0b" : "var(--accent)", borderRadius: 4 }} /><span style={{ position: "absolute", right: 6, top: 0, lineHeight: "16px", fontSize: 10, color: "var(--muted)", fontFamily: "var(--mono)" }}>{s.ms != null ? `${s.ms}ms` : ""}{b ? " · slow" : ""}</span></div>
                    </div>
                  ); })}
                </div>
              </>)}
              <label className="fld">Answer</label>
              <div style={{ background: "var(--panel)", border: "1px solid var(--border)", borderRadius: 7, padding: "10px 14px" }}><Markdown text={result.answer} /></div>
            </>)}
          </div>
        </div>
      </>)}

      {mode === "chat" && (
        <div className="card">
          <div className="card-h"><span className="t">Chat with {agentName}</span><button className="btn ghost sm r" onClick={() => setChat([])}>Clear</button></div>
          <div className="card-b">
            <div style={{ maxHeight: 380, overflow: "auto", display: "flex", flexDirection: "column", gap: 8, marginBottom: 10 }}>
              {chat.length === 0 && <div className="note">Say hello — the agent keeps the conversation in context.</div>}
              {chat.map((m, i) => (
                <div key={i} style={{ alignSelf: m.role === "user" ? "flex-end" : "flex-start", maxWidth: "82%", background: m.role === "user" ? "var(--accent-weak)" : "var(--panel)", border: "1px solid var(--border)", borderRadius: 10, padding: "8px 12px", fontSize: 13, lineHeight: 1.5 }}>{m.role === "user" ? <span style={{ whiteSpace: "pre-wrap" }}>{m.text}</span> : <Markdown text={m.text} />}</div>
              ))}
              {running && mode === "chat" && <div className="note">thinking…</div>}
            </div>
            <div className="row" style={{ gap: 8 }}>
              <input type="text" placeholder="Message the agent…" value={chatInput} onChange={(e) => setChatInput(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") sendChat(); }} disabled={!providerId || !model} />
              <button className="btn" onClick={sendChat} disabled={running || !chatInput.trim()}>Send</button>
            </div>
          </div>
        </div>
      )}

      {mode === "test" && (
        <div className="card">
          <div className="card-h"><span className="t">Test — one input per line</span></div>
          <div className="card-b">
            <textarea rows={4} value={testText} onChange={(e) => setTestText(e.target.value)} placeholder="One question per line…" />
            <button className="btn block" style={{ marginTop: 10 }} onClick={runTests} disabled={running || !providerId || !model}>{running ? "Running…" : "▶ Run all"}</button>
            {testRows.length > 0 && (
              <table className="tbl" style={{ marginTop: 14 }}>
                <thead><tr><th>Input</th><th>Answer</th><th style={{ textAlign: "right" }}>Latency</th><th style={{ textAlign: "right" }}>Cost</th></tr></thead>
                <tbody>{testRows.map((r, i) => (
                  <tr key={i}><td style={{ fontSize: 12, maxWidth: 180 }}>{r.input}</td><td style={{ fontSize: 12 }}>{r.answer}</td><td className="mono" style={{ textAlign: "right", color: "var(--muted)" }}>{(r.ms / 1000).toFixed(1)}s</td><td className="mono" style={{ textAlign: "right" }}>{fmtUsd(r.cost)}</td></tr>
                ))}</tbody>
              </table>
            )}
          </div>
        </div>
      )}

      {picker && (
        <div className="modal-wrap show" onClick={(e) => { if (e.target === e.currentTarget) setPicker(null); }}>
          <div className="modal" style={{ maxWidth: 460 }}>
            <div className="mh"><b>{picker === "tool" ? "Add a tool" : "Attach knowledge"}</b><button className="x" onClick={() => setPicker(null)}>×</button></div>
            <div className="mb">
              {picker === "tool" ? (<>
                <div className="note" style={{ marginBottom: 8 }}>Built-in</div>
                {SUPPORTED.map((t) => (
                  <div key={t.id} className="pick-row" onClick={() => toggle(setTools, t.id)}><span style={{ fontSize: 15 }}>{t.icon}</span><b>{t.label}</b><span className="note">{t.sub}</span><span className="pick-tick">{tools.has(t.id) ? "✓ added" : "+ add"}</span></div>
                ))}
                <div className="note" style={{ margin: "12px 0 8px" }}>MCP servers <span style={{ color: "var(--faint)" }}>· from Studio → MCP servers</span></div>
                {mcp.length === 0 ? <div className="note">no MCP servers — <a href="/studio/mcp">connect one</a></div>
                  : mcp.map((s) => (<div key={s.id} className="pick-row" onClick={() => toggle(setMcpIds, s.id)}><span style={{ fontSize: 15 }}>🔌</span><b>{s.name}</b><span className="badge">{s.transport}</span><span className="pick-tick">{mcpIds.has(s.id) ? "✓ added" : "+ add"}</span></div>))}
              </>) : (
                kbs.length === 0 ? <div className="note">no knowledge bases — <a href="/kb">create one</a></div>
                  : kbs.map((k) => (<div key={k.id} className="pick-row" onClick={() => toggle(setKbIds, k.id)}><span style={{ fontSize: 15 }}>📚</span><b>{k.name}</b><span className="note">{k.chunkCount} chunks</span><span className="pick-tick">{kbIds.has(k.id) ? "✓ attached" : "+ attach"}</span></div>))
              )}
              <div className="row" style={{ marginTop: 14, justifyContent: "flex-end" }}><button className="btn" onClick={() => setPicker(null)}>Done</button></div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
