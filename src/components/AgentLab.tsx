"use client";

import { useEffect, useRef, useState } from "react";
import {
  AGENT_TOOLS, buildKnowledge, reactSystemPrompt, parseReAct,
  type AgentTool, type ToolCtx,
} from "@/lib/agentTools";
import type { RagIndex } from "@/lib/ragUtils";

type AgentType = "react" | "workflow";
type Step = "type" | "build" | "run";
type Provider = { id: string; provider: string; label: string | null; defaultModel: string };

const TYPES: { id: AgentType; label: string; tag: string; desc: string }[] = [
  { id: "react", label: "ReAct Tool Agent", tag: "autonomous", desc: "Reasons in a loop — Thought → Action (tool) → Observation → … → Final Answer. Picks and calls tools on its own." },
  { id: "workflow", label: "Multi-step Workflow", tag: "deterministic", desc: "A fixed pipeline of LLM steps (e.g. plan → draft → critique → finalize), each feeding the next — deterministic & easy to debug." },
];
const TOOL_META: Record<string, { icon: string; label: string }> = {
  calculator: { icon: "🧮", label: "Calculator" },
  datetime: { icon: "🕐", label: "Date & time" },
  web_fetch: { icon: "🌐", label: "Web fetch" },
  knowledge: { icon: "📚", label: "Knowledge base" },
  http_request: { icon: "🔌", label: "HTTP request" },
  human_approval: { icon: "🙋", label: "Human approval" },
  statistics: { icon: "📊", label: "Statistics" },
  unit_convert: { icon: "📐", label: "Unit convert" },
  json_extract: { icon: "🔎", label: "JSON extract" },
};

// One-click starter agents — teach where/why agents are used.
const TEMPLATES: { id: string; label: string; tag: string; desc: string; type: AgentType; name: string; goal: string; tools: string[]; knowledge?: string; task: string }[] = [
  { id: "support", label: "Support bot", tag: "customer support", desc: "Answers policy questions from your docs and asks a human before big refunds.", type: "react", name: "Support bot", goal: "You are a customer-support agent. Answer refund & shipping questions grounded in the knowledge base, and use human_approval before authorising any refund over $500.", tools: ["knowledge", "human_approval"], knowledge: "Returns policy: damaged items may be returned within 30 days of delivery for a full refund. Refunds go to the original payment method within 5 business days. Shipping is free over $50, else a $6 flat fee. Gift cards are non-refundable. Refunds above $500 require a manager's approval.", task: "A customer says their $650 order arrived damaged and wants a full refund. What should we do?" },
  { id: "research", label: "Research assistant", tag: "knowledge work", desc: "Reads web pages and gives a concise, grounded answer.", type: "react", name: "Research assistant", goal: "You research questions using web_fetch and reply with a concise, well-structured answer. Cite the page you used.", tools: ["web_fetch", "calculator"], task: "Fetch https://en.wikipedia.org/wiki/Retrieval-augmented_generation and summarise what RAG is in 3 bullet points." },
  { id: "analyst", label: "Data analyst", tag: "analytics", desc: "Answers quantitative questions with the calculator + your data.", type: "react", name: "Data analyst", goal: "You answer quantitative business questions. Use the calculator for arithmetic and the knowledge base for figures.", tools: ["calculator", "knowledge"], knowledge: "Q3 figures: revenue = 240000, cost of goods = 175000, marketing spend = 22000, new customers = 480.", task: "What was the gross margin percentage in Q3, and the marketing cost per new customer?" },
  { id: "api", label: "API agent", tag: "integration", desc: "Calls a public REST API as a tool and reports the result.", type: "react", name: "API agent", goal: "You answer questions by calling public REST APIs with http_request and reading the JSON.", tools: ["http_request", "calculator"], task: "Use http_request on https://api.github.com/repos/vercel/next.js and tell me how many stars the repo has." },
];

// ── chat helper (OpenAI-compatible proxy) ──
async function chatOnce(messages: { role: string; content: string }[], temperature: number, maxTokens: number, providerId?: string, model?: string): Promise<string> {
  const res = await fetch("/api/chat", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ messages, temperature, maxTokens, streaming: false, providerId: providerId || undefined, model: model || undefined }) });
  if (!res.ok) { const j = await res.json().catch(() => ({ error: "request failed" })); throw new Error(j.error || "request failed"); }
  return (await res.text()).trim();
}

type TraceItem = { kind: "thought" | "action" | "observation" | "final" | "error"; text: string; tool?: string; state: string };
type ANode = { id: string; type: "trigger" | "agent" | "output" | "model" | "tool"; toolId?: string; icon: string; title: string; sub: string; w: number; h: number; bottom?: string[] };

const CANVAS_W = 980, CANVAS_H = 400;
const DEFAULT_POS: Record<string, { x: number; y: number }> = { trigger: { x: 24, y: 150 }, agent: { x: 306, y: 108 }, output: { x: 772, y: 150 }, model: { x: 150, y: 300 } };
const toolDefault = (i: number) => ({ x: 344 + i * 158, y: 300 });

export default function AgentLab() {
  const [step, setStep] = useState<Step>("type");
  const [agentType, setAgentType] = useState<AgentType>("react");
  const [buildMode, setBuildMode] = useState<"visual" | "manual" | "prompt">("visual");

  // providers / models
  const [providers, setProviders] = useState<Provider[]>([]);
  const [providerId, setProviderId] = useState("");
  const [modelList, setModelList] = useState<string[]>([]);
  const [model, setModel] = useState("");
  const [provKnown, setProvKnown] = useState(false);
  const [msg, setMsg] = useState("");

  // shared / react config
  const [name, setName] = useState("Support agent");
  const [description, setDescription] = useState("Answers questions and uses tools when it needs facts.");
  const [goal, setGoal] = useState("You are a helpful support agent. Use the connected tools when you need fresh facts, and cite what you use.");
  const [temperature, setTemperature] = useState(0.4);
  const [maxTokens, setMaxTokens] = useState(600);
  const [maxIters, setMaxIters] = useState(6);
  const [enabledTools, setEnabledTools] = useState<Set<string>>(new Set(["calculator", "datetime", "knowledge"]));
  const [placedTools, setPlacedTools] = useState<Set<string>>(new Set(["calculator", "datetime", "knowledge"]));
  const [knowledgeText, setKnowledgeText] = useState("Returns policy: damaged items may be returned within 30 days of delivery for a full refund. Shipping is free on orders over $50, otherwise a flat $6 fee applies. Gift cards are non-refundable.");
  const [uploading, setUploading] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  // workflow config
  const [steps, setSteps] = useState<{ id: string; name: string; instruction: string }[]>([
    { id: "s1", name: "Plan", instruction: "Break the user's request into a short bullet plan." },
    { id: "s2", name: "Draft", instruction: "Write a first draft that follows the plan." },
    { id: "s3", name: "Critique", instruction: "List concrete weaknesses in the draft." },
    { id: "s4", name: "Finalize", instruction: "Rewrite the draft addressing every critique. Output only the final result." },
  ]);

  // from-prompt
  const [describe, setDescribe] = useState("A finance helper that can do calculations and look things up online.");
  const [generating, setGenerating] = useState(false);

  // node canvas
  const [nodePos, setNodePos] = useState<Record<string, { x: number; y: number }>>({});
  const [aSel, setASel] = useState("agent");
  const [addOpen, setAddOpen] = useState(false);
  const [nodeStatus, setNodeStatus] = useState<Record<string, string>>({});
  const [fullscreen, setFullscreen] = useState(false);
  const [connectFrom, setConnectFrom] = useState<{ id: string; kind: "tool" | "agent" } | null>(null);
  const [connectXY, setConnectXY] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [saved, setSaved] = useState(false);
  const [savedAgents, setSavedAgents] = useState<{ id: string; name: string; config: Record<string, unknown> }[]>([]);
  const [loadOpen, setLoadOpen] = useState(false);
  const canvasRef = useRef<HTMLDivElement>(null);

  // run
  const [task, setTask] = useState("What is 15% of 240, and how many days until 2026-12-25?");
  const [running, setRunning] = useState(false);
  const [trace, setTrace] = useState<TraceItem[]>([]);
  const [finalOut, setFinalOut] = useState("");
  const [wfOutputs, setWfOutputs] = useState<{ name: string; text: string; state: string }[]>([]);
  const [showCode, setShowCode] = useState(false);
  const [copied, setCopied] = useState(false);
  const [pendingApproval, setPendingApproval] = useState<{ q: string; resolve: (v: string) => void } | null>(null);
  const [metrics, setMetrics] = useState<{ calls: number; tools: number; ms: number; tokens: number } | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  function applyTemplate(t: typeof TEMPLATES[number]) {
    setAgentType(t.type); setName(t.label); setDescription(t.desc); setGoal(t.goal);
    setEnabledTools(new Set(t.tools)); setPlacedTools(new Set(t.tools)); if (t.knowledge) setKnowledgeText(t.knowledge);
    setTask(t.task); setNodePos({}); setNodeStatus({}); setASel("agent"); setBuildMode("visual"); setStep("build");
  }

  useEffect(() => {
    fetch("/api/models").then((r) => r.json()).then((j) => {
      setProvKnown(true);
      setProviders(j.providers || []);
      if (j.providers?.length) { setProviderId(j.providerId || j.providers[0].id); setModelList(j.models || []); setModel(j.default || (j.models && j.models[0]) || ""); }
    }).catch(() => setProvKnown(true));
  }, []);
  useEffect(() => { if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight; }, [trace, wfOutputs]);
  // Delete / Backspace removes the selected tool node (or clicked wire's tool).
  useEffect(() => {
    if (step !== "build" || agentType !== "react" || buildMode !== "visual") return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Delete" && e.key !== "Backspace") return;
      const el = document.activeElement as HTMLElement | null;
      if (el && ["INPUT", "TEXTAREA", "SELECT"].includes(el.tagName)) return;
      if (aSel.startsWith("tool:")) { e.preventDefault(); const id = aSel.slice(5); setPlacedTools((s) => { const n = new Set(s); n.delete(id); return n; }); setEnabledTools((s) => { const n = new Set(s); n.delete(id); return n; }); setASel("agent"); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [step, agentType, buildMode, aSel]);

  const hasProvider = providers.length > 0;
  const toolList = AGENT_TOOLS.filter((t) => enabledTools.has(t.id));
  const togglePlaced = (id: string) => setPlacedTools((s) => { const n = new Set(s); if (n.has(id)) { n.delete(id); setEnabledTools((e) => { const m = new Set(e); m.delete(id); return m; }); } else n.add(id); return n; });
  const connectTool = (id: string) => { setPlacedTools((s) => new Set(s).add(id)); setEnabledTools((s) => new Set(s).add(id)); };
  const disconnectTool = (id: string) => setEnabledTools((s) => { const n = new Set(s); n.delete(id); return n; });
  const removeToolNode = (id: string) => { setPlacedTools((s) => { const n = new Set(s); n.delete(id); return n; }); setEnabledTools((s) => { const n = new Set(s); n.delete(id); return n; }); };
  const toggleManual = (id: string) => (enabledTools.has(id) ? disconnectTool(id) : connectTool(id));
  function loadModels(pid: string) {
    setProviderId(pid);
    fetch(`/api/models?providerId=${pid}`).then((r) => r.json()).then((j) => { setModelList(j.models || []); setModel(j.default || (j.models && j.models[0]) || ""); }).catch(() => {});
  }
  const providerLabel = providers.find((p) => p.id === providerId)?.label || providers.find((p) => p.id === providerId)?.provider || "provider";

  // ── knowledge upload ──
  async function onKnowledgeFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0]; if (!f) return;
    setUploading(true); setMsg("");
    const ext = (f.name.split(".").pop() || "txt").toLowerCase();
    try {
      if (["pdf", "docx", "doc", "xlsx", "xls", "xlsm"].includes(ext)) {
        const fd = new FormData(); fd.append("file", f);
        const r = await fetch("/api/rag/extract", { method: "POST", body: fd });
        const j = await r.json(); if (!r.ok) throw new Error(j.error || "parse failed");
        setKnowledgeText(j.text);
      } else setKnowledgeText(await f.text());
    } catch (err) { setMsg(`Could not read ${f.name}: ${(err as Error).message}`); }
    setUploading(false); e.target.value = "";
  }

  // ── node canvas model ──
  const placedOrder = [...placedTools];
  const nodes: ANode[] = [
    { id: "trigger", type: "trigger", icon: "💬", title: "User input", sub: "Trigger", w: 150, h: 56 },
    { id: "agent", type: "agent", icon: "🤖", title: name || "AI Agent", sub: "ReAct agent", w: 200, h: 88, bottom: ["model", "tools"] },
    { id: "output", type: "output", icon: "✅", title: "Final answer", sub: "Output", w: 150, h: 56 },
    { id: "model", type: "model", icon: "⚙️", title: (model || providerLabel).slice(0, 18), sub: "Model", w: 172, h: 54 },
    ...placedOrder.map((tid) => ({ id: "tool:" + tid, type: "tool" as const, toolId: tid, icon: TOOL_META[tid].icon, title: TOOL_META[tid].label, sub: tid === "knowledge" ? "Knowledge" : "Tool", w: 150, h: 54 })),
  ];
  const getPos = (id: string) => nodePos[id] || DEFAULT_POS[id] || toolDefault(Math.max(0, placedOrder.indexOf(id.replace("tool:", ""))));
  function portPos(n: ANode, which: "in" | "out" | "top" | number): [number, number] {
    const p = getPos(n.id);
    if (which === "in") return [p.x, p.y + n.h / 2];
    if (which === "out") return [p.x + n.w, p.y + n.h / 2];
    if (which === "top") return [p.x + n.w / 2, p.y];
    const cnt = n.bottom!.length; return [p.x + (n.w * ((which as number) + 1)) / (cnt + 1), p.y + n.h];
  }
  const agentNode = nodes.find((n) => n.id === "agent")!;
  const wires = [
    { from: "trigger", to: "agent", kind: "main", port: 0 },
    { from: "agent", to: "output", kind: "main", port: 0 },
    { from: "model", to: "agent", kind: "sub", port: 0 },
    ...[...enabledTools].filter((tid) => placedTools.has(tid)).map((tid) => ({ from: "tool:" + tid, to: "agent", kind: "sub", port: 1 })),
  ];
  function wirePath(w: { from: string; to: string; kind: string; port: number }): string {
    const a = nodes.find((n) => n.id === w.from), b = nodes.find((n) => n.id === w.to);
    if (!a || !b) return "";
    if (w.kind === "main") { const [x1, y1] = portPos(a, "out"); const [x2, y2] = portPos(b, "in"); const dx = Math.max(40, (x2 - x1) / 2); return `M${x1} ${y1} C${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`; }
    const [sx, sy] = portPos(a, "top"); const [bx, by] = portPos(agentNode, w.port); return `M${bx} ${by} C${bx} ${by + 38}, ${sx} ${sy - 38}, ${sx} ${sy}`;
  }
  function onNodeDown(e: React.PointerEvent, id: string) {
    const start = getPos(id); const sx = e.clientX, sy = e.clientY; let moved = false;
    const mv = (ev: PointerEvent) => { const dx = ev.clientX - sx, dy = ev.clientY - sy; if (Math.abs(dx) + Math.abs(dy) > 4) moved = true; setNodePos((p) => ({ ...p, [id]: { x: Math.max(0, start.x + dx), y: Math.max(0, start.y + dy) } })); };
    const up = () => { document.removeEventListener("pointermove", mv); document.removeEventListener("pointerup", up); if (!moved) setASel(id); };
    document.addEventListener("pointermove", mv); document.addEventListener("pointerup", up); e.preventDefault();
  }
  // Drag a port to another node to wire it up (n8n-style start→end connection).
  function portDown(e: React.PointerEvent, id: string, kind: "tool" | "agent") {
    e.stopPropagation(); e.preventDefault();
    const canvas = canvasRef.current; if (!canvas) return;
    const rel = (cx: number, cy: number) => { const r = canvas.getBoundingClientRect(); return { x: cx - r.left + canvas.scrollLeft, y: cy - r.top + canvas.scrollTop }; };
    setConnectFrom({ id, kind }); setConnectXY(rel(e.clientX, e.clientY));
    const mv = (ev: PointerEvent) => setConnectXY(rel(ev.clientX, ev.clientY));
    const up = (ev: PointerEvent) => {
      document.removeEventListener("pointermove", mv); document.removeEventListener("pointerup", up);
      const pt = rel(ev.clientX, ev.clientY);
      const hit = nodes.find((n) => { const p = getPos(n.id); return pt.x >= p.x && pt.x <= p.x + n.w && pt.y >= p.y && pt.y <= p.y + n.h; });
      if (hit) {
        if (kind === "tool" && hit.id === "agent") connectTool(id.replace("tool:", ""));
        else if (kind === "agent" && hit.type === "tool") connectTool(hit.toolId!);
      }
      setConnectFrom(null);
    };
    document.addEventListener("pointermove", mv); document.addEventListener("pointerup", up);
  }
  function tempWirePath(): string {
    if (!connectFrom) return "";
    let src: [number, number];
    if (connectFrom.kind === "agent") src = portPos(agentNode, 1);
    else { const tn = nodes.find((n) => n.id === connectFrom.id); if (!tn) return ""; src = portPos(tn, "top"); }
    return `M${src[0]} ${src[1]} C${src[0]} ${src[1] + 40}, ${connectXY.x} ${connectXY.y - 40}, ${connectXY.x} ${connectXY.y}`;
  }
  const flowCanvas = () => (
    <div className="acanvas" ref={canvasRef} onClick={() => setAddOpen(false)}>
      <svg className="wires2" width={CANVAS_W} height={CANVAS_H}>
        {wires.map((w, i) => { const act = ["running", "done"].includes(nodeStatus[w.from] || "") || ["running", "done"].includes(nodeStatus[w.to] || ""); const sel = w.kind === "sub" && aSel === w.from; return <path key={i} className={`${w.kind === "sub" ? "sub" : ""} ${act ? "active" : ""} ${sel ? "selw" : ""}`} d={wirePath(w)} />; })}
        {wires.filter((w) => w.kind === "sub").map((w, i) => <path key={"hit" + i} className="wire-hit" d={wirePath(w)} onClick={(e) => { e.stopPropagation(); setASel(w.from); }} />)}
        {connectFrom && <path className="sub selw" d={tempWirePath()} />}
      </svg>
      {nodes.map((n) => { const pos = getPos(n.id); const st = nodeStatus[n.id] || ""; const unwired = n.type === "tool" && !enabledTools.has(n.toolId!); return (
        <div key={n.id} className={`anode type-${n.type} ${aSel === n.id ? "sel" : ""} ${st} ${unwired ? "unwired" : ""}`} style={{ left: pos.x, top: pos.y, width: n.w }} onPointerDown={(e) => onNodeDown(e, n.id)}>
          <div className="ah"><span className="aic">{n.icon}</span><div><div className="atitle">{n.title}</div><div className="asub">{unwired ? "drag ↑ to connect" : n.sub}</div></div><span className="abadge" /></div>
          {n.type === "tool" && <span className="aport ap-top" title="drag to the Agent to connect" onPointerDown={(e) => portDown(e, n.id, "tool")} />}
          {n.type === "agent" && <span className="aport ap-agent" title="drag to a tool to connect" onPointerDown={(e) => portDown(e, "agent", "agent")} />}
          {n.type === "tool" && <button className="anode-x" title="Remove node" onPointerDown={(e) => e.stopPropagation()} onClick={(e) => { e.stopPropagation(); removeToolNode(n.toolId!); if (aSel === n.id) setASel("agent"); }}>×</button>}
        </div>
      ); })}
    </div>
  );

  // ── from-prompt generation ──
  async function generateFromPrompt() {
    setGenerating(true); setMsg("");
    try {
      const react = agentType === "react";
      const sys = react ? "Design a ReAct tool agent from the user's description." : "Design a multi-step LLM workflow from the user's description.";
      const schema = react
        ? `{"name": string, "goal": string, "tools": string[] (subset of ["calculator","datetime","web_fetch","knowledge","http_request","human_approval"]), "maxIters": number}`
        : `{"name": string, "steps": [{"name": string, "instruction": string}] (3-5)}`;
      const out = await chatOnce([{ role: "system", content: `${sys} Reply with ONLY minified JSON matching: ${schema}. No prose, no code fences.` }, { role: "user", content: describe }], 0.3, 500, providerId, model);
      const cfg = JSON.parse(out.slice(out.indexOf("{"), out.lastIndexOf("}") + 1));
      if (cfg.name) setName(String(cfg.name));
      if (react) {
        if (cfg.goal) setGoal(String(cfg.goal));
        if (Array.isArray(cfg.tools)) { const picked = new Set<string>(cfg.tools.filter((t: string) => AGENT_TOOLS.some((x) => x.id === t))); setEnabledTools(picked); setPlacedTools(new Set(picked)); }
        if (cfg.maxIters) setMaxIters(Math.max(1, Math.min(10, Number(cfg.maxIters) || 6)));
        setBuildMode("visual");
      } else if (Array.isArray(cfg.steps)) setSteps(cfg.steps.slice(0, 6).map((s: { name?: string; instruction?: string }, i: number) => ({ id: `s${i + 1}`, name: String(s.name || `Step ${i + 1}`), instruction: String(s.instruction || "") })));
    } catch (e) { setMsg("Could not generate config: " + (e as Error).message); }
    setGenerating(false);
  }

  // ── RUN ──
  async function runReact() {
    setRunning(true); setTrace([]); setFinalOut(""); setMsg(""); setMetrics(null); setPendingApproval(null);
    setNodeStatus({ trigger: "done", model: "done", agent: "running" });
    const ctx: ToolCtx = { requestApproval: (q) => new Promise<string>((resolve) => setPendingApproval({ q, resolve })) };
    if (enabledTools.has("knowledge")) { const kb = buildKnowledge(knowledgeText); if (kb) { ctx.knowledgeIndex = kb.index as RagIndex; ctx.knowledgeChunks = kb.chunks; } }
    const tools: AgentTool[] = toolList;
    const messages: { role: string; content: string }[] = [{ role: "system", content: reactSystemPrompt(tools, goal) }, { role: "user", content: task }];
    const push = (t: TraceItem) => setTrace((tr) => [...tr, t]);
    const est = (s: string) => Math.round(s.length / 4);
    let calls = 0, toolsUsed = 0, tokens = 0; const t0 = performance.now();
    try {
      for (let iter = 0; iter < maxIters; iter++) {
        push({ kind: "thought", text: "thinking…", state: "active" });
        setNodeStatus((s) => ({ ...s, agent: "running" }));
        tokens += messages.reduce((a, m) => a + est(m.content), 0); calls++;
        const resp = await chatOnce(messages, temperature, Math.min(maxTokens, 500), providerId, model);
        tokens += est(resp);
        const p = parseReAct(resp);
        setTrace((tr) => { const c = [...tr]; c[c.length - 1] = { kind: "thought", text: p.thought || "(reasoning)", state: "done" }; return c; });
        if (p.final || (!p.action && !p.final)) { const ans = p.final || resp; setFinalOut(ans); push({ kind: "final", text: ans, state: "done" }); setNodeStatus((s) => ({ ...s, agent: "done", output: "done" })); break; }
        const tool = tools.find((t) => t.name.toLowerCase() === (p.action || "").toLowerCase());
        push({ kind: "action", text: p.input || "", tool: p.action, state: "active" });
        if (tool) { toolsUsed++; setNodeStatus((s) => ({ ...s, ["tool:" + tool.id]: "running" })); }
        const obs = tool ? await tool.run(p.input || "", ctx) : `Unknown tool "${p.action}". Available: ${tools.map((t) => t.name).join(", ")}.`;
        if (tool) setNodeStatus((s) => ({ ...s, ["tool:" + tool.id]: "done" }));
        setTrace((tr) => { const c = [...tr]; c[c.length - 1] = { ...c[c.length - 1], state: "done" }; return c; });
        push({ kind: "observation", text: obs, state: "done" });
        messages.push({ role: "assistant", content: resp }); messages.push({ role: "user", content: `Observation: ${obs}` });
        if (iter === maxIters - 1) { push({ kind: "error", text: `Reached the ${maxIters}-step limit without a final answer.`, state: "done" }); setNodeStatus((s) => ({ ...s, agent: "done", output: "done" })); }
      }
    } catch (e) { push({ kind: "error", text: (e as Error).message, state: "done" }); }
    setMetrics({ calls, tools: toolsUsed, ms: Math.round(performance.now() - t0), tokens });
    setPendingApproval(null); setRunning(false);
  }
  async function runWorkflow() {
    setRunning(true); setMsg(""); setFinalOut(""); setWfOutputs(steps.map((s) => ({ name: s.name, text: "", state: "" })));
    let prev = task;
    try {
      for (let k = 0; k < steps.length; k++) {
        setWfOutputs((o) => { const n = [...o]; n[k] = { ...n[k], state: "active" }; return n; });
        const messages = [{ role: "system", content: `You are executing step "${steps[k].name}" of a workflow. ${steps[k].instruction}` }, { role: "user", content: `Original request: ${task}\n\nPrevious step output:\n${prev}` }];
        const res = await fetch("/api/chat", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ messages, temperature, providerId: providerId || undefined, model: model || undefined }) });
        if (!res.ok || !res.body) { const j = await res.json().catch(() => ({ error: "failed" })); throw new Error(j.error || "failed"); }
        const reader = res.body.getReader(); const dec = new TextDecoder(); let acc = "";
        for (; ;) { const { done, value } = await reader.read(); if (done) break; acc += dec.decode(value, { stream: true }); setWfOutputs((o) => { const n = [...o]; n[k] = { ...n[k], text: acc }; return n; }); }
        setWfOutputs((o) => { const n = [...o]; n[k] = { ...n[k], state: "done" }; return n; });
        prev = acc;
      }
      setFinalOut(prev);
    } catch (e) { setMsg("Workflow error: " + (e as Error).message); }
    setRunning(false);
  }
  function startRun() { setStep("run"); if (agentType === "react") runReact(); else runWorkflow(); }
  // Linear workflow node canvas (input → steps → output), lights up as it runs.
  const wfCanvas = () => {
    const list = [
      { id: "in", icon: "💬", title: "Input", sub: "Trigger", st: running || wfOutputs.length ? "done" : "" },
      ...steps.map((s, i) => ({ id: s.id, icon: "◆", title: s.name, sub: `Step ${i + 1}`, st: wfOutputs[i]?.state || "" })),
      { id: "out", icon: "✅", title: "Final output", sub: "Output", st: finalOut && !running ? "done" : "" },
    ];
    const W = 150, H = 60, GAP = 44, y = 100;
    const pos = (i: number) => ({ x: 20 + i * (W + GAP), y });
    const cw = Math.max(CANVAS_W, 20 + list.length * (W + GAP));
    return (
      <div className="acanvas" style={{ height: 220 }}>
        <svg className="wires2" width={cw} height={220}>
          {list.slice(0, -1).map((_, i) => { const a = pos(i), b = pos(i + 1); const x1 = a.x + W, y1 = a.y + H / 2, x2 = b.x, y2 = b.y + H / 2; const dx = Math.max(30, (x2 - x1) / 2); const act = ["running", "done"].includes(list[i].st); return <path key={i} className={act ? "active" : ""} d={`M${x1} ${y1} C${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`} />; })}
        </svg>
        {list.map((n, i) => { const p = pos(i); return (
          <div key={n.id} className={`anode ${n.st}`} style={{ left: p.x, top: p.y, width: W }}>
            <div className="ah"><span className="aic">{n.icon}</span><div><div className="atitle">{n.title}</div><div className="asub">{n.sub}</div></div><span className="abadge" /></div>
          </div>
        ); })}
      </div>
    );
  };

  // ── code export ──
  function buildCode(): string {
    const q3 = (s: string) => s.replace(/\\/g, "\\\\").replace(/"""/g, '\\"\\"\\"');
    const preamble = `import os
from openai import OpenAI

client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL"), api_key=os.environ.get("OPENAI_API_KEY"))
MODEL = ${JSON.stringify(model || "your-model")}`;
    if (agentType === "workflow") {
      const stepsPy = steps.map((s) => `    (${JSON.stringify(s.name)}, ${JSON.stringify(s.instruction)}),`).join("\n");
      return `# AI Workbench · workflow "${name}"  (pip install openai)
${preamble}

STEPS = [
${stepsPy}
]

def run(task):
    prev = task
    for step_name, instruction in STEPS:
        msgs = [
            {"role": "system", "content": f"You are executing step '{step_name}'. {instruction}"},
            {"role": "user", "content": f"Original request: {task}\\n\\nPrevious step output:\\n{prev}"},
        ]
        prev = client.chat.completions.create(model=MODEL, messages=msgs, temperature=${temperature}).choices[0].message.content
        print(f"--- {step_name} ---\\n{prev}\\n")
    return prev

if __name__ == "__main__":
    print(run(${JSON.stringify(task)}))`;
    }
    const tls = toolList;
    const imports = ["import re"]; const defs: string[] = []; const reg: string[] = [];
    if (tls.some((t) => t.id === "calculator")) { imports.push("import math"); reg.push(`"calculator": tool_calculator`); defs.push(`def tool_calculator(x):
    try:
        return str(eval(x, {"__builtins__": {}}, {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}))
    except Exception as e:
        return f"Error: {e}"`); }
    if (tls.some((t) => t.id === "datetime")) { imports.push("import datetime as _dt"); reg.push(`"datetime": tool_datetime`); defs.push(`def tool_datetime(x):
    m = re.search(r"(\\d{4}-\\d{1,2}-\\d{1,2})", x or "")
    if m:
        delta = (_dt.date.fromisoformat(m.group(1)) - _dt.date.today()).days
        return f"{m.group(1)} is {delta} day(s) from today."
    return f"Now: {_dt.datetime.now().isoformat()}"`); }
    if (tls.some((t) => t.id === "web_fetch")) { imports.push("import requests"); reg.push(`"web_fetch": tool_web_fetch`); defs.push(`def tool_web_fetch(x):
    try:
        html = requests.get(x.strip(), timeout=12).text
        return re.sub(r"\\s+", " ", re.sub(r"<[^>]+>", " ", html))[:1200]
    except Exception as e:
        return f"Error: {e}"`); }
    if (tls.some((t) => t.id === "knowledge")) { reg.push(`"knowledge": tool_knowledge`); defs.push(`KNOWLEDGE = """${q3(knowledgeText)}"""

def tool_knowledge(x):
    sents = re.split(r"(?<=[.!?])\\s+", KNOWLEDGE)
    q = set(re.findall(r"[a-z0-9]+", x.lower()))
    ranked = sorted(sents, key=lambda s: len(q & set(re.findall(r"[a-z0-9]+", s.lower()))), reverse=True)
    return " ".join(ranked[:3])[:1200]`); }
    if (tls.some((t) => t.id === "http_request")) { if (!imports.includes("import requests")) imports.push("import requests"); imports.push("import json as _json"); reg.push(`"http_request": tool_http_request`); defs.push(`def tool_http_request(x):
    x = x.strip()
    try:
        if x.startswith("{"):
            spec = _json.loads(x)
            r = requests.request(spec.get("method", "GET").upper(), spec["url"], json=spec.get("body"), timeout=12)
        else:
            r = requests.get(x, timeout=12)
        return f"HTTP {r.status_code}\\n{r.text[:1400]}"
    except Exception as e:
        return f"Error: {e}"`); }
    if (tls.some((t) => t.id === "human_approval")) { reg.push(`"human_approval": tool_human_approval`); defs.push(`def tool_human_approval(x):
    ans = input(f"[APPROVAL NEEDED] {x} (yes/no) > ").strip().lower()
    return "User APPROVED." if ans in ("y", "yes", "approve") else "User DENIED the action."`); }
    if (tls.some((t) => t.id === "statistics")) { imports.push("import statistics as _st"); reg.push(`"statistics": tool_statistics`); defs.push(`def tool_statistics(x):
    nums = [float(n) for n in re.findall(r"-?\\d+(?:\\.\\d+)?", x)]
    if not nums:
        return "Error: provide numbers."
    return f"count={len(nums)} mean={_st.mean(nums):.3f} median={_st.median(nums)} min={min(nums)} max={max(nums)} stdev={_st.pstdev(nums):.3f}"`); }
    if (tls.some((t) => t.id === "unit_convert")) { reg.push(`"unit_convert": tool_unit_convert`); defs.push(`def tool_unit_convert(x):
    m = re.search(r"(-?\\d+(?:\\.\\d+)?)\\s*\\u00b0?\\s*([a-z]+)\\s*(?:to|in|->)\\s*\\u00b0?\\s*([a-z]+)", x.lower())
    if not m:
        return "Error: use like '10 km to mi'."
    v, a, b = float(m.group(1)), m.group(2), m.group(3)
    f = {"m":1,"km":1000,"cm":0.01,"mm":0.001,"mi":1609.344,"ft":0.3048,"in":0.0254,"yd":0.9144,"g":1,"kg":1000,"mg":0.001,"lb":453.592,"oz":28.3495}
    if a in f and b in f:
        return f"{v} {a} = {v*f[a]/f[b]:.4f} {b}"
    to_c = {"c": lambda t: t, "f": lambda t: (t-32)*5/9, "k": lambda t: t-273.15}
    fr_c = {"c": lambda t: t, "f": lambda t: t*9/5+32, "k": lambda t: t+273.15}
    if a in to_c and b in fr_c:
        return f"{v}{a.upper()} = {fr_c[b](to_c[a](v)):.2f}{b.upper()}"
    return "Error: unknown units."`); }
    if (tls.some((t) => t.id === "json_extract")) { if (!imports.includes("import json as _json")) imports.push("import json as _json"); reg.push(`"json_extract": tool_json_extract`); defs.push(`def tool_json_extract(x):
    parts = x.split("\\n", 1)
    path = parts[0].strip() if len(parts) == 2 else ""
    body = re.sub(r"^HTTP\\s+\\d+\\s*", "", parts[1] if len(parts) == 2 else x).strip()
    try:
        data = _json.loads(body)
    except Exception:
        return "Error: invalid JSON."
    if not path:
        return "keys: " + ", ".join(str(k) for k in (data.keys() if isinstance(data, dict) else range(len(data))))
    cur = data
    for k in [p for p in re.split(r"[.\\[\\]]", path) if p]:
        cur = cur[int(k)] if isinstance(cur, list) else cur.get(k)
        if cur is None:
            break
    return _json.dumps(cur)[:800] if isinstance(cur, (dict, list)) else str(cur)`); }
    return `# AI Workbench · ReAct agent "${name}"  (pip install openai${tls.some((t) => ["web_fetch", "http_request"].includes(t.id)) ? " requests" : ""})
import os
${imports.join("\n")}
from openai import OpenAI

client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL"), api_key=os.environ.get("OPENAI_API_KEY"))
MODEL = ${JSON.stringify(model || "your-model")}

# ── tools ──
${defs.join("\n\n")}

TOOLS = { ${reg.join(", ")} }

SYSTEM = """${q3(reactSystemPrompt(tls, goal))}"""

def run(task, max_iters=${maxIters}):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]
    for _ in range(max_iters):
        text = client.chat.completions.create(model=MODEL, messages=messages, temperature=${temperature}).choices[0].message.content
        final = re.search(r"Final Answer:\\s*([\\s\\S]*)", text, re.I)
        if final:
            return final.group(1).strip()
        action = re.search(r"Action:\\s*([^\\n]+)", text, re.I)
        if not action:
            return text.strip()
        ainput = re.search(r"Action Input:\\s*([\\s\\S]*?)(?=\\n\\s*(?:Thought|Action|Observation)\\s*:|$)", text, re.I)
        name = action.group(1).strip()
        arg = ainput.group(1).strip() if ainput else ""
        obs = TOOLS.get(name, lambda a: f"Unknown tool: {name}")(arg)
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": f"Observation: {obs}"})
    return "Stopped: reached the step limit."

if __name__ == "__main__":
    print(run(${JSON.stringify(task)}))`;
  }
  function copyCode() { navigator.clipboard.writeText(buildCode()).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500); }); }
  function downloadCode() { const blob = new Blob([buildCode()], { type: "text/x-python" }); const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = `${(name || "agent").replace(/\s+/g, "_").toLowerCase()}.py`; a.click(); URL.revokeObjectURL(a.href); }
  function agentConfig() {
    return {
      type: agentType, name, description, systemPrompt: goal, provider: providerId, model, temperature, maxIterations: maxIters,
      tools: [...enabledTools], knowledge: enabledTools.has("knowledge") ? knowledgeText : undefined,
      steps: agentType === "workflow" ? steps.map((s) => ({ name: s.name, instruction: s.instruction })) : undefined, task,
    };
  }
  async function saveAgent() {
    setMsg("");
    try {
      const r = await fetch("/api/projects", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ lab: "agent", name: name || "agent", config: agentConfig() }) });
      if (r.ok) { setSaved(true); setTimeout(() => setSaved(false), 1600); } else { const j = await r.json().catch(() => ({})); setMsg(j.error || "Could not save the agent."); }
    } catch (e) { setMsg((e as Error).message); }
  }
  function exportJson() { const blob = new Blob([JSON.stringify(agentConfig(), null, 2)], { type: "application/json" }); const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = `${(name || "agent").replace(/\s+/g, "_").toLowerCase()}.json`; a.click(); URL.revokeObjectURL(a.href); }
  async function loadAgents() {
    if (loadOpen) { setLoadOpen(false); return; }
    try { const r = await fetch("/api/projects?lab=agent"); const j = await r.json(); setSavedAgents(j.projects || []); } catch { setSavedAgents([]); }
    setLoadOpen(true);
  }
  /* eslint-disable @typescript-eslint/no-explicit-any */
  function applyConfig(cfg: any) {
    if (!cfg) return;
    setAgentType(cfg.type === "workflow" ? "workflow" : "react");
    if (cfg.name) setName(String(cfg.name));
    if (cfg.description) setDescription(String(cfg.description));
    if (cfg.systemPrompt) setGoal(String(cfg.systemPrompt));
    if (cfg.provider) { setProviderId(String(cfg.provider)); fetch(`/api/models?providerId=${cfg.provider}`).then((r) => r.json()).then((j) => setModelList(j.models || [])).catch(() => {}); }
    if (cfg.model) setModel(String(cfg.model));
    if (typeof cfg.temperature === "number") setTemperature(cfg.temperature);
    if (cfg.maxIterations) setMaxIters(Math.max(1, Math.min(10, Number(cfg.maxIterations))));
    if (Array.isArray(cfg.tools)) { const s = new Set<string>(cfg.tools.filter((t: string) => AGENT_TOOLS.some((x) => x.id === t))); setEnabledTools(s); setPlacedTools(new Set(s)); }
    if (cfg.knowledge) setKnowledgeText(String(cfg.knowledge));
    if (Array.isArray(cfg.steps)) setSteps(cfg.steps.map((s: any, i: number) => ({ id: `s${i + 1}`, name: String(s.name || `Step ${i + 1}`), instruction: String(s.instruction || "") })));
    if (cfg.task) setTask(String(cfg.task));
    setBuildMode(cfg.type === "workflow" ? "manual" : "visual"); setNodePos({}); setNodeStatus({}); setASel("agent"); setLoadOpen(false); setStep("build");
  }
  /* eslint-enable @typescript-eslint/no-explicit-any */

  // Load a saved agent when opened from My Projects (?project=<id>).
  useEffect(() => {
    const id = new URLSearchParams(window.location.search).get("project");
    if (!id) return;
    fetch(`/api/projects?id=${id}`).then((r) => r.json()).then(({ project }) => { if (project?.config) applyConfig(project.config); }).catch(() => {});
  }, []);

  const curType = TYPES.find((t) => t.id === agentType)!;
  const stepBtn = (k: Step, n: number, label: string) => (<button className={step === k ? "on" : ""} onClick={() => setStep(k)}><b>{n}</b>{label}</button>);
  const selNode = nodes.find((n) => n.id === aSel);

  return (
    <>
      <div className="lab-head">
        <div><div className="eyebrow">Lab 03 · orchestration</div><h2 className="page-h">Agent Lab</h2><p className="page-sub" style={{ margin: 0 }}>Pick an agent type, wire it up on a node canvas (or by form / from a prompt), then run it and watch every step.</p></div>
        <div className="acts" style={{ position: "relative" }}>
          <button className="btn ghost sm" onClick={loadAgents}>📂 Load</button>
          <button className="btn ghost sm" onClick={saveAgent}>{saved ? "Saved ✓" : "💾 Save"}</button>
          <button className="btn ghost sm" onClick={exportJson}>⬇ Export JSON</button>
          <button className="btn ghost sm" onClick={() => setShowCode(true)}>&lt;/&gt; Get code</button>
          {loadOpen && <div className="addmenu2" style={{ top: 38 }}><div className="hd">Saved agents</div>{savedAgents.length ? savedAgents.map((a) => <div key={a.id} className="ai" onClick={() => applyConfig(a.config)}>{a.name}</div>) : <div className="ai" style={{ color: "var(--faint)" }}>none saved yet</div>}</div>}
        </div>
      </div>
      {provKnown && !hasProvider && <div className="warnbar">No provider configured — an admin must add one under Admin → Providers before you can run an agent.</div>}
      {msg && <div className="err">{msg}</div>}
      <input ref={fileRef} type="file" accept=".txt,.md,.csv,.pdf,.docx,.doc,.xlsx,.xls" onChange={onKnowledgeFile} style={{ display: "none" }} />

      <div className="stepper">{stepBtn("type", 1, "Type")}{stepBtn("build", 2, "Build")}{stepBtn("run", 3, "Run")}</div>

      {/* STEP 1 — TYPE */}
      {step === "type" && (<>
        <div className="card" style={{ marginBottom: 16 }}>
          <div className="card-h"><span className="t">When do you actually need an agent?</span></div>
          <div className="card-b">
            <div className="whenuse">
              <div className="wu"><b>Single prompt</b><span>One question, one answer, no tools or steps. Cheapest & fastest — use the Prompting Lab.</span></div>
              <div className="wu"><b>Workflow</b><span>A fixed sequence you always run in the same order (plan → draft → review). Predictable & debuggable.</span></div>
              <div className="wu on"><b>Tool agent (ReAct)</b><span>The task needs live facts, math, APIs, or decisions the model must make itself. Powerful, but slower & pricier — you trade control for autonomy.</span></div>
            </div>
          </div>
        </div>
        <div className="card">
          <div className="card-h"><span className="t">Choose an agent type</span></div>
          <div className="card-b">
            <div className="model-grid">
              {TYPES.map((t) => (
                <div key={t.id} className={`model-card ${agentType === t.id ? "on" : ""}`} onClick={() => { setAgentType(t.id); setBuildMode(t.id === "react" ? "visual" : "manual"); }}>
                  <div className="mc-top"><span className="mc-name">{t.label}</span><span className="mc-fam">{t.tag}</span></div>
                  <div className="mc-desc">{t.desc}</div>
                  <div className="mc-foot">{agentType === t.id ? "selected" : "click to select"}</div>
                </div>
              ))}
            </div>
            <div className="stepnav"><button className="btn" onClick={() => setStep("build")}>Build from scratch →</button></div>
          </div>
        </div>
        <div className="card" style={{ marginTop: 16 }}>
          <div className="card-h"><span className="t">…or start from a template</span><span className="mono r">real, editable</span></div>
          <div className="card-b">
            <div className="model-grid">
              {TEMPLATES.map((t) => (
                <div key={t.id} className="model-card" onClick={() => applyTemplate(t)}>
                  <div className="mc-top"><span className="mc-name">{t.label}</span><span className="mc-fam">{t.tag}</span></div>
                  <div className="mc-desc">{t.desc}</div>
                  <div className="mc-foot">{t.tools.map((x) => TOOL_META[x]?.label).join(" · ")}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </>)}

      {/* STEP 2 — BUILD */}
      {step === "build" && (<>
        {agentType === "react" && <div className="seg" style={{ maxWidth: 420, marginBottom: 16 }}>
          <button className={buildMode === "visual" ? "on" : ""} onClick={() => setBuildMode("visual")}>Visual builder</button>
          <button className={buildMode === "manual" ? "on" : ""} onClick={() => setBuildMode("manual")}>Manual</button>
          <button className={buildMode === "prompt" ? "on" : ""} onClick={() => setBuildMode("prompt")}>From prompt</button>
        </div>}

        {/* REACT · VISUAL NODE CANVAS */}
        {agentType === "react" && buildMode === "visual" && (
          <div className={`agent-flow ${fullscreen ? "fs" : ""}`}>
          <div className="split" style={{ gridTemplateColumns: "1fr 320px" }}>
            <div className="card">
              <div className="card-h"><span className="t">Flow</span>
                <div className="r" style={{ position: "relative" }}>
                  <button className="btn ghost sm" onClick={() => setFullscreen((f) => !f)}>{fullscreen ? "⤢ Exit" : "⛶ Fullscreen"}</button>
                  <button className="btn ghost sm" onClick={() => setAddOpen((o) => !o)}>+ Add node</button>
                  <button className="btn sm" onClick={startRun} disabled={!hasProvider}>▶ Run</button>
                  {addOpen && (
                    <div className="addmenu2">
                      <div className="hd">Add a tool node to the canvas</div>
                      {AGENT_TOOLS.map((t) => { const on = placedTools.has(t.id); return <div key={t.id} className="ai" onClick={() => { togglePlaced(t.id); if (!on) setASel("tool:" + t.id); }}><span>{TOOL_META[t.id].icon}</span>{TOOL_META[t.id].label}<span className={`ai-state ${on ? "on" : ""}`}>{on ? "✓ on canvas" : "+ add"}</span></div>; })}
                    </div>
                  )}
                </div>
              </div>
              <div className="card-b" style={{ padding: 0 }}>{flowCanvas()}</div>
            </div>
            <div className="card">
              <div className="card-h"><span className="t">Configure</span><span className="mono r">{selNode?.title || "—"}</span></div>
              <div className="card-b" style={{ maxHeight: CANVAS_H, overflow: "auto" }}>
                {!selNode && <div className="note">Click a node to configure it.</div>}
                {selNode?.type === "trigger" && <div className="note">This is where the user&apos;s task enters the agent. You&apos;ll type the actual task on the Run step.</div>}
                {selNode?.type === "output" && <div className="note">The agent&apos;s Final Answer is returned here after the reasoning loop finishes.</div>}
                {selNode?.type === "agent" && (<>
                  <div className="insp-field"><div className="k">Agent name</div><input type="text" value={name} onChange={(e) => setName(e.target.value)} /></div>
                  <div className="insp-field"><div className="k">Description</div><input type="text" value={description} onChange={(e) => setDescription(e.target.value)} /></div>
                  <div className="insp-field"><div className="k">System prompt (role & goal)</div><textarea rows={4} value={goal} onChange={(e) => setGoal(e.target.value)} /></div>
                  <div className="insp-field"><div className="k">Max reasoning steps · {maxIters}</div><input type="range" min={1} max={10} value={maxIters} onChange={(e) => setMaxIters(+e.target.value)} /></div>
                  <div className="insp-field"><div className="k">Connected</div><div className="chip-row"><span className="c">{model || providerLabel}</span>{[...enabledTools].map((t) => <span key={t} className="c">{TOOL_META[t]?.label}</span>)}</div></div>
                </>)}
                {selNode?.type === "model" && (<>
                  <div className="insp-field"><div className="k">Provider</div>
                    <select value={providerId} onChange={(e) => loadModels(e.target.value)}>{providers.length ? providers.map((p) => <option key={p.id} value={p.id}>{p.label || p.provider}</option>) : <option>none configured</option>}</select>
                  </div>
                  <div className="insp-field"><div className="k">Model</div>
                    {modelList.length ? <select value={model} onChange={(e) => setModel(e.target.value)}>{modelList.map((m) => <option key={m}>{m}</option>)}</select> : <input type="text" value={model} onChange={(e) => setModel(e.target.value)} placeholder="model id" />}
                  </div>
                  <div className="insp-field"><div className="k">Temperature · {temperature}</div><input type="range" min={0} max={1} step={0.05} value={temperature} onChange={(e) => setTemperature(+e.target.value)} /></div>
                  <div className="insp-field"><div className="k">Max tokens</div><input type="text" value={maxTokens} onChange={(e) => setMaxTokens(Number(e.target.value) || 600)} /></div>
                </>)}
                {selNode?.type === "tool" && selNode.toolId !== "knowledge" && (<>
                  <div className="insp-field"><div className="k">Tool</div><div className="note" style={{ margin: 0 }}>{AGENT_TOOLS.find((t) => t.id === selNode.toolId)?.desc}</div></div>
                  <div className="insp-field"><div className="k">Example input the agent sends</div><div className="mono" style={{ fontSize: 11, color: "var(--muted)" }}>{AGENT_TOOLS.find((t) => t.id === selNode.toolId)?.example}</div></div>
                  <div className="row" style={{ gap: 8 }}>
                    {enabledTools.has(selNode.toolId!) ? <button className="btn ghost sm" onClick={() => disconnectTool(selNode.toolId!)}>Disconnect</button> : <button className="btn sm" onClick={() => connectTool(selNode.toolId!)}>Connect to agent</button>}
                    <button className="btn ghost sm" onClick={() => { removeToolNode(selNode.toolId!); setASel("agent"); }}>Remove node</button>
                  </div>
                  <div className="note" style={{ marginTop: 8 }}>Drag its top dot to the Agent to wire it, or click a wire and press <b>Delete</b>. {enabledTools.has(selNode.toolId!) ? "" : "Unconnected tools aren't used at run."}</div>
                </>)}
                {selNode?.type === "tool" && selNode.toolId === "knowledge" && (<>
                  <div className="insp-field"><div className="k">Knowledge base (the agent searches this)</div><textarea rows={5} value={knowledgeText} onChange={(e) => setKnowledgeText(e.target.value)} /></div>
                  <div className="row" style={{ gap: 8, flexWrap: "wrap" }}>
                    <button className="btn ghost sm" onClick={() => fileRef.current?.click()} disabled={uploading}>{uploading ? "Parsing…" : "Upload doc"}</button>
                    {enabledTools.has("knowledge") ? <button className="btn ghost sm" onClick={() => disconnectTool("knowledge")}>Disconnect</button> : <button className="btn sm" onClick={() => connectTool("knowledge")}>Connect</button>}
                    <button className="btn ghost sm" onClick={() => { removeToolNode("knowledge"); setASel("agent"); }}>Remove</button>
                  </div>
                  <div className="note" style={{ marginTop: 8 }}>{knowledgeText.split(/\s+/).filter(Boolean).length} words · pdf / docx / xlsx supported</div>
                </>)}
              </div>
            </div>
          </div>
          </div>
        )}

        {/* REACT · MANUAL FORM */}
        {agentType === "react" && buildMode === "manual" && (
          <div className="card"><div className="card-h"><span className="t">Agent configuration</span></div>
            <div className="card-b">
              <div className="split col-2e"><div><label className="fld">Agent name</label><input type="text" value={name} onChange={(e) => setName(e.target.value)} /></div><div><label className="fld">Description</label><input type="text" value={description} onChange={(e) => setDescription(e.target.value)} /></div></div>
              <label className="fld" style={{ marginTop: 12 }}>System prompt (role & goal)</label><textarea rows={3} value={goal} onChange={(e) => setGoal(e.target.value)} />
              <div className="split col-2e" style={{ marginTop: 12 }}>
                <div><label className="fld">Provider</label><select value={providerId} onChange={(e) => loadModels(e.target.value)}>{providers.length ? providers.map((p) => <option key={p.id} value={p.id}>{p.label || p.provider}</option>) : <option>none</option>}</select></div>
                <div><label className="fld">Model</label>{modelList.length ? <select value={model} onChange={(e) => setModel(e.target.value)}>{modelList.map((m) => <option key={m}>{m}</option>)}</select> : <input type="text" value={model} onChange={(e) => setModel(e.target.value)} />}</div>
              </div>
              <div className="split col-2e" style={{ marginTop: 12 }}><div><label className="fld">Temperature · {temperature}</label><input type="range" min={0} max={1} step={0.05} value={temperature} onChange={(e) => setTemperature(+e.target.value)} /></div><div><label className="fld">Max reasoning steps · {maxIters}</label><input type="range" min={1} max={10} value={maxIters} onChange={(e) => setMaxIters(+e.target.value)} /></div></div>
              <label className="fld" style={{ marginTop: 12 }}>Tools</label>
              <div className="checklist">{AGENT_TOOLS.map((t) => <span key={t.id} className={`chk ${enabledTools.has(t.id) ? "on" : ""}`} onClick={() => toggleManual(t.id)} title={t.desc}>{t.name}</span>)}</div>
              {enabledTools.has("knowledge") && (<>
                <label className="fld" style={{ marginTop: 12 }}>Knowledge base</label>
                <textarea rows={3} value={knowledgeText} onChange={(e) => setKnowledgeText(e.target.value)} />
                <div className="row" style={{ marginTop: 8, gap: 8 }}><button className="btn ghost sm" onClick={() => fileRef.current?.click()} disabled={uploading}>{uploading ? "Parsing…" : "Upload doc (pdf/docx/xlsx)"}</button><span className="note">{knowledgeText.split(/\s+/).filter(Boolean).length} words loaded</span></div>
              </>)}
              <div className="stepnav"><button className="btn ghost" onClick={() => setStep("type")}>← Back</button><button className="btn" onClick={startRun} disabled={!hasProvider}>Next: Run →</button></div>
            </div>
          </div>
        )}

        {/* FROM PROMPT (all types) */}
        {(buildMode === "prompt" || agentType !== "react") && (
          <>
            {agentType === "react" && buildMode === "prompt" && (
              <div className="card" style={{ marginBottom: 16 }}><div className="card-h"><span className="t">Describe your agent</span></div>
                <div className="card-b"><label className="fld">Tell the builder what the agent should do — the LLM drafts the config</label><textarea rows={3} value={describe} onChange={(e) => setDescribe(e.target.value)} /><div className="row" style={{ marginTop: 10 }}><button className="btn" onClick={generateFromPrompt} disabled={generating || !hasProvider}>{generating ? <><span className="busy-dot" />Generating…</> : "✦ Generate config"}</button><span className="note">fills the Visual builder</span></div></div>
              </div>
            )}
            {agentType === "workflow" && (<>
              <div className="card" style={{ marginBottom: 16 }}>
                <div className="card-h"><span className="t">Pipeline</span><span className="mono r">{steps.length} steps</span></div>
                <div className="card-b" style={{ padding: 0 }}>{wfCanvas()}</div>
              </div>
              <div className="card"><div className="card-h"><span className="t">Workflow steps</span></div>
                <div className="card-b">
                  <div className="split col-2e"><div><label className="fld">Name</label><input type="text" value={name} onChange={(e) => setName(e.target.value)} /></div><div><label className="fld">Temperature · {temperature}</label><input type="range" min={0} max={1} step={0.05} value={temperature} onChange={(e) => setTemperature(+e.target.value)} /></div></div>
                  <div className="split col-2e" style={{ marginTop: 12 }}>
                    <div><label className="fld">Provider</label><select value={providerId} onChange={(e) => loadModels(e.target.value)}>{providers.length ? providers.map((p) => <option key={p.id} value={p.id}>{p.label || p.provider}</option>) : <option>none</option>}</select></div>
                    <div><label className="fld">Model</label>{modelList.length ? <select value={model} onChange={(e) => setModel(e.target.value)}>{modelList.map((m) => <option key={m}>{m}</option>)}</select> : <input type="text" value={model} onChange={(e) => setModel(e.target.value)} />}</div>
                  </div>
                  <label className="fld" style={{ marginTop: 12 }}>Steps (run top to bottom, each sees the previous output)</label>
                  {steps.map((s, i) => (
                    <div key={s.id} className="wf-edit"><div className="wf-idx">{i + 1}</div><input type="text" value={s.name} onChange={(e) => setSteps((ss) => ss.map((x) => x.id === s.id ? { ...x, name: e.target.value } : x))} style={{ maxWidth: 150 }} /><input type="text" value={s.instruction} onChange={(e) => setSteps((ss) => ss.map((x) => x.id === s.id ? { ...x, instruction: e.target.value } : x))} /><button className="x" onClick={() => setSteps((ss) => ss.filter((x) => x.id !== s.id))}>×</button></div>
                  ))}
                  <button className="btn ghost sm" style={{ marginTop: 8 }} onClick={() => setSteps((ss) => [...ss, { id: `s${Date.now()}`, name: `Step ${ss.length + 1}`, instruction: "" }])}>+ Add step</button>
                  <div className="stepnav"><button className="btn ghost" onClick={() => setStep("type")}>← Back</button><button className="btn" onClick={startRun} disabled={!hasProvider}>Next: Run →</button></div>
                </div>
              </div>
            </>)}
          </>
        )}

        {agentType === "react" && buildMode === "visual" && <div className="stepnav"><button className="btn ghost" onClick={() => setStep("type")}>← Back</button><button className="btn" onClick={startRun} disabled={!hasProvider}>Next: Run →</button></div>}
      </>)}

      {/* STEP 3 — RUN */}
      {step === "run" && (<>
        {agentType === "react" && (<>
          <div className="card" style={{ marginBottom: 16 }}>
            <div className="card-h"><span className="t">Execution flow</span><span className="mono r">{running ? "running…" : finalOut ? "finished ✓" : "idle"}</span></div>
            <div className="card-b" style={{ padding: 0 }}>{flowCanvas()}</div>
          </div>
          {pendingApproval && (
            <div className="approval">
              <div className="ap-q"><span className="ap-ic">🙋</span><div><div className="ap-title">Human approval needed</div><div className="ap-text">{pendingApproval.q}</div></div></div>
              <div className="row" style={{ gap: 8 }}><button className="btn" onClick={() => { pendingApproval.resolve("User APPROVED."); setPendingApproval(null); }}>✓ Approve</button><button className="btn ghost" onClick={() => { pendingApproval.resolve("User DENIED the action."); setPendingApproval(null); }}>✗ Deny</button></div>
            </div>
          )}
          {metrics && !running && <div className="ag-metrics"><span className="m"><b>{metrics.calls}</b> LLM calls</span><span className="m"><b>{metrics.tools}</b> tool calls</span><span className="m"><b>{(metrics.ms / 1000).toFixed(1)}s</b> latency</span><span className="m"><b>~{metrics.tokens >= 1000 ? (metrics.tokens / 1000).toFixed(1) + "k" : metrics.tokens}</b> tokens</span></div>}
          <div className="split col-2">
            <div className="card"><div className="card-h"><span className="t">Task</span></div>
              <div className="card-b">
                <label className="fld">What should {name} do?</label><textarea rows={3} value={task} onChange={(e) => setTask(e.target.value)} />
                <div className="row" style={{ marginTop: 12 }}><button className="btn" onClick={runReact} disabled={running || !hasProvider}>{running ? <><span className="busy-dot" />Running…</> : "▶ Run agent"}</button><span className="note">{toolList.length} tools · {model || providerLabel} · max {maxIters} steps</span></div>
                {finalOut && <><label className="fld" style={{ marginTop: 16 }}>Final answer</label><div className="out">{finalOut}</div></>}
              </div>
            </div>
            <div className="card"><div className="card-h"><span className="t">Reasoning trace</span><span className="mono r">{trace.length} steps</span></div>
              <div className="card-b" ref={scrollRef} style={{ maxHeight: 460, overflow: "auto" }}>
                {trace.length === 0 && <div className="note">Run the agent to watch the Thought → Action → Observation loop.</div>}
                {trace.map((t, i) => (<div key={i} className={`ag-step ${t.kind} ${t.state}`}><div className="ag-k">{t.kind === "action" ? `action · ${t.tool}` : t.kind === "final" ? "final answer" : t.kind}</div><div className="ag-t">{t.kind === "thought" && t.state === "active" ? <><span className="busy-dot" />{t.text}</> : t.text}</div></div>))}
              </div>
            </div>
          </div>
        </>)}
        {agentType === "workflow" && (<>
          <div className="card" style={{ marginBottom: 16 }}>
            <div className="card-h"><span className="t">Execution flow</span><span className="mono r">{running ? "running…" : finalOut ? "finished ✓" : "idle"}</span></div>
            <div className="card-b" style={{ padding: 0 }}>{wfCanvas()}</div>
          </div>
          <div className="card"><div className="card-h"><span className="t">Workflow · {name}</span><span className="mono r">{steps.length} steps</span></div>
            <div className="card-b" ref={scrollRef} style={{ maxHeight: 560, overflow: "auto" }}>
              <label className="fld">Input</label><textarea rows={2} value={task} onChange={(e) => setTask(e.target.value)} />
              <div className="row" style={{ margin: "12px 0" }}><button className="btn" onClick={runWorkflow} disabled={running || !hasProvider}>{running ? <><span className="busy-dot" />Running…</> : "▶ Run workflow"}</button></div>
              {wfOutputs.map((o, i) => (<div key={i} className={`wf-run ${o.state}`}><div className="wf-run-h"><span className="wf-idx">{i + 1}</span><b>{o.name}</b>{o.state === "active" && <span className="busy-dot" style={{ marginLeft: 8 }} />}{o.state === "done" && <span className="badge good" style={{ marginLeft: 8 }}>done</span>}</div>{o.text && <div className="out" style={{ marginTop: 8 }}>{o.text}</div>}</div>))}
            </div>
          </div>
        </>)}
        <div className="stepnav"><button className="btn ghost" onClick={() => setStep("build")}>← Back to build</button></div>
      </>)}

      <div className={`modal-wrap ${showCode ? "show" : ""}`} onClick={(e) => { if (e.target === e.currentTarget) setShowCode(false); }}>
        <div className="modal"><div className="mh"><b>Agent code · {curType.label}</b><div className="r" style={{ marginLeft: "auto", display: "flex", gap: 8 }}><button className="btn ghost sm" onClick={copyCode}>{copied ? "Copied ✓" : "Copy"}</button><button className="btn sm" onClick={downloadCode}>Download</button></div><button className="x" onClick={() => setShowCode(false)}>×</button></div><div className="mb"><div className="note" style={{ marginBottom: 10 }}>Where to use it: run this Python (<code>pip install openai</code>, set <code>OPENAI_BASE_URL</code> &amp; <code>OPENAI_API_KEY</code>) · or <b>💾 Save</b> to My Projects · or <b>⬇ Export JSON</b> to load the config into your own app.</div><div className="code">{buildCode()}</div></div></div>
      </div>
    </>
  );
}
