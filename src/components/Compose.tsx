"use client";

import { useEffect, useRef, useState } from "react";
import { chunkText, buildIndex } from "@/lib/ragUtils";
import { AGENT_TOOLS, reactSystemPrompt, parseReAct, type ToolCtx } from "@/lib/agentTools";

type RagCfg = { docs: { name: string; kind: string; text: string }[]; size?: number; overlap?: number };
type SavedRag = { id: string; name: string; config: RagCfg };
type TraceItem = { kind: string; text: string; tool?: string };

async function chatOnce(messages: { role: string; content: string }[], providerId?: string, model?: string): Promise<string> {
  const res = await fetch("/api/chat", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ messages, temperature: 0.2, maxTokens: 500, streaming: false, providerId: providerId || undefined, model: model || undefined }) });
  if (!res.ok) { const j = await res.json().catch(() => ({ error: "request failed" })); throw new Error(j.error || "request failed"); }
  return (await res.text()).trim();
}

export default function Compose() {
  const [rags, setRags] = useState<SavedRag[]>([]);
  const [ragId, setRagId] = useState("");
  const [question, setQuestion] = useState("My order arrived damaged — what can I do and by when?");
  const [running, setRunning] = useState(false);
  const [trace, setTrace] = useState<TraceItem[]>([]);
  const [answer, setAnswer] = useState("");
  const [status, setStatus] = useState<Record<string, string>>({});
  const [provReady, setProvReady] = useState(false);
  const [hasProvider, setHasProvider] = useState(false);
  const [providerId, setProviderId] = useState("");
  const [model, setModel] = useState("");
  const [msg, setMsg] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch("/api/projects?lab=rag").then((r) => r.json()).then((j) => { const list = (j.projects || []) as SavedRag[]; setRags(list); if (list[0]) setRagId(list[0].id); }).catch(() => {});
    fetch("/api/models").then((r) => r.json()).then((j) => { setProvReady(true); const ps = j.providers || []; setHasProvider(ps.length > 0); if (ps.length) { setProviderId(j.providerId || ps[0].id); setModel(j.default || (j.models && j.models[0]) || ""); } }).catch(() => setProvReady(true));
  }, []);
  useEffect(() => { if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight; }, [trace]);

  const rag = rags.find((r) => r.id === ragId);

  async function run() {
    if (!rag) { setMsg("Save a RAG build in the RAG Lab first, then pick it here."); return; }
    setMsg(""); setRunning(true); setTrace([]); setAnswer(""); setStatus({ user: "done", agent: "running" });
    // rebuild the RAG index from the saved documents
    const chunks: string[] = [];
    for (const d of rag.config.docs || []) for (const c of chunkText(d.text, rag.config.size || 40, rag.config.overlap || 8)) chunks.push(c);
    const ctx: ToolCtx = { knowledgeIndex: buildIndex(chunks), knowledgeChunks: chunks };
    const tools = AGENT_TOOLS.filter((t) => ["knowledge", "calculator"].includes(t.id));
    const messages = [
      { role: "system", content: reactSystemPrompt(tools, `You are a grounded assistant. Use the knowledge tool to answer strictly from the connected documents ("${rag.name}"), and cite what you used. If it isn't in the documents, say so.`) },
      { role: "user", content: question },
    ];
    const push = (t: TraceItem) => setTrace((tr) => [...tr, t]);
    try {
      for (let i = 0; i < 6; i++) {
        setStatus((s) => ({ ...s, agent: "running" }));
        const resp = await chatOnce(messages, providerId, model);
        const p = parseReAct(resp);
        if (p.thought) push({ kind: "thought", text: p.thought });
        if (p.final || (!p.action && !p.final)) { const ans = p.final || resp; setAnswer(ans); push({ kind: "final", text: ans }); setStatus((s) => ({ ...s, agent: "done", answer: "done" })); break; }
        const tool = tools.find((t) => t.name.toLowerCase() === (p.action || "").toLowerCase());
        push({ kind: "action", text: p.input || "", tool: p.action });
        if (p.action === "knowledge") setStatus((s) => ({ ...s, rag: "running" }));
        const obs = tool ? await tool.run(p.input || "", ctx) : `Unknown tool "${p.action}".`;
        if (p.action === "knowledge") setStatus((s) => ({ ...s, rag: "done" }));
        push({ kind: "observation", text: obs });
        messages.push({ role: "assistant", content: resp }); messages.push({ role: "user", content: `Observation: ${obs}` });
      }
    } catch (e) { push({ kind: "error", text: (e as Error).message }); }
    setRunning(false);
  }

  const chain = [
    { id: "user", icon: "💬", label: "You", sub: "question" },
    { id: "agent", icon: "🤖", label: "Agent", sub: "ReAct" },
    { id: "rag", icon: "📚", label: rag ? rag.name.slice(0, 16) : "RAG", sub: "tool" },
    { id: "answer", icon: "✅", label: "Answer", sub: "grounded" },
  ];

  return (
    <>
      <div className="lab-head">
        <div><div className="eyebrow">Studio · Compose</div><h2 className="page-h">Compose</h2><p className="page-sub" style={{ margin: 0 }}>Chain your builds into one system — here, an <b>Agent</b> that uses a saved <b>RAG</b> bot as a tool, grounded in your documents.</p></div>
      </div>
      {provReady && !hasProvider && <div className="warnbar">No provider configured — an admin must add one under Admin → Providers before running.</div>}
      {msg && <div className="err">{msg}</div>}

      {rags.length === 0 ? (
        <div className="hero"><h2>No saved RAG builds yet</h2><p>Open the <b>RAG Lab</b>, add sources, then click <b>💾 Save</b>. Your saved RAG bot will appear here to compose into an agent.</p></div>
      ) : (
        <>
          <div className="card">
            <div className="card-h"><span className="t">Wire it up</span></div>
            <div className="card-b">
              <div className="row" style={{ gap: 10, flexWrap: "wrap", marginBottom: 16 }}>
                <div><label className="fld">RAG bot (tool)</label><select value={ragId} onChange={(e) => setRagId(e.target.value)}>{rags.map((r) => <option key={r.id} value={r.id}>{r.name}</option>)}</select></div>
                {rag && <span className="pill" style={{ alignSelf: "flex-end" }}>{rag.config.docs?.length || 0} docs · {(rag.config.docs || []).reduce((a, d) => a + d.text.split(/\s+/).length, 0)} words</span>}
              </div>
              <div className="compose-chain">
                {chain.map((n, i) => (
                  <div key={n.id} style={{ display: "contents" }}>
                    <div className={`anode ${status[n.id] || ""}`} style={{ position: "static", width: 150 }}>
                      <div className="ah"><span className="aic">{n.icon}</span><div><div className="atitle">{n.label}</div><div className="asub">{n.sub}</div></div><span className="abadge" /></div>
                    </div>
                    {i < chain.length - 1 && <span className="compose-arrow">→</span>}
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="split col-2" style={{ marginTop: 16 }}>
            <div className="card">
              <div className="card-h"><span className="t">Ask</span></div>
              <div className="card-b">
                <label className="fld">Question for the composed system</label>
                <textarea rows={3} value={question} onChange={(e) => setQuestion(e.target.value)} />
                <div className="row" style={{ marginTop: 12 }}><button className="btn" onClick={run} disabled={running || !hasProvider}>{running ? <><span className="busy-dot" />Running…</> : "▶ Run system"}</button><span className="note">the agent calls your RAG bot as a tool</span></div>
                {answer && <><label className="fld" style={{ marginTop: 16 }}>Grounded answer</label><div className="out">{answer}</div></>}
              </div>
            </div>
            <div className="card">
              <div className="card-h"><span className="t">Trace</span><span className="mono r">{trace.length} steps</span></div>
              <div className="card-b" ref={scrollRef} style={{ maxHeight: 420, overflow: "auto" }}>
                {trace.length === 0 && <div className="note">Run to watch the agent reason and call your RAG bot.</div>}
                {trace.map((t, i) => <div key={i} className={`ag-step ${t.kind}`}><div className="ag-k">{t.kind === "action" ? `action · ${t.tool}` : t.kind === "final" ? "final answer" : t.kind}</div><div className="ag-t">{t.text}</div></div>)}
              </div>
            </div>
          </div>
        </>
      )}
    </>
  );
}
