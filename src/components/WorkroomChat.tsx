"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import Markdown from "@/components/Markdown";

export type AgentCfg = {
  agentName?: string; agentType?: string; providerId?: string; model?: string;
  temperature?: number; systemPrompt?: string; tools?: string[]; mcpIds?: string[]; kbIds?: string[];
};
type Msg = { role: "user" | "assistant"; text: string; ms?: number };

export default function WorkroomChat({ agentName, cfg }: { agentName: string; cfg: AgentCfg }) {
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const scroller = useRef<HTMLDivElement>(null);

  useEffect(() => { scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: "smooth" }); }, [msgs, busy]);

  async function send(text?: string) {
    const q = (text ?? input).trim();
    if (!q || busy) return;
    setInput("");
    const history = msgs.map((m) => `${m.role === "user" ? "User" : "Assistant"}: ${m.text}`).join("\n");
    setMsgs((m) => [...m, { role: "user", text: q }]);
    setBusy(true);
    const started = performance.now();
    try {
      const res = await fetch("/api/agent/nat-run", {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({
          task: history ? `${history}\nUser: ${q}` : q,
          providerId: cfg.providerId, model: cfg.model, systemPrompt: cfg.systemPrompt,
          agentType: cfg.agentType || "react_agent", tools: cfg.tools || [], temperature: cfg.temperature ?? 0.2,
          mcpServerIds: cfg.mcpIds || [], knowledgeBaseIds: cfg.kbIds || [],
        }),
      });
      const j = await res.json().catch(() => null);
      const ms = Math.round(performance.now() - started);
      setMsgs((m) => [...m, { role: "assistant", text: res.ok ? (j?.answer || "(no answer)") : `⚠ ${j?.error || `Failed (${res.status})`}`, ms }]);
    } catch (e) {
      setMsgs((m) => [...m, { role: "assistant", text: `⚠ ${(e as Error).message}` }]);
    }
    setBusy(false);
  }

  const starters = ["What can you help me with?", "Give me a quick example of what you do.", "Summarize your instructions in one line."];

  return (
    <div className="wr-chat">
      <div className="wr-chat-head">
        <Link href="/workroom" className="btn ghost sm">← Workroom</Link>
        <div className="wr-chat-title"><span className="agent-ic">🤖</span><b>{agentName}</b><span className="badge">{(cfg.agentType || "react_agent") === "tool_calling_agent" ? "Tool-calling" : "ReAct"}</span></div>
        <span style={{ flex: 1 }} />
        <Link href="/workroom/channels" className="btn ghost sm">🚀 Deploy</Link>
        {msgs.length > 0 && <button className="btn ghost sm" onClick={() => setMsgs([])}>Clear</button>}
      </div>

      <div className="wr-scroll" ref={scroller}>
        {msgs.length === 0 ? (
          <div className="wr-welcome">
            <div className="agent-ic big">🤖</div>
            <h3>Chat with {agentName}</h3>
            <p className="note">Powered by the NVIDIA NeMo Agent Toolkit. Ask anything — it uses its tools and knowledge automatically.</p>
            <div className="wr-starters">{starters.map((s) => <button key={s} className="wr-starter" onClick={() => send(s)}>{s}</button>)}</div>
          </div>
        ) : msgs.map((m, i) => (
          <div key={i} className={`wr-msg ${m.role}`}>
            <div className="wr-ava">{m.role === "user" ? "🧑" : "🤖"}</div>
            <div className="wr-bubble">
              {m.role === "user" ? <span style={{ whiteSpace: "pre-wrap" }}>{m.text}</span> : <Markdown text={m.text} />}
              {m.role === "assistant" && m.ms != null && <div className="wr-meta">{(m.ms / 1000).toFixed(1)}s</div>}
            </div>
          </div>
        ))}
        {busy && <div className="wr-msg assistant"><div className="wr-ava">🤖</div><div className="wr-bubble"><span className="wr-typing"><i /><i /><i /></span></div></div>}
      </div>

      <div className="wr-composer">
        <textarea rows={1} placeholder={`Message ${agentName}…`} value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }} />
        <button className="btn" onClick={() => send()} disabled={busy || !input.trim()}>Send</button>
      </div>
    </div>
  );
}
