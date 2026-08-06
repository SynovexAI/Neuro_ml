"use client";

import { useEffect, useRef, useState } from "react";
import Markdown from "@/components/Markdown";

type Msg = { role: "user" | "assistant"; text: string };

export default function EmbedChat({ channelId, agentName }: { channelId: string; agentName: string }) {
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const scroller = useRef<HTMLDivElement>(null);
  useEffect(() => { scroller.current?.scrollTo({ top: scroller.current.scrollHeight }); }, [msgs, busy]);

  async function send() {
    const q = input.trim(); if (!q || busy) return;
    setInput(""); setMsgs((m) => [...m, { role: "user", text: q }]); setBusy(true);
    try {
      const res = await fetch(`/api/agent/public/${channelId}`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ message: q }) });
      const j = await res.json().catch(() => null);
      setMsgs((m) => [...m, { role: "assistant", text: res.ok ? (j?.answer || "(no answer)") : `⚠ ${j?.error || "Failed"}` }]);
    } catch { setMsgs((m) => [...m, { role: "assistant", text: "⚠ Network error" }]); }
    setBusy(false);
  }

  return (
    <div className="embed-wrap">
      <div className="embed-head"><span className="agent-ic">🤖</span><b>{agentName}</b></div>
      <div className="embed-scroll" ref={scroller}>
        {msgs.length === 0 && <div className="note" style={{ padding: 12 }}>Ask me anything.</div>}
        {msgs.map((m, i) => (
          <div key={i} className={`embed-msg ${m.role}`}>{m.role === "user" ? m.text : <Markdown text={m.text} />}</div>
        ))}
        {busy && <div className="embed-msg assistant note">thinking…</div>}
      </div>
      <div className="embed-composer">
        <input value={input} placeholder="Message…" onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") send(); }} />
        <button className="btn" onClick={send} disabled={busy || !input.trim()}>Send</button>
      </div>
    </div>
  );
}
