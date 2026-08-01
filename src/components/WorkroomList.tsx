"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

type Agent = { id: string; name: string; model: string; runtime: string; kind: string; toolCount: number; kbCount: number };

export default function WorkroomList() {
  const [agents, setAgents] = useState<Agent[] | null>(null);

  useEffect(() => {
    fetch("/api/agent/published").then((r) => r.json()).then((j) => setAgents(j.agents || [])).catch(() => setAgents([]));
  }, []);

  if (agents === null) return <div className="note" style={{ marginTop: 20 }}>Loading…</div>;

  if (agents.length === 0) return (
    <div className="empty-card" style={{ marginTop: 20 }}>
      <div style={{ fontSize: 30, marginBottom: 8 }}>🚀</div>
      <b>No published agents yet</b>
      <p className="note" style={{ maxWidth: 420, margin: "6px auto 14px" }}>Build an agent in the Agent Lab, then hit <b>Publish</b>. It’ll appear here ready to chat with and deploy.</p>
      <Link href="/labs/agent" className="btn">Open Agent Lab</Link>
    </div>
  );

  return (
    <div className="card-grid" style={{ marginTop: 18 }}>
      {agents.map((a) => (
        <Link key={a.id} href={`/workroom/${a.id}`} className="agent-card">
          <div className="agent-card-top"><span className="agent-ic">🤖</span><b>{a.name}</b></div>
          <div className="note" style={{ marginTop: 4 }}>{a.kind} · {a.model || "no model"}</div>
          <div className="agent-card-meta">
            <span className="chip">{a.runtime === "nat" ? "⚡ NAT" : "🌐 In-browser"}</span>
            <span className="chip">{a.toolCount} tool{a.toolCount === 1 ? "" : "s"}</span>
            {a.kbCount > 0 && <span className="chip">📚 knowledge</span>}
          </div>
          <span className="agent-card-cta">Open chat →</span>
        </Link>
      ))}
    </div>
  );
}
