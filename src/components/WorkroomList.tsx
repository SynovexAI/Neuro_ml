"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { toast, confirmDialog } from "@/lib/toast";

type Agent = { id: string; name: string; model: string; runtime: string; kind: string; toolCount: number; kbCount: number; runs: number; tokens: number; lastUsed: string | null };

function ago(s: string | null): string {
  if (!s) return "never used";
  const d = new Date(s); if (isNaN(d.getTime())) return "";
  const mins = Math.floor((Date.now() - d.getTime()) / 60000);
  if (mins < 1) return "just now"; if (mins < 60) return `${mins}m ago`;
  const h = Math.floor(mins / 60); if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}
const fmt = (n: number) => (n >= 1000 ? (n / 1000).toFixed(1) + "k" : String(n));

export default function WorkroomList() {
  const [agents, setAgents] = useState<Agent[] | null>(null);

  function load() { fetch("/api/agent/published").then((r) => r.json()).then((j) => setAgents(j.agents || [])).catch(() => setAgents([])); }
  useEffect(() => { load(); }, []);

  async function unpublish(e: React.MouseEvent, a: Agent) {
    e.preventDefault(); e.stopPropagation();
    if (!(await confirmDialog(`Remove “${a.name}” from the Workroom? The saved agent stays in your projects; only its published copy and channels stop being usable.`, { danger: true, confirmLabel: "Unpublish" }))) return;
    const res = await fetch("/api/projects", { method: "PATCH", headers: { "content-type": "application/json" }, body: JSON.stringify({ id: a.id, published: false }) });
    toast(res.ok ? "Unpublished" : "Failed", res.ok ? "success" : "error");
    if (res.ok) setAgents((xs) => (xs || []).filter((x) => x.id !== a.id));
  }

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
        <div key={a.id} className="agent-card-wrap">
          <button className="card-unpub" title="Remove from Workroom" onClick={(e) => unpublish(e, a)}>×</button>
          <Link href={`/workroom/${a.id}`} className="agent-card">
            <div className="agent-card-top"><span className="agent-ic">🤖</span><b>{a.name}</b></div>
            <div className="note" style={{ marginTop: 4 }}>{a.kind} · {a.model || "no model"}</div>
            <div className="agent-card-meta">
              <span className="chip">{a.runtime === "nat" ? "⚡ NAT" : "🌐 In-browser"}</span>
              <span className="chip">{a.toolCount} tool{a.toolCount === 1 ? "" : "s"}</span>
              {a.kbCount > 0 && <span className="chip">📚 knowledge</span>}
            </div>
            <div className="agent-card-stats">{a.runs} run{a.runs === 1 ? "" : "s"} · {fmt(a.tokens)} tokens · {ago(a.lastUsed)}</div>
            <span className="agent-card-cta">Open chat →</span>
          </Link>
        </div>
      ))}
    </div>
  );
}
