"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { toast, confirmDialog } from "@/lib/toast";

type Agent = { id: string; name: string };
type Channel = { id: string; type: string; agentName: string; enabled: boolean; dailyLimit: number | null; hasSecret: boolean };
type Kind = "telegram" | "widget" | "api";
type Result = { type: Kind; id: string; apiKey?: string; botUsername?: string; webhookUrl?: string };

const KINDS: { id: Kind; icon: string; label: string; desc: string }[] = [
  { id: "telegram", icon: "✈️", label: "Telegram bot", desc: "Chat with your agent inside Telegram. Needs a bot token from @BotFather." },
  { id: "widget", icon: "💬", label: "Web widget", desc: "Embed a chat bubble on any website with one <iframe> snippet." },
  { id: "api", icon: "🔑", label: "REST API", desc: "Call your agent from code with an API key. Great for integrations." },
];

export default function ChannelsManager() {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [channels, setChannels] = useState<Channel[]>([]);
  const [origin, setOrigin] = useState("");
  const [kind, setKind] = useState<Kind | null>(null);
  const [projectId, setProjectId] = useState("");
  const [token, setToken] = useState("");
  const [publicUrl, setPublicUrl] = useState("");
  const [cap, setCap] = useState(200);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<Result | null>(null);

  async function load() {
    const [a, c] = await Promise.all([
      fetch("/api/agent/published").then((r) => r.json()).catch(() => ({ agents: [] })),
      fetch("/api/channels").then((r) => r.json()).catch(() => ({ channels: [] })),
    ]);
    setAgents(a.agents || []); setChannels(c.channels || []);
  }
  useEffect(() => { setOrigin(window.location.origin); setPublicUrl(window.location.origin); load(); }, []);

  function openKind(k: Kind) {
    if (agents.length === 0) { toast("Publish an agent first — deploy it from here after.", "info"); return; }
    setKind(k); setProjectId(agents[0].id); setToken(""); setCap(200); setResult(null);
  }

  async function create() {
    if (!projectId) return;
    setBusy(true);
    try {
      const body: Record<string, unknown> = { type: kind, projectId };
      if (kind === "telegram") { body.token = token.trim(); body.publicUrl = publicUrl.trim(); }
      if (kind === "widget" || kind === "api") body.dailyLimit = cap;
      const res = await fetch("/api/channels", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
      const j = await res.json().catch(() => null);
      if (!res.ok) { toast(j?.error || "Deploy failed", "error"); return; }
      setKind(null);
      setResult({ type: kind!, id: j.id, apiKey: j.apiKey, botUsername: j.botUsername, webhookUrl: j.webhookUrl });
      toast("Deployed", "success");
      load();
    } catch (e) { toast((e as Error).message, "error"); }
    finally { setBusy(false); }
  }

  async function patch(id: string, body: Record<string, unknown>) {
    const res = await fetch("/api/channels", { method: "PATCH", headers: { "content-type": "application/json" }, body: JSON.stringify({ id, ...body }) });
    if (!res.ok) { toast("Update failed", "error"); return; }
    load();
  }
  async function toggleEnabled(c: Channel) {
    await patch(c.id, { enabled: !c.enabled });
    toast(c.enabled ? "Paused" : "Resumed", "success");
  }
  async function editCap(c: Channel) {
    const cur = c.dailyLimit ?? 200;
    const v = window.prompt(`Daily message cap for this ${c.type} (max runs/day):`, String(cur));
    if (v == null) return;
    const n = Math.max(1, Math.min(100_000, Math.round(Number(v) || cur)));
    await patch(c.id, { dailyLimit: n });
    toast(`Daily cap set to ${n}`, "success");
  }

  async function remove(c: Channel) {
    const ok = await confirmDialog(`Remove this ${c.type} deployment of “${c.agentName}”?`, { danger: true, confirmLabel: "Remove" });
    if (!ok) return;
    const res = await fetch(`/api/channels?id=${c.id}`, { method: "DELETE" });
    toast(res.ok ? "Removed" : "Failed to remove", res.ok ? "success" : "error");
    load();
  }

  const copy = (s: string) => { navigator.clipboard?.writeText(s); toast("Copied", "success"); };
  const embedSnippet = (id: string) => `<iframe src="${origin}/embed/${id}" style="width:400px;height:600px;border:1px solid #e5e7eb;border-radius:12px" title="chat"></iframe>`;
  const curl = (id: string) => `curl -X POST ${origin}/api/agent/public/${id} \\\n  -H "Authorization: Bearer YOUR_API_KEY" \\\n  -H "content-type: application/json" \\\n  -d '{"message":"Hello!"}'`;
  const kindMeta = (t: string) => KINDS.find((k) => k.id === t);

  return (
    <>
      <div className="sec-head" style={{ marginTop: 18 }}><b>Deploy to a channel</b><span className="note">pick where your agent should run</span></div>
      <div className="card-grid">
        {KINDS.map((k) => (
          <button key={k.id} className="agent-card" style={{ textAlign: "left", cursor: "pointer" }} onClick={() => openKind(k.id)}>
            <div className="agent-card-top"><span className="agent-ic">{k.icon}</span><b>{k.label}</b></div>
            <div className="note" style={{ marginTop: 8, lineHeight: 1.5 }}>{k.desc}</div>
            <span className="agent-card-cta" style={{ marginTop: 12 }}>Set up →</span>
          </button>
        ))}
      </div>

      <div className="sec-head" style={{ marginTop: 26 }}><b>Live deployments</b><span className="note">{channels.length} active</span></div>
      {channels.length === 0 ? (
        <div className="note" style={{ padding: "10px 0" }}>No deployments yet. Pick a channel above to get started.</div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {channels.map((c) => (
            <div key={c.id} className="row" style={{ alignItems: "center", gap: 12, border: "1px solid var(--border)", borderRadius: 10, padding: "11px 14px", opacity: c.enabled ? 1 : 0.6 }}>
              <span className="agent-ic">{kindMeta(c.type)?.icon || "🔌"}</span>
              <div style={{ flex: 1 }}><b style={{ fontSize: 13.5 }}>{c.agentName}</b><div className="note">{kindMeta(c.type)?.label || c.type}{(c.type === "widget" || c.type === "api") ? ` · cap ${c.dailyLimit ?? 200}/day` : ""}</div></div>
              {c.type === "widget" && <Link href={`/embed/${c.id}`} target="_blank" className="btn ghost sm">Open</Link>}
              {c.type === "widget" && <button className="btn ghost sm" onClick={() => copy(embedSnippet(c.id))}>Copy embed</button>}
              {c.type === "api" && <button className="btn ghost sm" onClick={() => copy(`${origin}/api/agent/public/${c.id}`)}>Copy URL</button>}
              {(c.type === "widget" || c.type === "api") && <button className="btn ghost sm" onClick={() => editCap(c)}>Cap</button>}
              <button className="btn ghost sm" onClick={() => toggleEnabled(c)}>{c.enabled ? "Pause" : "Resume"}</button>
              <span className="badge" style={{ color: c.enabled ? "#3b9e5f" : "var(--faint)" }}>{c.enabled ? "live" : "paused"}</span>
              <button className="iconbtn" title="Remove" onClick={() => remove(c)}>🗑</button>
            </div>
          ))}
        </div>
      )}

      {/* Deploy form modal */}
      {kind && (
        <div className="modal-wrap show" onClick={(e) => { if (e.target === e.currentTarget) setKind(null); }}>
          <div className="modal" style={{ maxWidth: 520 }}>
            <div className="mh"><b>{kindMeta(kind)?.icon} {kindMeta(kind)?.label}</b><button className="x" onClick={() => setKind(null)}>×</button></div>
            <div className="mb">
              <label className="fld">Agent</label>
              <select value={projectId} onChange={(e) => setProjectId(e.target.value)}>{agents.map((a) => <option key={a.id} value={a.id}>{a.name}</option>)}</select>

              {kind === "telegram" && (<>
                <label className="fld" style={{ marginTop: 12 }}>Bot token <span className="note">from @BotFather</span></label>
                <input type="password" placeholder="123456789:ABCdef..." value={token} onChange={(e) => setToken(e.target.value)} />
                <label className="fld" style={{ marginTop: 12 }}>Public app URL <span className="note">must be public HTTPS — not localhost</span></label>
                <input type="text" placeholder="https://your-app.com" value={publicUrl} onChange={(e) => setPublicUrl(e.target.value)} />
                <div className="teach-note" style={{ marginTop: 12 }}><span className="ic">ℹ️</span><span>Telegram must reach a public HTTPS URL to deliver messages. On localhost, use a tunnel (e.g. ngrok) and paste that URL here.</span></div>
              </>)}
              {(kind === "widget" || kind === "api") && (<>
                <label className="fld" style={{ marginTop: 12 }}>Daily message cap <span className="note">runs/day before it pauses — protects your token budget</span></label>
                <input type="number" min={1} max={100000} value={cap} onChange={(e) => setCap(Math.max(1, Math.round(Number(e.target.value) || 200)))} />
              </>)}
              {kind === "widget" && <div className="teach-note" style={{ marginTop: 12 }}><span className="ic">💬</span><span>You’ll get an <code>&lt;iframe&gt;</code> snippet to paste on any site. Anyone with the link can chat with this agent — runs use <b>your</b> provider quota, so keep the cap sensible.</span></div>}
              {kind === "api" && <div className="teach-note" style={{ marginTop: 12 }}><span className="ic">🔑</span><span>You’ll get an API key shown <b>once</b>. Store it safely — it authorizes agent runs billed to your account.</span></div>}

              <div className="row" style={{ marginTop: 16, justifyContent: "flex-end", gap: 8 }}>
                <button className="btn ghost" onClick={() => setKind(null)}>Cancel</button>
                <button className="btn" onClick={create} disabled={busy || (kind === "telegram" && !token.trim())}>{busy ? "Deploying…" : "Deploy"}</button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Result modal */}
      {result && (
        <div className="modal-wrap show" onClick={(e) => { if (e.target === e.currentTarget) setResult(null); }}>
          <div className="modal" style={{ maxWidth: 560 }}>
            <div className="mh"><b>✅ Deployed</b><button className="x" onClick={() => setResult(null)}>×</button></div>
            <div className="mb">
              {result.type === "telegram" && <>
                <p className="note">Your bot is live{result.botUsername ? <> as <b>@{result.botUsername}</b></> : ""}. Open Telegram, find it, and send a message.</p>
                {result.botUsername && <a className="btn" href={`https://t.me/${result.botUsername}`} target="_blank" rel="noreferrer" style={{ marginTop: 8 }}>Open @{result.botUsername} →</a>}
              </>}
              {result.type === "widget" && <>
                <label className="fld">Embed snippet — paste into any web page</label>
                <pre className="md-pre">{embedSnippet(result.id)}</pre>
                <div className="row" style={{ gap: 8, marginTop: 8 }}><button className="btn sm" onClick={() => copy(embedSnippet(result.id))}>Copy snippet</button><Link className="btn ghost sm" href={`/embed/${result.id}`} target="_blank">Preview</Link></div>
              </>}
              {result.type === "api" && <>
                <label className="fld">Your API key — copy it now, it won’t be shown again</label>
                <pre className="md-pre" style={{ userSelect: "all" }}>{result.apiKey}</pre>
                <button className="btn sm" onClick={() => copy(result.apiKey || "")}>Copy key</button>
                <label className="fld" style={{ marginTop: 14 }}>Example request</label>
                <pre className="md-pre">{curl(result.id)}</pre>
              </>}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
