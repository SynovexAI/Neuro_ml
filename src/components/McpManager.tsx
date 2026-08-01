"use client";

import { useEffect, useState } from "react";
import { toast, confirmDialog } from "@/lib/toast";

type Server = {
  id: string; name: string; transport: "http" | "sse" | "stdio";
  url: string | null; command: string | null;
  authType: "none" | "apikey" | "bearer" | "oauth";
  headerName: string | null; envName: string | null;
  enabled: boolean; tools: string[] | null; hasSecret: boolean;
};

type Form = {
  name: string; transport: "http" | "sse" | "stdio"; url: string; command: string;
  authType: "none" | "apikey" | "bearer" | "oauth"; headerName: string; envName: string; secret: string;
};
const empty: Form = { name: "", transport: "http", url: "", command: "", authType: "none", headerName: "Authorization", envName: "", secret: "" };

// Popular MCP servers. Clicking one pre-fills the connect form; the admin just
// adds the key / token. (stdio ones must be installed in the deployed image.)
const CATALOG: (Partial<Form> & { title: string; icon: string; desc: string; needs: string })[] = [
  { title: "GitHub", icon: "🐙", desc: "Repos, issues, PRs", needs: "personal access token", name: "github", transport: "http", url: "https://api.githubcopilot.com/mcp", authType: "bearer", headerName: "Authorization" },
  { title: "Brave Search", icon: "🦁", desc: "Web search", needs: "Brave API key", name: "brave-search", transport: "stdio", command: "npx -y @modelcontextprotocol/server-brave-search", envName: "BRAVE_API_KEY" },
  { title: "Slack", icon: "💬", desc: "Channels & messages", needs: "bot token", name: "slack", transport: "stdio", command: "npx -y @modelcontextprotocol/server-slack", envName: "SLACK_BOT_TOKEN" },
  { title: "Google Maps", icon: "🗺", desc: "Places & directions", needs: "Maps API key", name: "google-maps", transport: "stdio", command: "npx -y @modelcontextprotocol/server-google-maps", envName: "GOOGLE_MAPS_API_KEY" },
  { title: "Postgres", icon: "🐘", desc: "Query a database", needs: "connection string", name: "postgres", transport: "stdio", command: "npx -y @modelcontextprotocol/server-postgres", envName: "DATABASE_URL" },
  { title: "Filesystem", icon: "📁", desc: "Local files", needs: "no auth", name: "filesystem", transport: "stdio", command: "npx -y @modelcontextprotocol/server-filesystem /data", authType: "none" },
  { title: "Fetch", icon: "🌐", desc: "Fetch & read URLs", needs: "no auth", name: "fetch", transport: "stdio", command: "npx -y @modelcontextprotocol/server-fetch", authType: "none" },
  { title: "Time", icon: "⏰", desc: "Dates & timezones", needs: "no auth", name: "time", transport: "stdio", command: "npx -y @modelcontextprotocol/server-time", authType: "none" },
];

export default function McpManager() {
  const [servers, setServers] = useState<Server[]>([]);
  const [loading, setLoading] = useState(true);
  const [f, setF] = useState<Form>({ ...empty });
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");
  const [connectOpen, setConnectOpen] = useState(false);

  async function load() {
    setLoading(true);
    try { const j = await fetch("/api/admin/mcp").then((r) => r.json()); setServers(j.servers || []); }
    catch { setServers([]); }
    setLoading(false);
  }
  useEffect(() => { load(); }, []);

  function applyPreset(c: Partial<Form>) { setF({ ...empty, ...c }); setMsg(""); setConnectOpen(true); }
  function openCustom() { setF({ ...empty }); setMsg(""); setConnectOpen(true); }

  async function add() {
    setBusy(true); setMsg("");
    try {
      const r = await fetch("/api/admin/mcp", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(f) });
      const j = await r.json().catch(() => ({}));
      if (!r.ok) { setMsg(j.error || "Could not add the server."); toast(j.error || "Could not connect the server", "error"); }
      else { toast(`Connected “${f.name}”`, "success"); setF({ ...empty }); setConnectOpen(false); await load(); }
    } catch (e) { setMsg((e as Error).message); toast((e as Error).message, "error"); }
    setBusy(false);
  }
  async function patch(id: string, body: object, note?: string) {
    await fetch(`/api/admin/mcp/${id}`, { method: "PATCH", headers: { "content-type": "application/json" }, body: JSON.stringify(body) }).catch(() => {});
    load();
    if (note) toast(note, "success");
  }
  async function remove(s: Server) {
    if (!(await confirmDialog(`Remove MCP server “${s.name}”?`, { confirmLabel: "Remove", danger: true }))) return;
    await fetch(`/api/admin/mcp/${s.id}`, { method: "DELETE" }).catch(() => {});
    load();
    toast("MCP server removed", "success");
  }

  const remoteForm = f.transport === "http" || f.transport === "sse";

  return (
    <>
      <div style={{ display: "flex", alignItems: "flex-end", marginBottom: 12, gap: 12 }}>
        <div><h3 style={{ margin: 0, fontSize: 15 }}>Marketplace</h3><div className="note" style={{ marginTop: 2 }}>Pick a server to connect — you add your own key.</div></div>
        <span style={{ flex: 1 }} />
        <button className="btn" onClick={openCustom}>+ Custom server</button>
      </div>

      <div className="cards" style={{ marginBottom: 18 }}>
        {CATALOG.map((c) => (
          <button key={c.title} type="button" className="lab-card" onClick={() => applyPreset(c)} style={{ cursor: "pointer", textAlign: "left", fontFamily: "inherit", alignItems: "stretch" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}><span className="zi" style={{ fontSize: 18 }}>{c.icon}</span><b style={{ fontSize: 14 }}>{c.title}</b></div>
            <p style={{ margin: "6px 0 0" }}>{c.desc}</p>
            <div style={{ marginTop: "auto", paddingTop: 10, display: "flex", alignItems: "center", gap: 6 }}>
              <span className="badge">{c.transport === "http" ? "HTTP" : c.transport}</span>
              <span className="note">needs {c.needs}</span>
              <span className="go" style={{ marginLeft: "auto" }}>Connect →</span>
            </div>
          </button>
        ))}
      </div>

      <div className="card">
        <div className="card-h"><span className="t">Your connected servers</span><span className="mono r">{servers.length}</span></div>
        <div className="card-b">
          {loading ? <div className="note">Loading…</div>
            : servers.length === 0 ? <div className="note">No MCP servers yet. Pick one from the Marketplace above.</div>
            : servers.map((s) => (
              <div key={s.id} style={{ borderBottom: "1px solid var(--border)", padding: "10px 0" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
                  <span style={{ width: 8, height: 8, borderRadius: "50%", background: s.enabled ? "var(--accent)" : "var(--faint)", flex: "0 0 auto" }} />
                  <b style={{ fontSize: 13.5 }}>{s.name}</b>
                  <span className="badge">{s.transport}</span>
                  {s.hasSecret && <span className="badge" title="secret stored encrypted">🔒 key</span>}
                  <span className="note">{s.url || s.command}</span>
                  <span style={{ marginLeft: "auto", display: "flex", gap: 6 }}>
                    <button className="btn ghost sm" onClick={() => patch(s.id, { enabled: !s.enabled }, s.enabled ? "Server disabled" : "Server enabled")}>{s.enabled ? "Disable" : "Enable"}</button>
                    <button className="btn ghost sm danger" onClick={() => remove(s)}>Remove</button>
                  </span>
                </div>
                <div className="note" style={{ marginTop: 6 }}>
                  {Array.isArray(s.tools) && s.tools.length ? <>tools: {s.tools.map((t) => <span key={t} className="chip" style={{ marginRight: 4, fontSize: 11 }}>{t}</span>)}</> : "tools resolve when an agent runs (discovery via the NAT service)"}
                </div>
              </div>
            ))}
        </div>
      </div>

      <div className="note" style={{ marginTop: 12 }}>Tools from <b>enabled</b> servers appear in the Agent Lab palette, tagged MCP. Execution runs through the NAT service.</div>

      {connectOpen && (
        <div className="modal-wrap show" onClick={(e) => { if (e.target === e.currentTarget) setConnectOpen(false); }}>
          <div className="modal" style={{ maxWidth: 560 }}>
            <div className="mh"><b>Connect MCP server</b><button className="x" onClick={() => setConnectOpen(false)}>×</button></div>
            <div className="mb">
              <div className="split col-2e">
                <div><label className="fld">Name</label><input type="text" placeholder="github" value={f.name} onChange={(e) => setF({ ...f, name: e.target.value })} /></div>
                <div><label className="fld">Transport</label>
                  <select value={f.transport} onChange={(e) => setF({ ...f, transport: e.target.value as typeof f.transport })}>
                    <option value="http">Streamable HTTP</option><option value="sse">SSE</option><option value="stdio">stdio (local command)</option>
                  </select>
                </div>
              </div>
              {remoteForm ? (
                <>
                  <label className="fld" style={{ marginTop: 12 }}>Server URL</label>
                  <input type="text" placeholder="https://mcp.example.com/mcp" value={f.url} onChange={(e) => setF({ ...f, url: e.target.value })} />
                  <div className="split col-2e" style={{ marginTop: 12 }}>
                    <div><label className="fld">Auth</label>
                      <select value={f.authType} onChange={(e) => setF({ ...f, authType: e.target.value as typeof f.authType })}>
                        <option value="none">None</option><option value="bearer">API key (Bearer)</option><option value="apikey">API key (custom header)</option><option value="oauth">OAuth 2.1 (soon)</option>
                      </select>
                    </div>
                    {f.authType === "apikey" && <div><label className="fld">Header name</label><input type="text" placeholder="X-API-Key" value={f.headerName} onChange={(e) => setF({ ...f, headerName: e.target.value })} /></div>}
                  </div>
                  {(f.authType === "apikey" || f.authType === "bearer") && (<><label className="fld" style={{ marginTop: 12 }}>Your key / token (stored encrypted)</label><input type="password" placeholder="paste your API key / token" value={f.secret} onChange={(e) => setF({ ...f, secret: e.target.value })} /></>)}
                  {f.authType === "oauth" && <div className="note" style={{ marginTop: 10 }}>OAuth sign-in is a later phase — use API key / Bearer for now.</div>}
                </>
              ) : (
                <>
                  <label className="fld" style={{ marginTop: 12 }}>Command</label>
                  <input type="text" placeholder="npx -y @modelcontextprotocol/server-filesystem /data" value={f.command} onChange={(e) => setF({ ...f, command: e.target.value })} />
                  <div className="split col-2e" style={{ marginTop: 12 }}>
                    <div><label className="fld">Env var name (optional)</label><input type="text" placeholder="API_TOKEN" value={f.envName} onChange={(e) => setF({ ...f, envName: e.target.value })} /></div>
                    <div><label className="fld">Your key (encrypted)</label><input type="password" placeholder="value for that env var" value={f.secret} onChange={(e) => setF({ ...f, secret: e.target.value })} /></div>
                  </div>
                  <div className="note" style={{ marginTop: 8 }}>⚠ the command must be installed in the deployed image (stdio can&apos;t launch arbitrary binaries on Render).</div>
                </>
              )}
              <div className="row" style={{ marginTop: 16, gap: 10 }}>
                <button className="btn" onClick={add} disabled={busy}>{busy ? "Connecting…" : "Connect"}</button>
                <button className="btn ghost" onClick={() => setConnectOpen(false)}>Cancel</button>
                {msg && <span className="err" style={{ margin: 0 }}>{msg}</span>}
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
