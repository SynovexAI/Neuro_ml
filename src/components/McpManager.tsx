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

type Entry = {
  id: string; title: string; icon: string; iconRgb: string; desc: string;
  transport: "http" | "stdio"; url?: string; command?: string;
  authType: "none" | "bearer"; envName?: string; headerName?: string;
  hosted?: boolean; native?: boolean; needs: string;   // short badge label
  keyLabel?: string; keyPlaceholder?: string; note?: string; // present ⇒ a secret is required
};

// MCP connections. Two kinds work on the free tier:
//   • "Connect your database" (native) — the agent queries it through the in-app DB
//     tool; no runtime needed, runs on Render free.
//   • GitHub (hosted) — GitHub hosts the MCP endpoint; you only supply a free token.
// Web search, Wikipedia, arXiv and memory are already built into the Agent Lab as
// native tools, so they're no longer listed here as servers-to-connect.
const CATALOG: Entry[] = [
  { id: "database", title: "Connect your database", icon: "🐘", iconRgb: "62,207,127", desc: "Chat with your Postgres — the agent queries it live", transport: "stdio", command: "uvx postgres-mcp", envName: "DATABASE_URI", authType: "none", native: true, needs: "works free", keyLabel: "Your database URL", keyPlaceholder: "postgresql://user:pass@host:5432/dbname", note: "Free Postgres: Supabase, Neon, or Render. Reachable from the internet (not localhost). Pick read-only or read-write below." },
  { id: "github", title: "GitHub", icon: "🐙", iconRgb: "160,160,170", desc: "Repos, issues, PRs, code search", transport: "http", url: "https://api.githubcopilot.com/mcp", authType: "bearer", headerName: "Authorization", hosted: true, needs: "free token", keyLabel: "Your GitHub personal access token", keyPlaceholder: "ghp_…", note: "Create one at github.com → Settings → Developer settings → Personal access tokens. No deploy — GitHub hosts it." },
  { id: "deepwiki", title: "DeepWiki", icon: "📘", iconRgb: "91,124,255", desc: "Ask questions about any public GitHub repo's docs", transport: "http", url: "https://mcp.deepwiki.com/mcp", authType: "none", hosted: true, needs: "no key", note: "Fully free — no account, no token. Great for exploring open-source projects." },
  { id: "context7", title: "Context7", icon: "📗", iconRgb: "62,207,127", desc: "Up-to-date docs for thousands of libraries & frameworks", transport: "http", url: "https://mcp.context7.com/mcp", authType: "none", hosted: true, needs: "no key", note: "Free & keyless. Pulls current API docs so the agent doesn't rely on stale training data." },
  { id: "huggingface", title: "Hugging Face", icon: "🤗", iconRgb: "245,158,11", desc: "Search models, datasets & spaces on the Hub", transport: "http", url: "https://huggingface.co/mcp", authType: "bearer", headerName: "Authorization", hosted: true, needs: "free token", keyLabel: "Your Hugging Face access token", keyPlaceholder: "hf_…", note: "Create a free token at huggingface.co → Settings → Access Tokens. Browsing the Hub is free." },
  { id: "semgrep", title: "Semgrep", icon: "🛡", iconRgb: "168,85,247", desc: "Static analysis & security scanning for code", transport: "http", url: "https://mcp.semgrep.ai/mcp", authType: "none", hosted: true, needs: "no key", note: "Free tier — scan code snippets for bugs and security issues." },
  { id: "exa", title: "Exa Search", icon: "🔍", iconRgb: "245,158,11", desc: "Neural web search — finds by meaning, returns page content", transport: "http", url: "https://mcp.exa.ai/mcp", authType: "none", hosted: true, needs: "no key", note: "Works free (rate-limited). For higher limits, add your own Exa key via + Custom server with an x-api-key header." },
  { id: "microsoft-learn", title: "Microsoft Learn", icon: "📘", iconRgb: "91,124,255", desc: "Official Azure / .NET / C# / Windows documentation", transport: "http", url: "https://learn.microsoft.com/api/mcp", authType: "none", hosted: true, needs: "no key", note: "Free & keyless — accurate Microsoft docs & code samples so the agent doesn't guess." },
  { id: "aws-knowledge", title: "AWS Knowledge", icon: "☁", iconRgb: "245,158,11", desc: "AWS service docs, guidance & best practices", transport: "http", url: "https://knowledge-mcp.global.api.aws", authType: "none", hosted: true, needs: "no key", note: "Free & keyless (no AWS account needed) — grounds answers in trusted AWS documentation." },
  { id: "grep", title: "Grep", icon: "🔎", iconRgb: "62,207,127", desc: "Search code across a million public GitHub repos", transport: "http", url: "https://mcp.grep.app", authType: "none", hosted: true, needs: "no key", note: "Free (by Vercel) — find real-world usage examples of any API or pattern." },
  { id: "globalping", title: "Globalping", icon: "🌍", iconRgb: "34,184,207", desc: "Run ping / traceroute / DNS / HTTP from probes worldwide", transport: "http", url: "https://mcp.globalping.dev/mcp", authType: "none", hosted: true, needs: "no key", note: "Free anonymous access — debug latency, DNS and uptime from many regions." },
];

export default function McpManager() {
  const [servers, setServers] = useState<Server[]>([]);
  const [loading, setLoading] = useState(true);
  const [f, setF] = useState<Form>({ ...empty });
  const [active, setActive] = useState<Entry | null>(null); // set ⇒ minimal preset modal; null + open ⇒ custom
  const [busy, setBusy] = useState(false);
  const [connectingId, setConnectingId] = useState("");
  const [msg, setMsg] = useState("");
  const [connectOpen, setConnectOpen] = useState(false);
  const [testing, setTesting] = useState(false);
  const [testRes, setTestRes] = useState<{ ok: boolean; msg: string } | null>(null);
  const [dbWrite, setDbWrite] = useState(false); // false = read-only (safe default)
  const dbCommand = (write: boolean) => `uvx postgres-mcp --access-mode=${write ? "unrestricted" : "restricted"}`;

  async function testDb() {
    setTesting(true); setTestRes(null);
    try {
      const r = await fetch("/api/admin/mcp/test", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ secret: f.secret }) });
      const j = await r.json();
      setTestRes(j.ok ? { ok: true, msg: `connected · ${j.latencyMs}ms${j.version ? " · " + j.version : ""}` } : { ok: false, msg: j.error || "connection failed" });
    } catch (e) { setTestRes({ ok: false, msg: (e as Error).message }); }
    setTesting(false);
  }

  async function load() {
    setLoading(true);
    try { const j = await fetch("/api/admin/mcp").then((r) => r.json()); setServers(j.servers || []); }
    catch { setServers([]); }
    setLoading(false);
  }
  useEffect(() => { load(); }, []);

  const connectedNames = new Set(servers.map((s) => s.name));

  function baseFrom(e: Entry): Form {
    return { ...empty, name: e.id, transport: e.transport, url: e.url || "", command: e.command || "", authType: e.authType, headerName: e.headerName || "Authorization", envName: e.envName || "", secret: "" };
  }
  async function post(form: Form): Promise<boolean> {
    const r = await fetch("/api/admin/mcp", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(form) });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) { setMsg(j.error || "Could not connect the server."); toast(j.error || "Could not connect the server", "error"); return false; }
    return true;
  }
  // Card click: no-key servers connect immediately; keyed / DB servers open a minimal modal.
  async function pick(e: Entry) {
    setMsg("");
    if (e.keyLabel) { setActive(e); setF(baseFrom(e)); setTestRes(null); setDbWrite(false); setConnectOpen(true); return; }
    setConnectingId(e.id);
    if (await post(baseFrom(e))) { toast(`Connected “${e.title}”`, "success"); await load(); }
    setConnectingId("");
  }
  function openCustom() { setActive(null); setF({ ...empty }); setMsg(""); setConnectOpen(true); }

  async function add() {
    setBusy(true); setMsg("");
    const form = active?.id === "database" ? { ...f, command: dbCommand(dbWrite) } : f;
    if (await post(form)) { toast(`Connected “${form.name}”`, "success"); setF({ ...empty }); setConnectOpen(false); setActive(null); await load(); }
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
  const pnl: React.CSSProperties = { border: "1px solid var(--border)", borderRadius: 13, background: "var(--panel)", overflow: "hidden" };

  return (
    <>
      <div style={{ border: "1px solid var(--border)", borderRadius: 13, background: "linear-gradient(180deg, rgba(62,207,127,.06), transparent)", padding: "12px 14px", marginBottom: 14, display: "flex", gap: 11, alignItems: "flex-start" }}>
        <span style={{ fontSize: 17, flex: "0 0 auto", marginTop: 1 }}>🧰</span>
        <div>
          <b style={{ fontSize: 13 }}>Built-in tools are already on — no setup</b>
          <p className="note" style={{ margin: "4px 0 0", fontSize: 12, color: "var(--muted)", fontFamily: "var(--sans)", lineHeight: 1.5 }}>
            Web search, Wikipedia, arXiv and agent memory run inside the Agent Lab out of the box — no server to connect, works on the free tier. Use this page to plug in things only you can provide: <b>your database</b> and <b>hosted MCP integrations</b> like GitHub.
          </p>
        </div>
      </div>
      <div style={{ display: "flex", alignItems: "flex-end", marginBottom: 14, gap: 12 }}>
        <div><h3 style={{ margin: 0, fontSize: 15 }}>Connect a server</h3><div className="note" style={{ marginTop: 3 }}>Your database runs through the in-app tool (free); GitHub is hosted — just paste a token.</div></div>
        <span style={{ flex: 1 }} />
        <button className="btn" onClick={openCustom}>+ Custom server</button>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(232px, 1fr))", gap: 12, marginBottom: 22 }}>
        {CATALOG.map((e) => {
          const connected = connectedNames.has(e.id);
          const on = connectedNames.has(e.id) && servers.find((s) => s.name === e.id)?.enabled;
          return (
            <div key={e.id} style={{ border: `1px solid ${connected ? "rgba(62,207,127,.4)" : "var(--border)"}`, boxShadow: connected ? "0 0 0 1px rgba(62,207,127,.2)" : "none", borderRadius: 13, background: "var(--panel)", padding: 14, display: "flex", flexDirection: "column" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <span style={{ width: 36, height: 36, borderRadius: 10, display: "grid", placeItems: "center", fontSize: 17, flex: "0 0 auto", background: `rgba(${e.iconRgb},.15)` }}>{e.icon}</span>
                <b style={{ fontSize: 14 }}>{e.title}</b>
              </div>
              <p className="note" style={{ margin: "9px 0 0", fontSize: 11.5, color: "var(--muted)", fontFamily: "var(--sans)" }}>{e.desc}</p>
              <div style={{ marginTop: "auto", paddingTop: 12, display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
                <span className="badge" style={e.native ? { color: "var(--good)", borderColor: "rgba(62,207,127,.35)" } : { color: e.hosted ? "var(--sky)" : "var(--accent)", borderColor: e.hosted ? "var(--border)" : "rgba(91,124,255,.35)" }}>{e.native ? "✓ built-in tool" : e.hosted ? "hosted" : "🐳 stdio"}</span>
                <span className="badge" style={e.needs === "no key" || e.needs === "works free" ? { color: "var(--good)", borderColor: "rgba(62,207,127,.3)" } : {}}>{e.needs}</span>
                {connected
                  ? <span style={{ marginLeft: "auto", display: "inline-flex", alignItems: "center", gap: 5, fontSize: 11.5, fontWeight: 600, color: on ? "var(--good)" : "var(--faint)" }}><span style={{ width: 7, height: 7, borderRadius: "50%", background: on ? "var(--good)" : "var(--faint)" }} />{on ? "connected" : "disabled"}</span>
                  : <button className="btn ghost sm" style={{ marginLeft: "auto" }} disabled={connectingId === e.id} onClick={() => pick(e)}>{connectingId === e.id ? "…" : e.keyLabel ? "Connect →" : "Enable →"}</button>}
              </div>
            </div>
          );
        })}
      </div>

      <div style={pnl}>
        <div className="row" style={{ alignItems: "center", justifyContent: "space-between", padding: "11px 15px", borderBottom: "1px solid var(--border)", background: "var(--surface)" }}>
          <span style={{ fontSize: 10.5, fontWeight: 600, textTransform: "uppercase", letterSpacing: ".06em", color: "var(--muted)" }}>● Your connected servers</span>
          <span className="mono" style={{ fontSize: 11, color: "var(--faint)" }}>{servers.length}</span>
        </div>
        <div style={{ padding: servers.length ? 0 : 15 }}>
          {loading ? <div className="note" style={{ padding: 15 }}>Loading…</div>
            : servers.length === 0 ? <div className="note">No servers yet — connect one above.</div>
            : servers.map((s) => { const cat = CATALOG.find((c) => c.id === s.name);
              return (
                <div key={s.id} className="row" style={{ gap: 11, alignItems: "center", padding: "12px 15px", borderBottom: "1px solid var(--border)" }}>
                  <span style={{ width: 32, height: 32, borderRadius: 9, display: "grid", placeItems: "center", fontSize: 15, flex: "0 0 auto", background: `rgba(${cat?.iconRgb || "150,150,160"},.15)` }}>{cat?.icon || "🔌"}</span>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 7, flexWrap: "wrap" }}><b style={{ fontSize: 13.5 }}>{cat?.title || s.name}</b><span className="badge">{s.transport}</span>{s.hasSecret && <span className="badge" style={{ color: "var(--good)" }}>🔒 key</span>}</div>
                    <div className="note" style={{ marginTop: 3 }}>{cat?.native ? "runs through the in-app database tool — works on the free tier" : Array.isArray(s.tools) && s.tools.length ? <>tools: {s.tools.slice(0, 6).map((t) => <span key={t} className="chip" style={{ marginRight: 4, fontSize: 10.5, padding: "2px 8px" }}>{t}</span>)}{s.tools.length > 6 ? ` +${s.tools.length - 6}` : ""}</> : cat?.hosted ? "hosted MCP — tools resolve when the agent connects" : "tools resolve when an agent runs (discovery via the NAT runtime)"}</div>
                  </div>
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 5, fontSize: 11.5, fontWeight: 600, color: s.enabled ? "var(--good)" : "var(--faint)", flex: "0 0 auto" }}><span style={{ width: 7, height: 7, borderRadius: "50%", background: s.enabled ? "var(--good)" : "var(--faint)" }} />{s.enabled ? "on" : "off"}</span>
                  <span style={{ display: "flex", gap: 6, flex: "0 0 auto" }}>
                    <button className="btn ghost sm" onClick={() => patch(s.id, { enabled: !s.enabled }, s.enabled ? "Server disabled" : "Server enabled")}>{s.enabled ? "Disable" : "Enable"}</button>
                    <button className="btn ghost sm danger" onClick={() => remove(s)}>Remove</button>
                  </span>
                </div>
              );
            })}
        </div>
      </div>

      <div className="note" style={{ marginTop: 12, lineHeight: 1.6 }}>Keys are stored encrypted. <b>Your database and the built-in Agent Lab tools run in-app and work on Render&apos;s free tier.</b> Hosted MCP (like GitHub) and any custom <span className="mono">stdio</span> command execute through the NAT agent runtime — a small always-on service that isn&apos;t on the free tier — so those are configured now and become executable once the NAT runtime (a paid instance) is online.</div>

      {connectOpen && (
        <div className="modal-wrap show" onClick={(e) => { if (e.target === e.currentTarget) { setConnectOpen(false); setActive(null); } }}>
          <div className="modal" style={{ maxWidth: active ? 460 : 560 }}>
            {active ? (
              <>
                <div className="mh" style={{ display: "flex", alignItems: "center", gap: 12 }}><span style={{ width: 34, height: 34, borderRadius: 9, display: "grid", placeItems: "center", fontSize: 16, background: `rgba(${active.iconRgb},.15)` }}>{active.icon}</span><div style={{ flex: 1 }}><b>Connect {active.title}</b><div className="note" style={{ marginTop: 2 }}>{active.desc}{active.native ? " · in-app tool, works free" : active.hosted ? " · hosted, no deploy" : " · runs in the NAT runtime"}</div></div><button className="x" onClick={() => { setConnectOpen(false); setActive(null); }}>×</button></div>
                <div className="mb">
                  {!active.native && <div className="row" style={{ gap: 6, flexWrap: "wrap", marginBottom: 14 }}>{[["transport", active.transport], active.url ? ["url", active.url] : active.command ? ["run", active.command] : null, active.envName ? ["env", active.envName] : null].filter(Boolean).map((x) => { const [k, v] = x as [string, string]; return <span key={k} className="mono" style={{ fontSize: 10.5, background: "var(--panel-2)", border: "1px solid var(--border)", borderRadius: 6, padding: "3px 9px", color: "var(--faint)", maxWidth: "100%", overflow: "hidden", textOverflow: "ellipsis" }}>{k}: {v}</span>; })}</div>}
                  <label className="fld">{active.keyLabel} <span className="note" style={{ textTransform: "none" }}>· stored encrypted, never shown again</span></label>
                  <input type="password" placeholder={active.keyPlaceholder} value={f.secret} onChange={(e) => { setF({ ...f, secret: e.target.value }); if (testRes) setTestRes(null); }} />
                  {active.id === "database" && <>
                    <label className="fld" style={{ marginTop: 14 }}>Access mode</label>
                    <div className="chips">
                      <button className={`chip ${!dbWrite ? "on" : ""}`} onClick={() => setDbWrite(false)}>🔒 Read-only</button>
                      <button className={`chip ${dbWrite ? "on" : ""}`} onClick={() => setDbWrite(true)}>✎ Read-write</button>
                    </div>
                    <div className="note" style={{ marginTop: 8, lineHeight: 1.5, color: dbWrite ? "var(--warn)" : "var(--faint)" }}>{dbWrite ? "⚠ The agent can INSERT / UPDATE / DELETE and change your data — use only on a database you're OK with an agent modifying." : "The agent can only SELECT / read — safest for chatting with your data."}</div>
                  </>}
                  {active.note && <div className="note" style={{ marginTop: 10, lineHeight: 1.5 }}>{active.note}</div>}
                  {active.id === "database" && testRes && <div className="note" style={{ marginTop: 10, color: testRes.ok ? "var(--good)" : "var(--crit)" }}>{testRes.ok ? "✓ " : "✗ "}{testRes.msg}</div>}
                  <div className="row" style={{ marginTop: 16, gap: 10 }}>
                    <button className="btn" onClick={add} disabled={busy || !f.secret.trim()}>{busy ? "Connecting…" : "Connect"}</button>
                    {active.id === "database" && <button className="btn ghost" onClick={testDb} disabled={testing || !f.secret.trim()}>{testing ? "Testing…" : "Test connection"}</button>}
                    <button className="btn ghost" onClick={() => { setConnectOpen(false); setActive(null); }}>Cancel</button>
                    {msg && <span className="err" style={{ margin: 0 }}>{msg}</span>}
                  </div>
                </div>
              </>
            ) : (
              <>
                <div className="mh"><b>Custom MCP server</b><button className="x" onClick={() => setConnectOpen(false)}>×</button></div>
                <div className="mb">
                  <div className="split col-2e">
                    <div><label className="fld">Name</label><input type="text" placeholder="my-server" value={f.name} onChange={(e) => setF({ ...f, name: e.target.value })} /></div>
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
                            <option value="none">None</option><option value="bearer">API key (Bearer)</option><option value="apikey">API key (custom header)</option>
                          </select>
                        </div>
                        {f.authType === "apikey" && <div><label className="fld">Header name</label><input type="text" placeholder="X-API-Key" value={f.headerName} onChange={(e) => setF({ ...f, headerName: e.target.value })} /></div>}
                      </div>
                      {(f.authType === "apikey" || f.authType === "bearer") && (<><label className="fld" style={{ marginTop: 12 }}>Your key / token (stored encrypted)</label><input type="password" placeholder="paste your API key / token" value={f.secret} onChange={(e) => setF({ ...f, secret: e.target.value })} /></>)}
                    </>
                  ) : (
                    <>
                      <label className="fld" style={{ marginTop: 12 }}>Command</label>
                      <input type="text" placeholder="npx -y @modelcontextprotocol/server-filesystem /data" value={f.command} onChange={(e) => setF({ ...f, command: e.target.value })} />
                      <div className="split col-2e" style={{ marginTop: 12 }}>
                        <div><label className="fld">Env var name (optional)</label><input type="text" placeholder="API_TOKEN" value={f.envName} onChange={(e) => setF({ ...f, envName: e.target.value })} /></div>
                        <div><label className="fld">Your key (encrypted)</label><input type="password" placeholder="value for that env var" value={f.secret} onChange={(e) => setF({ ...f, secret: e.target.value })} /></div>
                      </div>
                      <div className="note" style={{ marginTop: 8 }}>⚠ the command must be installed in the NAT runtime image.</div>
                    </>
                  )}
                  <div className="row" style={{ marginTop: 16, gap: 10 }}>
                    <button className="btn" onClick={add} disabled={busy}>{busy ? "Connecting…" : "Connect"}</button>
                    <button className="btn ghost" onClick={() => setConnectOpen(false)}>Cancel</button>
                    {msg && <span className="err" style={{ margin: 0 }}>{msg}</span>}
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </>
  );
}
