import { NextResponse } from "next/server";
import { and, eq } from "drizzle-orm";
import { db } from "@/lib/db";
import { mcpServers } from "@/lib/db/schema";
import { getSessionUser } from "@/lib/auth";
import { decrypt } from "@/lib/crypto";
import { resolvesToPrivate } from "@/lib/net";
import { rateLimitDb } from "@/lib/ratelimit";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Server-side proxy for HOSTED (Streamable-HTTP) MCP servers — e.g. GitHub's
// api.githubcopilot.com/mcp. The remote server runs itself; we just relay the
// JSON-RPC handshake with the user's decrypted token. This makes hosted MCP
// tools usable by the in-browser agent WITHOUT the NAT runtime — free-tier OK.

const PROTOCOL = "2025-06-18";

type JsonRpc = { jsonrpc: "2.0"; id?: number; method: string; params?: unknown };
type RpcResult = { result?: unknown; error?: { code: number; message: string } };

// Extract the JSON-RPC message for `id` from a text/event-stream body.
function parseSse(text: string, id: number): RpcResult | null {
  for (const block of text.split(/\n\n/)) {
    const data = block.split(/\n/).filter((l) => l.startsWith("data:")).map((l) => l.slice(5).trim()).join("");
    if (!data) continue;
    try { const j = JSON.parse(data); if (j && (j.id === id || j.result || j.error)) return j; } catch { /* skip */ }
  }
  return null;
}

async function rpc(url: string, token: string | null, headerName: string, sessionId: string, msg: JsonRpc, signal: AbortSignal): Promise<{ payload: RpcResult | null; sessionId: string; status: number }> {
  const headers: Record<string, string> = {
    "content-type": "application/json",
    accept: "application/json, text/event-stream",
    "mcp-protocol-version": PROTOCOL,
  };
  if (token) headers[headerName || "Authorization"] = /authorization/i.test(headerName || "Authorization") ? `Bearer ${token}` : token;
  if (sessionId) headers["mcp-session-id"] = sessionId;
  const r = await fetch(url, { method: "POST", headers, body: JSON.stringify(msg), signal });
  const newSession = r.headers.get("mcp-session-id") || sessionId;
  const ct = r.headers.get("content-type") || "";
  const text = await r.text();
  if (msg.id === undefined) return { payload: null, sessionId: newSession, status: r.status }; // notification
  const payload = ct.includes("text/event-stream") ? parseSse(text, msg.id) : text ? JSON.parse(text) : null;
  return { payload, sessionId: newSession, status: r.status };
}

// Look up a connected hosted (http/sse) MCP server for this user.
async function getServer(userId: string, name: string) {
  const rows = await db.select().from(mcpServers).where(and(eq(mcpServers.userId, userId), eq(mcpServers.name, name), eq(mcpServers.enabled, true))).limit(1);
  const r = rows[0];
  if (!r) return null;
  if (r.transport === "stdio") return { stdio: true } as const;
  return { url: r.url || "", token: r.secretEnc ? decrypt(r.secretEnc) : null, headerName: r.headerName || "Authorization" };
}

function summarizeTools(result: unknown): string {
  const tools = (result as { tools?: { name: string; description?: string }[] })?.tools || [];
  if (!tools.length) return "This MCP server exposes no tools.";
  return tools.slice(0, 40).map((t) => `- ${t.name}${t.description ? `: ${t.description.replace(/\s+/g, " ").slice(0, 90)}` : ""}`).join("\n") + (tools.length > 40 ? `\n… +${tools.length - 40} more` : "");
}

function summarizeCall(result: unknown): string {
  const content = (result as { content?: { type: string; text?: string }[]; isError?: boolean })?.content || [];
  const txt = content.filter((c) => c.type === "text" && c.text).map((c) => c.text).join("\n").trim();
  return (txt || JSON.stringify(result)).slice(0, 1800);
}

export async function POST(req: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  if (!(await rateLimitDb("mcp", user.id, 30, 60_000))) return NextResponse.json({ error: "Too many requests — wait a minute." }, { status: 429 });

  const b = await req.json().catch(() => ({}));
  const server = String(b.server || "github").trim();
  const action = String(b.action || "list").trim();      // "servers" | "list" | "call"
  const toolName = String(b.tool || "").trim();
  const argsRaw = b.args;

  // List the hosted MCP servers this user has connected (stdio needs the NAT runtime).
  if (action === "servers") {
    const rows = await db.select().from(mcpServers).where(and(eq(mcpServers.userId, user.id), eq(mcpServers.enabled, true)));
    const hosted = rows.filter((r) => r.transport !== "stdio").map((r) => r.name);
    if (!hosted.length) return NextResponse.json({ text: "No hosted MCP servers connected. Add one (GitHub, DeepWiki, Context7, Hugging Face, Semgrep, …) on the MCP servers page." });
    return NextResponse.json({ text: `Connected hosted MCP servers: ${hosted.join(", ")}. Use '<server> list' to see a server's tools.` });
  }

  const srv = await getServer(user.id, server);
  if (!srv) return NextResponse.json({ text: `No MCP server named "${server}" is connected. Connect it on the MCP servers page.` });
  if ("stdio" in srv) return NextResponse.json({ text: `"${server}" is a stdio server — it needs the NAT agent runtime to execute and isn't reachable from the free tier. Use a hosted MCP server (like GitHub) instead.` });
  if (!srv.url || !/^https?:\/\//i.test(srv.url)) return NextResponse.json({ text: "That server has no valid HTTP endpoint." });

  try {
    const host = new URL(srv.url).hostname;
    if (await resolvesToPrivate(host)) return NextResponse.json({ text: "Blocked: the MCP endpoint resolves to a private/internal address." });
  } catch { return NextResponse.json({ text: "That server URL is invalid." }); }

  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), 20_000);
  try {
    // 1) initialize
    const init = await rpc(srv.url, srv.token, srv.headerName, "", { jsonrpc: "2.0", id: 1, method: "initialize", params: { protocolVersion: PROTOCOL, capabilities: {}, clientInfo: { name: "ai-workbench", version: "1.0" } } }, ctrl.signal);
    if (init.status === 401 || init.status === 403) return NextResponse.json({ text: "Authentication failed — the token for this MCP server is missing, invalid, or expired. Reconnect it on the MCP servers page." });
    if (init.payload?.error) return NextResponse.json({ text: `MCP init error: ${init.payload.error.message}` });
    const sid = init.sessionId;
    // 2) initialized notification (best-effort)
    await rpc(srv.url, srv.token, srv.headerName, sid, { jsonrpc: "2.0", method: "notifications/initialized" }, ctrl.signal).catch(() => {});

    // 3) list or call
    if (action === "call") {
      if (!toolName) return NextResponse.json({ text: 'Provide a tool name, e.g. {"tool":"search_repositories","args":{"query":"nextjs"}}. Use action "list" first to see available tools.' });
      let args: Record<string, unknown> = {};
      if (typeof argsRaw === "string" && argsRaw.trim()) { try { args = JSON.parse(argsRaw); } catch { return NextResponse.json({ text: "The args field must be a JSON object." }); } }
      else if (argsRaw && typeof argsRaw === "object") args = argsRaw as Record<string, unknown>;
      const call = await rpc(srv.url, srv.token, srv.headerName, sid, { jsonrpc: "2.0", id: 3, method: "tools/call", params: { name: toolName, arguments: args } }, ctrl.signal);
      if (call.payload?.error) return NextResponse.json({ text: `Tool "${toolName}" error: ${call.payload.error.message}` });
      return NextResponse.json({ text: summarizeCall(call.payload?.result) });
    }

    const list = await rpc(srv.url, srv.token, srv.headerName, sid, { jsonrpc: "2.0", id: 2, method: "tools/list" }, ctrl.signal);
    if (list.payload?.error) return NextResponse.json({ text: `MCP tools/list error: ${list.payload.error.message}` });
    return NextResponse.json({ text: `Tools on "${server}":\n${summarizeTools(list.payload?.result)}` });
  } catch (e) {
    const m = (e as Error).name === "AbortError" ? "the MCP server timed out (20s)" : (e as Error).message;
    return NextResponse.json({ text: "MCP error: " + m.slice(0, 200) });
  } finally {
    clearTimeout(t);
  }
}
