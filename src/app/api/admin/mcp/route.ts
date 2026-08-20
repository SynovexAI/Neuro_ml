import { NextResponse } from "next/server";
import { desc, eq } from "drizzle-orm";
import { db } from "@/lib/db";
import { mcpServers } from "@/lib/db/schema";
import { getSessionUser, uid } from "@/lib/auth";
import { encrypt } from "@/lib/crypto";
import { audit } from "@/lib/monitor";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const TRANSPORTS = ["http", "sse", "stdio"];
const AUTH_TYPES = ["none", "apikey", "bearer", "oauth"];

// MCP servers are per-user: each user connects and uses their own.
export async function GET() {
  const u = await getSessionUser();
  if (!u) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const rows = await db.select().from(mcpServers).where(eq(mcpServers.userId, u.id)).orderBy(desc(mcpServers.createdAt));
  const safe = rows.map(({ secretEnc, ...r }) => ({ ...r, hasSecret: !!secretEnc }));
  return NextResponse.json({ servers: safe });
}

export async function POST(req: Request) {
  const u = await getSessionUser();
  if (!u) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const b = await req.json().catch(() => ({}));
  const name = String(b.name || "").trim();
  const transport = TRANSPORTS.includes(b.transport) ? b.transport : "http";
  const authType = AUTH_TYPES.includes(b.authType) ? b.authType : "none";
  if (!name) return NextResponse.json({ error: "A server name is required." }, { status: 400 });
  // stdio MCP servers need a local process (the self-hosted NAT runtime) and can't run
  // on the serverless deploy. Only the built-in "database" connector may use it (it's
  // served over HTTP by the in-app db tools, not executed as an MCP). Everything else
  // must be a hosted HTTP/SSE server.
  if (transport === "stdio" && name !== "database")
    return NextResponse.json({ error: "stdio MCP servers can't run on this deployment. Connect a hosted HTTP or SSE server instead — or use the built-in Database tool for databases." }, { status: 400 });
  if ((transport === "http" || transport === "sse") && !String(b.url || "").trim())
    return NextResponse.json({ error: "A server URL is required for HTTP/SSE." }, { status: 400 });
  if (transport === "stdio" && !String(b.command || "").trim())
    return NextResponse.json({ error: "A command is required for stdio." }, { status: 400 });

  const secret = b.secret ? String(b.secret) : "";
  const id = uid();
  await db.insert(mcpServers).values({
    id, userId: u.id, name, transport,
    url: transport === "stdio" ? null : String(b.url || "").trim() || null,
    command: transport === "stdio" ? String(b.command || "").trim() : null,
    authType,
    headerName: b.headerName ? String(b.headerName).slice(0, 80) : null,
    envName: b.envName ? String(b.envName).slice(0, 80) : null,
    secretEnc: secret ? encrypt(secret) : null,
    enabled: b.enabled !== false,
  });
  await audit("mcp_server_added", u.id, { name, transport, authType });
  return NextResponse.json({ ok: true, id });
}
