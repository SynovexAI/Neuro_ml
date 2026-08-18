import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Verify a user-supplied Postgres connection string before they save it — the
// same connection the agent will use at run time, so this is no extra exposure.
export async function POST(req: Request) {
  const u = await getSessionUser();
  if (!u) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const b = await req.json().catch(() => ({}));
  const conn = String(b.secret || "").trim();
  const isLibsql = /^libsql:|\.turso\.io/i.test(conn);
  if (!isLibsql && !/^postgres(ql)?:\/\//i.test(conn)) return NextResponse.json({ ok: false, error: "Enter a postgresql:// or a libsql:// (Turso) connection string." }, { status: 400 });

  // Turso / libSQL (SQLite-compatible over HTTP)
  if (isLibsql) {
    const t0b = Date.now();
    const m = conn.match(/[?&]authToken=([^&]+)/i);
    const authToken = m ? decodeURIComponent(m[1]) : undefined;
    const url = conn.replace(/([?&])authToken=[^&]+/i, "$1").replace(/[?&]+$/, "");
    const { createClient } = await import("@libsql/client");
    const client = createClient({ url, authToken });
    try { await client.execute("select 1"); return NextResponse.json({ ok: true, latencyMs: Date.now() - t0b, version: "Turso / libSQL" }); }
    catch (e) { return NextResponse.json({ ok: false, error: (e as Error).message.replace(/\s+/g, " ").slice(0, 200) }); }
    finally { client.close(); }
  }

  const { Client } = await import("pg");
  const t0 = Date.now();
  const attempt = async (ssl: false | { rejectUnauthorized: boolean }) => {
    const client = new Client({ connectionString: conn, connectionTimeoutMillis: 8000, query_timeout: 8000, ...(ssl ? { ssl } : {}) });
    await client.connect();
    try { const r = await client.query("select version()"); return String(r.rows?.[0]?.version || "connected"); }
    finally { await client.end().catch(() => {}); }
  };
  try {
    let version: string;
    try { version = await attempt(false); }
    catch (e) { // hosted Postgres (Neon/Supabase/Render) often needs SSL; retry allowing it
      if (/ssl|certificate|self.signed|no encryption/i.test((e as Error).message)) version = await attempt({ rejectUnauthorized: false });
      else throw e;
    }
    return NextResponse.json({ ok: true, latencyMs: Date.now() - t0, version: version.replace(/\s+/g, " ").slice(0, 80) });
  } catch (e) {
    return NextResponse.json({ ok: false, error: (e as Error).message.replace(/\s+/g, " ").slice(0, 200) });
  }
}
