import { NextResponse } from "next/server";
import mysql from "mysql2/promise";
import { getSessionUser } from "@/lib/auth";
import { resolvesToPrivate, rateLimit } from "@/lib/net";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Connect to a user-supplied MySQL/TiDB URL, run a read-only-ish query, return CSV.
export async function POST(req: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  if (!rateLimit(`dbq:${user.id}`, 10, 60_000)) return NextResponse.json({ error: "Too many queries — wait a minute." }, { status: 429 });
  const { url, query } = await req.json().catch(() => ({}));
  if (!/^mysql:\/\//i.test(url || "")) return NextResponse.json({ error: "Provide a mysql:// connection URL (Postgres support coming later)." }, { status: 400 });
  if (!/^\s*select\b/i.test(query || "")) return NextResponse.json({ error: "Only SELECT queries are allowed here." }, { status: 400 });
  if (/\b(into\s+outfile|load_file|information_schema|mysql\.|;\s*\S)/i.test(query)) return NextResponse.json({ error: "Query rejected (only a single, plain SELECT is allowed)." }, { status: 400 });
  let conn: mysql.Connection | null = null;
  try {
    const u = new URL(url);
    if (await resolvesToPrivate(u.hostname)) return NextResponse.json({ error: "That host is blocked (internal / private address)." }, { status: 400 });
    conn = await mysql.createConnection({
      host: u.hostname, port: Number(u.port || 3306),
      user: decodeURIComponent(u.username), password: decodeURIComponent(u.password),
      database: u.pathname.replace(/^\//, "") || undefined,
      ssl: u.hostname.includes("tidbcloud.com") ? { minVersion: "TLSv1.2", rejectUnauthorized: true } : undefined,
      connectTimeout: 12000,
    });
    const capped = /limit\s+\d+/i.test(query) ? query : query.replace(/;?\s*$/, " LIMIT 2000;");
    const [rows] = await conn.query(capped) as [Record<string, unknown>[], unknown];
    if (!Array.isArray(rows) || rows.length === 0) return NextResponse.json({ error: "Query returned no rows." }, { status: 400 });
    const cols = Object.keys(rows[0]);
    const esc = (v: unknown) => { const s = v == null ? "" : String(v); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
    const csv = [cols.join(","), ...rows.map((r) => cols.map((c) => esc(r[c])).join(","))].join("\n");
    return NextResponse.json({ csv, rows: rows.length });
  } catch (e) {
    return NextResponse.json({ error: `Database error: ${(e as Error).message}` }, { status: 502 });
  } finally {
    if (conn) await conn.end().catch(() => {});
  }
}
