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

// Native (no-MCP, no-NAT) agent tools that run inside the web service — usable by
// the in-browser agent on Render's free tier: web search, Wikipedia, arXiv, and
// a real database tool (queries the DB the user connected on the MCP page).

const strip = (s: string) => s.replace(/<[^>]+>/g, "").replace(/&amp;/g, "&").replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/\s+/g, " ").trim();

async function ext(url: string, opts?: RequestInit): Promise<string> {
  const ctrl = new AbortController(); const t = setTimeout(() => ctrl.abort(), 12000);
  try { const r = await fetch(url, { ...opts, signal: ctrl.signal, headers: { "user-agent": "AI-Workbench-Agent", ...(opts?.headers || {}) } }); return await r.text(); }
  finally { clearTimeout(t); }
}

async function webSearch(q: string): Promise<string> {
  const html = await ext("https://html.duckduckgo.com/html/?q=" + encodeURIComponent(q));
  const titles = [...html.matchAll(/<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/g)];
  const snips = [...html.matchAll(/<a[^>]*class="result__snippet"[^>]*>([\s\S]*?)<\/a>/g)].map((m) => strip(m[1]));
  if (!titles.length) return "No results.";
  const out = titles.slice(0, 5).map((m, i) => {
    let href = m[1]; const u = href.match(/uddg=([^&]+)/); if (u) href = decodeURIComponent(u[1]);
    return `${i + 1}. ${strip(m[2])}\n   ${snips[i] || ""}\n   ${href}`;
  });
  return out.join("\n");
}

async function wikipedia(q: string): Promise<string> {
  const txt = await ext("https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&srlimit=3&srsearch=" + encodeURIComponent(q));
  const j = JSON.parse(txt) as { query?: { search?: { title: string; snippet: string }[] } };
  const res = j.query?.search || [];
  if (!res.length) return "No Wikipedia results.";
  return res.map((r, i) => `${i + 1}. ${r.title}\n   ${strip(r.snippet)}\n   https://en.wikipedia.org/wiki/${encodeURIComponent(r.title.replace(/ /g, "_"))}`).join("\n");
}

async function arxiv(q: string): Promise<string> {
  const xml = await ext("http://export.arxiv.org/api/query?start=0&max_results=5&search_query=all:" + encodeURIComponent(q));
  const entries = [...xml.matchAll(/<entry>([\s\S]*?)<\/entry>/g)];
  if (!entries.length) return "No arXiv results.";
  return entries.slice(0, 5).map((e, i) => {
    const g = (tag: string) => (e[1].match(new RegExp(`<${tag}>([\\s\\S]*?)<\\/${tag}>`))?.[1] || "").trim();
    return `${i + 1}. ${strip(g("title"))}\n   ${strip(g("summary")).slice(0, 260)}…\n   ${g("id")}`;
  }).join("\n");
}// Resolve the DB the user connected on the MCP page (name "database"). Read-write
// unless they picked read-only (access-mode=restricted). Falls back to platform DATABASE_URL.
async function getDb(userId: string): Promise<{ conn: string; write: boolean } | null> {
  const rows = await db.select().from(mcpServers).where(and(eq(mcpServers.userId, userId), eq(mcpServers.name, "database"), eq(mcpServers.enabled, true))).limit(1);
  const r = rows[0]; if (r?.secretEnc) return { conn: decrypt(r.secretEnc), write: (r.command || "").includes("unrestricted") };
  if (process.env.DATABASE_URL) return { conn: process.env.DATABASE_URL, write: true };
  return null;
}

type SqlResult = { cols: string[]; rows: unknown[][]; rowCount: number };

// Which driver a connection string needs: libSQL/Turso vs MySQL vs Postgres.
function dbKind(conn: string): "libsql" | "mysql" | "pg" {
  if (/^mysql:|^mysql2:/i.test(conn)) return "mysql";
  return /^libsql:|\.turso\.io/i.test(conn) ? "libsql" : "pg";
}
function hostOf(conn: string): string { return (conn.match(/\/\/(?:[^/?@]+@)?([^/?:]+)/) || [])[1] || ""; }

// Turso / libSQL: conn is `libsql://<db>.turso.io?authToken=<token>`
async function libsqlRun(conn: string, sql: string): Promise<SqlResult> {
  const m = conn.match(/[?&]authToken=([^&]+)/i);
  const authToken = m ? decodeURIComponent(m[1]) : undefined;
  const url = conn.replace(/([?&])authToken=[^&]+/i, "$1").replace(/[?&]+$/, "");
  const { createClient } = await import("@libsql/client");
  const client = createClient({ url, authToken });
  try {
    const res = await client.execute(sql);
    const cols = res.columns ?? [];
    const rows = res.rows.slice(0, 50).map((r) => cols.map((c) => (r as unknown as Record<string, unknown>)[c]));
    return { cols, rows, rowCount: res.rowsAffected || res.rows.length };
  } finally { client.close(); }
}

async function mysqlRun(conn: string, sql: string): Promise<SqlResult> {
  const mysql = await import("mysql2/promise");
  const u = new URL(conn);
  const connection = await mysql.createConnection({
    host: u.hostname,
    port: Number(u.port || 3306),
    user: decodeURIComponent(u.username),
    password: decodeURIComponent(u.password),
    database: u.pathname.replace(/^\//, ""),
    ssl: { minVersion: "TLSv1.2", rejectUnauthorized: false },
    connectTimeout: 10000,
  });
  try {
    const [rows, fields] = await connection.query(sql);
    if (Array.isArray(rows)) {
      const cols = fields ? fields.map((f) => f.name) : Object.keys((rows[0] as object) || {});
      const rowData = (rows as Record<string, unknown>[]).slice(0, 50).map((r) => cols.map((c) => r[c]));
      return { cols, rows: rowData, rowCount: rows.length };
    }
    const affected = (rows as { affectedRows?: number })?.affectedRows || 0;
    return { cols: [], rows: [], rowCount: affected };
  } finally {
    await connection.end().catch(() => {});
  }
}

async function pgRun(conn: string, sql: string): Promise<SqlResult> {
  const { Client } = await import("pg");
  const attempt = async (ssl: false | { rejectUnauthorized: boolean }) => {
    const c = new Client({ connectionString: conn, connectionTimeoutMillis: 8000, query_timeout: 12000, statement_timeout: 12000, ...(ssl ? { ssl } : {}) });
    await c.connect();
    try { const res = await c.query(sql); return { cols: res.fields.map((f) => f.name), rows: (res.rows as Record<string, unknown>[]).slice(0, 50).map((row) => res.fields.map((f) => row[f.name])), rowCount: res.rowCount ?? res.rows.length }; }
    finally { await c.end().catch(() => {}); }
  };
  try { return await attempt(false); }
  catch (e) { if (/ssl|certificate|self.signed|no encryption/i.test((e as Error).message)) return attempt({ rejectUnauthorized: false }); throw e; }
}

async function runSql(conn: string, sql: string): Promise<SqlResult> {
  const host = hostOf(conn);
  if (host && await resolvesToPrivate(host)) throw new Error("Blocked host (internal/private address).");
  const kind = dbKind(conn);
  if (kind === "libsql") return libsqlRun(conn, sql);
  if (kind === "mysql") return mysqlRun(conn, sql);
  return pgRun(conn, sql);
}

export async function POST(req: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  if (!(await rateLimitDb("native", user.id, 40, 60_000))) return NextResponse.json({ error: "Too many requests — wait a minute." }, { status: 429 });
  const b = await req.json().catch(() => ({}));
  const tool = String(b.tool || ""); const input = String(b.input || "").trim();

  try {
    if (tool === "web_search") return NextResponse.json({ text: input ? await webSearch(input) : "Provide a search query." });
    if (tool === "wikipedia") return NextResponse.json({ text: input ? await wikipedia(input) : "Provide a topic." });
    if (tool === "arxiv") return NextResponse.json({ text: input ? await arxiv(input) : "Provide a search query." });

    if (tool === "db_schema" || tool === "db_query") {
      const d = await getDb(user.id);
      if (!d) return NextResponse.json({ text: "No database connected. Connect one on the MCP servers page ('Connect your database')." });
      if (tool === "db_schema") {
        const kind = dbKind(d.conn);
        const schemaSql = kind === "libsql"
          ? "SELECT m.name AS table_name, p.name AS column_name, p.type AS data_type FROM sqlite_master m, pragma_table_info(m.name) p WHERE m.type='table' AND m.name NOT LIKE 'sqlite_%' ORDER BY m.name, p.cid LIMIT 400"
          : kind === "mysql"
          ? "SELECT table_name, column_name, data_type FROM information_schema.columns WHERE table_schema = DATABASE() ORDER BY table_name, ordinal_position LIMIT 400"
          : "SELECT table_name, column_name, data_type FROM information_schema.columns WHERE table_schema = 'public' ORDER BY table_name, ordinal_position LIMIT 400";
        const r = await runSql(d.conn, schemaSql);
        const byTable = new Map<string, string[]>();
        r.rows.forEach((row) => { const t = String(row[0]); if (!byTable.has(t)) byTable.set(t, []); byTable.get(t)!.push(`${row[1]} ${row[2]}`); });
        if (!byTable.size) return NextResponse.json({ text: "No tables found in the database schema." });
        return NextResponse.json({ text: [...byTable.entries()].map(([t, cols]) => `${t}(${cols.join(", ")})`).join("\n") });
      }
      // db_query
      const sql = input.replace(/;\s*$/, "");
      const isRead = /^\s*(select|with|explain|show|table)\b/i.test(sql);
      if (!d.write && !isRead) return NextResponse.json({ text: "This database is connected read-only — only SELECT queries are allowed. Reconnect it as read-write to modify data." });
      const r = await runSql(d.conn, sql);
      if (!r.rows.length) return NextResponse.json({ text: `OK — ${r.rowCount} row(s) affected/returned. No rows to show.` });
      const header = r.cols.join(" | ");
      const body = r.rows.map((row) => row.map((v) => (v == null ? "null" : String(v))).join(" | ")).join("\n");
      return NextResponse.json({ text: `${header}\n${body}\n(${r.rows.length}${r.rowCount > r.rows.length ? " of " + r.rowCount : ""} rows)` });
    }

    return NextResponse.json({ error: "unknown tool" }, { status: 400 });
  } catch (e) {
    const errText = (e as Error).message;
    if (errText.includes("Unknown column") || errText.includes("doesn't exist") || errText.includes("Unknown table")) {
      return NextResponse.json({ text: `Database Query Note: ${errText}. If you are querying custom data (such as student gender, grades, or sales), attach your CSV dataset in the Agent Lab builder.` });
    }
    return NextResponse.json({ text: "Error: " + errText.slice(0, 200) });
  }
}
