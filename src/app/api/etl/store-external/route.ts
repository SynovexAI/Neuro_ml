import { NextResponse } from "next/server";
import mysql from "mysql2/promise";
import { getSessionUser } from "@/lib/auth";
import { resolvesToPrivate } from "@/lib/net";
import { rateLimitDb } from "@/lib/ratelimit";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const MAX_ROWS = 5000;
const ident = (s: string) => /^[A-Za-z_][A-Za-z0-9_]{0,63}$/.test(s);

// Loads a pipeline output into a table in the USER'S OWN external MySQL/TiDB
// (they supply the connection). SSRF-guarded; identifiers validated; values
// parameterized. Creates the table if missing, then appends the rows.
export async function POST(req: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  if (!(await rateLimitDb("etlext", user.id, 10, 60_000))) return NextResponse.json({ error: "Too many writes — wait a minute." }, { status: 429 });

  const b = await req.json().catch(() => ({}));
  const url = String(b.url || "");
  const table = String(b.table || "").trim();
  const cols: string[] = Array.isArray(b.cols) ? b.cols.map(String) : [];
  const rows: Record<string, unknown>[] = Array.isArray(b.rows) ? b.rows : [];
  const mode = b.mode === "upsert" ? "upsert" : "append";
  const keyCol = String(b.keyCol || "");

  if (!/^mysql:\/\//i.test(url)) return NextResponse.json({ error: "Provide a mysql:// connection URL." }, { status: 400 });
  if (!ident(table)) return NextResponse.json({ error: "Table name must be letters/digits/underscore and start with a letter." }, { status: 400 });
  if (!cols.length || !cols.every(ident)) return NextResponse.json({ error: "Column names must be valid SQL identifiers (letters/digits/underscore)." }, { status: 400 });
  if (mode === "upsert" && (!keyCol || !cols.includes(keyCol))) return NextResponse.json({ error: "Upsert needs a key column that exists in the output." }, { status: 400 });
  if (!rows.length) return NextResponse.json({ error: "Nothing to store — the output is empty." }, { status: 400 });
  if (rows.length > MAX_ROWS) return NextResponse.json({ error: `Too many rows (${rows.length}). Cap is ${MAX_ROWS} — add a Limit first.` }, { status: 413 });
  if (JSON.stringify(rows).length > 4_000_000) return NextResponse.json({ error: "Output is too large (over ~4 MB)." }, { status: 413 });

  // Infer a column type per column from the data (all-integer → BIGINT, all-number → DOUBLE, else TEXT).
  const typeOf = (c: string): string => {
    let allNum = true, allInt = true, any = false;
    for (const r of rows) { const v = r[c]; if (v == null || v === "") continue; any = true; const n = Number(v); if (typeof v === "number" || (!isNaN(n) && String(v).trim() !== "")) { if (!Number.isInteger(n)) allInt = false; } else { allNum = false; break; } }
    if (!any) return "TEXT"; return allNum ? (allInt ? "BIGINT" : "DOUBLE") : "TEXT";
  };
  const colTypes = cols.map((c) => `\`${c}\` ${typeOf(c)}`);

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
    // Schema-drift preflight: if the table already exists, its columns must
    // cover the output — otherwise the insert would fail confusingly.
    const [ex] = await conn.query("SELECT COLUMN_NAME FROM information_schema.COLUMNS WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = ?", [table]) as [Array<{ COLUMN_NAME: string }>, unknown];
    const existingCols = (ex || []).map((r) => r.COLUMN_NAME);
    const tableExisted = existingCols.length > 0;
    if (tableExisted) {
      const missing = cols.filter((c) => !existingCols.includes(c));
      if (missing.length) return NextResponse.json({ error: `Schema drift — the existing table \`${table}\` is missing column(s): ${missing.join(", ")}. Load into a new table, or add those columns first.`, drift: { missing, extra: existingCols.filter((c) => !cols.includes(c)) } }, { status: 409 });
    } else {
      await conn.query(`CREATE TABLE \`${table}\` (${colTypes.join(", ")})`);
    }

    if (mode === "upsert") {
      // Best-effort unique key on the key column so ON DUPLICATE KEY works.
      await conn.query(`ALTER TABLE \`${table}\` ADD UNIQUE KEY \`uk_${keyCol}\` (\`${keyCol}\`)`).catch(() => {});
    }

    const colList = cols.map((c) => `\`${c}\``).join(", ");
    const onDup = mode === "upsert" ? ` ON DUPLICATE KEY UPDATE ${cols.filter((c) => c !== keyCol).map((c) => `\`${c}\`=VALUES(\`${c}\`)`).join(", ")}` : "";
    const toVal = (v: unknown) => (v == null ? null : typeof v === "object" ? JSON.stringify(v) : v);
    for (let i = 0; i < rows.length; i += 500) {
      const batch = rows.slice(i, i + 500).map((r) => cols.map((c) => toVal(r[c])));
      await conn.query(`INSERT INTO \`${table}\` (${colList}) VALUES ?${onDup}`, [batch]);
    }
    return NextResponse.json({ ok: true, table, rowCount: rows.length, mode, created: !tableExisted });
  } catch (e) {
    return NextResponse.json({ error: `Database error: ${(e as Error).message}` }, { status: 502 });
  } finally {
    if (conn) await conn.end().catch(() => {});
  }
}
