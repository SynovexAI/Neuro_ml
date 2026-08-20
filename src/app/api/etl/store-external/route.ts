import { NextResponse } from "next/server";
import mysql from "mysql2/promise";
import { Client as PgClient } from "pg";
import { getSessionUser } from "@/lib/auth";
import { resolvesToPrivate } from "@/lib/net";
import { rateLimitDb } from "@/lib/ratelimit";
import { audit } from "@/lib/monitor";
import { MongoClient } from "mongodb";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 60;

const MAX_ROWS = 5000;
const ident = (s: string) => /^[A-Za-z_][A-Za-z0-9_]{0,63}$/.test(s);
// Overrides come from a dropdown, but validate against an allowlist anyway
// (these get interpolated into DDL).
const ALLOWED_TYPES = new Set(["TEXT", "BIGINT", "DOUBLE", "VARCHAR(255)", "DATE", "DATETIME", "BOOLEAN"]);
// MySQL type → Postgres equivalent for the DDL.
const PG_TYPE: Record<string, string> = { TEXT: "TEXT", BIGINT: "BIGINT", DOUBLE: "DOUBLE PRECISION", "VARCHAR(255)": "VARCHAR(255)", DATE: "DATE", DATETIME: "TIMESTAMP", BOOLEAN: "BOOLEAN" };

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
  const typeOverrides: Record<string, string> = b.types && typeof b.types === "object" ? b.types : {};

  const isPg = /^(postgres(ql)?|redshift):\/\//i.test(url);
  const isMy = /^mysql:\/\//i.test(url);
  const isLibsql = /^libsql:\/\//i.test(url) || /\.turso\.io/i.test(url);
  const isMongo = /^mongodb(\+srv)?:\/\//i.test(url);
  if (!isPg && !isMy && !isLibsql && !isMongo) return NextResponse.json({ error: "Provide a valid mysql://, postgres://, redshift://, libsql:// (Turso), or mongodb:// connection URL." }, { status: 400 });
  if (!ident(table)) return NextResponse.json({ error: "Table name must be letters/digits/underscore and start with a letter." }, { status: 400 });
  if (!cols.length || (!isMongo && !cols.every(ident))) return NextResponse.json({ error: "Column names must be valid SQL identifiers." }, { status: 400 });
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
  const colTypes = cols.map((c) => { const ov = typeOverrides[c]; const ty = ov && ov !== "auto" && ALLOWED_TYPES.has(ov) ? ov : typeOf(c); return `\`${c}\` ${ty}`; });

  try {
    // ── Turso / libSQL path (SQLite dialect: double-quoted idents, ? params, ON CONFLICT upsert) ──
    if (isLibsql) {
      const host = (url.match(/\/\/(?:[^/?@]+@)?([^/?:]+)/) || [])[1] || "";
      if (host && await resolvesToPrivate(host)) return NextResponse.json({ error: "That host is blocked (internal / private address)." }, { status: 400 });
      const m = url.match(/[?&]authToken=([^&]+)/i);
      const authToken = m ? decodeURIComponent(m[1]) : undefined;
      const clean = url.replace(/([?&])authToken=[^&]+/i, "$1").replace(/[?&]+$/, "");
      const { createClient } = await import("@libsql/client");
      const client = createClient({ url: clean, authToken });
      const sqliteType = (c: string): string => { const ov = typeOverrides[c]; const t = ov && ov !== "auto" && ALLOWED_TYPES.has(ov) ? ov : typeOf(c); return t === "BIGINT" ? "INTEGER" : t === "DOUBLE" ? "REAL" : "TEXT"; };
      try {
        const info = await client.execute(`PRAGMA table_info("${table}")`);
        const existingCols = info.rows.map((r) => String((r as unknown as Record<string, unknown>).name));
        const tableExisted = existingCols.length > 0;
        if (tableExisted) {
          const missing = cols.filter((c) => !existingCols.includes(c));
          if (missing.length) return NextResponse.json({ error: `Schema drift — the existing table "${table}" is missing column(s): ${missing.join(", ")}. Load into a new table, or add those columns first.`, drift: { missing, extra: existingCols.filter((c) => !cols.includes(c)) } }, { status: 409 });
        } else {
          await client.execute(`CREATE TABLE "${table}" (${cols.map((c) => `"${c}" ${sqliteType(c)}`).join(", ")})`);
        }
        if (mode === "upsert") await client.execute(`CREATE UNIQUE INDEX IF NOT EXISTS "uk_${keyCol}" ON "${table}"("${keyCol}")`).catch(() => {});
        const colList = cols.map((c) => `"${c}"`).join(", ");
        const onConf = mode === "upsert" ? ` ON CONFLICT("${keyCol}") DO UPDATE SET ${cols.filter((c) => c !== keyCol).map((c) => `"${c}"=excluded."${c}"`).join(", ")}` : "";
        const ph = `(${cols.map(() => "?").join(", ")})`;
        const toVal = (v: unknown) => (v == null ? null : typeof v === "object" ? JSON.stringify(v) : (v as string | number));
        for (let i = 0; i < rows.length; i += 200) {
          const batch = rows.slice(i, i + 200).map((r) => ({ sql: `INSERT INTO "${table}" (${colList}) VALUES ${ph}${onConf}`, args: cols.map((c) => toVal(r[c])) }));
          await client.batch(batch, "write");
        }
        { await audit("etl_stored", user.id, { table, rows: rows.length, mode, backend: isLibsql ? "turso" : isPg ? "postgres" : "mysql" }).catch(() => {}); return NextResponse.json({ ok: true, table, rowCount: rows.length, mode, created: !tableExisted }); }
      } finally { client.close(); }
    }

    const u = new URL(url);
    if (await resolvesToPrivate(u.hostname)) return NextResponse.json({ error: "That host is blocked (internal / private address)." }, { status: 400 });

    if (isMongo) {
      const dbName = String(b.db || "").trim();
      if (!dbName) return NextResponse.json({ error: "Database name is required for MongoDB." }, { status: 400 });
      const client = new MongoClient(url, { serverSelectionTimeoutMS: 5000 });
      try {
        await client.connect();
        const db = client.db(dbName);
        const coll = db.collection(table);
        if (mode === "upsert") {
          const bulkOps = rows.map((r) => {
            const filter: Record<string, unknown> = {}; filter[keyCol] = r[keyCol];
            return { updateOne: { filter, update: { $set: r }, upsert: true } };
          });
          await coll.bulkWrite(bulkOps);
        } else {
          await coll.insertMany(rows as Record<string, unknown>[]);
        }
        return NextResponse.json({ ok: true, table, rowCount: rows.length, mode, created: false });
      } finally { await client.close(); }
    }

    // ── Postgres path (double-quoted idents, $n params, ON CONFLICT upsert) ──
    if (isPg) {
      const pgTypeOf = (c: string): string => { const ov = typeOverrides[c]; if (ov && ov !== "auto" && ALLOWED_TYPES.has(ov)) return PG_TYPE[ov]; const t = typeOf(c); return t === "BIGINT" ? "BIGINT" : t === "DOUBLE" ? "DOUBLE PRECISION" : "TEXT"; };
      const ssl = /sslmode=disable/i.test(url) ? false : { rejectUnauthorized: false };
      const client = new PgClient({ connectionString: url, ssl, connectionTimeoutMillis: 12000 });
      try {
        await client.connect();
        const ex = await client.query("SELECT column_name FROM information_schema.columns WHERE table_name = $1 AND table_schema = current_schema()", [table]);
        const existingCols = ex.rows.map((r: { column_name: string }) => r.column_name);
        const tableExisted = existingCols.length > 0;
        if (tableExisted) {
          const missing = cols.filter((c) => !existingCols.includes(c));
          if (missing.length) return NextResponse.json({ error: `Schema drift — the existing table "${table}" is missing column(s): ${missing.join(", ")}. Load into a new table, or add those columns first.`, drift: { missing, extra: existingCols.filter((c: string) => !cols.includes(c)) } }, { status: 409 });
        } else {
          await client.query(`CREATE TABLE "${table}" (${cols.map((c) => `"${c}" ${pgTypeOf(c)}`).join(", ")})`);
        }
        if (mode === "upsert") await client.query(`ALTER TABLE "${table}" ADD CONSTRAINT "uk_${keyCol}" UNIQUE ("${keyCol}")`).catch(() => {});
        const colList = cols.map((c) => `"${c}"`).join(", ");
        const onConf = mode === "upsert" ? ` ON CONFLICT ("${keyCol}") DO UPDATE SET ${cols.filter((c) => c !== keyCol).map((c) => `"${c}"=EXCLUDED."${c}"`).join(", ")}` : "";
        const toVal = (v: unknown) => (v == null ? null : typeof v === "object" ? JSON.stringify(v) : v);
        const B = 200;
        for (let i = 0; i < rows.length; i += B) {
          const batch = rows.slice(i, i + B);
          const params: unknown[] = [];
          const tuples = batch.map((r, ri) => { cols.forEach((c) => params.push(toVal(r[c]))); return `(${cols.map((_, ci) => `$${ri * cols.length + ci + 1}`).join(", ")})`; });
          await client.query(`INSERT INTO "${table}" (${colList}) VALUES ${tuples.join(", ")}${onConf}`, params);
        }
        { await audit("etl_stored", user.id, { table, rows: rows.length, mode, backend: isLibsql ? "turso" : isPg ? "postgres" : "mysql" }).catch(() => {}); return NextResponse.json({ ok: true, table, rowCount: rows.length, mode, created: !tableExisted }); }
      } finally { await client.end().catch(() => {}); }
    }

    let conn: mysql.Connection | null = null;
    try {
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
    { await audit("etl_stored", user.id, { table, rows: rows.length, mode, backend: isLibsql ? "turso" : isPg ? "postgres" : "mysql" }).catch(() => {}); return NextResponse.json({ ok: true, table, rowCount: rows.length, mode, created: !tableExisted }); }
    } finally {
      if (conn) await conn.end().catch(() => {});
    }
  } catch (e) {
    return NextResponse.json({ error: `Database error: ${(e as Error).message}` }, { status: 502 });
  }
}
