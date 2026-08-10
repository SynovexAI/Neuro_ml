import "server-only";
import mysql from "mysql2/promise";
import { Client as PgClient } from "pg";
import { resolvesToPrivate } from "@/lib/net";

// Server-side EXTRACT and LOAD against a user-supplied MySQL/TiDB, Postgres, or
// Turso (libSQL) URL. Mirrors /api/ml/query (extract) and /api/etl/store-external
// (load) — SSRF-guarded, SELECT-only reads, parameterized writes — so the ETL
// "full run" API can do extract → transform → load entirely on the server.

export type RunTable = { cols: string[]; rows: Record<string, unknown>[] };

const kind = (url: string) => ({
  isPg: /^postgres(ql)?:\/\//i.test(url),
  isMy: /^mysql:\/\//i.test(url),
  isLibsql: /^libsql:\/\//i.test(url) || /\.turso\.io/i.test(url),
});
const ident = (s: string) => /^[A-Za-z_][A-Za-z0-9_]{0,63}$/.test(s);
const EXTRACT_CAP = 5000;

function colsFrom(rows: Record<string, unknown>[]): string[] { return rows.length ? Object.keys(rows[0]) : []; }

// ── EXTRACT: run a read-only SELECT and return rows ──
export async function extractRows(url: string, query: string): Promise<RunTable> {
  const { isPg, isMy, isLibsql } = kind(url);
  if (!isPg && !isMy && !isLibsql) throw new Error("Source must be a mysql://, postgres://, or libsql:// (Turso) URL.");
  if (!/^\s*select\b/i.test(query || "")) throw new Error("The source query must be a single SELECT.");
  if (/\b(into\s+outfile|load_file|information_schema|mysql\.|pg_|;\s*\S)/i.test(query)) throw new Error("Source query rejected (only a single, plain SELECT is allowed).");
  const capped = /limit\s+\d+/i.test(query) ? query : query.replace(/;?\s*$/, ` LIMIT ${EXTRACT_CAP}`);

  if (isLibsql) {
    const host = (url.match(/\/\/(?:[^/?@]+@)?([^/?:]+)/) || [])[1] || "";
    if (host && await resolvesToPrivate(host)) throw new Error("That source host is blocked (internal / private address).");
    const m = url.match(/[?&]authToken=([^&]+)/i);
    const authToken = m ? decodeURIComponent(m[1]) : undefined;
    const clean = url.replace(/([?&])authToken=[^&]+/i, "$1").replace(/[?&]+$/, "");
    const { createClient } = await import("@libsql/client");
    const client = createClient({ url: clean, authToken });
    try {
      const res = await client.execute(capped);
      const cols = res.columns ?? [];
      const rows = res.rows.map((r) => cols.reduce((o, c) => { o[c] = (r as unknown as Record<string, unknown>)[c]; return o; }, {} as Record<string, unknown>));
      return { cols, rows };
    } finally { client.close(); }
  }

  const u = new URL(url);
  if (await resolvesToPrivate(u.hostname)) throw new Error("That source host is blocked (internal / private address).");

  if (isPg) {
    const ssl = /sslmode=disable/i.test(url) ? false : { rejectUnauthorized: false };
    const client = new PgClient({ connectionString: url, ssl, connectionTimeoutMillis: 12000, statement_timeout: 15000 });
    try { await client.connect(); const res = await client.query(capped); const rows = res.rows as Record<string, unknown>[]; return { cols: colsFrom(rows), rows }; }
    finally { await client.end().catch(() => {}); }
  }

  let conn: mysql.Connection | null = null;
  try {
    conn = await mysql.createConnection({
      host: u.hostname, port: Number(u.port || 3306), user: decodeURIComponent(u.username), password: decodeURIComponent(u.password),
      database: u.pathname.replace(/^\//, "") || undefined,
      ssl: u.hostname.includes("tidbcloud.com") ? { minVersion: "TLSv1.2", rejectUnauthorized: true } : undefined, connectTimeout: 12000,
    });
    const [rows] = await conn.query(capped + ";") as [Record<string, unknown>[], unknown];
    const arr = Array.isArray(rows) ? rows : [];
    return { cols: colsFrom(arr), rows: arr };
  } finally { if (conn) await conn.end().catch(() => {}); }
}

// ── LOAD: create the table if needed, then insert/upsert the rows ──
export async function loadRows(url: string, opts: { table: string; cols: string[]; rows: Record<string, unknown>[]; mode?: string; keyCol?: string }): Promise<{ rowCount: number; created: boolean; mode: string }> {
  const { isPg, isMy, isLibsql } = kind(url);
  if (!isPg && !isMy && !isLibsql) throw new Error("Target must be a mysql://, postgres://, or libsql:// (Turso) URL.");
  const table = String(opts.table || "").trim();
  const cols = (opts.cols || []).map(String);
  const rows = opts.rows || [];
  const mode = opts.mode === "upsert" ? "upsert" : "append";
  const keyCol = String(opts.keyCol || cols[0] || "");
  if (!ident(table)) throw new Error("Target table name must be a valid SQL identifier.");
  if (!cols.length || !cols.every(ident)) throw new Error("Output column names must be valid SQL identifiers (letters/digits/underscore).");
  if (!rows.length) throw new Error("Nothing to load — the pipeline produced 0 rows.");
  if (rows.length > 5000) throw new Error(`Too many rows to load (${rows.length}); cap is 5000. Add a Limit or use the Python export.`);

  const typeOf = (c: string): string => {
    let allNum = true, allInt = true, any = false;
    for (const r of rows) { const v = r[c]; if (v == null || v === "") continue; any = true; const n = Number(v); if (typeof v === "number" || (!isNaN(n) && String(v).trim() !== "")) { if (!Number.isInteger(n)) allInt = false; } else { allNum = false; break; } }
    if (!any) return "TEXT"; return allNum ? (allInt ? "BIGINT" : "DOUBLE") : "TEXT";
  };
  const toVal = (v: unknown) => (v == null ? null : typeof v === "object" ? JSON.stringify(v) : (v as string | number));

  if (isLibsql) {
    const host = (url.match(/\/\/(?:[^/?@]+@)?([^/?:]+)/) || [])[1] || "";
    if (host && await resolvesToPrivate(host)) throw new Error("That target host is blocked (internal / private address).");
    const m = url.match(/[?&]authToken=([^&]+)/i);
    const authToken = m ? decodeURIComponent(m[1]) : undefined;
    const clean = url.replace(/([?&])authToken=[^&]+/i, "$1").replace(/[?&]+$/, "");
    const { createClient } = await import("@libsql/client");
    const client = createClient({ url: clean, authToken });
    const sqliteType = (c: string) => { const t = typeOf(c); return t === "BIGINT" ? "INTEGER" : t === "DOUBLE" ? "REAL" : "TEXT"; };
    try {
      const info = await client.execute(`PRAGMA table_info("${table}")`);
      const created = info.rows.length === 0;
      if (created) await client.execute(`CREATE TABLE "${table}" (${cols.map((c) => `"${c}" ${sqliteType(c)}`).join(", ")})`);
      if (mode === "upsert") await client.execute(`CREATE UNIQUE INDEX IF NOT EXISTS "uk_${keyCol}" ON "${table}"("${keyCol}")`).catch(() => {});
      const colList = cols.map((c) => `"${c}"`).join(", ");
      const onConf = mode === "upsert" ? ` ON CONFLICT("${keyCol}") DO UPDATE SET ${cols.filter((c) => c !== keyCol).map((c) => `"${c}"=excluded."${c}"`).join(", ")}` : "";
      const ph = `(${cols.map(() => "?").join(", ")})`;
      for (let i = 0; i < rows.length; i += 200) {
        const batch = rows.slice(i, i + 200).map((r) => ({ sql: `INSERT INTO "${table}" (${colList}) VALUES ${ph}${onConf}`, args: cols.map((c) => toVal(r[c])) }));
        await client.batch(batch, "write");
      }
      return { rowCount: rows.length, created, mode };
    } finally { client.close(); }
  }

  const u = new URL(url);
  if (await resolvesToPrivate(u.hostname)) throw new Error("That target host is blocked (internal / private address).");

  if (isPg) {
    const pgTypeOf = (c: string) => { const t = typeOf(c); return t === "BIGINT" ? "BIGINT" : t === "DOUBLE" ? "DOUBLE PRECISION" : "TEXT"; };
    const ssl = /sslmode=disable/i.test(url) ? false : { rejectUnauthorized: false };
    const client = new PgClient({ connectionString: url, ssl, connectionTimeoutMillis: 12000 });
    try {
      await client.connect();
      const ex = await client.query("SELECT column_name FROM information_schema.columns WHERE table_name = $1 AND table_schema = current_schema()", [table]);
      const created = ex.rows.length === 0;
      if (created) await client.query(`CREATE TABLE "${table}" (${cols.map((c) => `"${c}" ${pgTypeOf(c)}`).join(", ")})`);
      if (mode === "upsert") await client.query(`ALTER TABLE "${table}" ADD CONSTRAINT "uk_${keyCol}" UNIQUE ("${keyCol}")`).catch(() => {});
      const colList = cols.map((c) => `"${c}"`).join(", ");
      const onConf = mode === "upsert" ? ` ON CONFLICT ("${keyCol}") DO UPDATE SET ${cols.filter((c) => c !== keyCol).map((c) => `"${c}"=EXCLUDED."${c}"`).join(", ")}` : "";
      for (let i = 0; i < rows.length; i += 200) {
        const batch = rows.slice(i, i + 200); const params: unknown[] = [];
        const tuples = batch.map((r, ri) => { cols.forEach((c) => params.push(toVal(r[c]))); return `(${cols.map((_, ci) => `$${ri * cols.length + ci + 1}`).join(", ")})`; });
        await client.query(`INSERT INTO "${table}" (${colList}) VALUES ${tuples.join(", ")}${onConf}`, params);
      }
      return { rowCount: rows.length, created, mode };
    } finally { await client.end().catch(() => {}); }
  }

  let conn: mysql.Connection | null = null;
  try {
    conn = await mysql.createConnection({
      host: u.hostname, port: Number(u.port || 3306), user: decodeURIComponent(u.username), password: decodeURIComponent(u.password),
      database: u.pathname.replace(/^\//, "") || undefined,
      ssl: u.hostname.includes("tidbcloud.com") ? { minVersion: "TLSv1.2", rejectUnauthorized: true } : undefined, connectTimeout: 12000,
    });
    const [ex] = await conn.query("SELECT COLUMN_NAME FROM information_schema.COLUMNS WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = ?", [table]) as [Array<{ COLUMN_NAME: string }>, unknown];
    const created = !(ex && ex.length);
    if (created) await conn.query(`CREATE TABLE \`${table}\` (${cols.map((c) => `\`${c}\` ${typeOf(c)}`).join(", ")})`);
    if (mode === "upsert") await conn.query(`ALTER TABLE \`${table}\` ADD UNIQUE KEY \`uk_${keyCol}\` (\`${keyCol}\`)`).catch(() => {});
    const colList = cols.map((c) => `\`${c}\``).join(", ");
    const onDup = mode === "upsert" ? ` ON DUPLICATE KEY UPDATE ${cols.filter((c) => c !== keyCol).map((c) => `\`${c}\`=VALUES(\`${c}\`)`).join(", ")}` : "";
    for (let i = 0; i < rows.length; i += 500) {
      const batch = rows.slice(i, i + 500).map((r) => cols.map((c) => toVal(r[c])));
      await conn.query(`INSERT INTO \`${table}\` (${colList}) VALUES ?${onDup}`, [batch]);
    }
    return { rowCount: rows.length, created, mode };
  } finally { if (conn) await conn.end().catch(() => {}); }
}
