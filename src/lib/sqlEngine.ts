import type { Table, Cell, Rec } from "./etlUtils";

/* eslint-disable @typescript-eslint/no-explicit-any */

const cell = (v: unknown): Cell =>
  v == null ? null
    : typeof v === "number" ? v
    : typeof v === "boolean" ? (v ? 1 : 0)
    : typeof v === "object" ? JSON.stringify(v)
    : String(v);

// Run real SQL over the landed rows (ELT transform). alasql is loaded lazily so
// it only ships when the ELT mode is actually used. Tables: `raw` (+ `b` if a
// Source B is loaded). Returns the result as a Table.
export async function runSql(sql: string, raw: Table, b?: Table | null): Promise<Table> {
  // alasql's node build pulls in react-native/fs shims; next.config stubs those
  // so the browser build compiles. We only use in-memory SQL here.
  const mod: any = await import("alasql");
  const alasql = mod.default || mod;
  const db = new alasql.Database();
  db.exec("CREATE TABLE raw"); db.tables.raw.data = raw.rows.map((r) => ({ ...r }));
  if (b && b.cols.length) { db.exec("CREATE TABLE b"); db.tables.b.data = b.rows.map((r) => ({ ...r })); }
  const res = db.exec(sql);
  if (!Array.isArray(res)) return { cols: [], rows: [] };
  const out = res as Record<string, unknown>[];
  const cols: string[] = [];
  out.forEach((r) => Object.keys(r).forEach((k) => { if (!cols.includes(k)) cols.push(k); }));
  const rows: Rec[] = out.map((r) => { const o: Rec = {}; cols.forEach((c) => (o[c] = cell(r[c]))); return o; });
  return { cols, rows };
}
