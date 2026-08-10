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
const IDENT = /^[A-Za-z_][A-Za-z0-9_]*$/;

export async function runSql(sql: string, raw: Table, b?: Table | null, extra?: { name: string; table: Table }[]): Promise<Table> {
  // alasql's node build pulls in react-native/fs shims; next.config stubs those
  // so the browser build compiles. We only use in-memory SQL here.
  const mod: any = await import("alasql");
  const alasql = mod.default || mod;
  const db = new alasql.Database();
  const register = (name: string, t: Table) => {
    if (!IDENT.test(name)) return;
    if (!db.tables[name]) {
      db.exec(`CREATE TABLE ${name}`);
    }
    db.tables[name].data = t.rows.map((r) => ({ ...r }));
  };
  register("raw", raw);
  if (b && b.cols.length) register("b", b);
  (extra || []).forEach((e) => register(e.name, e.table));

  // Dynamically register any table name referenced in the SQL query (e.g. FROM tableid, FROM student_performance_dataset)
  const referencedTables = Array.from(sql.matchAll(/\b(?:FROM|JOIN|INTO|UPDATE|TABLE)\s+([A-Za-z0-9_]+)/gi)).map((m) => m[1]);
  referencedTables.forEach((tblName) => {
    if (IDENT.test(tblName) && !db.tables[tblName]) {
      register(tblName, raw);
    }
  });

  const res = db.exec(sql);
  if (!Array.isArray(res)) return { cols: [], rows: [] };
  const out = res as Record<string, unknown>[];
  const cols: string[] = [];
  out.forEach((r) => Object.keys(r).forEach((k) => { if (!cols.includes(k)) cols.push(k); }));
  const rows: Rec[] = out.map((r) => { const o: Rec = {}; cols.forEach((c) => (o[c] = cell(r[c]))); return o; });
  return { cols, rows };
}
