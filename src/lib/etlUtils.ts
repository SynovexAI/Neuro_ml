// Real in-browser ETL: parse records, apply Spark-style transforms, run the pipeline.
// Every operator genuinely transforms the row data — no mocks.

export type Cell = string | number | null;
export type Rec = Record<string, Cell>;
export interface Table { cols: string[]; rows: Rec[]; }

// RFC-4180-style tokenizer: walks the whole text so quoted fields may contain
// the delimiter AND embedded newlines. Returns an array of rows (each a string[]).
function tokenizeCsv(text: string, d: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [], cur = "", q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) {
      if (c === '"') { if (text[i + 1] === '"') { cur += '"'; i++; } else q = false; }
      else cur += c;
    } else if (c === '"') { q = true; }
    else if (c === d) { row.push(cur); cur = ""; }
    else if (c === "\n") { row.push(cur); rows.push(row); row = []; cur = ""; }
    else if (c === "\r") { /* handled by \n */ }
    else cur += c;
  }
  if (cur.length || row.length) { row.push(cur); rows.push(row); }
  return rows;
}
export function parseRecords(csv: string): Table {
  const text = csv.replace(/\r\n?/g, "\n").replace(/\n+$/, "");
  if (!text.trim()) return { cols: [], rows: [] };
  const first = text.split("\n", 1)[0];
  const d = (first.match(/;/g)?.length || 0) > (first.match(/,/g)?.length || 0) ? ";"
    : (first.match(/\t/g)?.length || 0) > (first.match(/,/g)?.length || 0) ? "\t" : ",";
  const grid = tokenizeCsv(text, d).filter((r) => r.length > 1 || (r[0] ?? "").trim() !== "");
  if (!grid.length) return { cols: [], rows: [] };
  const cols = grid[0].map((h) => h.trim());
  const rows = grid.slice(1).map((cells) => {
    const r: Rec = {};
    cols.forEach((h, i) => { const v = (cells[i] ?? "").trim(); r[h] = v === "" ? null : (!isNaN(Number(v)) ? Number(v) : v); });
    return r;
  });
  return { cols, rows };
}

export function sampleSources(): { key: string; label: string; csv: string }[] {
  return [
    { key: "orders", label: "orders (e-commerce)", csv:
`order_id,customer,region,amount,status
1001,Alice,US,120.5,paid
1002,Bob,EU,0,failed
1003,Cara,US,89.9,paid
1004,Dan,APAC,240,paid
1005,Eve,EU,55.25,refunded
1006,Finn,US,17.4,paid
1007,Gina,APAC,310,paid
1008,Hugo,EU,,failed
1009,Ivy,US,42,paid
1010,Jack,APAC,128.75,refunded
1011,Kira,EU,66,paid
1012,Leo,US,205.1,paid
1013,Mia,APAC,19.99,paid
1014,Nate,EU,150,paid
1015,Omar,US,0,failed
1016,Pia,APAC,88,paid` },
    { key: "events", label: "events (clickstream)", csv:
`user_id,event,page,duration
u1,view,home,12
u1,click,pricing,4
u2,view,home,30
u2,view,blog,45
u3,click,signup,8
u1,view,pricing,22
u3,view,home,15
u2,click,pricing,3
u4,view,home,9
u4,click,signup,11` },
    { key: "sensors", label: "sensors (IoT)", csv:
`sensor,zone,reading,ok
s1,north,21.4,1
s2,north,58.2,1
s3,south,102.5,0
s1,north,22.1,1
s2,south,49.8,1
s3,south,110,0
s1,north,20.9,1
s2,north,61.3,1` },
  ];
}

export type OpType = "filter" | "select" | "derive" | "aggregate" | "sort" | "dedupe" | "clean" | "rename" | "limit" | "sample" | "map" | "fillna" | "bucket";
export interface EtlOp {
  id: string; type: OpType;
  col?: string; op?: string; value?: string;                 // filter / limit / sample / fillna / bucket
  cols?: string[];                                            // select / dedupe
  name?: string; left?: string; arith?: string; right?: string; // derive / rename / bucket
  groupBy?: string; agg?: string; aggCol?: string;            // aggregate
  dir?: string;                                               // sort
  mode?: string;                                              // clean
  fn?: string;                                                // map
}
export const OP_META: Record<OpType, { label: string; icon: string; hint: string }> = {
  filter: { label: "Filter", icon: "🔻", hint: "keep only rows matching a condition (WHERE)" },
  select: { label: "Select", icon: "🧲", hint: "project a subset of columns" },
  derive: { label: "Derive", icon: "➕", hint: "add a computed column from two fields" },
  aggregate: { label: "Aggregate", icon: "∑", hint: "group by a column and aggregate (groupBy)" },
  sort: { label: "Sort", icon: "↕", hint: "order rows by a column" },
  dedupe: { label: "Dedupe", icon: "🧹", hint: "remove duplicate rows" },
  clean: { label: "Clean", icon: "🚿", hint: "handle nulls (drop rows or fill 0)" },
  rename: { label: "Rename", icon: "🏷", hint: "rename a column" },
  limit: { label: "Limit", icon: "✂", hint: "keep only the first N rows" },
  sample: { label: "Sample", icon: "🎲", hint: "randomly sample rows (fraction ≤1, or a count)" },
  map: { label: "Map column", icon: "🔧", hint: "apply a function to one column (upper / round / …)" },
  fillna: { label: "Fill nulls", icon: "🩹", hint: "replace nulls in a column with a value" },
  bucket: { label: "Bucketize", icon: "🪣", hint: "bin a numeric column into N buckets" },
};

const numify = (v: Cell): number => (v == null ? NaN : Number(v));
function compare(a: Cell, op: string, raw: string): boolean {
  const bn = Number(raw), an = numify(a);
  const numeric = !isNaN(an) && !isNaN(bn) && raw.trim() !== "";
  if (op === "contains") return String(a ?? "").toLowerCase().includes(raw.toLowerCase());
  if (op === "!=") return numeric ? an !== bn : String(a ?? "") !== raw;
  if (op === "==") return numeric ? an === bn : String(a ?? "") === raw;
  if (numeric) { if (op === ">") return an > bn; if (op === "<") return an < bn; if (op === ">=") return an >= bn; if (op === "<=") return an <= bn; }
  const as = String(a ?? ""); if (op === ">") return as > raw; if (op === "<") return as < raw; if (op === ">=") return as >= raw; if (op === "<=") return as <= raw;
  return false;
}

export function applyOp(t: Table, op: EtlOp): Table {
  const rows = t.rows;
  switch (op.type) {
    case "filter": {
      const col = op.col || t.cols[0];
      return { cols: t.cols, rows: rows.filter((r) => compare(r[col], op.op || "==", op.value ?? "")) };
    }
    case "select": {
      const cols = (op.cols && op.cols.length ? op.cols : t.cols).filter((c) => t.cols.includes(c));
      return { cols, rows: rows.map((r) => { const o: Rec = {}; cols.forEach((c) => (o[c] = r[c])); return o; }) };
    }
    case "derive": {
      const name = op.name || "derived"; const arith = op.arith || "+";
      const val = (r: Rec, k?: string) => { if (k == null) return NaN; return t.cols.includes(k) ? numify(r[k]) : Number(k); };
      const cols = [...t.cols, name];
      return { cols, rows: rows.map((r) => { const l = val(r, op.left), rr = val(r, op.right); let v: number = NaN; if (arith === "+") v = l + rr; else if (arith === "-") v = l - rr; else if (arith === "*") v = l * rr; else if (arith === "/") v = rr ? l / rr : NaN; return { ...r, [name]: isNaN(v) ? null : Math.round(v * 1000) / 1000 }; }) };
    }
    case "aggregate": {
      const gb = op.groupBy || t.cols[0]; const agg = op.agg || "count"; const ac = op.aggCol || t.cols[0];
      const groups = new Map<string, Rec[]>();
      rows.forEach((r) => { const k = String(r[gb] ?? "∅"); if (!groups.has(k)) groups.set(k, []); groups.get(k)!.push(r); });
      const outCol = agg === "count" ? "count" : `${agg}_${ac}`;
      const out: Rec[] = [...groups.entries()].map(([k, rs]) => {
        let val: number;
        if (agg === "count") val = rs.length;
        else { const nums = rs.map((x) => numify(x[ac])).filter((n) => !isNaN(n)); const sum = nums.reduce((a, b) => a + b, 0); val = agg === "sum" ? sum : agg === "avg" ? sum / (nums.length || 1) : agg === "min" ? Math.min(...nums) : Math.max(...nums); }
        return { [gb]: k, [outCol]: Math.round(val * 1000) / 1000 };
      });
      return { cols: [gb, outCol], rows: out };
    }
    case "sort": {
      const col = op.col || t.cols[0]; const dir = op.dir === "desc" ? -1 : 1;
      const sorted = [...rows].sort((a, b) => { const x = a[col], y = b[col]; const xn = numify(x), yn = numify(y); if (!isNaN(xn) && !isNaN(yn)) return (xn - yn) * dir; return String(x ?? "").localeCompare(String(y ?? "")) * dir; });
      return { cols: t.cols, rows: sorted };
    }
    case "dedupe": {
      const keys = (op.cols && op.cols.length ? op.cols : t.cols).filter((c) => t.cols.includes(c));
      const seen = new Set<string>(); const out: Rec[] = [];
      rows.forEach((r) => { const k = keys.map((c) => String(r[c])).join("¦"); if (!seen.has(k)) { seen.add(k); out.push(r); } });
      return { cols: t.cols, rows: out };
    }
    case "clean": {
      if (op.mode === "fill0") return { cols: t.cols, rows: rows.map((r) => { const o: Rec = { ...r }; t.cols.forEach((c) => { if (o[c] == null) o[c] = 0; }); return o; }) };
      return { cols: t.cols, rows: rows.filter((r) => t.cols.every((c) => r[c] != null && r[c] !== "")) }; // drop nulls
    }
    case "rename": {
      const from = op.col || t.cols[0]; const to = op.name || from;
      return { cols: t.cols.map((c) => (c === from ? to : c)), rows: rows.map((r) => { const o: Rec = {}; t.cols.forEach((c) => (o[c === from ? to : c] = r[c])); return o; }) };
    }
    case "limit": { const n = Math.max(0, parseInt(op.value || "10") || 0); return { cols: t.cols, rows: rows.slice(0, n) }; }
    case "sample": { const v = Number(op.value || "0.5"); const frac = v <= 1 ? v : v / (rows.length || 1); return { cols: t.cols, rows: rows.filter(() => Math.random() < frac) }; }
    case "map": {
      const col = op.col || t.cols[0]; const fn = op.fn || "round";
      const ap = (x: Cell): Cell => {
        if (x == null) return null;
        if (fn === "upper") return String(x).toUpperCase(); if (fn === "lower") return String(x).toLowerCase(); if (fn === "trim") return String(x).trim(); if (fn === "length") return String(x).length;
        const n = Number(x); if (isNaN(n)) return x;
        if (fn === "round") return Math.round(n); if (fn === "abs") return Math.abs(n); if (fn === "floor") return Math.floor(n); if (fn === "ceil") return Math.ceil(n); return x;
      };
      return { cols: t.cols, rows: rows.map((r) => ({ ...r, [col]: ap(r[col]) })) };
    }
    case "fillna": {
      const raw = op.value ?? "0"; const fv: Cell = raw.trim() !== "" && !isNaN(Number(raw)) ? Number(raw) : raw;
      const cols = op.col ? [op.col] : t.cols;
      return { cols: t.cols, rows: rows.map((r) => { const o: Rec = { ...r }; cols.forEach((c) => { if (o[c] == null) o[c] = fv; }); return o; }) };
    }
    case "bucket": {
      const col = op.col || t.cols[0]; const name = op.name || `${col}_bin`; const bins = Math.max(2, parseInt(op.value || "4") || 4);
      const nums = rows.map((r) => numify(r[col])).filter((n) => !isNaN(n)); const mn = Math.min(...nums), mx = Math.max(...nums), sp = (mx - mn) || 1;
      return { cols: [...t.cols, name], rows: rows.map((r) => { const v = numify(r[col]); let b: Cell = null; if (!isNaN(v)) { b = Math.floor(((v - mn) / sp) * bins); if (b >= bins) b = bins - 1; if (b < 0) b = 0; } return { ...r, [name]: b }; }) };
    }
  }
}

export function runPipeline(src: Table, ops: EtlOp[]): { stages: { op: EtlOp | null; table: Table }[]; final: Table } {
  const stages: { op: EtlOp | null; table: Table }[] = [{ op: null, table: src }];
  let cur = src;
  for (const op of ops) { cur = applyOp(cur, op); stages.push({ op, table: cur }); }
  return { stages, final: cur };
}

// Build a Table from parsed JSON (an array of objects, or {data:[…]} / {rows:[…]}).
export function tableFromRecords(arr: unknown[]): Table {
  const cols: string[] = []; const rows: Rec[] = [];
  for (const item of arr) {
    if (!item || typeof item !== "object" || Array.isArray(item)) continue;
    const o = item as Record<string, unknown>; const r: Rec = {};
    for (const k of Object.keys(o)) {
      if (!cols.includes(k)) cols.push(k);
      const v = o[k];
      r[k] = v == null ? null : typeof v === "number" ? v : typeof v === "boolean" ? (v ? 1 : 0) : typeof v === "object" ? JSON.stringify(v) : (String(v).trim() !== "" && !isNaN(Number(v)) ? Number(v) : String(v));
    }
    rows.push(r);
  }
  rows.forEach((r) => cols.forEach((c) => { if (!(c in r)) r[c] = null; }));
  return { cols, rows };
}
export function profile(t: Table): { name: string; type: string; nulls: number }[] {
  return t.cols.map((c) => { const vals = t.rows.map((r) => r[c]); const nn = vals.filter((v) => v != null); const numeric = nn.length > 0 && nn.every((v) => typeof v === "number"); return { name: c, type: numeric ? "num" : "text", nulls: vals.length - nn.length }; });
}

export function toCSV(t: Table): string {
  const esc = (v: Cell) => { const s = v == null ? "" : String(v); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
  return [t.cols.join(","), ...t.rows.map((r) => t.cols.map((c) => esc(r[c])).join(","))].join("\n");
}
