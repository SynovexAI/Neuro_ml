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
`order_id,customer,region,amount,status,ordered_at
1001,Alice,US,120.5,paid,2026-01-04
1002,Bob,EU,0,failed,2026-01-05
1003,Cara,US,89.9,paid,2026-01-05
1004,Dan,APAC,240,paid,2026-01-07
1005,Eve,EU,55.25,refunded,2026-01-09
1006,Finn,US,17.4,paid,2026-01-11
1007,Gina,APAC,310,paid,2026-01-12
1008,Hugo,EU,,failed,2026-01-13
1009,Ivy,US,42,paid,2026-01-15
1010,Jack,APAC,128.75,refunded,2026-01-16
1011,Kira,EU,66,paid,2026-01-18
1012,Leo,US,205.1,paid,2026-01-20
1013,Mia,APAC,19.99,paid,2026-01-22
1014,Nate,EU,150,paid,2026-01-24
1015,Omar,US,0,failed,2026-01-25
1016,Pia,APAC,88,paid,2026-01-27` },
    { key: "customers", label: "customers (join with orders)", csv:
`customer,tier,country,signup_year
Alice,gold,USA,2021
Bob,silver,Germany,2022
Cara,gold,USA,2020
Dan,bronze,Japan,2023
Eve,silver,France,2021
Finn,bronze,USA,2024
Gina,gold,Australia,2019
Hugo,silver,Germany,2022
Ivy,bronze,USA,2023
Jack,gold,Singapore,2020
Kira,silver,Spain,2021
Leo,gold,USA,2018` },
    { key: "regions", label: "regions (lookup)", csv:
`region,manager,quota
US,Sam,1000
EU,Lena,800
APAC,Ravi,1200` },
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

export type OpType =
  | "filter" | "select" | "derive" | "aggregate" | "sort" | "dedupe" | "clean"
  | "rename" | "limit" | "sample" | "map" | "fillna" | "bucket"
  | "join" | "union" | "pivot" | "unpivot" | "window" | "regex" | "dateparse"
  | "lookup" | "merge" | "append";
export interface EtlOp {
  id: string; type: OpType;
  col?: string; op?: string; value?: string;                 // filter / limit / sample / fillna / bucket / regex
  cols?: string[];                                            // select / dedupe / unpivot
  name?: string; left?: string; arith?: string; right?: string; // derive / rename / bucket / unpivot (name=var,value=val)
  groupBy?: string; agg?: string; aggCol?: string;            // aggregate / pivot / window (partition)
  dir?: string;                                               // sort
  mode?: string;                                              // clean / union (all|distinct)
  fn?: string;                                                // map / window / dateparse
  joinType?: string; rightKey?: string;                       // join
}
export const OP_META: Record<OpType, { label: string; icon: string; hint: string; needsB?: boolean }> = {
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
  pivot: { label: "Pivot", icon: "📊", hint: "reshape data by turning rows into columns", needsB: true },
  join: { label: "Join", icon: "🔗", hint: "join with Source B on a key (inner / left / right)", needsB: true },
  union: { label: "Union", icon: "⧉", hint: "stack Source B's rows under this table", needsB: true },
  lookup: { label: "Lookup", icon: "🔎", hint: "lookup matching rows from secondary table (left join)", needsB: true },
  merge: { label: "Merge", icon: "🔀", hint: "merge fields from secondary into primary based on key", needsB: true },
  append: { label: "Append", icon: "➕", hint: "append rows from secondary table (union)", needsB: true },
  unpivot: { label: "Unpivot", icon: "🔃", hint: "melt several columns into variable/value rows (long)" },
  window: { label: "Window", icon: "🪟", hint: "running total, rank, row number, lag or lead" },
  regex: { label: "Regex extract", icon: "🔍", hint: "extract the first match/group of a pattern into a new column" },
  dateparse: { label: "Parse date", icon: "📅", hint: "pull year/month/day/weekday or days-since from a date column" },
};

const numify = (v: Cell): number => {
  if (v == null || v === "") return NaN;
  if (typeof v === "number") return v;
  const s = String(v).replace(/[$,\s]/g, "");
  return s === "" ? NaN : Number(s);
};
function aggNums(nums: number[], agg: string): number {
  const sum = nums.reduce((a, b) => a + b, 0);
  const v = agg === "sum" ? sum : agg === "avg" ? sum / (nums.length || 1) : agg === "min" ? (nums.length ? Math.min(...nums) : 0) : agg === "max" ? (nums.length ? Math.max(...nums) : 0) : nums.length;
  return Math.round(v * 1000) / 1000;
}
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

export interface OpCtx { secondary?: Table | null; }

export function applyOp(t: Table, op: EtlOp, ctx?: OpCtx): Table {
  const rows = t.rows;
  switch (op.type) {
    case "filter": {
      const col = op.col || t.cols[0];
      return { cols: t.cols, rows: rows.filter((r) => compare(r[col], op.op || "==", op.value ?? "")) };
    }
    case "select": {
      const cols = op.cols ? op.cols.filter((c) => t.cols.includes(c)) : t.cols;
      return { cols, rows: rows.map((r) => { const o: Rec = {}; cols.forEach((c) => (o[c] = r[c])); return o; }) };
    }
    case "derive": {
      const name = op.name || "derived"; const arith = op.arith || "+";
      const val = (r: Rec, k?: string) => { if (k == null) return NaN; return t.cols.includes(k) ? numify(r[k]) : numify(k); };
      const cols = [...t.cols, name];
      return { cols, rows: rows.map((r) => { const l = val(r, op.left), rr = val(r, op.right); let v: number = NaN; if (arith === "+") v = l + rr; else if (arith === "-") v = l - rr; else if (arith === "*") v = l * rr; else if (arith === "/") v = rr ? l / rr : NaN; return { ...r, [name]: isNaN(v) ? null : Math.round(v * 1000) / 1000 }; }) };
    }
    case "aggregate": {
      const gb = op.groupBy || t.cols[0]; const agg = op.agg || "count"; const ac = op.aggCol || t.cols[0];
      const groups = new Map<string, Rec[]>();
      rows.forEach((r) => { const k = String(r[gb] ?? "∅"); if (!groups.has(k)) groups.set(k, []); groups.get(k)!.push(r); });
      const outCol = agg === "count" ? "count" : `${agg}_${ac}`;
      const out: Rec[] = [...groups.entries()].map(([k, rs]) => {
        const val = agg === "count" ? rs.length : aggNums(rs.map((x) => numify(x[ac])).filter((n) => !isNaN(n)), agg);
        return { [gb]: k, [outCol]: val };
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
      return { cols: t.cols, rows: rows.filter((r) => t.cols.every((c) => r[c] != null && r[c] !== "")) };
    }
    case "rename": {
      const from = op.col || t.cols[0]; const to = op.name || from;
      return { cols: t.cols.map((c) => (c === from ? to : c)), rows: rows.map((r) => { const o: Rec = {}; t.cols.forEach((c) => (o[c === from ? to : c] = r[c])); return o; }) };
    }
    case "limit": { const n = Math.max(0, parseInt(op.value || "10") || 0); return { cols: t.cols, rows: rows.slice(0, n) }; }
    case "sample": { const v = Number(op.value || "0.5"); const frac = v <= 1 ? v : v / (rows.length || 1); return { cols: t.cols, rows: rows.filter((_, i) => { const h = Math.sin(i * 12.9898 + 78.233) * 43758.5453; return (h - Math.floor(h)) < frac; }) }; }
    case "map": {
      const col = op.col || t.cols[0]; const fn = op.fn || "round";
      const ap = (x: Cell): Cell => {
        if (x == null) return null;
        if (fn === "upper") return String(x).toUpperCase(); if (fn === "lower") return String(x).toLowerCase(); if (fn === "trim") return String(x).trim(); if (fn === "length") return String(x).length;
        const n = numify(x); if (isNaN(n)) return x;
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
    case "join": {
      const b = ctx?.secondary; if (!b || !b.cols.length) return t;
      const lk = op.col || t.cols[0]; const rk = op.rightKey || b.cols[0]; const jt = op.joinType || "inner";
      const bCols = b.cols.map((c) => (t.cols.includes(c) ? `b_${c}` : c));
      const cols = [...t.cols, ...bCols];
      const idx = new Map<string, Rec[]>();
      b.rows.forEach((r) => { const k = String(r[rk] ?? "∅"); if (!idx.has(k)) idx.set(k, []); idx.get(k)!.push(r); });
      const blank = (side: "b") => { const o: Rec = {}; (side === "b" ? bCols : t.cols).forEach((c) => (o[c] = null)); return o; };
      const bRow = (r: Rec) => { const o: Rec = {}; b.cols.forEach((c, i) => (o[bCols[i]] = r[c])); return o; };
      const out: Rec[] = []; const matchedB = new Set<Rec>();
      for (const l of rows) {
        const ms = idx.get(String(l[lk] ?? "∅")) || [];
        if (ms.length) ms.forEach((m) => { matchedB.add(m); out.push({ ...l, ...bRow(m) }); });
        else if (jt !== "inner") out.push({ ...l, ...blank("b") });
      }
      if (jt === "right") { const lBlank: Rec = {}; t.cols.forEach((c) => (lBlank[c] = null)); b.rows.forEach((m) => { if (!matchedB.has(m)) out.push({ ...lBlank, ...bRow(m) }); }); }
      return { cols, rows: out };
    }
    case "union": {
      const b = ctx?.secondary; if (!b || !b.cols.length) return t;
      const cols = Array.from(new Set([...t.cols, ...b.cols]));
      const norm = (r: Rec) => { const o: Rec = {}; cols.forEach((c) => (o[c] = c in r ? r[c] : null)); return o; };
      let out = [...rows.map(norm), ...b.rows.map(norm)];
      if (op.mode === "distinct") { const seen = new Set<string>(); out = out.filter((r) => { const k = cols.map((c) => String(r[c])).join("¦"); if (seen.has(k)) return false; seen.add(k); return true; }); }
      return { cols, rows: out };
    }
    case "lookup": {
        // Lookup is essentially a left join – bring in matching columns without duplicating primary rows.
        const lookupOp: EtlOp = { ...op, type: "join", joinType: "left" };
        return applyOp(t, lookupOp, ctx);
      }
      case "merge": {
        // Merge updates primary rows with fields from secondary based on a key.
        // Perform a left join first.
        const mergeOp: EtlOp = { ...op, type: "join", joinType: "left" };
        const joined = applyOp(t, mergeOp, ctx);
        // Overwrite primary columns with secondary values where they share the same name.
        const bPrefix = "b_";
        const mergedCols = new Set<string>();
        const resultRows = joined.rows.map((r) => {
          const out: Rec = { ...r };
          for (const col of Object.keys(r)) {
            if (col.startsWith(bPrefix)) {
              const primaryName = col.slice(bPrefix.length);
              if (t.cols.includes(primaryName)) {
                const val = r[col];
                if (val !== null && val !== undefined && val !== "") out[primaryName] = val as Cell;
                delete out[col];
                mergedCols.add(col);
              }
            }
          }
          return out;
        });
        return { cols: joined.cols.filter((c) => !mergedCols.has(c)), rows: resultRows };
      }
      case "append": {
        // Append is equivalent to union (stack rows from secondary).
        const appendOp: EtlOp = { ...op, type: "union" };
        return applyOp(t, appendOp, ctx);
      }
    case "pivot": {
      const idxCol = op.groupBy || t.cols[0]; const pivCol = op.col || t.cols[1] || t.cols[0]; const agg = op.agg || "sum"; const ac = op.aggCol || t.cols[t.cols.length - 1];
      const pivVals = Array.from(new Set(rows.map((r) => String(r[pivCol] ?? "∅"))));
      const groups = new Map<string, Rec[]>();
      rows.forEach((r) => { const k = String(r[idxCol] ?? "∅"); if (!groups.has(k)) groups.set(k, []); groups.get(k)!.push(r); });
      const cols = [idxCol, ...pivVals];
      const out: Rec[] = [...groups.entries()].map(([k, rs]) => {
        const o: Rec = { [idxCol]: k };
        pivVals.forEach((pv) => { const nums = rs.filter((r) => String(r[pivCol] ?? "∅") === pv).map((r) => numify(r[ac])).filter((n) => !isNaN(n)); o[pv] = nums.length ? aggNums(nums, agg) : null; });
        return o;
      });
      return { cols, rows: out };
    }
    case "unpivot": {
      const melt = op.cols ? op.cols.filter((c) => t.cols.includes(c)) : [];
      const keep = t.cols.filter((c) => !melt.includes(c));
      const varName = op.name || "variable"; const valName = op.value || "value";
      const cols = [...keep, varName, valName];
      const out: Rec[] = [];
      rows.forEach((r) => melt.forEach((c) => { const o: Rec = {}; keep.forEach((k) => (o[k] = r[k])); o[varName] = c; o[valName] = r[c]; out.push(o); }));
      return { cols, rows: out };
    }
    case "window": {
      const part = op.groupBy && op.groupBy !== "(none)" ? op.groupBy : ""; const col = op.col || t.cols[0]; const fn = op.fn || "running_sum"; const name = op.name || fn;
      const partsMap = new Map<string, number[]>(); // partition key → row indices
      rows.forEach((_, i) => { const k = part ? String(rows[i][part] ?? "∅") : "_"; if (!partsMap.has(k)) partsMap.set(k, []); partsMap.get(k)!.push(i); });
      const outVal = new Array<Cell>(rows.length).fill(null);
      for (const idxs of partsMap.values()) {
        if (fn === "running_sum") { let acc = 0; idxs.forEach((i) => { const n = numify(rows[i][col]); if (!isNaN(n)) acc += n; outVal[i] = Math.round(acc * 1000) / 1000; }); }
        else if (fn === "row_number") { idxs.forEach((i, k) => (outVal[i] = k + 1)); }
        else if (fn === "lag" || fn === "lead") { idxs.forEach((i, k) => { const j = fn === "lag" ? k - 1 : k + 1; outVal[i] = j >= 0 && j < idxs.length ? rows[idxs[j]][col] : null; }); }
        else if (fn === "rank") { const sorted = [...idxs].sort((a, b) => numify(rows[b][col]) - numify(rows[a][col])); let rank = 0, prev: number | null = null; sorted.forEach((i, k) => { const v = numify(rows[i][col]); if (prev === null || v !== prev) { rank = k + 1; prev = v; } outVal[i] = rank; }); }
      }
      return { cols: [...t.cols, name], rows: rows.map((r, i) => ({ ...r, [name]: outVal[i] })) };
    }
    case "regex": {
      const col = op.col || t.cols[0]; const name = op.name || `${col}_match`;
      let re: RegExp | null = null; try { re = new RegExp(op.value || ""); } catch { re = null; }
      return { cols: [...t.cols, name], rows: rows.map((r) => { let v: Cell = null; if (re) { const m = String(r[col] ?? "").match(re); if (m) v = m[1] ?? m[0]; } return { ...r, [name]: v }; }) };
    }
    case "dateparse": {
      const col = op.col || t.cols[0]; const fn = op.fn || "year"; const name = op.name || `${col}_${fn}`;
      const days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
      return { cols: [...t.cols, name], rows: rows.map((r) => {
        let d = new Date(String(r[col] ?? "")); let v: Cell = null;
        if (isNaN(d.getTime())) {
          const m = String(r[col] ?? "").trim().match(/^(\d{2})[-/](\d{2})[-/](\d{4})$/);
          if (m) d = new Date(`${m[3]}-${m[2]}-${m[1]}T00:00:00Z`);
        }
        if (!isNaN(d.getTime())) {
          if (fn === "year") v = d.getUTCFullYear(); else if (fn === "month") v = d.getUTCMonth() + 1; else if (fn === "day") v = d.getUTCDate();
          else if (fn === "weekday") v = days[d.getUTCDay()]; else if (fn === "iso") v = d.toISOString().slice(0, 10);
          else if (fn === "days_since") v = Math.round((Date.parse(new Date().toISOString().slice(0, 10)) - d.getTime()) / 86_400_000);
        }
        return { ...r, [name]: v };
      }) };
    }
  }
}

export function runPipeline(src: Table, ops: EtlOp[], ctx?: OpCtx): { stages: { op: EtlOp | null; table: Table }[]; final: Table } {
  const stages: { op: EtlOp | null; table: Table }[] = [{ op: null, table: src }];
  let cur = src;
  for (const op of ops) { cur = applyOp(cur, op, ctx); stages.push({ op, table: cur }); }
  return { stages, final: cur };
}

// ── Data-quality expectations (Great-Expectations-style) with remediation ──
export type RuleType = "not_null" | "unique" | "in_range" | "regex" | "in_set";
export type RuleAction = "reject" | "drop" | "fix" | "warn";
export interface Expectation { id: string; col: string; type: RuleType; min?: string; max?: string; pattern?: string; set?: string; action?: RuleAction; fix?: string; }
export const RULE_META: Record<RuleType, { label: string; hint: string }> = {
  not_null: { label: "Not null", hint: "every value must be present" },
  unique: { label: "Unique", hint: "no duplicate values in the column" },
  in_range: { label: "In range", hint: "numeric value between min and max (inclusive)" },
  regex: { label: "Matches regex", hint: "text value must match the pattern" },
  in_set: { label: "In set", hint: "value must be one of a comma-separated list" },
};
export const ACTION_META: Record<RuleAction, { label: string; hint: string }> = {
  reject: { label: "Reject", hint: "quarantine the row to the rejects sink with reasons" },
  drop: { label: "Drop", hint: "remove the row entirely" },
  fix: { label: "Fix", hint: "auto-repair the value (fill / clamp / default; dedupe for unique)" },
  warn: { label: "Warn only", hint: "keep the row, just count the violation" },
};
export function ruleDesc(r: Expectation): string {
  if (r.type === "in_range") return `${r.col} in [${r.min ?? "-∞"}, ${r.max ?? "∞"}]`;
  if (r.type === "regex") return `${r.col} ~ /${r.pattern ?? ""}/`;
  if (r.type === "in_set") return `${r.col} ∈ {${r.set ?? ""}}`;
  return `${r.col} — ${RULE_META[r.type].label}`;
}
// Non-unique pass check (unique is stateful → handled in the loop for keep-first).
function passesSingle(v: Cell, r: Expectation): boolean {
  switch (r.type) {
    case "not_null": return v != null && v !== "";
    case "in_range": { const n = numify(v); if (isNaN(n)) return false; if (r.min != null && r.min !== "" && n < Number(r.min)) return false; if (r.max != null && r.max !== "" && n > Number(r.max)) return false; return true; }
    case "regex": { try { return new RegExp(r.pattern || "").test(String(v ?? "")); } catch { return true; } }
    case "in_set": { const set = (r.set || "").split(",").map((s) => s.trim()); return set.includes(String(v)); }
    default: return true;
  }
}
function fixCell(r: Expectation, cur: Cell): Cell {
  const raw = r.fix ?? "";
  const coerced: Cell = raw !== "" && !isNaN(Number(raw)) ? Number(raw) : raw;
  switch (r.type) {
    case "not_null": return raw === "" ? 0 : coerced;
    case "in_range": { let n = numify(cur); if (isNaN(n)) return r.min != null && r.min !== "" ? Number(r.min) : (coerced || 0); if (r.min != null && r.min !== "" && n < Number(r.min)) n = Number(r.min); if (r.max != null && r.max !== "" && n > Number(r.max)) n = Number(r.max); return n; }
    case "in_set": { if (raw !== "") return coerced; const first = (r.set || "").split(",").map((s) => s.trim()).filter(Boolean)[0]; return first ?? cur; }
    case "regex": return raw === "" ? "" : coerced;
    default: return cur;
  }
}
export function evaluate(t: Table, rules: Expectation[]): {
  report: { id: string; desc: string; ok: boolean; fails: number; action: RuleAction; fixed: number }[];
  clean: Table; rejects: { row: Rec; reasons: string[] }[];
  dropped: number; warned: number; fixedCells: number;
} {
  const fails = new Map<string, number>(rules.map((r) => [r.id, 0]));
  const fixedBy = new Map<string, number>(rules.map((r) => [r.id, 0]));
  const seenByRule = new Map<string, Set<string>>(); // unique → values seen (keep-first)
  const clean: Rec[] = []; const rejects: { row: Rec; reasons: string[] }[] = [];
  let dropped = 0, warned = 0, fixedCells = 0;
  for (const row of t.rows) {
    const work: Rec = { ...row };
    let reject = false, drop = false; const reasons: string[] = [];
    for (const r of rules) {
      const cur = work[r.col];
      let pass: boolean;
      if (r.type === "unique") { const s = seenByRule.get(r.id) || new Set<string>(); const k = String(cur); pass = !s.has(k); if (pass) s.add(k); seenByRule.set(r.id, s); }
      else pass = passesSingle(cur, r);
      if (pass) continue;
      fails.set(r.id, (fails.get(r.id) || 0) + 1);
      const action = r.action || "reject";
      if (action === "warn") warned++;
      else if (action === "drop") drop = true;
      else if (action === "fix") { if (r.type === "unique") { drop = true; } else { work[r.col] = fixCell(r, cur); fixedCells++; fixedBy.set(r.id, (fixedBy.get(r.id) || 0) + 1); } }
      else { reject = true; reasons.push(ruleDesc(r)); }
    }
    if (reject) rejects.push({ row: work, reasons });
    else if (drop) dropped++;
    else clean.push(work);
  }
  const report = rules.map((r) => ({ id: r.id, desc: ruleDesc(r), ok: (fails.get(r.id) || 0) === 0, fails: fails.get(r.id) || 0, action: r.action || "reject", fixed: fixedBy.get(r.id) || 0 }));
  return { report, clean: { cols: t.cols, rows: clean }, rejects, dropped, warned, fixedCells };
}

// ── Column-level lineage: which source columns feed each output column ──
export function lineage(srcCols: string[], ops: EtlOp[], secondaryCols: string[] = []): { col: string; from: string[] }[] {
  let map = new Map<string, Set<string>>(srcCols.map((c) => [c, new Set([c])]));
  const src = (k?: string): Set<string> => (k && map.has(k) ? map.get(k)! : new Set(k && !isNaN(Number(k)) ? [] : (k ? [k] : [])));
  const union = (...sets: Set<string>[]) => { const s = new Set<string>(); sets.forEach((x) => x.forEach((v) => s.add(v))); return s; };
  for (const o of ops) {
    switch (o.type) {
      case "select": { const keep = (o.cols && o.cols.length ? o.cols : [...map.keys()]); map = new Map(keep.filter((c) => map.has(c)).map((c) => [c, map.get(c)!])); break; }
      case "rename": { const from = o.col || ""; const to = o.name || from; if (map.has(from)) { const nm = new Map<string, Set<string>>(); map.forEach((v, k) => nm.set(k === from ? to : k, v)); map = nm; } break; }
      case "derive": map.set(o.name || "derived", union(src(o.left), src(o.right))); break;
      case "bucket": map.set(o.name || `${o.col}_bin`, src(o.col)); break;
      case "regex": map.set(o.name || `${o.col}_match`, src(o.col)); break;
      case "dateparse": map.set(o.name || `${o.col}_${o.fn}`, src(o.col)); break;
      case "window": map.set(o.name || (o.fn || "window"), union(src(o.col), o.groupBy ? src(o.groupBy) : new Set())); break;
      case "aggregate": { const outCol = o.agg === "count" ? "count" : `${o.agg}_${o.aggCol}`; map = new Map([[o.groupBy || "", src(o.groupBy)], [outCol, o.agg === "count" ? new Set([...map.keys()].flatMap((k) => [...(map.get(k) || [])])) : src(o.aggCol)]]); break; }
      case "pivot": { const nm = new Map<string, Set<string>>(); nm.set(o.groupBy || "", src(o.groupBy)); nm.set("(pivoted)", union(src(o.col), src(o.aggCol))); map = nm; break; }
      case "unpivot": { const melt = (o.cols && o.cols.length ? o.cols : [...map.keys()]); const keep = [...map.keys()].filter((c) => !melt.includes(c)); const nm = new Map<string, Set<string>>(); keep.forEach((k) => nm.set(k, map.get(k) || new Set([k]))); nm.set(o.name || "variable", union(...melt.map((c) => src(c)))); nm.set(o.value || "value", union(...melt.map((c) => src(c)))); map = nm; break; }
      case "join": { secondaryCols.forEach((c) => { const key = srcCols.includes(c) ? `b_${c}` : c; if (!map.has(key)) map.set(key, new Set([`B.${c}`])); }); break; }
      case "union": secondaryCols.forEach((c) => { if (!map.has(c)) map.set(c, new Set([`B.${c}`])); }); break;
      default: break; // filter/sort/dedupe/clean/limit/sample/map/fillna keep columns
    }
  }
  return [...map.entries()].map(([col, from]) => ({ col, from: [...from] }));
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

// Rich per-column profile: type, nulls, distinct, numeric stats, top values, histogram.
export interface ColProfile { name: string; type: string; nulls: number; distinct: number; min?: number; max?: number; mean?: number; top: { v: string; count: number }[]; hist?: number[]; }
export function profile(t: Table): ColProfile[] {
  return t.cols.map((c) => {
    const vals = t.rows.map((r) => r[c]);
    const nn = vals.filter((v) => v != null && v !== "");
    const numeric = nn.length > 0 && nn.every((v) => typeof v === "number");
    const counts = new Map<string, number>();
    nn.forEach((v) => { const k = String(v); counts.set(k, (counts.get(k) || 0) + 1); });
    const top = [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 5).map(([v, count]) => ({ v, count }));
    const base: ColProfile = { name: c, type: numeric ? "num" : "text", nulls: vals.length - nn.length, distinct: counts.size, top };
    if (numeric) {
      const nums = nn as number[];
      const mn = Math.min(...nums), mx = Math.max(...nums);
      base.min = mn; base.max = mx; base.mean = Math.round((nums.reduce((a, b) => a + b, 0) / nums.length) * 1000) / 1000;
      const B = 10, sp = (mx - mn) || 1; const hist = new Array(B).fill(0);
      nums.forEach((v) => { let b = Math.floor(((v - mn) / sp) * B); if (b >= B) b = B - 1; if (b < 0) b = 0; hist[b]++; });
      base.hist = hist;
    }
    return base;
  });
}

export function toCSV(t: Table): string {
  const esc = (v: Cell) => { const s = v == null ? "" : String(v); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
  return [t.cols.join(","), ...t.rows.map((r) => t.cols.map((c) => esc(r[c])).join(","))].join("\n");
}
export function toJSON(t: Table): string {
  return JSON.stringify(t.rows, null, 2);
}
