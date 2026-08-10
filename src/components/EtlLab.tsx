"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  parseRecords, sampleSources, runPipeline, toCSV, toJSON, tableFromRecords, profile, evaluate, lineage,
  OP_META, RULE_META, ACTION_META, ruleDesc,
  type Table, type EtlOp, type OpType, type Expectation, type RuleType, type RuleAction,
} from "@/lib/etlUtils";
import { toast, confirmDialog } from "@/lib/toast";
import { runSql } from "@/lib/sqlEngine";

type Step = "extract" | "transform" | "load";
type PipeMode = "etl" | "elt";
type LoadTab = "run" | "quality" | "deliver" | "schedule" | "lineage";
const rid = () => Math.random().toString(36).slice(2, 9);

function opSummary(o: EtlOp): string {
  switch (o.type) {
    case "filter": return `${o.col} ${o.op} ${o.value ?? ""}`;
    case "select": return (o.cols || []).join(", ").slice(0, 22) || "all";
    case "derive": return `${o.name} = ${o.left} ${o.arith} ${o.right}`;
    case "aggregate": return `${o.agg} by ${o.groupBy}`;
    case "sort": return `${o.col} ${o.dir}`;
    case "dedupe": return (o.cols && o.cols.length ? o.cols.join(",") : "all cols").slice(0, 22);
    case "clean": return o.mode === "fill0" ? "fill nulls → 0" : "drop nulls";
    case "rename": return `${o.col} → ${o.name}`;
    case "limit": return `first ${o.value}`;
    case "sample": return `~${o.value}`;
    case "map": return `${o.fn}(${o.col})`;
    case "fillna": return `${o.col || "all"} ← ${o.value}`;
    case "bucket": return `${o.col} → ${o.value} bins`;
    case "join": return `${o.joinType} on ${o.col}=${o.rightKey}`;
    case "union": return `+ Source B (${o.mode})`;
    case "pivot": return `${o.col} → columns`;
    case "unpivot": return `melt ${(o.cols || []).length || "cols"}`;
    case "window": return `${o.fn}(${o.col})`;
    case "regex": return `/${o.value}/ → ${o.name}`;
    case "dateparse": return `${o.fn}(${o.col})`;
    default: return "";
  }
}

function download(text: string, filename: string, mime: string) {
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([text], { type: mime })); a.download = filename; a.click(); URL.revokeObjectURL(a.href);
}
function downloadBlob(blob: Blob, filename: string) {
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = filename; a.click(); URL.revokeObjectURL(a.href);
}
// Referenced table names (raw / b / other models) in a SQL string — for the DAG.
function refsOf(sql: string, known: string[]): string[] {
  const found = new Set<string>();
  const re = /\b(?:from|join)\s+([A-Za-z_][A-Za-z0-9_]*)/gi;
  let m: RegExpExecArray | null;
  while ((m = re.exec(sql))) { if (known.includes(m[1])) found.add(m[1]); }
  return [...found];
}
const EXT_TYPES = ["auto", "TEXT", "BIGINT", "DOUBLE", "VARCHAR(255)", "DATE", "DATETIME", "BOOLEAN"];

import DataVizExplorer from "./visualization/DataVizExplorer";
export default function EtlLab() {
  const [step, setStep] = useState<Step>("extract");
  const [pipeMode, setPipeMode] = useState<PipeMode>("etl");
  const [srcTable, setSrcTable] = useState<Table | null>(null);
  const [srcName, setSrcName] = useState("");
  const [srcTab, setSrcTab] = useState<"sample" | "upload" | "json" | "db">("sample");
  const [sampleKey, setSampleKey] = useState("orders");
  const [jsonText, setJsonText] = useState('[\n  {"id": 1, "region": "US", "amount": 120.5, "status": "paid"},\n  {"id": 2, "region": "EU", "amount": 0, "status": "failed"}\n]');
  const [dbUrl, setDbUrl] = useState("mysql://user:pass@host:4000/db");
  const [dbQuery, setDbQuery] = useState("SELECT * FROM orders LIMIT 500;");
  const [busy, setBusy] = useState(false);
  const [viewMode, setViewMode] = useState<"head" | "tail">("head");
  const [profOpen, setProfOpen] = useState(false);

  // Source B (for join / union)
  const [srcTableB, setSrcTableB] = useState<Table | null>(null);
  const [srcNameB, setSrcNameB] = useState("");
  const [sampleKeyB, setSampleKeyB] = useState("customers");

  const [ops, setOps] = useState<EtlOp[]>([]);
  const [sel, setSel] = useState<string | null>(null);
  const [mode, setMode] = useState<"batch" | "stream">("batch");
  const [running, setRunning] = useState(false);
  const [nodeStatus, setNodeStatus] = useState<Record<string, string>>({});
  const [metrics, setMetrics] = useState<{ records: number; throughput: number; batches: number; out: number } | null>(null);
  const [addOpen, setAddOpen] = useState(false);
  const [showCode, setShowCode] = useState(false);
  const [copied, setCopied] = useState(false);
  const [savedMsg, setSavedMsg] = useState("");
  const [msg, setMsg] = useState("");

  // Load sub-tabs + features
  const [loadTab, setLoadTab] = useState<LoadTab>("run");
  const [rules, setRules] = useState<Expectation[]>([]);
  const [stored, setStored] = useState<{ id: string; name: string; rowCount: number; createdAt?: string }[]>([]);
  const [storeBusy, setStoreBusy] = useState(false);
  const [storeName, setStoreName] = useState("");
  const [schedEvery, setSchedEvery] = useState(5);
  const [schedOn, setSchedOn] = useState(false);
  const [schedRuns, setSchedRuns] = useState<{ t: string; inn: number; out: number; rej: number }[]>([]);

  // store target (both ETL + ELT) + ELT SQL transform
  const [storeTarget, setStoreTarget] = useState<"platform" | "external">("platform");
  const [extUrl, setExtUrl] = useState("mysql://user:pass@host:4000/db");
  const [extTable, setExtTable] = useState("etl_output");
  const [sqlText, setSqlText] = useState("SELECT region,\n       COUNT(*) AS orders,\n       ROUND(SUM(amount), 2) AS revenue\nFROM raw\nWHERE status = 'paid'\nGROUP BY region\nORDER BY revenue DESC");
  const [sqlResult, setSqlResult] = useState<Table | null>(null);
  const [sqlErr, setSqlErr] = useState("");
  const [sqlBusy, setSqlBusy] = useState(false);
  const [storeMode, setStoreMode] = useState<"new" | "replace">("new");
  const [extMode, setExtMode] = useState<"append" | "upsert">("append");
  const [extKey, setExtKey] = useState("");
  const [extConn, setExtConn] = useState<"url" | "fields">("url");
  const [extHost, setExtHost] = useState("");
  const [extPort, setExtPort] = useState("4000");
  const [extUser, setExtUser] = useState("");
  const [extPass, setExtPass] = useState("");
  const [extDb, setExtDb] = useState("");
  const [extTypes, setExtTypes] = useState<Record<string, string>>({});
  const [extTypesOpen, setExtTypesOpen] = useState(false);
  const [models, setModels] = useState<{ name: string; sql: string }[]>([]);
  const [modelName, setModelName] = useState("");

  const fileRef = useRef<HTMLInputElement>(null);
  const timers = useRef<ReturnType<typeof setTimeout>[]>([]);
  const schedRef = useRef<ReturnType<typeof setInterval> | null>(null);

  function setSource(t: Table, name: string) {
    if (!t.cols.length || !t.rows.length) { setMsg("The source has no usable rows / columns."); return; }
    setSrcTable(t); setSrcName(name); setMsg(""); setOps([]); setSel(null); setNodeStatus({}); setMetrics(null); setRules([]);
  }
  function loadSample() { const s = sampleSources().find((x) => x.key === sampleKey)!; setSource(parseRecords(s.csv), s.label); }
  function loadSampleB() { const s = sampleSources().find((x) => x.key === sampleKeyB)!; setSrcTableB(parseRecords(s.csv)); setSrcNameB(s.label); }
  useEffect(() => { const s = sampleSources().find((x) => x.key === "orders")!; setSource(parseRecords(s.csv), s.label); const t = timers; return () => t.current.forEach(clearTimeout); }, []);
  useEffect(() => () => { if (schedRef.current) clearInterval(schedRef.current); }, []);
  useEffect(() => {
    const id = new URLSearchParams(window.location.search).get("project");
    if (!id) return;
    fetch(`/api/projects?id=${id}`).then((r) => r.json()).then(({ project }) => {
      const c = project?.config; if (!c) return;
      const s = sampleSources().find((x) => x.label === c.srcName || x.key === c.srcName);
      if (s) setSource(parseRecords(s.csv), s.label);
      else setMsg(`Loaded pipeline "${project.name}". Its source "${c.srcName}" was custom data — re-load it in Extract to run.`);
      if (c.mode) setMode(c.mode);
      if (Array.isArray(c.ops)) setOps(c.ops.map((o: EtlOp) => ({ ...o, id: rid() })));
      if (Array.isArray(c.rules)) setRules(c.rules.map((r: Expectation) => ({ ...r, id: rid() })));
      if (Array.isArray(c.models)) setModels(c.models);
      if (c.pipeMode === "elt" || c.pipeMode === "etl") setPipeMode(c.pipeMode);
    }).catch(() => {});
  }, []);
  function loadJson(text: string, name: string) {
    try { const data = JSON.parse(text); const arr = Array.isArray(data) ? data : Array.isArray(data?.data) ? data.data : Array.isArray(data?.rows) ? data.rows : null; if (!arr) throw new Error("expected an array of objects, or {data:[…]}"); const t = tableFromRecords(arr); if (!t.rows.length) throw new Error("no object rows found"); setSource(t, name); }
    catch (e) { setMsg("JSON error: " + (e as Error).message); }
  }
  async function onFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0]; if (!f) return; const ext = (f.name.split(".").pop() || "").toLowerCase();
    setBusy(true); setMsg("");
    try {
      if (["xlsx", "xls", "xlsm"].includes(ext)) { const fd = new FormData(); fd.append("file", f); const r = await fetch("/api/rag/extract", { method: "POST", body: fd }); const j = await r.json(); if (!r.ok) throw new Error(j.error || "parse failed"); setSource(parseRecords(j.text), f.name); }
      else if (ext === "json") loadJson(await f.text(), f.name);
      else setSource(parseRecords(await f.text()), f.name);
    } catch (err) { setMsg((err as Error).message); }
    setBusy(false); e.target.value = "";
  }
  async function runDb() {
    setBusy(true); setMsg("");
    try { const r = await fetch("/api/ml/query", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ url: dbUrl, query: dbQuery }) }); const j = await r.json(); if (!r.ok) throw new Error(j.error || "query failed"); setSource(parseRecords(j.csv), "db query"); }
    catch (e) { setMsg((e as Error).message); } finally { setBusy(false); }
  }

  const pipe = useMemo(() => (srcTable ? runPipeline(srcTable, ops, { secondary: srcTableB }) : null), [srcTable, ops, srcTableB]);
  const prof = useMemo(() => (srcTable ? profile(srcTable) : []), [srcTable]);
  const qc = useMemo(() => (pipe ? evaluate(pipe.final, rules) : null), [pipe, rules]);
  const lin = useMemo(() => (srcTable ? lineage(srcTable.cols, ops, srcTableB?.cols || []) : []), [srcTable, ops, srcTableB]);
  const deliver = useMemo(() => (pipe ? (rules.length ? qc!.clean : pipe.final) : null), [pipe, qc, rules]);
  const selIdx = ops.findIndex((o) => o.id === sel);
  const inCols = srcTable ? (selIdx >= 0 && pipe ? pipe.stages[selIdx].table.cols : srcTable.cols) : [];
  const bCols = srcTableB?.cols || [];

  function addOp(type: OpType) {
    if (!srcTable) return;
    const cols = srcTable.cols; const numCol = cols[cols.findIndex((c) => srcTable.rows.some((r) => typeof r[c] === "number"))] || cols[0];
    const base: EtlOp = { id: rid(), type };
    if (type === "filter") Object.assign(base, { col: numCol, op: ">", value: "0" });
    else if (type === "select") Object.assign(base, { cols: cols.slice(0, Math.min(3, cols.length)) });
    else if (type === "derive") Object.assign(base, { name: "derived", left: numCol, arith: "*", right: "1" });
    else if (type === "aggregate") Object.assign(base, { groupBy: cols[0], agg: "count", aggCol: numCol });
    else if (type === "sort") Object.assign(base, { col: numCol, dir: "desc" });
    else if (type === "dedupe") Object.assign(base, { cols: [] });
    else if (type === "clean") Object.assign(base, { mode: "dropnull" });
    else if (type === "rename") Object.assign(base, { col: cols[0], name: cols[0] + "_new" });
    else if (type === "limit") Object.assign(base, { value: "10" });
    else if (type === "sample") Object.assign(base, { value: "0.5" });
    else if (type === "map") Object.assign(base, { col: numCol, fn: "round" });
    else if (type === "fillna") Object.assign(base, { col: "", value: "0" });
    else if (type === "bucket") Object.assign(base, { col: numCol, name: numCol + "_bin", value: "4" });
    else if (type === "join") Object.assign(base, { joinType: "inner", col: cols[0], rightKey: bCols[0] || cols[0] });
    else if (type === "union") Object.assign(base, { mode: "all" });
    else if (type === "pivot") Object.assign(base, { groupBy: cols[0], col: cols[1] || cols[0], agg: "sum", aggCol: numCol });
    else if (type === "unpivot") Object.assign(base, { cols: [numCol], name: "variable", value: "value" });
    else if (type === "window") Object.assign(base, { groupBy: "(none)", col: numCol, fn: "running_sum", name: "running_sum" });
    else if (type === "regex") Object.assign(base, { col: cols[0], value: "(\\d+)", name: cols[0] + "_match" });
    else if (type === "dateparse") Object.assign(base, { col: cols[0], fn: "year", name: cols[0] + "_year" });
    setOps((o) => [...o, base]); setSel(base.id); setAddOpen(false);
  }
  const patch = (p: Partial<EtlOp>) => setOps((os) => os.map((o) => (o.id === sel ? { ...o, ...p } : o)));
  const removeOp = (id: string) => { setOps((os) => os.filter((o) => o.id !== id)); if (sel === id) setSel(null); };

  function run() {
    if (!srcTable || !pipe) return;
    timers.current.forEach(clearTimeout); timers.current = [];
    setRunning(true); setNodeStatus({}); setMetrics(null);
    const ids = ["src", ...ops.map((o) => o.id), "sink"]; const t0 = performance.now();
    if (mode === "batch") {
      let i = 0;
      const tick = () => {
        if (i > 0) setNodeStatus((s) => ({ ...s, [ids[i - 1]]: "done" }));
        if (i < ids.length) { setNodeStatus((s) => ({ ...s, [ids[i]]: "running" })); i++; timers.current.push(setTimeout(tick, 460)); }
        else { setNodeStatus(Object.fromEntries(ids.map((id) => [id, "done"]))); const ms = Math.max(1, performance.now() - t0); setMetrics({ records: srcTable.rows.length, throughput: Math.round(srcTable.rows.length / (ms / 1000)), batches: 1, out: pipe.final.rows.length }); setRunning(false); }
      };
      tick();
    } else {
      const total = srcTable.rows.length, B = Math.max(1, Math.ceil(total / 8));
      setNodeStatus(Object.fromEntries(ids.map((id) => [id, "running"])));
      let done = 0, b = 0;
      const tick = () => {
        done = Math.min(total, done + B); b++; const ms = Math.max(1, performance.now() - t0);
        setMetrics({ records: done, throughput: Math.round(done / (ms / 1000)), batches: b, out: pipe.final.rows.length });
        if (done < total) timers.current.push(setTimeout(tick, 400));
        else { setNodeStatus(Object.fromEntries(ids.map((id) => [id, "done"]))); setRunning(false); }
      };
      timers.current.push(setTimeout(tick, 300));
    }
  }

  // Schedule simulation — logs a run entry every N seconds.
  function toggleSchedule() {
    if (schedOn) { if (schedRef.current) clearInterval(schedRef.current); schedRef.current = null; setSchedOn(false); return; }
    setSchedOn(true);
    const fire = () => setSchedRuns((rs) => [{ t: new Date().toLocaleTimeString(), inn: srcTable?.rows.length || 0, out: deliver?.rows.length || 0, rej: qc?.rejects.length || 0 }, ...rs].slice(0, 12));
    fire();
    schedRef.current = setInterval(fire, Math.max(2, schedEvery) * 1000);
  }

  async function loadStored() { try { const j = await fetch("/api/etl/store").then((r) => r.json()); setStored(j.datasets || []); } catch { setStored([]); } }
  useEffect(() => { if (step === "load" && loadTab === "deliver") loadStored(); }, [step, loadTab]);

  async function storePlatform(t: Table) {
    setStoreBusy(true);
    try {
      const r = await fetch("/api/etl/store", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ name: storeName || srcName || "ETL output", cols: t.cols, rows: t.rows, mode: storeMode }) });
      const j = await r.json().catch(() => null);
      if (r.ok) { toast(`${storeMode === "replace" ? "Replaced" : "Stored"} ${j.rowCount} rows in the platform database`, "success"); setStoreName(""); loadStored(); }
      else toast(j?.error || "Store failed", "error");
    } catch (e) { toast((e as Error).message, "error"); }
    setStoreBusy(false);
  }
  async function storeExternal(t: Table) {
    const url = extConn === "fields"
      ? `mysql://${encodeURIComponent(extUser)}:${encodeURIComponent(extPass)}@${extHost.trim()}:${extPort.trim() || "3306"}/${extDb.trim()}`
      : extUrl;
    setStoreBusy(true);
    try {
      const r = await fetch("/api/etl/store-external", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ url, table: extTable, cols: t.cols, rows: t.rows, mode: extMode, keyCol: extKey || t.cols[0], types: extTypes }) });
      const j = await r.json().catch(() => null);
      if (r.ok) toast(`${extMode === "upsert" ? "Upserted" : "Loaded"} ${j.rowCount} rows into \`${j.table}\` on your database`, "success");
      else toast(j?.error || "Load failed", "error");
    } catch (e) { toast((e as Error).message, "error"); }
    setStoreBusy(false);
  }
  // Compute all saved ELT models in order (each can query the earlier ones).
  async function computeModels(): Promise<{ name: string; table: Table }[]> {
    if (!srcTable) return [];
    const acc: { name: string; table: Table }[] = [];
    for (const m of models) { acc.push({ name: m.name, table: await runSql(m.sql, srcTable, srcTableB, acc) }); }
    return acc;
  }
  async function runSqlNow() {
    if (!srcTable) return;
    setSqlBusy(true); setSqlErr("");
    try { const ms = await computeModels(); setSqlResult(await runSql(sqlText, srcTable, srcTableB, ms)); }
    catch (e) { setSqlErr((e as Error).message); setSqlResult(null); }
    setSqlBusy(false);
  }
  function saveModel() {
    const nm = modelName.trim();
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(nm)) { toast("Model name must be a valid identifier (letters/digits/underscore).", "error"); return; }
    if (nm === "raw" || nm === "b") { toast("“raw” and “b” are reserved.", "error"); return; }
    setModels((ms) => [...ms.filter((x) => x.name !== nm), { name: nm, sql: sqlText }]);
    setModelName(""); toast(`Saved model “${nm}”`, "success");
  }
  const removeModel = (i: number) => setModels((ms) => ms.filter((_, j) => j !== i));
  async function downloadParquet(t: Table) {
    try {
      const res = await fetch("/api/etl/parquet", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ cols: t.cols, rows: t.rows }) });
      if (!res.ok) { const j = await res.json().catch(() => null); toast(j?.error || "Parquet export failed", "error"); return; }
      downloadBlob(await res.blob(), "etl_output.parquet");
    } catch (e) { toast((e as Error).message, "error"); }
  }

  // Layered dependency graph of the ELT models: sources (raw/b) → models → query.
  function modelDag() {
    const known = ["raw", ...(srcTableB ? ["b"] : []), ...models.map((m) => m.name)];
    const depth: Record<string, number> = { raw: 0 };
    if (srcTableB) depth.b = 0;
    const edges: { from: string; to: string }[] = [];
    models.forEach((m) => {
      const refs = refsOf(m.sql, known).filter((n) => n !== m.name);
      depth[m.name] = refs.length ? 1 + Math.max(...refs.map((r) => depth[r] ?? 0)) : 1;
      refs.forEach((r) => edges.push({ from: r, to: m.name }));
    });
    depth.__out = 1 + Math.max(0, ...models.map((m) => depth[m.name] ?? 1));
    refsOf(sqlText, known).forEach((r) => edges.push({ from: r, to: "__out" }));
    const nodes = [
      { id: "raw", label: "raw", cls: "type-trigger" },
      ...(srcTableB ? [{ id: "b", label: "b", cls: "type-trigger" }] : []),
      ...models.map((m) => ({ id: m.name, label: m.name, cls: "type-tool" })),
      { id: "__out", label: "query ▶", cls: "type-output" },
    ];
    const byDepth: Record<number, string[]> = {};
    nodes.forEach((n) => { const d = depth[n.id] ?? 1; (byDepth[d] = byDepth[d] || []).push(n.id); });
    const W = 104, H = 38, GX = 64, GY = 14;
    const pos: Record<string, { x: number; y: number }> = {};
    Object.entries(byDepth).forEach(([d, ids]) => ids.forEach((id, i) => { pos[id] = { x: 12 + Number(d) * (W + GX), y: 10 + i * (H + GY) }; }));
    const maxD = Math.max(...Object.keys(byDepth).map(Number));
    const maxRows = Math.max(...Object.values(byDepth).map((a) => a.length));
    const cw = 24 + (maxD + 1) * (W + GX), chh = 20 + maxRows * (H + GY);
    const label: Record<string, string> = Object.fromEntries(nodes.map((n) => [n.id, n.label]));
    return (
      <div style={{ overflowX: "auto", marginTop: 10 }}>
        <div style={{ position: "relative", width: cw, height: chh }}>
          <svg className="wires2" width={cw} height={chh} style={{ position: "absolute", inset: 0 }}>
            {edges.map((e, i) => { const a = pos[e.from], b2 = pos[e.to]; if (!a || !b2) return null; const x1 = a.x + W, y1 = a.y + H / 2, x2 = b2.x, y2 = b2.y + H / 2; const dx = GX / 2; return <path key={i} className="active" d={`M${x1} ${y1} C${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`} />; })}
          </svg>
          {nodes.map((n) => { const p = pos[n.id]; return <div key={n.id} className={`anode ${n.cls}`} style={{ position: "absolute", left: p.x, top: p.y, width: W, height: H }}><div className="ah" style={{ padding: "0 9px", height: "100%" }}><div className="atitle" style={{ fontSize: 12, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{label[n.id]}</div></div></div>; })}
        </div>
      </div>
    );
  }
  async function downloadStored(id: string, name: string) {
    const j = await fetch(`/api/etl/store?id=${id}`).then((r) => r.json()).catch(() => null);
    if (!j?.dataset) { toast("Could not load dataset", "error"); return; }
    download(toCSV({ cols: j.dataset.cols || [], rows: j.rows || [] }), `${name.replace(/\s+/g, "_")}.csv`, "text/csv");
  }
  async function deleteStored(id: string) {
    if (!(await confirmDialog("Delete this stored dataset?", { danger: true, confirmLabel: "Delete" }))) return;
    const r = await fetch(`/api/etl/store?id=${id}`, { method: "DELETE" });
    toast(r.ok ? "Deleted" : "Delete failed", r.ok ? "success" : "error"); loadStored();
  }

  // Expectation rule helpers
  const addRule = () => setRules((rs) => [...rs, { id: rid(), col: (srcTable?.cols[0] || ""), type: "not_null" }]);
  const patchRule = (id: string, p: Partial<Expectation>) => setRules((rs) => rs.map((r) => (r.id === id ? { ...r, ...p } : r)));
  const removeRule = (id: string) => setRules((rs) => rs.filter((r) => r.id !== id));

  function buildCode(): string {
    const hasB = ops.some((o) => o.type === "join" || o.type === "union");
    const hasWin = ops.some((o) => o.type === "window");
    const lines = ops.map((o) => {
      switch (o.type) {
        case "filter": { const q = /^-?\d+(\.\d+)?$/.test(o.value ?? "") ? o.value : `"${o.value}"`; if (o.op === "contains") return `df = df.filter(F.col("${o.col}").contains("${o.value}"))`; return `df = df.filter(F.col("${o.col}") ${o.op} ${q})`; }
        case "select": return `df = df.select(${(o.cols || []).map((c) => `"${c}"`).join(", ")})`;
        case "derive": { const term = (k?: string) => (srcTable?.cols.includes(k || "") ? `F.col("${k}")` : `F.lit(${k})`); return `df = df.withColumn("${o.name}", ${term(o.left)} ${o.arith} ${term(o.right)})`; }
        case "aggregate": return o.agg === "count" ? `df = df.groupBy("${o.groupBy}").count()` : `df = df.groupBy("${o.groupBy}").agg(F.${o.agg}("${o.aggCol}").alias("${o.agg}_${o.aggCol}"))`;
        case "sort": return `df = df.orderBy(F.col("${o.col}").${o.dir === "desc" ? "desc" : "asc"}())`;
        case "dedupe": return `df = df.dropDuplicates(${o.cols && o.cols.length ? `[${o.cols.map((c) => `"${c}"`).join(", ")}]` : ""})`;
        case "clean": return o.mode === "fill0" ? `df = df.fillna(0)` : `df = df.dropna()`;
        case "rename": return `df = df.withColumnRenamed("${o.col}", "${o.name}")`;
        case "limit": return `df = df.limit(${parseInt(o.value || "10") || 10})`;
        case "sample": return `df = df.sample(${Number(o.value || "0.5") <= 1 ? o.value : (Number(o.value) / (srcTable?.rows.length || 1)).toFixed(3)})`;
        case "map": { const fmap: Record<string, string> = { upper: "upper", lower: "lower", trim: "trim", length: "length", round: "round", abs: "abs", floor: "floor", ceil: "ceil" }; return `df = df.withColumn("${o.col}", F.${fmap[o.fn || "round"]}(F.col("${o.col}")))`; }
        case "fillna": return `df = df.fillna(${/^-?\d+(\.\d+)?$/.test(o.value ?? "") ? o.value : `"${o.value}"`}${o.col ? `, subset=["${o.col}"]` : ""})`;
        case "bucket": return `df = df.withColumn("${o.name}", (((F.col("${o.col}") - F.min("${o.col}").over(W)) / (F.max("${o.col}").over(W) - F.min("${o.col}").over(W))) * ${parseInt(o.value || "4") || 4}).cast("int"))`;
        case "join": return `df = df.join(dfB, df["${o.col}"] == dfB["${o.rightKey}"], "${o.joinType}")`;
        case "union": return `df = df.unionByName(dfB, allowMissingColumns=True)${o.mode === "distinct" ? ".distinct()" : ""}`;
        case "pivot": return `df = df.groupBy("${o.groupBy}").pivot("${o.col}").agg(F.${o.agg}("${o.aggCol}"))`;
        case "unpivot": return `df = df.unpivot([${srcTable!.cols.filter((c) => !(o.cols || []).includes(c)).map((c) => `"${c}"`).join(", ")}], [${(o.cols || []).map((c) => `"${c}"`).join(", ")}], "${o.name}", "${o.value}")`;
        case "window": { const w = o.groupBy && o.groupBy !== "(none)" ? `Window.partitionBy("${o.groupBy}").orderBy("${o.col}")` : `Window.orderBy("${o.col}")`; const fmap: Record<string, string> = { running_sum: `F.sum("${o.col}")`, row_number: "F.row_number()", rank: "F.rank()", lag: `F.lag("${o.col}")`, lead: `F.lead("${o.col}")` }; return `df = df.withColumn("${o.name}", ${fmap[o.fn || "running_sum"]}.over(${w}))`; }
        case "regex": return `df = df.withColumn("${o.name}", F.regexp_extract(F.col("${o.col}"), r"${o.value}", 1))`;
        case "dateparse": { const fmap: Record<string, string> = { year: "F.year", month: "F.month", day: "F.dayofmonth", weekday: "F.date_format", iso: "F.to_date", days_since: "F.datediff" }; if (o.fn === "days_since") return `df = df.withColumn("${o.name}", F.datediff(F.current_date(), F.to_date(F.col("${o.col}"))))`; if (o.fn === "weekday") return `df = df.withColumn("${o.name}", F.date_format(F.col("${o.col}"), "EEEE"))`; if (o.fn === "iso") return `df = df.withColumn("${o.name}", F.to_date(F.col("${o.col}")))`; return `df = df.withColumn("${o.name}", ${fmap[o.fn || "year"]}(F.to_date(F.col("${o.col}"))))`; }
        default: return "";
      }
    });
    return `# AI Workbench · ETL pipeline (PySpark)  ·  ${mode === "stream" ? "structured streaming" : "batch"}
from pyspark.sql import SparkSession, functions as F${hasWin ? "\nfrom pyspark.sql.window import Window" : ""}
spark = SparkSession.builder.appName("workbench_etl").getOrCreate()

${mode === "stream"
        ? `df = (spark.readStream.format("kafka").option("subscribe", "orders").load())  # extract`
        : `df = spark.read.option("header", True).option("inferSchema", True).csv("source.csv")  # extract`}${hasB ? `\ndfB = spark.read.option("header", True).option("inferSchema", True).csv("source_b.csv")  # Source B` : ""}

${lines.filter(Boolean).join("\n") || "# (no transforms yet)"}
${rules.length ? `\n# ── data-quality expectations (route failures to a rejects sink) ──\n${rules.map((r) => `# expect ${ruleDesc(r)}`).join("\n")}` : ""}

${mode === "stream"
        ? `query = df.writeStream.format("console").outputMode("complete").start()\nquery.awaitTermination()`
        : `df.show()\ndf.write.mode("overwrite").parquet("output/")   # load`}`;
  }
  function copyCode() { navigator.clipboard.writeText(buildCode()).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500); }); }
  async function saveProject() {
    const config = { srcName, mode, ops, rules, models, pipeMode };
    try { const r = await fetch("/api/projects", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ lab: "etl", name: srcName || "ETL pipeline", config }) }); setSavedMsg(r.ok ? "Saved ✓" : "Save failed"); }
    catch { setSavedMsg("Save failed"); }
    setTimeout(() => setSavedMsg(""), 2000);
  }

  // ── DAG canvas ──
  const W = 158, H = 62, GAP = 46, y = 74;
  const nodeList = srcTable && pipe ? [
    { id: "src", icon: "📥", title: srcName.slice(0, 16), sub: mode === "stream" ? "Kafka topic" : "extract", count: srcTable.rows.length },
    ...ops.map((o, i) => ({ id: o.id, icon: OP_META[o.type].icon, title: OP_META[o.type].label, sub: opSummary(o), count: pipe.stages[i + 1].table.rows.length })),
    { id: "sink", icon: "🏬", title: "Sink", sub: "load → warehouse", count: pipe.final.rows.length },
  ] : [];
  const cw = Math.max(720, 20 + nodeList.length * (W + GAP));
  const dagCanvas = (editable: boolean) => (
    <div className="acanvas" style={{ height: 190 }}>
      <svg className="wires2" width={cw} height={190}>
        {nodeList.slice(0, -1).map((_, i) => { const x1 = 20 + i * (W + GAP) + W, x2 = 20 + (i + 1) * (W + GAP), yy = y + H / 2; const dx = Math.max(24, GAP / 2); const act = ["running", "done"].includes(nodeStatus[nodeList[i].id] || ""); return <path key={i} className={act ? "active" : ""} d={`M${x1} ${yy} C${x1 + dx} ${yy}, ${x2 - dx} ${yy}, ${x2} ${yy}`} />; })}
      </svg>
      {nodeList.map((n, i) => { const isOp = n.id !== "src" && n.id !== "sink"; const st = nodeStatus[n.id] || ""; return (
        <div key={n.id} className={`anode ${st} ${editable && sel === n.id ? "sel" : ""}`} style={{ left: 20 + i * (W + GAP), top: y, width: W, cursor: editable && isOp ? "pointer" : "default" }} onClick={() => editable && isOp && setSel(n.id)}>
          <div className="ah"><span className="aic">{n.icon}</span><div style={{ minWidth: 0 }}><div className="atitle">{n.title}</div><div className="asub" style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{n.sub}</div></div><span className="etl-count">{n.count}</span></div>
          {editable && isOp && <button className="anode-x" title="Remove" onClick={(e) => { e.stopPropagation(); removeOp(n.id); }}>×</button>}
        </div>
      ); })}
    </div>
  );

  const dtable = (t: Table, n: number) => (
    <div style={{ overflowX: "auto" }}><table className="dtable"><tbody>
      <tr>{t.cols.map((c) => <th key={c}>{c}</th>)}</tr>
      {t.rows.slice(0, n).map((r, i) => <tr key={i}>{t.cols.map((c) => <td key={c}>{r[c] == null ? "—" : String(r[c])}</td>)}</tr>)}
    </tbody></table></div>
  );

  // Shared "load to a database" box — platform DB or the user's own external DB.
  const storeBox = (t: Table) => (
    <div>
      <div className="seg" style={{ maxWidth: 320, marginBottom: 10 }}>
        <button className={storeTarget === "platform" ? "on" : ""} onClick={() => setStoreTarget("platform")}>Platform DB</button>
        <button className={storeTarget === "external" ? "on" : ""} onClick={() => setStoreTarget("external")}>External DB</button>
      </div>
      {storeTarget === "platform" ? (<>
        <div className="seg" style={{ maxWidth: 280, marginBottom: 8 }}><button className={storeMode === "new" ? "on" : ""} onClick={() => setStoreMode("new")}>New snapshot</button><button className={storeMode === "replace" ? "on" : ""} onClick={() => setStoreMode("replace")}>Replace same name</button></div>
        <div className="row" style={{ gap: 8 }}><input type="text" placeholder="dataset name" value={storeName} onChange={(e) => setStoreName(e.target.value)} /><button className="btn sm" onClick={() => storePlatform(t)} disabled={storeBusy || !t.rows.length}>{storeBusy ? "Storing…" : "🗄 Store"}</button></div>
        <div className="row" style={{ alignItems: "center", marginTop: 14 }}><label className="fld" style={{ margin: 0 }}>Stored datasets</label><button className="btn ghost sm" style={{ marginLeft: "auto" }} onClick={loadStored}>↻ Refresh</button></div>
        {stored.length === 0 ? <div className="note" style={{ marginTop: 8 }}>None yet.</div> : (
          <div style={{ display: "flex", flexDirection: "column", gap: 6, marginTop: 8 }}>
            {stored.map((d) => (
              <div key={d.id} className="row" style={{ alignItems: "center", gap: 10, border: "1px solid var(--border)", borderRadius: 8, padding: "8px 12px" }}>
                <span>🗄</span><b style={{ fontSize: 13 }}>{d.name}</b><span className="note">{d.rowCount} rows</span><span style={{ flex: 1 }} />
                <button className="btn ghost sm" onClick={() => downloadStored(d.id, d.name)}>Download</button>
                <button className="btn ghost sm danger" onClick={() => deleteStored(d.id)}>Delete</button>
              </div>
            ))}
          </div>
        )}
      </>) : (<>
        <div className="seg" style={{ maxWidth: 260, marginBottom: 8 }}><button className={extConn === "url" ? "on" : ""} onClick={() => setExtConn("url")}>Connection URL</button><button className={extConn === "fields" ? "on" : ""} onClick={() => setExtConn("fields")}>Host &amp; fields</button></div>
        {extConn === "url" ? (<>
          <label className="fld">Your MySQL / TiDB connection URL</label>
          <input type="text" value={extUrl} onChange={(e) => setExtUrl(e.target.value)} placeholder="mysql://user:pass@host:4000/db" />
        </>) : (<>
          <div className="split col-2e"><div><label className="fld">Host</label><input type="text" value={extHost} onChange={(e) => setExtHost(e.target.value)} placeholder="gateway.tidbcloud.com" /></div><div><label className="fld">Port</label><input type="text" value={extPort} onChange={(e) => setExtPort(e.target.value)} /></div></div>
          <div className="split col-2e" style={{ marginTop: 8 }}><div><label className="fld">User</label><input type="text" value={extUser} onChange={(e) => setExtUser(e.target.value)} /></div><div><label className="fld">Database</label><input type="text" value={extDb} onChange={(e) => setExtDb(e.target.value)} /></div></div>
          <label className="fld" style={{ marginTop: 8 }}>Password</label><input type="password" value={extPass} onChange={(e) => setExtPass(e.target.value)} />
        </>)}
        <label className="fld" style={{ marginTop: 10 }}>Target table name</label>
        <input type="text" value={extTable} onChange={(e) => setExtTable(e.target.value)} />
        <div className="seg" style={{ maxWidth: 280, margin: "10px 0" }}><button className={extMode === "append" ? "on" : ""} onClick={() => setExtMode("append")}>Append</button><button className={extMode === "upsert" ? "on" : ""} onClick={() => setExtMode("upsert")}>Upsert</button></div>
        {extMode === "upsert" && <div className="insp-field" style={{ marginBottom: 10 }}><div className="k">Key column (insert new / update existing on this)</div><select value={extKey || t.cols[0]} onChange={(e) => setExtKey(e.target.value)}>{t.cols.map((c) => <option key={c}>{c}</option>)}</select></div>}
        <button className="btn ghost sm" style={{ marginBottom: 8 }} onClick={() => setExtTypesOpen((o) => !o)}>{extTypesOpen ? "Hide column types" : "⚙ Column types (advanced)"}</button>
        {extTypesOpen && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6, marginBottom: 10 }}>
            {t.cols.map((c) => (
              <div key={c} className="row" style={{ gap: 6, alignItems: "center" }}>
                <span className="note" style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={c}>{c}</span>
                <select value={extTypes[c] || "auto"} onChange={(e) => setExtTypes((m) => ({ ...m, [c]: e.target.value }))} style={{ maxWidth: 120 }}>{EXT_TYPES.map((ty) => <option key={ty} value={ty}>{ty}</option>)}</select>
              </div>
            ))}
          </div>
        )}
        <button className="btn sm" onClick={() => storeExternal(t)} disabled={storeBusy || !t.rows.length}>{storeBusy ? "Loading…" : "🛢 Load to my DB"}</button>
        <div className="teach-note" style={{ marginTop: 10 }}><span className="ic">🔒</span><span>Connects to <b>your</b> database, creates the table if missing (types inferred), then {extMode === "upsert" ? "inserts new rows and updates existing ones on the key" : "appends the rows"}. Warns on <b>schema drift</b>. Credentials are used only for this request; private/internal hosts are blocked.</span></div>
      </>)}
    </div>
  );

  return (
    <>
      <div className="lab-head">
        <div><div className="eyebrow">Lab 06 · data engineering</div><h2 className="page-h">ETL Lab</h2><p className="page-sub" style={{ margin: 0 }}>Extract, profile & join sources, transform with a Spark-style DAG, validate, then load — to a file or a real database.</p></div>
        <div className="acts"><button className="btn ghost sm" onClick={saveProject}>{savedMsg || "💾 Save"}</button><button className="btn ghost sm" onClick={() => setShowCode(true)}>&lt;/&gt; Get code (PySpark)</button></div>
      </div>
      {msg && <div className="err">{msg}</div>}

      <div className="row" style={{ alignItems: "center", gap: 12, marginBottom: 12, flexWrap: "wrap" }}>
        <div className="seg" style={{ width: 260 }}>
          <button className={pipeMode === "etl" ? "on" : ""} onClick={() => { setPipeMode("etl"); setStep("extract"); }}>ETL</button>
          <button className={pipeMode === "elt" ? "on" : ""} onClick={() => { setPipeMode("elt"); setStep("extract"); }}>ELT</button>
        </div>
        <span className="note" style={{ flex: 1, minWidth: 220 }}>{pipeMode === "elt"
          ? "ELT — load raw first, then transform in SQL. The modern default (Snowflake / BigQuery + dbt)."
          : "ETL — transform before load. Best for streaming, PII masking, or constrained targets."}</span>
      </div>
      <div className="teach-note"><span className="ic">🎓</span><span><b>Teaching engine.</b> {pipeMode === "elt" ? <>Raw data lands first, then you transform it with <b>real SQL</b> in-browser. </> : <>Transforms run for real on your rows so you can watch each stage. </>}For production scale, use the <b>Get code (PySpark)</b> export.</span></div>

      <div className="stepper">
        <button className={step === "extract" ? "on" : ""} onClick={() => setStep("extract")}><b>1</b>Extract</button>
        <button className={step === "transform" ? "on" : ""} disabled={!srcTable} onClick={() => setStep("transform")}><b>2</b>{pipeMode === "elt" ? "Load raw" : "Transform"}</button>
        <button className={step === "load" ? "on" : ""} disabled={!srcTable} onClick={() => setStep("load")}><b>3</b>{pipeMode === "elt" ? "Transform (SQL)" : "Load & Run"}</button>
      </div>

      {/* STEP 1 — EXTRACT */}
      {step === "extract" && (
        <div className="card">
          <div className="card-h"><span className="t">Extract — connect a source</span>
            <div className="tabs">
              <button className={srcTab === "sample" ? "on" : ""} onClick={() => setSrcTab("sample")}>Sample</button>
              <button className={srcTab === "upload" ? "on" : ""} onClick={() => setSrcTab("upload")}>File</button>
              <button className={srcTab === "json" ? "on" : ""} onClick={() => setSrcTab("json")}>JSON</button>
              <button className={srcTab === "db" ? "on" : ""} onClick={() => setSrcTab("db")}>Database</button>
            </div>
          </div>
          <div className="card-b">
            {srcTab === "sample" && <div className="row" style={{ gap: 8 }}><select value={sampleKey} onChange={(e) => setSampleKey(e.target.value)}>{sampleSources().map((s) => <option key={s.key} value={s.key}>{s.label}</option>)}</select><button className="btn sm" onClick={loadSample}>Load</button></div>}
            {srcTab === "upload" && <><div className="dropzone" onClick={() => fileRef.current?.click()}>{busy ? "Parsing…" : <>Click to upload <b>CSV · TSV · Excel (.xlsx) · JSON</b> — parsed locally (Excel/xlsx via the server extractor)</>}</div><input ref={fileRef} type="file" accept=".csv,.tsv,.xlsx,.xls,.xlsm,.json" onChange={onFile} style={{ display: "none" }} /></>}
            {srcTab === "json" && <><label className="fld">Paste a JSON array of objects (or {"{ data: [ … ] }"})</label><textarea rows={5} value={jsonText} onChange={(e) => setJsonText(e.target.value)} /><div style={{ marginTop: 10 }}><button className="btn sm" onClick={() => loadJson(jsonText, "pasted JSON")}>Load JSON</button></div></>}
            {srcTab === "db" && <><label className="fld">MySQL / TiDB connection URL</label><input type="text" value={dbUrl} onChange={(e) => setDbUrl(e.target.value)} /><label className="fld" style={{ marginTop: 10 }}>Query</label><textarea rows={2} value={dbQuery} onChange={(e) => setDbQuery(e.target.value)} /><div style={{ marginTop: 10 }}><button className="btn sm" onClick={runDb} disabled={busy}>{busy ? "Running…" : "Run query & extract"}</button></div></>}

            {srcTable && (<>
              <div className="row" style={{ gap: 10, margin: "16px 0 12px", flexWrap: "wrap", alignItems: "center" }}>
                <span className="pill"><span className="dot" />{srcName}</span><span className="pill">rows {srcTable.rows.length}</span><span className="pill">cols {srcTable.cols.length}</span>
                <button className="btn ghost sm" onClick={() => setProfOpen((o) => !o)}>{profOpen ? "Hide profile" : "📊 Profile data"}</button>
                <div className="rangebtns" style={{ marginLeft: "auto" }}><button className={viewMode === "head" ? "on" : ""} onClick={() => setViewMode("head")}>Head</button><button className={viewMode === "tail" ? "on" : ""} onClick={() => setViewMode("tail")}>Tail</button></div>
              </div>

              {profOpen && (
                <div className="prof-grid">
                  {prof.map((p) => (
                    <div key={p.name} className="prof-card">
                      <div className="prof-h"><b title={p.name}>{p.name}</b><span className="badge">{p.type}</span></div>
                      <div className="prof-stats"><span>{srcTable.rows.length - p.nulls}/{srcTable.rows.length} filled</span><span>{p.distinct} distinct</span>{p.nulls > 0 && <span style={{ color: "var(--crit)" }}>{p.nulls} null</span>}</div>
                      {p.type === "num" ? (<>
                        <div className="prof-stats"><span>min {p.min}</span><span>μ {p.mean}</span><span>max {p.max}</span></div>
                        <div className="prof-hist">{(p.hist || []).map((h, i) => { const mx = Math.max(1, ...(p.hist || [])); return <span key={i} style={{ height: `${Math.max(2, (h / mx) * 34)}px` }} title={`${h}`} />; })}</div>
                      </>) : (
                        <div className="prof-top">{p.top.map((tv) => <div key={tv.v} className="prof-top-row"><span className="v" title={tv.v}>{tv.v}</span><span className="c">{tv.count}</span></div>)}</div>
                      )}
                    </div>
                  ))}
                </div>
              )}

              <div style={{ overflowX: "auto" }}><table className="dtable"><tbody>
                <tr><th>#</th>{srcTable.cols.map((c) => { const p = prof.find((x) => x.name === c)!; return <th key={c}>{c} <span style={{ color: "var(--faint)" }}>{p.type}{p.nulls ? `·${p.nulls}na` : ""}</span></th>; })}</tr>
                {(viewMode === "head" ? srcTable.rows.slice(0, 8) : srcTable.rows.slice(-8)).map((r, i) => <tr key={i}><td style={{ color: "var(--faint)" }}>{viewMode === "head" ? i : srcTable.rows.length - 8 + i}</td>{srcTable.cols.map((c) => <td key={c}>{r[c] == null ? "—" : String(r[c])}</td>)}</tr>)}
              </tbody></table></div>

              {/* Source B for join / union */}
              <div style={{ marginTop: 18, borderTop: "1px solid var(--border)", paddingTop: 14 }}>
                <label className="fld">Source B <span className="note">— a second table to Join or Union with (optional)</span></label>
                <div className="row" style={{ gap: 8, flexWrap: "wrap", alignItems: "center" }}>
                  <select value={sampleKeyB} onChange={(e) => setSampleKeyB(e.target.value)} style={{ maxWidth: 260 }}>{sampleSources().map((s) => <option key={s.key} value={s.key}>{s.label}</option>)}</select>
                  <button className="btn ghost sm" onClick={loadSampleB}>Load as Source B</button>
                  {srcTableB && <><span className="pill"><span className="dot" />{srcNameB}</span><span className="pill">rows {srcTableB.rows.length}</span><span className="pill">cols {srcTableB.cols.length}</span><button className="btn ghost sm" onClick={() => { setSrcTableB(null); setSrcNameB(""); }}>Clear</button></>}
                </div>
              </div>

              <div className="stepnav"><button className="btn" onClick={() => setStep("transform")}>Next: Transform →</button></div>
            </>)}
          </div>
        </div>
      )}

      {/* STEP 2 (ELT) — LOAD RAW */}
      {step === "transform" && srcTable && pipeMode === "elt" && (
        <div className="card">
          <div className="card-h"><span className="t">Load raw → landing zone</span><span className="mono r">{srcTable.rows.length} rows</span></div>
          <div className="card-b">
            <div className="teach-note"><span className="ic">📥</span><span>In <b>ELT</b> the raw data lands <b>untransformed</b> first — this preserves the original so you can re-transform it any time. Optionally persist the raw table below, then move on to transform it in SQL.</span></div>
            <div style={{ maxHeight: 220, overflowY: "auto" }}>{dtable(srcTable, 10)}</div>
            <label className="fld" style={{ marginTop: 14 }}>Persist the raw data (optional)</label>
            {storeBox(srcTable)}
            <div className="stepnav"><button className="btn ghost" onClick={() => setStep("extract")}>← Extract</button><button className="btn" onClick={() => setStep("load")}>Next: Transform (SQL) →</button></div>
          </div>
        </div>
      )}

      {/* STEP 2 (ETL) — TRANSFORM */}
      {step === "transform" && srcTable && pipeMode === "etl" && (
        <div className="split" style={{ gridTemplateColumns: "1fr 300px" }}>
          <div className="card">
            <div className="card-h"><span className="t">Transform — build the DAG</span>
              <div className="r" style={{ position: "relative" }}>
                <button className="btn ghost sm" onClick={() => setAddOpen((o) => !o)}>+ Add transform</button>
                {addOpen && <div className="addmenu2" style={{ maxHeight: 320, overflowY: "auto" }}><div className="hd">Add a transform</div>{(Object.keys(OP_META) as OpType[]).map((t) => <div key={t} className="ai" onClick={() => addOp(t)}><span>{OP_META[t].icon}</span>{OP_META[t].label}{OP_META[t].needsB && <span className="ai-state">needs B</span>}</div>)}</div>}
              </div>
            </div>
            <div className="card-b" style={{ padding: 0 }}>{dagCanvas(true)}</div>
            <div className="card-b" style={{ borderTop: "1px solid var(--border)" }}>
              <label className="fld">Live preview — sink output ({pipe!.final.rows.length} rows)</label>
              <div style={{ maxHeight: 180, overflowY: "auto" }}>{dtable(pipe!.final, 6)}</div>
              {/* Data Visualization Explorer */}
              {pipe && srcTable && (
                <DataVizExplorer
                  sourceTable={srcTable}
                  finalTable={pipe.final}
                  ops={ops}
                  rules={rules}
                />
              )}
              <div className="stepnav"><button className="btn ghost" onClick={() => setStep("extract")}>← Extract</button><button className="btn" onClick={() => setStep("load")}>Next: Load &amp; Run →</button></div>
            </div>
          </div>
          <div className="card">
            <div className="card-h"><span className="t">Configure</span><span className="mono r">{selIdx >= 0 ? OP_META[ops[selIdx].type].label : "—"}</span></div>
            <div className="card-b" style={{ maxHeight: 470, overflow: "auto" }}>
              {selIdx < 0 && <div className="note">Add a transform, then click its node to configure it.</div>}
              {selIdx >= 0 && (() => {
                const o = ops[selIdx];
                const colSel = (val: string | undefined, on: (v: string) => void, list = inCols) => <select value={val} onChange={(e) => on(e.target.value)}>{list.map((c) => <option key={c}>{c}</option>)}</select>;
                return (<>
                  <div className="note" style={{ marginBottom: 10 }}>{OP_META[o.type].hint}</div>
                  {o.type === "filter" && (<>
                    <div className="insp-field"><div className="k">Column</div>{colSel(o.col, (v) => patch({ col: v }))}</div>
                    <div className="insp-field"><div className="k">Operator</div><select value={o.op} onChange={(e) => patch({ op: e.target.value })}>{["==", "!=", ">", "<", ">=", "<=", "contains"].map((x) => <option key={x}>{x}</option>)}</select></div>
                    <div className="insp-field"><div className="k">Value</div><input type="text" value={o.value ?? ""} onChange={(e) => patch({ value: e.target.value })} /></div>
                  </>)}
                  {o.type === "select" && <div className="insp-field"><div className="k">Keep columns</div><div className="checklist">{inCols.map((c) => <span key={c} className={`chk ${(o.cols || []).includes(c) ? "on" : ""}`} onClick={() => patch({ cols: (o.cols || []).includes(c) ? (o.cols || []).filter((x) => x !== c) : [...(o.cols || []), c] })}>{c}</span>)}</div></div>}
                  {o.type === "derive" && (<>
                    <div className="insp-field"><div className="k">New column name</div><input type="text" value={o.name ?? ""} onChange={(e) => patch({ name: e.target.value })} /></div>
                    <div className="insp-field"><div className="k">Left (column or number)</div><input type="text" value={o.left ?? ""} onChange={(e) => patch({ left: e.target.value })} list="etl-cols" /></div>
                    <div className="insp-field"><div className="k">Operation</div><select value={o.arith} onChange={(e) => patch({ arith: e.target.value })}>{["+", "-", "*", "/"].map((x) => <option key={x}>{x}</option>)}</select></div>
                    <div className="insp-field"><div className="k">Right (column or number)</div><input type="text" value={o.right ?? ""} onChange={(e) => patch({ right: e.target.value })} list="etl-cols" /></div>
                    <datalist id="etl-cols">{inCols.map((c) => <option key={c} value={c} />)}</datalist>
                  </>)}
                  {o.type === "aggregate" && (<>
                    <div className="insp-field"><div className="k">Group by</div>{colSel(o.groupBy, (v) => patch({ groupBy: v }))}</div>
                    <div className="insp-field"><div className="k">Aggregate</div><select value={o.agg} onChange={(e) => patch({ agg: e.target.value })}>{["count", "sum", "avg", "min", "max"].map((x) => <option key={x}>{x}</option>)}</select></div>
                    {o.agg !== "count" && <div className="insp-field"><div className="k">Of column</div>{colSel(o.aggCol, (v) => patch({ aggCol: v }))}</div>}
                  </>)}
                  {o.type === "sort" && (<>
                    <div className="insp-field"><div className="k">Column</div>{colSel(o.col, (v) => patch({ col: v }))}</div>
                    <div className="insp-field"><div className="k">Direction</div><select value={o.dir} onChange={(e) => patch({ dir: e.target.value })}><option value="asc">ascending</option><option value="desc">descending</option></select></div>
                  </>)}
                  {o.type === "dedupe" && <div className="insp-field"><div className="k">Unique by (empty = whole row)</div><div className="checklist">{inCols.map((c) => <span key={c} className={`chk ${(o.cols || []).includes(c) ? "on" : ""}`} onClick={() => patch({ cols: (o.cols || []).includes(c) ? (o.cols || []).filter((x) => x !== c) : [...(o.cols || []), c] })}>{c}</span>)}</div></div>}
                  {o.type === "clean" && <div className="insp-field"><div className="k">Mode</div><select value={o.mode} onChange={(e) => patch({ mode: e.target.value })}><option value="dropnull">drop rows with nulls</option><option value="fill0">fill nulls with 0</option></select></div>}
                  {o.type === "rename" && (<>
                    <div className="insp-field"><div className="k">Column</div>{colSel(o.col, (v) => patch({ col: v }))}</div>
                    <div className="insp-field"><div className="k">New name</div><input type="text" value={o.name ?? ""} onChange={(e) => patch({ name: e.target.value })} /></div>
                  </>)}
                  {o.type === "limit" && <div className="insp-field"><div className="k">Keep first N rows</div><input type="text" value={o.value ?? ""} onChange={(e) => patch({ value: e.target.value })} /></div>}
                  {o.type === "sample" && <div className="insp-field"><div className="k">Fraction (≤1) or count</div><input type="text" value={o.value ?? ""} onChange={(e) => patch({ value: e.target.value })} /></div>}
                  {o.type === "map" && (<>
                    <div className="insp-field"><div className="k">Column</div>{colSel(o.col, (v) => patch({ col: v }))}</div>
                    <div className="insp-field"><div className="k">Function</div><select value={o.fn} onChange={(e) => patch({ fn: e.target.value })}>{["round", "abs", "floor", "ceil", "upper", "lower", "trim", "length"].map((x) => <option key={x}>{x}</option>)}</select></div>
                  </>)}
                  {o.type === "fillna" && (<>
                    <div className="insp-field"><div className="k">Column</div><select value={o.col} onChange={(e) => patch({ col: e.target.value })}><option value="">(all columns)</option>{inCols.map((c) => <option key={c}>{c}</option>)}</select></div>
                    <div className="insp-field"><div className="k">Fill value</div><input type="text" value={o.value ?? ""} onChange={(e) => patch({ value: e.target.value })} /></div>
                  </>)}
                  {o.type === "bucket" && (<>
                    <div className="insp-field"><div className="k">Numeric column</div>{colSel(o.col, (v) => patch({ col: v }))}</div>
                    <div className="insp-field"><div className="k">New column name</div><input type="text" value={o.name ?? ""} onChange={(e) => patch({ name: e.target.value })} /></div>
                    <div className="insp-field"><div className="k">Number of buckets</div><input type="text" value={o.value ?? ""} onChange={(e) => patch({ value: e.target.value })} /></div>
                  </>)}
                  {o.type === "join" && (<>
                    {!srcTableB && <div className="warnbar" style={{ marginBottom: 8 }}>Load a <b>Source B</b> in the Extract step first.</div>}
                    <div className="insp-field"><div className="k">Join type</div><select value={o.joinType} onChange={(e) => patch({ joinType: e.target.value })}><option value="inner">inner</option><option value="left">left</option><option value="right">right</option></select></div>
                    <div className="insp-field"><div className="k">Left key (this table)</div>{colSel(o.col, (v) => patch({ col: v }))}</div>
                    <div className="insp-field"><div className="k">Right key (Source B)</div>{colSel(o.rightKey, (v) => patch({ rightKey: v }), bCols)}</div>
                  </>)}
                  {o.type === "union" && (<>
                    {!srcTableB && <div className="warnbar" style={{ marginBottom: 8 }}>Load a <b>Source B</b> in the Extract step first.</div>}
                    <div className="insp-field"><div className="k">Mode</div><select value={o.mode} onChange={(e) => patch({ mode: e.target.value })}><option value="all">union all (keep dupes)</option><option value="distinct">distinct</option></select></div>
                  </>)}
                  {o.type === "pivot" && (<>
                    <div className="insp-field"><div className="k">Index (rows)</div>{colSel(o.groupBy, (v) => patch({ groupBy: v }))}</div>
                    <div className="insp-field"><div className="k">Spread column → new cols</div>{colSel(o.col, (v) => patch({ col: v }))}</div>
                    <div className="insp-field"><div className="k">Value column</div>{colSel(o.aggCol, (v) => patch({ aggCol: v }))}</div>
                    <div className="insp-field"><div className="k">Aggregate</div><select value={o.agg} onChange={(e) => patch({ agg: e.target.value })}>{["sum", "avg", "min", "max", "count"].map((x) => <option key={x}>{x}</option>)}</select></div>
                  </>)}
                  {o.type === "unpivot" && (<>
                    <div className="insp-field"><div className="k">Columns to melt</div><div className="checklist">{inCols.map((c) => <span key={c} className={`chk ${(o.cols || []).includes(c) ? "on" : ""}`} onClick={() => patch({ cols: (o.cols || []).includes(c) ? (o.cols || []).filter((x) => x !== c) : [...(o.cols || []), c] })}>{c}</span>)}</div></div>
                    <div className="insp-field"><div className="k">Variable column name</div><input type="text" value={o.name ?? ""} onChange={(e) => patch({ name: e.target.value })} /></div>
                    <div className="insp-field"><div className="k">Value column name</div><input type="text" value={o.value ?? ""} onChange={(e) => patch({ value: e.target.value })} /></div>
                  </>)}
                  {o.type === "window" && (<>
                    <div className="insp-field"><div className="k">Partition by</div><select value={o.groupBy} onChange={(e) => patch({ groupBy: e.target.value })}><option value="(none)">(whole table)</option>{inCols.map((c) => <option key={c}>{c}</option>)}</select></div>
                    <div className="insp-field"><div className="k">Column</div>{colSel(o.col, (v) => patch({ col: v }))}</div>
                    <div className="insp-field"><div className="k">Function</div><select value={o.fn} onChange={(e) => patch({ fn: e.target.value })}><option value="running_sum">running sum</option><option value="row_number">row number</option><option value="rank">rank</option><option value="lag">lag</option><option value="lead">lead</option></select></div>
                    <div className="insp-field"><div className="k">Output column</div><input type="text" value={o.name ?? ""} onChange={(e) => patch({ name: e.target.value })} /></div>
                  </>)}
                  {o.type === "regex" && (<>
                    <div className="insp-field"><div className="k">Column</div>{colSel(o.col, (v) => patch({ col: v }))}</div>
                    <div className="insp-field"><div className="k">Pattern (first group extracted)</div><input type="text" value={o.value ?? ""} onChange={(e) => patch({ value: e.target.value })} /></div>
                    <div className="insp-field"><div className="k">Output column</div><input type="text" value={o.name ?? ""} onChange={(e) => patch({ name: e.target.value })} /></div>
                  </>)}
                  {o.type === "dateparse" && (<>
                    <div className="insp-field"><div className="k">Date column</div>{colSel(o.col, (v) => patch({ col: v }))}</div>
                    <div className="insp-field"><div className="k">Extract</div><select value={o.fn} onChange={(e) => patch({ fn: e.target.value })}><option value="year">year</option><option value="month">month</option><option value="day">day</option><option value="weekday">weekday</option><option value="iso">ISO date</option><option value="days_since">days since</option></select></div>
                    <div className="insp-field"><div className="k">Output column</div><input type="text" value={o.name ?? ""} onChange={(e) => patch({ name: e.target.value })} /></div>
                  </>)}
                  <button className="btn ghost sm" onClick={() => removeOp(o.id)}>Remove transform</button>
                </>);
              })()}
            </div>
          </div>
        </div>
      )}

      {/* STEP 3 (ETL) — LOAD & RUN */}
      {step === "load" && srcTable && pipe && deliver && qc && pipeMode === "etl" && (
        <>
          <div className="seg" style={{ maxWidth: 620, marginBottom: 14 }}>
            <button className={loadTab === "run" ? "on" : ""} onClick={() => setLoadTab("run")}>Run</button>
            <button className={loadTab === "quality" ? "on" : ""} onClick={() => setLoadTab("quality")}>Quality{rules.length ? ` (${qc.rejects.length})` : ""}</button>
            <button className={loadTab === "deliver" ? "on" : ""} onClick={() => setLoadTab("deliver")}>Deliver</button>
            <button className={loadTab === "schedule" ? "on" : ""} onClick={() => setLoadTab("schedule")}>Schedule</button>
            <button className={loadTab === "lineage" ? "on" : ""} onClick={() => setLoadTab("lineage")}>Lineage</button>
          </div>

          {loadTab === "run" && (<>
            <div className="seg" style={{ maxWidth: 380, marginBottom: 16 }}>
              <button className={mode === "batch" ? "on" : ""} onClick={() => setMode("batch")}>Batch (Spark)</button>
              <button className={mode === "stream" ? "on" : ""} onClick={() => setMode("stream")}>Streaming (Kafka + Spark)</button>
            </div>
            <div className="card">
              <div className="card-h"><span className="t">Run pipeline</span><div className="r"><button className="btn sm" onClick={run} disabled={running}>{running ? <><span className="busy-dot" />running…</> : "▶ Run pipeline"}</button></div></div>
              <div className="card-b" style={{ padding: 0 }}>
                <div className="etl-metrics">
                  <div className="m">records in<b>{metrics ? metrics.records : srcTable.rows.length}</b></div>
                  <div className="m">throughput<b>{metrics ? `${metrics.throughput}/s` : "—"}</b></div>
                  <div className="m">{mode === "stream" ? "micro-batches" : "stages"}<b>{metrics ? (mode === "stream" ? metrics.batches : ops.length + 1) : ops.length + 1}</b></div>
                  <div className="m">rows out<b>{pipe.final.rows.length}</b></div>
                  {rules.length > 0 && <div className="m">rejected<b style={{ color: qc.rejects.length ? "var(--crit)" : undefined }}>{qc.rejects.length}</b></div>}
                </div>
                {dagCanvas(false)}
              </div>
            </div>
            <div className="card" style={{ marginTop: 16 }}>
              <div className="card-h"><span className="t">Sink output</span><div className="r"><span className="mono">{pipe.final.rows.length} × {pipe.final.cols.length}</span></div></div>
              <div className="card-b">
                {dtable(pipe.final, 14)}
                {pipe.final.rows.length > 14 && <div className="note" style={{ marginTop: 8 }}>+ {pipe.final.rows.length - 14} more rows</div>}
                <div className="stepnav"><button className="btn ghost" onClick={() => setStep("transform")}>← Transform</button></div>
              </div>
            </div>
          </>)}

          {loadTab === "quality" && (
            <div className="card">
              <div className="card-h"><span className="t">Data-quality expectations</span><div className="r"><button className="btn ghost sm" onClick={addRule}>+ Add rule</button></div></div>
              <div className="card-b">
                <div className="note" style={{ marginBottom: 10 }}>Each rule has an action — <b>Reject</b> (quarantine to dead-letter), <b>Drop</b>, <b>Fix</b> (auto-repair: fill / clamp / dedupe), or <b>Warn</b>. Clean rows flow on to Deliver.</div>
                {rules.length === 0 && <div className="note">No rules yet — add one to validate the output.</div>}
                {rules.map((r) => (
                  <div key={r.id} className="row" style={{ gap: 8, marginBottom: 8, flexWrap: "wrap", alignItems: "center" }}>
                    <select value={r.col} onChange={(e) => patchRule(r.id, { col: e.target.value })} style={{ maxWidth: 150 }}>{pipe.final.cols.map((c) => <option key={c}>{c}</option>)}</select>
                    <select value={r.type} onChange={(e) => patchRule(r.id, { type: e.target.value as RuleType })} style={{ maxWidth: 150 }}>{(Object.keys(RULE_META) as RuleType[]).map((t) => <option key={t} value={t}>{RULE_META[t].label}</option>)}</select>
                    {r.type === "in_range" && <><input type="text" placeholder="min" value={r.min ?? ""} onChange={(e) => patchRule(r.id, { min: e.target.value })} style={{ maxWidth: 80 }} /><input type="text" placeholder="max" value={r.max ?? ""} onChange={(e) => patchRule(r.id, { max: e.target.value })} style={{ maxWidth: 80 }} /></>}
                    {r.type === "regex" && <input type="text" placeholder="pattern" value={r.pattern ?? ""} onChange={(e) => patchRule(r.id, { pattern: e.target.value })} style={{ maxWidth: 180 }} />}
                    {r.type === "in_set" && <input type="text" placeholder="a, b, c" value={r.set ?? ""} onChange={(e) => patchRule(r.id, { set: e.target.value })} style={{ maxWidth: 150 }} />}
                    <span className="note">→</span>
                    <select value={r.action || "reject"} onChange={(e) => patchRule(r.id, { action: e.target.value as RuleAction })} style={{ maxWidth: 120 }} title={ACTION_META[r.action || "reject"].hint}>{(Object.keys(ACTION_META) as RuleAction[]).map((a) => <option key={a} value={a}>{ACTION_META[a].label}</option>)}</select>
                    {r.action === "fix" && r.type !== "unique" && r.type !== "in_range" && <input type="text" placeholder="fill value" value={r.fix ?? ""} onChange={(e) => patchRule(r.id, { fix: e.target.value })} style={{ maxWidth: 100 }} />}
                    <span className="note" style={{ flex: 1 }} />
                    <button className="btn ghost sm" onClick={() => removeRule(r.id)}>×</button>
                  </div>
                ))}
                {rules.length > 0 && (<>
                  <div className="etl-metrics" style={{ margin: "14px 0" }}>
                    <div className="m">rules<b>{rules.length}</b></div>
                    <div className="m">clean out<b style={{ color: "var(--good)" }}>{qc.clean.rows.length}</b></div>
                    <div className="m">fixed<b style={{ color: qc.fixedCells ? "var(--sky)" : undefined }}>{qc.fixedCells}</b></div>
                    <div className="m">dropped<b>{qc.dropped}</b></div>
                    <div className="m">rejected<b style={{ color: qc.rejects.length ? "var(--crit)" : undefined }}>{qc.rejects.length}</b></div>
                  </div>
                  <label className="fld">Report</label>
                  <div style={{ display: "flex", flexDirection: "column", gap: 6, marginBottom: 14 }}>
                    {qc.report.map((rp) => <div key={rp.id} className="qc-row"><span className={`qc-dot ${rp.ok ? "ok" : rp.action === "reject" ? "bad" : "warn"}`} />{rp.desc}<span className="badge" style={{ marginLeft: 8 }}>{ACTION_META[rp.action].label}</span><span className="r mono" style={{ marginLeft: "auto", color: rp.ok ? "var(--good)" : rp.action === "reject" ? "var(--crit)" : "var(--warn)" }}>{rp.ok ? "PASS" : `${rp.fails} hit${rp.action === "fix" ? ` · ${rp.fixed} fixed` : rp.action === "drop" ? " · dropped" : rp.action === "warn" ? " · warned" : ""}`}</span></div>)}
                  </div>
                  {qc.rejects.length > 0 && (<>
                    <div className="row" style={{ alignItems: "center" }}><label className="fld" style={{ margin: 0 }}>Rejects (dead-letter) — {qc.rejects.length} rows</label><button className="btn ghost sm" style={{ marginLeft: "auto" }} onClick={() => download(toCSV({ cols: [...pipe.final.cols, "_reasons"], rows: qc.rejects.map((rj) => ({ ...rj.row, _reasons: rj.reasons.join("; ") })) }), "rejects.csv", "text/csv")}>⬇ Download rejects</button></div>
                    <div style={{ overflowX: "auto", maxHeight: 220, overflowY: "auto", marginTop: 8 }}><table className="dtable"><tbody>
                      <tr>{pipe.final.cols.map((c) => <th key={c}>{c}</th>)}<th>reasons</th></tr>
                      {qc.rejects.slice(0, 30).map((rj, i) => <tr key={i}>{pipe.final.cols.map((c) => <td key={c}>{rj.row[c] == null ? "—" : String(rj.row[c])}</td>)}<td style={{ color: "var(--crit)" }}>{rj.reasons.join("; ")}</td></tr>)}
                    </tbody></table></div>
                  </>)}
                </>)}
              </div>
            </div>
          )}

          {loadTab === "deliver" && (
            <div className="card">
              <div className="card-h"><span className="t">Deliver output</span><span className="mono r">{deliver.rows.length} rows{rules.length ? " (clean)" : ""}</span></div>
              <div className="card-b">
                <div className="row" style={{ gap: 8, flexWrap: "wrap" }}>
                  <button className="btn ghost sm" onClick={() => download(toCSV(deliver), "etl_output.csv", "text/csv")}>⬇ Export CSV</button>
                  <button className="btn ghost sm" onClick={() => download(toJSON(deliver), "etl_output.json", "application/json")}>⬇ Export JSON</button>
                  <button className="btn ghost sm" onClick={() => downloadParquet(deliver)}>⬇ Parquet</button>
                </div>
                <div style={{ marginTop: 16, borderTop: "1px solid var(--border)", paddingTop: 14 }}>
                  <label className="fld">Load to a database <span className="note">— the platform DB, or your own external database</span></label>
                  {storeBox(deliver)}
                </div>
              </div>
            </div>
          )}

          {loadTab === "schedule" && (
            <div className="card">
              <div className="card-h"><span className="t">Schedule (simulation)</span></div>
              <div className="card-b">
                <div className="note" style={{ marginBottom: 12 }}>Simulate a scheduled job — runs the pipeline every N seconds and logs each run. In production this is an Airflow DAG / cron trigger (see the code export).</div>
                <div className="row" style={{ gap: 8, alignItems: "center" }}>
                  <span className="note">Run every</span>
                  <input type="number" min={2} max={120} value={schedEvery} onChange={(e) => setSchedEvery(Math.max(2, Number(e.target.value) || 5))} style={{ maxWidth: 90 }} disabled={schedOn} />
                  <span className="note">seconds</span>
                  <button className={`btn sm ${schedOn ? "ghost" : ""}`} onClick={toggleSchedule}>{schedOn ? "■ Stop" : "▶ Start schedule"}</button>
                  {schedOn && <span className="pill"><span className="dot" style={{ background: "var(--good)" }} />running</span>}
                </div>
                {schedRuns.length > 0 && (
                  <table className="dtable" style={{ marginTop: 14 }}><tbody>
                    <tr><th>#</th><th>time</th><th>records in</th><th>rows out</th><th>rejected</th><th>status</th></tr>
                    {schedRuns.map((r, i) => <tr key={i}><td style={{ color: "var(--faint)" }}>{schedRuns.length - i}</td><td>{r.t}</td><td>{r.inn}</td><td>{r.out}</td><td style={{ color: r.rej ? "var(--crit)" : undefined }}>{r.rej}</td><td style={{ color: "var(--good)" }}>✓ success</td></tr>)}
                  </tbody></table>
                )}
              </div>
            </div>
          )}

          {loadTab === "lineage" && (
            <div className="card">
              <div className="card-h"><span className="t">Column lineage</span><span className="mono r">{deliver.cols.length} output columns</span></div>
              <div className="card-b">
                <div className="note" style={{ marginBottom: 12 }}>Which source columns each output column traces back to (B.* = Source B).</div>
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  {lin.filter((l) => deliver.cols.includes(l.col)).map((l) => (
                    <div key={l.col} className="row" style={{ alignItems: "center", gap: 10, flexWrap: "wrap" }}>
                      <b style={{ fontSize: 13, minWidth: 120 }}>{l.col}</b><span className="note">←</span>
                      <div className="chips">{l.from.length ? l.from.map((f) => <span key={f} className="chip" style={{ cursor: "default" }}>{f}</span>) : <span className="note">literal / constant</span>}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </>
      )}

      {/* STEP 3 (ELT) — TRANSFORM WITH SQL */}
      {step === "load" && srcTable && pipeMode === "elt" && (
        <>
          <div className="card">
            <div className="card-h"><span className="t">Transform with SQL</span><div className="r"><button className="btn sm" onClick={runSqlNow} disabled={sqlBusy}>{sqlBusy ? <><span className="busy-dot" />running…</> : "▶ Run SQL"}</button></div></div>
            <div className="card-b">
              <div className="note" style={{ marginBottom: 8 }}>Real SQL over the landed data (dbt-style). Tables: <code>raw</code>{srcTableB ? <>, <code>b</code></> : ""}{models.length ? <> + your models ({models.map((m) => m.name).join(", ")})</> : ""}. Supports SELECT / WHERE / JOIN / GROUP BY / aggregates / ORDER BY / LIMIT.</div>
              <textarea rows={8} value={sqlText} onChange={(e) => setSqlText(e.target.value)} spellCheck={false} />
              <div style={{ marginTop: 12, borderTop: "1px solid var(--border)", paddingTop: 12 }}>
                <label className="fld">Models <span className="note">— save a query as a named table; later models &amp; the editor can query it (mini-dbt)</span></label>
                <div className="row" style={{ gap: 8 }}><input type="text" placeholder="model name (e.g. paid_orders)" value={modelName} onChange={(e) => setModelName(e.target.value)} style={{ maxWidth: 240 }} /><button className="btn ghost sm" onClick={saveModel} disabled={!modelName.trim()}>+ Save current SQL as model</button></div>
                {models.length > 0 && <div className="chips" style={{ marginTop: 8 }}>{models.map((m, i) => <span key={i} className="chip" style={{ cursor: "pointer" }} onClick={() => setSqlText(m.sql)} title="click to load into the editor">{m.name} <b onClick={(e) => { e.stopPropagation(); removeModel(i); }} style={{ color: "var(--faint)" }}>×</b></span>)}</div>}
                {models.length > 0 && (<><label className="fld" style={{ marginTop: 12 }}>Dependency graph</label>{modelDag()}</>)}
              </div>
              {sqlErr && <div className="err" style={{ marginTop: 10 }}>{sqlErr}</div>}
              {sqlResult && (<>
                <label className="fld" style={{ marginTop: 14 }}>Result — {sqlResult.rows.length} rows × {sqlResult.cols.length} cols</label>
                {dtable(sqlResult, 14)}
                {sqlResult.rows.length > 14 && <div className="note" style={{ marginTop: 8 }}>+ {sqlResult.rows.length - 14} more rows</div>}
              </>)}
              <div className="stepnav"><button className="btn ghost" onClick={() => setStep("transform")}>← Load raw</button></div>
            </div>
          </div>
          {sqlResult && sqlResult.rows.length > 0 && (
            <div className="card" style={{ marginTop: 16 }}>
              <div className="card-h"><span className="t">Deliver transformed table</span><div className="r"><button className="btn ghost sm" onClick={() => download(toCSV(sqlResult), "elt_output.csv", "text/csv")}>⬇ CSV</button><button className="btn ghost sm" onClick={() => download(toJSON(sqlResult), "elt_output.json", "application/json")}>⬇ JSON</button><button className="btn ghost sm" onClick={() => downloadParquet(sqlResult)}>⬇ Parquet</button></div></div>
              <div className="card-b"><label className="fld">Load to a database</label>{storeBox(sqlResult)}</div>
            </div>
          )}
        </>
      )}

      <div className={`modal-wrap ${showCode ? "show" : ""}`} onClick={(e) => { if (e.target === e.currentTarget) setShowCode(false); }}>
        <div className="modal"><div className="mh"><b>ETL pipeline · PySpark</b><div className="r" style={{ marginLeft: "auto", display: "flex", gap: 8 }}><button className="btn ghost sm" onClick={copyCode}>{copied ? "Copied ✓" : "Copy"}</button></div><button className="x" onClick={() => setShowCode(false)}>×</button></div><div className="mb"><div className="code">{buildCode()}</div></div></div>
      </div>
    </>
  );
}
