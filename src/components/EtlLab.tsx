"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  parseRecords, sampleSources, runPipeline, toCSV, tableFromRecords, profile, OP_META,
  type Table, type EtlOp, type OpType,
} from "@/lib/etlUtils";

type Step = "extract" | "transform" | "load";
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
  }
}

export default function EtlLab() {
  const [step, setStep] = useState<Step>("extract");
  const [srcTable, setSrcTable] = useState<Table | null>(null);
  const [srcName, setSrcName] = useState("");
  const [srcTab, setSrcTab] = useState<"sample" | "upload" | "json" | "db">("sample");
  const [sampleKey, setSampleKey] = useState("orders");
  const [jsonText, setJsonText] = useState('[\n  {"id": 1, "region": "US", "amount": 120.5, "status": "paid"},\n  {"id": 2, "region": "EU", "amount": 0, "status": "failed"}\n]');
  const [dbUrl, setDbUrl] = useState("mysql://user:pass@host:4000/db");
  const [dbQuery, setDbQuery] = useState("SELECT * FROM orders LIMIT 500;");
  const [busy, setBusy] = useState(false);
  const [viewMode, setViewMode] = useState<"head" | "tail">("head");

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
  const fileRef = useRef<HTMLInputElement>(null);
  const timers = useRef<ReturnType<typeof setTimeout>[]>([]);

  function setSource(t: Table, name: string) {
    if (!t.cols.length || !t.rows.length) { setMsg("The source has no usable rows / columns."); return; }
    setSrcTable(t); setSrcName(name); setMsg(""); setOps([]); setSel(null); setNodeStatus({}); setMetrics(null);
  }
  function loadSample() { const s = sampleSources().find((x) => x.key === sampleKey)!; setSource(parseRecords(s.csv), s.label); }
  useEffect(() => { const s = sampleSources().find((x) => x.key === "orders")!; setSource(parseRecords(s.csv), s.label); const t = timers; return () => t.current.forEach(clearTimeout); }, []);
  // Load a saved pipeline when opened from My Projects (?project=<id>). Only the
  // config is stored (not raw data), so re-seed a named sample and re-apply ops.
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

  const pipe = useMemo(() => (srcTable ? runPipeline(srcTable, ops) : null), [srcTable, ops]);
  const prof = useMemo(() => (srcTable ? profile(srcTable) : []), [srcTable]);
  const selIdx = ops.findIndex((o) => o.id === sel);
  const inCols = srcTable ? (selIdx >= 0 && pipe ? pipe.stages[selIdx].table.cols : srcTable.cols) : [];

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

  function buildCode(): string {
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
        case "bucket": return `df = df.withColumn("${o.name}", (((F.col("${o.col}") - F.min("${o.col}").over(W)) / (F.max("${o.col}").over(W) - F.min("${o.col}").over(W))) * ${parseInt(o.value || "4") || 4}).cast("int"))  # or use pyspark.ml.feature.Bucketizer`;
      }
    });
    return `# AI Workbench · ETL pipeline (PySpark)  ·  ${mode === "stream" ? "structured streaming" : "batch"}
from pyspark.sql import SparkSession, functions as F
spark = SparkSession.builder.appName("workbench_etl").getOrCreate()

${mode === "stream"
        ? `df = (spark.readStream.format("kafka").option("subscribe", "orders").load())  # extract`
        : `df = spark.read.option("header", True).option("inferSchema", True).csv("source.csv")  # extract`}

${lines.join("\n") || "# (no transforms yet)"}

${mode === "stream"
        ? `query = df.writeStream.format("console").outputMode("complete").start()\nquery.awaitTermination()`
        : `df.show()\ndf.write.mode("overwrite").parquet("output/")   # load`}`;
  }
  function copyCode() { navigator.clipboard.writeText(buildCode()).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500); }); }
  async function saveProject() {
    const config = { srcName, mode, ops };
    try { const r = await fetch("/api/projects", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ lab: "etl", name: srcName || "ETL pipeline", config }) }); setSavedMsg(r.ok ? "Saved ✓" : "Save failed"); }
    catch { setSavedMsg("Save failed"); }
    setTimeout(() => setSavedMsg(""), 2000);
  }
  function exportCsv() { if (!pipe) return; const blob = new Blob([toCSV(pipe.final)], { type: "text/csv" }); const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "etl_output.csv"; a.click(); URL.revokeObjectURL(a.href); }

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

  return (
    <>
      <div className="lab-head">
        <div><div className="eyebrow">Lab 06 · data engineering</div><h2 className="page-h">ETL Lab</h2><p className="page-sub" style={{ margin: 0 }}>Extract from any source, transform with a Kafka + Spark-style DAG, then load — and watch records flow.</p></div>
        <div className="acts"><button className="btn ghost sm" onClick={saveProject}>{savedMsg || "💾 Save"}</button><button className="btn ghost sm" onClick={() => setShowCode(true)}>&lt;/&gt; Get code (PySpark)</button></div>
      </div>
      {msg && <div className="err">{msg}</div>}
      <div className="teach-note"><span className="ic">🎓</span><span><b>Teaching engine.</b> Transforms run for real on your rows in the browser so you can watch each stage. For production scale, use the <b>Get code (PySpark)</b> export.</span></div>

      <div className="stepper">
        <button className={step === "extract" ? "on" : ""} onClick={() => setStep("extract")}><b>1</b>Extract</button>
        <button className={step === "transform" ? "on" : ""} disabled={!srcTable} onClick={() => setStep("transform")}><b>2</b>Transform</button>
        <button className={step === "load" ? "on" : ""} disabled={!srcTable} onClick={() => setStep("load")}><b>3</b>Load &amp; Run</button>
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
                <div className="rangebtns" style={{ marginLeft: "auto" }}><button className={viewMode === "head" ? "on" : ""} onClick={() => setViewMode("head")}>Head</button><button className={viewMode === "tail" ? "on" : ""} onClick={() => setViewMode("tail")}>Tail</button></div>
              </div>
              <div style={{ overflowX: "auto" }}><table className="dtable"><tbody>
                <tr><th>#</th>{srcTable.cols.map((c) => { const p = prof.find((x) => x.name === c)!; return <th key={c}>{c} <span style={{ color: "var(--faint)" }}>{p.type}{p.nulls ? `·${p.nulls}na` : ""}</span></th>; })}</tr>
                {(viewMode === "head" ? srcTable.rows.slice(0, 8) : srcTable.rows.slice(-8)).map((r, i) => <tr key={i}><td style={{ color: "var(--faint)" }}>{viewMode === "head" ? i : srcTable.rows.length - 8 + i}</td>{srcTable.cols.map((c) => <td key={c}>{r[c] == null ? "—" : String(r[c])}</td>)}</tr>)}
              </tbody></table></div>
              <div className="stepnav"><button className="btn" onClick={() => setStep("transform")}>Next: Transform →</button></div>
            </>)}
          </div>
        </div>
      )}

      {/* STEP 2 — TRANSFORM */}
      {step === "transform" && srcTable && (
        <div className="split" style={{ gridTemplateColumns: "1fr 300px" }}>
          <div className="card">
            <div className="card-h"><span className="t">Transform — build the DAG</span>
              <div className="r" style={{ position: "relative" }}>
                <button className="btn ghost sm" onClick={() => setAddOpen((o) => !o)}>+ Add transform</button>
                {addOpen && <div className="addmenu2"><div className="hd">Add a transform</div>{(Object.keys(OP_META) as OpType[]).map((t) => <div key={t} className="ai" onClick={() => addOp(t)}><span>{OP_META[t].icon}</span>{OP_META[t].label}</div>)}</div>}
              </div>
            </div>
            <div className="card-b" style={{ padding: 0 }}>{dagCanvas(true)}</div>
            <div className="card-b" style={{ borderTop: "1px solid var(--border)" }}>
              <label className="fld">Live preview — sink output ({pipe!.final.rows.length} rows)</label>
              <div style={{ overflowX: "auto", maxHeight: 180, overflowY: "auto" }}><table className="dtable"><tbody><tr>{pipe!.final.cols.map((c) => <th key={c}>{c}</th>)}</tr>{pipe!.final.rows.slice(0, 6).map((r, i) => <tr key={i}>{pipe!.final.cols.map((c) => <td key={c}>{r[c] == null ? "—" : String(r[c])}</td>)}</tr>)}</tbody></table></div>
              <div className="stepnav"><button className="btn ghost" onClick={() => setStep("extract")}>← Extract</button><button className="btn" onClick={() => setStep("load")}>Next: Load &amp; Run →</button></div>
            </div>
          </div>
          <div className="card">
            <div className="card-h"><span className="t">Configure</span><span className="mono r">{selIdx >= 0 ? OP_META[ops[selIdx].type].label : "—"}</span></div>
            <div className="card-b" style={{ maxHeight: 430, overflow: "auto" }}>
              {selIdx < 0 && <div className="note">Add a transform, then click its node to configure it.</div>}
              {selIdx >= 0 && (() => {
                const o = ops[selIdx];
                const colSel = (val: string | undefined, on: (v: string) => void) => <select value={val} onChange={(e) => on(e.target.value)}>{inCols.map((c) => <option key={c}>{c}</option>)}</select>;
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
                  <button className="btn ghost sm" onClick={() => removeOp(o.id)}>Remove transform</button>
                </>);
              })()}
            </div>
          </div>
        </div>
      )}

      {/* STEP 3 — LOAD & RUN */}
      {step === "load" && srcTable && pipe && (
        <>
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
              </div>
              {dagCanvas(false)}
            </div>
          </div>
          <div className="card" style={{ marginTop: 16 }}>
            <div className="card-h"><span className="t">Sink output</span><div className="r"><span className="mono">{pipe.final.rows.length} × {pipe.final.cols.length}</span><button className="btn ghost sm" onClick={exportCsv}>⬇ Export CSV</button></div></div>
            <div className="card-b">
              <div style={{ overflowX: "auto" }}><table className="dtable"><tbody>
                <tr>{pipe.final.cols.map((c) => <th key={c}>{c}</th>)}</tr>
                {pipe.final.rows.slice(0, 14).map((r, i) => <tr key={i}>{pipe.final.cols.map((c) => <td key={c}>{r[c] == null ? "—" : String(r[c])}</td>)}</tr>)}
              </tbody></table></div>
              {pipe.final.rows.length > 14 && <div className="note" style={{ marginTop: 8 }}>+ {pipe.final.rows.length - 14} more rows</div>}
              <div className="stepnav"><button className="btn ghost" onClick={() => setStep("transform")}>← Transform</button></div>
            </div>
          </div>
        </>
      )}

      <div className={`modal-wrap ${showCode ? "show" : ""}`} onClick={(e) => { if (e.target === e.currentTarget) setShowCode(false); }}>
        <div className="modal"><div className="mh"><b>ETL pipeline · PySpark</b><div className="r" style={{ marginLeft: "auto", display: "flex", gap: 8 }}><button className="btn ghost sm" onClick={copyCode}>{copied ? "Copied ✓" : "Copy"}</button></div><button className="x" onClick={() => setShowCode(false)}>×</button></div><div className="mb"><div className="code">{buildCode()}</div></div></div>
      </div>
    </>
  );
}
