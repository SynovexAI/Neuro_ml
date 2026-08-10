"use client";

// ETL Lab — node-canvas edition. Project-2-style React Flow canvas driven by our
// guided, step-gated flow (Extract → Transform → Load → Analytics). Click a node to
// configure its real function; the output panel always shows the live result.
// Runs on the real engine in etlUtils; reference cards come from etlNodeInfo.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ReactFlow, ReactFlowProvider, Background, BackgroundVariant, Controls, MiniMap,
  Handle, Position, addEdge, useNodesState, useEdgesState, useReactFlow, MarkerType,
  type Node, type Edge, type Connection, type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import {
  parseRecords, sampleSources, runPipeline, toCSV, tableFromRecords, OP_META,
  profile, evaluate, lineage, RULE_META, ACTION_META,
  type Table, type EtlOp, type OpType, type Expectation, type RuleType, type RuleAction,
} from "@/lib/etlUtils";
import { toPython, type CodegenSpec } from "@/lib/etlCodegen";
import { runSql } from "@/lib/sqlEngine";
import { toast, confirmDialog } from "@/lib/toast";
import { OP_INFO, SRC_INFO, LOAD_INFO } from "@/lib/etlNodeInfo";
import DataVizExplorer from "./visualization/DataVizExplorer";

type Mode = "etl" | "elt" | "streaming" | "reverse";
type Kind = "source" | "op" | "load" | "analytics" | "sql";
const DEFAULT_SQL = "SELECT region,\n       COUNT(*) AS orders,\n       ROUND(SUM(amount), 2) AS revenue\nFROM raw\nWHERE status = 'paid'\nGROUP BY region\nORDER BY revenue DESC";
const PLAY_CAP = 24; // max input rows animated per node
const FILTER_LIKE = new Set<OpType>(["filter", "clean", "dedupe", "limit", "sample"]);
type Cat = "extraction" | "integration" | "transformation" | "loading" | "streaming" | "analytics";

const CATCOLOR: Record<Cat, string> = {
  extraction: "#3b82f6", integration: "#14b8a6", transformation: "#a855f7",
  loading: "#10b981", streaming: "#f59e0b", analytics: "#ec4899",
};
const CATLABEL: Record<Cat, string> = {
  extraction: "Extraction", integration: "Integration", transformation: "Transformation",
  loading: "Loading", streaming: "Streaming", analytics: "Analytics",
};
const rid = () => Math.random().toString(36).slice(2, 9);

// ── palette per mode + step (the step-gating) ──
type PalItem = { key: string; label: string; icon: string; cat: Cat; kind: Kind; opType?: OpType; srcType?: string; target?: string };
const SOURCES: PalItem[] = [
  { key: "s-sample", label: "Sample / CSV", icon: "📄", cat: "extraction", kind: "source", srcType: "csv" },
  { key: "s-excel", label: "Excel", icon: "📗", cat: "extraction", kind: "source", srcType: "excel" },
  { key: "s-pg", label: "PostgreSQL", icon: "🐘", cat: "extraction", kind: "source", srcType: "postgres" },
  { key: "s-mysql", label: "MySQL / TiDB", icon: "🗄", cat: "extraction", kind: "source", srcType: "mysql" },
  { key: "s-rest", label: "REST / API", icon: "🌐", cat: "extraction", kind: "source", srcType: "rest" },
  { key: "s-json", label: "JSON", icon: "🧾", cat: "extraction", kind: "source", srcType: "json" },
  { key: "s-mongo", label: "MongoDB", icon: "🍃", cat: "extraction", kind: "source", srcType: "mongodb" },
  { key: "s-api", label: "REST API", icon: "🌐", cat: "extraction", kind: "source", srcType: "rest" },
  { key: "s-s3", label: "Amazon S3", icon: "🪣", cat: "extraction", kind: "source", srcType: "s3" },
];
const STREAM_SOURCES: PalItem[] = [
  { key: "s-kafka", label: "Kafka", icon: "📡", cat: "streaming", kind: "source", srcType: "kafka" },
  { key: "s-rabbit", label: "RabbitMQ", icon: "📨", cat: "streaming", kind: "source", srcType: "rabbitmq" },
];
const opItem = (t: OpType, cat: Cat = "transformation"): PalItem => ({ key: "op-" + t, label: OP_META[t].label, icon: OP_META[t].icon, cat, kind: "op", opType: t });
const TRANSFORMS: PalItem[] = [
  ...(["filter", "select", "derive", "aggregate", "sort", "dedupe", "clean", "rename", "limit", "sample", "map", "fillna", "bucket", "pivot", "unpivot", "window", "regex", "dateparse"] as OpType[]).map((t) => opItem(t, "transformation")),
  opItem("join", "integration"), opItem("union", "integration"),
  // New integration ops
  opItem("lookup", "integration"),
  opItem("merge", "integration"),
  opItem("append", "integration"),
];
const LOADS: PalItem[] = [
  { key: "l-internal", label: "Platform DB", icon: "🗃", cat: "loading", kind: "load", target: "platform" },
  { key: "l-pg", label: "PostgreSQL", icon: "🐘", cat: "loading", kind: "load", target: "postgres" },
  { key: "l-mysql", label: "MySQL / TiDB", icon: "🗄", cat: "loading", kind: "load", target: "mysql" },
  { key: "l-snow", label: "Snowflake", icon: "❄", cat: "loading", kind: "load", target: "snowflake" },
  { key: "l-bq", label: "BigQuery", icon: "📊", cat: "loading", kind: "load", target: "bigquery" },
  { key: "l-mongo", label: "MongoDB", icon: "🍃", cat: "loading", kind: "load", target: "mongodb" },
  { key: "l-redshift", label: "Redshift", icon: "☁", cat: "loading", kind: "load", target: "redshift" },
  { key: "l-csv", label: "CSV file", icon: "📄", cat: "loading", kind: "load", target: "csv" },
  { key: "l-xlsx", label: "Excel file", icon: "📗", cat: "loading", kind: "load", target: "xlsx" },
];
const ANALYTICS: PalItem[] = [
  { key: "a-dash", label: "Dashboard", icon: "📈", cat: "analytics", kind: "analytics", target: "dashboard" },
];
const SQL_ITEM: PalItem = { key: "sql-model", label: "SQL model", icon: "🧮", cat: "transformation", kind: "sql" };

type ExtractionField = { id: string; label: string; type: "text" | "password" | "number" | "textarea" | "select"; placeholder?: string; required?: boolean; options?: string[]; tooltip?: string; };
type ExtractionSource = { name: string; category: "Database" | "File" | "Streaming" | "API" | "Cloud Storage"; difficulty: "Beginner" | "Intermediate" | "Advanced"; apiKeyRequired: boolean; oauthRequired: boolean; fields: ExtractionField[]; };

const extractionConfig: Record<string, ExtractionSource> = {
  csv: { name: "CSV", category: "File", difficulty: "Beginner", apiKeyRequired: false, oauthRequired: false, fields: [] },
  excel: { name: "Excel", category: "File", difficulty: "Beginner", apiKeyRequired: false, oauthRequired: false, fields: [{ id: "sheet", label: "Sheet selector (optional)", type: "text", placeholder: "e.g., Sheet1" }] },
  json: { name: "JSON", category: "File", difficulty: "Beginner", apiKeyRequired: false, oauthRequired: false, fields: [{ id: "jsonUrl", label: "JSON URL (optional)", type: "text", placeholder: "https://api.example.com/data.json" }] },
  mysql: { name: "MySQL", category: "Database", difficulty: "Intermediate", apiKeyRequired: false, oauthRequired: false, fields: [
    { id: "dbHost", label: "Host", type: "text", required: true, placeholder: "db.example.com" },
    { id: "dbPort", label: "Port", type: "number", required: true, placeholder: "3306" },
    { id: "dbName", label: "Database name", type: "text", required: true, placeholder: "production_db" },
    { id: "dbUser", label: "Username", type: "text", required: true },
    { id: "dbPass", label: "Password", type: "password", required: true },
    { id: "dbTable", label: "Table name", type: "text", required: true }
  ] },
  postgres: { name: "PostgreSQL", category: "Database", difficulty: "Intermediate", apiKeyRequired: false, oauthRequired: false, fields: [
    { id: "dbHost", label: "Host", type: "text", required: true, placeholder: "db.example.com" },
    { id: "dbPort", label: "Port", type: "number", required: true, placeholder: "5432" },
    { id: "dbName", label: "Database", type: "text", required: true, placeholder: "production_db" },
    { id: "dbUser", label: "Username", type: "text", required: true },
    { id: "dbPass", label: "Password", type: "password", required: true },
    { id: "dbSchema", label: "Schema", type: "text", required: true, placeholder: "public" },
    { id: "dbTable", label: "Table", type: "text", required: true }
  ] },
  mongodb: { name: "MongoDB", category: "Database", difficulty: "Intermediate", apiKeyRequired: false, oauthRequired: false, fields: [
    { id: "mongoConn", label: "Connection string", type: "text", required: true, placeholder: "mongodb+srv://user:pass@cluster..." },
    { id: "mongoDb", label: "Database", type: "text", required: true },
    { id: "mongoColl", label: "Collection", type: "text", required: true }
  ] },
  kafka: { name: "Kafka", category: "Streaming", difficulty: "Advanced", apiKeyRequired: false, oauthRequired: false, fields: [
    { id: "kafkaBroker", label: "Broker URL", type: "text", required: true, placeholder: "broker:9092" },
    { id: "kafkaTopic", label: "Topic", type: "text", required: true },
    { id: "kafkaGroup", label: "Consumer group", type: "text", required: true }
  ] },
  rabbitmq: { name: "RabbitMQ", category: "Streaming", difficulty: "Intermediate", apiKeyRequired: false, oauthRequired: false, fields: [
    { id: "rabbitUrl", label: "Queue URL", type: "text", required: true, placeholder: "amqp://localhost" },
    { id: "rabbitQueue", label: "Queue name", type: "text", required: true }
  ] },
  rest: { name: "REST API", category: "API", difficulty: "Beginner", apiKeyRequired: true, oauthRequired: false, fields: [
    { id: "apiUrl", label: "Endpoint URL", type: "text", required: true, placeholder: "https://api.domain.com/v1/data", tooltip: "The full URL endpoint to fetch data from." },
    { id: "apiMethod", label: "HTTP Method", type: "select", options: ["GET", "POST"], required: true },
    { id: "apiHeaders", label: "Headers (JSON)", type: "textarea", placeholder: '{"Authorization": "Bearer token"}' },
    { id: "apiKey", label: "API Key (optional)", type: "password" },
    { id: "apiParams", label: "Query parameters (JSON)", type: "textarea", placeholder: '{"limit": 100}' }
  ] },
  s3: { name: "Amazon S3", category: "Cloud Storage", difficulty: "Intermediate", apiKeyRequired: true, oauthRequired: false, fields: [
    { id: "s3Access", label: "Access key", type: "text", required: true },
    { id: "s3Secret", label: "Secret key", type: "password", required: true },
    { id: "s3Bucket", label: "Bucket name", type: "text", required: true },
    { id: "s3Region", label: "Region", type: "text", required: true, placeholder: "us-east-1" },
    { id: "s3Folder", label: "Folder path", type: "text", placeholder: "data/logs/" }
  ] }
};

type StepDef = { label: string; items: PalItem[] };
const MODES: Record<Mode, { accent: string; label: string; steps: StepDef[] }> = {
  etl: { accent: "#5b7cff", label: "ETL", steps: [
    { label: "Extract", items: SOURCES },
    { label: "Transform", items: TRANSFORMS },
    { label: "Load", items: [...LOADS, ...ANALYTICS] },
  ] },
  elt: { accent: "#a855f7", label: "ELT", steps: [
    { label: "Extract", items: SOURCES },
    { label: "Load raw", items: LOADS },
    { label: "Transform (SQL)", items: [SQL_ITEM, ...TRANSFORMS] },
  ] },
  streaming: { accent: "#f59e0b", label: "Streaming", steps: [
    { label: "Stream source", items: STREAM_SOURCES },
    { label: "Process", items: TRANSFORMS },
    { label: "Sink", items: [...LOADS, ...ANALYTICS] },
  ] },
  reverse: { accent: "#ec4899", label: "Reverse ETL", steps: [
    { label: "Warehouse", items: LOADS },
    { label: "Map & model", items: TRANSFORMS },
    { label: "Sync to app", items: LOADS },
  ] },
};

// Default config for a freshly-added op node (mirrors the classic lab's addOp).
function makeOp(type: OpType, cols: string[], rows: Record<string, unknown>[], bCols: string[]): EtlOp {
  const numCol = cols[cols.findIndex((c) => rows.some((r) => typeof r[c] === "number"))] || cols[0] || "";
  const base = { id: rid(), type } as EtlOp;
  const a = Object.assign as <T>(t: T, s: Partial<EtlOp>) => T;
  if (type === "filter") a(base, { col: numCol, op: ">", value: "0" });
  else if (type === "select") a(base, { cols: cols.slice(0, Math.min(3, cols.length)) });
  else if (type === "derive") a(base, { name: "derived", left: numCol, arith: "*", right: "1" });
  else if (type === "aggregate") a(base, { groupBy: cols[0], agg: "count", aggCol: numCol });
  else if (type === "sort") a(base, { col: numCol, dir: "desc" });
  else if (type === "dedupe") a(base, { cols: [] });
  else if (type === "clean") a(base, { mode: "dropnull" });
  else if (type === "rename") a(base, { col: cols[0], name: (cols[0] || "col") + "_new" });
  else if (type === "limit") a(base, { value: "10" });
  else if (type === "sample") a(base, { value: "0.5" });
  else if (type === "map") a(base, { col: numCol, fn: "round" });
  else if (type === "fillna") a(base, { col: "", value: "0" });
  else if (type === "bucket") a(base, { col: numCol, name: numCol + "_bin", value: "4" });
  else if (type === "join" || type === "lookup" || type === "merge") a(base, { joinType: type === "join" ? "inner" : "left", col: cols[0], rightKey: bCols[0] || cols[0] });
  else if (type === "union" || type === "append") a(base, { mode: "all" });
  else if (type === "pivot") a(base, { groupBy: cols[0], col: cols[1] || cols[0], agg: "sum", aggCol: numCol });
  else if (type === "unpivot") a(base, { cols: [numCol], name: "variable", value: "value" });
  else if (type === "window") a(base, { groupBy: "(none)", col: numCol, fn: "running_sum", name: "running_sum" });
  else if (type === "regex") a(base, { col: cols[0], value: "(\\d+)", name: (cols[0] || "col") + "_match" });
  else if (type === "dateparse") a(base, { col: cols[0], fn: "year", name: (cols[0] || "col") + "_year" });
  else if (type === "scd2") a(base, { businessKey: cols[0] });
  else if (type === "fuzzydedupe") a(base, { col: cols[0], threshold: 0.8 });
  else if (type === "quality") a(base, { qualityCol: cols[0], rule: "not_null", name: "_quality_status" });
  return base;
}
function opSummary(o: EtlOp): string {
  switch (o.type) {
    case "filter": return `${o.col} ${o.op} ${o.value ?? ""}`;
    case "select": return (o.cols || []).join(", ").slice(0, 20) || "all";
    case "aggregate": return `${o.agg} by ${o.groupBy}`;
    case "sort": return `${o.col} ${o.dir}`;
    case "join": return `${o.joinType} on ${o.col}`;
    case "derive": return `${o.name}`;
    default: return OP_META[o.type].label.toLowerCase();
  }
}

// ── node data + custom node renderer ──
type NData = {
  kind: Kind; label: string; cat: Cat; icon: string; color: string;
  op?: EtlOp; table?: Table; srcType?: string; srcName?: string; target?: string;
  dbUrl?: string; dbQuery?: string; jsonText?: string; sqlText?: string;
  extUrl?: string; extTable?: string; extMode?: string;
  extHost?: string; extPort?: string; extDb?: string; extUser?: string; extPass?: string;
  restUrl?: string; restPath?: string;
  count?: number | null; run?: "running" | "done";
};
type FNode = Node<NData>;
type SavedWF = { mode?: Mode; nodes?: { id: string; position?: { x: number; y: number }; data: NData }[]; edges?: { id?: string; source: string; target: string }[] };
type PlayNode = { label: string; rule: string; input: Table; steps: number; capped: boolean; outAt: (k: number) => Table; state: (i: number, cur: number) => string; cap: (cur: number) => string; hlOut?: boolean };

function FlowNode({ id, data, selected }: NodeProps<FNode>) {
  const { setNodes, setEdges } = useReactFlow();
  const [hover, setHover] = useState(false);
  const c = data.color;
  const running = data.run === "running", done = data.run === "done";
  const border = running ? "#f59e0b" : done ? "#3ecf7f" : selected ? c : "var(--border)";
  const shadow = running ? "0 0 18px rgba(245,158,11,.45)" : selected ? `0 0 0 3px color-mix(in srgb, ${c} 30%, transparent)` : "var(--shadow-sm)";
  return (
    <div className="flow-node-wrapper" onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)} style={{ width: 158, borderRadius: 12, border: `1px solid ${border}`, background: "var(--panel)", boxShadow: shadow, padding: "10px 11px", position: "relative", transition: "box-shadow .2s, border-color .2s" }}>
      <button 
        className="node-info-btn"
        onClick={(e) => {
          e.stopPropagation();
          window.dispatchEvent(new CustomEvent("open-node-info", { detail: id }));
        }}
        style={{
          position: "absolute", top: -8, left: -8, width: 22, height: 22, borderRadius: "50%",
          background: "var(--surface)", color: "var(--text)", border: "1px solid var(--border)",
          boxShadow: "0 0 0 4px #0e121d, 0 2px 6px rgba(0,0,0,0.3)", display: "flex", alignItems: "center", justifyContent: "center", 
          cursor: "pointer", zIndex: 10, opacity: hover ? 1 : 0, transform: hover ? "scale(1)" : "scale(0.8)", 
          transition: "all 0.15s cubic-bezier(0.4, 0, 0.2, 1)", pointerEvents: hover ? "auto" : "none"
        }}
        onMouseEnter={(e) => { e.currentTarget.style.color = "var(--accent)"; e.currentTarget.style.borderColor = "var(--accent)"; }}
        onMouseLeave={(e) => { e.currentTarget.style.color = "var(--text)"; e.currentTarget.style.borderColor = "var(--border)"; }}
        title="View details"
      >
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
      </button>
      <button 
        className="node-del-btn"
        onClick={(e) => {
          e.stopPropagation();
          setNodes((nds) => nds.filter((n) => n.id !== id));
          setEdges((eds) => eds.filter((edge) => edge.source !== id && edge.target !== id));
        }}
        style={{
          position: "absolute", top: -8, right: -8, width: 22, height: 22, borderRadius: "50%",
          background: "var(--surface)", color: "var(--text)", border: "1px solid var(--border)",
          boxShadow: "0 0 0 4px #0e121d, 0 2px 6px rgba(0,0,0,0.3)", display: "flex", alignItems: "center", justifyContent: "center", 
          cursor: "pointer", zIndex: 10, opacity: hover ? 1 : 0, transform: hover ? "scale(1)" : "scale(0.8)", 
          transition: "all 0.15s cubic-bezier(0.4, 0, 0.2, 1)", pointerEvents: hover ? "auto" : "none"
        }}
        onMouseEnter={(e) => { e.currentTarget.style.color = "#ef4444"; e.currentTarget.style.borderColor = "#ef4444"; }}
        onMouseLeave={(e) => { e.currentTarget.style.color = "var(--text)"; e.currentTarget.style.borderColor = "var(--border)"; }}
        title="Remove node"
      >
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
      </button>
      {/* Row count badge (Extract -> Transform -> Load) */}
      {data.count != null && (
        <div
          className="node-count-badge"
          title={`${data.count.toLocaleString()} rows in node output`}
          style={{
            position: "absolute",
            top: -9,
            right: hover ? 18 : -6,
            minWidth: 20,
            height: 19,
            padding: "0 6px",
            borderRadius: 10,
            background: done ? "rgba(16,185,129,0.25)" : "var(--surface)",
            border: `1px solid ${done ? "#10b981" : "var(--border-strong)"}`,
            color: done ? "#3ecf7f" : "var(--text)",
            fontSize: 10,
            fontFamily: "var(--mono)",
            fontWeight: 700,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 9,
            boxShadow: "0 2px 6px rgba(0,0,0,0.4)",
            transition: "all 0.15s cubic-bezier(0.4, 0, 0.2, 1)",
            pointerEvents: "none",
          }}
        >
          {data.count.toLocaleString()}
        </div>
      )}

      {data.kind !== "source" && <Handle type="target" position={Position.Left} style={{ background: "#0a0d17", border: `2px solid ${c}`, width: 9, height: 9 }} />}
      {data.kind !== "analytics" && <Handle type="source" position={Position.Right} style={{ background: "#0a0d17", border: `2px solid ${c}`, width: 9, height: 9 }} />}
      <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
        <span style={{ width: 30, height: 30, borderRadius: 8, display: "grid", placeItems: "center", fontSize: 15, flex: "0 0 auto", background: `color-mix(in srgb, ${c} 18%, transparent)`, color: c }}>{data.icon}</span>
        <div style={{ minWidth: 0 }}>
          <div style={{ fontSize: 12.5, fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 96 }}>{data.label}</div>
          <div style={{ fontSize: 9, color: c, textTransform: "uppercase", letterSpacing: ".04em", marginTop: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 96 }}>{data.op ? opSummary(data.op) : data.cat}</div>
        </div>
      </div>
    </div>
  );
}
const nodeTypes = { flow: FlowNode };

function Inner() {
  const [mode, setMode] = useState<Mode>("etl");
  const [step, setStep] = useState(0);
  const [nodes, setNodes, onNodesChange] = useNodesState<FNode>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [selId, setSelId] = useState<string | null>(null);
  const [runMsg, setRunMsg] = useState("");
  const [runStatus, setRunStatus] = useState<Record<string, "running" | "done">>({});
  const [metrics, setMetrics] = useState<{ inn: number; out: number; ms: number; batches?: number; thr?: number } | null>(null);
  const [running, setRunning] = useState(false);
  const [dashOpen, setDashOpen] = useState(false);
  const [playOpen, setPlayOpen] = useState(false);
  const [savedMsg, setSavedMsg] = useState("");
  const [projects, setProjects] = useState<{ id: string; name: string }[]>([]);
  const [loadOpen, setLoadOpen] = useState(false);
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  // Deploy-as-API state
  const [deploy, setDeploy] = useState<{ open: boolean; busy: boolean; key?: string; chId?: string; err?: string; mode?: string; hasTarget?: boolean } | null>(null);
  const [apiLang, setApiLang] = useState<"curl" | "js" | "python">("curl");
  const [runs, setRuns] = useState<{ id: string; name: string | null; mode: string | null; target: string | null; rowsOut: number; rowsLoaded: number; durationMs: number; status: string | null; ts: string | null }[]>([]);
  const [showQuality, setShowQuality] = useState(false);
  const [showRuns, setShowRuns] = useState(false);
  const [rules, setRules] = useState<Expectation[]>([]);
  const [renameText, setRenameText] = useState("");
  const runTimers = useRef<ReturnType<typeof setTimeout>[]>([]);
  const fileRef = useRef<HTMLInputElement>(null);
  const importRef = useRef<HTMLInputElement>(null);
  const { screenToFlowPosition } = useReactFlow();
  const accent = MODES[mode].accent;
  
  const [infoNodeId, setInfoNodeId] = useState<string | null>(null);
  useEffect(() => {
    const h = (e: Event) => setInfoNodeId((e as CustomEvent<string>).detail);
    window.addEventListener("open-node-info", h);
    return () => window.removeEventListener("open-node-info", h);
  }, []);

  // ── compute the live pipeline over the current chain (ordered left→right) ──
  const sources = useMemo(() => nodes.filter((n) => n.data.kind === "source" && n.data.table), [nodes]);
  const primary = sources[0]?.data.table || null;
  const secondary = sources[1]?.data.table || null;
  const opNodes = useMemo(() => nodes.filter((n) => n.data.kind === "op").sort((a, b) => a.position.x - b.position.x), [nodes]);
  const pipe = useMemo(() => (primary ? runPipeline(primary, opNodes.map((n) => n.data.op!).filter(Boolean), { secondary }) : null), [primary, secondary, opNodes]);
  const sqlNode = useMemo(() => nodes.find((n) => n.data.kind === "sql" && n.data.table), [nodes]);
  const finalTable = sqlNode?.data.table || pipe?.final || primary;
  const activeTable = useMemo(() => {
    if (!selId) return finalTable;
    const s = nodes.find((n) => n.id === selId);
    if (!s) return finalTable;
    if (s.data.kind === "source" || s.data.kind === "sql") return s.data.table || finalTable;
    if (s.data.kind === "op" && pipe) {
      const idx = opNodes.findIndex((n) => n.id === selId);
      if (idx >= 0 && pipe.stages[idx + 1]) return pipe.stages[idx + 1].table;
    }
    return finalTable;
  }, [selId, nodes, finalTable, pipe, opNodes]);

  // Node-by-node execution model for the Play walkthrough. Each node's input is
  // the previous node's output; partial output per row is real (runPipeline on a
  // row prefix), so the animation shows the actual transform executing.
  const playNodes = useMemo<PlayNode[]>(() => {
    if (!pipe || !primary) return [];
    const arr: PlayNode[] = [];
    opNodes.forEach((n, i) => {
      const op = n.data.op!;
      const input = pipe.stages[i].table;
      const rows = input.rows.slice(0, PLAY_CAP);
      const inp: Table = { cols: input.cols, rows };
      const steps = rows.map((_, k) => runPipeline({ cols: input.cols, rows: rows.slice(0, k + 1) }, [op], { secondary }).final);
      const counts = steps.map((s) => s.rows.length);
      const delta = (k: number) => counts[k] - (k > 0 ? counts[k - 1] : 0);
      const state = (idx: number, cur: number): string => {
        if (idx === cur) return "cur";
        if (idx > cur) return "pend";
        return FILTER_LIKE.has(op.type) ? (delta(idx) > 0 ? "keep" : "drop") : "keep";
      };
      const cap = (cur: number): string => {
        if (cur < 0) return `Node ${i + 1} · ${OP_META[op.type].label}. Input = ${i === 0 ? "the raw source" : "the output of " + OP_META[opNodes[i - 1].data.op!.type].label}.`;
        if (cur >= rows.length) return `${OP_META[op.type].label} done → ${pipe.stages[i + 1].table.rows.length} rows.`;
        const r = rows[cur];
        switch (op.type) {
          case "filter": return `Row ${cur + 1}: ${op.col} ${op.op} ${op.value ?? ""} → ${delta(cur) > 0 ? "✓ keep" : "✗ drop"}`;
          case "clean": return `Row ${cur + 1}: ${delta(cur) > 0 ? "✓ no nulls · keep" : "✗ null found · drop"}`;
          case "aggregate": return `Row ${cur + 1}: fold "${String(r[op.groupBy ?? ""] ?? "")}" into its ${op.groupBy} group`;
          case "map": return `Row ${cur + 1}: apply ${op.fn}(${op.col})`;
          case "derive": return `Row ${cur + 1}: compute ${op.name}`;
          case "join": return `Row ${cur + 1}: match on ${op.col} = ${op.rightKey}`;
          case "sort": return `Row ${cur + 1}: place by ${op.col} (${op.dir})`;
          case "dedupe": return `Row ${cur + 1}: ${delta(cur) > 0 ? "✓ first seen · keep" : "✗ duplicate · drop"}`;
          case "limit": return `Row ${cur + 1}: ${delta(cur) > 0 ? "✓ within limit" : "✗ beyond limit · drop"}`;
          default: return `Row ${cur + 1} → produces output`;
        }
      };
      arr.push({ label: OP_META[op.type].label, rule: opSummary(op), input: inp, steps: rows.length, capped: input.rows.length > PLAY_CAP, outAt: (k) => (k < 0 ? { cols: steps[0]?.cols || input.cols, rows: [] } : steps[Math.min(k, steps.length - 1)]), state, cap });
    });
    if (sqlNode?.data.table) {
      const outT = sqlNode.data.table; const orows = outT.rows.slice(0, PLAY_CAP);
      arr.push({ label: "SQL model", rule: "in-browser SQL", input: { cols: primary.cols, rows: primary.rows.slice(0, PLAY_CAP) }, steps: orows.length, capped: outT.rows.length > PLAY_CAP, outAt: (k) => ({ cols: outT.cols, rows: k < 0 ? [] : orows.slice(0, k + 1) }), state: () => "proc", cap: (cur) => (cur < 0 ? "SQL model — runs over the raw source." : cur >= orows.length ? "SQL done." : `SQL produces output row ${cur + 1}`), hlOut: true });
    }
    return arr;
  }, [pipe, primary, opNodes, sqlNode, secondary]);

  // counts per node id, for the badge
  const counts = useMemo(() => {
    const m: Record<string, number> = {};
    sources.forEach((s) => { if (s.data.table) m[s.id] = s.data.table.rows.length; });
    opNodes.forEach((n, i) => { if (pipe?.stages[i + 1]?.table) m[n.id] = pipe.stages[i + 1].table.rows.length; });
    nodes.filter((n) => n.data.kind === "load" || n.data.kind === "analytics" || n.data.kind === "sql").forEach((n) => {
      if (n.data.table) {
        m[n.id] = n.data.table.rows.length;
      } else {
        const inEdge = edges.find((e) => e.target === n.id);
        if (inEdge && m[inEdge.source] != null) {
          m[n.id] = m[inEdge.source];
        } else if (finalTable) {
          m[n.id] = finalTable.rows.length;
        }
      }
    });
    return m;
  }, [sources, opNodes, pipe, nodes, finalTable, edges]);

  const displayNodes = useMemo(() => nodes.map((n) => ({ ...n, data: { ...n.data, count: counts[n.id] ?? null, run: runStatus[n.id] } })), [nodes, counts, runStatus]);
  const sel = nodes.find((n) => n.id === selId) || null;
  const cols = primary?.cols || [];

  const onConnect = useCallback((c: Connection) => setEdges((eds) => addEdge({ ...c, animated: true, style: { stroke: accent, strokeWidth: 2 }, markerEnd: { type: MarkerType.ArrowClosed, color: accent } }, eds)), [setEdges, accent]);

  // add a palette item to the canvas, auto-chaining an animated edge from the previous node.
  // `pos` (flow coords) is passed on drag-drop; otherwise we lay it out to the right.
  function addFromPalette(it: PalItem, pos?: { x: number; y: number }) {
    const id = rid();
    const color = CATCOLOR[it.cat];
    const data: NData = { kind: it.kind, label: it.label, cat: it.cat, icon: it.icon, color };
    if (it.kind === "source") {
      data.srcType = it.srcType;
      if (it.srcType === "csv") { const s = sampleSources()[0]; data.table = parseRecords(s.csv); data.srcName = s.label; data.label = s.label.slice(0, 14); }
    } else if (it.kind === "op") {
      data.op = makeOp(it.opType!, cols, primary?.rows || [], secondary?.cols || []);
    } else if (it.kind === "load") {
      data.target = it.target;
    } else if (it.kind === "sql") {
      data.sqlText = DEFAULT_SQL;
    }
    const order = nodes.length;
    const position = pos || { x: 40 + order * 210, y: 90 + (it.kind === "source" && sources.length ? 150 : 0) };
    const node: FNode = { id, type: "flow", position, data };
    setNodes((nds) => [...nds, node]);
    // chain from the right-most existing non-analytics node
    const prev = [...nodes].filter((n) => n.data.kind !== "analytics").sort((a, b) => b.position.x - a.position.x)[0];
    if (prev && it.kind !== "source") setEdges((eds) => addEdge({ id: `e-${prev.id}-${id}`, source: prev.id, target: id, animated: true, style: { stroke: color, strokeWidth: 2 }, markerEnd: { type: MarkerType.ArrowClosed, color } }, eds));
    setSelId(id);
  }
  // Run the whole pipeline: animate each node in flow order, then report metrics.
  function runAll() {
    runTimers.current.forEach(clearTimeout); runTimers.current = [];
    if (!primary || !finalTable) { setRunMsg("Add a source and load data first."); return; }
    const order = [...nodes].sort((a, b) => a.position.x - b.position.x);
    setRunning(true); setRunStatus({}); setMetrics(null); setRunMsg("");
    const t0 = performance.now();
    let i = 0;
    const tick = () => {
      if (i > 0) setRunStatus((s) => ({ ...s, [order[i - 1].id]: "done" }));
      if (i < order.length) { const id = order[i].id; setRunStatus((s) => ({ ...s, [id]: "running" })); i++; runTimers.current.push(setTimeout(tick, 320)); }
      else { setRunStatus(Object.fromEntries(order.map((n) => [n.id, "done"]))); const ms = Math.max(1, Math.round(performance.now() - t0)); const extra = mode === "streaming" ? { batches: Math.max(1, Math.ceil(primary.rows.length / 64)), thr: Math.round(finalTable.rows.length / (ms / 1000)) } : {}; setMetrics({ inn: primary.rows.length, out: finalTable.rows.length, ms, ...extra }); setRunning(false); if (nodes.some((n) => n.data.kind === "analytics")) setDashOpen(true); }
    };
    tick();
  }
  const onDragStartTile = (e: React.DragEvent, key: string) => { e.dataTransfer.setData("application/etlflow", key); e.dataTransfer.effectAllowed = "move"; };
  const onCanvasDragOver = (e: React.DragEvent) => { e.preventDefault(); e.dataTransfer.dropEffect = "move"; };
  const onCanvasDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const key = e.dataTransfer.getData("application/etlflow");
    const it = MODES[mode].steps[step].items.find((p) => p.key === key);
    if (!it) return;
    const pos = screenToFlowPosition({ x: e.clientX, y: e.clientY });
    addFromPalette(it, { x: pos.x - 79, y: pos.y - 24 });
  };

  const patchSel = (patch: Partial<NData>) => setNodes((nds) => nds.map((n) => (n.id === selId ? { ...n, data: { ...n.data, ...patch } } : n)));
  const patchOp = (p: Partial<EtlOp>) => setNodes((nds) => nds.map((n) => (n.id === selId && n.data.op ? { ...n, data: { ...n.data, op: { ...n.data.op, ...p } as EtlOp, label: OP_META[n.data.op.type].label } } : n)));
  const removeSel = () => { if (!selId) return; setNodes((nds) => nds.filter((n) => n.id !== selId)); setEdges((eds) => eds.filter((e) => e.source !== selId && e.target !== selId)); setSelId(null); };

  async function onFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0]; if (!f) return;
    const ext = (f.name.split(".").pop() || "").toLowerCase();
    try {
      let t: Table;
      if (["xlsx", "xls", "xlsm"].includes(ext)) { const fd = new FormData(); fd.append("file", f); const r = await fetch("/api/rag/extract", { method: "POST", body: fd }); const j = await r.json(); if (!r.ok) throw new Error(j.error || "parse failed"); t = parseRecords(j.text); }
      else t = parseRecords(await f.text());
      patchSel({ table: t, srcName: f.name, label: f.name.slice(0, 14) });
    } catch (err) { setRunMsg((err as Error).message); }
    e.target.value = "";
  }

  function download(text: string, name: string, mime: string) { const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([text], { type: mime })); a.download = name; a.click(); URL.revokeObjectURL(a.href); }

  // ── save / load / import the workflow (reuses /api/projects) ──
  // Strips transient fields (count/run) and credentials (dbUrl/extUrl); keeps
  // node config + the loaded data (trimmed) so a reload restores the pipeline.
  function buildWorkflow(): SavedWF {
    const strip = (d: NData): NData => ({ ...d, count: undefined, run: undefined, dbUrl: undefined, extUrl: undefined, table: d.table ? { cols: d.table.cols, rows: d.table.rows.slice(0, 2000) } : undefined });
    return { mode, nodes: nodes.map((n) => ({ id: n.id, position: n.position, data: strip(n.data) })), edges: edges.map((e) => ({ id: e.id, source: e.source, target: e.target })) };
  }
  function applyWorkflow(wf: SavedWF | undefined) {
    if (!wf || !Array.isArray(wf.nodes)) { setSavedMsg("Nothing to load"); return; }
    const acc = (wf.mode && MODES[wf.mode]?.accent) || accent;
    if (wf.mode) { setMode(wf.mode); setStep(0); }
    setNodes(wf.nodes.map((n) => ({ id: n.id, type: "flow", position: n.position || { x: 40, y: 90 }, data: n.data })));
    setEdges((wf.edges || []).map((e) => ({ id: e.id || `e-${e.source}-${e.target}`, source: e.source, target: e.target, animated: true, style: { stroke: acc, strokeWidth: 2 }, markerEnd: { type: MarkerType.ArrowClosed, color: acc } })));
    setSelId(null); setRunStatus({}); setMetrics(null); setRunMsg("");
  }
  async function loadProjects() { try { const j = await fetch("/api/projects?lab=etl").then((r) => r.json()); setProjects(((j.projects || []) as { id: string; name: string }[]).map((p) => ({ id: p.id, name: p.name }))); } catch { /* ignore */ } }
  async function loadProject(id: string) { if (!id) return; try { const j = await fetch(`/api/projects?id=${id}`).then((r) => r.json()); applyWorkflow(j.project?.config as SavedWF); setCurrentId(id); setSavedMsg("Loaded ✓"); setTimeout(() => setSavedMsg(""), 2000); } catch { setSavedMsg("Load failed"); } }
  async function deleteProject(id: string) {
    if (!(await confirmDialog("Delete this saved workflow?", { danger: true, confirmLabel: "Delete" }))) return;
    try { const r = await fetch(`/api/projects?id=${id}`, { method: "DELETE" }); toast(r.ok ? "Workflow deleted" : "Delete failed", r.ok ? "success" : "error"); if (id === currentId) setCurrentId(null); loadProjects(); }
    catch { toast("Delete failed", "error"); }
  }
  async function renameProject(id: string) {
    const name = renameText.trim(); setRenamingId(null);
    if (!name) return;
    try { const r = await fetch("/api/projects", { method: "PATCH", headers: { "content-type": "application/json" }, body: JSON.stringify({ id, name }) }); toast(r.ok ? "Renamed" : "Rename failed", r.ok ? "success" : "error"); loadProjects(); }
    catch { toast("Rename failed", "error"); }
  }
  const wfName = () => nodes.find((n) => n.data.kind === "source")?.data.srcName || `${MODES[mode].label} pipeline`;
  // Save: update the loaded workflow in place if one is open, else create new.
  async function saveWorkflow() {
    try {
      if (currentId) {
        const r = await fetch("/api/projects", { method: "PATCH", headers: { "content-type": "application/json" }, body: JSON.stringify({ id: currentId, config: buildWorkflow() }) });
        setSavedMsg(r.ok ? "Updated ✓" : "Update failed"); if (r.ok) loadProjects();
      } else {
        const r = await fetch("/api/projects", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ lab: "etl", name: wfName(), config: buildWorkflow() }) });
        const j = await r.json().catch(() => null); setSavedMsg(r.ok ? "Saved ✓" : (j?.error || "Save failed")); if (r.ok && j?.id) { setCurrentId(j.id); loadProjects(); }
      }
    } catch { setSavedMsg("Save failed"); }
    setTimeout(() => setSavedMsg(""), 2500);
  }
  // Save as a brand-new copy (does not touch the loaded workflow).
  async function saveAsNew() {
    try { const r = await fetch("/api/projects", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ lab: "etl", name: wfName() + " (copy)", config: buildWorkflow() }) }); const j = await r.json().catch(() => null); setSavedMsg(r.ok ? "Saved copy ✓" : (j?.error || "Save failed")); if (r.ok && j?.id) { setCurrentId(j.id); loadProjects(); } }
    catch { setSavedMsg("Save failed"); }
    setTimeout(() => setSavedMsg(""), 2500);
  }
  // Start a fresh, unsaved workflow.
  function newWorkflow() { setNodes([]); setEdges([]); setCurrentId(null); setSelId(null); setRunStatus({}); setMetrics(null); setRunMsg(""); }
  function exportJson() { download(JSON.stringify(buildWorkflow(), null, 2), "etl-workflow.json", "application/json"); }
  // Build a codegen spec from the current canvas (primary source → ordered ops → load node).
  function currentSpec(): CodegenSpec {
    const src = sources[0]?.data;
    const loadNode = nodes.find((n) => n.data.kind === "load")?.data;
    return {
      source: { type: src?.srcType || "csv", url: src?.dbUrl, query: src?.dbQuery, restUrl: src?.restUrl },
      ops: opNodes.map((n) => n.data.op!).filter(Boolean),
      load: loadNode ? { target: loadNode.target, url: loadNode.extUrl, table: loadNode.extTable, mode: loadNode.extMode } : undefined,
      secondary: sources.length > 1,
    };
  }
  function exportPython() { download(toPython(currentSpec()), "etl_pipeline.py", "text/x-python"); toast("Downloaded runnable Python ✓", "success"); }
  // Deploy the pipeline as a Bearer-key API: save + publish the project, then mint an api channel.
  async function deployApi() {
    if (!opNodes.length) { setDeploy({ open: true, busy: false, err: "Add at least one transform before deploying." }); return; }
    setDeploy({ open: true, busy: true });
    try {
      let pid = currentId;
      const cfg = buildWorkflow();
      if (pid) {
        await fetch("/api/projects", { method: "PATCH", headers: { "content-type": "application/json" }, body: JSON.stringify({ id: pid, config: cfg, published: true }) });
      } else {
        const r = await fetch("/api/projects", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ lab: "etl", name: wfName(), config: cfg, published: true }) });
        const j = await r.json(); if (!r.ok) throw new Error(j.error || "save failed"); pid = j.id; setCurrentId(pid); loadProjects();
      }
      // Build the connection config for a full extract→transform→load run (only if a real DB source URL is set).
      const src = sources[0]?.data; const loadNode = nodes.find((n) => n.data.kind === "load")?.data;
      const conn = src?.dbUrl ? { sourceUrl: src.dbUrl, query: src.dbQuery || "", srcType: src.srcType || "", targetUrl: loadNode?.extUrl || "", table: loadNode?.extTable || "etl_output", mode: loadNode?.extMode || "append", keyCol: primary?.cols?.[0] || "" } : null;
      const rc = await fetch("/api/etl/deploy", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ projectId: pid, conn, dailyLimit: 200 }) });
      const jc = await rc.json(); if (!rc.ok) throw new Error(jc.error || "deploy failed");
      setDeploy({ open: true, busy: false, key: jc.apiKey, chId: jc.id, mode: jc.mode, hasTarget: jc.hasTarget });
    } catch (e) { setDeploy({ open: true, busy: false, err: (e as Error).message }); }
  }
  async function importJson(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0]; if (!f) return;
    try { applyWorkflow(JSON.parse(await f.text()) as SavedWF); setCurrentId(null); setSavedMsg("Imported ✓"); } catch { setSavedMsg("Bad JSON file"); }
    e.target.value = ""; setTimeout(() => setSavedMsg(""), 2500);
  }
  // Load the saved project list, and any ?project=<id> deep link, on mount.
  useEffect(() => {
    loadProjects();
    loadRuns();
    const id = new URLSearchParams(window.location.search).get("project");
    if (id) loadProject(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  async function runLoad() {
    if (!sel || !finalTable) return;
    const target = sel.data.target;
    if (target === "csv") { download(toCSV(finalTable), "etl_output.csv", "text/csv"); setRunMsg("Downloaded CSV ✓"); logRun({ mode: "load", target: "csv", rowsOut: finalTable.rows.length, rowsLoaded: finalTable.rows.length, status: "ok" }); return; }
    if (target === "xlsx") { download(toCSV(finalTable), "etl_output.csv", "text/csv"); setRunMsg("Exported (CSV; xlsx export next) ✓"); logRun({ mode: "load", target: "csv", rowsOut: finalTable.rows.length, status: "ok" }); return; }
    if (target === "platform") {
      try { const r = await fetch("/api/etl/store", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ name: "ETL flow output", cols: finalTable.cols, rows: finalTable.rows, mode: "new" }) }); const ok = r.ok; setRunMsg(ok ? "Stored in platform DB ✓" : "Store failed"); logRun({ mode: "load", target: "platform DB", rowsOut: finalTable.rows.length, rowsLoaded: ok ? finalTable.rows.length : 0, status: ok ? "ok" : "error" }); }
      catch (err) { setRunMsg((err as Error).message); logRun({ mode: "load", target: "platform DB", status: "error", error: (err as Error).message }); }
      return;
    }
    setRunMsg(`${target} connector — configure credentials in Admin → Providers (wiring next).`);
  }
  // Real DB source: run a SELECT against the node's mysql:// URL and load the rows.
  async function runSourceQuery() {
    if (!sel) return;
    setRunMsg("querying…");
    try {
      const r = await fetch("/api/ml/query", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ url: sel.data.dbUrl || "", query: sel.data.dbQuery || "" }) });
      const j = await r.json(); if (!r.ok) throw new Error(j.error || "query failed");
      patchSel({ table: parseRecords(j.csv), srcName: "db query", label: "db query" });
      setRunMsg(`loaded ${j.rows} rows ✓`);
    } catch (e) { setRunMsg((e as Error).message); }
  }
  // REST/API source: GET a public JSON endpoint and load its records.
  async function runRestSource() {
    if (!sel) return;
    setRunMsg("fetching…");
    try {
      const r = await fetch("/api/etl/fetch-json", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ url: sel.data.restUrl || "", path: sel.data.restPath || "" }) });
      const j = await r.json(); if (!r.ok) throw new Error(j.error || "fetch failed");
      const t = tableFromRecords(j.records || []);
      patchSel({ table: t, srcName: "REST", label: "REST source" });
      setRunMsg(`loaded ${t.rows.length} rows ✓`);
    } catch (e) { setRunMsg((e as Error).message); }
  }
  // Fire-and-forget: record a run in the log (etl_runs) so the Runs console shows history.
  function logRun(r: { mode: string; target?: string; rowsOut?: number; rowsLoaded?: number; status?: string; error?: string; durationMs?: number }) {
    fetch("/api/etl/runs", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ name: wfName(), ...r }) }).then(() => loadRuns()).catch(() => {});
  }
  async function loadRuns() { try { const j = await fetch("/api/etl/runs").then((x) => x.json()); setRuns(j.runs || []); } catch { /* ignore */ } }
  // ELT: run the SQL model node against the raw source (real in-browser SQL).
  async function runSqlNode() {
    if (!sel || !primary) { setRunMsg("Add a source with data first."); return; }
    setRunMsg("running SQL…");
    try { const res = await runSql(sel.data.sqlText || "", primary, secondary, []); patchSel({ table: res }); setRunMsg(`SQL ran · ${res.rows.length} rows ✓`); }
    catch (e) { setRunMsg("SQL error: " + (e as Error).message); }
  }
  // Load pasted JSON into the source node.
  function loadJsonSource() {
    if (!sel) return;
    try {
      const data = JSON.parse(sel.data.jsonText || "");
      const arr = Array.isArray(data) ? data : Array.isArray(data?.data) ? data.data : Array.isArray(data?.rows) ? data.rows : null;
      if (!arr || !arr.length) throw new Error("expected a non-empty array of objects");
      patchSel({ table: tableFromRecords(arr), srcName: "pasted JSON", label: "JSON" });
      setRunMsg("JSON loaded ✓");
    } catch (e) { setRunMsg("JSON error: " + (e as Error).message); }
  }
  // Real external load into the user's own MySQL/TiDB via the existing route.
  async function runExternal() {
    if (!sel || !finalTable) return;
    setRunMsg("loading…");
    try {
      const dbUrl = sel.data.extUrl || "";
      const r = await fetch("/api/etl/store-external", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ url: dbUrl, table: sel.data.extTable || "etl_output", cols: finalTable.cols, rows: finalTable.rows, mode: sel.data.extMode || "append", keyCol: finalTable.cols[0], db: sel.data.extDb }) });
      const j = await r.json().catch(() => null);
      setRunMsg(r.ok ? `${(sel.data.extMode === "upsert" ? "Upserted" : "Loaded")} ${j.rowCount} rows into \`${j.table}\` ✓` : (j?.error || "load failed"));
      logRun({ mode: "load", target: `${sel.data.target || "db"}:${j?.table || sel.data.extTable || "etl_output"}`, rowsOut: finalTable.rows.length, rowsLoaded: r.ok ? (j?.rowCount || finalTable.rows.length) : 0, status: r.ok ? "ok" : "error", error: r.ok ? undefined : (j?.error || "load failed") });
    } catch (e) { setRunMsg((e as Error).message); logRun({ mode: "load", target: "external db", status: "error", error: (e as Error).message }); }
  }

  const paletteItems = MODES[mode].steps[step].items;

  return (
    <div>
      <div className="lab-head">
        <div><div className="eyebrow">Lab 06 · data engineering</div><h2 className="page-h">ETL Lab</h2><p className="page-sub" style={{ margin: 0 }}>Drag nodes onto the canvas, wire them, configure each, and watch the output — guided step by step across four pipeline modes.</p></div>
        <div className="acts" style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
          {currentId && <button className="btn ghost sm" onClick={newWorkflow} title="Start a new, empty workflow">＋ New</button>}
          <button className="btn ghost sm" onClick={saveWorkflow} title={currentId ? "Update the loaded workflow" : "Save as a new workflow"}>{savedMsg || (currentId ? "💾 Update" : "💾 Save")}</button>
          {currentId && <button className="btn ghost sm" onClick={saveAsNew} title="Save as a new copy">⧉ Save as new</button>}
          <div style={{ position: "relative" }}>
            <button className="btn ghost sm" onClick={() => { setLoadOpen((o) => !o); if (!loadOpen) loadProjects(); }}>📂 Load… ▾</button>
            {loadOpen && <>
              <div onClick={() => { setLoadOpen(false); setRenamingId(null); }} style={{ position: "fixed", inset: 0, zIndex: 40 }} />
              <div style={{ position: "absolute", top: "112%", right: 0, zIndex: 50, width: 288, maxHeight: 340, overflowY: "auto", background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 10, boxShadow: "var(--shadow-md)", padding: 6 }}>
                <div className="note" style={{ padding: "4px 8px 6px", fontSize: 10, textTransform: "uppercase", letterSpacing: ".06em" }}>Saved workflows</div>
                {projects.length === 0 ? <div className="note" style={{ padding: "6px 8px" }}>None yet — build a pipeline and Save.</div> :
                  projects.map((p) => (
                    <div key={p.id} className="etl-load-row" style={{ display: "flex", alignItems: "center", gap: 4, padding: "5px 8px", borderRadius: 7, background: p.id === currentId ? "var(--panel)" : undefined }}>
                      {renamingId === p.id ? (
                        <input autoFocus value={renameText} onChange={(e) => setRenameText(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") renameProject(p.id); if (e.key === "Escape") setRenamingId(null); }} onBlur={() => renameProject(p.id)} style={{ flex: 1, height: 26, fontSize: 12 }} />
                      ) : (
                        <button onClick={() => { loadProject(p.id); setLoadOpen(false); }} style={{ flex: 1, textAlign: "left", background: "none", border: "none", color: "var(--text)", fontSize: 12.5, cursor: "pointer", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>📄 {p.name}{p.id === currentId ? " •" : ""}</button>
                      )}
                      <button onClick={() => { setRenamingId(p.id); setRenameText(p.name); }} title="Rename" style={{ background: "none", border: "none", color: "var(--faint)", fontSize: 13, cursor: "pointer", padding: "0 3px" }}>✎</button>
                      <button onClick={() => deleteProject(p.id)} title="Delete workflow" style={{ background: "none", border: "none", color: "var(--faint)", fontSize: 16, lineHeight: 1, cursor: "pointer", padding: "0 3px" }}>×</button>
                    </div>
                  ))}
              </div>
            </>}
          </div>
          <button className="btn ghost sm" onClick={exportPython} title="Download a runnable pandas + SQLAlchemy script — run it in your own project">⤓ Python</button>
          <button className="btn ghost sm" onClick={exportJson} title="Export the workflow as JSON">⤓ JSON</button>
          <button className="btn ghost sm" onClick={() => importRef.current?.click()}>⤒ Import</button>
          <button className="btn sm" onClick={deployApi} title="Deploy this pipeline as a callable API (small data)">🚀 Deploy as API</button>
        </div>
      </div>
      <input ref={importRef} type="file" accept=".json,application/json" onChange={importJson} style={{ display: "none" }} />

      {deploy?.open && (
        <div style={{ border: "1px solid var(--border-strong)", borderRadius: 12, background: "var(--panel)", padding: 14, marginBottom: 12 }}>
          <div className="row" style={{ justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
            <b style={{ fontSize: 13 }}>🚀 Deploy as API {deploy.busy ? "· publishing…" : ""}</b>
            <button className="btn ghost sm" onClick={() => setDeploy(null)}>✕ close</button>
          </div>
          {deploy.err && <div className="err">{deploy.err}</div>}
          {deploy.busy && <div className="note"><span className="busy-dot" /> saving, publishing & minting a key…</div>}
          {deploy.key && deploy.chId && (() => {
            const origin = typeof window !== "undefined" ? window.location.origin : "https://your-app.vercel.app";
            const url = `${origin}/api/etl/public/${deploy.chId}`;
            const cols = (primary?.cols && primary.cols.length ? primary.cols : ["col1", "col2"]).slice(0, 6);
            const sample: Record<string, unknown> = {};
            cols.forEach((c) => { sample[c] = primary?.rows?.[0]?.[c] ?? "value"; });
            const full = deploy.mode === "full_run"; const hasTarget = !!deploy.hasTarget;
            const curl = full
              ? `curl -X POST ${url} \\\n  -H "Authorization: Bearer ${deploy.key}" \\\n  -H "content-type: application/json" \\\n  -d '{}'`
              : `curl -X POST ${url} \\\n  -H "Authorization: Bearer ${deploy.key}" \\\n  -H "content-type: application/json" \\\n  -d '${JSON.stringify({ data: [sample] })}'`;
            const js = full
              ? `const res = await fetch(${JSON.stringify(url)}, {\n  method: "POST",\n  headers: { "Authorization": "Bearer ${deploy.key}", "content-type": "application/json" },\n  body: "{}",\n});\nconst summary = await res.json();\nconsole.log(summary); // { rowsExtracted, rowsOut, rowsLoaded }`
              : `const res = await fetch(${JSON.stringify(url)}, {\n  method: "POST",\n  headers: { "Authorization": "Bearer ${deploy.key}", "content-type": "application/json" },\n  body: JSON.stringify({ data: [${JSON.stringify(sample)}] }),\n});\nconst { cols, rows } = await res.json();\nconsole.log(rows);`;
            const python = full
              ? `import requests\n\nr = requests.post(\n    ${JSON.stringify(url)},\n    headers={"Authorization": "Bearer ${deploy.key}", "content-type": "application/json"},\n    json={},\n)\nprint(r.json())  # {"rowsExtracted": .., "rowsLoaded": ..}`
              : `import requests\n\nr = requests.post(\n    ${JSON.stringify(url)},\n    headers={"Authorization": "Bearer ${deploy.key}", "content-type": "application/json"},\n    json={"data": [${JSON.stringify(sample)}]},\n)\nprint(r.json())  # {"cols": [...], "rows": [...]}`;
            const snippet = apiLang === "curl" ? curl : apiLang === "js" ? js : python;
            const respStr = full
              ? JSON.stringify(hasTarget ? { ok: true, mode: "full_run", pipeline: wfName(), rowsExtracted: 1000, rowsOut: 42, rowsLoaded: 42, loadedTable: "etl_output" } : { ok: true, mode: "full_run", pipeline: wfName(), rowsExtracted: 1000, rowsOut: 42, cols, rows: [sample] }, null, 2)
              : JSON.stringify({ ok: true, mode: "transform_only", pipeline: wfName(), cols, rows: [sample], rowsIn: 1, rowsOut: 1 }, null, 2);
            const codeBox: React.CSSProperties = { fontFamily: "var(--mono)", fontSize: 11, background: "var(--panel-2)", padding: "10px 12px", borderRadius: 8, overflowX: "auto", margin: 0, lineHeight: 1.5, whiteSpace: "pre" };
            const label = (t: string) => <div className="k" style={{ fontSize: 10.5, textTransform: "uppercase", letterSpacing: ".05em", color: "var(--muted)", margin: "10px 0 4px" }}>{t}</div>;
            return (
              <div>
                <div className="note" style={{ color: "var(--good)", marginBottom: 6 }}>✓ Deployed — {full ? (hasTarget ? "full run: extract → transform → load, server-side." : "full run: extract → transform, returns rows.") : "transform-only: POST rows, get rows back."} Your pipeline is now a live API.</div>

                {label("① API key — copy now, shown only once")}
                <div className="row" style={{ gap: 6 }}><code style={{ flex: 1, ...codeBox, whiteSpace: "nowrap", color: "var(--warn)" }}>{deploy.key}</code><button className="btn ghost sm" onClick={() => { navigator.clipboard?.writeText(deploy.key!); toast("Key copied", "success"); }}>copy</button></div>

                {label("② How to connect")}
                <div style={codeBox}>{`POST   ${url}\nAuth   Authorization: Bearer <your-key>\nType   application/json`}</div>

                {label("③ Input — what to send")}
                {full ? (
                  <>
                    <div style={codeBox}>{`# No body needed — just call with your key.\n# The server extracts from your source DB, transforms,\n# and ${hasTarget ? "loads into your warehouse table." : "returns the rows."}\n\n# Optional: override the source SELECT for this run\n{ "query": "SELECT * FROM orders WHERE created_at > '2026-01-01'" }`}</div>
                    <div className="note" style={{ marginTop: 4, lineHeight: 1.5 }}>No data goes over the wire — your source & target URLs are stored <b>encrypted</b> with your pipeline and used server-side.</div>
                  </>
                ) : (
                  <>
                    <div style={codeBox}>{`{
  "data": [        // an array of row objects (≤ 5000)
    {
${cols.map((c) => `      ${JSON.stringify(c)}: ${JSON.stringify(sample[c])}`).join(",\n")}
    }
  ]
}`}</div>
                    <div className="note" style={{ marginTop: 4, lineHeight: 1.5 }}>Each row is a JSON object. The fields above are the columns this pipeline expects. Extra fields are ignored; missing ones become null.</div>
                  </>
                )}

                {label("④ Example call")}
                <div className="row" style={{ gap: 4, marginBottom: 6 }}>
                  {(["curl", "js", "python"] as const).map((l) => <button key={l} className={`btn sm ${apiLang === l ? "" : "ghost"}`} onClick={() => setApiLang(l)}>{l === "js" ? "JavaScript" : l === "python" ? "Python" : "cURL"}</button>)}
                  <button className="btn ghost sm" style={{ marginLeft: "auto" }} onClick={() => { navigator.clipboard?.writeText(snippet); toast("Copied", "success"); }}>copy</button>
                </div>
                <pre style={codeBox}>{snippet}</pre>

                {label("⑤ Response")}
                <pre style={codeBox}>{respStr}</pre>

                <div className="note" style={{ marginTop: 10, lineHeight: 1.5 }}><b>Limits:</b> ≤5000 rows/request, 60s, 200 calls/day. For bigger jobs use <b>⤓ Python</b>. Revoke or re-issue the key in <b>Workroom → Channels</b>. On a deployed app the host is your Vercel domain (not localhost).</div>
              </div>
            );
          })()}
        </div>
      )}

      {/* Runs console */}
      {runs.length > 0 && (
        <div className="card" style={{ marginBottom: 12 }}>
          <div className="card-h" onClick={() => setShowRuns((s) => !s)} style={{ cursor: "pointer" }}><span className="t">🗂 Runs</span><span className="mono r">{runs.length} logged · {showRuns ? "▾ hide" : "▸ show"}</span></div>
          {showRuns && <div className="card-b" style={{ overflowX: "auto" }}>
            <table className="dtable" style={{ width: "100%", fontSize: 12 }}><tbody>
              <tr><th style={{ textAlign: "left" }}>when</th><th style={{ textAlign: "left" }}>pipeline</th><th style={{ textAlign: "left" }}>mode</th><th style={{ textAlign: "left" }}>target</th><th style={{ textAlign: "right" }}>out</th><th style={{ textAlign: "right" }}>loaded</th><th style={{ textAlign: "left" }}>status</th></tr>
              {runs.map((r) => (
                <tr key={r.id}>
                  <td style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--muted)", whiteSpace: "nowrap" }}>{r.ts ? new Date(r.ts).toLocaleString() : "—"}</td>
                  <td>{r.name || "—"}</td>
                  <td style={{ fontFamily: "var(--mono)", fontSize: 11 }}>{r.mode || "—"}</td>
                  <td style={{ fontSize: 11.5 }}>{r.target || "—"}</td>
                  <td style={{ textAlign: "right", fontFamily: "var(--mono)" }}>{r.rowsOut}</td>
                  <td style={{ textAlign: "right", fontFamily: "var(--mono)" }}>{r.rowsLoaded}</td>
                  <td style={{ color: r.status === "ok" ? "var(--good)" : "var(--crit)", fontWeight: 600 }}>{r.status || "—"}</td>
                </tr>
              ))}
            </tbody></table>
          </div>}
        </div>
      )}

      {/* Data quality · profile · lineage */}
      {finalTable && (
        <div className="card" style={{ marginBottom: 12 }}>
          <div className="card-h" onClick={() => setShowQuality((s) => !s)} style={{ cursor: "pointer" }}><span className="t">🔎 Quality · Profile · Lineage</span><span className="mono r">{finalTable.rows.length} rows · {showQuality ? "▾ hide" : "▸ show"}</span></div>
          {showQuality && (() => {
            const ops = opNodes.map((n) => n.data.op!).filter(Boolean);
            const prof = profile(finalTable);
            const lin = lineage(primary?.cols || [], ops, secondary?.cols || []);
            const ev = rules.length ? evaluate(finalTable, rules) : null;
            const rid = () => Math.random().toString(36).slice(2, 9);
            const box: React.CSSProperties = { border: "1px solid var(--border)", borderRadius: 8, overflowX: "auto" };
            return (
              <div className="card-b">
                <label className="fld">Column profile — types, nulls, distinct, stats</label>
                <div style={box}>
                  <table className="dtable" style={{ width: "100%", fontSize: 11.5 }}><tbody>
                    <tr><th style={{ textAlign: "left" }}>column</th><th>type</th><th>nulls</th><th>distinct</th><th>min</th><th>max</th><th>mean</th><th style={{ textAlign: "left" }}>top values</th></tr>
                    {prof.map((c) => (
                      <tr key={c.name}>
                        <td style={{ fontWeight: 600 }}>{c.name}</td>
                        <td style={{ textAlign: "center", fontFamily: "var(--mono)", color: "var(--accent)" }}>{c.type}</td>
                        <td style={{ textAlign: "center", color: c.nulls ? "var(--warn)" : "var(--muted)" }}>{c.nulls}</td>
                        <td style={{ textAlign: "center" }}>{c.distinct}</td>
                        <td style={{ textAlign: "center", fontFamily: "var(--mono)" }}>{c.min ?? "—"}</td>
                        <td style={{ textAlign: "center", fontFamily: "var(--mono)" }}>{c.max ?? "—"}</td>
                        <td style={{ textAlign: "center", fontFamily: "var(--mono)" }}>{c.mean != null ? Math.round(c.mean * 100) / 100 : "—"}</td>
                        <td style={{ fontSize: 10.5, color: "var(--muted)" }}>{c.top.slice(0, 3).map((t) => `${t.v}(${t.count})`).join(", ")}</td>
                      </tr>
                    ))}
                  </tbody></table>
                </div>

                <label className="fld" style={{ marginTop: 14 }}>Validation rules — expectations with an action on failure</label>
                <div className="row" style={{ gap: 8, marginBottom: 8, flexWrap: "wrap" }}>
                  <button className="btn ghost sm" onClick={() => setRules((rs) => [...rs, { id: rid(), col: finalTable.cols[0], type: "not_null", action: "reject" }])}>+ Add rule</button>
                  {rules.length > 0 && <button className="btn ghost sm" onClick={() => setRules([])}>clear</button>}
                  {ev && <span className="note">clean <b style={{ color: "var(--good)" }}>{ev.clean.rows.length}</b> · rejected <b style={{ color: "var(--crit)" }}>{ev.rejects.length}</b> · dropped {ev.dropped} · fixed {ev.fixedCells}</span>}
                </div>
                {rules.map((r) => (
                  <div key={r.id} className="row" style={{ gap: 6, marginBottom: 6, flexWrap: "wrap", alignItems: "center" }}>
                    <select value={r.col} onChange={(e) => setRules((rs) => rs.map((x) => x.id === r.id ? { ...x, col: e.target.value } : x))} style={{ width: 130 }}>{finalTable.cols.map((c) => <option key={c} value={c}>{c}</option>)}</select>
                    <select value={r.type} onChange={(e) => setRules((rs) => rs.map((x) => x.id === r.id ? { ...x, type: e.target.value as RuleType } : x))} style={{ width: 130 }}>{(Object.keys(RULE_META) as RuleType[]).map((t) => <option key={t} value={t}>{RULE_META[t].label}</option>)}</select>
                    {r.type === "in_range" && <><input placeholder="min" value={r.min ?? ""} onChange={(e) => setRules((rs) => rs.map((x) => x.id === r.id ? { ...x, min: e.target.value } : x))} style={{ width: 66 }} /><input placeholder="max" value={r.max ?? ""} onChange={(e) => setRules((rs) => rs.map((x) => x.id === r.id ? { ...x, max: e.target.value } : x))} style={{ width: 66 }} /></>}
                    {r.type === "regex" && <input placeholder="pattern" value={r.pattern ?? ""} onChange={(e) => setRules((rs) => rs.map((x) => x.id === r.id ? { ...x, pattern: e.target.value } : x))} style={{ width: 150 }} />}
                    {r.type === "in_set" && <input placeholder="a, b, c" value={r.set ?? ""} onChange={(e) => setRules((rs) => rs.map((x) => x.id === r.id ? { ...x, set: e.target.value } : x))} style={{ width: 150 }} />}
                    <select value={r.action} onChange={(e) => setRules((rs) => rs.map((x) => x.id === r.id ? { ...x, action: e.target.value as RuleAction } : x))} style={{ width: 120 }}>{(Object.keys(ACTION_META) as RuleAction[]).map((a) => <option key={a} value={a}>{ACTION_META[a].label}</option>)}</select>
                    {ev && (() => { const rep = ev.report.find((x) => x.id === r.id); return rep ? <span className="note" style={{ color: rep.ok ? "var(--good)" : "var(--warn)" }}>{rep.ok ? "✓ pass" : `✗ ${rep.fails} fail`}</span> : null; })()}
                    <button onClick={() => setRules((rs) => rs.filter((x) => x.id !== r.id))} style={{ background: "none", border: "none", color: "var(--faint)", fontSize: 15, cursor: "pointer" }}>×</button>
                  </div>
                ))}
                {rules.length === 0 && <div className="note">No rules — add expectations like <b>not-null</b>, <b>in-range</b>, or <b>unique</b>, and pick an action (reject / drop / fix / warn) on failure.</div>}

                <label className="fld" style={{ marginTop: 14 }}>Column lineage — where each output column came from</label>
                <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                  {lin.map((l) => (
                    <div key={l.col} style={{ fontSize: 12, fontFamily: "var(--mono)" }}><span style={{ color: "var(--accent)", fontWeight: 600 }}>{l.col}</span> <span style={{ color: "var(--faint)" }}>←</span> <span style={{ color: "var(--muted)" }}>{l.from.length ? l.from.join(", ") : "constant / new"}</span></div>
                  ))}
                </div>
              </div>
            );
          })()}
        </div>
      )}

      {/* mode selector */}
      <div className="row" style={{ gap: 8, flexWrap: "wrap", marginBottom: 12 }}>
        {(Object.keys(MODES) as Mode[]).map((m) => (
          <button key={m} onClick={() => { setMode(m); setStep(0); }} style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 14px", borderRadius: 10, cursor: "pointer", fontSize: 12.5, fontWeight: 500, border: `1px solid ${mode === m ? MODES[m].accent : "var(--border)"}`, background: mode === m ? `color-mix(in srgb, ${MODES[m].accent} 14%, transparent)` : "var(--panel)", color: mode === m ? "var(--text)" : "var(--muted)" }}>
            <span style={{ width: 8, height: 8, borderRadius: "50%", background: MODES[m].accent }} />{MODES[m].label}
          </button>
        ))}
      </div>

      {/* stepper (gates the palette) */}
      <div className="stepper">
        {MODES[mode].steps.map((s, i) => (
          <button key={i} className={step === i ? "on" : ""} onClick={() => setStep(i)}><b>{i + 1}</b>{s.label}</button>
        ))}
      </div>
      <div className="note" style={{ margin: "0 0 12px" }}>🔒 Palette shows only <b style={{ color: accent }}>{MODES[mode].steps[step].label}</b> nodes for this step.</div>

      <div className="split" style={{ gridTemplateColumns: "196px 1fr 300px", gap: 12 }}>
        {/* step-gated palette — project-2-style icon tiles, draggable */}
        <div className="card"><div className="card-h"><span className="t">{MODES[mode].steps[step].label} nodes</span><span className="mono r">{paletteItems.length}</span></div>
          <div className="card-b" style={{ maxHeight: 470, overflowY: "auto" }}>
            {Object.entries(paletteItems.reduce((g, it) => { (g[it.cat] = g[it.cat] || []).push(it); return g; }, {} as Record<string, PalItem[]>)).map(([cat, items]) => {
              const cc = CATCOLOR[cat as Cat];
              return (
                <div key={cat} style={{ marginBottom: 10 }}>
                  <div className="row" style={{ alignItems: "center", gap: 7, marginBottom: 8 }}>
                    <span style={{ width: 16, height: 16, borderRadius: 5, background: `color-mix(in srgb, ${cc} 22%, transparent)`, border: `1px solid ${cc}` }} />
                    <span style={{ fontSize: 10, fontWeight: 600, textTransform: "uppercase", letterSpacing: ".06em", color: "var(--muted)", flex: 1 }}>{CATLABEL[cat as Cat]}</span>
                    <span style={{ fontSize: 10, color: "var(--faint)" }}>{items.length}</span>
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 7 }}>
                    {items.map((it) => (
                      <div key={it.key} draggable onDragStart={(e) => onDragStartTile(e, it.key)} onClick={() => addFromPalette(it)} title={it.opType ? OP_META[it.opType].hint : it.label}
                        style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6, padding: "10px 6px", borderRadius: 11, border: "1px solid var(--border)", background: "var(--panel-2)", cursor: "grab" }}>
                        <span style={{ width: 32, height: 32, borderRadius: 9, display: "grid", placeItems: "center", fontSize: 15, background: `color-mix(in srgb, ${cc} 18%, transparent)`, color: cc }}>{it.icon}</span>
                        <span style={{ fontSize: 10.5, color: "var(--muted)", textAlign: "center", lineHeight: 1.2 }}>{it.label}</span>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
            <div className="note" style={{ fontSize: 10, lineHeight: 1.5 }}>Drag a tile onto the canvas, or click to add. Wire nodes by dragging handle → handle.</div>
          </div>
        </div>

        {/* canvas */}
        <div className="card" style={{ padding: 0, overflow: "hidden", position: "relative" }}>
          <div style={{ position: "absolute", right: 12, top: 12, zIndex: 10, display: "flex", gap: 8 }}>
            <button onClick={() => setPlayOpen(true)} disabled={!finalTable} title="Animate the data journey node by node, row by row" style={{ display: "flex", alignItems: "center", gap: 7, fontSize: 12.5, fontWeight: 600, color: "#fff", background: "var(--accent)", border: "none", borderRadius: 9, padding: "8px 15px", cursor: !finalTable ? "default" : "pointer", opacity: !finalTable ? 0.5 : 1 }}>▶ Play</button>
            <button onClick={runAll} disabled={running || !finalTable} style={{ display: "flex", alignItems: "center", gap: 7, fontSize: 12.5, fontWeight: 600, color: "#06210f", background: running ? "#9fe1cb" : "#3ecf7f", border: "none", borderRadius: 9, padding: "8px 15px", cursor: running || !finalTable ? "default" : "pointer", opacity: !finalTable ? 0.5 : 1 }}>{running ? "● running…" : "▶ Run pipeline"}</button>
          </div>
          <div style={{ height: 480 }} onDrop={onCanvasDrop} onDragOver={onCanvasDragOver}>
            <ReactFlow nodes={displayNodes} edges={edges} nodeTypes={nodeTypes} onNodesChange={onNodesChange} onEdgesChange={onEdgesChange} onConnect={onConnect} onNodeClick={(_, n) => setSelId(n.id)} onPaneClick={() => setSelId(null)} proOptions={{ hideAttribution: true }} defaultEdgeOptions={{ animated: true, style: { stroke: accent, strokeWidth: 2 } }} zoomOnDoubleClick={false}>
              <Background variant={BackgroundVariant.Dots} gap={20} size={1.4} color="var(--border-strong)" />
              <Controls />
              <MiniMap nodeColor={(n) => (n.data as NData).color} nodeStrokeWidth={2} bgColor="#0e121d" maskColor="rgba(8,11,19,0.72)" style={{ background: "#0e121d", border: "1px solid var(--border)", borderRadius: 8, width: 150, height: 96 }} pannable zoomable />
            </ReactFlow>
          </div>
        </div>

        {/* node config + reference */}
        <div className="card"><div className="card-h"><span className="t">Node details</span>{sel && <button className="btn ghost sm" onClick={removeSel}>Remove</button>}</div>
          <div className="card-b" style={{ maxHeight: 460, overflowY: "auto" }}>
            {!sel && <div className="note">Add a node from the palette, then click it to configure.</div>}
            {sel && <NodeConfig node={sel} cols={cols} bCols={secondary?.cols || []} patchSel={patchSel} patchOp={patchOp} onUpload={() => fileRef.current?.click()} runLoad={runLoad} runSourceQuery={runSourceQuery} runRestSource={runRestSource} loadJsonSource={loadJsonSource} runExternal={runExternal} runSqlNode={runSqlNode} openDash={() => setDashOpen(true)} />}
          </div>
        </div>
      </div>
      <input ref={fileRef} type="file" accept=".csv,.tsv,.json,.xlsx,.xls" onChange={onFile} style={{ display: "none" }} />

      {/* always-on output */}
      <div className="card" style={{ marginTop: 12 }}>
        <div className="card-h"><span className="t">Output — live result{sel ? ` (${sel.data.label})` : ""}</span><span className="mono r">{metrics ? `ran · ${metrics.inn}→${metrics.out} rows · ${metrics.ms}ms${metrics.batches ? ` · ${metrics.batches} micro-batches · ${metrics.thr}/s` : ""}` : activeTable ? `${activeTable.rows.length} rows · ${activeTable.cols.length} cols` : "no source yet"}{runMsg ? ` · ${runMsg}` : ""}</span></div>
        <div className="card-b" style={{ maxHeight: 260, overflow: "auto" }}>
          {!activeTable ? <div className="note">Add a source node and upload / pick data to see the output.</div> : (
            sel?.data.kind === "analytics" ? <AnalyticsView table={activeTable} /> : (
              <>
                {sel?.data.kind === "load" && <div className="note" style={{ marginBottom: 8 }}>→ Table being loaded into <b>{sel.data.target}</b> · {activeTable.rows.length} rows × {activeTable.cols.length} cols</div>}
                <div style={{ overflowX: "auto" }}><table className="dtable"><tbody>
                  <tr>{activeTable.cols.map((c) => <th key={c}>{c}</th>)}</tr>
                  {activeTable.rows.slice(0, 8).map((r, i) => <tr key={i}>{activeTable.cols.map((c) => <td key={c}>{r[c] == null ? "—" : String(r[c])}</td>)}</tr>)}
                </tbody></table></div>
              </>
            )
          )}
        </div>
      </div>

      {/* Data Visualization Explorer */}
      {finalTable && (
        <details className="collapsible" style={{ marginTop: 12 }} open>
          <summary style={{ cursor: "pointer", fontWeight: "bold", padding: "10px 0" }}>Data Visualization Explorer</summary>
          <div style={{ marginTop: 8 }}>
            <DataVizExplorer
              sourceTable={primary || finalTable}
              finalTable={finalTable}
              ops={opNodes.map((n) => n.data.op!).filter(Boolean)}
              rules={[]}
            />
          </div>
        </details>
      )}

      {/* Power BI-style interactive dashboard modal */}
      {dashOpen && finalTable && <DashboardModal table={finalTable} title={sel?.data.kind === "analytics" ? sel.data.label : "Pipeline dashboard"} onClose={() => setDashOpen(false)} />}

      {/* Animated node-by-node, row-by-row pipeline walkthrough */}
      {playOpen && finalTable && primary && <PlayModal raw={primary} nodes={playNodes} final={finalTable} onClose={() => setPlayOpen(false)} />}

      {infoNodeId && (() => {
        const infoNode = nodes.find((n) => n.id === infoNodeId);
        return infoNode ? <NodeInfoModal node={infoNode} onClose={() => setInfoNodeId(null)} /> : null;
      })()}
    </div>
  );
}

// ── per-node config editor ──
function NodeConfig({ node, cols, bCols, patchSel, patchOp, onUpload, runLoad, runSourceQuery, runRestSource, loadJsonSource, runExternal, runSqlNode, openDash }: { node: FNode; cols: string[]; bCols: string[]; patchSel: (p: Partial<NData>) => void; patchOp: (p: Partial<EtlOp>) => void; onUpload: () => void; runLoad: () => void; runSourceQuery: () => void; runRestSource: () => void; loadJsonSource: () => void; runExternal: () => void; runSqlNode: () => void; openDash: () => void }) {
  const d = node.data; const c = d.color;
  const isExternalDb = d.kind === "load" && ["mysql", "postgres", "snowflake", "bigquery", "mongodb", "redshift"].includes(d.target || "");
  const nameField = (
    <div className="insp-field"><div className="k">Node name</div><input type="text" value={d.label} onChange={(e) => patchSel({ label: e.target.value })} /></div>
  );
  const colSel = (val: string | undefined, on: (v: string) => void, list = cols) => <select value={val || ""} onChange={(e) => on(e.target.value)}>{list.map((x) => <option key={x}>{x}</option>)}</select>;

  return (
    <div>
      <div className="row" style={{ alignItems: "center", gap: 9, marginBottom: 10 }}>
        <span style={{ width: 34, height: 34, borderRadius: 9, display: "grid", placeItems: "center", fontSize: 16, background: `color-mix(in srgb, ${c} 18%, transparent)`, color: c }}>{d.icon}</span>
        <span style={{ fontSize: 9.5, letterSpacing: ".07em", textTransform: "uppercase", fontWeight: 600, padding: "2px 8px", borderRadius: 6, background: `color-mix(in srgb, ${c} 16%, transparent)`, color: c }}>{d.cat}</span>
      </div>
      {nameField}

      {d.kind === "source" && (
        <div className="fade-in">
          {extractionConfig[d.srcType!] && (
            <div style={{ padding: "12px", background: "var(--panel)", borderRadius: "10px", marginBottom: "16px", border: "1px solid var(--border)", boxShadow: "var(--shadow-sm)" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
                <b style={{ fontSize: 14 }}>{extractionConfig[d.srcType!].name} Setup</b>
                <span className={`badge ${extractionConfig[d.srcType!].difficulty === "Beginner" ? "good" : extractionConfig[d.srcType!].difficulty === "Intermediate" ? "accent" : "warn"}`}>{extractionConfig[d.srcType!].difficulty}</span>
              </div>
              <div style={{ display: "flex", gap: "8px", flexWrap: "wrap", fontSize: 11, color: "var(--faint)" }}>
                <span>🗂 {extractionConfig[d.srcType!].category}</span>
                {extractionConfig[d.srcType!].apiKeyRequired && <span title="API Key Required">🔑 API Key</span>}
                {extractionConfig[d.srcType!].oauthRequired && <span title="OAuth Required">🛡️ OAuth</span>}
              </div>
            </div>
          )}

          {/* Dynamic Fields */}
          {extractionConfig[d.srcType!]?.fields.map((field) => {
            const val = String((d as unknown as Record<string, unknown>)[field.id] || "");
            const isValid = field.required && val.length > 0;
            return (
              <div key={field.id} className="insp-field group-section">
                <div className="k" title={field.tooltip} style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span>{field.label} {field.required && <span style={{ color: "var(--warn)" }}>*</span>} {field.tooltip && <span style={{ cursor: "help", color: "var(--muted)" }}>ⓘ</span>}</span>
                  {isValid && <span style={{ color: "var(--good)" }}>✓</span>}
                </div>
                {field.type === "textarea" ? (
                  <textarea rows={3} value={val} placeholder={field.placeholder} onChange={(e) => patchSel({ [field.id]: e.target.value })} />
                ) : field.type === "select" ? (
                  <select value={val} onChange={(e) => patchSel({ [field.id]: e.target.value })}>
                    <option value="" disabled>Select...</option>
                    {field.options?.map((opt) => <option key={opt} value={opt}>{opt}</option>)}
                  </select>
                ) : (
                  <input type={field.type} value={val} placeholder={field.placeholder} onChange={(e) => patchSel({ [field.id]: e.target.value })} />
                )}
              </div>
            );
          })}

          {/* Preserve Existing Actions */}
          {d.srcType === "csv" && <>
            <div className="note" style={{ marginBottom: 8, marginTop: 8 }}>{d.srcName ? `Loaded: ${d.srcName}` : "Pick a sample or upload a file."}</div>
            <div className="insp-field"><div className="k">Sample dataset</div><select onChange={(e) => { const s = sampleSources().find((x) => x.key === e.target.value)!; patchSel({ table: parseRecords(s.csv), srcName: s.label, label: s.label.slice(0, 14) }); }} defaultValue="">{[<option key="" value="" disabled>choose…</option>, ...sampleSources().map((s) => <option key={s.key} value={s.key}>{s.label}</option>)]}</select></div>
            <button className="btn sm block" onClick={onUpload}>⬆ Upload CSV / Excel / JSON</button>
          </>}
          {(d.srcType === "mysql" || d.srcType === "postgres") && <>
            <div className="insp-field"><div className="k">Query</div><textarea rows={2} value={d.dbQuery ?? "SELECT * FROM orders LIMIT 500;"} onChange={(e) => patchSel({ dbQuery: e.target.value })} /></div>
            <button className="btn sm block" onClick={runSourceQuery}>▶ Run query &amp; load</button>
            <div className="teach-note" style={{ marginTop: 8 }}><span className="ic">🔒</span><span>Runs a read-only SELECT against your <b>{d.srcType === "postgres" ? "Postgres" : "MySQL / TiDB"}</b> (private hosts blocked, capped at 2000 rows).</span></div>
          </>}
          {d.srcType === "rest" && <>
            <div className="insp-field"><div className="k">API URL (GET, returns JSON)</div><input type="text" value={d.restUrl ?? ""} placeholder="https://api.example.com/v1/orders" onChange={(e) => patchSel({ restUrl: e.target.value })} /></div>
            <div className="insp-field"><div className="k">JSON path (optional)</div><input type="text" value={d.restPath ?? ""} placeholder="data.items" onChange={(e) => patchSel({ restPath: e.target.value })} /></div>
            <button className="btn sm" onClick={runRestSource}>▶ Fetch &amp; load</button>
            <div className="teach-note" style={{ marginTop: 8 }}><span className="ic">🌐</span><span>GETs a public JSON API and flattens it to rows (array, or <code>data</code>/<code>rows</code>/<code>results</code>, or a dot-path). Private hosts blocked, capped at 5000 rows.</span></div>
          </>}
          {d.srcType === "json" && <>
            <div className="insp-field"><div className="k">JSON array of objects</div><textarea rows={5} value={d.jsonText ?? '[\n  {"id": 1, "region": "US", "amount": 120}\n]'} onChange={(e) => patchSel({ jsonText: e.target.value })} /></div>
            <button className="btn sm block" onClick={loadJsonSource}>Load JSON</button>
          </>}
          {d.srcType === "kafka" && <div className="teach-note" style={{ marginTop: 8 }}><span className="ic">📡</span><span>Streaming source — micro-batches over your rows. Attach a sample table to simulate the stream.</span></div>}
        </div>
      )}

      {d.kind === "op" && d.op && <OpConfig op={d.op} cols={cols} bCols={bCols} patchOp={patchOp} colSel={colSel} />}

      {d.kind === "sql" && <>
        <div className="insp-field"><div className="k">SQL — query the source as <code>raw</code>{bCols.length ? <>, Source B as <code>b</code></> : null}</div><textarea rows={7} value={d.sqlText ?? DEFAULT_SQL} onChange={(e) => patchSel({ sqlText: e.target.value })} style={{ fontFamily: "var(--mono)", fontSize: 11.5 }} /></div>
        <button className="btn sm" onClick={runSqlNode}>▶ Run SQL</button>
        <div className="teach-note" style={{ marginTop: 8 }}><span className="ic">🧮</span><span>Real in-browser SQL over your rows (SELECT / WHERE / GROUP BY / JOIN). This is the <b>ELT</b> pattern — land raw, transform here. Result flows to the Output panel.</span></div>
      </>}

      {d.kind === "load" && (
        <>
          <div className="note" style={{ margin: "4px 0 10px" }}>Target: <b>{d.target}</b></div>
          {(d.target === "platform" || d.target === "csv" || d.target === "xlsx") && <button className="btn sm" onClick={runLoad}>▶ Run load</button>}
          {isExternalDb && (() => {
            const coming = d.target === "snowflake" || d.target === "bigquery";
            return <>
              {coming ? (
                <div className="teach-note" style={{ marginTop: 8 }}><span className="ic">🔒</span><span><b>{d.target}</b> connector lands next.</span></div>
              ) : (
                <>
                  <div className="insp-field"><div className="k">Host / URI</div><input type="text" value={d.extHost ?? ""} placeholder={d.target === "mongodb" ? "mongodb+srv://..." : "db.example.com"} onChange={(e) => patchSel({ extHost: e.target.value })} /></div>
                  {d.target !== "mongodb" && <div className="insp-field"><div className="k">Port</div><input type="text" value={d.extPort ?? ""} placeholder={d.target === "postgres" || d.target === "redshift" ? "5432" : "3306"} onChange={(e) => patchSel({ extPort: e.target.value })} /></div>}
                  <div className="insp-field"><div className="k">Database</div><input type="text" value={d.extDb ?? ""} placeholder="production_db" onChange={(e) => patchSel({ extDb: e.target.value })} /></div>
                  {d.target !== "mongodb" && <div className="insp-field"><div className="k">Username</div><input type="text" value={d.extUser ?? ""} onChange={(e) => patchSel({ extUser: e.target.value })} /></div>}
                  {d.target !== "mongodb" && <div className="insp-field"><div className="k">Password</div><input type="password" value={d.extPass ?? ""} onChange={(e) => patchSel({ extPass: e.target.value })} /></div>}
                  
                  <div className="insp-field"><div className="k">{d.target === "mongodb" ? "Target collection" : "Target table"}</div><input type="text" value={d.extTable ?? "etl_output"} onChange={(e) => patchSel({ extTable: e.target.value })} /></div>
                  <div className="insp-field"><div className="k">Mode</div><select value={d.extMode ?? "append"} onChange={(e) => patchSel({ extMode: e.target.value })}><option value="append">append</option><option value="upsert">upsert (on first column)</option></select></div>
                  <button className="btn sm" onClick={() => {
                    let url = "";
                    if (d.target === "mongodb") {
                      url = d.extHost || "";
                      if (!url.includes("://")) url = "mongodb+srv://" + url;
                    } else {
                      url = `${d.target}://${d.extUser || ""}:${d.extPass || ""}@${d.extHost || ""}:${d.extPort || (d.target==="mysql" ? "3306" : "5432")}/${d.extDb || ""}`;
                    }
                    patchSel({ extUrl: url });
                    setTimeout(runExternal, 0);
                  }}>▶ Load to my DB</button>
                  <div className="teach-note" style={{ marginTop: 8 }}><span className="ic">🛢</span><span>Loads into your <b>{d.target}</b> — creates the {d.target === "mongodb" ? "collection" : "table"} if missing, {d.extMode === "upsert" ? "upserts on the first column" : "appends rows"}.</span></div>
                </>
              )}
            </>;
          })()}
        </>
      )}

      {d.kind === "analytics" && <>
        <div className="note" style={{ marginTop: 6, marginBottom: 8 }}>Builds a Power BI-style dashboard (KPIs + charts) from the live pipeline output.</div>
        <button className="btn sm" onClick={openDash}>📊 Open dashboard</button>
      </>}

      {d.op && OP_INFO[d.op.type] && <Reference op={d.op.type} />}
    </div>
  );
}

function OpConfig({ op, cols, bCols, patchOp, colSel }: { op: EtlOp; cols: string[]; bCols: string[]; patchOp: (p: Partial<EtlOp>) => void; colSel: (v: string | undefined, on: (v: string) => void, list?: string[]) => React.ReactNode }) {
  const o = op;
  const checklist = (sel: string[], on: (next: string[]) => void, list = cols) => <div className="checklist">{list.map((c) => <span key={c} className={`chk ${sel.includes(c) ? "on" : ""}`} onClick={() => on(sel.includes(c) ? sel.filter((x) => x !== c) : [...sel, c])}>{c}</span>)}</div>;
  return (
    <>
      {o.type === "filter" && <>
        <div className="insp-field"><div className="k">Column</div>{colSel(o.col, (v) => patchOp({ col: v }))}</div>
        <div className="insp-field"><div className="k">Operator</div><select value={o.op} onChange={(e) => patchOp({ op: e.target.value })}>{["==", "!=", ">", "<", ">=", "<=", "contains"].map((x) => <option key={x}>{x}</option>)}</select></div>
        <div className="insp-field"><div className="k">Value</div><input type="text" value={o.value ?? ""} onChange={(e) => patchOp({ value: e.target.value })} /></div>
      </>}
      {o.type === "select" && <div className="insp-field"><div className="k">Keep columns</div><div className="checklist">{cols.map((c) => <span key={c} className={`chk ${(o.cols || []).includes(c) ? "on" : ""}`} onClick={() => patchOp({ cols: (o.cols || []).includes(c) ? (o.cols || []).filter((x) => x !== c) : [...(o.cols || []), c] })}>{c}</span>)}</div></div>}
      {o.type === "aggregate" && <>
        <div className="insp-field"><div className="k">Group by</div>{colSel(o.groupBy, (v) => patchOp({ groupBy: v }))}</div>
        <div className="insp-field"><div className="k">Aggregate</div><select value={o.agg} onChange={(e) => patchOp({ agg: e.target.value })}>{["count", "sum", "avg", "min", "max"].map((x) => <option key={x}>{x}</option>)}</select></div>
        {o.agg !== "count" && <div className="insp-field"><div className="k">Of column</div>{colSel(o.aggCol, (v) => patchOp({ aggCol: v }))}</div>}
      </>}
      {o.type === "sort" && <>
        <div className="insp-field"><div className="k">Column</div>{colSel(o.col, (v) => patchOp({ col: v }))}</div>
        <div className="insp-field"><div className="k">Direction</div><select value={o.dir} onChange={(e) => patchOp({ dir: e.target.value })}><option value="asc">ascending</option><option value="desc">descending</option></select></div>
      </>}
      {o.type === "clean" && <div className="insp-field"><div className="k">Mode</div><select value={o.mode} onChange={(e) => patchOp({ mode: e.target.value })}><option value="dropnull">drop nulls</option><option value="fill0">fill 0</option></select></div>}
      {o.type === "limit" && <div className="insp-field"><div className="k">First N rows</div><input type="text" value={o.value ?? ""} onChange={(e) => patchOp({ value: e.target.value })} /></div>}
      {o.type === "rename" && <>
        <div className="insp-field"><div className="k">Column</div>{colSel(o.col, (v) => patchOp({ col: v }))}</div>
        <div className="insp-field"><div className="k">New name</div><input type="text" value={o.name ?? ""} onChange={(e) => patchOp({ name: e.target.value })} /></div>
      </>}
      {o.type === "derive" && <>
        <div className="insp-field"><div className="k">New column</div><input type="text" value={o.name ?? ""} onChange={(e) => patchOp({ name: e.target.value })} /></div>
        <div className="insp-field"><div className="k">Left</div><input type="text" value={o.left ?? ""} onChange={(e) => patchOp({ left: e.target.value })} /></div>
        <div className="insp-field"><div className="k">Op</div><select value={o.arith} onChange={(e) => patchOp({ arith: e.target.value })}>{["+", "-", "*", "/"].map((x) => <option key={x}>{x}</option>)}</select></div>
        <div className="insp-field"><div className="k">Right</div><input type="text" value={o.right ?? ""} onChange={(e) => patchOp({ right: e.target.value })} /></div>
      </>}
      {o.type === "dedupe" && <div className="insp-field"><div className="k">Unique by (empty = whole row)</div>{checklist(o.cols || [], (n) => patchOp({ cols: n }))}</div>}
      {o.type === "sample" && <div className="insp-field"><div className="k">Fraction (≤1) or count</div><input type="text" value={o.value ?? ""} onChange={(e) => patchOp({ value: e.target.value })} /></div>}
      {o.type === "map" && <>
        <div className="insp-field"><div className="k">Column</div>{colSel(o.col, (v) => patchOp({ col: v }))}</div>
        <div className="insp-field"><div className="k">Function</div><select value={o.fn} onChange={(e) => patchOp({ fn: e.target.value })}>{["round", "abs", "floor", "ceil", "upper", "lower", "trim", "length"].map((x) => <option key={x}>{x}</option>)}</select></div>
      </>}
      {o.type === "fillna" && <>
        <div className="insp-field"><div className="k">Column</div><select value={o.col ?? ""} onChange={(e) => patchOp({ col: e.target.value })}><option value="">(all columns)</option>{cols.map((c) => <option key={c}>{c}</option>)}</select></div>
        <div className="insp-field"><div className="k">Fill value</div><input type="text" value={o.value ?? ""} onChange={(e) => patchOp({ value: e.target.value })} /></div>
      </>}
      {o.type === "bucket" && <>
        <div className="insp-field"><div className="k">Numeric column</div>{colSel(o.col, (v) => patchOp({ col: v }))}</div>
        <div className="insp-field"><div className="k">New column name</div><input type="text" value={o.name ?? ""} onChange={(e) => patchOp({ name: e.target.value })} /></div>
        <div className="insp-field"><div className="k">Number of buckets</div><input type="text" value={o.value ?? ""} onChange={(e) => patchOp({ value: e.target.value })} /></div>
      </>}
      {["join", "lookup", "merge"].includes(o.type) && <>
        {!bCols.length && <div className="warnbar" style={{ marginBottom: 8 }}>Add a <b>second source</b> node (Source B) to join against.</div>}
        {o.type === "join" && <div className="insp-field"><div className="k">Join type</div><select value={o.joinType} onChange={(e) => patchOp({ joinType: e.target.value })}><option value="inner">inner</option><option value="left">left</option><option value="right">right</option></select></div>}
        <div className="insp-field"><div className="k">Left key (this table)</div>{colSel(o.col, (v) => patchOp({ col: v }))}</div>
        <div className="insp-field"><div className="k">Right key (Source B)</div>{colSel(o.rightKey, (v) => patchOp({ rightKey: v }), bCols)}</div>
      </>}
      {["union", "append"].includes(o.type) && <>
        {!bCols.length && <div className="warnbar" style={{ marginBottom: 8 }}>Add a <b>second source</b> node (Source B) to union with.</div>}
        <div className="insp-field"><div className="k">Mode</div><select value={o.mode} onChange={(e) => patchOp({ mode: e.target.value })}><option value="all">union all (keep dupes)</option><option value="distinct">distinct</option></select></div>
      </>}
      {o.type === "pivot" && <>
        <div className="insp-field"><div className="k">Index (rows)</div>{colSel(o.groupBy, (v) => patchOp({ groupBy: v }))}</div>
        <div className="insp-field"><div className="k">Spread column → new cols</div>{colSel(o.col, (v) => patchOp({ col: v }))}</div>
        <div className="insp-field"><div className="k">Value column</div>{colSel(o.aggCol, (v) => patchOp({ aggCol: v }))}</div>
        <div className="insp-field"><div className="k">Aggregate</div><select value={o.agg} onChange={(e) => patchOp({ agg: e.target.value })}>{["sum", "avg", "min", "max", "count"].map((x) => <option key={x}>{x}</option>)}</select></div>
      </>}
      {o.type === "unpivot" && <>
        <div className="insp-field"><div className="k">Columns to melt</div>{checklist(o.cols || [], (n) => patchOp({ cols: n }))}</div>
        <div className="insp-field"><div className="k">Variable column name</div><input type="text" value={o.name ?? ""} onChange={(e) => patchOp({ name: e.target.value })} /></div>
        <div className="insp-field"><div className="k">Value column name</div><input type="text" value={o.value ?? ""} onChange={(e) => patchOp({ value: e.target.value })} /></div>
      </>}
      {o.type === "window" && <>
        <div className="insp-field"><div className="k">Partition by</div><select value={o.groupBy} onChange={(e) => patchOp({ groupBy: e.target.value })}><option value="(none)">(whole table)</option>{cols.map((c) => <option key={c}>{c}</option>)}</select></div>
        <div className="insp-field"><div className="k">Column</div>{colSel(o.col, (v) => patchOp({ col: v }))}</div>
        <div className="insp-field"><div className="k">Function</div><select value={o.fn} onChange={(e) => patchOp({ fn: e.target.value })}><option value="running_sum">running sum</option><option value="row_number">row number</option><option value="rank">rank</option><option value="lag">lag</option><option value="lead">lead</option></select></div>
        <div className="insp-field"><div className="k">Output column</div><input type="text" value={o.name ?? ""} onChange={(e) => patchOp({ name: e.target.value })} /></div>
      </>}
      {o.type === "regex" && <>
        <div className="insp-field"><div className="k">Column</div>{colSel(o.col, (v) => patchOp({ col: v }))}</div>
        <div className="insp-field"><div className="k">Pattern (first group)</div><input type="text" value={o.value ?? ""} onChange={(e) => patchOp({ value: e.target.value })} /></div>
        <div className="insp-field"><div className="k">Output column</div><input type="text" value={o.name ?? ""} onChange={(e) => patchOp({ name: e.target.value })} /></div>
      </>}
      {o.type === "dateparse" && <>
        <div className="insp-field"><div className="k">Date column</div>{colSel(o.col, (v) => patchOp({ col: v }))}</div>
        <div className="insp-field"><div className="k">Extract</div><select value={o.fn} onChange={(e) => patchOp({ fn: e.target.value })}><option value="year">year</option><option value="month">month</option><option value="day">day</option><option value="weekday">weekday</option><option value="iso">ISO date</option><option value="days_since">days since</option></select></div>
        <div className="insp-field"><div className="k">Output column</div><input type="text" value={o.name ?? ""} onChange={(e) => patchOp({ name: e.target.value })} /></div>
      </>}
    </>
  );
}

function Reference({ op }: { op: OpType }) {
  const info = OP_INFO[op];
  return (
    <div style={{ marginTop: 12, borderTop: "1px solid var(--border)", paddingTop: 10 }}>
      <div className="note" style={{ lineHeight: 1.55, marginBottom: 9 }}>{info.definition}</div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
        <div style={{ border: "1px solid var(--border)", borderRadius: 9, padding: "7px 9px" }}><div style={{ fontSize: 9.5, fontWeight: 600, color: "var(--good)", marginBottom: 4 }}>✓ Strengths</div>{info.advantages.map((a) => <div key={a} style={{ fontSize: 10, color: "var(--muted)" }}>{a}</div>)}</div>
        <div style={{ border: "1px solid var(--border)", borderRadius: 9, padding: "7px 9px" }}><div style={{ fontSize: 9.5, fontWeight: 600, color: "var(--crit)", marginBottom: 4 }}>✕ Watch-outs</div>{info.disadvantages.map((a) => <div key={a} style={{ fontSize: 10, color: "var(--muted)" }}>{a}</div>)}</div>
      </div>
    </div>
  );
}

// Animated node-by-node walkthrough: each node runs row by row, its input is the
// previous node's output, and it ends on the final output with a raw-vs-final compare.
function PlayModal({ raw, nodes, final, onClose }: { raw: Table; nodes: PlayNode[]; final: Table; onClose: () => void }) {
  const [ni, setNi] = useState(0);
  const [cur, setCur] = useState(-1);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(950);
  const summary = ni >= nodes.length; // one past the last node = final output view
  const node = summary ? null : nodes[ni];
  useEffect(() => {
    if (!playing) return;
    if (summary) { setPlaying(false); return; }
    const tm = setTimeout(() => {
      setCur((c) => {
        if (node && c < node.steps - 1) return c + 1;
        setNi((n) => n + 1); return -1;
      });
    }, speed);
    return () => clearTimeout(tm);
  }, [playing, cur, ni, speed, node, summary]);
  const inRef = useRef<HTMLDivElement>(null);
  useEffect(() => { const el = inRef.current?.querySelector(".play-cur") as HTMLElement | null; el?.scrollIntoView({ block: "nearest" }); }, [ni, cur]);
  const restart = () => { setPlaying(false); setNi(0); setCur(-1); };
  const stepFwd = () => { setPlaying(false); if (summary) return; if (node && cur < node.steps - 1) setCur(cur + 1); else { setNi(ni + 1); setCur(-1); } };
  const jumpNode = (i: number) => { setPlaying(false); setNi(i); setCur(-1); };
  const totalSteps = nodes.reduce((a, n) => a + Math.max(1, n.steps), 0) || 1;
  const doneSteps = nodes.slice(0, ni).reduce((a, n) => a + Math.max(1, n.steps), 0) + (summary ? 0 : Math.max(0, cur + 1));
  const pct = summary ? 100 : (doneSteps / totalSteps) * 100;
  const chip = (label: string, sub: string, i: number, active: boolean, doneN: boolean, onClick?: () => void) => (
    <button key={i} onClick={onClick} disabled={!onClick} style={{ flex: "0 0 auto", display: "flex", flexDirection: "column", alignItems: "flex-start", gap: 1, padding: "6px 11px", borderRadius: 10, cursor: onClick ? "pointer" : "default", border: `1px solid ${active ? "var(--accent)" : doneN ? "rgba(62,207,127,.4)" : "var(--border)"}`, background: active ? "color-mix(in srgb, var(--accent) 14%, transparent)" : doneN ? "color-mix(in srgb, var(--good) 12%, transparent)" : "var(--panel-2)" }}>
      <b style={{ fontSize: 11.5, color: active ? "var(--text)" : "var(--muted)" }}>{label}</b><span style={{ fontSize: 9.5, color: "var(--faint)" }}>{sub}</span>
    </button>
  );
  const tbl = (t: Table, rowCls?: (i: number) => string, hlRow = -1) => (
    <div style={{ overflowX: "auto", maxHeight: 300 }}><table className="dtable"><tbody>
      <tr>{t.cols.map((c) => <th key={c}>{c}</th>)}</tr>
      {t.rows.slice(0, PLAY_CAP).map((r, i) => <tr key={i} className={rowCls ? rowCls(i) : ""} style={i === hlRow ? { outline: "1px solid var(--accent)", background: "color-mix(in srgb, var(--accent) 14%, transparent)" } : undefined}>{t.cols.map((c) => <td key={c}>{r[c] == null ? "—" : String(r[c])}</td>)}</tr>)}
    </tbody></table></div>
  );
  const rowClsName = (s: string) => (s === "keep" ? "play-keep" : s === "drop" ? "play-drop" : s === "cur" ? "play-cur" : s === "pend" ? "play-pend" : "");
  const outTable = node ? node.outAt(cur) : final;
  const rowD = final.rows.length - raw.rows.length, colD = final.cols.length - raw.cols.length;
  return (
    <div onClick={onClose} style={{ position: "fixed", inset: 0, zIndex: 1000, background: "rgba(4,6,12,.78)", backdropFilter: "blur(4px)", display: "flex", alignItems: "center", justifyContent: "center", padding: 20 }}>
      <div onClick={(e) => e.stopPropagation()} style={{ width: "min(1080px, 97vw)", maxHeight: "95vh", overflow: "auto", background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 14, boxShadow: "0 20px 60px rgba(0,0,0,.5)" }}>
        <div className="row" style={{ alignItems: "center", justifyContent: "space-between", padding: "12px 16px", borderBottom: "1px solid var(--border)" }}>
          <b style={{ fontSize: 14 }}>▶ Pipeline walkthrough <span className="note" style={{ fontWeight: 400 }}>· node by node, row by row</span></b>
          <div className="row" style={{ gap: 6, alignItems: "center" }}>
            <button className="btn ghost sm" onClick={restart} title="Restart">⏮</button>
            <button className="btn ghost sm" onClick={stepFwd} disabled={summary} title="Step one row">›</button>
            <button className="btn sm" onClick={() => { if (summary) restart(); setPlaying((p) => !p); }}>{playing ? "⏸ Pause" : summary ? "↻ Replay" : "▶ Play"}</button>
            <span className="note" style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>speed<select value={speed} onChange={(e) => setSpeed(+e.target.value)}><option value={1600}>0.5×</option><option value={950}>1×</option><option value={500}>2×</option></select></span>
            <button className="btn ghost sm" onClick={onClose}>Close</button>
          </div>
        </div>
        <div style={{ height: 3, background: "var(--panel-2)" }}><div style={{ width: `${pct}%`, height: "100%", background: "var(--accent)", transition: "width .35s ease" }} /></div>
        {/* node chain */}
        <div className="row" style={{ gap: 6, padding: "10px 14px", overflowX: "auto", borderBottom: "1px solid var(--border)" }}>
          {chip("Raw", `${raw.rows.length} rows`, -1, false, true)}
          <span style={{ color: "var(--faint)", alignSelf: "center" }}>›</span>
          {nodes.map((n, i) => <span key={i} style={{ display: "flex", alignItems: "center", gap: 6 }}>{chip(`${i + 1} ${n.label}`, node && ni === i ? "running" : "", i, ni === i, ni > i, () => jumpNode(i))}{i < nodes.length - 1 && <span style={{ color: "var(--faint)" }}>›</span>}</span>)}
          <span style={{ color: "var(--faint)", alignSelf: "center" }}>›</span>
          {chip("Output", `${final.rows.length} rows`, -2, summary, summary, () => jumpNode(nodes.length))}
        </div>

        <div style={{ padding: "12px 16px" }}>
          {!summary && node ? (<>
            <div style={{ fontSize: 15, fontWeight: 600 }}>Node {ni + 1} · {node.label} <span style={{ fontWeight: 400, fontSize: 12.5, color: "var(--muted)", fontFamily: "var(--mono)" }}>· {node.rule}</span></div>
            <div style={{ marginTop: 8, padding: "9px 13px", borderRadius: 9, background: "var(--panel-2)", border: "1px solid var(--border-strong)", fontSize: 13, fontFamily: "var(--mono)", fontWeight: 600, color: "var(--muted)", minHeight: 22 }}>{(() => { const s = node.cap(cur); const a = s.lastIndexOf("→"); if (a < 0) return s; const head = s.slice(0, a + 1), tail = s.slice(a + 1); const col = /keep/i.test(tail) ? "#5fe0a0" : /drop/i.test(tail) ? "#f0808b" : "var(--muted)"; return <>{head}<span style={{ color: col, fontWeight: 700 }}>{tail}</span></>; })()}</div>
            <div className="split" style={{ gridTemplateColumns: "1fr 26px 1fr", gap: 8, marginTop: 10, alignItems: "start" }}>
              <div ref={inRef} style={{ border: "1px solid rgba(245,158,11,.35)", borderRadius: 10, overflow: "hidden" }}>
                <div className="note" style={{ padding: "7px 10px", borderBottom: "1px solid var(--border)", fontSize: 10.5, display: "flex", justifyContent: "space-between" }}><span style={{ color: "#f0b45f" }}>Input · {ni === 0 ? "raw source" : `← output of ${nodes[ni - 1].label}`}</span><span>{node.input.rows.length}{node.capped ? "+" : ""}</span></div>
                <div key={`in-${ni}`} className="etl-stage-in">{tbl(node.input, (i) => rowClsName(node.hlOut ? "" : node.state(i, cur)))}</div>
              </div>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "center", paddingTop: 40, color: "var(--faint)" }}>→</div>
              <div style={{ border: "1px solid rgba(62,207,127,.35)", borderRadius: 10, overflow: "hidden" }}>
                <div className="note" style={{ padding: "7px 10px", borderBottom: "1px solid var(--border)", fontSize: 10.5, display: "flex", justifyContent: "space-between" }}><span style={{ color: "var(--good)" }}>Output {ni < nodes.length - 1 ? `→ feeds ${nodes[ni + 1].label}` : "→ final output"}</span><span>{outTable.rows.length}</span></div>
                <div key={`out-${ni}-${cur}`} className="etl-stage-in">{outTable.rows.length ? tbl(outTable, undefined, node.hlOut ? cur : -1) : <div className="note" style={{ padding: 12 }}>building…</div>}</div>
              </div>
            </div>
          </>) : (<>
            <div style={{ fontSize: 15, fontWeight: 600 }}>Final output <span className="note" style={{ fontWeight: 400 }}>· pipeline complete</span></div>
            <div className="note" style={{ marginTop: 2 }}>Raw {raw.rows.length} rows → after {nodes.length} node{nodes.length === 1 ? "" : "s"} → {final.rows.length} rows × {final.cols.length} cols.</div>
            <div key="final" className="etl-stage-in" style={{ marginTop: 10, border: "1px solid var(--accent)", borderRadius: 10, overflow: "hidden" }}>{tbl(final)}</div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr auto 1fr", gap: 12, marginTop: 14, alignItems: "center" }}>
              <div style={{ border: "1px solid var(--border)", borderRadius: 10, padding: "10px 12px" }}><div className="note" style={{ fontSize: 10, textTransform: "uppercase" }}>Raw input</div><div style={{ fontSize: 19, fontWeight: 600, marginTop: 4 }}>{raw.rows.length} rows</div><div className="note">{raw.cols.length} cols</div></div>
              <div style={{ textAlign: "center" }}><div className="note" style={{ fontSize: 10 }}>Δ rows</div><div style={{ fontSize: 17, fontWeight: 600, color: rowD < 0 ? "var(--crit)" : rowD > 0 ? "var(--good)" : "var(--muted)" }}>{rowD > 0 ? "+" : ""}{rowD}</div><div className="note" style={{ marginTop: 4 }}>Δ cols {colD > 0 ? "+" : ""}{colD}</div></div>
              <div style={{ border: "1px solid var(--accent)", borderRadius: 10, padding: "10px 12px", background: "color-mix(in srgb, var(--accent) 8%, transparent)" }}><div className="note" style={{ fontSize: 10, textTransform: "uppercase" }}>Final output</div><div style={{ fontSize: 19, fontWeight: 600, marginTop: 4 }}>{final.rows.length} rows</div><div className="note">{final.cols.length} cols</div></div>
            </div>
          </>)}
        </div>
      </div>
    </div>
  );
}

// Interactive dashboard modal: column pickers (dimension + measure), a slicer
// (filter), and the 4-chart SVG. Download rasterizes the current view to PNG.
function DashboardModal({ table, title, onClose }: { table: Table; title: string; onClose: () => void }) {
  const catCols = useMemo(() => table.cols.filter((c) => table.rows.some((r) => typeof r[c] === "string")), [table]);
  const numCols = useMemo(() => table.cols.filter((c) => table.rows.some((r) => typeof r[c] === "number")), [table]);
  const [dim, setDim] = useState(catCols[0] || table.cols[0] || "");
  const [measure, setMeasure] = useState(numCols[0] || "");
  const [slicerCol, setSlicerCol] = useState(catCols[1] || catCols[0] || "");
  const [slicer, setSlicer] = useState<Set<string>>(new Set());
  const slicerVals = useMemo(() => Array.from(new Set(table.rows.map((r) => String(r[slicerCol] ?? "—")))).slice(0, 14), [table, slicerCol]);
  const filtered = useMemo<Table>(() => (slicer.size ? { cols: table.cols, rows: table.rows.filter((r) => slicer.has(String(r[slicerCol] ?? "—"))) } : table), [table, slicer, slicerCol]);
  const svg = useMemo(() => buildDashboardSvg(filtered, dim, measure, title), [filtered, dim, measure, title]);
  const toggle = (v: string) => setSlicer((s) => { const n = new Set(s); if (n.has(v)) n.delete(v); else n.add(v); return n; });
  function downloadPng() {
    const url = URL.createObjectURL(new Blob([svg], { type: "image/svg+xml;charset=utf-8" }));
    const img = new Image();
    img.onload = () => { const c = document.createElement("canvas"); c.width = 960 * 2; c.height = 748 * 2; const ctx = c.getContext("2d"); if (!ctx) return; ctx.scale(2, 2); ctx.drawImage(img, 0, 0); c.toBlob((b) => { const a = document.createElement("a"); a.href = b ? URL.createObjectURL(b) : url; a.download = b ? "dashboard.png" : "dashboard.svg"; a.click(); URL.revokeObjectURL(url); }, "image/png"); };
    img.onerror = () => { const a = document.createElement("a"); a.href = url; a.download = "dashboard.svg"; a.click(); };
    img.src = url;
  }
  const lbl: React.CSSProperties = { fontSize: 9, letterSpacing: ".05em", textTransform: "uppercase", color: "var(--faint)", marginBottom: 4 };
  return (
    <div onClick={onClose} style={{ position: "fixed", inset: 0, zIndex: 1000, background: "rgba(4,6,12,.76)", backdropFilter: "blur(4px)", display: "flex", alignItems: "center", justifyContent: "center", padding: 20 }}>
      <div onClick={(e) => e.stopPropagation()} style={{ width: "min(1050px, 97vw)", maxHeight: "94vh", overflow: "auto", background: "#0e121d", border: "1px solid var(--border)", borderRadius: 14, boxShadow: "0 20px 60px rgba(0,0,0,.5)" }}>
        <div className="row" style={{ alignItems: "center", justifyContent: "space-between", padding: "12px 16px", borderBottom: "1px solid var(--border)" }}>
          <b style={{ fontSize: 14 }}>📊 {title}</b>
          <div className="row" style={{ gap: 8 }}><button className="btn sm" onClick={downloadPng}>⬇ Download image</button><button className="btn ghost sm" onClick={onClose}>Close</button></div>
        </div>
        <div style={{ padding: "12px 16px", display: "flex", gap: 14, flexWrap: "wrap", alignItems: "flex-end", borderBottom: "1px solid var(--border)" }}>
          <div><div style={lbl}>Dimension</div><select value={dim} onChange={(e) => setDim(e.target.value)} style={{ minWidth: 130 }}>{table.cols.map((c) => <option key={c}>{c}</option>)}</select></div>
          <div><div style={lbl}>Measure</div><select value={measure} onChange={(e) => setMeasure(e.target.value)} style={{ minWidth: 130 }}><option value="">(row count)</option>{numCols.map((c) => <option key={c}>{c}</option>)}</select></div>
          <div><div style={lbl}>Slicer field</div><select value={slicerCol} onChange={(e) => { setSlicerCol(e.target.value); setSlicer(new Set()); }} style={{ minWidth: 130 }}>{catCols.map((c) => <option key={c}>{c}</option>)}</select></div>
          <div style={{ flex: 1, minWidth: 220 }}><div style={lbl}>Slicer · {slicerCol} {slicer.size ? `(${slicer.size})` : ""}</div><div className="row" style={{ gap: 6, flexWrap: "wrap", alignItems: "center" }}>{slicerVals.map((v) => <button key={v} onClick={() => toggle(v)} style={{ fontSize: 11, padding: "4px 10px", borderRadius: 7, cursor: "pointer", border: `1px solid ${slicer.has(v) ? "var(--accent)" : "var(--border)"}`, background: slicer.has(v) ? "color-mix(in srgb, var(--accent) 16%, transparent)" : "var(--panel-2)", color: slicer.has(v) ? "var(--text)" : "var(--muted)" }}>{v.length > 14 ? v.slice(0, 13) + "…" : v}</button>)}{slicer.size > 0 && <button className="btn ghost sm" onClick={() => setSlicer(new Set())}>clear</button>}</div></div>
        </div>
        <div style={{ padding: 14 }} dangerouslySetInnerHTML={{ __html: svg.replace('width="960" height="748"', 'style="width:100%;height:auto;display:block"') }} />
      </div>
    </div>
  );
}

// Build a self-contained Power BI-style dashboard SVG: KPIs + 4 charts (bar,
// donut, line, count) driven by the chosen dimension + measure. Hardcoded
// colors (no CSS vars) so it rasterizes to PNG cleanly.
const DASH_PAL = ["#5b7cff", "#a855f7", "#14b8a6", "#f59e0b", "#ec4899", "#10b981", "#38bdf8"];
function buildDashboardSvg(table: Table, dim: string, measure: string, title = "Pipeline dashboard"): string {
  const W = 960, H = 748;
  const rows = table.rows, cols = table.cols;
  const hasM = !!measure && cols.includes(measure);
  const nf = (n: number) => (Math.abs(n) >= 1000 ? Math.round(n).toLocaleString() : String(Math.round(n * 100) / 100));
  const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const trunc = (s: string, n: number) => (s.length > n ? s.slice(0, n - 1) + "…" : s);
  const sumM = hasM ? rows.reduce((a, r) => a + (Number(r[measure]) || 0), 0) : rows.length;
  const distinct = new Set(rows.map((r) => String(r[dim]))).size;
  const group = (valFn: (r: Record<string, unknown>) => number, n = 8): [string, number][] => { const m: Record<string, number> = {}; rows.forEach((r) => { const k = String(r[dim] ?? "—"); m[k] = (m[k] || 0) + valFn(r); }); return Object.entries(m).sort((a, b) => b[1] - a[1]).slice(0, n); };
  const gSum = group((r) => (hasM ? Number(r[measure]) || 0 : 1));
  const gCount = group(() => 1);
  const kpis = [
    { label: "Rows", value: nf(rows.length), color: DASH_PAL[0] },
    { label: hasM ? "Sum of " + measure : "Columns", value: hasM ? nf(sumM) : String(cols.length), color: DASH_PAL[1] },
    { label: hasM ? "Avg " + measure : "Distinct " + dim, value: hasM ? nf(sumM / (rows.length || 1)) : nf(distinct), color: DASH_PAL[2] },
    { label: "Distinct " + dim, value: nf(distinct), color: DASH_PAL[3] },
  ];
  const panel = (x: number, y: number, w: number, h: number, t: string) => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="12" fill="#131824" stroke="#2a3348"/><text x="${x + 18}" y="${y + 26}" fill="#e8ecf6" font-size="13" font-family="sans-serif">${esc(t)}</text>`;
  const hbars = (e: [string, number][], x: number, y: number, w: number, color: string) => {
    const max = Math.max(1, ...e.map((v) => v[1])); const rh = 26, bx = x + 96, bw = w - 118;
    return e.slice(0, 7).map((v, i) => { const yy = y + i * rh, b = Math.max(2, (v[1] / max) * bw);
      return `<text x="${x}" y="${yy + 12}" fill="#c3ccdf" font-size="11.5" font-family="sans-serif">${esc(trunc(String(v[0]), 11))}</text><rect x="${bx}" y="${yy + 2}" width="${b.toFixed(1)}" height="13" rx="3" fill="${color}"/><text x="${(bx + b + 6).toFixed(1)}" y="${yy + 12}" fill="#9aa4bd" font-size="10.5" font-family="monospace">${nf(v[1])}</text>`;
    }).join("");
  };
  const donut = (e: [string, number][], cx: number, cy: number, r: number, ir: number) => {
    const tot = e.reduce((a, v) => a + v[1], 0) || 1; let a0 = -Math.PI / 2; const P = Math.PI;
    const arcs = e.slice(0, 6).map((v, i) => { const a1 = a0 + (v[1] / tot) * 2 * P; const col = DASH_PAL[i % DASH_PAL.length];
      const x0 = cx + r * Math.cos(a0), y0 = cy + r * Math.sin(a0), x1 = cx + r * Math.cos(a1), y1 = cy + r * Math.sin(a1);
      const xi1 = cx + ir * Math.cos(a1), yi1 = cy + ir * Math.sin(a1), xi0 = cx + ir * Math.cos(a0), yi0 = cy + ir * Math.sin(a0);
      const lg = a1 - a0 > P ? 1 : 0; a0 = a1;
      return `<path d="M${x0.toFixed(1)} ${y0.toFixed(1)} A${r} ${r} 0 ${lg} 1 ${x1.toFixed(1)} ${y1.toFixed(1)} L${xi1.toFixed(1)} ${yi1.toFixed(1)} A${ir} ${ir} 0 ${lg} 0 ${xi0.toFixed(1)} ${yi0.toFixed(1)} Z" fill="${col}"/>`;
    }).join("");
    const legend = e.slice(0, 6).map((v, i) => `<rect x="${cx + r + 20}" y="${cy - 44 + i * 20}" width="10" height="10" rx="2" fill="${DASH_PAL[i % DASH_PAL.length]}"/><text x="${cx + r + 36}" y="${cy - 35 + i * 20}" fill="#c3ccdf" font-size="11" font-family="sans-serif">${esc(trunc(String(v[0]), 10))} ${Math.round((v[1] / tot) * 100)}%</text>`).join("");
    return arcs + legend + `<text x="${cx}" y="${cy + 4}" text-anchor="middle" fill="#e8ecf6" font-size="15" font-weight="600" font-family="sans-serif">${nf(tot)}</text>`;
  };
  const line = (e: [string, number][], x: number, y: number, w: number, h: number, color: string) => {
    const pts = e.slice(0, 10); const max = Math.max(1, ...pts.map((v) => v[1])); const min = Math.min(...pts.map((v) => v[1]), 0);
    const px = (i: number) => x + 34 + (pts.length <= 1 ? 0 : (i / (pts.length - 1)) * (w - 50));
    const py = (val: number) => y + h - 28 - ((val - min) / (max - min || 1)) * (h - 50);
    const path = pts.map((v, i) => `${i ? "L" : "M"}${px(i).toFixed(1)} ${py(v[1]).toFixed(1)}`).join(" ");
    const area = `${path} L${px(pts.length - 1).toFixed(1)} ${(y + h - 28).toFixed(1)} L${px(0).toFixed(1)} ${(y + h - 28).toFixed(1)} Z`;
    const dots = pts.map((v, i) => `<circle cx="${px(i).toFixed(1)}" cy="${py(v[1]).toFixed(1)}" r="3" fill="${color}"/>`).join("");
    const xl = pts.map((v, i) => `<text x="${px(i).toFixed(1)}" y="${y + h - 10}" text-anchor="middle" fill="#6b748c" font-size="9" font-family="sans-serif">${esc(trunc(String(v[0]), 6))}</text>`).join("");
    return `<path d="${area}" fill="${color}" opacity="0.14"/><path d="${path}" fill="none" stroke="${color}" stroke-width="2.2"/>${dots}${xl}`;
  };
  const kpiTiles = kpis.map((k, i) => { const x = 24 + i * 232;
    return `<rect x="${x}" y="56" width="216" height="82" rx="10" fill="#151a28" stroke="#2a3348"/><text x="${x + 16}" y="82" fill="#9aa4bd" font-size="12" font-family="sans-serif">${esc(k.label)}</text><text x="${x + 16}" y="118" fill="${k.color}" font-size="25" font-weight="600" font-family="sans-serif">${esc(k.value)}</text>`;
  }).join("");
  return `<svg xmlns="http://www.w3.org/2000/svg" width="960" height="748" viewBox="0 0 ${W} ${H}">`
    + `<rect width="${W}" height="${H}" fill="#0e121d"/>`
    + `<text x="24" y="34" fill="#e8ecf6" font-size="18" font-weight="600" font-family="sans-serif">${esc(title)}</text>`
    + `<text x="${W - 24}" y="34" text-anchor="end" fill="#6b748c" font-size="12" font-family="monospace">${rows.length} rows · ${cols.length} cols · dim ${esc(dim)}${hasM ? " · measure " + esc(measure) : ""}</text>`
    + kpiTiles
    + panel(24, 156, 452, 272, `${hasM ? "Sum of " + measure : "Count"} by ${dim}`) + `<g>${hbars(gSum, 44, 196, 432, DASH_PAL[0])}</g>`
    + panel(484, 156, 452, 272, `Share by ${dim}`) + `<g>${donut(gSum, 600, 300, 78, 46)}</g>`
    + panel(24, 444, 452, 276, `${hasM ? measure : "Count"} trend by ${dim}`) + `<g>${line(gSum, 44, 460, 424, 246, DASH_PAL[4])}</g>`
    + panel(484, 444, 452, 276, `Count by ${dim}`) + `<g>${hbars(gCount, 504, 484, 432, DASH_PAL[2])}</g>`
    + `</svg>`;
}

// simple real bar chart from the output (count by first categorical column)
function AnalyticsView({ table }: { table: Table }) {
  const catCol = table.cols.find((c) => table.rows.some((r) => typeof r[c] === "string")) || table.cols[0];
  const numCol = table.cols.find((c) => table.rows.some((r) => typeof r[c] === "number"));
  const groups: Record<string, number> = {};
  table.rows.forEach((r) => { const k = String(r[catCol] ?? "—"); groups[k] = (groups[k] || 0) + (numCol ? Number(r[numCol]) || 0 : 1); });
  const entries = Object.entries(groups).sort((a, b) => b[1] - a[1]).slice(0, 8);
  const max = Math.max(1, ...entries.map((e) => e[1]));
  return (
    <div>
      <div className="note" style={{ marginBottom: 8 }}>{numCol ? `${numCol} by ${catCol}` : `count by ${catCol}`} — from the live output</div>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {entries.map(([k, v]) => (
          <div key={k} style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ width: 90, fontSize: 11, color: "var(--muted)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{k}</span>
            <div style={{ flex: 1, background: "var(--panel-2)", borderRadius: 5, height: 16 }}><div style={{ width: `${(v / max) * 100}%`, height: "100%", background: "#ec4899", borderRadius: 5 }} /></div>
            <span style={{ width: 56, textAlign: "right", fontSize: 11, fontFamily: "var(--mono)", color: "var(--muted)" }}>{Math.round(v * 100) / 100}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── NodeInfoModal (Detailed Reference Panel) ──
function NodeInfoModal({ node, onClose }: { node: FNode; onClose: () => void }) {
  const d = node.data;
  const c = d.color;
  let def = "", purp = "", adv: string[] = [], dis: string[] = [], speed = "", mem = "", scale = "", cost = "", py = "", sp = "";

  if (d.op && OP_INFO[d.op.type]) {
    const i = OP_INFO[d.op.type];
    def = i.definition; purp = i.purpose; adv = i.advantages; dis = i.disadvantages;
    speed = i.speed; mem = i.memory; scale = i.scalability; cost = i.cost; py = i.pandas; sp = i.spark;
  } else if (d.kind === "source") {
    let key: "sample" | "upload" | "json" | "db" | "" = "";
    if (["csv", "excel"].includes(d.srcType || "")) key = "upload";
    if (["postgres", "mysql"].includes(d.srcType || "")) key = "db";
    if (d.srcType === "json") key = "json";
    if (key && SRC_INFO[key]) {
      const i = SRC_INFO[key];
      def = i.definition; adv = i.advantages; dis = i.disadvantages; py = i.code;
    } else def = "Source node configuration.";
  } else if (d.kind === "load") {
    const key = d.target === "platform" ? "platform" : "external";
    const i = LOAD_INFO[key];
    if (i) { def = i.definition; adv = i.advantages; dis = i.disadvantages; py = i.code; }
  }

  const Section = ({ title, icon, children }: { title: string, icon: string, children: React.ReactNode }) => (
    <div style={{ marginBottom: 16 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, fontWeight: 700, letterSpacing: ".06em", color: "var(--muted)", textTransform: "uppercase", marginBottom: 8 }}>
        <span style={{ fontSize: 13 }}>{icon}</span> {title}
      </div>
      <div style={{ fontSize: 13, lineHeight: 1.5, color: "var(--text)" }}>{children}</div>
    </div>
  );

  return (
    <div onClick={onClose} style={{ position: "fixed", inset: 0, zIndex: 1000, background: "rgba(4,6,12,.78)", backdropFilter: "blur(4px)", display: "flex", justifyContent: "flex-end" }}>
      <div onClick={(e) => e.stopPropagation()} style={{ width: 420, height: "100%", background: "#0a0c10", borderLeft: "1px solid var(--border)", boxShadow: "-10px 0 30px rgba(0,0,0,.5)", display: "flex", flexDirection: "column" }}>
        
        {/* Header */}
        <div style={{ padding: "20px 24px", display: "flex", alignItems: "flex-start", gap: 16, borderBottom: "1px solid var(--border)" }}>
          <span style={{ width: 44, height: 44, borderRadius: 12, display: "grid", placeItems: "center", fontSize: 22, background: `color-mix(in srgb, ${c} 15%, transparent)`, color: c }}>{d.icon}</span>
          <div style={{ flex: 1 }}>
            <h2 style={{ fontSize: 22, fontWeight: 700, margin: "0 0 4px", color: "#fff" }}>{d.op ? OP_META[d.op.type].label : d.label}</h2>
            <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: ".1em", textTransform: "uppercase", color: c }}>{d.cat} Node</div>
          </div>
          <button onClick={onClose} style={{ background: "none", border: "none", color: "var(--muted)", cursor: "pointer", fontSize: 20 }}>×</button>
        </div>

        {/* Content */}
        <div style={{ flex: 1, overflowY: "auto", padding: "24px" }}>
          {def && <Section title="1. Definition" icon="💡">{def}</Section>}
          {purp && <Section title="2. Purpose" icon="🎯">{purp}</Section>}

          {speed && (
            <Section title="3. Performance Metrics" icon="⏱">
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {[
                  { label: "Speed", value: speed, icon: "⚡" },
                  { label: "Memory", value: mem, icon: "💾" },
                  { label: "Scalability", value: scale, icon: "📈" },
                  { label: "Cost", value: cost, icon: "💰" }
                ].map(m => (
                  <div key={m.label} style={{ display: "flex", justifyContent: "space-between", padding: "8px 12px", background: "rgba(255,255,255,.03)", border: "1px solid rgba(255,255,255,.06)", borderRadius: 8 }}>
                    <span style={{ color: "var(--muted)", display: "flex", alignItems: "center", gap: 6 }}><span>{m.icon}</span> {m.label}</span>
                    <strong style={{ color: "#fff" }}>{m.value}</strong>
                  </div>
                ))}
              </div>
            </Section>
          )}

          {(py || sp) && (
            <Section title="4. Examples" icon="💻">
              {py && (
                <div style={{ marginBottom: 12 }}>
                  <div style={{ fontSize: 10, color: "var(--muted)", marginBottom: 4 }}>PYTHON / PANDAS</div>
                  <pre style={{ background: "#000", padding: "10px 12px", borderRadius: 8, border: "1px solid var(--border)", fontSize: 12, color: "#4ade80", overflowX: "auto" }}>{py}</pre>
                </div>
              )}
              {sp && (
                <div>
                  <div style={{ fontSize: 10, color: "var(--muted)", marginBottom: 4 }}>APACHE SPARK</div>
                  <pre style={{ background: "#000", padding: "10px 12px", borderRadius: 8, border: "1px solid var(--border)", fontSize: 12, color: "#facc15", overflowX: "auto" }}>{sp}</pre>
                </div>
              )}
            </Section>
          )}

          {(adv.length > 0 || dis.length > 0) && (
            <Section title="5. Pros & Cons" icon="⚖">
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                <div style={{ background: "rgba(62,207,127,.05)", border: "1px solid rgba(62,207,127,.15)", borderRadius: 10, padding: "10px 12px" }}>
                  <div style={{ color: "var(--good)", fontSize: 10, fontWeight: 700, marginBottom: 8, textTransform: "uppercase" }}>Advantages</div>
                  <ul style={{ margin: 0, paddingLeft: 16, color: "var(--text)", fontSize: 12 }}>
                    {adv.map(a => <li key={a} style={{ marginBottom: 4 }}>{a}</li>)}
                  </ul>
                </div>
                <div style={{ background: "rgba(239,68,68,.05)", border: "1px solid rgba(239,68,68,.15)", borderRadius: 10, padding: "10px 12px" }}>
                  <div style={{ color: "var(--crit)", fontSize: 10, fontWeight: 700, marginBottom: 8, textTransform: "uppercase" }}>Disadvantages</div>
                  <ul style={{ margin: 0, paddingLeft: 16, color: "var(--text)", fontSize: 12 }}>
                    {dis.map(a => <li key={a} style={{ marginBottom: 4 }}>{a}</li>)}
                  </ul>
                </div>
              </div>
            </Section>
          )}
        </div>
      </div>
    </div>
  );
}

export default function EtlFlowLab() {
  return <ReactFlowProvider><Inner /></ReactFlowProvider>;
}
