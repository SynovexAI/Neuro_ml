"use client";

// RAG Lab — node-canvas edition. One guided flow (Source → Chunk → Index →
// Retrieve & Answer) built on a React Flow canvas with a step-gated palette.
// Index holds Vector store / Knowledge graph / Hybrid as nodes. The endpoint is
// a multi-turn grounded chat. Wired to the real ragUtils engine.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ReactFlow, ReactFlowProvider, Background, BackgroundVariant, Controls, MiniMap,
  Handle, Position, addEdge, useNodesState, useEdgesState, useReactFlow, MarkerType,
  type Node, type Edge, type Connection, type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import {
  chunkText, buildIndex, retrieve, retrieveDense, mmrRerank, retrievalMetrics, cosine, denseCos,
  type RagIndex, type Strategy, type Metric,
} from "@/lib/ragUtils";
import { extractGraph, retrieveGraph, layoutGraph, type KnowledgeGraph } from "@/lib/kgUtils";
import { toast, confirmDialog } from "@/lib/toast";

type Kind = "source" | "chunk" | "embed" | "index" | "retriever" | "mmr" | "llm" | "dashboard";
type Cat = Kind;
const COLOR: Record<Cat, string> = {
  source: "#3b82f6", chunk: "#f59e0b", embed: "#a855f7", index: "#14b8a6",
  retriever: "#ec4899", mmr: "#f472b6", llm: "#10b981", dashboard: "#db2777",
};
const CATLABEL: Record<Cat, string> = {
  source: "Source", chunk: "Chunk", embed: "Embed", index: "Index",
  retriever: "Retrieve", mmr: "Re-rank", llm: "Generate", dashboard: "Analytics",
};
const rid = () => Math.random().toString(36).slice(2, 9);
const SAMPLE = `Returns policy. Damaged items may be returned within 30 days of delivery for a full refund, provided the original packaging is included. Refunds are issued to the original payment method within 5 business days of the returned item being received. To start a return, sign in and open the order, then select the item and a reason for return. Store hours are 9am to 6pm on weekdays, closed on public holidays. Shipping is free on orders over $50, otherwise a flat $6 fee applies. International orders may take 10 to 15 business days to arrive. Gift cards are non-refundable. Warranty claims for electronics are handled by the manufacturer for the first 12 months.`;

type NData = {
  kind: Kind; label: string; icon: string; color: string;
  text?: string; srcName?: string; srcType?: string;
  size?: number; overlap?: number;
  mode?: "tfidf" | "neural"; indexType?: "vector" | "kg" | "hybrid";
  strategy?: Strategy; topK?: number; metric?: Metric; lambda?: number; hops?: number; method?: string;
  embModel?: string; semanticChunks?: string[];
  model?: string; count?: number | null; run?: "running" | "done";
};
type FNode = Node<NData>;
type SavedWF = { nodes?: { id: string; position?: { x: number; y: number }; data: NData }[]; edges?: { id?: string; source: string; target: string }[]; step?: number };
type PlayData = { question: string; chunks: string[]; ranked: { i: number; score: number }[]; top: number[]; strategy: Strategy; hasMmr: boolean; indexType: string };

type PalItem = { key: string; label: string; icon: string; kind: Kind; make: () => Partial<NData> };
const SOURCES: PalItem[] = [
  { key: "s-sample", label: "Sample", icon: "📚", kind: "source", make: () => ({ srcType: "sample", text: SAMPLE, srcName: "sample-returns.txt" }) },
  { key: "s-file", label: "File", icon: "📄", kind: "source", make: () => ({ srcType: "file", text: "", srcName: "" }) },
  { key: "s-url", label: "URL", icon: "🌐", kind: "source", make: () => ({ srcType: "url", text: "", srcName: "" }) },
];
// One tile per node kind — the type is chosen via a dropdown in the node config,
// so clicking a tile never spawns duplicate nodes.
const CHUNKS: PalItem[] = [
  { key: "c-chunk", label: "Chunk", icon: "⟷", kind: "chunk", make: () => ({ size: 40, overlap: 8, method: "sliding", label: "Sliding" }) },
];
const INDEXES: PalItem[] = [
  { key: "i-embed", label: "Embed", icon: "🧬", kind: "embed", make: () => ({ mode: "tfidf", label: "TF-IDF" }) },
  { key: "i-index", label: "Index", icon: "◆", kind: "index", make: () => ({ indexType: "vector", label: "Vector store" }) },
];
const ANSWER: PalItem[] = [
  { key: "r-ret", label: "Retriever", icon: "🔍", kind: "retriever", make: () => ({ strategy: "vector", topK: 3, metric: "cosine", label: "Vector retriever" }) },
  { key: "r-mmr", label: "MMR re-rank", icon: "🎯", kind: "mmr", make: () => ({ lambda: 0.7 }) },
  { key: "r-llm", label: "LLM", icon: "🤖", kind: "llm", make: () => ({ model: "" }) },
  { key: "r-dash", label: "Dashboard", icon: "📈", kind: "dashboard", make: () => ({}) },
];
type StepDef = { label: string; items: PalItem[] };
const STEPS: StepDef[] = [
  { label: "Source", items: SOURCES },
  { label: "Chunk", items: CHUNKS },
  { label: "Index", items: INDEXES },
  { label: "Retrieve & Answer", items: ANSWER },
];

function mkNode(it: PalItem, pos: { x: number; y: number }): FNode {
  return { id: rid(), type: "flow", position: pos, data: { kind: it.kind, label: it.label, icon: it.icon, color: COLOR[it.kind], ...it.make() } };
}
const DEFAULT_ITEMS = ["s-sample", "c-chunk", "i-embed", "i-index", "r-ret", "r-llm"];
function defaultNodes(): FNode[] {
  const all = [...SOURCES, ...CHUNKS, ...INDEXES, ...ANSWER];
  return DEFAULT_ITEMS.map((k, i) => { const it = all.find((p) => p.key === k)!; return mkNode(it, { x: 30 + i * 165, y: 100 }); });
}
function defaultEdges(nodes: FNode[]): Edge[] {
  const e: Edge[] = [];
  for (let i = 0; i < nodes.length - 1; i++) e.push({ id: `e${i}`, source: nodes[i].id, target: nodes[i + 1].id, animated: true, style: { stroke: "#5b7cff", strokeWidth: 2 }, markerEnd: { type: MarkerType.ArrowClosed, color: "#5b7cff" } });
  return e;
}

function FlowNode({ data, selected }: NodeProps<FNode>) {
  const c = data.color;
  const running = data.run === "running", rdone = data.run === "done";
  const border = running ? "#f59e0b" : rdone ? "#3ecf7f" : selected ? c : "var(--border)";
  const shadow = running ? "0 0 16px rgba(245,158,11,.45)" : selected ? `0 0 0 3px color-mix(in srgb, ${c} 30%, transparent)` : "var(--shadow-sm)";
  return (
    <div style={{ width: 150, borderRadius: 12, border: `1px solid ${border}`, background: "var(--panel)", boxShadow: shadow, padding: "9px 11px", position: "relative", transition: "box-shadow .2s, border-color .2s" }}>
      {data.kind !== "source" && <Handle type="target" position={Position.Left} style={{ background: "#0a0d17", border: `2px solid ${c}`, width: 9, height: 9 }} />}
      {data.kind !== "llm" && data.kind !== "dashboard" && <Handle type="source" position={Position.Right} style={{ background: "#0a0d17", border: `2px solid ${c}`, width: 9, height: 9 }} />}
      <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
        <span style={{ width: 28, height: 28, borderRadius: 8, display: "grid", placeItems: "center", fontSize: 14, flex: "0 0 auto", background: `color-mix(in srgb, ${c} 18%, transparent)`, color: c }}>{data.icon}</span>
        <div style={{ minWidth: 0 }}>
          <div style={{ fontSize: 12, fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 92 }}>{data.label}</div>
          <div style={{ fontSize: 9, color: c, textTransform: "uppercase", letterSpacing: ".04em", marginTop: 1 }}>{CATLABEL[data.kind]}</div>
        </div>
      </div>
      {data.count != null && <span style={{ position: "absolute", top: -8, right: -7, background: "var(--surface)", border: "1px solid var(--border-strong)", color: "var(--muted)", fontSize: 9.5, fontFamily: "var(--mono)", borderRadius: 20, padding: "1px 7px" }}>{data.count}</span>}
    </div>
  );
}
const nodeTypes = { flow: FlowNode };

type ChatMsg = { role: "user" | "assistant"; text: string; chunks?: { i: number; score: number; text: string }[] };

function Inner() {
  const [step, setStep] = useState(0);
  const initial = useMemo(() => defaultNodes(), []);
  const [nodes, setNodes, onNodesChange] = useNodesState<FNode>(initial);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>(defaultEdges(initial));
  const [selId, setSelId] = useState<string | null>(null);
  const [denseVecs, setDenseVecs] = useState<number[][] | null>(null);
  const [embedding, setEmbedding] = useState(false);
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [question, setQuestion] = useState("What is the refund policy for damaged items?");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");
  const [models, setModels] = useState<string[]>([]);
  const [model, setModel] = useState("");
  const [providers, setProviders] = useState<{ id: string; provider: string; label: string | null }[]>([]);
  const [providerId, setProviderId] = useState("");
  const [provKnown, setProvKnown] = useState(false);
  const [provider, setProvider] = useState<string | null>(null);
  const { screenToFlowPosition } = useReactFlow();
  const fileRef = useRef<HTMLInputElement>(null);
  const importRef = useRef<HTMLInputElement>(null);
  const threadRef = useRef<HTMLDivElement>(null);
  const [savedMsg, setSavedMsg] = useState("");
  const [projects, setProjects] = useState<{ id: string; name: string }[]>([]);
  const [loadOpen, setLoadOpen] = useState(false);
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameText, setRenameText] = useState("");
  const [playOpen, setPlayOpen] = useState(false);
  const [playData, setPlayData] = useState<PlayData | null>(null);
  const [dashOpen, setDashOpen] = useState(false);
  const [runStatus, setRunStatus] = useState<Record<string, "running" | "done">>({});
  const [openCite, setOpenCite] = useState<string | null>(null);
  const runTimers = useRef<ReturnType<typeof setTimeout>[]>([]);

  // ── derive the pipeline config from the nodes ──
  const chunkNode = nodes.find((n) => n.data.kind === "chunk");
  const embedNode = nodes.find((n) => n.data.kind === "embed");
  const idxNode = nodes.find((n) => n.data.kind === "index");
  const retrNode = nodes.find((n) => n.data.kind === "retriever");
  const hasMmr = nodes.some((n) => n.data.kind === "mmr");
  const mmrNode = nodes.find((n) => n.data.kind === "mmr");
  const indexType = idxNode?.data.indexType ?? "vector";
  const hops = idxNode?.data.hops ?? (indexType === "hybrid" ? 2 : 1);

  // Multi-document: concatenate every Source node's text.
  const text = useMemo(() => nodes.filter((n) => n.data.kind === "source").map((n) => n.data.text || "").filter(Boolean).join("\n\n"), [nodes]);
  const srcCount = nodes.filter((n) => n.data.kind === "source" && n.data.text).length;
  const size = chunkNode?.data.size ?? 40;
  const overlap = chunkNode?.data.overlap ?? 8;
  const chunks = useMemo(() => {
    if (chunkNode?.data.method === "semantic" && chunkNode.data.semanticChunks?.length) return chunkNode.data.semanticChunks;
    return text ? chunkText(text, size, overlap) : [];
  }, [text, size, overlap, chunkNode]);
  const index: RagIndex | null = useMemo(() => (chunks.length ? buildIndex(chunks) : null), [chunks]);
  // Real knowledge graph (extracted from chunks) when the index node is KG or Hybrid.
  const graph: KnowledgeGraph | null = useMemo(() => (indexType !== "vector" && chunks.length ? extractGraph(chunks, { maxNodes: 24 }) : null), [indexType, chunks]);
  const embedMode = embedNode?.data.mode ?? "tfidf";
  const strategy: Strategy = retrNode?.data.strategy ?? "vector";
  const topK = retrNode?.data.topK ?? 3;
  const metric: Metric = retrNode?.data.metric ?? "cosine";
  const lambda = mmrNode?.data.lambda ?? 0.7;
  useEffect(() => { setDenseVecs(null); }, [chunks]); // chunks changed → neural vecs stale

  // node counts (chunks on the chunk node, etc.)
  const counts = useMemo(() => {
    const m: Record<string, number> = {};
    nodes.forEach((n) => {
      if (n.data.kind === "chunk") m[n.id] = chunks.length;
      else if (n.data.kind === "index" || n.data.kind === "embed") m[n.id] = chunks.length;
      else if (n.data.kind === "retriever") m[n.id] = topK;
    });
    return m;
  }, [nodes, chunks, topK]);
  const displayNodes = useMemo(() => nodes.map((n) => ({ ...n, data: { ...n.data, count: counts[n.id] ?? null, run: runStatus[n.id] } })), [nodes, counts, runStatus]);
  const sel = nodes.find((n) => n.id === selId) || null;

  async function loadModels(id?: string) {
    try {
      const r = await fetch(`/api/models${id ? `?providerId=${encodeURIComponent(id)}` : ""}`);
      const j = await r.json();
      setProviders(j.providers || []); setProvider(j.provider ?? null); setProviderId(j.providerId || id || (j.providers?.[0]?.id ?? "")); setModels(j.models || []); setModel(j.default || (j.models?.[0] ?? ""));
    } catch { /* unconfigured */ } finally { setProvKnown(true); }
  }
  useEffect(() => { loadModels(); }, []);
  useEffect(() => { if (threadRef.current) threadRef.current.scrollTop = threadRef.current.scrollHeight; }, [messages]);

  const onConnect = useCallback((c: Connection) => setEdges((eds) => addEdge({ ...c, animated: true, style: { stroke: "#5b7cff", strokeWidth: 2 }, markerEnd: { type: MarkerType.ArrowClosed, color: "#5b7cff" } }, eds)), [setEdges]);
  function addFromPalette(it: PalItem, pos?: { x: number; y: number }) {
    const order = nodes.length;
    const node = mkNode(it, pos || { x: 30 + order * 165, y: 100 });
    setNodes((nds) => [...nds, node]);
    const prev = [...nodes].sort((a, b) => b.position.x - a.position.x)[0];
    if (prev && it.kind !== "source") setEdges((eds) => addEdge({ id: `e-${prev.id}-${node.id}`, source: prev.id, target: node.id, animated: true, style: { stroke: "#5b7cff", strokeWidth: 2 }, markerEnd: { type: MarkerType.ArrowClosed, color: "#5b7cff" } }, eds));
    setSelId(node.id);
  }
  const onDrop = (e: React.DragEvent) => { e.preventDefault(); const key = e.dataTransfer.getData("application/ragflow"); const it = STEPS[step].items.find((p) => p.key === key); if (!it) return; const p = screenToFlowPosition({ x: e.clientX, y: e.clientY }); addFromPalette(it, { x: p.x - 75, y: p.y - 22 }); };
  const patchSel = (p: Partial<NData>) => setNodes((nds) => nds.map((n) => (n.id === selId ? { ...n, data: { ...n.data, ...p } } : n)));
  const removeSel = () => { if (!selId) return; setNodes((nds) => nds.filter((n) => n.id !== selId)); setEdges((eds) => eds.filter((e) => e.source !== selId && e.target !== selId)); setSelId(null); };

  // ── save / load / import the workflow (reuses /api/projects, lab=rag) ──
  function download(txt: string, name: string, mime: string) { const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([txt], { type: mime })); a.download = name; a.click(); URL.revokeObjectURL(a.href); }
  function buildWorkflow(): SavedWF {
    const strip = (dd: NData): NData => ({ ...dd, count: undefined, text: dd.text ? dd.text.slice(0, 400000) : dd.text });
    return { nodes: nodes.map((n) => ({ id: n.id, position: n.position, data: strip(n.data) })), edges: edges.map((e) => ({ id: e.id, source: e.source, target: e.target })), step };
  }
  function applyWorkflow(wf: SavedWF | undefined) {
    if (!wf || !Array.isArray(wf.nodes)) { setSavedMsg("Nothing to load"); return; }
    setNodes(wf.nodes.map((n) => ({ id: n.id, type: "flow", position: n.position || { x: 30, y: 100 }, data: n.data })));
    setEdges((wf.edges || []).map((e) => ({ id: e.id || `e-${e.source}-${e.target}`, source: e.source, target: e.target, animated: true, style: { stroke: "#5b7cff", strokeWidth: 2 }, markerEnd: { type: MarkerType.ArrowClosed, color: "#5b7cff" } })));
    if (wf.step != null) setStep(wf.step); setSelId(null); setMessages([]); setDenseVecs(null);
  }
  async function loadProjects() { try { const j = await fetch("/api/projects?lab=rag").then((r) => r.json()); setProjects(((j.projects || []) as { id: string; name: string }[]).map((p) => ({ id: p.id, name: p.name }))); } catch { /* ignore */ } }
  async function loadProject(id: string) { if (!id) return; try { const j = await fetch(`/api/projects?id=${id}`).then((r) => r.json()); applyWorkflow(j.project?.config as SavedWF); setCurrentId(id); setSavedMsg("Loaded ✓"); setTimeout(() => setSavedMsg(""), 2000); } catch { setSavedMsg("Load failed"); } }
  const wfName = () => nodes.find((n) => n.data.kind === "source")?.data.srcName || "RAG pipeline";
  async function saveWorkflow() {
    try {
      if (currentId) { const r = await fetch("/api/projects", { method: "PATCH", headers: { "content-type": "application/json" }, body: JSON.stringify({ id: currentId, config: buildWorkflow() }) }); setSavedMsg(r.ok ? "Updated ✓" : "Update failed"); if (r.ok) loadProjects(); }
      else { const r = await fetch("/api/projects", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ lab: "rag", name: wfName(), config: buildWorkflow() }) }); const j = await r.json().catch(() => null); setSavedMsg(r.ok ? "Saved ✓" : (j?.error || "Save failed")); if (r.ok && j?.id) { setCurrentId(j.id); loadProjects(); } }
    } catch { setSavedMsg("Save failed"); }
    setTimeout(() => setSavedMsg(""), 2500);
  }
  async function deleteProject(id: string) { if (!(await confirmDialog("Delete this saved workflow?", { danger: true, confirmLabel: "Delete" }))) return; try { const r = await fetch(`/api/projects?id=${id}`, { method: "DELETE" }); toast(r.ok ? "Deleted" : "Delete failed", r.ok ? "success" : "error"); if (id === currentId) setCurrentId(null); loadProjects(); } catch { toast("Delete failed", "error"); } }
  async function renameProject(id: string) { const name = renameText.trim(); setRenamingId(null); if (!name) return; try { const r = await fetch("/api/projects", { method: "PATCH", headers: { "content-type": "application/json" }, body: JSON.stringify({ id, name }) }); toast(r.ok ? "Renamed" : "Rename failed", r.ok ? "success" : "error"); loadProjects(); } catch { toast("Rename failed", "error"); } }
  function newWorkflow() { const nn = defaultNodes(); setNodes(nn); setEdges(defaultEdges(nn)); setCurrentId(null); setSelId(null); setMessages([]); setStep(0); }
  function exportJson() { download(JSON.stringify(buildWorkflow(), null, 2), "rag-workflow.json", "application/json"); }
  async function importJson(e: React.ChangeEvent<HTMLInputElement>) { const f = e.target.files?.[0]; if (!f) return; try { applyWorkflow(JSON.parse(await f.text()) as SavedWF); setCurrentId(null); setSavedMsg("Imported ✓"); } catch { setSavedMsg("Bad JSON file"); } e.target.value = ""; setTimeout(() => setSavedMsg(""), 2500); }
  useEffect(() => {
    loadProjects();
    const id = new URLSearchParams(window.location.search).get("project");
    if (id) loadProject(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── ▶ Play: score every chunk for the question, then reveal top-k (chunk-by-chunk) ──
  async function openPlay() {
    if (!index || !chunks.length) { setMsg("Add a source and chunk it first."); return; }
    const q = question.trim() || "What is the refund policy for damaged items?";
    let ranked: { i: number; score: number }[];
    try {
      if (embedMode === "neural" && denseVecs) { const qv = (await embedViaApi([q]))[0]; ranked = retrieveDense(index, q, qv, denseVecs, strategy, chunks.length, metric); }
      else ranked = retrieve(index, q, strategy, chunks.length, metric);
    } catch (e) { setMsg((e as Error).message); return; }
    const top = (await retrieveTop(q)).map((h) => h.i);
    setPlayData({ question: q, chunks, ranked, top, strategy, hasMmr, indexType });
    setPlayOpen(true);
  }

  async function embedViaApi(texts: string[], embModel?: string): Promise<number[][]> {
    const res = await fetch("/api/rag/embed", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ texts, ...(providerId ? { providerId } : {}), ...(embModel ? { model: embModel } : {}) }) });
    const j = await res.json(); if (!res.ok) throw new Error(j.error || "embed failed"); return j.vectors as number[][];
  }
  // Real semantic chunking: embed sentences, then start a new chunk when the
  // running meaning drifts (low cosine) or the size cap is hit.
  async function runSemanticChunk() {
    if (!sel || !text) { setMsg("Add a source with data first."); return; }
    const sentences = text.split(/(?<=[.!?])\s+/).map((s) => s.trim()).filter(Boolean);
    if (sentences.length < 2) { setNodes((nds) => nds.map((n) => (n.id === sel.id ? { ...n, data: { ...n.data, semanticChunks: [text] } } : n))); return; }
    setEmbedding(true); setMsg("semantic chunking — embedding sentences…");
    try {
      const vecs = await embedViaApi(sentences, sel.data.embModel);
      const cap = sel.data.size ?? 60; const out: string[] = [];
      let cur = [sentences[0]]; let cen = vecs[0].slice();
      for (let i = 1; i < sentences.length; i++) {
        const sim = denseCos(cen, vecs[i]); const words = cur.join(" ").split(/\s+/).length;
        if (sim < 0.55 || words > cap) { out.push(cur.join(" ")); cur = [sentences[i]]; cen = vecs[i].slice(); }
        else { cur.push(sentences[i]); cen = cen.map((c, j) => (c + vecs[i][j]) / 2); }
      }
      if (cur.length) out.push(cur.join(" "));
      setNodes((nds) => nds.map((n) => (n.id === sel.id ? { ...n, data: { ...n.data, semanticChunks: out } } : n)));
      setMsg(`Semantic chunking → ${out.length} chunks`);
    } catch (e) { setMsg("Semantic chunking failed: " + (e as Error).message); }
    setEmbedding(false);
  }
  async function runNeural() {
    if (!chunks.length) return; setEmbedding(true); setMsg("");
    try { const v = await embedViaApi(chunks, embedNode?.data.embModel); setDenseVecs(v); setMsg(`Neural embeddings ready ✓ · ${v.length} vectors · ${v[0]?.length ?? 0} dims`); }
    catch (e) { setMsg((e as Error).message); setDenseVecs(null); }
    setEmbedding(false);
  }
  async function onFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0]; if (!f) return; const ext = (f.name.split(".").pop() || "").toLowerCase();
    try {
      let txt: string;
      if (["pdf", "docx", "doc", "xlsx", "xls"].includes(ext)) { const fd = new FormData(); fd.append("file", f); const r = await fetch("/api/rag/extract", { method: "POST", body: fd }); const j = await r.json(); if (!r.ok) throw new Error(j.error || "parse failed"); txt = j.text; }
      else txt = await f.text();
      patchSel({ text: txt, srcName: f.name });
    } catch (err) { setMsg((err as Error).message); }
    e.target.value = "";
  }
  async function fetchUrl(url: string) {
    if (!/^https?:\/\//i.test(url)) { setMsg("Enter a valid http(s) URL."); return; }
    setMsg("fetching…");
    try { const r = await fetch("/api/rag/fetch-url", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ url }) }); const j = await r.json(); if (!r.ok) throw new Error(j.error || "failed"); patchSel({ text: j.text, srcName: j.title || url }); setMsg(""); }
    catch (e) { setMsg((e as Error).message); }
  }

  // ── retrieve for a query (real) ──
  async function retrieveTop(q: string): Promise<{ i: number; score: number }[]> {
    if (!index) return [];
    // Pure knowledge-graph retrieval: traverse the graph from the query entities.
    if (indexType === "kg" && graph) {
      const r = retrieveGraph(graph, q, topK, hops);
      return r.chunkIds.map((i, rank) => ({ i, score: Math.max(0.05, 1 - rank * 0.12) }));
    }
    let ranked: { i: number; score: number }[];
    if (embedMode === "neural" && denseVecs) { const qv = (await embedViaApi([q]))[0]; ranked = retrieveDense(index, q, qv, denseVecs, strategy, chunks.length, metric); }
    else ranked = retrieve(index, q, strategy, chunks.length, metric);
    // Hybrid (GraphRAG): the graph narrows to relevant chunks, vectors rank within them.
    if (indexType === "hybrid" && graph) {
      const cand = new Set(retrieveGraph(graph, q, Math.max(topK * 3, 8), hops).chunkIds);
      const inGraph = ranked.filter((h) => cand.has(h.i));
      if (inGraph.length) ranked = inGraph;
    }
    let order: number[];
    if (hasMmr) {
      const cand = ranked.slice(0, Math.max(topK * 3, topK)).map((h) => h.i);
      const relMap = new Map(ranked.map((h) => [h.i, h.score])); const rel = (i: number) => relMap.get(i) ?? 0;
      const sim = (embedMode === "neural" && denseVecs) ? (i: number, j: number) => denseCos(denseVecs[i], denseVecs[j]) : (i: number, j: number) => cosine(index.vectors[i], index.vectors[j]);
      order = mmrRerank(cand, rel, sim, lambda, topK);
    } else order = ranked.slice(0, topK).map((h) => h.i);
    const scoreOf = new Map(ranked.map((h) => [h.i, h.score]));
    return order.map((i) => ({ i, score: scoreOf.get(i) ?? 0 }));
  }
  // Pulse each pipeline node in flow order; the LLM node stays "running" until
  // the answer finishes streaming (finishFlow marks everything done, then clears).
  function animateFlow() {
    runTimers.current.forEach(clearTimeout); runTimers.current = [];
    const order = [...nodes].sort((a, b) => a.position.x - b.position.x);
    setRunStatus({});
    order.forEach((n, idx) => {
      runTimers.current.push(setTimeout(() => setRunStatus((s) => ({ ...s, [n.id]: "running" })), idx * 300));
      if (n.data.kind !== "llm") runTimers.current.push(setTimeout(() => setRunStatus((s) => ({ ...s, [n.id]: "done" })), idx * 300 + 340));
    });
  }
  function finishFlow() {
    runTimers.current.forEach(clearTimeout); runTimers.current = [];
    setRunStatus((s) => { const n = { ...s }; nodes.forEach((nd) => { n[nd.id] = "done"; }); return n; });
    runTimers.current.push(setTimeout(() => setRunStatus({}), 1000));
  }
  async function ask(qArg?: string) {
    const q = (qArg ?? question).trim(); if (!q || !index || busy) return;
    setQuestion(""); setBusy(true); setMsg(""); animateFlow();
    setMessages((m) => [...m, { role: "user", text: q }]);
    let raw: { i: number; score: number }[] = [];
    try { raw = await retrieveTop(q); } catch (e) { setMsg((e as Error).message); }
    const top = raw.map((h) => ({ i: h.i, score: h.score, text: chunks[h.i] }));
    const context = top.map((h) => `[chunk ${h.i + 1}] ${h.text}`).join("\n\n");
    setMessages((m) => [...m, { role: "assistant", text: "", chunks: top }]);
    // No LLM provider → extractive fallback: show the retrieved context as the answer.
    if (provKnown && provider === null) {
      const extractive = top.length ? "No LLM provider configured — showing the retrieved context (add one under Admin → Providers for a generated answer):\n\n" + top.map((h) => `[chunk ${h.i + 1}] ${h.text}`).join("\n\n") : "No matching context found.";
      setMessages((m) => updateLast(m, extractive)); finishFlow(); setBusy(false); return;
    }
    const chatMessages = [
      { role: "system", content: "You are a helpful assistant. Answer using ONLY the provided context. Cite inline like [chunk N]. If the answer is not in the context, say you don't know." },
      { role: "user", content: `Context:\n${context}\n\nQuestion: ${q}` },
    ];
    try {
      const res = await fetch("/api/chat", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ messages: chatMessages, temperature: 0.2, lab: "rag", ...(providerId ? { providerId } : {}), ...(model ? { model } : {}) }) });
      if (!res.ok || !res.body) { const j = await res.json().catch(() => ({ error: "failed" })); setMessages((m) => updateLast(m, "⚠ " + (j.error || "failed"))); finishFlow(); setBusy(false); return; }
      const reader = res.body.getReader(); const dec = new TextDecoder(); let acc = "";
      for (; ;) { const { done, value } = await reader.read(); if (done) break; acc += dec.decode(value, { stream: true }); setMessages((m) => updateLast(m, acc)); }
    } catch (e) { setMessages((m) => updateLast(m, "⚠ " + (e as Error).message)); }
    finishFlow(); setBusy(false);
  }
  function updateLast(m: ChatMsg[], text: string): ChatMsg[] { const c = [...m]; for (let i = c.length - 1; i >= 0; i--) if (c[i].role === "assistant") { c[i] = { ...c[i], text }; break; } return c; }

  const stepBtn = (i: number, s: StepDef) => <button key={i} className={step === i ? "on" : ""} onClick={() => setStep(i)}><b>{i + 1}</b>{s.label}</button>;

  return (
    <div>
      <div className="lab-head">
        <div><div className="eyebrow">Lab 02 · flagship</div><h2 className="page-h">RAG Lab · Canvas</h2><p className="page-sub" style={{ margin: 0 }}>Build the retrieval pipeline as nodes — source, chunk, index (vector / graph / hybrid), retriever — then chat, grounded with citations.</p></div>
        <div className="acts" style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
          <button className="btn ghost sm" onClick={openPlay} title="Animate chunk-by-chunk retrieval">▶ Play</button>
          {currentId && <button className="btn ghost sm" onClick={newWorkflow}>＋ New</button>}
          <button className="btn ghost sm" onClick={saveWorkflow}>{savedMsg || (currentId ? "💾 Update" : "💾 Save")}</button>
          <div style={{ position: "relative" }}>
            <button className="btn ghost sm" onClick={() => { setLoadOpen((o) => !o); if (!loadOpen) loadProjects(); }}>📂 Load… ▾</button>
            {loadOpen && <>
              <div onClick={() => { setLoadOpen(false); setRenamingId(null); }} style={{ position: "fixed", inset: 0, zIndex: 40 }} />
              <div style={{ position: "absolute", top: "112%", right: 0, zIndex: 50, width: 288, maxHeight: 340, overflowY: "auto", background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 10, boxShadow: "var(--shadow-md)", padding: 6 }}>
                <div className="note" style={{ padding: "4px 8px 6px", fontSize: 10, textTransform: "uppercase", letterSpacing: ".06em" }}>Saved workflows</div>
                {projects.length === 0 ? <div className="note" style={{ padding: "6px 8px" }}>None yet — build a pipeline and Save.</div> :
                  projects.map((p) => (
                    <div key={p.id} className="etl-load-row" style={{ display: "flex", alignItems: "center", gap: 4, padding: "5px 8px", borderRadius: 7, background: p.id === currentId ? "var(--panel)" : undefined }}>
                      {renamingId === p.id ? <input autoFocus value={renameText} onChange={(e) => setRenameText(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") renameProject(p.id); if (e.key === "Escape") setRenamingId(null); }} onBlur={() => renameProject(p.id)} style={{ flex: 1, height: 26, fontSize: 12 }} />
                        : <button onClick={() => { loadProject(p.id); setLoadOpen(false); }} style={{ flex: 1, textAlign: "left", background: "none", border: "none", color: "var(--text)", fontSize: 12.5, cursor: "pointer", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>📄 {p.name}{p.id === currentId ? " •" : ""}</button>}
                      <button onClick={() => { setRenamingId(p.id); setRenameText(p.name); }} title="Rename" style={{ background: "none", border: "none", color: "var(--faint)", fontSize: 13, cursor: "pointer", padding: "0 3px" }}>✎</button>
                      <button onClick={() => deleteProject(p.id)} title="Delete" style={{ background: "none", border: "none", color: "var(--faint)", fontSize: 16, lineHeight: 1, cursor: "pointer", padding: "0 3px" }}>×</button>
                    </div>
                  ))}
              </div>
            </>}
          </div>
          <button className="btn ghost sm" onClick={exportJson}>⤓ Export</button>
          <button className="btn ghost sm" onClick={() => importRef.current?.click()}>⤒ Import</button>
        </div>
      </div>
      <input ref={importRef} type="file" accept=".json,application/json" onChange={importJson} style={{ display: "none" }} />
      {provKnown && provider === null && <div className="warnbar">No provider configured — add one under Admin → Providers to generate answers (retrieval still works).</div>}

      <div className="stepper">{STEPS.map((s, i) => stepBtn(i, s))}</div>
      <div className="note" style={{ margin: "0 0 8px" }}>🔒 Palette shows only <b style={{ color: "var(--accent)" }}>{STEPS[step].label}</b> nodes{step === 2 ? " — Vector store / Knowledge graph / Hybrid are here." : "."}</div>
      {(() => { const miss: string[] = []; if (!text) miss.push("a Source with data"); if (!nodes.some((n) => n.data.kind === "chunk")) miss.push("a Chunk node"); if (!nodes.some((n) => n.data.kind === "llm")) miss.push("an LLM node (extractive fallback runs without one)"); return miss.length ? <div className="warnbar" style={{ marginBottom: 12 }}>Pipeline incomplete — add {miss.join(", ")}.</div> : null; })()}

      <div className="split" style={{ gridTemplateColumns: "190px 1fr 300px", gap: 12 }}>
        {/* palette */}
        <div className="card"><div className="card-h"><span className="t">{STEPS[step].label} nodes</span></div>
          <div className="card-b" style={{ maxHeight: 460, overflowY: "auto" }}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 7 }}>
              {STEPS[step].items.map((it) => (
                <div key={it.key} draggable onDragStart={(e) => { e.dataTransfer.setData("application/ragflow", it.key); e.dataTransfer.effectAllowed = "move"; }} onClick={() => addFromPalette(it)}
                  style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6, padding: "10px 6px", borderRadius: 11, border: "1px solid var(--border)", background: "var(--panel-2)", cursor: "grab" }}>
                  <span style={{ width: 32, height: 32, borderRadius: 9, display: "grid", placeItems: "center", fontSize: 15, background: `color-mix(in srgb, ${COLOR[it.kind]} 18%, transparent)`, color: COLOR[it.kind] }}>{it.icon}</span>
                  <span style={{ fontSize: 10.5, color: "var(--muted)", textAlign: "center", lineHeight: 1.2 }}>{it.label}</span>
                </div>
              ))}
            </div>
            <div className="note" style={{ fontSize: 10, marginTop: 10, lineHeight: 1.5 }}>Drag a tile onto the canvas, or click to add. Wire handle → handle.</div>
          </div>
        </div>

        {/* canvas */}
        <div className="card" style={{ padding: 0, overflow: "hidden" }}>
          <div style={{ height: 460 }} onDrop={onDrop} onDragOver={(e) => { e.preventDefault(); e.dataTransfer.dropEffect = "move"; }}>
            <ReactFlow nodes={displayNodes} edges={edges} nodeTypes={nodeTypes} onNodesChange={onNodesChange} onEdgesChange={onEdgesChange} onConnect={onConnect} onNodeClick={(_, n) => setSelId(n.id)} onPaneClick={() => setSelId(null)} fitView proOptions={{ hideAttribution: true }}>
              <Background variant={BackgroundVariant.Dots} gap={20} size={1.4} color="var(--border-strong)" />
              <Controls />
              <MiniMap nodeColor={(n) => (n.data as NData).color} bgColor="#0e121d" maskColor="rgba(8,11,19,0.72)" style={{ background: "#0e121d", border: "1px solid var(--border)", borderRadius: 8, width: 140, height: 90 }} pannable zoomable />
            </ReactFlow>
          </div>
        </div>

        {/* node config */}
        <div className="card"><div className="card-h"><span className="t">Node details</span>{sel && <button className="btn ghost sm" onClick={removeSel}>Remove</button>}</div>
          <div className="card-b" style={{ maxHeight: 460, overflowY: "auto" }}>
            {!sel && <div className="note">Add a node from the palette, then click it to configure.</div>}
            {sel && <NodeConfig node={sel} patchSel={patchSel} onUpload={() => fileRef.current?.click()} fetchUrl={fetchUrl} runNeural={runNeural} embedding={embedding} models={models} model={model} setModel={setModel} chunks={chunks.length} denseReady={!!denseVecs} openDash={() => setDashOpen(true)} graph={graph} index={index} runSemantic={runSemanticChunk} providers={providers} providerId={providerId} setProviderId={setProviderId} loadModels={loadModels} denseDim={denseVecs?.[0]?.length ?? 0} />}
          </div>
        </div>
      </div>
      <input ref={fileRef} type="file" accept=".txt,.md,.csv,.json,.pdf,.docx" onChange={onFile} style={{ display: "none" }} />
      {msg && <div className="note" style={{ marginTop: 8 }}>{msg}</div>}

      {/* multi-turn chat */}
      <div className="card" style={{ marginTop: 12 }}>
        <div className="card-h"><span className="t">Retrieve &amp; Answer — chat</span><span className="mono r" style={{ display: "flex", alignItems: "center", gap: 10 }}>{srcCount} doc{srcCount === 1 ? "" : "s"} · {chunks.length} chunks · {indexType === "kg" ? `graph · ${hops}-hop` : indexType === "hybrid" ? `hybrid · ${strategy}` : strategy}{hasMmr && indexType !== "kg" ? " + MMR" : ""}{embedMode === "neural" && !denseVecs ? " · TF-IDF fallback" : embedMode === "neural" ? " · neural" : ""} · top-{topK}{messages.length > 0 && <button className="btn ghost sm" onClick={() => { setMessages([]); setOpenCite(null); }} title="Clear chat history">🗑 Clear</button>}</span></div>
        <div className="card-b">
          <div ref={threadRef} style={{ maxHeight: 320, overflowY: "auto", display: "flex", flexDirection: "column", gap: 12, marginBottom: 12 }}>
            {messages.length === 0 && <div className="note">Ask a question — the answer is grounded in the retrieved chunks, and you can keep asking.</div>}
            {messages.map((m, i) => m.role === "user" ? (
              <div key={i} style={{ alignSelf: "flex-end", maxWidth: "78%", background: "var(--accent)", color: "#fff", borderRadius: "12px 12px 3px 12px", padding: "8px 12px", fontSize: 13 }}>{m.text}</div>
            ) : (
              <div key={i} style={{ alignSelf: "flex-start", maxWidth: "90%" }}>
                <div style={{ fontSize: 10, color: "var(--faint)", textTransform: "uppercase", letterSpacing: ".05em", marginBottom: 4 }}>🤖 grounded answer</div>
                {(() => {
                  const openIdx = openCite && openCite.startsWith(`${i}-`) ? +openCite.slice(`${i}-`.length) : null;
                  const openChunk = openIdx != null ? m.chunks?.find((c) => c.i === openIdx) : null;
                  const toggle = (ci: number) => setOpenCite((k) => (k === `${i}-${ci}` ? null : `${i}-${ci}`));
                  return (
                    <div style={{ background: "var(--panel-2)", border: "1px solid var(--border)", borderRadius: "12px 12px 12px 3px", padding: "9px 12px", fontSize: 13, lineHeight: 1.55 }}>
                      {highlightCites(m.text || "", toggle)}{busy && i === messages.length - 1 && <span className="rag-cur-blink" />}
                      {m.chunks && m.chunks.length > 0 && <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 7 }}>{m.chunks.map((c) => <button key={c.i} onClick={() => toggle(c.i)} style={{ fontSize: 10, fontFamily: "var(--mono)", border: `1px solid ${openIdx === c.i ? "var(--accent)" : "var(--border)"}`, borderRadius: 20, padding: "2px 8px", color: openIdx === c.i ? "var(--text)" : "var(--muted)", background: openIdx === c.i ? "color-mix(in srgb, var(--accent) 14%, transparent)" : "transparent", cursor: "pointer" }}>chunk {c.i + 1} · <b style={{ color: "var(--orange)" }}>{c.score.toFixed(2)}</b></button>)}</div>}
                      {openChunk && <div style={{ marginTop: 8, borderTop: "1px solid var(--border)", paddingTop: 8, fontSize: 11.5, color: "var(--muted)", fontFamily: "var(--mono)", lineHeight: 1.55 }}><b style={{ color: "#82aaff" }}>chunk {openChunk.i + 1}</b> · score {openChunk.score.toFixed(2)}<br />{openChunk.text}</div>}
                    </div>
                  );
                })()}
              </div>
            ))}
          </div>
          {messages.length < 2 && <div className="row" style={{ gap: 6, flexWrap: "wrap", marginBottom: 8 }}>{["How much is shipping?", "What are the store hours?", "Are gift cards refundable?", "Electronics warranty?"].map((s) => <button key={s} className="btn ghost sm" style={{ fontSize: 11 }} onClick={() => ask(s)} disabled={busy || !index}>{s}</button>)}</div>}
          <div className="row" style={{ gap: 8 }}>
            <input type="text" value={question} onChange={(e) => setQuestion(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") ask(); }} placeholder="Ask a question… (ask as many times as you like)" style={{ flex: 1 }} />
            <button className="btn" onClick={() => ask()} disabled={busy || !index}>{busy ? "…" : "Ask ↗"}</button>
          </div>
          <div className="note" style={{ fontSize: 10, marginTop: 6 }}>Model: <b>{model || "none"}</b> — set it in the LLM node.</div>
        </div>
      </div>

      {playOpen && playData && <RagPlayModal data={playData} onClose={() => setPlayOpen(false)} />}
      {dashOpen && index && <RagDashModal index={index} chunks={chunks} question={question || "What is the refund policy for damaged items?"} topK={topK} metric={metric} onClose={() => setDashOpen(false)} />}
    </div>
  );
}

// Retrieval-quality dashboard: mark relevant chunks, then compare strategies by P@k / recall / MRR / nDCG.
const DASH_STRATS: Strategy[] = ["vector", "keyword", "hybrid"];
function RagDashModal({ index, chunks, question, topK, metric, onClose }: { index: RagIndex; chunks: string[]; question: string; topK: number; metric: Metric; onClose: () => void }) {
  const candidates = useMemo(() => {
    const s = new Set<number>();
    DASH_STRATS.forEach((st) => retrieve(index, question, st, topK, metric).forEach((h) => s.add(h.i)));
    return [...s].sort((a, b) => a - b);
  }, [index, question, topK, metric]);
  const [relevant, setRelevant] = useState<Set<number>>(new Set());
  const [rows, setRows] = useState<{ name: string; p: number; r: number; mrr: number; ndcg: number }[]>([]);
  const toggle = (i: number) => setRelevant((s) => { const n = new Set(s); if (n.has(i)) n.delete(i); else n.add(i); return n; });
  const evaluate = () => setRows(DASH_STRATS.map((st) => { const ranked = retrieve(index, question, st, chunks.length, metric).map((h) => h.i); return { name: st, ...retrievalMetrics(ranked, relevant, topK) }; }));
  const METRICS: [string, "p" | "r" | "mrr" | "ndcg"][] = [["P@k", "p"], ["Recall", "r"], ["MRR", "mrr"], ["nDCG", "ndcg"]];
  const SC = ["#5b7cff", "#ec4899", "#14b8a6"];
  return (
    <div onClick={onClose} style={{ position: "fixed", inset: 0, zIndex: 1000, background: "rgba(4,6,12,.78)", backdropFilter: "blur(4px)", display: "flex", alignItems: "center", justifyContent: "center", padding: 20 }}>
      <div onClick={(e) => e.stopPropagation()} style={{ width: "min(820px, 96vw)", maxHeight: "94vh", overflow: "auto", background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 14, boxShadow: "0 20px 60px rgba(0,0,0,.5)" }}>
        <div className="row" style={{ alignItems: "center", justifyContent: "space-between", padding: "12px 16px", borderBottom: "1px solid var(--border)" }}>
          <b style={{ fontSize: 14 }}>📈 Retrieval dashboard</b><button className="btn ghost sm" onClick={onClose}>Close</button>
        </div>
        <div style={{ padding: "12px 16px" }}>
          <div className="note" style={{ marginBottom: 4 }}>Question</div>
          <div style={{ background: "var(--accent)", color: "#fff", borderRadius: 9, padding: "8px 12px", fontSize: 13, marginBottom: 12 }}>{question}</div>
          <div className="note" style={{ marginBottom: 8 }}>Tick the chunks that actually answer the question, then evaluate to compare retrievers objectively.</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6, marginBottom: 12 }}>
            {candidates.map((i) => (
              <label key={i} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, cursor: "pointer", border: "1px solid var(--border)", borderRadius: 8, padding: "6px 9px", background: relevant.has(i) ? "color-mix(in srgb, var(--good) 8%, transparent)" : "var(--panel-2)" }}>
                <input type="checkbox" checked={relevant.has(i)} onChange={() => toggle(i)} />
                <span style={{ fontFamily: "var(--mono)", color: "var(--faint)", flex: "0 0 auto" }}>chunk {i + 1}</span>
                <span style={{ color: "var(--muted)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{chunks[i].slice(0, 80)}…</span>
              </label>
            ))}
          </div>
          <button className="btn sm" onClick={evaluate} disabled={relevant.size === 0}>▶ Evaluate ({relevant.size} relevant)</button>
          {rows.length > 0 && (
            <div style={{ marginTop: 14, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              {METRICS.map(([label, key]) => {
                const max = Math.max(1e-6, ...rows.map((r) => r[key]));
                return (
                  <div key={key} style={{ border: "1px solid var(--border)", borderRadius: 10, padding: "10px 12px" }}>
                    <div style={{ fontSize: 11.5, fontWeight: 600, marginBottom: 8 }}>{label} <span className="note" style={{ fontWeight: 400 }}>@top-{topK}</span></div>
                    {rows.map((r, ri) => (
                      <div key={r.name} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 5 }}>
                        <span style={{ width: 58, fontSize: 10.5, color: "var(--muted)", flex: "0 0 auto" }}>{r.name}</span>
                        <div style={{ flex: 1, height: 12, background: "var(--panel-2)", borderRadius: 4, overflow: "hidden" }}><div style={{ width: `${(r[key] / max) * 100}%`, height: "100%", background: SC[ri % SC.length], borderRadius: 4 }} /></div>
                        <span style={{ width: 34, textAlign: "right", fontSize: 10.5, fontFamily: "var(--mono)", color: "var(--muted)" }}>{r[key].toFixed(2)}</span>
                      </div>
                    ))}
                  </div>
                );
              })}
            </div>
          )}
          {rows.length > 0 && <div className="note" style={{ marginTop: 10 }}>Higher is better. This is how you pick a retriever objectively instead of by eye.</div>}
        </div>
      </div>
    </div>
  );
}

// Animated chunk-by-chunk retrieval: score each chunk for the query, reveal one
// at a time, then highlight the top-k that get retrieved.
function RagPlayModal({ data, onClose }: { data: PlayData; onClose: () => void }) {
  const ordered = useMemo(() => [...data.ranked].sort((a, b) => b.score - a.score), [data]);
  const [cur, setCur] = useState(-1);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(700);
  const topSet = useMemo(() => new Set(data.top), [data]);
  const total = ordered.length;
  const listRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!playing) return;
    if (cur >= total - 1) { setPlaying(false); return; }
    const t = setTimeout(() => setCur((c) => Math.min(c + 1, total - 1)), speed);
    return () => clearTimeout(t);
  }, [playing, cur, speed, total]);
  useEffect(() => { const el = listRef.current?.querySelector(".rag-cur") as HTMLElement | null; el?.scrollIntoView({ block: "nearest" }); }, [cur]);
  const max = Math.max(1e-6, ...ordered.map((o) => o.score));
  const done = cur >= total - 1;
  const shownTop = ordered.filter((o, i) => i <= cur && topSet.has(o.i)).length;
  return (
    <div onClick={onClose} style={{ position: "fixed", inset: 0, zIndex: 1000, background: "rgba(4,6,12,.78)", backdropFilter: "blur(4px)", display: "flex", alignItems: "center", justifyContent: "center", padding: 20 }}>
      <div onClick={(e) => e.stopPropagation()} style={{ width: "min(760px, 96vw)", maxHeight: "94vh", overflow: "auto", background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 14, boxShadow: "0 20px 60px rgba(0,0,0,.5)" }}>
        <div className="row" style={{ alignItems: "center", justifyContent: "space-between", padding: "12px 16px", borderBottom: "1px solid var(--border)" }}>
          <b style={{ fontSize: 14 }}>▶ Retrieval walkthrough <span className="note" style={{ fontWeight: 400 }}>· {data.strategy}{data.hasMmr ? " + MMR" : ""} · scoring chunk by chunk</span></b>
          <div className="row" style={{ gap: 6, alignItems: "center" }}>
            <button className="btn ghost sm" onClick={() => { setPlaying(false); setCur(-1); }}>⏮</button>
            <button className="btn ghost sm" onClick={() => { setPlaying(false); setCur((c) => Math.min(c + 1, total - 1)); }} disabled={done}>›</button>
            <button className="btn sm" onClick={() => { if (done) { setCur(-1); setPlaying(true); } else setPlaying((p) => !p); }}>{playing ? "⏸ Pause" : done ? "↻ Replay" : "▶ Play"}</button>
            <span className="note" style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>speed<select value={speed} onChange={(e) => setSpeed(+e.target.value)}><option value={1200}>0.5×</option><option value={700}>1×</option><option value={380}>2×</option></select></span>
            <button className="btn ghost sm" onClick={onClose}>Close</button>
          </div>
        </div>
        <div style={{ padding: "12px 16px" }}>
          <div className="note" style={{ marginBottom: 4 }}>Query</div>
          <div style={{ background: "var(--accent)", color: "#fff", borderRadius: 9, padding: "8px 12px", fontSize: 13, marginBottom: 12 }}>{data.question}</div>
          <div className="note" style={{ marginBottom: 8 }}>Scoring {total} chunks by <b>{data.strategy}</b> similarity — top-{data.top.length} (highlighted) get retrieved{data.hasMmr ? ", then MMR re-ranks for diversity" : ""}.</div>
          {data.indexType === "kg" && <div className="teach-note" style={{ marginBottom: 8 }}><span className="ic">🕸</span><span>This index is a <b>knowledge graph</b> — the ✓ chunks are chosen by <b>graph traversal</b>, so they may differ from the highest similarity bars (shown for reference).</span></div>}
          <div ref={listRef} style={{ display: "flex", flexDirection: "column", gap: 6, maxHeight: 360, overflowY: "auto" }}>
            {ordered.map((o, i) => {
              const scored = i <= cur, isCur = i === cur, retr = topSet.has(o.i);
              return (
                <div key={o.i} className={isCur ? "rag-cur" : ""} style={{ display: "flex", alignItems: "center", gap: 10, padding: "7px 9px", borderRadius: 9, border: `1px solid ${isCur ? "var(--accent)" : scored && retr ? "rgba(62,207,127,.4)" : "var(--border)"}`, background: isCur ? "color-mix(in srgb, var(--accent) 12%, transparent)" : scored && retr ? "color-mix(in srgb, var(--good) 8%, transparent)" : "var(--panel-2)", opacity: scored ? 1 : 0.45 }}>
                  <span style={{ fontSize: 10.5, fontFamily: "var(--mono)", color: "var(--faint)", width: 58, flex: "0 0 auto" }}>chunk {o.i + 1}</span>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: 11, color: "var(--muted)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{data.chunks[o.i].slice(0, 90)}…</div>
                    <div style={{ height: 6, background: "var(--surface)", borderRadius: 3, marginTop: 4, overflow: "hidden" }}><div style={{ width: scored ? `${(o.score / max) * 100}%` : "0%", height: "100%", background: retr ? "var(--good)" : "var(--accent)", transition: "width .3s" }} /></div>
                  </div>
                  <span style={{ fontSize: 11, fontFamily: "var(--mono)", color: scored ? (retr ? "var(--good)" : "var(--muted)") : "transparent", width: 68, textAlign: "right", flex: "0 0 auto" }}>{scored ? `${o.score.toFixed(2)}${retr ? " ✓" : ""}` : "—"}</span>
                </div>
              );
            })}
          </div>
          <div className="note" style={{ marginTop: 10, fontFamily: "var(--mono)", color: "var(--muted)" }}>{done ? `Done — ${data.top.length} chunks retrieved and sent to the LLM as context.` : `scored ${Math.max(0, cur + 1)} / ${total} · ${shownTop} of top-${data.top.length} found so far`}</div>
        </div>
      </div>
    </div>
  );
}

// Top TF-IDF terms across all chunk vectors — a peek at what the embeddings capture.
function embedPreview(index: RagIndex) {
  const totals: Record<string, number> = {};
  index.vectors.forEach((v) => Object.entries(v).forEach(([t, w]) => { totals[t] = (totals[t] || 0) + w; }));
  const top = Object.entries(totals).sort((a, b) => b[1] - a[1]).slice(0, 8);
  const max = top[0]?.[1] || 1;
  return (
    <div style={{ marginTop: 8 }}>
      <div className="fld">Top terms · {Object.keys(index.df).length}-dim vocabulary</div>
      {top.map(([t, w]) => (
        <div key={t} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 3 }}>
          <span style={{ width: 76, fontSize: 10.5, fontFamily: "var(--mono)", color: "var(--muted)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{t}</span>
          <div style={{ flex: 1, height: 8, background: "var(--panel-2)", borderRadius: 3, overflow: "hidden" }}><div style={{ width: `${(w / max) * 100}%`, height: "100%", background: "#a855f7", borderRadius: 3 }} /></div>
        </div>
      ))}
    </div>
  );
}
// Compact knowledge-graph preview (nodes coloured by type, sized by frequency).
function miniGraph(g: KnowledgeGraph) {
  const W = 268, H = 176;
  const pos = layoutGraph(g, W, H);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} style={{ background: "var(--panel-2)", borderRadius: 8, border: "1px solid var(--border)", marginTop: 6 }}>
      {g.edges.slice(0, 40).map((e, i) => { const a = pos[e.s], b = pos[e.o]; if (!a || !b) return null; return <line key={i} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="var(--border-strong)" strokeWidth={1} opacity={0.55} />; })}
      {g.nodes.map((n, i) => { const p = pos[n.id]; if (!p) return null; const r = 4 + Math.min(6, n.freq * 1.2); return <g key={i}><circle cx={p.x} cy={p.y} r={r} fill={n.type === "proper" ? "#a855f7" : "#5b7cff"} opacity={0.9} /><text x={p.x} y={p.y + r + 8} fontSize={7.5} fill="var(--muted)" textAnchor="middle">{n.label.length > 12 ? n.label.slice(0, 11) + "…" : n.label}</text></g>; })}
    </svg>
  );
}
function highlightCites(text: string, onCite?: (i: number) => void) {
  return text.split(/(\[chunk \d+\])/g).map((part, i) => {
    const m = /^\[chunk (\d+)\]$/.exec(part);
    if (m) { const idx = +m[1] - 1; return <span key={i} onClick={onCite ? () => onCite(idx) : undefined} style={{ color: "#82aaff", fontWeight: 500, cursor: onCite ? "pointer" : "default", textDecoration: onCite ? "underline dotted" : "none" }}>{part}</span>; }
    return <span key={i}>{part}</span>;
  });
}

function NodeConfig({ node, patchSel, onUpload, fetchUrl, runNeural, embedding, models, model, setModel, chunks, denseReady, openDash, graph, index, runSemantic, providers, providerId, setProviderId, loadModels, denseDim }: {
  node: FNode; patchSel: (p: Partial<NData>) => void; onUpload: () => void; fetchUrl: (u: string) => void; runNeural: () => void; embedding: boolean;
  models: string[]; model: string; setModel: (m: string) => void; chunks: number; denseReady: boolean; openDash: () => void; graph: KnowledgeGraph | null; index: RagIndex | null;
  runSemantic: () => void; providers: { id: string; provider: string; label: string | null }[]; providerId: string; setProviderId: (id: string) => void;
  loadModels: (id?: string) => void; denseDim: number;
}) {
  const d = node.data; const c = d.color;
  const [url, setUrl] = useState("https://en.wikipedia.org/wiki/Product_return");
  return (
    <div>
      <div className="row" style={{ alignItems: "center", gap: 9, marginBottom: 10 }}>
        <span style={{ width: 34, height: 34, borderRadius: 9, display: "grid", placeItems: "center", fontSize: 16, background: `color-mix(in srgb, ${c} 18%, transparent)`, color: c }}>{d.icon}</span>
        <span style={{ fontSize: 9.5, letterSpacing: ".07em", textTransform: "uppercase", fontWeight: 600, padding: "2px 8px", borderRadius: 6, background: `color-mix(in srgb, ${c} 16%, transparent)`, color: c }}>{CATLABEL[d.kind]}</span>
      </div>

      {d.kind === "source" && <>
        {d.srcType === "sample" && <div className="note" style={{ marginBottom: 8 }}>Built-in returns-policy corpus ({(d.text || "").split(/\s+/).length} words).</div>}
        {d.srcType === "file" && <><div className="note" style={{ marginBottom: 8 }}>{d.srcName ? `Loaded: ${d.srcName}` : "Upload a document."}</div><button className="btn sm" onClick={onUpload}>⬆ Upload file</button></>}
        {d.srcType === "url" && <><div className="insp-field"><div className="k">Web page URL</div><input type="text" value={url} onChange={(e) => setUrl(e.target.value)} /></div><button className="btn sm" onClick={() => fetchUrl(url)}>🌐 Fetch &amp; scrape</button>{d.srcName && <div className="note" style={{ marginTop: 6 }}>Loaded: {d.srcName}</div>}</>}
        {d.text ? <div style={{ marginTop: 10, maxHeight: 120, overflow: "auto", fontFamily: "var(--mono)", fontSize: 11, color: "var(--muted)", border: "1px solid var(--border)", borderRadius: 8, padding: 8 }}>{d.text.slice(0, 500)}{d.text.length > 500 ? "…" : ""}</div> : null}
      </>}

      {d.kind === "chunk" && <>
        <div className="insp-field"><div className="k">Method</div><select value={d.method || "sliding"} onChange={(e) => { const meth = e.target.value; patchSel({ method: meth, ...(meth === "fixed" ? { overlap: 0 } : meth === "semantic" ? { overlap: 10 } : {}), label: meth === "sliding" ? "Sliding" : meth === "fixed" ? "Fixed" : "Semantic" }); }}><option value="sliding">Sliding window</option><option value="fixed">Fixed</option><option value="semantic">Semantic</option></select></div>
        <div className="insp-field"><div className="k">Chunk size (words) · {d.size}</div><input type="range" min={15} max={120} step={5} value={d.size ?? 40} onChange={(e) => patchSel({ size: +e.target.value })} /></div>
        <div className="insp-field"><div className="k">Overlap (words) · {d.overlap}</div><input type="range" min={0} max={40} step={2} value={d.overlap ?? 8} onChange={(e) => patchSel({ overlap: +e.target.value })} /></div>
        {d.method === "semantic" ? <div style={{ border: "1px solid var(--border)", borderRadius: 10, padding: "10px 11px", background: "var(--panel-2)", marginBottom: 8 }}>
          <div className="note" style={{ marginBottom: 8 }}>Splits on <b>meaning</b> — embeds each sentence, then starts a new chunk when the topic drifts. Needs an embedder.</div>
          {providers.length > 1 && <div className="insp-field"><div className="k">Provider</div><select value={providerId} onChange={(e) => { setProviderId(e.target.value); loadModels(e.target.value); }}>{providers.map((p) => <option key={p.id} value={p.id}>{p.label || p.provider}</option>)}</select></div>}
          <div className="insp-field"><div className="k">Embedding model {models.length ? `· ${models.length} fetched` : "· none fetched"}</div>
            {models.length ? <select value={d.embModel || ""} onChange={(e) => patchSel({ embModel: e.target.value })}><option value="" disabled>choose a model…</option>{[...models.filter((m) => /embed/i.test(m)), ...models.filter((m) => !/embed/i.test(m))].map((m) => <option key={m} value={m}>{m}</option>)}</select>
              : <input type="text" value={d.embModel ?? ""} placeholder="gemini-embedding-001" onChange={(e) => patchSel({ embModel: e.target.value })} />}
          </div>
          <div className="row" style={{ gap: 8 }}><button className="btn ghost sm" onClick={() => loadModels(providerId || undefined)}>↻ Fetch models</button><button className="btn sm" onClick={runSemantic} disabled={embedding}>{embedding ? "chunking…" : "▶ Run semantic chunking"}</button></div>
          {d.semanticChunks && d.semanticChunks.length > 0 && <div style={{ marginTop: 10 }}><div className="fld">✓ Output · {d.semanticChunks.length} semantic chunks</div>{d.semanticChunks.slice(0, 3).map((c, ci) => <div key={ci} style={{ fontSize: 10.5, color: "var(--muted)", fontFamily: "var(--mono)", lineHeight: 1.45, border: "1px solid var(--border)", borderRadius: 8, padding: "6px 8px", marginBottom: 5 }}><b style={{ color: "#82aaff" }}>chunk {ci + 1}</b> · {c.split(/\s+/).length}w<br />{c.slice(0, 140)}…</div>)}</div>}
        </div> : <div className="note">{chunks} chunks from the source.</div>}
      </>}

      {d.kind === "embed" && <>
        <div className="insp-field"><div className="k">Backend</div><select value={d.mode} onChange={(e) => { const mode = e.target.value as "tfidf" | "neural"; patchSel({ mode, label: mode === "tfidf" ? "TF-IDF" : "Neural embed" }); }}><option value="tfidf">TF-IDF · lexical</option><option value="neural">Neural · semantic</option></select></div>
        {d.mode === "neural" && <div style={{ border: "1px solid var(--border)", borderRadius: 10, padding: "10px 11px", background: "var(--panel-2)", marginTop: 8 }}>
          {providers.length > 1 && <div className="insp-field"><div className="k">Provider</div><select value={providerId} onChange={(e) => { setProviderId(e.target.value); loadModels(e.target.value); }}>{providers.map((p) => <option key={p.id} value={p.id}>{p.label || p.provider}</option>)}</select></div>}
          <div className="insp-field"><div className="k">Embedding model {models.length ? `· ${models.length} fetched` : "· none fetched"}</div>
            {models.length ? <select value={d.embModel || ""} onChange={(e) => patchSel({ embModel: e.target.value })}><option value="" disabled>choose a model…</option>{[...models.filter((m) => /embed/i.test(m)), ...models.filter((m) => !/embed/i.test(m))].map((m) => <option key={m} value={m}>{m}</option>)}</select>
              : <input type="text" value={d.embModel ?? ""} placeholder="gemini-embedding-001" onChange={(e) => patchSel({ embModel: e.target.value })} />}
          </div>
          <div className="row" style={{ gap: 8 }}><button className="btn ghost sm" onClick={() => loadModels(providerId || undefined)}>↻ Fetch models</button><button className="btn sm" onClick={runNeural} disabled={embedding || !chunks}>{embedding ? "embedding…" : denseReady ? "↻ Re-embed" : "▶ Embed"}</button></div>
          {denseReady ? <div className="fld" style={{ marginTop: 8, color: "var(--good)" }}>✓ Output · {chunks} vectors · {denseDim} dims · <b>semantic</b></div>
            : <div className="teach-note" style={{ marginTop: 8 }}><span className="ic">🧬</span><span>Real dense embeddings via the provider. Free: <b>gemini-embedding-001</b>. (TF-IDF works without a provider.)</span></div>}
        </div>}
        {d.mode === "tfidf" && <><div className="note" style={{ marginTop: 8 }}>TF-IDF sparse vectors — no provider needed. Switch to Neural for semantic search.</div>{index && embedPreview(index)}</>}
      </>}

      {d.kind === "index" && <>
        <div className="insp-field"><div className="k">Index type</div><select value={d.indexType} onChange={(e) => { const t = e.target.value as "vector" | "kg" | "hybrid"; patchSel({ indexType: t, label: t === "vector" ? "Vector store" : t === "kg" ? "Knowledge graph" : "Hybrid" }); }}><option value="vector">Vector store</option><option value="kg">Knowledge graph</option><option value="hybrid">Hybrid (GraphRAG)</option></select></div>
        {d.indexType === "vector" && <div className="note" style={{ marginTop: 8 }}>ANN-style cosine index over the embeddings — fuzzy, paraphrase-friendly.</div>}
        {d.indexType !== "vector" && <>
          <div className="insp-field"><div className="k">Graph hops · {d.hops ?? (d.indexType === "hybrid" ? 2 : 1)}</div><input type="range" min={1} max={3} value={d.hops ?? (d.indexType === "hybrid" ? 2 : 1)} onChange={(e) => patchSel({ hops: +e.target.value })} /></div>
          <div className="teach-note" style={{ marginTop: 8 }}><span className="ic">🕸</span><span>{d.indexType === "kg" ? <>Entities + relations extracted from the chunks; retrieval <b>traverses the graph</b> from the query entities ({d.hops ?? 1}-hop).</> : <>GraphRAG — the graph finds relevant entities, then <b>vectors rank</b> the chunks they point to.</>}</span></div>
          {graph && graph.nodes.length > 0 && <><div className="fld" style={{ marginTop: 10 }}>Extracted graph · {graph.nodes.length} entities · {graph.edges.length} relations</div>{miniGraph(graph)}</>}
        </>}
      </>}

      {d.kind === "retriever" && <>
        <div className="insp-field"><div className="k">Strategy</div><select value={d.strategy} onChange={(e) => { const st = e.target.value as Strategy; patchSel({ strategy: st, label: st === "vector" ? "Vector retriever" : st === "keyword" ? "BM25" : "Hybrid retriever" }); }}><option value="vector">Vector (embeddings)</option><option value="keyword">Keyword (BM25)</option><option value="hybrid">Hybrid</option></select></div>
        <div className="insp-field"><div className="k">Top-k · {d.topK}</div><input type="range" min={1} max={6} value={d.topK ?? 3} onChange={(e) => patchSel({ topK: +e.target.value })} /></div>
        <div className="insp-field"><div className="k">Similarity</div><select value={d.metric} onChange={(e) => patchSel({ metric: e.target.value as Metric })}><option value="cosine">cosine</option><option value="dot">dot (IP)</option><option value="euclidean">euclidean (L2)</option></select></div>
      </>}

      {d.kind === "mmr" && <div className="insp-field"><div className="k">λ (relevance ↔ diversity) · {(d.lambda ?? 0.7).toFixed(2)}</div><input type="range" min={0} max={1} step={0.05} value={d.lambda ?? 0.7} onChange={(e) => patchSel({ lambda: +e.target.value })} /></div>}

      {d.kind === "llm" && <>
        {providers.length > 1 && <div className="insp-field"><div className="k">Provider</div><select value={providerId} onChange={(e) => { setProviderId(e.target.value); loadModels(e.target.value); }}>{providers.map((p) => <option key={p.id} value={p.id}>{p.label || p.provider}</option>)}</select></div>}
        <div className="insp-field"><div className="k">Model</div><select value={model} onChange={(e) => setModel(e.target.value)} disabled={!models.length}>{models.length ? models.map((m) => <option key={m} value={m}>{m}</option>) : <option value="">no models</option>}</select></div>
        <div className="note" style={{ marginTop: 6 }}>Generates the grounded, cited answer over the retrieved context.</div>
      </>}

      {d.kind === "dashboard" && <>
        <div className="note" style={{ marginBottom: 8 }}>Retrieval-quality metrics — P@k, recall, MRR, nDCG — comparing vector / keyword / hybrid on the current question.</div>
        <button className="btn sm" onClick={openDash}>📈 Open dashboard</button>
      </>}
    </div>
  );
}

export default function RagFlowLab() {
  return <ReactFlowProvider><Inner /></ReactFlowProvider>;
}
