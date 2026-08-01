"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { chunkText, buildIndex, retrieve, retrieveDense, denseCos, mmrRerank, retrievalMetrics, pca2, cosine, tokenize, queryVector, type RagIndex, type Strategy, type Vec } from "@/lib/ragUtils";
import { extractGraph, graphFromTriples, retrieveGraph, layoutGraph, type KnowledgeGraph, type KgEdge } from "@/lib/kgUtils";
import Plot from "@/components/Plot";
import { plotlyTheme } from "@/lib/edaCharts";

type Doc = { id: string; name: string; kind: string; text: string };
type Chunk = { text: string; docName: string; docKind: string };
type Step = "source" | "chunk" | "embed" | "query";
type Backend = "vector" | "kg" | "hybrid";

const SAMPLE = `Returns policy. Damaged items may be returned within 30 days of delivery for a full refund, provided the original packaging is included. Refunds are issued to the original payment method within 5 business days of the returned item being received. To start a return, sign in and open the order, then select the item and a reason for return. Store hours are 9am to 6pm on weekdays, closed on public holidays. Shipping is free on orders over $50, otherwise a flat $6 fee applies. International orders may take 10 to 15 business days to arrive. Gift cards are non-refundable. Warranty claims for electronics are handled by the manufacturer for the first 12 months.`;

const rid = () => Math.random().toString(36).slice(2, 10);

function computeChunks(docs: Doc[], size: number, overlap: number): Chunk[] {
  const out: Chunk[] = [];
  for (const d of docs) for (const t of chunkText(d.text, size, overlap)) out.push({ text: t, docName: d.name, docKind: d.kind });
  return out;
}
function topTermsW(v: Vec, n = 6): { term: string; w: number }[] {
  const arr = Object.entries(v).sort((a, b) => b[1] - a[1]).slice(0, n);
  const mx = arr[0]?.[1] || 1;
  return arr.map(([term, w]) => ({ term, w: w / mx }));
}

// Renders text with the matched query terms wrapped in <mark> (case-insensitive).
function highlightTerms(text: string, terms: string[]) {
  if (!terms.length) return text;
  const set = new Set(terms.map((t) => t.toLowerCase()));
  return text.split(/(\s+)/).map((tok, i) =>
    set.has(tok.toLowerCase().replace(/[^a-z0-9]/gi, "")) && tok.trim()
      ? <mark key={i} className="q-hl">{tok}</mark>
      : <span key={i}>{tok}</span>
  );
}

// A real cosine k-means over the chunk vectors — mirrors how Milvus's IVF_FLAT
// index partitions vectors into `nlist` buckets (deterministic, no randomness).
function ivfCluster(vectors: Vec[], k: number): number[] {
  const n = vectors.length;
  if (n === 0) return [];
  k = Math.max(1, Math.min(k, n));
  let centroids: Vec[] = [];
  for (let c = 0; c < k; c++) centroids.push({ ...vectors[Math.floor((c * n) / k)] });
  const assign = new Array(n).fill(0);
  for (let iter = 0; iter < 6; iter++) {
    for (let i = 0; i < n; i++) { let best = 0, bs = -Infinity; for (let c = 0; c < k; c++) { const s = cosine(vectors[i], centroids[c]); if (s > bs) { bs = s; best = c; } } assign[i] = best; }
    const sums: Vec[] = Array.from({ length: k }, () => ({})); const counts = new Array(k).fill(0);
    for (let i = 0; i < n; i++) { const c = assign[i]; counts[c]++; const v = vectors[i]; for (const t in v) sums[c][t] = (sums[c][t] || 0) + v[t]; }
    centroids = sums.map((s, c) => { const out: Vec = {}; const cnt = counts[c] || 1; for (const t in s) out[t] = s[t] / cnt; return out; });
  }
  return assign;
}

export default function RagLab() {
  const [step, setStep] = useState<Step>("source");
  const [docs, setDocs] = useState<Doc[]>([{ id: "sample", name: "sample-returns-policy.txt", kind: "txt", text: SAMPLE }]);
  const [size, setSize] = useState(40);
  const [overlap, setOverlap] = useState(8);
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [index, setIndex] = useState<RagIndex | null>(null);
  const [chunking, setChunking] = useState(false);
  const [embedding, setEmbedding] = useState(false);
  const [chunkPlayKey, setChunkPlayKey] = useState(0);
  // per-chunk inspector (steps through chunks, showing the actual words + overlap)
  const [inspIdx, setInspIdx] = useState(0);
  const [inspPlaying, setInspPlaying] = useState(false);
  const [inspSpeed, setInspSpeed] = useState(1100);
  // per-chunk embedding inspector (words → tokens → TF-IDF → vector → store)
  const [embIdx, setEmbIdx] = useState(0);
  const [embPlaying, setEmbPlaying] = useState(false);
  const [embSpeed, setEmbSpeed] = useState(1400);

  // ── knowledge-graph backend ──
  const [backend, setBackend] = useState<Backend>("vector");
  const [graph, setGraph] = useState<KnowledgeGraph | null>(null);
  const [kgExtract, setKgExtract] = useState<"heuristic" | "llm">("heuristic");
  const [kgHops, setKgHops] = useState(1);
  const [maxNodes, setMaxNodes] = useState(22);
  const [buildingKg, setBuildingKg] = useState(false);
  const [kgPath, setKgPath] = useState<KgEdge[]>([]);
  const [kgVisited, setKgVisited] = useState<string[]>([]);
  const [kgSeeds, setKgSeeds] = useState<string[]>([]);

  const [strategy, setStrategy] = useState<Strategy>("hybrid");
  const [topK, setTopK] = useState(3);
  // neural embeddings (real, via provider) + re-ranking + retrieval metrics
  const [embedMode, setEmbedMode] = useState<"tfidf" | "neural">("tfidf");
  const [denseVecs, setDenseVecs] = useState<number[][] | null>(null);
  const [embedInfo, setEmbedInfo] = useState<{ model: string; dim: number } | null>(null);
  const [qVec, setQVec] = useState<number[] | null>(null);
  const [rerank, setRerank] = useState<"none" | "mmr">("none");
  const [mmrLambda, setMmrLambda] = useState(0.7);
  const [relevant, setRelevant] = useState<Set<number>>(new Set());
  const [metricRows, setMetricRows] = useState<{ name: string; p: number; r: number; mrr: number; ndcg: number }[]>([]);
  const [question, setQuestion] = useState("What is the refund policy for damaged items?");
  const [url, setUrl] = useState("https://en.wikipedia.org/wiki/Product_return");
  const [fetching, setFetching] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [openDocs, setOpenDocs] = useState<Set<string>>(new Set());
  const [msg, setMsg] = useState("");

  const [answer, setAnswer] = useState("Ask a question to generate a grounded answer.");
  const [meta, setMeta] = useState("idle");
  const [tab, setTab] = useState<"out" | "trace">("out");
  const [trace, setTrace] = useState<{ who: string; what: string; state: string }[]>([]);
  const [hits, setHits] = useState<{ i: number; score: number }[]>([]);
  const [running, setRunning] = useState(false);
  const [compareRows, setCompareRows] = useState<{ size: number; overlap: number; chunks: number; top: number; avg: number; best: string }[]>([]);
  const [provider, setProvider] = useState<string | null>(null);
  const [provKnown, setProvKnown] = useState(false);
  const [providers, setProviders] = useState<{ id: string; provider: string; label: string | null }[]>([]);
  const [providerId, setProviderId] = useState("");
  const [models, setModels] = useState<string[]>([]);
  const [model, setModel] = useState("");
  const [modelsLoading, setModelsLoading] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Load a saved build when opened from My Projects (?project=<id>). RAG stores
  // the document text, so this fully restores the build.
  useEffect(() => {
    const id = new URLSearchParams(window.location.search).get("project");
    if (!id) return;
    fetch(`/api/projects?id=${id}`).then((r) => r.json()).then(({ project }) => {
      const c = project?.config; if (!c) return;
      if (Array.isArray(c.docs) && c.docs.length) setDocs(c.docs.map((d: { name: string; kind?: string; text?: string }) => ({ id: rid(), name: d.name, kind: d.kind || "txt", text: d.text || "" })));
      if (c.size != null) setSize(c.size);
      if (c.overlap != null) setOverlap(c.overlap);
      if (c.strategy) setStrategy(c.strategy);
      if (c.topK != null) setTopK(c.topK);
      if (c.question) setQuestion(c.question);
    }).catch(() => {});
  }, []);

  // Chunks changed → any neural vectors are stale; drop back to TF-IDF until re-embedded.
  useEffect(() => { setDenseVecs(null); setEmbedMode("tfidf"); setQVec(null); setMetricRows([]); setRelevant(new Set()); setGraph(null); setKgPath([]); setKgVisited([]); setKgSeeds([]); }, [chunks]);
  useEffect(() => { setQVec(null); setMetricRows([]); }, [question]);
  const combined = useMemo(() => docs.map((d) => d.text).join("\n\n"), [docs]);
  const totalWords = useMemo(() => combined.split(/\s+/).filter(Boolean).length, [combined]);
  const vocab = index ? Object.keys(index.df).length : 0;
  // IVF-style clustering of the stored vectors, shared by the Embed & Retrieve steps.
  const clusters = useMemo(() => {
    if (!index) return null;
    const k = Math.min(4, index.vectors.length);
    const assign = ivfCluster(index.vectors, k);
    const buckets = Array.from({ length: k }, (_, c) => index.vectors.map((_, i) => i).filter((i) => assign[i] === c));
    const centroids: Vec[] = buckets.map((ids) => { const s: Vec = {}; ids.forEach((i) => { const v = index.vectors[i]; for (const t in v) s[t] = (s[t] || 0) + v[t]; }); const out: Vec = {}; const cnt = ids.length || 1; for (const t in s) out[t] = s[t] / cnt; return out; });
    return { assign, buckets, centroids, nlist: k };
  }, [index]);

  const GW = 620, GH = 360;
  const graphPos = useMemo(() => (graph ? layoutGraph(graph, GW, GH) : null), [graph]);
  const canQuery = backend === "vector" ? !!index : backend === "kg" ? !!graph : (!!index && !!graph);
  const pnl: React.CSSProperties = { border: "1px solid var(--border)", borderRadius: 14, background: "var(--panel)", overflow: "hidden" };
  const kgHead = (dot: string, title: string, right?: React.ReactNode) => <div className="row" style={{ alignItems: "center", justifyContent: "space-between", padding: "10px 14px", borderBottom: "1px solid var(--border)", background: "var(--surface)" }}><div className="row" style={{ gap: 8, alignItems: "center" }}><span style={{ width: 7, height: 7, borderRadius: "50%", background: dot }} /><span style={{ fontWeight: 600, fontSize: 10.5, textTransform: "uppercase", letterSpacing: ".07em", color: "var(--muted)" }}>{title}</span></div>{right}</div>;
  const backendSel = (
    <div style={{ marginBottom: 16 }}>
      <label className="fld">Index backend — how the knowledge is stored &amp; searched</label>
      <div style={{ display: "inline-flex", background: "var(--panel-2)", border: "1px solid var(--border)", borderRadius: 10, padding: 3, flexWrap: "wrap" }}>
        {([["vector", "◆ Vector store"], ["kg", "🕸 Knowledge graph"], ["hybrid", "⚡ Hybrid (GraphRAG)"]] as [Backend, string][]).map(([b, l]) => <button key={b} onClick={() => setBackend(b)} style={{ padding: "7px 14px", borderRadius: 8, border: "none", background: backend === b ? "var(--accent)" : "transparent", color: backend === b ? "#fff" : "var(--muted)", fontSize: 12.5, fontWeight: 500, cursor: "pointer" }}>{l}</button>)}
      </div>
      <div className="note" style={{ marginTop: 6 }}>{backend === "vector" ? "Similarity search over embeddings — fuzzy, paraphrase-friendly." : backend === "kg" ? "Entities linked by relations, retrieved by traversal — explainable, multi-hop." : "Graph finds the relevant entities; vectors rank the chunks they point to."}</div>
    </div>
  );

  // Premium graph renderer — nodes coloured by type, matched subgraph highlighted, traversed edges lit.
  function graphSvg(g: KnowledgeGraph, pos: Record<string, { x: number; y: number }>, visited?: Set<string>, path?: KgEdge[]) {
    const pathSet = new Set((path || []).map((e) => e.s + "→" + e.o));
    const seedSet = new Set(kgSeeds);
    return (
      <svg viewBox={`0 0 ${GW} ${GH}`} width="100%" height={GH} style={{ display: "block", background: "var(--panel-2)", borderRadius: 10 }}>
        {g.edges.map((e, i) => { const a = pos[e.s], b = pos[e.o]; if (!a || !b) return null; const on = pathSet.has(e.s + "→" + e.o); const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
          return <g key={i}><line x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke={on ? "#5b7cff" : "var(--border-strong)"} strokeWidth={on ? 2.2 : 1} opacity={on ? 0.95 : visited ? 0.22 : 0.5} />
            {(on || g.edges.length <= 16) && <text x={mx} y={my - 3} fontSize={8.5} fill={on ? "#f59e0b" : "var(--faint)"} textAnchor="middle" fontStyle="italic">{e.rel.length > 14 ? e.rel.slice(0, 13) + "…" : e.rel}</text>}
          </g>; })}
        {g.nodes.map((n, i) => { const p = pos[n.id]; if (!p) return null; const on = visited?.has(n.id); const seed = seedSet.has(n.id); const r = 6 + Math.min(8, n.freq * 1.4);
          return <g key={i} opacity={visited && !on ? 0.32 : 1}>
            {on && <circle cx={p.x} cy={p.y} r={r + 4} fill="none" stroke="#5b7cff" strokeWidth={2} opacity={0.55} />}
            <circle cx={p.x} cy={p.y} r={r} fill={seed ? "#3ecf7f" : n.type === "proper" ? "#a855f7" : "#5b7cff"} opacity={0.92} />
            <text x={p.x} y={p.y + r + 10} fontSize={9.5} fill="var(--text)" textAnchor="middle" fontWeight={600}>{n.label.length > 16 ? n.label.slice(0, 15) + "…" : n.label}</text>
          </g>; })}
      </svg>
    );
  }

  async function loadModels(id?: string) {
    setModelsLoading(true);
    try {
      const r = await fetch(`/api/models${id ? `?providerId=${encodeURIComponent(id)}` : ""}`);
      const j = await r.json();
      setProviders(j.providers || []);
      setProvider(j.provider ?? null);
      setProviderId(j.providerId || id || (j.providers?.[0]?.id ?? ""));
      setModels(j.models || []);
      setModel(j.default || (j.models?.[0] ?? ""));
    } catch { /* leave as unconfigured */ } finally { setProvKnown(true); setModelsLoading(false); }
  }
  useEffect(() => {
    loadModels();
    return () => { if (timer.current) clearTimeout(timer.current); };
  }, []);

  // Auto-run chunking / embedding the first time you enter a step (explicit buttons re-run).
  useEffect(() => {
    if (step === "chunk" && chunks.length === 0 && docs.length) runChunking();
    if (step === "embed" && !index && chunks.length) runEmbedding();
    if (step === "embed" && backend !== "vector" && !graph && chunks.length) buildGraphHeuristic();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step]);
  // switching to a graph backend on the index step auto-extracts the graph
  useEffect(() => {
    if (step === "embed" && backend !== "vector" && !graph && chunks.length && !buildingKg) buildGraphHeuristic();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [backend]);

  // ── source ──
  async function onFiles(e: React.ChangeEvent<HTMLInputElement>) {
    const files = e.target.files; if (!files) return;
    setUploading(true); setMsg("");
    for (const f of Array.from(files)) {
      const ext = (f.name.split(".").pop() || "txt").toLowerCase();
      const binary = ["pdf", "docx", "doc", "xlsx", "xls", "xlsm"].includes(ext);
      try {
        if (binary) {
          const fd = new FormData(); fd.append("file", f);
          const r = await fetch("/api/rag/extract", { method: "POST", body: fd });
          const j = await r.json();
          if (!r.ok) throw new Error(j.error || "parse failed");
          setDocs((d) => [...d, { id: rid(), name: f.name, kind: ext, text: j.text }]);
        } else {
          const text = await f.text();
          setDocs((d) => [...d, { id: rid(), name: f.name, kind: ext, text }]);
        }
      } catch (err) { setMsg(`Could not read ${f.name}: ${(err as Error).message}`); }
    }
    setUploading(false);
    e.target.value = "";
    invalidate();
  }
  async function fetchUrl() {
    if (!/^https?:\/\//i.test(url)) { setMsg("Enter a valid http(s) URL."); return; }
    setFetching(true); setMsg("");
    try {
      const r = await fetch("/api/rag/fetch-url", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ url }) });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || "failed");
      setDocs((d) => [...d, { id: rid(), name: j.title || url, kind: "url", text: j.text }]);
      invalidate();
    } catch (e) { setMsg((e as Error).message); } finally { setFetching(false); }
  }
  const removeDoc = (id: string) => { setDocs((d) => d.filter((x) => x.id !== id)); invalidate(); };
  function invalidate() { setChunks([]); setIndex(null); }

  // ── chunking (cards fade in with a CSS stagger) ──
  function runChunking() {
    if (timer.current) clearTimeout(timer.current);
    const cs = computeChunks(docs, size, overlap);
    setChunks(cs); setIndex(null); setChunking(true); setChunkPlayKey((k) => k + 1);
    setInspIdx(0); setInspPlaying(cs.length > 0);
    timer.current = setTimeout(() => setChunking(false), Math.min(2000, cs.length * 90 + 400));
  }
  // step the chunk inspector through the chunks one at a time
  useEffect(() => {
    if (!inspPlaying || chunks.length === 0) return;
    if (inspIdx >= chunks.length - 1) { setInspPlaying(false); return; }
    const t = setTimeout(() => setInspIdx((i) => Math.min(i + 1, chunks.length - 1)), inspSpeed);
    return () => clearTimeout(t);
  }, [inspPlaying, inspIdx, inspSpeed, chunks.length]);

  // ── embedding (vector cards fade in with a CSS stagger) ──
  function runEmbedding() {
    if (chunks.length === 0) { runChunking(); return; }
    if (timer.current) clearTimeout(timer.current);
    const idx = buildIndex(chunks.map((c) => c.text));
    setIndex(idx); setEmbedding(true);
    setEmbIdx(0); setEmbPlaying(chunks.length > 0);
    timer.current = setTimeout(() => setEmbedding(false), Math.min(2200, chunks.length * 110 + 400));
  }
  // step the embedding inspector through the chunks
  useEffect(() => {
    if (!embPlaying || !index || chunks.length === 0) return;
    if (embIdx >= chunks.length - 1) { setEmbPlaying(false); return; }
    const t = setTimeout(() => setEmbIdx((i) => Math.min(i + 1, chunks.length - 1)), embSpeed);
    return () => clearTimeout(t);
  }, [embPlaying, embIdx, embSpeed, index, chunks.length]);

  function goStep(s: Step) { setStep(s); }

  function setTraceStep(steps: { who: string; what: string }[], active: number) {
    setTrace(steps.map((s, i) => ({ ...s, state: i < active ? "done" : i === active ? "active" : "" })));
  }

  const [saved, setSaved] = useState("");

  // Rebuild the index at several chunk sizes and score retrieval for the current
  // question — teaches that chunk size is a real, measurable knob.
  function compareChunking() {
    const cfgs = [{ size: 20, overlap: 4 }, { size: 40, overlap: 8 }, { size: 80, overlap: 16 }, { size: 120, overlap: 24 }];
    setCompareRows(cfgs.map(({ size: sz, overlap: ov }) => {
      const cks = chunkText(combined, sz, ov);
      if (!cks.length) return { size: sz, overlap: ov, chunks: 0, top: 0, avg: 0, best: "" };
      const hs = retrieve(buildIndex(cks), question, strategy, topK);
      const top = hs[0]?.score || 0;
      const avg = hs.length ? hs.reduce((a, h) => a + h.score, 0) / hs.length : 0;
      return { size: sz, overlap: ov, chunks: cks.length, top, avg, best: hs[0] != null ? cks[hs[0].i] : "" };
    }));
  }
  async function saveProject() {
    const MAX = 600_000; // cap saved document text (~0.6 MB) so a project row stays small
    let used = 0;
    const savedDocs = docs.map((d) => {
      const room = Math.max(0, MAX - used);
      const text = d.text.length > room ? d.text.slice(0, room) : d.text;
      used += text.length;
      return { name: d.name, kind: d.kind, text, truncated: text.length < d.text.length };
    });
    const trimmed = savedDocs.some((d) => d.truncated);
    const config = { docs: savedDocs, size, overlap, strategy, topK, question };
    try {
      const r = await fetch("/api/projects", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ lab: "rag", name: docs[0]?.name || "RAG build", config }) });
      setSaved(r.ok ? (trimmed ? "Saved (text trimmed)" : "Saved ✓") : "Save failed");
    }
    catch { setSaved("Save failed"); }
    setTimeout(() => setSaved(""), 2500);
  }
  // ── neural embeddings (real, via the provider's /embeddings endpoint) ──
  async function embedViaApi(texts: string[]): Promise<number[][]> {
    const res = await fetch("/api/rag/embed", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ texts }) });
    const j = await res.json();
    if (!res.ok) throw new Error(j.error || "embed failed");
    return j.vectors as number[][];
  }
  async function runNeuralEmbed() {
    if (!chunks.length) return; setEmbedding(true); setMsg("");
    try {
      const res = await fetch("/api/rag/embed", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ texts: chunks.map((c) => c.text) }) });
      const j = await res.json(); if (!res.ok) throw new Error(j.error || "embed failed");
      setDenseVecs(j.vectors); setEmbedInfo({ model: j.model, dim: j.dim }); setEmbedMode("neural"); setQVec(null); setMetricRows([]);
    } catch (e) { setMsg((e as Error).message); setEmbedMode("tfidf"); setDenseVecs(null); }
    setEmbedding(false);
  }
  // ── knowledge-graph construction ──
  function buildGraphHeuristic() {
    if (!chunks.length) return;
    setBuildingKg(true); setMsg("");
    // defer so the spinner paints before the (fast) synchronous extraction
    setTimeout(() => { setGraph(extractGraph(chunks.map((c) => c.text), { maxNodes })); setKgPath([]); setKgVisited([]); setKgSeeds([]); setBuildingKg(false); }, 20);
  }
  async function buildGraphLLM() {
    if (!chunks.length) return; setBuildingKg(true); setMsg("");
    try {
      const corpus = chunks.map((c, i) => `[chunk ${i}] ${c.text}`).join("\n").slice(0, 8000);
      const messages = [
        { role: "system", content: "Extract a knowledge graph. Return ONLY a JSON array of triples, each {\"s\":\"subject\",\"r\":\"relation\",\"o\":\"object\"}. Use short noun-phrase entities and concise relations. Max 30 triples. No prose." },
        { role: "user", content: corpus },
      ];
      const res = await fetch("/api/chat", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ messages, temperature: 0, lab: "rag", ...(providerId ? { providerId } : {}), ...(model ? { model } : {}) }) });
      if (!res.ok || !res.body) throw new Error("extraction failed");
      const reader = res.body.getReader(); const dec = new TextDecoder(); let text = "";
      for (; ;) { const { done, value } = await reader.read(); if (done) break; text += dec.decode(value, { stream: true }); }
      const m = text.match(/\[[\s\S]*\]/); const triples = m ? JSON.parse(m[0]) : [];
      const clean = (Array.isArray(triples) ? triples : []).filter((t: unknown): t is { s: string; r: string; o: string } => !!t && typeof (t as { s: unknown }).s === "string" && typeof (t as { o: unknown }).o === "string");
      if (!clean.length) throw new Error("no triples returned");
      setGraph(graphFromTriples(clean, chunks.map((c) => c.text))); setKgPath([]); setKgVisited([]); setKgSeeds([]);
    } catch (e) { setMsg("LLM extraction: " + (e as Error).message + " — falling back to heuristic."); setGraph(extractGraph(chunks.map((c) => c.text), { maxNodes })); }
    setBuildingKg(false);
  }
  function buildGraph() { if (kgExtract === "llm" && provider) buildGraphLLM(); else buildGraphHeuristic(); }

  async function ensureQVec(): Promise<number[] | null> {
    if (embedMode !== "neural" || !denseVecs) return null;
    if (qVec) return qVec;
    try { const v = (await embedViaApi([question]))[0] || null; setQVec(v); return v; } catch { return null; }
  }
  // Full ranking of every chunk for a strategy (dense when neural, else TF-IDF/BM25).
  function fullRank(strat: Strategy, qv: number[] | null): { i: number; score: number }[] {
    if (embedMode === "neural" && denseVecs && qv) return retrieveDense(index!, question, qv, denseVecs, strat, chunks.length);
    return retrieve(index!, question, strat, chunks.length);
  }
  // Apply MMR re-ranking (diversity) to a ranked candidate list, returning top-k indices.
  function applyRerank(ranked: { i: number; score: number }[], k: number): number[] {
    if (rerank !== "mmr") return ranked.slice(0, k).map((h) => h.i);
    const cand = ranked.slice(0, Math.min(ranked.length, Math.max(k * 3, k))).map((h) => h.i);
    const relMap = new Map(ranked.map((h) => [h.i, h.score]));
    const rel = (i: number) => relMap.get(i) ?? 0;
    const sim = (embedMode === "neural" && denseVecs) ? (i: number, j: number) => denseCos(denseVecs[i], denseVecs[j]) : (i: number, j: number) => cosine(index!.vectors[i], index!.vectors[j]);
    return mmrRerank(cand, rel, sim, mmrLambda, k);
  }
  async function evalRetrieval() {
    if (!index) return; const qv = await ensureQVec();
    const strats: Strategy[] = ["keyword", "vector", "hybrid"];
    const rows = strats.map((s) => { const ranked = fullRank(s, qv); const order = applyRerank(ranked, chunks.length); const m = retrievalMetrics(order, relevant, topK); return { name: rerank === "mmr" ? `${s}+mmr` : s, ...m }; });
    setMetricRows(rows);
  }
  async function ask() {
    if (chunks.length === 0 || !canQuery) { setAnswer("Build the index first (step 3)."); return; }
    setRunning(true); setTab("out"); setAnswer(""); setMeta("retrieving…");
    const neural = embedMode === "neural" && !!denseVecs;
    const steps = [
      { who: backend === "kg" ? "link entities" : "embed query", what: backend === "kg" ? "match query terms → graph nodes" : neural ? `question → ${embedInfo?.dim ?? 0}-d neural vector` : "question → TF-IDF vector" },
      { who: "retrieve", what: backend === "kg" ? `graph traversal · ${kgHops}-hop · top ${topK}` : backend === "hybrid" ? `graph → vector rank · top ${topK}` : `${strategy}${rerank === "mmr" ? " + MMR" : ""} · top-k ${topK}` },
      { who: "prompt", what: "inject retrieved context + sources" },
      { who: "generate", what: `stream → ${model || provider || "provider"}` },
    ];
    setTraceStep(steps, 1);
    let top: { i: number; score: number }[];
    if (backend === "kg" && graph) {
      const r = retrieveGraph(graph, question, topK, kgHops);
      setKgSeeds(r.seeds); setKgVisited(r.nodes); setKgPath(r.path);
      top = r.chunkIds.map((i, rank) => ({ i, score: Math.max(0.05, 1 - rank * 0.12) }));
    } else if (backend === "hybrid" && graph && index) {
      const r = retrieveGraph(graph, question, Math.max(topK * 3, 8), Math.max(kgHops, 2));
      setKgSeeds(r.seeds); setKgVisited(r.nodes); setKgPath(r.path);
      const cand = new Set(r.chunkIds); const qv = await ensureQVec();
      const ranked = fullRank(strategy, qv); const inGraph = ranked.filter((h) => cand.has(h.i));
      const order = applyRerank(inGraph.length ? inGraph : ranked, topK);
      const scoreOf = new Map(ranked.map((h) => [h.i, h.score]));
      top = order.map((i) => ({ i, score: scoreOf.get(i) ?? 0 }));
    } else {
      const qv = await ensureQVec();
      const ranked = fullRank(strategy, qv); const order = applyRerank(ranked, topK);
      const scoreOf = new Map(ranked.map((h) => [h.i, h.score]));
      top = order.map((i) => ({ i, score: scoreOf.get(i) ?? 0 }));
    }
    setHits(top);
    setTraceStep(steps, 3);
    const context = top.map((h) => `[chunk ${h.i + 1} · source: ${chunks[h.i].docName}] ${chunks[h.i].text}`).join("\n\n");
    const messages = [
      { role: "system", content: "You are a helpful assistant. Answer using ONLY the provided context. Cite inline like [chunk N]. If the answer is not in the context, say you don't know." },
      { role: "user", content: `Context:\n${context}\n\nQuestion: ${question}` },
    ];
    setMeta("generating…");
    const t0 = performance.now();
    try {
      const res = await fetch("/api/chat", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ messages, temperature: 0.2, lab: "rag", ...(providerId ? { providerId } : {}), ...(model ? { model } : {}) }) });
      if (!res.ok || !res.body) { const j = await res.json().catch(() => ({ error: "failed" })); setAnswer("⚠ " + (j.error || "failed")); setMeta("error"); setRunning(false); return; }
      const reader = res.body.getReader(); const dec = new TextDecoder(); let text = "";
      for (; ;) { const { done, value } = await reader.read(); if (done) break; text += dec.decode(value, { stream: true }); setAnswer(text); }
      setMeta(`${model ? model + " · " : ""}grounded · ${top.length} sources · ${Math.round(performance.now() - t0)}ms`);
      setTrace(steps.map((s) => ({ ...s, state: "done" })));
    } catch (e) { setAnswer("⚠ " + (e as Error).message); setMeta("error"); }
    setRunning(false);
  }

  const stepBtn = (k: Step, n: number, label: string, enabled: boolean) => (
    <button className={step === k ? "on" : ""} disabled={!enabled} onClick={() => goStep(k)}><b>{n}</b>{label}</button>
  );
  const stepWords = Math.max(1, size - overlap);
  const pLayout = (t: ReturnType<typeof plotlyTheme>, title: string, extra: Record<string, unknown> = {}) => ({ title: { text: title, font: { size: 13, color: t.text } }, paper_bgcolor: t.paper, plot_bgcolor: t.plot, font: { color: t.muted, size: 11 }, margin: { l: 44, r: 16, t: 40, b: 60 }, xaxis: { gridcolor: t.grid, zerolinecolor: t.grid }, yaxis: { gridcolor: t.grid, zerolinecolor: t.grid }, colorway: t.colorway, ...extra });

  return (
    <div className="rag-lab">
      <div className="lab-head">
        <div>
          <div className="eyebrow">Lab 02 · flagship</div>
          <h2 className="page-h">RAG Lab</h2>
          <p className="page-sub" style={{ margin: 0 }}>Add sources, chunk them, then index into a <b>vector store</b>, a <b>knowledge graph</b>, or both (hybrid) — ask, and the answer is grounded with citations.</p>
        </div>
        <div className="acts"><button className="btn ghost sm" onClick={saveProject}>{saved || "💾 Save"}</button></div>
      </div>

      {provKnown && provider === null && <div className="warnbar">No provider configured — an admin must add one under Admin → Providers before the answer step (source/chunk/embed still work).</div>}

      <div className="stepper">
        {stepBtn("source", 1, "Source", true)}
        {stepBtn("chunk", 2, "Chunk", docs.length > 0)}
        {stepBtn("embed", 3, "Index", docs.length > 0)}
        {stepBtn("query", 4, "Retrieve & Answer", docs.length > 0)}
      </div>

      {/* STEP 1 — SOURCE */}
      {step === "source" && (
        <div className="card">
          <div className="card-h"><span className="t">Add knowledge sources</span><span className="mono r">{docs.length} docs · {totalWords} words</span></div>
          <div className="card-b">
            {msg && <div className="err">{msg}</div>}
            <div className="row" style={{ flexWrap: "wrap", gap: 10 }}>
              <button className="btn ghost sm" onClick={() => fileRef.current?.click()} disabled={uploading}>{uploading ? <><span className="busy-dot" />Parsing…</> : "Upload files"}</button>
              <input ref={fileRef} type="file" multiple accept=".txt,.md,.csv,.json,.log,.html,.tsv,.pdf,.docx,.doc,.xlsx,.xls,.xlsm,text/*,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" onChange={onFiles} style={{ display: "none" }} />
              <span className="note">txt · md · csv · json · html · <b>pdf</b> · <b>docx</b> · <b>xlsx</b></span>
            </div>
            <div className="row" style={{ marginTop: 12, gap: 8 }}>
              <input type="text" value={url} onChange={(e) => setUrl(e.target.value)} placeholder="https://… (web page to scrape)" />
              <button className="btn ghost sm" onClick={fetchUrl} disabled={fetching} style={{ whiteSpace: "nowrap" }}>{fetching ? "Fetching…" : "Fetch URL"}</button>
            </div>
            <label className="fld" style={{ marginTop: 16 }}>Loaded documents — check the preview before chunking</label>
            {docs.map((d) => {
              const words = d.text.split(/\s+/).filter(Boolean).length;
              const open = openDocs.has(d.id);
              const toggle = () => setOpenDocs((s) => { const n = new Set(s); if (n.has(d.id)) n.delete(d.id); else n.add(d.id); return n; });
              return (
                <div key={d.id} className={`doc-row ${open ? "open" : ""}`}>
                  <div className="doc-item">
                    <span className="kind">{d.kind}</span>
                    <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{d.name}</span>
                    <span className="note">{words.toLocaleString()} words · {d.text.length.toLocaleString()} chars</span>
                    <button className="doc-prev-btn" onClick={toggle}>{open ? "Hide" : "Preview"}</button>
                    <button className="x" onClick={() => removeDoc(d.id)} title="Remove">×</button>
                  </div>
                  {open && (
                    d.text.trim()
                      ? <div className="doc-prev">{d.text.slice(0, 800)}{d.text.length > 800 ? " …" : ""}</div>
                      : <div className="doc-prev empty">⚠ No text was extracted — this file may be a scanned image or an unsupported layout.</div>
                  )}
                </div>
              );
            })}
            {docs.length === 0 && <div className="note">No documents yet — upload a file or fetch a URL.</div>}
            <label className="fld" style={{ marginTop: 14 }}>Combined data preview</label>
            <div className="dprev">{combined.slice(0, 1500)}{combined.length > 1500 ? "\n…" : ""}</div>
            <div className="stepnav"><button className="btn" disabled={docs.length === 0} onClick={() => goStep("chunk")}>Next: Chunk →</button></div>
          </div>
        </div>
      )}

      {/* STEP 2 — CHUNK */}
      {step === "chunk" && (
        <div className="card">
          <div className="card-h"><span className="t">Chunking</span><span className="mono r">{chunking ? <><span className="busy-dot" />splitting…</> : `${chunks.length} chunks`}</span></div>
          <div className="card-b">
            <div className="stat-row">
              <div className="stat">documents<b>{docs.length}</b></div>
              <div className="stat">words<b>{totalWords}</b></div>
              <div className="stat">chunks<b>{chunks.length}</b></div>
              <div className="stat">overlap<b>{overlap}w</b></div>
            </div>
            <div className="row" style={{ flexWrap: "wrap", gap: 16, marginBottom: 12 }}>
              <div className="knob" style={{ margin: 0, minWidth: 200 }}><div className="kr"><span>Chunk size (words)</span><b>{size}</b></div><input type="range" min={15} max={120} step={5} value={size} onChange={(e) => setSize(+e.target.value)} /></div>
              <div className="knob" style={{ margin: 0, minWidth: 170 }}><div className="kr"><span>Overlap (words)</span><b>{overlap}</b></div><input type="range" min={0} max={40} step={2} value={overlap} onChange={(e) => setOverlap(+e.target.value)} /></div>
              <button className="btn" onClick={runChunking} disabled={chunking}>▶ Run chunking</button>
            </div>

            {chunks.length > 0 && (
              <>
                <label className="fld">How the extracted text is split into chunks — a sliding window of {size} words moving {stepWords} words each step ({overlap}w overlap)</label>
                <div className="flow" key={`cf-${chunkPlayKey}`} style={{ ["--sweepdur"]: `${(chunks.length * 0.35 + 0.6).toFixed(1)}s` } as React.CSSProperties}>
                  <div className="flow-label">extracted text · {totalWords} words</div>
                  <div className="doc-track">
                    <div className="readhead" />
                    {chunks.map((c, i) => (
                      <div key={i} className="cwin" style={{ left: `${Math.min(98, (i * stepWords / Math.max(1, totalWords)) * 100)}%`, width: `${Math.min(60, (size / Math.max(1, totalWords)) * 100)}%`, background: i % 2 ? "var(--sky)" : "var(--accent)", animationDelay: `${(i * 0.35).toFixed(2)}s` }}>c{i + 1}</div>
                    ))}
                  </div>
                  <div className="flow-arrow">↓ splits into {chunks.length} chunks</div>
                  <div className="flow-chunks">
                    {chunks.map((c, i) => <div key={i} className="fchunk" style={{ animationDelay: `${(i * 0.35 + 0.2).toFixed(2)}s` }}>chunk {i + 1}</div>)}
                  </div>
                </div>
                <div className="row" style={{ marginBottom: 12 }}><button className="btn ghost sm" onClick={() => setChunkPlayKey((k) => k + 1)}>↻ Replay animation</button></div>

                {/* per-chunk inspector — how the text turns into ONE chunk (with overlap) */}
                <label className="fld">Watch one chunk form — step through to see the exact words &amp; the shared overlap</label>
                {(() => {
                  const ci = Math.min(inspIdx, chunks.length - 1);
                  const c = chunks[ci];
                  const words = c.text.split(/\s+/).filter(Boolean);
                  const ov = Math.min(overlap, Math.floor(words.length / 2));
                  const start = ci * stepWords;
                  return (
                    <div className="chunk-inspect">
                      <div className="pp-player" style={{ margin: "0 0 8px" }}>
                        <button className="pp-ctrl" title="First" onClick={() => { setInspPlaying(false); setInspIdx(0); }}>⏮</button>
                        <button className="pp-ctrl" title="Previous" onClick={() => { setInspPlaying(false); setInspIdx((i) => Math.max(0, i - 1)); }}>‹</button>
                        <button className="pp-ctrl play" onClick={() => { if (ci >= chunks.length - 1) { setInspIdx(0); setInspPlaying(true); } else setInspPlaying((p) => !p); }}>{inspPlaying ? "⏸ Pause" : (ci >= chunks.length - 1 ? "↻ Replay" : "▶ Play")}</button>
                        <button className="pp-ctrl" title="Next" onClick={() => { setInspPlaying(false); setInspIdx((i) => Math.min(chunks.length - 1, i + 1)); }}>›</button>
                        <span className="pp-count">chunk {ci + 1} / {chunks.length} · words {start + 1}–{start + words.length}</span>
                        <span className="pp-speed">speed<select value={inspSpeed} onChange={(e) => setInspSpeed(+e.target.value)}><option value={1800}>0.5×</option><option value={1100}>1×</option><option value={550}>2×</option></select></span>
                      </div>
                      <div className="pp-progress"><i style={{ width: `${(ci / Math.max(1, chunks.length - 1)) * 100}%` }} /></div>
                      <div className="ci-meta"><span className="src-tag">{c.docKind}</span> {c.docName} · {words.length} words{ov > 0 ? ` · ${ov}w shared with neighbours` : ""}</div>
                      <div className="ci-words" key={ci}>
                        {words.map((w, j) => {
                          const prevOv = ci > 0 && j < ov;
                          const nextOv = ci < chunks.length - 1 && j >= words.length - ov;
                          return <span key={j} className={`ci-w ${prevOv ? "ov-prev" : ""} ${nextOv ? "ov-next" : ""}`} style={{ animationDelay: `${Math.min(j * 0.012, 0.5)}s` }}>{w}</span>;
                        })}
                      </div>
                      <div className="ci-legend"><span className="lg ov-prev">overlap ← previous</span><span className="lg mid">unique to this chunk</span><span className="lg ov-next">overlap → next</span></div>
                      <div className="note" style={{ marginTop: 8 }}>A sliding window of {size} words becomes chunk {ci + 1}; its first &amp; last {overlap} words are shared with the neighbouring chunks so context isn&apos;t cut mid-sentence.</div>
                    </div>
                  );
                })()}
              </>
            )}

            <label className="fld">All chunks (with source) — scroll to see more</label>
            <div className="chunk-scroll">
              {chunks.map((c, i) => (
                <div key={i} className={`chunk-card ${i === Math.min(inspIdx, chunks.length - 1) ? "on" : ""}`}>
                  <div className="ch"><span>chunk {i + 1}<span className="src-tag">{c.docKind}</span> {c.docName}</span><span>words {i * stepWords + 1}–{i * stepWords + size} · {c.text.split(/\s+/).length}w</span></div>
                  <div>{c.text.length > 220 ? c.text.slice(0, 220) + "…" : c.text}</div>
                </div>
              ))}
            </div>
            {chunks.length === 0 && !chunking && <div className="note">Click Run chunking to split the documents.</div>}
            <div className="stepnav"><button className="btn ghost" onClick={() => goStep("source")}>← Back</button><button className="btn" disabled={chunks.length === 0} onClick={() => goStep("embed")}>Next: Embed →</button></div>
          </div>
        </div>
      )}

      {/* STEP 3 — INDEX (vector store / knowledge graph / hybrid) */}
      {step === "embed" && (
        <div className="card">
          <div className="card-h"><span className="t">{backend === "kg" ? "Build the knowledge graph" : backend === "hybrid" ? "Index — vectors + graph" : "Embed & store in vector index"}</span><span className="mono r">{embedding || buildingKg ? <><span className="busy-dot" />{buildingKg ? "extracting…" : "vectorizing…"}</> : backend === "kg" ? (graph ? `${graph.nodes.length} nodes · ${graph.edges.length} edges` : "not built") : index ? `${index.vectors.length} vectors` : "not built"}</span></div>
          <div className="card-b">
            {backendSel}

            {backend !== "vector" && (
              <div style={{ ...pnl, marginBottom: 16 }}>
                {kgHead("#a855f7", "Knowledge graph", graph ? <span className="note" style={{ fontSize: 10 }}>{graph.nodes.length} entities · {graph.edges.length} relations</span> : undefined)}
                <div style={{ padding: 14 }}>
                  <div className="row" style={{ gap: 12, alignItems: "center", flexWrap: "wrap", marginBottom: 12 }}>
                    <div><label className="note" style={{ display: "block", marginBottom: 4 }}>Extraction</label>
                      <div className="chips">
                        <button className={`chip ${kgExtract === "heuristic" ? "on" : ""}`} onClick={() => setKgExtract("heuristic")}>Heuristic</button>
                        <button className={`chip ${kgExtract === "llm" ? "on" : ""}`} onClick={() => setKgExtract("llm")} disabled={provider === null && provKnown} title={provider === null ? "needs a provider" : ""}>LLM-assisted</button>
                      </div>
                    </div>
                    {kgExtract === "heuristic" && <div className="knob" style={{ margin: 0, minWidth: 160 }}><div className="kr"><span>Max entities</span><b>{maxNodes}</b></div><input type="range" min={10} max={36} step={2} value={maxNodes} onChange={(e) => setMaxNodes(+e.target.value)} /></div>}
                    <button className="btn" onClick={buildGraph} disabled={buildingKg || !chunks.length}>{buildingKg ? "extracting…" : graph ? "↻ Rebuild graph" : "▶ Build graph"}</button>
                    {kgExtract === "llm" && provider && <span className="note">uses {provider} to extract clean triples</span>}
                    {kgExtract === "llm" && provider === null && provKnown && <span className="note">no provider — heuristic used instead</span>}
                  </div>
                  {graph && graphPos && graph.nodes.length > 0 ? (
                    <div className="split col-2e" style={{ gap: 14, alignItems: "start" }}>
                      <div>
                        {graphSvg(graph, graphPos)}
                        <div className="row" style={{ gap: 14, marginTop: 8, fontSize: 11, color: "var(--muted)", flexWrap: "wrap" }}>
                          <span className="row" style={{ gap: 5, alignItems: "center" }}><span style={{ width: 8, height: 8, borderRadius: "50%", background: "#5b7cff" }} />concept</span>
                          <span className="row" style={{ gap: 5, alignItems: "center" }}><span style={{ width: 8, height: 8, borderRadius: "50%", background: "#a855f7" }} />proper noun</span>
                          <span className="note">node size = frequency · edge label = relation</span>
                        </div>
                      </div>
                      <div>
                        <label className="fld">Extracted triples · {graph.edges.length}</label>
                        <div style={{ maxHeight: 300, overflowY: "auto", border: "1px solid var(--border)", borderRadius: 10 }}>
                          <table className="dtable" style={{ width: "100%" }}><tbody>
                            <tr><th style={{ textAlign: "left" }}>subject</th><th style={{ textAlign: "left" }}>relation</th><th style={{ textAlign: "left" }}>object</th></tr>
                            {graph.edges.slice(0, 40).map((e, i) => <tr key={i}><td style={{ color: "#5b7cff" }}>{graph.nodes.find((n) => n.id === e.s)?.label}</td><td style={{ color: "var(--orange)", fontStyle: "italic" }}>{e.rel}</td><td style={{ color: "#3ecf7f" }}>{graph.nodes.find((n) => n.id === e.o)?.label}</td></tr>)}
                          </tbody></table>
                        </div>
                        <div className="note" style={{ marginTop: 8 }}>{kgExtract === "heuristic" ? "Heuristic extraction: frequent entities linked by intra-sentence co-occurrence. Rough but deterministic — switch to LLM-assisted for cleaner triples." : "LLM-extracted triples grounded back to the chunks they appear in."}</div>
                      </div>
                    </div>
                  ) : !buildingKg && <div className="note">Build the graph to extract entities and relations from the {chunks.length} chunks.</div>}
                </div>
              </div>
            )}

            {backend !== "kg" && (<>
            <div className="stat-row">
              <div className="stat">chunks<b>{chunks.length}</b></div>
              <div className="stat">vectors stored<b>{index ? index.vectors.length : 0}</b></div>
              <div className="stat">vocabulary (dims)<b>{vocab}</b></div>
              <div className="stat">similarity<b>cosine</b></div>
            </div>
            <div className="row" style={{ marginBottom: 12 }}><button className="btn" onClick={runEmbedding} disabled={embedding}>▶ Run embedding</button><span className="note">turns each chunk into a vector and stores it</span></div>

            <div style={{ marginBottom: 14, borderTop: "1px solid var(--border)", paddingTop: 12 }}>
              <label className="fld">Vector backend — lexical (TF-IDF) or real semantic embeddings</label>
              <div className="row" style={{ gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                <div className="chips">
                  <button className={`chip ${embedMode === "tfidf" ? "on" : ""}`} onClick={() => setEmbedMode("tfidf")}>TF-IDF · lexical</button>
                  <button className={`chip ${embedMode === "neural" ? "on" : ""}`} onClick={() => { if (denseVecs) setEmbedMode("neural"); else runNeuralEmbed(); }} disabled={provider === null && provKnown}>Neural · semantic</button>
                </div>
                {embedMode === "neural" && !denseVecs && <button className="btn sm" onClick={runNeuralEmbed} disabled={embedding || (provider === null && provKnown)}>{embedding ? "embedding…" : "▶ Embed with neural model"}</button>}
                {embedMode === "neural" && denseVecs && embedInfo && <span className="note">real embeddings · <b>{embedInfo.model}</b> · {embedInfo.dim} dims — vector search is now <b>semantic</b>, not lexical</span>}
                {provider === null && provKnown && <span className="note">neural embeddings need a provider (Admin → Providers) — TF-IDF works without one</span>}
              </div>
              {embedMode === "neural" && denseVecs && denseVecs.length > 2 && (() => {
                const t = plotlyTheme(); const pts = pca2(denseVecs); const asg = clusters?.assign ?? denseVecs.map(() => 0); const K = clusters?.nlist ?? 1;
                const traces = [...Array(K).keys()].map((c) => ({ type: "scatter", mode: "markers+text", name: `cluster ${c + 1}`, x: pts.map((p, i) => (asg[i] === c ? p.x : null)), y: pts.map((p, i) => (asg[i] === c ? p.y : null)), text: pts.map((_, i) => (asg[i] === c ? String(i + 1) : "")), textposition: "top center", textfont: { size: 9, color: t.muted }, marker: { size: 11, opacity: 0.85 }, hovertemplate: "chunk %{text}<extra></extra>" }));
                return <div style={{ marginTop: 10 }}><Plot data={traces} layout={{ ...pLayout(t, "Embedding space (PCA → 2-D) — semantically similar chunks sit close together", { showlegend: true, legend: { orientation: "h", y: -0.2 }, height: 340, xaxis: { visible: false }, yaxis: { visible: false } }) }} style={{ height: 340, width: "100%" }} /></div>;
              })()}
            </div>

            {index && (() => {
              const ei = Math.min(embIdx, chunks.length - 1);
              const toks = index.docs[ei];
              const vec = index.vectors[ei];
              const tf: Record<string, number> = {};
              toks.forEach((t) => { tf[t] = (tf[t] || 0) + 1; });
              const top = Object.entries(vec).sort((a, b) => b[1] - a[1]).slice(0, 8);
              const maxW = top[0]?.[1] || 1;
              const N = index.N;
              const totals: Record<string, number> = {};
              index.vectors.forEach((v) => Object.entries(v).forEach(([t, w]) => { totals[t] = (totals[t] || 0) + w; }));
              const mTerms = Object.entries(totals).sort((a, b) => b[1] - a[1]).slice(0, 10).map((e) => e[0]);
              let mMax = 0; index.vectors.forEach((v) => mTerms.forEach((t) => { if ((v[t] || 0) > mMax) mMax = v[t] || 0; })); mMax = mMax || 1;
              const nlist = clusters?.nlist ?? Math.min(4, chunks.length);
              const buckets = clusters?.buckets ?? [];
              return (
                <>
                  <label className="fld">Watch one chunk become a vector — step through the embedding pipeline</label>
                  <div className="chunk-inspect">
                    <div className="pp-player" style={{ margin: "0 0 8px" }}>
                      <button className="pp-ctrl" title="First" onClick={() => { setEmbPlaying(false); setEmbIdx(0); }}>⏮</button>
                      <button className="pp-ctrl" title="Previous" onClick={() => { setEmbPlaying(false); setEmbIdx((i) => Math.max(0, i - 1)); }}>‹</button>
                      <button className="pp-ctrl play" onClick={() => { if (ei >= chunks.length - 1) { setEmbIdx(0); setEmbPlaying(true); } else setEmbPlaying((p) => !p); }}>{embPlaying ? "⏸ Pause" : (ei >= chunks.length - 1 ? "↻ Replay" : "▶ Play")}</button>
                      <button className="pp-ctrl" title="Next" onClick={() => { setEmbPlaying(false); setEmbIdx((i) => Math.min(chunks.length - 1, i + 1)); }}>›</button>
                      <span className="pp-count">chunk {ei + 1} / {chunks.length} · {vocab} dims total</span>
                      <span className="pp-speed">speed<select value={embSpeed} onChange={(e) => setEmbSpeed(+e.target.value)}><option value={2200}>0.5×</option><option value={1400}>1×</option><option value={700}>2×</option></select></span>
                    </div>
                    <div className="emb-pipe" key={ei}>
                      <div className="emb-stage">
                        <div className="emb-h"><span className="emb-n">1</span>text → tokens</div>
                        <div className="emb-toks">{toks.slice(0, 40).map((t, j) => <span key={j} className="emb-tok" style={{ animationDelay: `${Math.min(j * 0.015, 0.6)}s` }}>{t}</span>)}{toks.length > 40 ? <span className="note">+{toks.length - 40} more</span> : null}</div>
                        <div className="note">{toks.length} tokens — lowercased, split on non-alphanumeric</div>
                      </div>
                      <div className="emb-stage">
                        <div className="emb-h"><span className="emb-n">2</span>weigh each term · TF-IDF = (count ÷ length) × ln(N ÷ df + 1)</div>
                        <div style={{ overflowX: "auto" }}><table className="emb-table"><tbody>
                          <tr><th>term</th><th>count</th><th>df</th><th>idf</th><th>weight</th><th style={{ width: "38%" }} /></tr>
                          {top.map(([t, w]) => { const df = index.df[t] || 1; const idf = Math.log(N / df + 1); return (
                            <tr key={t}><td>{t}</td><td>{tf[t]}</td><td>{df}/{N}</td><td>{idf.toFixed(2)}</td><td>{w.toFixed(3)}</td><td className="emb-barcell"><div className="emb-bar"><i style={{ width: `${Math.round((w / maxW) * 100)}%` }} /></div></td></tr>
                          ); })}
                        </tbody></table></div>
                        <div className="note" style={{ marginTop: 6 }}>Rare words (low df) get a high idf → they represent the chunk more; common words are down-weighted.</div>
                      </div>
                      <div className="emb-stage">
                        <div className="emb-h"><span className="emb-n">3</span>assemble the vector → store it</div>
                        <div className="emb-vecrow"><div className="emb-vec">{top.map(([t, w], j) => <i key={j} style={{ height: `${Math.round(15 + (w / maxW) * 85)}%` }} title={`${t}: ${w.toFixed(3)}`} />)}</div><span className="arrow-anim">→</span><span className="note">a {vocab}-dim sparse vector ({Object.keys(vec).length} non-zero) stored as row {ei + 1} of the vector database</span></div>
                      </div>
                    </div>
                  </div>

                  <label className="fld">Vector store — chunks (rows) × top terms (columns), shaded by weight</label>
                  <div className="vstore" style={{ gridTemplateColumns: `64px repeat(${mTerms.length}, 1fr)` }}>
                    <div className="vs-corner" />
                    {mTerms.map((t) => <div key={t} className="vs-col" title={t}>{t}</div>)}
                    {chunks.flatMap((c, r) => [
                      <div key={`r${r}`} className={`vs-row ${r === ei ? "on" : ""}`}>c{r + 1}</div>,
                      ...mTerms.map((t) => { const w = index.vectors[r][t] || 0; return <div key={`${r}-${t}`} className={`vs-cell ${r === ei ? "on" : ""}`} style={{ background: "var(--accent)", opacity: w ? 0.12 + 0.88 * (w / mMax) : 0 }} title={`c${r + 1} · ${t}: ${w.toFixed(3)}`} />; }),
                    ])}
                  </div>
                  <div className="note" style={{ marginTop: 8 }}>Darker = that term matters more to that chunk. A question is embedded the exact same way; retrieval scores chunks whose strong terms <b>overlap</b> the question (cosine similarity).</div>

                  {/* how a real vector DB (Milvus) stores this */}
                  <label className="fld" style={{ marginTop: 14 }}>How a real vector DB stores this — Milvus collection</label>
                  <div className="milvus">
                    <div className="mv-schema">
                      <div className="mv-line"><b>Collection</b><span>rag_chunks</span></div>
                      <div className="mv-badges"><span className="mv-b">metric: COSINE</span><span className="mv-b">index: IVF_FLAT</span><span className="mv-b">nlist: {nlist}</span><span className="mv-b">dim: {vocab}</span><span className="mv-b">entities: {chunks.length}</span></div>
                      <div className="mv-fields">
                        <div className="mv-f"><span className="mv-fn">id</span><span className="mv-ft">INT64</span><span className="mv-fk">primary key</span></div>
                        <div className="mv-f"><span className="mv-fn">embedding</span><span className="mv-ft">FLOAT_VECTOR</span><span className="mv-fk">dim = {vocab}</span></div>
                        <div className="mv-f"><span className="mv-fn">text</span><span className="mv-ft">VARCHAR</span><span className="mv-fk">scalar field</span></div>
                        <div className="mv-f"><span className="mv-fn">source</span><span className="mv-ft">VARCHAR</span><span className="mv-fk">scalar field</span></div>
                      </div>
                    </div>
                    <label className="fld" style={{ marginTop: 12 }}>Entities (rows stored in the collection)</label>
                    <div className="chunk-scroll" style={{ maxHeight: 220, padding: 0 }}>
                      <table className="dtable"><tbody>
                        <tr><th>id</th><th>embedding (first dims)</th><th>text</th><th>source</th></tr>
                        {chunks.map((c, i) => { const preview = mTerms.slice(0, 6).map((t) => (index.vectors[i][t] || 0).toFixed(2)).join(", "); return (
                          <tr key={i} className={i === ei ? "mv-on" : ""}><td>{i}</td><td className="mv-vec">[{preview}, …]</td><td style={{ maxWidth: 260, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{c.text.slice(0, 64)}…</td><td>{c.docName.length > 18 ? c.docName.slice(0, 18) + "…" : c.docName}</td></tr>
                        ); })}
                      </tbody></table>
                    </div>
                    <label className="fld" style={{ marginTop: 12 }}>IVF_FLAT index — vectors clustered into {nlist} buckets; a search scans only the nearest bucket(s)</label>
                    <div className="mv-buckets">
                      {buckets.map((ids, c) => (
                        <div key={c} className="mv-bucket">
                          <div className="mv-bh">bucket {c + 1} <span className="note">· {ids.length} vectors</span></div>
                          <div className="mv-chips">{ids.map((i) => <span key={i} className={`mv-chip ${i === ei ? "on" : ""}`}>c{i + 1}</span>)}{ids.length === 0 && <span className="note">empty</span>}</div>
                        </div>
                      ))}
                    </div>
                    <div className="note" style={{ marginTop: 8 }}>Milvus stores each chunk as an <b>entity</b> (id + float vector + metadata) inside a <b>collection</b>, then builds an ANN index — IVF_FLAT clusters vectors into <b>nlist</b> buckets so a query only compares against the closest few (nprobe), staying fast at millions of vectors. Our in-browser store uses the same shape (TF-IDF vectors + cosine); swap in a neural embedder and it maps 1:1 to Milvus.</div>
                  </div>

                  <label className="fld" style={{ marginTop: 14 }}>All chunk vectors (top terms) — scroll to see more</label>
                  <div className="chunk-scroll">
                    {chunks.map((c, i) => { const v = index.vectors[i]; const terms = topTermsW(v); return (
                      <div key={i} className={`chunk-card ${i === ei ? "on" : ""}`} style={{ borderLeftColor: "var(--sky)" }}>
                        <div className="ch"><span>chunk {i + 1} <span className="arrow-anim">→</span> vector<span className="src-tag">{c.docKind}</span></span><span>{Object.keys(v).length} dims</span></div>
                        {terms.map((t) => (<div key={t.term} className="tbar"><span className="tl">{t.term}</span><div className="tbaro"><i style={{ width: `${Math.round(t.w * 100)}%` }} /></div><span className="tw">{t.w.toFixed(2)}</span></div>))}
                      </div>); })}
                  </div>
                </>
              );
            })()}
            {!index && !embedding && <div className="note">Click Run embedding to vectorize the chunks.</div>}
            <div className="note" style={{ marginTop: 8 }}>Shown as TF-IDF term weights (clear to read). Neural embeddings (e.g. bge-small) can replace this backend — the pipeline is identical.</div>
            </>)}
            <div className="stepnav"><button className="btn ghost" onClick={() => goStep("chunk")}>← Back</button><button className="btn" disabled={!canQuery} onClick={() => goStep("query")}>Next: Retrieve &amp; Answer →</button></div>
          </div>
        </div>
      )}

      {/* STEP 4 — QUERY */}
      {step === "query" && (
        <div className="split col-2">
          <div className="card">
            <div className="card-h"><span className="t">Retrieve &amp; ask</span></div>
            <div className="card-b">
              {!canQuery && <div className="warnbar">Build the index first (step 3).</div>}
              <div className="row" style={{ justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}><label className="fld" style={{ margin: 0 }}>Retrieval parameters</label><span className="note" style={{ fontSize: 10.5, textTransform: "uppercase", letterSpacing: ".05em", color: backend === "kg" ? "#a855f7" : backend === "hybrid" ? "#f59e0b" : "var(--accent)" }}>{backend === "kg" ? "🕸 knowledge graph" : backend === "hybrid" ? "⚡ hybrid" : "◆ vector store"}</span></div>
              <div className="row" style={{ flexWrap: "wrap", gap: 10, marginBottom: 12 }}>
                {backend !== "kg" && <select value={strategy} onChange={(e) => setStrategy(e.target.value as Strategy)} style={{ width: 168 }}>
                  <option value="hybrid">Hybrid</option><option value="vector">Vector ({embedMode === "neural" ? "neural" : "TF-IDF"})</option><option value="keyword">Keyword (BM25)</option>
                </select>}
                {backend !== "kg" && <select value={rerank} onChange={(e) => setRerank(e.target.value as "none" | "mmr")} style={{ width: 168 }} title="Re-ranking">
                  <option value="none">No re-ranking</option><option value="mmr">MMR re-rank (diversity)</option>
                </select>}
                {backend !== "kg" && rerank === "mmr" && <div className="knob" style={{ margin: 0, minWidth: 150 }}><div className="kr"><span>λ (relevance↔diversity)</span><b>{mmrLambda.toFixed(2)}</b></div><input type="range" min={0} max={1} step={0.05} value={mmrLambda} onChange={(e) => setMmrLambda(+e.target.value)} /></div>}
                {backend !== "vector" && <div className="knob" style={{ margin: 0, minWidth: 150 }}><div className="kr"><span>Graph hops</span><b>{kgHops}</b></div><input type="range" min={1} max={3} value={kgHops} onChange={(e) => setKgHops(+e.target.value)} /></div>}
                <div className="knob" style={{ margin: 0, minWidth: 140 }}><div className="kr"><span>Top-k</span><b>{topK}</b></div><input type="range" min={1} max={6} value={topK} onChange={(e) => setTopK(+e.target.value)} /></div>
              </div>
              <label className="fld">Generation model — the LLM that writes the grounded answer</label>
              <div className="row" style={{ gap: 10, flexWrap: "wrap", marginBottom: 12, alignItems: "center" }}>
                {providers.length > 1 && <select value={providerId} onChange={(e) => { setProviderId(e.target.value); loadModels(e.target.value); }} style={{ width: 176 }} title="Provider">{providers.map((p) => <option key={p.id} value={p.id}>{p.label || p.provider}</option>)}</select>}
                <select value={model} onChange={(e) => setModel(e.target.value)} style={{ width: 244 }} disabled={modelsLoading || !models.length} title="Model">
                  {modelsLoading ? <option>loading…</option> : models.length ? models.map((m) => <option key={m} value={m}>{m}</option>) : <option value="">no models available</option>}
                </select>
                <button className="btn ghost sm" onClick={() => loadModels(providerId || undefined)} disabled={modelsLoading} title="Refresh model list">↻</button>
                {provider && <span className="note">{providers.length > 1 ? provider + " · " : ""}{models.length} model{models.length === 1 ? "" : "s"}</span>}
                {provKnown && !provider && <span className="note" style={{ color: "var(--warn)" }}>no provider configured — add one under Admin → Providers</span>}
              </div>
              <label className="fld">Question</label>
              <input type="text" value={question} onChange={(e) => setQuestion(e.target.value)} />
              <div className="row" style={{ marginTop: 12 }}><button className="btn" onClick={ask} disabled={running || !canQuery}>▶ Ask</button></div>

              {backend !== "vector" && graph && graphPos && kgVisited.length > 0 && (
                <div style={{ ...pnl, marginTop: 16 }}>
                  {kgHead("#a855f7", "Graph traversal", <span className="note" style={{ fontSize: 10 }}>{kgSeeds.length} matched · {kgVisited.length} in subgraph</span>)}
                  <div style={{ padding: 14 }}>
                    {kgPath.length > 0 && <div className="row" style={{ gap: 6, flexWrap: "wrap", alignItems: "center", marginBottom: 12, fontSize: 11.5 }}>
                      {kgPath.slice(0, 6).map((e, i) => <span key={i} className="row" style={{ gap: 6, alignItems: "center" }}><span style={{ padding: "3px 9px", borderRadius: 20, background: "rgba(62,207,127,.14)", color: "#3ecf7f", fontWeight: 600 }}>{graph.nodes.find((n) => n.id === e.s)?.label}</span><span style={{ color: "var(--orange)", fontStyle: "italic", fontSize: 10.5 }}>{e.rel} →</span><span style={{ padding: "3px 9px", borderRadius: 20, background: "rgba(91,124,255,.14)", color: "#5b7cff", fontWeight: 600 }}>{graph.nodes.find((n) => n.id === e.o)?.label}</span></span>)}
                    </div>}
                    {graphSvg(graph, graphPos, new Set(kgVisited), kgPath)}
                    <div className="note" style={{ marginTop: 6 }}>Green = query-matched entities · highlighted = subgraph within {kgHops} hop(s). The chunks attached to these entities become the context.</div>
                  </div>
                </div>
              )}
              {hits.length === 0 && <div className="note" style={{ marginTop: 16 }}>Ask to retrieve.</div>}
              {index && hits.length > 0 && (() => {
                const qTerms = Array.from(new Set(tokenize(question)));
                const qv = queryVector(index, question);
                let probed = 0, bs = -Infinity;
                if (clusters) clusters.centroids.forEach((ct, c) => { const s = cosine(qv, ct); if (s > bs) { bs = s; probed = c; } });
                return (
                  <>
                    <label className="fld" style={{ marginTop: 16 }}>Query → tokens (matched against the {backend === "kg" ? "graph entities" : "chunk vectors"})</label>
                    <div className="q-terms">{qTerms.map((t) => <span key={t} className="q-term">{t}</span>)}</div>

                    {backend !== "kg" && clusters && (
                      <>
                        <label className="fld" style={{ marginTop: 14 }}>ANN search path (Milvus IVF) — the query vector lands in its nearest bucket; only that bucket is scanned</label>
                        <div className="mv-buckets">
                          {clusters.buckets.map((ids, c) => (
                            <div key={c} className={`mv-bucket ${c === probed ? "probed" : "dim"}`}>
                              <div className="mv-bh">bucket {c + 1} {c === probed ? <span className="q-badge">◎ probed</span> : <span className="note">skipped</span>}</div>
                              <div className="mv-chips">{ids.map((i) => <span key={i} className={`mv-chip ${hits.some((h) => h.i === i) ? "hit" : ""}`}>c{i + 1}</span>)}{ids.length === 0 && <span className="note">empty</span>}</div>
                            </div>
                          ))}
                        </div>
                        <div className="note" style={{ marginTop: 6 }}>Milvus compares the query only against vectors in the closest bucket(s) (nprobe), not all {chunks.length}, then ranks them by cosine — green chips are the chunks that were retrieved.</div>
                      </>
                    )}

                    <label className="fld" style={{ marginTop: 14 }}>Retrieved chunks — which query terms matched</label>
                    {hits.map((h) => {
                      const chunkToks = new Set(index.docs[h.i]);
                      const matched = qTerms.filter((t) => chunkToks.has(t));
                      return (
                        <div key={h.i} className="chunk-card reveal-in">
                          <div className="ch"><span>chunk {h.i + 1}<span className="src-tag">{chunks[h.i].docKind}</span> {chunks[h.i].docName}</span><span style={{ display: "inline-flex", gap: 10, alignItems: "center" }}><label className="note" style={{ cursor: "pointer", display: "inline-flex", alignItems: "center", gap: 4 }}><input type="checkbox" checked={relevant.has(h.i)} onChange={() => setRelevant((s) => { const n = new Set(s); if (n.has(h.i)) n.delete(h.i); else n.add(h.i); return n; })} />relevant</label><span style={{ color: "var(--accent)" }}>score {h.score.toFixed(2)}</span></span></div>
                          <div className="q-match">{qTerms.map((t) => <span key={t} className={`q-chip ${matched.includes(t) ? "on" : ""}`}>{t}</span>)}</div>
                          <div className="q-matchnote">{matched.length}/{qTerms.length} query terms overlap this chunk</div>
                          <div style={{ color: "var(--muted)", marginTop: 6 }}>{highlightTerms(chunks[h.i].text.slice(0, 170), matched)}…</div>
                        </div>
                      );
                    })}
                    <div style={{ marginTop: 14, borderTop: "1px solid var(--border)", paddingTop: 12 }}>
                      <label className="fld">Retrieval quality — tick the chunks that actually answer the question, then evaluate</label>
                      <div className="row" style={{ gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                        <button className="btn sm" onClick={evalRetrieval} disabled={relevant.size === 0}>▶ Evaluate P@k / recall / MRR / nDCG</button>
                        <span className="note">{relevant.size} chunk{relevant.size === 1 ? "" : "s"} marked relevant</span>
                        {relevant.size > 0 && <button className="btn ghost sm" onClick={() => { setRelevant(new Set()); setMetricRows([]); }}>clear</button>}
                      </div>
                      {metricRows.length > 0 && (() => {
                        const t = plotlyTheme(); const mk: [string, "p" | "r" | "mrr" | "ndcg"][] = [["P@k", "p"], ["recall@k", "r"], ["MRR", "mrr"], ["nDCG", "ndcg"]];
                        const traces = metricRows.map((row, ri) => ({ type: "bar", name: row.name, x: mk.map((m) => m[0]), y: mk.map((m) => row[m[1]]), marker: { color: t.colorway[ri % t.colorway.length] }, text: mk.map((m) => row[m[1]].toFixed(2)), textposition: "outside", cliponaxis: false }));
                        return <div style={{ marginTop: 10 }}><Plot data={traces} layout={{ ...pLayout(t, `Retrieval metrics @top-${topK} vs your ${relevant.size} relevant chunk(s)`, { barmode: "group", showlegend: true, legend: { orientation: "h", y: -0.2 }, height: 320, yaxis: { range: [0, 1.12] } }) }} style={{ height: 320, width: "100%" }} /></div>;
                      })()}
                      {metricRows.length > 0 && <div className="note" style={{ marginTop: 6 }}>Higher is better. Compare strategies (and MMR on/off) to see which retrieval surfaces the relevant chunks first — this is how you’d pick a retriever objectively instead of by eye.</div>}
                    </div>
                  </>
                );
              })()}
            </div>
          </div>
          <div className="card">
            <div className="card-h"><span className="t">Grounded answer</span><span className="mono">{meta}</span>
              <div className="tabs"><button className={tab === "out" ? "on" : ""} onClick={() => setTab("out")}>Output</button><button className={tab === "trace" ? "on" : ""} onClick={() => setTab("trace")}>Trace</button></div>
            </div>
            <div className="card-b">
              {tab === "out"
                ? <div className="out">{answer}{running && <span className="cur" />}</div>
                : <div className="tracebox">{trace.length === 0 ? "Ask to see the pipeline run." : trace.map((s, i) => <div key={i} className={`trow ${s.state}`}><span className="who">{s.who}</span><span>{s.what}</span></div>)}</div>}
              {tab === "out" && hits.length > 0 && !running && (
                <div className="sources-box">
                  <label className="fld">Sources cited</label>
                  {hits.map((h) => <div key={h.i} className="src"><span className="src-tag">{chunks[h.i].docKind}</span>[chunk {h.i + 1}] {chunks[h.i].docName}</div>)}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {step === "query" && index && (
        <div className="card" style={{ marginTop: 16 }}>
          <div className="card-h"><span className="t">Compare chunking</span><span className="mono r">{strategy} · top-{topK}</span></div>
          <div className="card-b">
            <div className="note" style={{ marginBottom: 10 }}>Rebuilds the index at several chunk sizes and retrieves for “{question}”. Higher <b>top score</b> = the best chunk matches more strongly — bigger chunks aren&apos;t always better. (Independent of your main pipeline above.)</div>
            <button className="btn sm" onClick={compareChunking} style={{ marginBottom: compareRows.length ? 14 : 0 }}>{compareRows.length ? "↻ Re-run comparison" : "▶ Compare chunk sizes"}</button>
            {compareRows.length > 0 && (() => {
              const best = Math.max(...compareRows.map((r) => r.top), 0.0001);
              const winner = compareRows.reduce((a, b) => (b.top > a.top ? b : a));
              return (
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  {compareRows.map((r, i) => {
                    const win = r.top === best && r.top > 0;
                    return (
                      <div key={i} style={{ display: "flex", alignItems: "center", gap: 10 }}>
                        <span className="mono" style={{ width: 88, flex: "0 0 auto", fontSize: 12 }}>{r.size}w·{r.overlap}o</span>
                        <div style={{ flex: 1, background: "var(--panel-2)", borderRadius: 5, height: 18, position: "relative", overflow: "hidden" }}>
                          <div style={{ width: `${Math.max(3, (r.top / best) * 100)}%`, height: "100%", background: win ? "var(--good)" : "var(--accent)", borderRadius: 5, transition: "width .3s" }} />
                          <span style={{ position: "absolute", right: 8, top: 0, lineHeight: "18px", fontSize: 10.5, fontFamily: "var(--mono)", color: "var(--muted)" }}>{r.top.toFixed(3)}{win ? " ⭐" : ""}</span>
                        </div>
                        <span className="note" style={{ width: 62, flex: "0 0 auto", textAlign: "right" }}>{r.chunks} ck</span>
                      </div>
                    );
                  })}
                  <div className="note" style={{ marginTop: 6, lineHeight: 1.5 }}>Best: <b>{winner.size}-word chunks</b> → “{winner.best.slice(0, 130)}{winner.best.length > 130 ? "…" : ""}”</div>
                </div>
              );
            })()}
          </div>
        </div>
      )}
    </div>
  );
}
