"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import AgentOutput from "@/components/AgentOutput";
import { explainProviderError } from "@/lib/providerErrors";
import {
  buildIndex,
  bm25ScoresTuned,
  bm25Explain,
  queryVector,
  simSparse,
  tokenize,
  rrfFuse,
  chunkBy,
  CHUNK_STRATEGY_LABEL,
  type ChunkStrategy,
  type Metric,
} from "@/lib/ragUtils";
import { extractGraph, retrieveGraph, type KnowledgeGraph } from "@/lib/kgUtils";
import KnowledgeGraphPlot from "@/components/KnowledgeGraphPlot";
import { planRetrievers, runAgenticRetrieval, detectSignals, type AgenticResult, type RouteSignal } from "@/lib/agenticRag";
import { gradeDocs, gradeRetrieval, rewriteQuery, type DocGrade, type Grade } from "@/lib/cragUtils";
import {
  buildMultimodalIndex,
  retrieveMultimodal,
  sampleMultimodalCorpus,
  type ModalHit,
  type ModalIndex,
  type ModalItem,
  type Modality,
} from "@/lib/multimodalUtils";

// Every architecture gets its own stepper, and every step is a WORKING stage in
// the same shape as the RAG Lab's Source → Chunk → Index → Answer: stat tiles,
// a controls panel with real knobs, an explicit ▶ Run that commits output into
// state, an animated before → after, and the next step gated on having run.
type ArchId = "hybrid" | "graph" | "agentic" | "crag" | "multimodal";

const ARCH_STEPS: Record<ArchId, string[]> = {
  hybrid: ["Source & query", "Dense path", "Sparse path", "Fusion", "Answer"],
  graph: ["Source & query", "Entity extraction", "Knowledge graph", "Subgraph retrieval", "Answer"],
  agentic: ["Source & query", "Planner agent", "Tool loop", "Answer"],
  crag: ["Source & query", "Retrieve", "Evaluator / grader", "Branch", "Answer"],
  multimodal: ["Sources", "Per-modality encoding", "Unified index", "Retrieval", "Answer"],
};

const ARCHES: { id: ArchId; n: string; label: string; tagline: string }[] = [
  { id: "hybrid", n: "01", label: "Hybrid", tagline: "Dense vectors meet sparse keywords." },
  { id: "graph", n: "02", label: "GraphRAG", tagline: "Answers live in the relationships." },
  { id: "agentic", n: "03", label: "Agentic", tagline: "Retrieval becomes a plan, not a step." },
  { id: "crag", n: "04", label: "Corrective", tagline: "Grade the retrieval before you trust it." },
  { id: "multimodal", n: "05", label: "Multimodal", tagline: "One index across text, images, and tables." },
];

const DEFAULT_CORPUS = [
  "Ada Lovelace worked with Charles Babbage on the Analytical Engine project in London.",
  "Charles Babbage designed the Analytical Engine, a mechanical general purpose computer.",
  "Error code E4021 indicates a thermal shutdown in the battery management module.",
  "Grace Hopper developed the first compiler and worked on the Harvard Mark I computer.",
  "Automobiles powered by electricity are increasingly common in Norway and Sweden.",
  "The Harvard Mark I was an electromechanical computer used by the United States Navy.",
  "Thermal shutdown protects lithium cells when the pack exceeds its safe temperature.",
].join("\n");

const GRADE_COLOR: Record<Grade, string> = { correct: "var(--good)", ambiguous: "var(--warn)", incorrect: "var(--crit)" };
const MODALITY_COLOR: Record<Modality, string> = { text: "var(--accent)", table: "var(--sky)", image: "var(--good)" };

const pnl: React.CSSProperties = { border: "1px solid var(--border)", borderRadius: 14, background: "var(--panel)", overflow: "hidden" };
const inputStyle: React.CSSProperties = {
  background: "var(--panel-2)", color: "var(--text)", border: "1px solid var(--border)",
  borderRadius: "var(--rs)", padding: "8px 10px", fontSize: 12.5, fontFamily: "inherit", width: "100%",
};

function head(dot: string, title: string, right?: React.ReactNode) {
  return (
    <div className="row" style={{ alignItems: "center", justifyContent: "space-between", padding: "10px 14px", borderBottom: "1px solid var(--border)", background: "var(--surface)" }}>
      <div className="row" style={{ gap: 8, alignItems: "center" }}>
        <span style={{ width: 7, height: 7, borderRadius: "50%", background: dot }} />
        <span style={{ fontWeight: 600, fontSize: 10.5, textTransform: "uppercase", letterSpacing: ".07em", color: "var(--muted)" }}>{title}</span>
      </div>
      {right}
    </div>
  );
}

function Tiles({ items }: { items: [string, string][] }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: `repeat(${items.length},1fr)`, gap: 12, marginBottom: 16 }}>
      {items.map(([v, k]) => (
        <div key={k} style={{ background: "var(--panel)", border: "1px solid var(--border)", borderRadius: 12, padding: "12px 15px" }}>
          <div style={{ fontFamily: "var(--mono)", fontSize: 22, fontWeight: 600, letterSpacing: "-.02em", lineHeight: 1.1 }}>{v}</div>
          <div style={{ fontSize: 9.5, textTransform: "uppercase", letterSpacing: ".06em", color: "var(--faint)", marginTop: 3 }}>{k}</div>
        </div>
      ))}
    </div>
  );
}

function Knob({ label, value, min, max, step, onChange, fmt }: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; fmt?: (v: number) => string }) {
  return (
    <div className="knob" style={{ margin: 0, minWidth: 185 }}>
      <div className="kr"><span>{label}</span><b>{fmt ? fmt(value) : value}</b></div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(+e.target.value)} />
    </div>
  );
}

function Cfg({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ marginTop: 12, fontFamily: "var(--mono)", fontSize: 11, color: "var(--faint)", background: "var(--panel-2)", border: "1px solid var(--border)", borderRadius: 8, padding: "8px 11px" }}>
      {children}
    </div>
  );
}
const K = ({ children }: { children: React.ReactNode }) => <b style={{ color: "var(--accent)" }}>{children}</b>;

function Chunk({ text, rank, score, tone, delay = 0 }: { text: string; rank?: number; score?: number; tone?: string; delay?: number }) {
  return (
    <div style={{ display: "flex", gap: 9, padding: "8px 10px", border: "1px solid var(--border)", borderLeft: `2px solid ${tone || "var(--border-strong)"}`, borderRadius: "var(--rs)", background: "var(--panel-2)", opacity: 0, animation: "popIn .4s ease forwards", animationDelay: `${delay}s` }}>
      {rank !== undefined && <span className="mono" style={{ fontSize: 10.5, color: "var(--faint)", flex: "none", width: 16 }}>#{rank}</span>}
      <span style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.5, minWidth: 0 }}>{text}</span>
      {score !== undefined && <span className="mono" style={{ fontSize: 10.5, color: tone || "var(--faint)", flex: "none", marginLeft: "auto" }}>{score.toFixed(3)}</span>}
    </div>
  );
}

function Note({ children }: { children: React.ReactNode }) {
  return <div className="note" style={{ marginTop: 8, fontSize: 11, lineHeight: 1.6 }}>{children}</div>;
}

/** ⏮ ‹ › ⏭ player for stepping through items one at a time, as the Index step does. */
function Inspect({ i, n, set, label }: { i: number; n: number; set: (v: number) => void; label: string }) {
  return (
    <div className="row" style={{ gap: 6, alignItems: "center", marginBottom: 10, flexWrap: "wrap" }}>
      <button className="btn ghost sm" onClick={() => set(0)} disabled={i <= 0} title="first">⏮</button>
      <button className="btn ghost sm" onClick={() => set(Math.max(0, i - 1))} disabled={i <= 0}>‹</button>
      <span className="mono" style={{ fontSize: 11, color: "var(--muted)", minWidth: 96, textAlign: "center" }}>{label} {i + 1} / {n}</span>
      <button className="btn ghost sm" onClick={() => set(Math.min(n - 1, i + 1))} disabled={i >= n - 1}>›</button>
      <button className="btn ghost sm" onClick={() => set(n - 1)} disabled={i >= n - 1} title="last">⏭</button>
    </div>
  );
}

/** Horizontal weight bar — the shape used across the platform's teaching panels. */
function Bar({ label, value, max, tone, right }: { label: string; value: number; max: number; tone: string; right?: string }) {
  return (
    <div className="row" style={{ gap: 9, alignItems: "center", fontSize: 11.5 }}>
      <span className="mono" style={{ width: 96, flex: "none", color: "var(--muted)", overflow: "hidden", textOverflow: "ellipsis" }}>{label}</span>
      <div style={{ flex: 1, height: 7, background: "var(--panel)", borderRadius: 4, overflow: "hidden", minWidth: 40 }}>
        <div style={{ width: `${Math.max(0, Math.min(100, (value / (max || 1)) * 100))}%`, height: "100%", background: tone, transition: "width .35s ease" }} />
      </div>
      <span className="mono" style={{ width: 58, flex: "none", textAlign: "right", color: tone }}>{right ?? value.toFixed(3)}</span>
    </div>
  );
}

function Empty({ children }: { children: React.ReactNode }) {
  return <div style={{ fontSize: 11.5, color: "var(--faint)", textAlign: "center", padding: "22px 0", border: "1px dashed var(--border)", borderRadius: "var(--rs)" }}>{children}</div>;
}

/**
 * The final stage of every architecture: send the retrieved context to the LLM
 * and render a grounded, cited answer. Citations map [chunk N] back to the exact
 * chunk that architecture retrieved, so you can check the answer against source.
 */
function AnswerPanel({ ctx, sources, note, llm }: {
  ctx: string[]; sources: string[]; note?: string;
  llm: { providers: { id: string; provider: string; label: string | null }[]; models: string[]; providerId: string; model: string; loading: boolean; setProviderId: (v: string) => void; setModel: (v: string) => void };
}) {
  const [answer, setAnswer] = useState("");
  const [busy, setBusy] = useState(false);
  const [meta, setMeta] = useState("");
  const [err, setErr] = useState("");
  const [openCite, setOpenCite] = useState<number | null>(null);

  // a new retrieval invalidates the previous answer
  const ctxKey = ctx.join("|");
  useEffect(() => { setAnswer(""); setMeta(""); setOpenCite(null); }, [ctxKey]);

  async function generate() {
    if (!ctx.length) return;
    setBusy(true); setAnswer(""); setErr(""); setMeta("generating…");
    const context = ctx.map((c, i) => `[chunk ${i + 1}] ${c}`).join("\n\n");
    const messages = [
      { role: "system", content: "You are a helpful assistant. Answer using ONLY the provided context. Cite inline like [chunk N] after each claim. If the answer is not in the context, say you don't know." },
      { role: "user", content: `Context:\n${context}\n\nQuestion: ${note || "Answer from the context."}` },
    ];
    const t0 = performance.now();
    try {
      const res = await fetch("/api/chat", {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ messages, temperature: 0.2, lab: "rag", ...(llm.providerId ? { providerId: llm.providerId } : {}), ...(llm.model ? { model: llm.model } : {}) }),
      });
      if (!res.ok || !res.body) {
        const j = await res.json().catch(() => ({ error: "request failed" }));
        setErr(explainProviderError(j.error || "request failed")); setMeta("error"); setBusy(false); return;
      }
      const reader = res.body.getReader(); const dec = new TextDecoder(); let text = "";
      for (;;) { const { done, value } = await reader.read(); if (done) break; text += dec.decode(value, { stream: true }); setAnswer(text); }
      setMeta(`${llm.model ? llm.model + " · " : ""}grounded · ${ctx.length} source(s) · ${Math.round(performance.now() - t0)}ms`);
    } catch (e) { setErr(explainProviderError((e as Error).message)); setMeta("error"); }
    setBusy(false);
  }

  // turn "[chunk 2]" in the answer into a clickable chip
  const cited = new Set<number>();
  const rendered = answer.replace(/\[chunk (\d+)\]/gi, (_m, n) => { cited.add(+n); return `\`[${n}]\``; });

  return (
    <>
      <div style={{ ...pnl, marginBottom: 16 }}>
        {head("var(--good)", "Context assembled for the LLM", <span className="note" style={{ fontSize: 10 }}>{ctx.length} chunk(s) · ~{Math.round(ctx.join(" ").length / 4)} tokens</span>)}
        <div style={{ padding: 15 }}>
          {ctx.length ? (
            <>
              <pre style={{ margin: 0, fontFamily: "var(--mono)", fontSize: 11, lineHeight: 1.65, color: "var(--muted)", whiteSpace: "pre-wrap", wordBreak: "break-word", border: "1px solid var(--border)", borderRadius: "var(--rs)", background: "var(--panel-2)", padding: "10px 12px", maxHeight: 220, overflow: "auto" }}>
                {ctx.map((c, i) => `[chunk ${i + 1}] ${c}`).join("\n\n")}
              </pre>
              <div className="row" style={{ marginTop: 12, gap: 10, alignItems: "center", flexWrap: "wrap" }}>
                <span className="mono" style={{ fontSize: 10, color: "var(--faint)", textTransform: "uppercase", letterSpacing: ".08em" }}>Model</span>
                <select aria-label="Provider" value={llm.providerId} disabled={busy || !llm.providers.length}
                  onChange={(e) => llm.setProviderId(e.target.value)}
                  style={{ background: "var(--panel-2)", color: "var(--text)", border: "1px solid var(--border-strong)", borderRadius: "var(--rs)", padding: "6px 9px", fontSize: 12, fontFamily: "inherit", maxWidth: 190 }}>
                  {!llm.providers.length && <option value="">No provider configured</option>}
                  {llm.providers.map((pr) => <option key={pr.id} value={pr.id}>{pr.label || pr.provider}</option>)}
                </select>
                <select aria-label="Model" value={llm.model} disabled={busy || !llm.models.length}
                  onChange={(e) => llm.setModel(e.target.value)}
                  style={{ background: "var(--panel-2)", color: "var(--text)", border: "1px solid var(--border-strong)", borderRadius: "var(--rs)", padding: "6px 9px", fontSize: 12, fontFamily: "var(--mono)", maxWidth: 230 }}>
                  {!llm.models.length && <option value="">{llm.loading ? "Loading…" : "No models"}</option>}
                  {llm.models.map((m) => <option key={m} value={m}>{m}</option>)}
                </select>
                <button className="btn" onClick={generate} disabled={busy || !llm.providers.length}>{busy ? <><span className="busy-dot" />generating…</> : "▶ Generate cited answer"}</button>
                {meta && meta !== "error" && <span className="note" style={{ fontSize: 10.5 }}>{meta}</span>}
              </div>
              {!llm.providers.length && !llm.loading && (
                <Note><span style={{ color: "var(--warn)" }}>⚠ No LLM provider is configured.</span> Add one under <b>Admin → Providers</b> — Groq, Cerebras and Gemini all have free tiers.</Note>
              )}
              {err && (
                <div style={{ marginTop: 10, padding: "10px 12px", border: "1px solid var(--crit)", borderRadius: "var(--rs)", background: "var(--panel-2)", fontSize: 11.5, color: "var(--crit)", lineHeight: 1.6 }}>
                  ⚠ {err}
                </div>
              )}
            </>
          ) : (
            <Empty><b style={{ color: "var(--crit)" }}>Nothing to send.</b><br />No context survived retrieval — generating here would mean inventing the answer.</Empty>
          )}
        </div>
      </div>

      {answer && (
        <div style={pnl}>
          {head("var(--accent)", "Answer", <span className="note" style={{ fontSize: 10 }}>{cited.size} of {ctx.length} chunk(s) cited</span>)}
          <div style={{ padding: 15 }}>
            <AgentOutput text={rendered} />
            <div style={{ marginTop: 14, borderTop: "1px solid var(--border)", paddingTop: 12 }}>
              <div className="flow-label">sources — click to read the chunk the answer cites</div>
              <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
                {ctx.map((c, i) => {
                  const used = cited.has(i + 1);
                  const open = openCite === i;
                  return (
                    <div key={i}>
                      <button onClick={() => setOpenCite(open ? null : i)}
                        style={{ width: "100%", textAlign: "left", display: "flex", gap: 9, alignItems: "center", padding: "7px 10px", cursor: "pointer", fontFamily: "inherit",
                          border: `1px solid ${used ? "var(--good)" : "var(--border)"}`, borderRadius: "var(--rs)", background: "var(--panel-2)", color: "var(--text)" }}>
                        <span className="mono" style={{ fontSize: 10.5, color: used ? "var(--good)" : "var(--faint)", flex: "none" }}>[{i + 1}]</span>
                        <span style={{ fontSize: 11.5, color: "var(--muted)", flex: 1, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: open ? "normal" : "nowrap" }}>
                          {sources[i] || c}
                        </span>
                        <span className="mono" style={{ fontSize: 9.5, color: used ? "var(--good)" : "var(--faint)", flex: "none" }}>{used ? "cited" : "unused"}</span>
                      </button>
                      {open && <pre style={{ margin: "5px 0 0", fontFamily: "var(--mono)", fontSize: 11, lineHeight: 1.6, color: "var(--muted)", whiteSpace: "pre-wrap", border: "1px solid var(--border)", borderRadius: "var(--rs)", background: "var(--surface)", padding: "9px 11px" }}>{c}</pre>}
                    </div>
                  );
                })}
              </div>
              {cited.size < ctx.length && (
                <Note><b>{ctx.length - cited.size} retrieved chunk(s) went uncited.</b> That is a retrieval-precision signal: the pipeline pulled context the model did not need.</Note>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}

type Ranked = { i: number; score: number }[];

export default function RagArchitecturesLab() {
  const [arch, setArch] = useState<ArchId>("hybrid");
  const [step, setStep] = useState(0);
  const [play, setPlay] = useState(0); // re-triggers the popIn animation on each run

  // ── sources: uploaded / pasted / sample, then chunked into the working corpus ──
  type Doc = { id: string; name: string; kind: string; text: string };
  const [docs, setDocs] = useState<Doc[]>([{ id: "sample", name: "sample-computing-notes.txt", kind: "txt", text: DEFAULT_CORPUS }]);
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState("");
  const [paste, setPaste] = useState("");
  const [openDoc, setOpenDoc] = useState<string | null>(null);

  // Which LLM answers at the final step. Lifted to lab level so switching
  // architecture keeps your choice.
  const [providers, setProviders] = useState<{ id: string; provider: string; label: string | null }[]>([]);
  const [models, setModels] = useState<string[]>([]);
  const [providerId, setProviderId] = useState("");
  const [model, setModel] = useState("");
  const [llmLoading, setLlmLoading] = useState(false);
  useEffect(() => {
    let cancelled = false;
    setLlmLoading(true);
    fetch(`/api/models${providerId ? `?providerId=${encodeURIComponent(providerId)}` : ""}`)
      .then((r) => r.json())
      .then((j) => {
        if (cancelled) return;
        setProviders(j.providers || []);
        setModels(j.models || []);
        if (!providerId && j.providers?.length) setProviderId(j.providerId || j.providers[0].id);
        setModel((cur) => (cur && (j.models || []).includes(cur) ? cur : j.default || (j.models || [])[0] || ""));
      })
      .catch(() => { if (!cancelled) setProviders([]); })
      .finally(() => { if (!cancelled) setLlmLoading(false); });
    return () => { cancelled = true; };
  }, [providerId]);
  const llm = { providers, models, providerId, model, loading: llmLoading, setProviderId, setModel };
  const [chunkStrategy, setChunkStrategy] = useState<ChunkStrategy>("sentence");
  const [size, setSize] = useState(40);
  const [overlap, setOverlap] = useState(0);
  const fileRef = useRef<HTMLInputElement | null>(null);
  // the corpus only exists once you press ▶ Build chunks — same contract as the Steps tab
  const [chunks, setChunks] = useState<string[]>([]);
  const [chunkSrc, setChunkSrc] = useState<string[]>([]);
  const idx = useMemo(() => buildIndex(chunks), [chunks]);
  const totalWords = docs.reduce((a, d) => a + d.text.split(/\s+/).filter(Boolean).length, 0);

  async function onFiles(e: React.ChangeEvent<HTMLInputElement>) {
    const files = e.target.files; if (!files?.length) return;
    setUploading(true); setUploadMsg("");
    for (const f of Array.from(files)) {
      const ext = (f.name.split(".").pop() || "txt").toLowerCase();
      try {
        if (["pdf", "docx", "doc", "xlsx", "xls", "xlsm"].includes(ext)) {
          const fd = new FormData(); fd.append("file", f);
          const r = await fetch("/api/rag/extract", { method: "POST", body: fd });
          const j = await r.json();
          if (!r.ok) throw new Error(j.error || "parse failed");
          setDocs((d) => [...d, { id: `${Date.now()}-${f.name}`, name: f.name, kind: ext, text: j.text }]);
        } else {
          setDocs((d0) => [...d0, { id: `${Date.now()}-${f.name}`, name: f.name, kind: ext, text: "" }]);
          const text = await f.text();
          setDocs((d0) => d0.map((x) => (x.name === f.name && !x.text ? { ...x, text } : x)));
        }
      } catch (err) { setUploadMsg(`"${f.name}" — ${(err as Error).message}`); }
    }
    setUploading(false); e.target.value = "";
    setChunks([]); setChunkSrc([]);
  }

  // ── committed stage outputs. Reactive recomputation is what made the previous
  // version a dashboard; here a stage only produces output when you run it. ──
  const [tokens, setTokens] = useState<string[] | null>(null);
  const [dense, setDense] = useState<Ranked | null>(null);
  const [sparse, setSparse] = useState<Ranked | null>(null);
  const [fused, setFused] = useState<{ i: number; score: number; ranks?: (number | null)[] }[] | null>(null);
  const [graph, setGraph] = useState<KnowledgeGraph | null>(null);
  const [sub, setSub] = useState<ReturnType<typeof retrieveGraph> | null>(null);
  const [routePlan, setRoutePlan] = useState<RouteSignal[] | null>(null);
  const [loop, setLoop] = useState<AgenticResult | null>(null);
  const [retrieved, setRetrieved] = useState<number[] | null>(null);
  const [grades, setGrades] = useState<DocGrade[] | null>(null);
  const [branch, setBranch] = useState<{ grade: Grade; reason: string; finalHits: number[]; rewritten?: string; detail: string[] } | null>(null);
  const [sources, setSources] = useState<ModalItem[] | null>(null);
  const [encoded, setEncoded] = useState<ModalItem[] | null>(null);
  const [mIndex, setMIndex] = useState<ModalIndex | null>(null);
  const [mHits, setMHits] = useState<{ hits: ModalHit[]; perModality: Record<Modality, number> } | null>(null);

  const resetFrom = (n: number) => {
    // invalidate every downstream stage so the UI can never show stale output
    if (arch === "hybrid") { if (n <= 0) { setTokens(null); } if (n <= 1) setDense(null); if (n <= 2) setSparse(null); if (n <= 3) setFused(null); }
    if (arch === "graph") { if (n <= 1) setGraph(null); if (n <= 3) setSub(null); }
    if (arch === "agentic") { if (n <= 1) setRoutePlan(null); if (n <= 2) setLoop(null); }
    if (arch === "crag") { if (n <= 1) setRetrieved(null); if (n <= 2) setGrades(null); if (n <= 3) setBranch(null); }
    if (arch === "multimodal") { if (n <= 0) setSources(null); if (n <= 1) setEncoded(null); if (n <= 2) setMIndex(null); if (n <= 3) setMHits(null); }
  };
  const ran = () => setPlay((p) => p + 1);

  useEffect(() => {
    setStep(0);
    setTokens(null); setDense(null); setSparse(null); setFused(null);
    setGraph(null); setSub(null);
    setRoutePlan(null); setLoop(null);
    setRetrieved(null); setGrades(null); setBranch(null);
    setSources(null); setEncoded(null); setMIndex(null); setMHits(null);
  }, [arch]);

  // ── controls ──
  const [query, setQuery] = useState("Analytical Engine computer");
  const [metric, setMetric] = useState<Metric>("cosine");
  const [k1, setK1] = useState(1.5);
  const [b, setB] = useState(0.75);
  const [fusionMode, setFusionMode] = useState<"rrf" | "alpha">("rrf");
  const [rrfK, setRrfK] = useState(60);
  const [alpha, setAlpha] = useState(0.5);
  const [topK, setTopK] = useState(4);
  const [maxNodes, setMaxNodes] = useState(40);
  const [hops, setHops] = useState(2);
  const [threshold, setThreshold] = useState(0.5);
  const [maxSteps, setMaxSteps] = useState(3);
  const [upper, setUpper] = useState(0.5);
  const [lower, setLower] = useState(0.2);
  const [modalities, setModalities] = useState<Modality[]>(["text", "table", "image"]);
  const [mStrategy, setMStrategy] = useState<"hybrid" | "vector" | "keyword">("hybrid");
  const [inspect, setInspect] = useState(0);

  useEffect(() => {
    // sensible default question per architecture
    setQuery(arch === "hybrid" ? "Analytical Engine computer"
      : arch === "graph" ? "Babbage"
      : arch === "agentic" ? "what does error E4021 mean"
      : arch === "crag" ? "photosynthesis in chloroplasts"
      : "Q4 2024 revenue and margin");
  }, [arch]);

  const steps = ARCH_STEPS[arch];
  const activeArch = ARCHES.find((a) => a.id === arch)!;

  // gating — you cannot skip ahead of work you have not done
  const done: Record<ArchId, boolean[]> = {
    hybrid: [!!tokens && chunks.length > 0, !!dense, !!sparse, !!fused, !!fused],
    graph: [!!tokens && chunks.length > 0, !!graph, !!graph, !!sub, !!sub],
    agentic: [!!tokens && chunks.length > 0, !!routePlan, !!loop, !!loop],
    crag: [!!tokens && chunks.length > 0, !!retrieved, !!grades, !!branch, !!branch],
    multimodal: [!!sources, !!encoded, !!mIndex, !!mHits, !!mHits],
  };
  const reached = (n: number) => n === 0 || done[arch].slice(0, n).every(Boolean);

  const corpusCard = (
    <>
      <Tiles items={[[String(docs.length), "documents"], [totalWords.toLocaleString(), "words"], [String(chunks.length), "chunks"]]} />
      <div style={{ ...pnl, marginBottom: 16 }}>
        {head("var(--accent)", "Knowledge source", <span className="note" style={{ fontSize: 10 }}>{uploading ? <><span className="busy-dot" />parsing…</> : `${docs.length} doc(s)`}</span>)}
        <div style={{ padding: 15 }}>
          <input ref={fileRef} type="file" multiple hidden onChange={onFiles}
            accept=".txt,.md,.csv,.json,.html,.pdf,.docx,.xlsx" />
          <div onClick={() => fileRef.current?.click()}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => { e.preventDefault(); const f = e.dataTransfer.files; if (f?.length) onFiles({ target: { files: f, value: "" } } as unknown as React.ChangeEvent<HTMLInputElement>); }}
            style={{ border: "1.5px dashed var(--border-strong)", borderRadius: 14, padding: "20px 18px", textAlign: "center", cursor: "pointer", background: "var(--panel-2)" }}>
            <div style={{ fontSize: 18, marginBottom: 4 }}>⬆</div>
            <b style={{ fontSize: 12.5 }}>Drop files here or click to upload</b>
            <div className="note" style={{ marginTop: 4, fontSize: 10 }}>parsed in-browser or on the server</div>
            <div className="row" style={{ justifyContent: "center", gap: 5, marginTop: 8, flexWrap: "wrap" }}>
              {["TXT", "MD", "CSV", "JSON", "HTML", "PDF", "DOCX", "XLSX"].map((t) => (
                <span key={t} className="mono" style={{ fontSize: 9, padding: "1px 6px", borderRadius: 4, border: "1px solid var(--border)", color: "var(--faint)" }}>{t}</span>
              ))}
            </div>
          </div>
          {uploadMsg && <Note><span style={{ color: "var(--crit)" }}>⚠ {uploadMsg}</span></Note>}

          <div style={{ marginTop: 12 }}>
            <label className="fld">Or paste text directly</label>
            <textarea value={paste} onChange={(e) => setPaste(e.target.value)} rows={3} placeholder="Paste notes, an article, a policy…"
              style={{ ...inputStyle, fontFamily: "var(--mono)", fontSize: 11.5, lineHeight: 1.6, resize: "vertical" }} />
            <div className="row" style={{ gap: 8, marginTop: 8 }}>
              <button className="btn ghost sm" disabled={!paste.trim()} onClick={() => {
                setDocs((d) => [...d, { id: `p${Date.now()}`, name: `pasted-${d.length + 1}.txt`, kind: "txt", text: paste.trim() }]);
                setPaste(""); setChunks([]); setChunkSrc([]); resetFrom(0);
              }}>+ Add pasted text</button>
              <button className="btn ghost sm" onClick={() => {
                setDocs([{ id: "sample", name: "sample-computing-notes.txt", kind: "txt", text: DEFAULT_CORPUS }]);
                setChunks([]); setChunkSrc([]); resetFrom(0);
              }}>↺ Reset to sample</button>
            </div>
          </div>

          {docs.length > 0 && (
            <div style={{ marginTop: 12, display: "flex", flexDirection: "column", gap: 5 }}>
              <div className="flow-label">loaded documents — preview the extracted text before chunking</div>
              {docs.map((d) => {
                const open = openDoc === d.id;
                const words = d.text.split(/\s+/).filter(Boolean).length;
                return (
                  <div key={d.id}>
                    <div className="row" style={{ gap: 9, alignItems: "center", padding: "7px 10px", border: `1px solid ${open ? "var(--accent)" : "var(--border)"}`, borderRadius: "var(--rs)", background: "var(--panel-2)" }}>
                      <span className="mono" style={{ fontSize: 9, padding: "1px 6px", borderRadius: 4, border: "1px solid var(--border)", color: "var(--faint)", flex: "none" }}>{d.kind.toUpperCase()}</span>
                      <span style={{ fontSize: 11.5, flex: 1, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{d.name}</span>
                      <span className="note" style={{ fontSize: 10, flex: "none" }}>{words}w · {d.text.length} chars</span>
                      <button className="btn ghost sm" onClick={() => setOpenDoc(open ? null : d.id)}>{open ? "Hide" : "Preview"}</button>
                      <button className="btn ghost sm" onClick={() => { setDocs((x) => x.filter((y) => y.id !== d.id)); setOpenDoc(null); setChunks([]); setChunkSrc([]); resetFrom(0); }}>✕</button>
                    </div>
                    {open && (
                      d.text.trim()
                        ? <pre style={{ margin: "5px 0 0", fontFamily: "var(--mono)", fontSize: 11, lineHeight: 1.65, color: "var(--muted)", whiteSpace: "pre-wrap", wordBreak: "break-word", border: "1px solid var(--border)", borderRadius: "var(--rs)", background: "var(--surface)", padding: "10px 12px", maxHeight: 260, overflow: "auto" }}>{d.text}</pre>
                        : <div style={{ margin: "5px 0 0", padding: "9px 12px", border: "1px solid var(--warn)", borderRadius: "var(--rs)", background: "var(--panel-2)", fontSize: 11.5, color: "var(--warn)" }}>
                            ⚠ No text was extracted — this file may be a scanned image or an unsupported layout. It will contribute no chunks.
                          </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      <div style={{ ...pnl, marginBottom: 16 }}>
        {head("var(--sky)", "Chunking", <span className="note" style={{ fontSize: 10 }}>{chunks.length} chunks</span>)}
        <div style={{ padding: 15 }}>
          <div className="row" style={{ gap: 20, alignItems: "flex-end", flexWrap: "wrap" }}>
            <div style={{ minWidth: 180 }}>
              <label className="fld">Strategy</label>
              <select value={chunkStrategy} onChange={(e) => { setChunkStrategy(e.target.value as ChunkStrategy); setChunks([]); resetFrom(0); }} style={{ width: 180 }}>
                {(["fixed", "sentence", "paragraph", "semantic"] as ChunkStrategy[]).map((s) => <option key={s} value={s}>{CHUNK_STRATEGY_LABEL[s]}</option>)}
              </select>
            </div>
            <Knob label="target size (words)" value={size} min={10} max={300} step={5} onChange={(v) => { setSize(v); setChunks([]); resetFrom(0); }} />
            <Knob label="overlap (words)" value={overlap} min={0} max={60} step={2} onChange={(v) => { setOverlap(Math.min(v, size - 1)); setChunks([]); resetFrom(0); }} />
            <button className="btn" disabled={!docs.length} onClick={() => {
              const out: string[] = []; const src: string[] = [];
              docs.forEach((d) => chunkBy(d.text, chunkStrategy, size, overlap).forEach((c) => { out.push(c); src.push(d.name); }));
              setChunks(out); setChunkSrc(src); resetFrom(0); ran();
            }}>▶ Build chunks</button>
          </div>
          <Note>Every architecture below retrieves over <b>these chunks</b>. Change the strategy and the whole pipeline changes with it.</Note>
          {chunks.length > 0 && <Cfg><K>{totalWords.toLocaleString()}</K> words → <K>{chunks.length}</K> chunks · {CHUNK_STRATEGY_LABEL[chunkStrategy]} · {overlap}w overlap</Cfg>}

          {chunks.length > 0 && (
            <div style={{ marginTop: 14, borderTop: "1px solid var(--border)", paddingTop: 12 }} key={`ch${play}`}>
              <div className="flow-label">chunks by document — what each file actually became</div>
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {docs.map((d) => {
                  const mine = chunks.map((c, i) => ({ c, i })).filter(({ i }) => chunkSrc[i] === d.name);
                  if (!mine.length) return null;
                  return (
                    <div key={d.id} style={{ border: "1px solid var(--border)", borderRadius: "var(--rs)", overflow: "hidden" }}>
                      <div className="row" style={{ gap: 8, alignItems: "center", padding: "7px 11px", background: "var(--surface)", borderBottom: "1px solid var(--border)" }}>
                        <span className="mono" style={{ fontSize: 9, padding: "1px 6px", borderRadius: 4, border: "1px solid var(--border)", color: "var(--faint)" }}>{d.kind.toUpperCase()}</span>
                        <b style={{ fontSize: 11.5 }}>{d.name}</b>
                        <span className="note" style={{ fontSize: 10, marginLeft: "auto" }}>{mine.length} chunk{mine.length === 1 ? "" : "s"}</span>
                      </div>
                      <div style={{ padding: 10, display: "flex", flexDirection: "column", gap: 5, maxHeight: 240, overflow: "auto" }}>
                        {mine.map(({ c, i }, n) => (
                          <div key={i} style={{ display: "flex", gap: 9, padding: "7px 10px", border: "1px solid var(--border)", borderLeft: "2px solid var(--sky)", borderRadius: "var(--rs)", background: "var(--panel-2)", opacity: 0, animation: "popIn .4s ease forwards", animationDelay: `${Math.min(n * 0.04, 1.2)}s` }}>
                            <span className="mono" style={{ fontSize: 10, color: "var(--faint)", flex: "none" }}>c{i + 1}</span>
                            <span style={{ fontSize: 11.5, color: "var(--muted)", lineHeight: 1.5, minWidth: 0 }}>{c}</span>
                            <span className="mono" style={{ fontSize: 9.5, color: "var(--faint)", flex: "none", marginLeft: "auto" }}>{c.split(/\s+/).filter(Boolean).length}w</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );

  const queryPanel = (runLabel: string, onRun: () => void, note: React.ReactNode) => (
    <>
      {corpusCard}
      <div style={pnl}>
        {head("var(--accent)", "Query")}
        <div style={{ padding: 15 }}>
          <div className="row" style={{ gap: 12, alignItems: "flex-end", flexWrap: "wrap" }}>
            <div style={{ flex: 1, minWidth: 240 }}>
              <label className="fld">Question to run through the pipeline</label>
              <input style={inputStyle} value={query} onChange={(e) => { setQuery(e.target.value); resetFrom(0); }} />
            </div>
            <button className="btn" onClick={onRun}>▶ {runLabel}</button>
          </div>
          <Note>{note}</Note>
        </div>
      </div>

      {/* every ▶ Run must produce visible output — this is what analysing the query yields */}
      {tokens && (
        <div style={{ ...pnl, marginTop: 16 }}>
          {head("var(--sky)", "Query terms", <span className="note" style={{ fontSize: 10 }}>{tokens.length} term(s) · {tokens.filter((t) => idx.df[t]).length} present in the corpus</span>)}
          <div style={{ padding: 15 }} key={`t${play}`}>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {tokens.map((t, i) => (
                <span key={i} className="mono" style={{ fontSize: 10.5, padding: "3px 9px", borderRadius: 999, border: `1px solid ${idx.df[t] ? "var(--border-strong)" : "var(--crit)"}`, color: idx.df[t] ? "var(--muted)" : "var(--crit)", background: "var(--panel-2)", opacity: 0, animation: "popIn .4s ease forwards", animationDelay: `${i * 0.05}s` }}
                  title={idx.df[t] ? `appears in ${idx.df[t]} of ${idx.N} chunks` : "appears in NO chunk"}>
                  {t} · df {idx.df[t] || 0}
                </span>
              ))}
            </div>
            <Cfg>
              terms in <span style={{ color: "var(--crit)" }}>red</span> appear in <K>0</K> chunks — they contribute nothing downstream
              {tokens.every((t) => !idx.df[t]) && <> · <span style={{ color: "var(--crit)" }}>no term is in the corpus at all, so retrieval cannot succeed</span></>}
            </Cfg>
          </div>
        </div>
      )}
    </>
  );

  // ── STEP BODIES ────────────────────────────────────────────────────────────
  function body(): React.ReactNode {
    /* ══════════════ 01 HYBRID ══════════════ */
    if (arch === "hybrid") {
      if (step === 0) return (
        <>
          {queryPanel("Tokenise query", () => { setTokens(tokenize(query)); resetFrom(1); ran(); },
            <>Hybrid RAG sends one query down <b>two independent retrievers</b>. This is the shared input to both.</>)}
        </>
      );

      if (step === 1) return (
        <>
          <Tiles items={[[String(chunks.length), "candidates"], [metric, "metric"], [dense ? String(dense.filter((d) => d.score > 0).length) : "—", "non-zero hits"]]} />
          <div style={{ ...pnl, marginBottom: 16 }}>
            {head("var(--sky)", "Dense path controls")}
            <div style={{ padding: 15 }}>
              <div className="row" style={{ gap: 20, alignItems: "flex-end", flexWrap: "wrap" }}>
                <div style={{ minWidth: 180 }}>
                  <label className="fld">Similarity metric</label>
                  <select value={metric} onChange={(e) => { setMetric(e.target.value as Metric); resetFrom(1); }} style={{ width: 180 }}>
                    <option value="cosine">cosine (angle)</option>
                    <option value="dot">dot product (IP)</option>
                    <option value="euclidean">euclidean (L2)</option>
                  </select>
                </div>
                <button className="btn" onClick={() => {
                  const qv = queryVector(idx, query);
                  setDense(idx.vectors.map((v, i) => ({ i, score: simSparse(qv, v, metric) })).sort((a, c) => c.score - a.score));
                  resetFrom(2); ran();
                }}>▶ Run dense retrieval</button>
              </div>
              <Note>Embeds the query into the same space as the chunks and scores <b>every</b> candidate — semantic similarity is continuous, so nothing is ever fully excluded.</Note>
              <Cfg>vector space: <K>TF-IDF</K> · {metric} over <K>{chunks.length}</K> chunk vectors · {Object.keys(idx.df).length} vocabulary terms</Cfg>
            </div>
          </div>
          {dense ? (
            <>
              <div style={{ ...pnl, marginBottom: 16 }}>
                {head("var(--sky)", "Dense results", <button className="btn ghost sm" onClick={ran}>↻ Replay</button>)}
                <div style={{ padding: 15, display: "flex", flexDirection: "column", gap: 6 }} key={`d${play}`}>
                  {dense.map((r, n) => (
                    <div key={r.i} onClick={() => setInspect(r.i)} style={{ cursor: "pointer" }} title="click to see how this score was produced">
                      <Chunk rank={n + 1} score={r.score} tone={inspect === r.i ? "var(--accent)" : "var(--sky)"} text={chunks[r.i]} delay={n * 0.06} />
                    </div>
                  ))}
                </div>
              </div>

              {/* how it works — the same "watch one chunk" depth as the Index step */}
              <div style={pnl}>
                {head("var(--accent)", "Watch the query meet one chunk", <span className="note" style={{ fontSize: 10 }}>click a result, or step through</span>)}
                <div style={{ padding: 15 }}>
                  <Inspect i={inspect} n={chunks.length} set={setInspect} label="chunk" />
                  {(() => {
                    const qv = queryVector(idx, query);
                    const cv = idx.vectors[inspect] || {};
                    const qTerms = Object.entries(qv).sort((a, c) => c[1] - a[1]);
                    const shared = qTerms.filter(([t]) => cv[t]);
                    const dot = shared.reduce((a, [t, w]) => a + w * (cv[t] || 0), 0);
                    const nq = Math.sqrt(Object.values(qv).reduce((a, w) => a + w * w, 0));
                    const nc = Math.sqrt(Object.values(cv).reduce((a, w) => a + w * w, 0));
                    const maxW = Math.max(...qTerms.map(([, w]) => w), ...Object.values(cv), 1e-9);
                    return (
                      <>
                        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                          <div>
                            <div className="flow-label">query vector — {qTerms.length} weighted terms</div>
                            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                              {qTerms.map(([t, w]) => <Bar key={t} label={t} value={w} max={maxW} tone={cv[t] ? "var(--good)" : "var(--faint)"} />)}
                            </div>
                          </div>
                          <div>
                            <div className="flow-label">chunk {inspect + 1} vector — top terms</div>
                            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                              {Object.entries(cv).sort((a, c) => c[1] - a[1]).slice(0, 8).map(([t, w]) => <Bar key={t} label={t} value={w} max={maxW} tone={qv[t] ? "var(--good)" : "var(--faint)"} />)}
                            </div>
                          </div>
                        </div>
                        <div style={{ marginTop: 14, borderTop: "1px solid var(--border)", paddingTop: 12 }}>
                          <div className="flow-label">shared terms — only these contribute to the score</div>
                          {shared.length ? (
                            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                              {shared.map(([t, w]) => (
                                <div key={t} className="mono" style={{ fontSize: 11, color: "var(--muted)" }}>
                                  <span style={{ color: "var(--good)" }}>{t}</span>: q<sub>w</sub> {w.toFixed(3)} × c<sub>w</sub> {(cv[t] || 0).toFixed(3)} = <b style={{ color: "var(--accent)" }}>{(w * (cv[t] || 0)).toFixed(4)}</b>
                                </div>
                              ))}
                            </div>
                          ) : (
                            <div style={{ padding: "9px 12px", border: "1px solid var(--crit)", borderRadius: "var(--rs)", background: "var(--panel-2)", fontSize: 11.5, color: "var(--crit)" }}>
                              No shared terms → dot product is 0 → <b>score 0.000</b>. The chunk may be about exactly this topic in different words; TF-IDF cannot see that.
                            </div>
                          )}
                          <Cfg>
                            dot = <K>{dot.toFixed(4)}</K> · ‖q‖ = <K>{nq.toFixed(3)}</K> · ‖c‖ = <K>{nc.toFixed(3)}</K> → cos = {dot.toFixed(4)} / ({nq.toFixed(3)} × {nc.toFixed(3)}) = <K>{(dot / ((nq * nc) || 1)).toFixed(4)}</K>
                          </Cfg>
                        </div>
                        <Note><b>Honest limit:</b> this vector is <b>TF-IDF</b> — still lexical. Try <span className="mono">electric cars</span> against the &ldquo;Automobiles powered by electricity&rdquo; chunk: <b>zero shared terms, score 0.000</b>. Real paraphrase matching needs neural embeddings (Steps tab → neural mode).</Note>
                      </>
                    );
                  })()}
                </div>
              </div>
            </>
          ) : <Empty>Run the dense retriever to see its ranking.</Empty>}
        </>
      );

      if (step === 2) return (
        <>
          <Tiles items={[[k1.toFixed(1), "k1 saturation"], [b.toFixed(2), "b length norm"], [sparse ? String(sparse.filter((s) => s.score > 0).length) : "—", "non-zero hits"]]} />
          <div style={{ ...pnl, marginBottom: 16 }}>
            {head("var(--warn)", "BM25 controls")}
            <div style={{ padding: 15 }}>
              <div className="row" style={{ gap: 20, alignItems: "flex-end", flexWrap: "wrap" }}>
                <Knob label="k1 — term-frequency saturation" value={k1} min={0.1} max={3} step={0.1} onChange={(v) => { setK1(v); resetFrom(2); }} fmt={(v) => v.toFixed(1)} />
                <Knob label="b — length normalisation" value={b} min={0} max={1} step={0.05} onChange={(v) => { setB(v); resetFrom(2); }} fmt={(v) => v.toFixed(2)} />
                <button className="btn" onClick={() => {
                  setSparse(bm25ScoresTuned(idx, query, k1, b).map((s, i) => ({ i, score: s })).sort((a, c) => c.score - a.score));
                  resetFrom(3); ran();
                }}>▶ Run BM25</button>
              </div>
              <Note>
                <b>k1</b> low = &ldquo;mentioned once is nearly as good as ten times&rdquo;; high = repetition keeps paying.
                <b> b</b> = 0 ignores document length; 1 fully penalises long chunks for having more chances to match.
              </Note>
              <Cfg>BM25(k1=<K>{k1.toFixed(1)}</K>, b=<K>{b.toFixed(2)}</K>) · avg chunk length <K>{idx.avgdl.toFixed(1)}</K> words · N=<K>{idx.N}</K></Cfg>
            </div>
          </div>
          {sparse ? (
            <>
              <div style={{ ...pnl, marginBottom: 16 }}>
                {head("var(--warn)", "Sparse results", <button className="btn ghost sm" onClick={ran}>↻ Replay</button>)}
                <div style={{ padding: 15, display: "flex", flexDirection: "column", gap: 6 }} key={`s${play}`}>
                  {sparse.map((r, n) => (
                    <div key={r.i} onClick={() => setInspect(r.i)} style={{ cursor: "pointer" }} title="click to break down this score">
                      <Chunk rank={n + 1} score={r.score} tone={inspect === r.i ? "var(--accent)" : "var(--warn)"} text={chunks[r.i]} delay={n * 0.06} />
                    </div>
                  ))}
                </div>
              </div>
              <div style={pnl}>
                {head("var(--accent)", `Score breakdown — chunk ${inspect + 1}`, <span className="note" style={{ fontSize: 10 }}>click any result above</span>)}
                <div style={{ padding: 15 }}>
                  {(() => {
                    const ex = bm25Explain(idx, query, inspect, k1, b);
                    return (
                      <>
                        <div style={{ display: "grid", gridTemplateColumns: "1fr auto auto auto auto", gap: "6px 14px", fontSize: 11.5, alignItems: "center" }}>
                          {["term", "tf", "df", "idf", "adds"].map((h) => <span key={h} className="mono" style={{ fontSize: 9.5, textTransform: "uppercase", color: "var(--faint)" }}>{h}</span>)}
                          {ex.terms.map((t) => (
                            <ContribRow key={t.term} t={t} />
                          ))}
                        </div>
                        <Cfg>chunk length <K>{ex.dl}</K> vs average <K>{ex.avgdl.toFixed(1)}</K> → BM25 total <K>{ex.total.toFixed(3)}</K></Cfg>
                        <Note>Terms with <b>tf 0</b> contribute nothing — BM25 only ranks chunks sharing a literal term. That is its strength (exact codes like <span className="mono">E4021</span>) and its limit (no paraphrase).</Note>
                      </>
                    );
                  })()}
                </div>
              </div>
            </>
          ) : <Empty>Run BM25 to see its ranking.</Empty>}
        </>
      );

      if (step === 3) return (
        <>
          <Tiles items={[[fusionMode === "rrf" ? "RRF" : "alpha", "method"], [fusionMode === "rrf" ? String(rrfK) : alpha.toFixed(2), fusionMode === "rrf" ? "k constant" : "alpha"], [String(topK), "top-k"]]} />
          <div style={{ ...pnl, marginBottom: 16 }}>
            {head("var(--good)", "Fusion controls")}
            <div style={{ padding: 15 }}>
              <div className="row" style={{ gap: 20, alignItems: "flex-end", flexWrap: "wrap" }}>
                <div style={{ minWidth: 200 }}>
                  <label className="fld">Fusion method</label>
                  <select value={fusionMode} onChange={(e) => { setFusionMode(e.target.value as "rrf" | "alpha"); resetFrom(3); }} style={{ width: 200 }}>
                    <option value="rrf">Reciprocal Rank Fusion</option>
                    <option value="alpha">Weighted score blend</option>
                  </select>
                </div>
                {fusionMode === "rrf"
                  ? <Knob label="k — rank damping" value={rrfK} min={1} max={120} step={1} onChange={(v) => { setRrfK(v); resetFrom(3); }} />
                  : <Knob label="alpha — 0 keyword ⟷ 1 vector" value={alpha} min={0} max={1} step={0.05} onChange={(v) => { setAlpha(v); resetFrom(3); }} fmt={(v) => v.toFixed(2)} />}
                <Knob label="top-k" value={topK} min={1} max={Math.max(2, chunks.length)} step={1} onChange={(v) => { setTopK(v); resetFrom(3); }} />
                <button className="btn" disabled={!dense || !sparse} onClick={() => {
                  if (!dense || !sparse) return;
                  if (fusionMode === "rrf") setFused(rrfFuse([sparse.filter((r) => r.score > 0), dense.filter((r) => r.score > 0)], rrfK, topK));
                  else {
                    const mx = (a: Ranked) => Math.max(...a.map((r) => r.score), 1e-9);
                    const bm = new Map(sparse.map((r) => [r.i, r.score / mx(sparse)]));
                    const dv = new Map(dense.map((r) => [r.i, r.score / mx(dense)]));
                    setFused(chunks.map((_, i) => ({ i, score: (1 - alpha) * (bm.get(i) || 0) + alpha * (dv.get(i) || 0) })).sort((a, c) => c.score - a.score).slice(0, topK));
                  }
                  ran();
                }}>▶ Fuse rankings</button>
              </div>
              <Note>
                {fusionMode === "rrf"
                  ? <>BM25 is unbounded, cosine is [-1,1] — they cannot be added directly. RRF discards the scores and sums <span className="mono">1/(k + rank)</span>, so a chunk ranking well on <b>both</b> lists beats one topping only a single list. Lower <b>k</b> makes the top ranks dominate.</>
                  : <>Normalises both score arrays to 0–1 and blends them. Works, but the right <b>alpha</b> shifts as the corpus changes — which is exactly what RRF avoids.</>}
              </Note>
              {!dense || !sparse ? <Cfg><span style={{ color: "var(--crit)" }}>run both retrievers first — fusion needs two lists</span></Cfg> : <Cfg>fusing <K>{sparse.filter((r) => r.score > 0).length}</K> sparse + <K>{dense.filter((r) => r.score > 0).length}</K> dense results → top <K>{topK}</K></Cfg>}
            </div>
          </div>
          {fused ? (
            <div style={pnl}>
              {head("var(--good)", "Fused ranking", <button className="btn ghost sm" onClick={ran}>↻ Replay</button>)}
              <div style={{ padding: 15, display: "flex", flexDirection: "column", gap: 8 }} key={`f${play}`}>
                {fused.map((r, n) => (
                  <div key={r.i} style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                    <Chunk rank={n + 1} score={r.score} tone="var(--good)" text={chunks[r.i]} delay={n * 0.06} />
                    <span className="mono" style={{ fontSize: 9.5, color: "var(--faint)", paddingLeft: 25 }}>
                      {r.ranks
                        ? <>bm25 #{r.ranks[0] ?? "—"} {r.ranks[0] ? `→ 1/(${rrfK}+${r.ranks[0]})` : ""} · vector #{r.ranks[1] ?? "—"} {r.ranks[1] ? `→ 1/(${rrfK}+${r.ranks[1]})` : ""} = {r.score.toFixed(5)}</>
                        : <>({(1 - alpha).toFixed(2)} × bm25norm) + ({alpha.toFixed(2)} × vecnorm) = {r.score.toFixed(4)}</>}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          ) : <Empty>Fuse the two rankings to continue.</Empty>}
        </>
      );

      return <AnswerPanel llm={llm} ctx={(fused || []).map((r) => chunks[r.i])} sources={(fused || []).map((r) => chunkSrc[r.i] || "corpus")} note={query} />;
    }

    /* ══════════════ 02 GRAPHRAG ══════════════ */
    if (arch === "graph") {
      if (step === 0) return queryPanel("Analyse query", () => { setTokens(tokenize(query)); resetFrom(1); ran(); },
        <>GraphRAG answers questions whose answer is a <b>relationship spanning documents</b> — something no single chunk contains. Name an entity here; it is matched against the graph in step 4, once the graph exists.</>);

      if (step === 1) return (
        <>
          <Tiles items={[[String(chunks.length), "chunks scanned"], [graph ? String(graph.nodes.length) : "—", "entities"], [String(maxNodes), "node cap"]]} />
          <div style={{ ...pnl, marginBottom: 16 }}>
            {head("var(--accent)", "Entity extraction controls")}
            <div style={{ padding: 15 }}>
              <div className="row" style={{ gap: 20, alignItems: "flex-end", flexWrap: "wrap" }}>
                <Knob label="max entities to keep" value={maxNodes} min={5} max={80} step={5} onChange={(v) => { setMaxNodes(v); resetFrom(1); }} />
                <button className="btn" onClick={() => { setGraph(extractGraph(chunks, { maxNodes })); resetFrom(3); ran(); }}>▶ Extract entities</button>
              </div>
              <Note>Scans every chunk for proper nouns and repeated concepts. These become the graph&apos;s <b>nodes</b> — the anchors a query attaches to.</Note>
              <Cfg>scanning <K>{chunks.length}</K> chunks · keeping the <K>{maxNodes}</K> most frequent entities</Cfg>
            </div>
          </div>
          {graph ? (
            <div style={pnl}>
              {head("var(--accent)", "Entities found", <button className="btn ghost sm" onClick={ran}>↻ Replay</button>)}
              <div style={{ padding: 15 }} key={`e${play}`}>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                  {graph.nodes.map((n, i) => (
                    <span key={n.id} className="mono" style={{ fontSize: 10.5, padding: "3px 9px", borderRadius: 999, border: `1px solid ${n.type === "proper" ? "var(--accent)" : "var(--border-strong)"}`, color: n.type === "proper" ? "var(--accent-strong)" : "var(--muted)", background: "var(--panel-2)", opacity: 0, animation: "popIn .4s ease forwards", animationDelay: `${i * 0.03}s` }}
                      title={`${n.type} · appears in ${n.chunks.length} chunk(s) · freq ${n.freq}`}>
                      {n.label}
                    </span>
                  ))}
                </div>
                <Cfg>blue = proper nouns (people, places, systems) · grey = concepts · <K>{graph.nodes.filter((n) => n.type === "proper").length}</K> proper of <K>{graph.nodes.length}</K></Cfg>
              </div>
            </div>
          ) : <Empty>Extract entities to continue.</Empty>}
        </>
      );

      if (step === 2) return (
        <>
          <Tiles items={[[graph ? String(graph.nodes.length) : "—", "nodes"], [graph ? String(graph.edges.length) : "—", "relations"], [graph && graph.nodes.length ? (graph.edges.length / graph.nodes.length).toFixed(1) : "—", "edges / node"]]} />
          {graph && graph.nodes.length > 0 && (
            <div style={{ ...pnl, marginBottom: 16 }}>
              {head("var(--accent)", "The graph, drawn", <span className="note" style={{ fontSize: 10 }}>nodes = entities · edges = relations</span>)}
              <div style={{ padding: 15 }}>
                <KnowledgeGraphPlot graph={graph} height={460} />
                <Note>
                  <b>Scroll to zoom, drag to pan.</b> Hover a node for its type, how often it appears and which chunks it came from;
                  hover an edge for the relation and its source chunk. Circle size = entity frequency, line thickness = relation strength.
                </Note>
              </div>
            </div>
          )}
          {graph ? (
            <div style={pnl}>
              {head("var(--sky)", "Knowledge graph — every relation and where it came from", <button className="btn ghost sm" onClick={ran}>↻ Replay</button>)}
              <div style={{ padding: 15, display: "flex", flexDirection: "column", gap: 5 }} key={`g${play}`}>
                {graph.edges.slice(0, 18).map((e, n) => (
                  <div key={n} className="mono" style={{ fontSize: 11, color: "var(--muted)", padding: "6px 10px", border: "1px solid var(--border)", borderRadius: "var(--rs)", background: "var(--panel-2)", opacity: 0, animation: "popIn .4s ease forwards", animationDelay: `${n * 0.04}s` }}>
                    {graph.nodes.find((x) => x.id === e.s)?.label || e.s}
                    <span style={{ color: "var(--accent-strong)" }}> —{e.rel}→ </span>
                    {graph.nodes.find((x) => x.id === e.o)?.label || e.o}
                    <span style={{ color: "var(--faint)" }}> · chunk {e.chunks.map((c) => c + 1).join(", ")}</span>
                  </div>
                ))}
                <Note>Each edge remembers <b>which chunk produced it</b> — that back-pointer is how a graph hop turns back into retrievable text in the next step.</Note>
              </div>
            </div>
          ) : <Empty>Extract entities first.</Empty>}
        </>
      );

      if (step === 3) return (
        <>
          <Tiles items={[[String(hops), "hops"], [sub ? String(sub.nodes.length) : "—", "nodes reached"], [sub ? String(sub.chunkIds.length) : "—", "chunks pulled"]]} />
          <div style={{ ...pnl, marginBottom: 16 }}>
            {head("var(--good)", "Traversal controls")}
            <div style={{ padding: 15 }}>
              <div className="row" style={{ gap: 20, alignItems: "flex-end", flexWrap: "wrap" }}>
                <Knob label="hops from the seed" value={hops} min={1} max={3} step={1} onChange={(v) => { setHops(v); resetFrom(3); }} />
                <Knob label="top-k chunks" value={topK} min={1} max={Math.max(2, chunks.length)} step={1} onChange={(v) => { setTopK(v); resetFrom(3); }} />
                <button className="btn" disabled={!graph} onClick={() => { if (graph) { setSub(retrieveGraph(graph, query, topK, hops)); ran(); } }}>▶ Traverse graph</button>
              </div>
              <Note>Breadth-first from the entities matching your query. Each hop pulls in neighbours — and their chunks.</Note>
              <Cfg>seed = <K>{query}</K> · expanding <K>{hops}</K> hop(s) · returning top <K>{topK}</K> chunks</Cfg>
            </div>
          </div>
          {sub && graph && graph.nodes.length > 0 && (
            <div style={{ ...pnl, marginBottom: 16 }}>
              {head("var(--good)", "The subgraph, highlighted against the whole graph", <span className="note" style={{ fontSize: 10 }}>{sub.nodes.length} of {graph.nodes.length} nodes reached</span>)}
              <div style={{ padding: 15 }}>
                <KnowledgeGraphPlot graph={graph} height={440} highlight={new Set(sub.nodes)} seeds={sub.seeds} layers={sub.layers} />
                <Note>
                  Dimmed nodes were <b>not reached</b> — the traversal never got to them within {hops} hop(s).
                  Amber-ringed nodes are <b>seeds</b> (matched your query directly); hover any lit node to see how many hops away it was found.
                </Note>
              </div>
            </div>
          )}
          {sub ? (
            <div style={pnl}>
              {head("var(--good)", "Subgraph reached", <button className="btn ghost sm" onClick={ran}>↻ Replay</button>)}
              <div style={{ padding: 15, display: "flex", flexDirection: "column", gap: 10 }} key={`sg${play}`}>
                <div>
                  <div className="flow-label">nodes by hop distance</div>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                    {sub.nodes.length ? sub.nodes.map((id, i) => {
                      const seed = sub.seeds.includes(id);
                      const layer = sub.layers[id] ?? 0;
                      return (
                        <span key={id} className="mono" style={{ fontSize: 10.5, padding: "3px 9px", borderRadius: 999, border: `1px solid ${seed ? "var(--accent)" : "var(--border-strong)"}`, color: seed ? "var(--accent-strong)" : "var(--muted)", background: "var(--panel-2)", opacity: 0, animation: "popIn .4s ease forwards", animationDelay: `${i * 0.05}s` }}
                          title={seed ? "matched the query directly" : `reached in ${layer} hop(s)`}>
                          {graph?.nodes.find((n) => n.id === id)?.label || id}{!seed && ` ·${layer}`}
                        </span>
                      );
                    }) : <span className="note">no entity matches that query</span>}
                  </div>
                </div>
                <div>
                  <div className="flow-label">chunks reached through the graph</div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                    {sub.chunkIds.length ? sub.chunkIds.map((i, n) => <Chunk key={i} rank={n + 1} tone="var(--good)" text={chunks[i]} delay={n * 0.06} />) : <span className="note">nothing reached</span>}
                  </div>
                </div>
                <Note>Raise hops to 2 and chunks arrive that share <b>no words</b> with your query — reached purely through relationships. That is the class of question chunk similarity cannot answer.</Note>
              </div>
            </div>
          ) : <Empty>Traverse the graph to continue.</Empty>}
        </>
      );

      return (
        <>
          <AnswerPanel llm={llm} ctx={(sub?.chunkIds || []).map((i) => chunks[i])} sources={(sub?.chunkIds || []).map((i) => chunkSrc[i] || "corpus")} note={query} />
          <Note><b>Not implemented:</b> community detection and global summarisation (Leiden clustering + per-cluster summaries). This is local subgraph search only, so &ldquo;what are the main themes?&rdquo; is out of reach.</Note>
        </>
      );
    }

    /* ══════════════ 03 AGENTIC ══════════════ */
    if (arch === "agentic") {
      if (step === 0) return queryPanel("Analyse query", () => { setTokens(tokenize(query)); resetFrom(1); ran(); },
        <><b>Try:</b> <span className="mono">what does error E4021 mean</span> (literal → BM25) · <span className="mono">who worked with Babbage</span> (relationship → graph) · <span className="mono">tell me about early computing</span> (open → vectors).</>);

      if (step === 1) return (
        <>
          <Tiles items={[[routePlan ? routePlan[0].retriever : "—", "first choice"], [routePlan ? routePlan[0].strength.toFixed(2) : "—", "confidence"], [String(routePlan?.length ?? 0), "options ranked"]]} />
          <div style={{ ...pnl, marginBottom: 16 }}>
            {head("var(--accent)", "Planner controls")}
            <div style={{ padding: 15 }}>
              <button className="btn" onClick={() => { setRoutePlan(planRetrievers(query, { hasGraph: true, entityLabels: extractGraph(chunks).nodes.map((n) => n.label) })); resetFrom(2); ran(); }}>▶ Build retrieval plan</button>
              <Note>The planner reads the <b>shape</b> of the question, not its topic: exact literals favour BM25, relationship phrasing favours the graph, everything else favours vectors.</Note>
              {/* how it works — show the signals firing on the actual query text */}
              <div style={{ marginTop: 14, borderTop: "1px solid var(--border)", paddingTop: 12 }}>
                <div className="flow-label">signals detected in your question</div>
                {(() => {
                  const g2 = extractGraph(chunks);
                  const sig = detectSignals(query, g2.nodes.map((n) => n.label));
                  const rows: { label: string; hit: string | null; tone: string; yes: string; no: string }[] = [
                    { label: "names a graph entity", hit: sig.named, tone: "var(--good)",
                      yes: "→ graph · its neighbours are one hop away", no: "→ the graph has no anchor to start from" },
                    { label: "relationship phrasing", hit: sig.relational, tone: "var(--accent)",
                      yes: "→ graph · the answer is likely an edge, not a chunk", no: "→ no edge-shaped question detected" },
                    { label: "exact literal", hit: sig.exact, tone: "var(--warn)",
                      yes: "→ keyword · embeddings blur rare literals, BM25 does not", no: "→ nothing for keyword search to lock onto" },
                  ];
                  return (
                    <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
                      {rows.map((r) => (
                        <div key={r.label} style={{ display: "grid", gridTemplateColumns: "18px 150px minmax(90px, auto) 1fr", gap: 10, alignItems: "center", padding: "7px 11px", border: `1px solid ${r.hit ? r.tone : "var(--border)"}`, borderRadius: "var(--rs)", background: "var(--panel-2)" }}>
                          <span className="mono" style={{ fontSize: 12, color: r.hit ? r.tone : "var(--faint)", textAlign: "center" }}>{r.hit ? "✓" : "·"}</span>
                          <span className="mono" style={{ fontSize: 10.5, color: r.hit ? "var(--text)" : "var(--faint)" }}>{r.label}</span>
                          {r.hit
                            ? <span className="mono" style={{ fontSize: 10.5, color: r.tone, background: "var(--surface)", padding: "2px 8px", borderRadius: 4, justifySelf: "start", whiteSpace: "nowrap" }}>&ldquo;{r.hit}&rdquo;</span>
                            : <span className="mono" style={{ fontSize: 10.5, color: "var(--faint)" }}>not found</span>}
                          <span style={{ fontSize: 10.5, color: r.hit ? "var(--muted)" : "var(--faint)", textAlign: "right" }}>{r.hit ? r.yes : r.no}</span>
                        </div>
                      ))}
                      {!sig.named && !sig.exact && !sig.relational && (
                        <Note>No signal fired, so the planner falls back to <b>vector search</b> — the safe default for an open-ended question.</Note>
                      )}
                    </div>
                  );
                })()}
              </div>
            </div>
          </div>
          {routePlan ? (
            <div style={pnl}>
              {head("var(--accent)", "Plan", <button className="btn ghost sm" onClick={ran}>↻ Replay</button>)}
              <div style={{ padding: 15, display: "flex", flexDirection: "column", gap: 6 }} key={`p${play}`}>
                {routePlan.map((p, i) => (
                  <div key={p.retriever} style={{ display: "flex", gap: 10, alignItems: "center", padding: "8px 11px", border: "1px solid var(--border)", borderRadius: "var(--rs)", background: "var(--panel-2)", opacity: 0, animation: "popIn .4s ease forwards", animationDelay: `${i * 0.1}s` }}>
                    <span className="mono" style={{ fontSize: 11, color: "var(--accent-strong)", width: 66, flex: "none" }}>{p.retriever}</span>
                    <div style={{ flex: 1, height: 4, background: "var(--panel)", borderRadius: 2, overflow: "hidden" }}>
                      <div style={{ width: `${p.strength * 100}%`, height: "100%", background: "var(--accent)" }} />
                    </div>
                    <span style={{ fontSize: 11, color: "var(--muted)", flex: "2 1 0", minWidth: 0 }}>{p.why}</span>
                  </div>
                ))}
              </div>
            </div>
          ) : <Empty>Build the plan to continue.</Empty>}
        </>
      );

      if (step === 2) return (
        <>
          <Tiles items={[[loop ? String(loop.steps.length) : "—", "attempts"], [threshold.toFixed(2), "confidence bar"], [loop ? (loop.confident ? "yes" : "no") : "—", "cleared bar"]]} />
          <div style={{ ...pnl, marginBottom: 16 }}>
            {head("var(--good)", "Loop controls")}
            <div style={{ padding: 15 }}>
              <div className="row" style={{ gap: 20, alignItems: "flex-end", flexWrap: "wrap" }}>
                <Knob label="confidence bar to accept" value={threshold} min={0.1} max={0.9} step={0.05} onChange={(v) => { setThreshold(v); resetFrom(2); }} fmt={(v) => v.toFixed(2)} />
                <Knob label="max attempts" value={maxSteps} min={1} max={3} step={1} onChange={(v) => { setMaxSteps(v); resetFrom(2); }} />
                <Knob label="top-k per attempt" value={topK} min={1} max={Math.max(2, chunks.length)} step={1} onChange={(v) => { setTopK(v); resetFrom(2); }} />
                <button className="btn" disabled={!routePlan} onClick={() => {
                  setLoop(runAgenticRetrieval(idx, query, topK, { graph: extractGraph(chunks), threshold, maxSteps, hops }));
                  ran();
                }}>▶ Run the loop</button>
              </div>
              <Note>Each attempt is <b>graded</b> before it is accepted. Raise the bar to 0.9 and watch the agent exhaust every retriever and report failure instead of returning weak chunks silently.</Note>
              <Cfg>accept at ≥ <K>{threshold.toFixed(2)}</K> · up to <K>{maxSteps}</K> attempt(s) · top-<K>{topK}</K> each</Cfg>
            </div>
          </div>
          {loop ? (
            <div style={pnl}>
              {head("var(--good)", "Loop trace", <button className="btn ghost sm" onClick={ran}>↻ Replay</button>)}
              <div style={{ padding: 15, display: "flex", flexDirection: "column", gap: 8 }} key={`l${play}`}>
                {loop.steps.map((s, i) => {
                  const ok = s.verdict === "accepted";
                  return (
                    <div key={s.n} style={{ border: `1px solid ${ok ? "var(--good)" : "var(--border)"}`, borderRadius: "var(--r)", padding: "10px 12px", background: "var(--panel-2)", display: "flex", flexDirection: "column", gap: 6, opacity: 0, animation: "popIn .4s ease forwards", animationDelay: `${i * 0.25}s` }}>
                      <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                        <span className="mono" style={{ fontSize: 10.5, color: "var(--faint)" }}>ATTEMPT {s.n}</span>
                        <b style={{ fontSize: 12.5 }}>{s.retriever}</b>
                        <span className="mono" style={{ fontSize: 10.5, color: ok ? "var(--good)" : "var(--warn)" }}>{ok ? "✓ accepted" : "✕ rejected"} · best {s.best.toFixed(2)} vs bar {threshold.toFixed(2)}</span>
                      </div>
                      <span style={{ fontSize: 11.5, color: "var(--muted)" }}>{s.note}</span>
                      {s.hits.slice(0, 2).map((i2, n) => <Chunk key={i2} rank={n + 1} tone={ok ? "var(--good)" : "var(--border-strong)"} text={chunks[i2]} />)}
                    </div>
                  );
                })}
              </div>
            </div>
          ) : <Empty>Run the loop to continue.</Empty>}
        </>
      );

      return (
        <>
          <AnswerPanel llm={llm} ctx={(loop?.finalHits || []).map((i) => chunks[i])} sources={(loop?.finalHits || []).map((i) => chunkSrc[i] || "corpus")} note={query} />
          <Note>
            {loop?.confident
              ? <><b>Answered by {loop.answeredBy}.</b> Evidence cleared the bar on attempt {loop.steps.length}, so no further retrieval was spent.</>
              : <><b>No retriever cleared the bar.</b> Plain RAG would have returned these same weak chunks with no signal they were weak.</>}
          </Note>
        </>
      );
    }

    /* ══════════════ 04 CRAG ══════════════ */
    if (arch === "crag") {
      if (step === 0) return queryPanel("Accept query", () => { setTokens(tokenize(query)); resetFrom(1); ran(); },
        <>The default query is <b>deliberately absent</b> from the corpus. Plain RAG would still answer it. Work through the steps and watch CRAG refuse.</>);

      if (step === 1) return (
        <>
          <Tiles items={[[String(topK), "top-k"], [retrieved ? String(retrieved.length) : "—", "docs retrieved"], [String(chunks.length), "candidates"]]} />
          <div style={{ ...pnl, marginBottom: 16 }}>
            {head("var(--accent)", "Retriever controls")}
            <div style={{ padding: 15 }}>
              <div className="row" style={{ gap: 20, alignItems: "flex-end", flexWrap: "wrap" }}>
                <Knob label="top-k to retrieve" value={topK} min={1} max={Math.max(2, chunks.length)} step={1} onChange={(v) => { setTopK(v); resetFrom(1); }} />
                <button className="btn" onClick={() => {
                  const mx = (a: number[]) => Math.max(...a, 1e-9);
                  const bm = bm25ScoresTuned(idx, query, k1, b); const qv = queryVector(idx, query);
                  const vec = idx.vectors.map((v) => simSparse(qv, v, "cosine"));
                  const s = bm.map((x, i) => ({ i, score: 0.5 * (x / mx(bm)) + 0.5 * (vec[i] / mx(vec)) }));
                  setRetrieved(s.sort((a, c) => c.score - a.score).slice(0, topK).map((r) => r.i));
                  resetFrom(2); ran();
                }}>▶ Retrieve</button>
              </div>
              <Note>Plain hybrid retrieval — <b>no judgement yet</b>. Top-k always returns something, however unrelated.</Note>
            </div>
          </div>
          {retrieved ? (
            <div style={pnl}>
              {head("var(--accent)", "Retrieved docs — before any judgement", <button className="btn ghost sm" onClick={ran}>↻ Replay</button>)}
              <div style={{ padding: 15, display: "flex", flexDirection: "column", gap: 6 }} key={`r${play}`}>
                {retrieved.map((i, n) => <Chunk key={i} rank={n + 1} tone="var(--border-strong)" text={chunks[i]} delay={n * 0.06} />)}
                <Note>Notice these may have nothing to do with the query. <b>This is the failure CRAG exists to catch</b> — plain RAG would hand exactly these to the LLM.</Note>
              </div>
            </div>
          ) : <Empty>Retrieve to continue.</Empty>}
        </>
      );

      if (step === 2) return (
        <>
          <Tiles items={[[grades ? grades.filter((g) => g.verdict === "relevant").length + "" : "—", "relevant"], [grades ? grades.filter((g) => g.verdict === "partial").length + "" : "—", "partial"], [grades ? grades.filter((g) => g.verdict === "irrelevant").length + "" : "—", "irrelevant"]]} />
          <div style={{ ...pnl, marginBottom: 16 }}>
            {head("var(--warn)", "Evaluator controls")}
            <div style={{ padding: 15 }}>
              <div className="row" style={{ gap: 20, alignItems: "flex-end", flexWrap: "wrap" }}>
                <Knob label="CORRECT at score ≥" value={upper} min={0.3} max={0.9} step={0.05} onChange={(v) => { setUpper(v); resetFrom(2); }} fmt={(v) => v.toFixed(2)} />
                <Knob label="INCORRECT below" value={lower} min={0.05} max={0.4} step={0.05} onChange={(v) => { setLower(v); resetFrom(2); }} fmt={(v) => v.toFixed(2)} />
                <button className="btn" disabled={!retrieved} onClick={() => { if (retrieved) { setGrades(gradeDocs(idx, query, retrieved)); resetFrom(3); ran(); } }}>▶ Grade every doc</button>
              </div>
              <Note><b>Coverage</b> = share of query terms the chunk actually contains. <b>Similarity</b> = vector closeness. Coverage is weighted higher, because a chunk missing most of the question is not evidence however close its embedding sits.</Note>
              <Cfg>score = <K>0.5</K>×coverage + <K>0.3</K>×similarity + <K>0.2</K>×bm25 · relevant ≥ 0.5 · irrelevant &lt; 0.2</Cfg>
            </div>
          </div>
          {grades && (
            <div style={{ ...pnl, marginBottom: 16 }}>
              {head("var(--accent)", "How coverage is measured", <span className="note" style={{ fontSize: 10 }}>every query term × every retrieved chunk</span>)}
              <div style={{ padding: 15, overflowX: "auto" }}>
                {(() => {
                  const qTerms = [...new Set(tokenize(query))];
                  return (
                    <table style={{ borderCollapse: "collapse", fontSize: 11 }}>
                      <thead>
                        <tr>
                          <th style={{ textAlign: "left", padding: "4px 10px", color: "var(--faint)", fontWeight: 500, fontFamily: "var(--mono)", fontSize: 9.5, textTransform: "uppercase" }}>chunk</th>
                          {qTerms.map((t) => <th key={t} style={{ padding: "4px 8px", color: "var(--faint)", fontWeight: 500, fontFamily: "var(--mono)", fontSize: 10 }}>{t}</th>)}
                          <th style={{ padding: "4px 10px", color: "var(--faint)", fontWeight: 500, fontFamily: "var(--mono)", fontSize: 9.5, textTransform: "uppercase" }}>coverage</th>
                        </tr>
                      </thead>
                      <tbody>
                        {grades.map((g) => {
                          const docTerms = new Set(idx.docs[g.i] || []);
                          return (
                            <tr key={g.i}>
                              <td style={{ padding: "4px 10px", fontFamily: "var(--mono)", color: "var(--muted)" }}>c{g.i + 1}</td>
                              {qTerms.map((t) => {
                                const hit = docTerms.has(t);
                                return <td key={t} style={{ padding: "4px 8px", textAlign: "center", color: hit ? "var(--good)" : "var(--crit)", fontFamily: "var(--mono)" }}>{hit ? "✓" : "·"}</td>;
                              })}
                              <td style={{ padding: "4px 10px", textAlign: "right", fontFamily: "var(--mono)", color: g.coverage >= 0.5 ? "var(--good)" : g.coverage > 0 ? "var(--warn)" : "var(--crit)" }}>{(g.coverage * 100).toFixed(0)}%</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  );
                })()}
                <Note>A column of all <span style={{ color: "var(--crit)" }}>·</span> means <b>no retrieved chunk contains that query term</b> — the corpus simply does not discuss it. That is what drives the INCORRECT verdict, and it is checkable rather than a model&apos;s opinion.</Note>
              </div>
            </div>
          )}
          {grades ? (
            <div style={pnl}>
              {head("var(--warn)", "Grades", <button className="btn ghost sm" onClick={ran}>↻ Replay</button>)}
              <div style={{ padding: 15, display: "flex", flexDirection: "column", gap: 5 }} key={`gr${play}`}>
                {grades.map((g, n) => {
                  const tone = g.verdict === "relevant" ? "var(--good)" : g.verdict === "partial" ? "var(--warn)" : "var(--crit)";
                  return (
                    <div key={g.i} style={{ display: "flex", gap: 10, alignItems: "center", padding: "8px 11px", border: "1px solid var(--border)", borderLeft: `2px solid ${tone}`, borderRadius: "var(--rs)", background: "var(--panel-2)", opacity: 0, animation: "popIn .4s ease forwards", animationDelay: `${n * 0.08}s` }}>
                      <span className="mono" style={{ fontSize: 10.5, color: tone, width: 62, flex: "none" }}>{g.verdict}</span>
                      <span style={{ fontSize: 11.5, color: "var(--muted)", flex: 1, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{chunks[g.i]}</span>
                      <span className="mono" style={{ fontSize: 10, color: "var(--faint)", flex: "none" }}>cover {(g.coverage * 100).toFixed(0)}% · sim {g.similarity.toFixed(2)} · <b style={{ color: tone }}>{g.score.toFixed(2)}</b></span>
                    </div>
                  );
                })}
              </div>
            </div>
          ) : <Empty>Grade the docs to continue.</Empty>}
        </>
      );

      if (step === 3) return (
        <>
          <Tiles items={[[branch ? branch.grade : "—", "verdict"], [grades ? Math.max(...grades.map((g) => g.score)).toFixed(2) : "—", "best score"], [branch ? String(branch.finalHits.length) : "—", "docs kept"]]} />
          <div style={{ ...pnl, marginBottom: 16 }}>
            {head("var(--crit)", "Branch controls")}
            <div style={{ padding: 15 }}>
              {grades && (() => {
                const best = Math.max(...grades.map((g) => g.score));
                return (
                  <div style={{ marginBottom: 14 }}>
                    <div className="flow-label">the decision — where the best chunk falls on the scale</div>
                    <div style={{ position: "relative", height: 44, background: "var(--panel-2)", border: "1px solid var(--border)", borderRadius: 8 }}>
                      <div style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: `${lower * 100}%`, background: "color-mix(in srgb, var(--crit) 16%, transparent)", borderRadius: "8px 0 0 8px" }} />
                      <div style={{ position: "absolute", left: `${lower * 100}%`, top: 0, bottom: 0, width: `${(upper - lower) * 100}%`, background: "color-mix(in srgb, var(--warn) 16%, transparent)" }} />
                      <div style={{ position: "absolute", left: `${upper * 100}%`, right: 0, top: 0, bottom: 0, background: "color-mix(in srgb, var(--good) 16%, transparent)", borderRadius: "0 8px 8px 0" }} />
                      <div style={{ position: "absolute", left: `${best * 100}%`, top: 6, bottom: 6, width: 2, background: "var(--text)", transform: "translateX(-1px)" }} />
                      <span className="mono" style={{ position: "absolute", left: `${best * 100}%`, top: -2, fontSize: 9.5, color: "var(--text)", transform: "translateX(-50%)" }}>{best.toFixed(2)}</span>
                    </div>
                    <div className="row" style={{ justifyContent: "space-between", fontFamily: "var(--mono)", fontSize: 9.5, color: "var(--faint)", marginTop: 3 }}>
                      <span style={{ color: "var(--crit)" }}>0 · incorrect</span>
                      <span style={{ color: "var(--warn)" }}>{lower.toFixed(2)} · ambiguous</span>
                      <span style={{ color: "var(--good)" }}>{upper.toFixed(2)} · correct · 1</span>
                    </div>
                  </div>
                );
              })()}
              <button className="btn" disabled={!grades} onClick={() => {
                if (!grades) return;
                const { grade, reason } = gradeRetrieval(grades, { upper, lower });
                const detail = [reason];
                if (grade === "correct") { setBranch({ grade, reason, detail: [...detail, "Retrieval trusted — generating from these chunks."], finalHits: grades.map((g) => g.i) }); }
                else if (grade === "ambiguous") {
                  const partial = grades.filter((g) => g.verdict !== "irrelevant").map((g) => g.i);
                  const rw = rewriteQuery(idx, query, partial.length ? partial : grades.map((g) => g.i));
                  const mx = (a: number[]) => Math.max(...a, 1e-9);
                  const bm = bm25ScoresTuned(idx, rw.rewritten, k1, b);
                  const second = bm.map((x, i) => ({ i, score: x / mx(bm) })).sort((a, c) => c.score - a.score).slice(0, topK).map((r) => r.i);
                  setBranch({ grade, reason, rewritten: rw.rewritten, finalHits: [...new Set([...second, ...partial])].slice(0, topK),
                    detail: [...detail, `Rewrote query → "${rw.rewritten}"`, rw.dropped.length ? `Dropped terms the index has never seen: ${rw.dropped.join(", ")}` : "No unknown terms to drop.", rw.added.length ? `Expanded with corpus feedback: ${rw.added.join(", ")}` : "No expansion terms found.", "Re-retrieved and merged with the partial hits."] });
                } else {
                  setBranch({ grade, reason, finalHits: [], detail: [...detail, "Corpus rejected. No external source is wired, so answering is REFUSED rather than hallucinated."] });
                }
                ran();
              }}>▶ Execute branch</button>
              <Note>Drag the thresholds on the previous step to force each branch. <b>AMBIGUOUS</b> rewrites the query using the corpus itself — dropped terms appear nowhere in the index; added terms come from the chunks that partially matched.</Note>
            </div>
          </div>
          {branch ? (
            <div style={pnl}>
              {head(GRADE_COLOR[branch.grade], "Branch taken", <button className="btn ghost sm" onClick={ran}>↻ Replay</button>)}
              <div style={{ padding: 15, display: "flex", flexDirection: "column", gap: 8 }} key={`br${play}`}>
                <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                  {(["correct", "ambiguous", "incorrect"] as Grade[]).map((g) => (
                    <span key={g} className="mono" style={{ fontSize: 11, padding: "5px 14px", borderRadius: 999, textTransform: "uppercase", letterSpacing: ".06em", border: `1px solid ${branch.grade === g ? GRADE_COLOR[g] : "var(--border)"}`, color: branch.grade === g ? GRADE_COLOR[g] : "var(--faint)", background: branch.grade === g ? "var(--panel-2)" : "transparent", fontWeight: branch.grade === g ? 700 : 400 }}>
                      {branch.grade === g ? "▶ " : ""}{g}
                    </span>
                  ))}
                </div>
                {branch.detail.map((d, n) => (
                  <div key={n} style={{ fontSize: 11.5, color: "var(--muted)", lineHeight: 1.55, padding: "7px 11px", border: "1px solid var(--border)", borderRadius: "var(--rs)", background: "var(--panel-2)", opacity: 0, animation: "popIn .4s ease forwards", animationDelay: `${n * 0.14}s` }}>{d}</div>
                ))}
              </div>
            </div>
          ) : <Empty>Execute the branch to continue.</Empty>}
        </>
      );

      return (
        <>
          <AnswerPanel llm={llm} ctx={(branch?.finalHits || []).map((i) => chunks[i]).filter(Boolean)} sources={(branch?.finalHits || []).map((i) => chunkSrc[i] || "corpus")} note={query} />
          <Note><b>This is the whole point:</b> plain RAG hands the top-k to the LLM regardless of quality, and the LLM answers fluently from irrelevant context. Here a bad retrieval produces <i>no answer</i> instead of a confident wrong one.</Note>
        </>
      );
    }

    /* ══════════════ 05 MULTIMODAL ══════════════ */
    if (step === 0) return (
      <>
        <Tiles items={[[sources ? String(sources.filter((s) => s.modality === "text").length) : "—", "text"], [sources ? String(sources.filter((s) => s.modality === "table").length) : "—", "table rows"], [sources ? String(sources.filter((s) => s.modality === "image").length) : "—", "figures"]]} />
        <div style={{ ...pnl, marginBottom: 16 }}>
          {head("var(--accent)", "Source controls")}
          <div style={{ padding: 15 }}>
            <button className="btn" onClick={() => { setSources(sampleMultimodalCorpus()); resetFrom(1); ran(); }}>▶ Load sources</button>
            <Note>A question&apos;s answer often lives in a <b>chart or a table row</b>, not in prose. A text-only index cannot reach those at all.</Note>
          </div>
        </div>
        {sources ? (
          <div style={pnl}>
            {head("var(--accent)", "Loaded sources", <button className="btn ghost sm" onClick={ran}>↻ Replay</button>)}
            <div style={{ padding: 15, display: "flex", flexDirection: "column", gap: 5 }} key={`src${play}`}>
              {sources.map((it, i) => (
                <div key={i} style={{ display: "flex", gap: 9, padding: "8px 11px", border: "1px solid var(--border)", borderLeft: `2px solid ${MODALITY_COLOR[it.modality]}`, borderRadius: "var(--rs)", background: "var(--panel-2)", alignItems: "center", opacity: 0, animation: "popIn .4s ease forwards", animationDelay: `${i * 0.05}s` }}>
                  <span className="mono" style={{ fontSize: 10, padding: "1px 7px", borderRadius: 999, border: `1px solid ${MODALITY_COLOR[it.modality]}`, color: MODALITY_COLOR[it.modality], flex: "none" }}>{it.modality}</span>
                  <b style={{ fontSize: 12 }}>{it.title}</b>
                  {it.textSource && <span className="mono" style={{ fontSize: 9.5, color: "var(--faint)" }}>text from {it.textSource}</span>}
                </div>
              ))}
            </div>
          </div>
        ) : <Empty>Load the sources to continue.</Empty>}
      </>
    );

    if (step === 1) return (
      <>
        <div style={{ ...pnl, marginBottom: 16 }}>
          {head("var(--sky)", "Encoding controls")}
          <div style={{ padding: 15 }}>
            <button className="btn" disabled={!sources} onClick={() => { setEncoded(sources); resetFrom(2); ran(); }}>▶ Encode every modality</button>
            <Note>
              <b>Tables</b> are serialised one row at a time <i>with their column headers</i> (&ldquo;Revenue: $2.42M&rdquo;) — headers are the words a question actually contains. <b>Images</b> contribute their caption / OCR / alt text.
            </Note>
            <Cfg>3 modalities → <K>one</K> shared text representation each → one vector space</Cfg>
          </div>
        </div>
        {encoded && (() => {
          const t = encoded.find((x) => x.modality === "table" && x.rows);
          if (!t?.rows) return null;
          return (
            <div style={{ ...pnl, marginBottom: 16 }}>
              {head("var(--accent)", "How a table row becomes retrievable", <span className="note" style={{ fontSize: 10 }}>the key trick</span>)}
              <div style={{ padding: 15 }}>
                <div style={{ display: "grid", gridTemplateColumns: "1fr auto 1fr", gap: 14, alignItems: "center" }}>
                  <div>
                    <div className="flow-label">before — a row in a grid</div>
                    <table style={{ borderCollapse: "collapse", fontSize: 11.5 }}>
                      <tbody>{t.rows.map((r, ri) => (
                        <tr key={ri}>{r.map((c, ci) => <td key={ci} style={{ border: "1px solid var(--border)", padding: "4px 9px", color: ri === 0 ? "var(--faint)" : "var(--text)", fontFamily: ri === 0 ? "inherit" : "var(--mono)" }}>{c}</td>)}</tr>
                      ))}</tbody>
                    </table>
                  </div>
                  <span style={{ color: "var(--accent)", fontSize: 20 }}>→</span>
                  <div>
                    <div className="flow-label">after — one retrievable string</div>
                    <div className="mono" style={{ fontSize: 11, color: "var(--muted)", lineHeight: 1.6, border: "1px solid var(--border)", borderLeft: "2px solid var(--sky)", borderRadius: "var(--rs)", background: "var(--panel-2)", padding: "9px 11px" }}>{t.content}</div>
                  </div>
                </div>
                <Note>
                  The cell <span className="mono">$2.42M</span> alone is unretrievable — no question contains it. Pairing every value with its
                  <b> column header</b> (<span className="mono">Revenue: $2.42M</span>) injects the words a question actually uses, so
                  &ldquo;what was Q4 revenue&rdquo; now shares terms with the row.
                </Note>
              </div>
            </div>
          );
        })()}
        {encoded ? (
          <div style={pnl}>
            {head("var(--sky)", "What each source became", <button className="btn ghost sm" onClick={ran}>↻ Replay</button>)}
            <div style={{ padding: 15, display: "flex", flexDirection: "column", gap: 6 }} key={`enc${play}`}>
              {encoded.map((it, i) => (
                <div key={i} style={{ border: "1px solid var(--border)", borderLeft: `2px solid ${MODALITY_COLOR[it.modality]}`, borderRadius: "var(--rs)", padding: "9px 11px", background: "var(--panel-2)", display: "flex", flexDirection: "column", gap: 4, opacity: 0, animation: "popIn .4s ease forwards", animationDelay: `${i * 0.06}s` }}>
                  <span className="mono" style={{ fontSize: 10, color: MODALITY_COLOR[it.modality] }}>{it.modality} · {it.title}</span>
                  <span className="mono" style={{ fontSize: 11, color: "var(--muted)", lineHeight: 1.5 }}>{it.content}</span>
                </div>
              ))}
              <Note><b>Honest scope:</b> there is no vision encoder here. Production systems embed pixels with CLIP or ColPali so a chart is findable even when nothing describes it; this index reaches images only through their attached text.</Note>
            </div>
          </div>
        ) : <Empty>Encode the sources to continue.</Empty>}
      </>
    );

    if (step === 2) return (
      <>
        <Tiles items={[[mIndex ? String(mIndex.items.length) : "—", "documents"], [String(modalities.length), "modalities on"], ["1", "shared index"]]} />
        <div style={{ ...pnl, marginBottom: 16 }}>
          {head("var(--good)", "Index controls")}
          <div style={{ padding: 15 }}>
            <label className="fld">Modalities to include (untick to simulate a text-only index)</label>
            <div className="row" style={{ gap: 8, marginBottom: 12 }}>
              {(["text", "table", "image"] as Modality[]).map((m) => {
                const on = modalities.includes(m);
                return (
                  <button key={m} onClick={() => { setModalities((cur) => (cur.includes(m) ? cur.filter((x) => x !== m) : [...cur, m])); resetFrom(2); }}
                    style={{ flex: 1, fontSize: 11, padding: "8px 4px", borderRadius: "var(--rs)", cursor: "pointer", fontFamily: "var(--mono)", border: `1px solid ${on ? MODALITY_COLOR[m] : "var(--border)"}`, color: on ? MODALITY_COLOR[m] : "var(--faint)", background: on ? "var(--panel-2)" : "transparent" }}>
                    {m} {on ? "✓" : ""}
                  </button>
                );
              })}
            </div>
            <button className="btn" disabled={!encoded} onClick={() => { if (encoded) { setMIndex(buildMultimodalIndex(encoded)); resetFrom(3); ran(); } }}>▶ Build unified index</button>
            <Note>Every modality contributes <b>exactly one document</b>, so their scores are directly comparable. That shared ranking <i>is</i> the architecture — a table row can now outrank a paragraph.</Note>
          </div>
        </div>
        {mIndex ? <Cfg>indexed <K>{mIndex.items.length}</K> documents into one vector space · vocabulary <K>{Object.keys(mIndex.idx.df).length}</K> terms</Cfg> : <Empty>Build the index to continue.</Empty>}
      </>
    );

    if (step === 3) return (
      <>
        <Tiles items={[[mHits ? String(mHits.perModality.text) : "—", "text hits"], [mHits ? String(mHits.perModality.table) : "—", "table hits"], [mHits ? String(mHits.perModality.image) : "—", "image hits"]]} />
        <div style={{ ...pnl, marginBottom: 16 }}>
          {head("var(--good)", "Retrieval controls")}
          <div style={{ padding: 15 }}>
            <div className="row" style={{ gap: 12, alignItems: "flex-end", flexWrap: "wrap" }}>
              <div style={{ flex: 1, minWidth: 220 }}>
                <label className="fld">Query</label>
                <input style={inputStyle} value={query} onChange={(e) => { setQuery(e.target.value); resetFrom(3); }} />
              </div>
              <div style={{ minWidth: 150 }}>
                <label className="fld">Strategy</label>
                <select value={mStrategy} onChange={(e) => { setMStrategy(e.target.value as typeof mStrategy); resetFrom(3); }} style={{ width: 150 }}>
                  <option value="hybrid">hybrid</option><option value="vector">vector</option><option value="keyword">keyword</option>
                </select>
              </div>
              <Knob label="top-k" value={topK} min={1} max={8} step={1} onChange={(v) => { setTopK(v); resetFrom(3); }} />
              <button className="btn" disabled={!mIndex} onClick={() => { if (mIndex) { setMHits(retrieveMultimodal(mIndex, query, topK, { strategy: mStrategy, only: modalities })); ran(); } }}>▶ Retrieve</button>
            </div>
            <Note>Try <span className="mono">Q4 2024 revenue and margin</span> — a <b>table row</b> should outrank the prose. Try <span className="mono">pie chart revenue by region APAC</span> — a <b>figure</b> should win.</Note>
          </div>
        </div>
        {mHits ? (
          <div style={pnl}>
            {head("var(--good)", "Cross-modality ranking", <button className="btn ghost sm" onClick={ran}>↻ Replay</button>)}
            <div style={{ padding: 15, display: "flex", flexDirection: "column", gap: 7 }} key={`mh${play}`}>
              {mHits.hits.length ? mHits.hits.map((h, n) => (
                <div key={h.i} style={{ border: "1px solid var(--border)", borderLeft: `2px solid ${MODALITY_COLOR[h.modality]}`, borderRadius: "var(--rs)", padding: "9px 11px", background: "var(--panel-2)", display: "flex", flexDirection: "column", gap: 6, opacity: 0, animation: "popIn .4s ease forwards", animationDelay: `${n * 0.07}s` }}>
                  <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                    <span className="mono" style={{ fontSize: 10.5, color: "var(--faint)" }}>#{n + 1}</span>
                    <span className="mono" style={{ fontSize: 10, padding: "1px 7px", borderRadius: 999, border: `1px solid ${MODALITY_COLOR[h.modality]}`, color: MODALITY_COLOR[h.modality] }}>{h.modality}</span>
                    <b style={{ fontSize: 12 }}>{h.item.title}</b>
                    <span className="mono" style={{ fontSize: 10.5, color: "var(--muted)", marginLeft: "auto" }}>{h.score.toFixed(3)}</span>
                  </div>
                  {h.item.rows ? (
                    <table style={{ borderCollapse: "collapse", fontSize: 11.5 }}>
                      <tbody>{h.item.rows.map((r, ri) => (
                        <tr key={ri}>{r.map((c, ci) => <td key={ci} style={{ border: "1px solid var(--border)", padding: "4px 9px", color: ri === 0 ? "var(--faint)" : "var(--text)", fontFamily: ri === 0 ? "inherit" : "var(--mono)" }}>{c}</td>)}</tr>
                      ))}</tbody>
                    </table>
                  ) : <span style={{ fontSize: 11.5, color: "var(--muted)", lineHeight: 1.55 }}>{h.item.content}</span>}
                </div>
              )) : <Empty>Nothing matched in the selected modalities.</Empty>}
            </div>
          </div>
        ) : <Empty>Retrieve to continue.</Empty>}
      </>
    );

    return <AnswerPanel llm={llm} ctx={(mHits?.hits || []).map((h) => `(${h.modality}) ${h.item.title} — ${h.item.content}`)} sources={(mHits?.hits || []).map((h) => `${h.modality} · ${h.item.title}`)} note={query} />;
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))", gap: 8 }}>
        {ARCHES.map((a) => {
          const on = a.id === arch;
          return (
            <button key={a.id} onClick={() => setArch(a.id)}
              style={{
                textAlign: "left", padding: "10px 12px", borderRadius: "var(--r)", cursor: "pointer",
                border: on ? "1px solid var(--accent)" : "1px solid var(--border)",
                background: on ? "var(--accent-weak)" : "var(--panel)",
                boxShadow: on ? "0 0 0 3px var(--accent-weak)" : "none",
                fontFamily: "inherit", color: "var(--text)", display: "flex", flexDirection: "column", gap: 3,
              }}>
              <span className="mono" style={{ fontSize: 10, color: on ? "var(--accent-strong)" : "var(--faint)", letterSpacing: ".08em" }}>{a.n}</span>
              <b style={{ fontSize: 13 }}>{a.label} RAG</b>
              <span style={{ fontSize: 10.5, color: "var(--muted)", lineHeight: 1.4 }}>{a.tagline}</span>
            </button>
          );
        })}
      </div>

      <div className="stepper" style={{ marginBottom: 0 }}>
        {steps.map((s, i) => (
          <button key={s} className={step === i ? "on" : ""} disabled={!reached(i)} onClick={() => setStep(i)} title={reached(i) ? undefined : "run the previous step first"}>
            <b>{done[arch][i] ? "✓" : i + 1}</b> {s}
          </button>
        ))}
      </div>

      <div className="card rag-lab">
        <div className="card-h">
          <span className="t">{activeArch.n} · {activeArch.label} RAG — step {step + 1}: {steps[step]}</span>
          <span className="mono r" style={{ fontSize: 11, color: "var(--faint)" }}>{arch === "multimodal" && step < 3 ? `${sources?.length ?? 0} sources` : `query: ${query.slice(0, 34) || "—"}`}</span>
        </div>
        <div className="card-b">
          {body()}
          <div className="stepnav">
            <button className="btn ghost" disabled={step === 0} onClick={() => setStep((s) => Math.max(0, s - 1))}>← Back</button>
            <button className="btn" disabled={step === steps.length - 1 || !done[arch][step]} onClick={() => setStep((s) => Math.min(steps.length - 1, s + 1))}
              title={done[arch][step] ? undefined : "run this step first"}>
              Next: {steps[Math.min(step + 1, steps.length - 1)]} →
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function ContribRow({ t }: { t: { term: string; tf: number; df: number; idf: number; contribution: number } }) {
  const dead = t.tf === 0;
  return (
    <>
      <span className="mono" style={{ fontSize: 11, color: dead ? "var(--faint)" : "var(--text)" }}>{t.term}</span>
      <span className="mono" style={{ fontSize: 11, color: dead ? "var(--crit)" : "var(--muted)" }}>{t.tf}</span>
      <span className="mono" style={{ fontSize: 11, color: "var(--muted)" }}>{t.df}</span>
      <span className="mono" style={{ fontSize: 11, color: "var(--muted)" }}>{t.idf.toFixed(2)}</span>
      <span className="mono" style={{ fontSize: 11, color: dead ? "var(--faint)" : "var(--accent)" }}>{t.contribution.toFixed(3)}</span>
    </>
  );
}
