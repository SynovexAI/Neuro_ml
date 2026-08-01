"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { chunkText, buildIndex, retrieve, cosine, tokenize, queryVector, type RagIndex, type Strategy, type Vec } from "@/lib/ragUtils";

type Doc = { id: string; name: string; kind: string; text: string };
type Chunk = { text: string; docName: string; docKind: string };
type Step = "source" | "chunk" | "embed" | "query";

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

  const [strategy, setStrategy] = useState<Strategy>("hybrid");
  const [topK, setTopK] = useState(3);
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

  useEffect(() => {
    fetch("/api/models").then((r) => r.json()).then((j) => { setProvider(j.provider); setProvKnown(true); }).catch(() => setProvKnown(true));
    return () => { if (timer.current) clearTimeout(timer.current); };
  }, []);

  // Auto-run chunking / embedding the first time you enter a step (explicit buttons re-run).
  useEffect(() => {
    if (step === "chunk" && chunks.length === 0 && docs.length) runChunking();
    if (step === "embed" && !index && chunks.length) runEmbedding();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step]);

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
  async function ask() {
    if (!index || chunks.length === 0) { setAnswer("Run chunking and embedding first."); return; }
    setRunning(true); setTab("out"); setAnswer(""); setMeta("retrieving…");
    const steps = [
      { who: "embed query", what: "question → vector" },
      { who: "retrieve", what: `${strategy} · top-k ${topK}` },
      { who: "prompt", what: "inject retrieved context + sources" },
      { who: "generate", what: `stream → ${provider || "provider"}` },
    ];
    setTraceStep(steps, 1);
    const top = retrieve(index, question, strategy, topK);
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
      const res = await fetch("/api/chat", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ messages, temperature: 0.2 }) });
      if (!res.ok || !res.body) { const j = await res.json().catch(() => ({ error: "failed" })); setAnswer("⚠ " + (j.error || "failed")); setMeta("error"); setRunning(false); return; }
      const reader = res.body.getReader(); const dec = new TextDecoder(); let text = "";
      for (; ;) { const { done, value } = await reader.read(); if (done) break; text += dec.decode(value, { stream: true }); setAnswer(text); }
      setMeta(`grounded · ${top.length} sources · ${Math.round(performance.now() - t0)}ms`);
      setTrace(steps.map((s) => ({ ...s, state: "done" })));
    } catch (e) { setAnswer("⚠ " + (e as Error).message); setMeta("error"); }
    setRunning(false);
  }

  const stepBtn = (k: Step, n: number, label: string, enabled: boolean) => (
    <button className={step === k ? "on" : ""} disabled={!enabled} onClick={() => goStep(k)}><b>{n}</b>{label}</button>
  );
  const stepWords = Math.max(1, size - overlap);

  return (
    <>
      <div className="lab-head">
        <div>
          <div className="eyebrow">Lab 02 · flagship</div>
          <h2 className="page-h">RAG Lab</h2>
          <p className="page-sub" style={{ margin: 0 }}>Add sources, watch them get chunked and embedded into a vector store, then ask — the answer is grounded and cites the exact chunk &amp; source.</p>
        </div>
        <div className="acts"><button className="btn ghost sm" onClick={saveProject}>{saved || "💾 Save"}</button></div>
      </div>

      {provKnown && provider === null && <div className="warnbar">No provider configured — an admin must add one under Admin → Providers before the answer step (source/chunk/embed still work).</div>}

      <div className="stepper">
        {stepBtn("source", 1, "Source", true)}
        {stepBtn("chunk", 2, "Chunk", docs.length > 0)}
        {stepBtn("embed", 3, "Embed & Index", docs.length > 0)}
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

      {/* STEP 3 — EMBED */}
      {step === "embed" && (
        <div className="card">
          <div className="card-h"><span className="t">Embed &amp; store in vector index</span><span className="mono r">{embedding ? <><span className="busy-dot" />vectorizing…</> : index ? `${index.vectors.length} vectors` : "not built"}</span></div>
          <div className="card-b">
            <div className="stat-row">
              <div className="stat">chunks<b>{chunks.length}</b></div>
              <div className="stat">vectors stored<b>{index ? index.vectors.length : 0}</b></div>
              <div className="stat">vocabulary (dims)<b>{vocab}</b></div>
              <div className="stat">similarity<b>cosine</b></div>
            </div>
            <div className="row" style={{ marginBottom: 12 }}><button className="btn" onClick={runEmbedding} disabled={embedding}>▶ Run embedding</button><span className="note">turns each chunk into a vector and stores it</span></div>

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
            <div className="stepnav"><button className="btn ghost" onClick={() => goStep("chunk")}>← Back</button><button className="btn" disabled={!index} onClick={() => goStep("query")}>Next: Retrieve &amp; Answer →</button></div>
          </div>
        </div>
      )}

      {/* STEP 4 — QUERY */}
      {step === "query" && (
        <div className="split col-2">
          <div className="card">
            <div className="card-h"><span className="t">Retrieve &amp; ask</span></div>
            <div className="card-b">
              {!index && <div className="warnbar">Run embedding first (step 3).</div>}
              <label className="fld">Retrieval parameters</label>
              <div className="row" style={{ flexWrap: "wrap", gap: 10, marginBottom: 12 }}>
                <select value={strategy} onChange={(e) => setStrategy(e.target.value as Strategy)} style={{ width: 160 }}>
                  <option value="hybrid">Hybrid</option><option value="vector">Vector (TF-IDF)</option><option value="keyword">Keyword (BM25)</option>
                </select>
                <div className="knob" style={{ margin: 0, minWidth: 140 }}><div className="kr"><span>Top-k</span><b>{topK}</b></div><input type="range" min={1} max={6} value={topK} onChange={(e) => setTopK(+e.target.value)} /></div>
              </div>
              <label className="fld">Question</label>
              <input type="text" value={question} onChange={(e) => setQuestion(e.target.value)} />
              <div className="row" style={{ marginTop: 12 }}><button className="btn" onClick={ask} disabled={running || !index}>▶ Ask</button></div>
              {hits.length === 0 && <div className="note" style={{ marginTop: 16 }}>Ask to retrieve.</div>}
              {index && hits.length > 0 && (() => {
                const qTerms = Array.from(new Set(tokenize(question)));
                const qv = queryVector(index, question);
                let probed = 0, bs = -Infinity;
                if (clusters) clusters.centroids.forEach((ct, c) => { const s = cosine(qv, ct); if (s > bs) { bs = s; probed = c; } });
                return (
                  <>
                    <label className="fld" style={{ marginTop: 16 }}>Query → tokens (embedded the same way as the chunks)</label>
                    <div className="q-terms">{qTerms.map((t) => <span key={t} className="q-term">{t}</span>)}</div>

                    {clusters && (
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
                          <div className="ch"><span>chunk {h.i + 1}<span className="src-tag">{chunks[h.i].docKind}</span> {chunks[h.i].docName}</span><span style={{ color: "var(--accent)" }}>score {h.score.toFixed(2)}</span></div>
                          <div className="q-match">{qTerms.map((t) => <span key={t} className={`q-chip ${matched.includes(t) ? "on" : ""}`}>{t}</span>)}</div>
                          <div className="q-matchnote">{matched.length}/{qTerms.length} query terms overlap this chunk</div>
                          <div style={{ color: "var(--muted)", marginTop: 6 }}>{highlightTerms(chunks[h.i].text.slice(0, 170), matched)}…</div>
                        </div>
                      );
                    })}
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

      {index && (
        <div className="card" style={{ marginTop: 16 }}>
          <div className="card-h"><span className="t">Compare chunking</span><div className="r"><button className="btn sm" onClick={compareChunking}>▶ Compare sizes</button></div></div>
          <div className="card-b">
            <div className="note" style={{ marginBottom: 10 }}>Rebuilds the index at several chunk sizes and retrieves for your question — bigger isn&apos;t always better. Higher <b>top score</b> = the best chunk matches the question more strongly ({strategy}, top-{topK}).</div>
            {compareRows.length === 0 ? <div className="note">Click compare to score chunk sizes for “{question}”.</div> : (
              <div style={{ overflowX: "auto" }}><table className="tbl">
                <thead><tr><th>Chunk size</th><th>Overlap</th><th style={{ textAlign: "right" }}>Chunks</th><th style={{ textAlign: "right" }}>Top score</th><th style={{ textAlign: "right" }}>Avg top-{topK}</th><th>Best chunk</th></tr></thead>
                <tbody>{compareRows.map((r, i) => { const best = Math.max(...compareRows.map((x) => x.top)); const win = r.top === best && r.top > 0; return <tr key={i}><td>{r.size}w {win && <span title="best">⭐</span>}</td><td>{r.overlap}</td><td className="mono" style={{ textAlign: "right" }}>{r.chunks}</td><td className="mono" style={{ textAlign: "right", color: win ? "var(--good)" : undefined, fontWeight: win ? 600 : 400 }}>{r.top.toFixed(3)}</td><td className="mono" style={{ textAlign: "right" }}>{r.avg.toFixed(3)}</td><td style={{ fontSize: 11, color: "var(--muted)", maxWidth: 260 }}>{r.best.slice(0, 120)}{r.best.length > 120 ? "…" : ""}</td></tr>; })}</tbody>
              </table></div>
            )}
          </div>
        </div>
      )}
    </>
  );
}
