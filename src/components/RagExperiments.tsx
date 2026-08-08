"use client";
import { useEffect, useMemo, useState } from "react";

// Experiment tracking for the RAG Lab. Saves ONLY configuration + metrics (never
// document or embedding copies) so it's storage-cheap. Lets students build a
// history of runs and compare them side by side — the practical "change a knob,
// measure the effect" loop that a real RAG engineer works in.

export type ExpConfig = {
  backend: string; size: number; overlap: number; strategy: string; metric: string;
  topK: number; rerank: string; mmrLambda: number; embedMode: string; embModel: string; kgHops: number;
};
export type ExpMetrics = { p: number; r: number; mrr: number; ndcg: number } | null;
export type CurrentExperiment = {
  config: ExpConfig; metrics: ExpMetrics; question: string; dataset: string; chunkCount: number; latencyMs: number;
};
type Row = {
  id: string; label: string; dataset: string | null; question: string | null;
  config: ExpConfig; metrics: ExpMetrics; chunkCount: number; latencyMs: number; ts: string | null;
};

const METRIC_COLS: [string, keyof NonNullable<ExpMetrics>][] = [["P@k", "p"], ["Recall@k", "r"], ["MRR", "mrr"], ["nDCG", "ndcg"]];

export default function RagExperiments({ current, onLoad }: { current: CurrentExperiment; onLoad: (c: ExpConfig) => void }) {
  const [rows, setRows] = useState<Row[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [sel, setSel] = useState<Set<string>>(new Set());
  const [note, setNote] = useState("");

  const defaultLabel = useMemo(() => {
    const c = current.config;
    return `${current.dataset || "docs"} · ${c.size}w/${c.overlap} · ${c.strategy}${c.rerank === "mmr" ? "+mmr" : ""} · k${c.topK}`.slice(0, 80);
  }, [current]);
  const [label, setLabel] = useState(defaultLabel);
  useEffect(() => { setLabel(defaultLabel); }, [defaultLabel]);

  async function load() {
    setLoading(true);
    try { const r = await fetch("/api/rag/experiments"); const j = await r.json(); if (r.ok) setRows(j.experiments || []); } catch { /* ignore */ } finally { setLoading(false); }
  }
  useEffect(() => { load(); }, []);

  async function save() {
    setSaving(true); setNote("");
    try {
      const r = await fetch("/api/rag/experiments", {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ label, dataset: current.dataset, question: current.question, config: current.config, metrics: current.metrics, chunkCount: current.chunkCount, latencyMs: current.latencyMs }),
      });
      if (!r.ok) { const j = await r.json().catch(() => ({})); throw new Error(j.error || "save failed"); }
      setNote("Saved ✓"); await load();
    } catch (e) { setNote((e as Error).message); } finally { setSaving(false); setTimeout(() => setNote(""), 2500); }
  }
  async function remove(id: string) {
    setRows((rs) => rs.filter((r) => r.id !== id));
    setSel((s) => { const n = new Set(s); n.delete(id); return n; });
    try { await fetch(`/api/rag/experiments/${id}`, { method: "DELETE" }); } catch { /* ignore */ }
  }
  const toggle = (id: string) => setSel((s) => { const n = new Set(s); if (n.has(id)) n.delete(id); else n.add(id); return n; });

  const selected = rows.filter((r) => sel.has(r.id));
  const fmtTs = (t: string | null) => (t ? new Date(t).toLocaleString() : "—");
  const cfgSummary = (c: ExpConfig) => `${c.size}w/${c.overlap}ov · ${c.strategy}${c.rerank === "mmr" ? "+mmr" : ""} · ${c.embedMode === "neural" ? "neural" : "tf-idf"} · top-${c.topK}`;

  // Best value per metric across the selected experiments (max for quality metrics, min for latency).
  const bestOf = (key: keyof NonNullable<ExpMetrics>) => Math.max(...selected.map((r) => r.metrics?.[key] ?? -1));
  const bestLatency = Math.min(...selected.filter((r) => r.latencyMs > 0).map((r) => r.latencyMs), Infinity);

  return (
    <div className="card" style={{ marginTop: 16 }}>
      <div className="card-h"><span className="t">🧪 Experiments</span><span className="mono r">{rows.length} saved · config + metrics only</span></div>
      <div className="card-b">
        <div className="note" style={{ marginBottom: 12, lineHeight: 1.5 }}>Save this pipeline&apos;s <b>settings and metrics</b> (not the documents), then compare runs to see which knobs actually help. Storage cost is a few hundred bytes per experiment.</div>

        {/* save current */}
        <div className="row" style={{ gap: 8, alignItems: "center", flexWrap: "wrap", marginBottom: 14 }}>
          <input type="text" value={label} onChange={(e) => setLabel(e.target.value)} placeholder="experiment name" style={{ flex: 1, minWidth: 220 }} />
          <button className="btn sm" onClick={save} disabled={saving || !label.trim()}>{saving ? "saving…" : "💾 Save current experiment"}</button>
          {note && <span className="note" style={{ color: note.includes("✓") ? "var(--good)" : "var(--warn)" }}>{note}</span>}
        </div>
        {!current.metrics && (
          <div className="note" style={{ marginBottom: 12, fontSize: 11, color: "var(--warn)" }}>Tip: mark relevant chunks and click <b>Evaluate</b> above before saving, so this experiment records P@k / recall / MRR / nDCG — otherwise it saves config only.</div>
        )}

        {/* history */}
        {loading ? <div className="note"><span className="busy-dot" /> loading…</div>
          : rows.length === 0 ? <div className="note">No experiments yet — run the pipeline, then save it above.</div>
          : (
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {rows.map((r) => (
                <div key={r.id} className="row" style={{ gap: 10, alignItems: "center", border: `1px solid ${sel.has(r.id) ? "var(--accent)" : "var(--border)"}`, borderRadius: 10, padding: "9px 12px", background: sel.has(r.id) ? "var(--accent-weak)" : "var(--surface)" }}>
                  <input type="checkbox" checked={sel.has(r.id)} onChange={() => toggle(r.id)} title="select to compare" />
                  <span style={{ flex: 1, minWidth: 0 }}>
                    <span style={{ display: "block", fontWeight: 600, fontSize: 12.5, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{r.label}</span>
                    <span className="note" style={{ fontSize: 10, fontFamily: "var(--mono)" }}>{cfgSummary(r.config)} · {r.chunkCount} chunks{r.latencyMs ? ` · ${r.latencyMs}ms` : ""}</span>
                  </span>
                  {r.metrics
                    ? <span className="mono" style={{ fontSize: 10.5, color: "var(--muted)", flex: "0 0 auto" }}>MRR {r.metrics.mrr.toFixed(2)} · nDCG {r.metrics.ndcg.toFixed(2)}</span>
                    : <span className="note" style={{ fontSize: 9.5, flex: "0 0 auto" }}>no metrics</span>}
                  <span className="note" style={{ fontSize: 9.5, flex: "0 0 auto", whiteSpace: "nowrap" }}>{fmtTs(r.ts)}</span>
                  <button className="btn ghost sm" onClick={() => onLoad(r.config)} title="apply these settings to the lab">Load</button>
                  <button onClick={() => remove(r.id)} title="Delete" style={{ background: "none", border: "none", color: "var(--faint)", fontSize: 16, cursor: "pointer", lineHeight: 1, padding: "0 2px" }}>×</button>
                </div>
              ))}
            </div>
          )}

        {/* side-by-side comparison */}
        {selected.length >= 2 && (
          <div style={{ marginTop: 16 }}>
            <label className="fld">⚔️ Side-by-side comparison — {selected.length} experiments</label>
            <div style={{ overflowX: "auto", border: "1px solid var(--border)", borderRadius: 10 }}>
              <table className="dtable" style={{ width: "100%", fontSize: 12 }}><tbody>
                <tr>
                  <th style={{ textAlign: "left", position: "sticky", left: 0, background: "var(--panel)" }}>parameter</th>
                  {selected.map((r) => <th key={r.id} style={{ textAlign: "left", minWidth: 130 }}>{r.label.length > 22 ? r.label.slice(0, 21) + "…" : r.label}</th>)}
                </tr>
                {([
                  ["Dataset", (r: Row) => r.dataset || "—"],
                  ["Chunk size / overlap", (r: Row) => `${r.config.size}w / ${r.config.overlap}w`],
                  ["Strategy", (r: Row) => r.config.strategy + (r.config.rerank === "mmr" ? " + MMR" : "")],
                  ["Embeddings", (r: Row) => r.config.embedMode === "neural" ? `neural${r.config.embModel ? " · " + r.config.embModel : ""}` : "TF-IDF"],
                  ["Similarity", (r: Row) => r.config.metric],
                  ["Top-K", (r: Row) => String(r.config.topK)],
                  ["Chunks", (r: Row) => String(r.chunkCount)],
                ] as [string, (r: Row) => string][]).map(([lbl, get]) => (
                  <tr key={lbl}>
                    <td style={{ color: "var(--muted)", position: "sticky", left: 0, background: "var(--panel)", fontWeight: 500 }}>{lbl}</td>
                    {selected.map((r) => <td key={r.id} style={{ fontFamily: "var(--mono)", fontSize: 11.5 }}>{get(r)}</td>)}
                  </tr>
                ))}
                <tr>
                  <td style={{ color: "var(--muted)", position: "sticky", left: 0, background: "var(--panel)", fontWeight: 500 }}>Latency</td>
                  {selected.map((r) => {
                    const best = r.latencyMs > 0 && r.latencyMs === bestLatency;
                    return <td key={r.id} style={{ fontFamily: "var(--mono)", fontSize: 11.5, color: best ? "var(--good)" : undefined, fontWeight: best ? 700 : undefined }}>{r.latencyMs ? `${r.latencyMs}ms${best ? " ⚡" : ""}` : "—"}</td>;
                  })}
                </tr>
                {METRIC_COLS.map(([lbl, key]) => {
                  const best = bestOf(key);
                  const anyHas = selected.some((r) => r.metrics);
                  if (!anyHas) return null;
                  return (
                    <tr key={key}>
                      <td style={{ color: "var(--muted)", position: "sticky", left: 0, background: "var(--panel)", fontWeight: 500 }}>{lbl}</td>
                      {selected.map((r) => {
                        const v = r.metrics?.[key];
                        const isBest = v != null && best >= 0 && v === best && selected.filter((x) => x.metrics).length > 1;
                        return <td key={r.id} style={{ fontFamily: "var(--mono)", fontSize: 11.5, color: isBest ? "var(--good)" : undefined, fontWeight: isBest ? 700 : undefined }}>{v == null ? "—" : v.toFixed(2) + (isBest ? " ⭐" : "")}</td>;
                      })}
                    </tr>
                  );
                })}
              </tbody></table>
            </div>
            <div className="note" style={{ marginTop: 6, lineHeight: 1.5 }}>Green/⭐ marks the best value in each row (higher for quality metrics, lower for latency). Metrics appear only for experiments you evaluated against relevant chunks.</div>
          </div>
        )}
        {selected.length === 1 && <div className="note" style={{ marginTop: 12 }}>Select one more experiment to compare side by side.</div>}
      </div>
    </div>
  );
}
