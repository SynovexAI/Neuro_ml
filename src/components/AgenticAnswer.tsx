"use client";
import { useRef, useState, type ReactNode } from "react";

// Self-contained Agentic RAG panel. It runs a real ReAct loop over /api/chat (injected as `chat`)
// and the parent's retrieval (injected as `retrieve`) — so it reuses the existing engine and
// never modifies the current single-shot answer path. Used by both RagLab (Steps) and RagFlowLab (Canvas).

export type AgentTool = "vector" | "keyword" | "hybrid" | "kg" | "web";
export type AgentHit = { i: number; score: number };
export type ChatMessage = { role: "system" | "user" | "assistant"; content: string };

export type TraceEvent =
  | { kind: "plan" | "reflect" | "grade"; text: string; step: number }
  | { kind: "tool"; tool: AgentTool; query: string; hits: AgentHit[]; step: number }
  | { kind: "web"; url: string; ok: boolean; step: number }
  | { kind: "answer"; text: string; citations: number[]; step: number }
  | { kind: "error"; text: string; step: number };

const TOOL_META: Record<AgentTool, { label: string; icon: string; color: string; desc: string }> = {
  vector: { label: "vector search", icon: "◆", color: "#5b7cff", desc: "meaning match — good for paraphrased questions" },
  keyword: { label: "keyword (BM25)", icon: "🔤", color: "#22b8cf", desc: "exact terms — codes, names, IDs" },
  hybrid: { label: "hybrid", icon: "⚡", color: "#f59e0b", desc: "blends meaning + exact terms (safe default)" },
  kg: { label: "knowledge graph", icon: "🕸", color: "#a855f7", desc: "follows entity → relation links" },
  web: { label: "web fetch", icon: "🌐", color: "#3ecf7f", desc: "fetch a public URL — reaches outside your docs" },
};

type Props = {
  chunks: { text: string; docName: string; docKind?: string }[];
  tools: AgentTool[];                                              // tools the parent can actually serve
  retrieve: (query: string, tool: AgentTool, k: number) => Promise<AgentHit[]>;
  chat: (messages: ChatMessage[]) => Promise<string>;             // returns full assistant text
  fetchWeb?: (url: string) => Promise<{ title: string; text: string } | null>;
  modelLabel?: string;
  modelPicker?: ReactNode;                                        // provider + model selector, owned by the parent
  note?: ReactNode;                                               // small hint under the tool list (e.g. "build a graph to enable KG")
  defaultQuestion?: string;
  disabled?: boolean;
  compact?: boolean;                                              // tighter layout for the Canvas node
};

// Pull the first balanced {...} object out of an LLM reply (handles ```json fences and stray prose).
function parseAction(raw: string): { thought?: string; action?: string; tool?: string; query?: string; url?: string; answer?: string; citations?: number[] } | null {
  if (!raw) return null;
  const fenced = raw.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const body = fenced ? fenced[1] : raw;
  const start = body.indexOf("{");
  if (start < 0) return null;
  let depth = 0, inStr = false, esc = false;
  for (let i = start; i < body.length; i++) {
    const ch = body[i];
    if (inStr) { if (esc) esc = false; else if (ch === "\\") esc = true; else if (ch === '"') inStr = false; continue; }
    if (ch === '"') inStr = true;
    else if (ch === "{") depth++;
    else if (ch === "}") { depth--; if (depth === 0) { try { return JSON.parse(body.slice(start, i + 1)); } catch { return null; } } }
  }
  return null;
}

const snippet = (s: string, n = 260) => (s.length > n ? s.slice(0, n).trimEnd() + "…" : s);

const CopySvg = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
  </svg>
);

const CheckSvg = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="20 6 9 17 4 12" />
  </svg>
);

function CopyBtn({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const onCopy = (e: React.MouseEvent) => {
    e.stopPropagation();
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };
  return (
    <button className="out-copy-btn" onClick={onCopy} title={copied ? "Copied!" : "Copy to clipboard"}>
      {copied ? <CheckSvg /> : <CopySvg />}
    </button>
  );
}

export default function AgenticAnswer({ chunks, tools, retrieve, chat, fetchWeb, modelLabel, modelPicker, note, defaultQuestion, disabled, compact }: Props) {
  const [question, setQuestion] = useState(defaultQuestion || "What is the refund policy for damaged items?");
  const [enabled, setEnabled] = useState<Set<AgentTool>>(new Set(tools.filter((t) => t !== "web")));
  const [maxSteps, setMaxSteps] = useState(5);
  const [selfCheck, setSelfCheck] = useState(true);
  const [topK, setTopK] = useState(4);
  const [events, setEvents] = useState<TraceEvent[]>([]);
  const [running, setRunning] = useState(false);
  const [final, setFinal] = useState<{ answer: string; citations: number[] } | null>(null);
  const [stats, setStats] = useState<{ steps: number; retrievals: number; requeries: number; ms: number } | null>(null);
  const [openChunk, setOpenChunk] = useState<number | null>(null);
  const cancelRef = useRef(false);

  const activeTools = tools.filter((t) => enabled.has(t));
  const toggle = (t: AgentTool) => setEnabled((s) => { const n = new Set(s); if (n.has(t)) n.delete(t); else n.add(t); return n; });

  async function run() {
    if (running || disabled || !question.trim() || activeTools.length === 0) return;
    cancelRef.current = false;
    setRunning(true); setEvents([]); setFinal(null); setStats(null);
    const t0 = performance.now();
    const trace: TraceEvent[] = [];
    const push = (e: TraceEvent) => { trace.push(e); setEvents([...trace]); };

    const toolList = activeTools.map((t) => `"${t}" (${TOOL_META[t].desc})`).join(", ");
    const sys = [
      "You are a retrieval agent that answers a question using ONLY a document store you search through tools.",
      "On EACH turn reply with a SINGLE JSON object and nothing else, shaped:",
      `{"thought": string, "action": "search" | "answer", "tool"?: string, "query"?: string, "url"?: string, "answer"?: string, "citations"?: number[]}`,
      `- action "search": pick a tool from [${toolList}] and a focused "query" to fetch more context. For the "web" tool put a full URL in "url" instead of a query.`,
      `- action "answer": use ONLY when your observations contain enough evidence. Put the grounded answer in "answer" and the chunk numbers you relied on in "citations" (e.g. [3, 7]).`,
      "- Cite chunk numbers exactly as they appear in observations, e.g. the number 3 for a chunk shown as [chunk 3].",
      selfCheck ? "- Before answering, in your thought briefly grade whether the retrieved chunks actually address every part of the question; if a part is unsupported, search again instead of answering." : "",
      "- If, after searching, the answer is genuinely not in the documents, answer that you don't know rather than guessing.",
      `You may take at most ${maxSteps} search steps.`,
    ].filter(Boolean).join("\n");

    const messages: ChatMessage[] = [{ role: "system", content: sys }, { role: "user", content: `Question: ${question}` }];
    let retrievals = 0, requeries = 0; const usedQueries = new Set<string>();
    const seenIds = new Set<number>();
    let stepNo = 0;

    try {
      for (let iter = 0; iter <= maxSteps; iter++) {
        if (cancelRef.current) { push({ kind: "error", text: "stopped", step: stepNo }); break; }
        const forceAnswer = iter === maxSteps; // last turn: must answer with what it has
        if (forceAnswer) messages.push({ role: "user", content: "You have no search steps left. Answer now with the evidence gathered, citing chunk numbers. If insufficient, say you don't know." });

        const raw = await chat(messages);
        const act = parseAction(raw);
        messages.push({ role: "assistant", content: raw });

        if (!act) { // couldn't parse — treat the reply as the final answer
          stepNo++;
          setFinal({ answer: raw.trim() || "(no answer)", citations: [...seenIds].slice(0, topK) });
          push({ kind: "answer", text: raw.trim(), citations: [...seenIds].slice(0, topK), step: stepNo });
          break;
        }

        if (act.action === "answer" || forceAnswer) {
          stepNo++;
          const cites = (act.citations || []).map((n) => n - 1).filter((i) => i >= 0 && i < chunks.length);
          if (selfCheck && act.thought) push({ kind: "grade", text: act.thought, step: stepNo });
          const ans = (act.answer || raw).trim();
          setFinal({ answer: ans, citations: cites.length ? cites : [...seenIds].slice(0, topK) });
          push({ kind: "answer", text: ans, citations: cites.length ? cites : [...seenIds].slice(0, topK), step: stepNo });
          break;
        }

        // action === "search"
        stepNo++;
        const isReflect = iter > 0;
        if (act.thought) push({ kind: isReflect ? "reflect" : "plan", text: act.thought, step: stepNo });

        let tool = (act.tool as AgentTool) || activeTools[0];
        if (!activeTools.includes(tool)) tool = activeTools[0];

        stepNo++;
        if (tool === "web") {
          const url = act.url || act.query || "";
          if (fetchWeb && /^https?:\/\//i.test(url)) {
            const doc = await fetchWeb(url).catch(() => null);
            push({ kind: "web", url, ok: !!doc, step: stepNo });
            messages.push({ role: "user", content: doc ? `OBSERVATION (web ${url}):\n${snippet(doc.text, 900)}` : `OBSERVATION (web ${url}): fetch failed.` });
          } else {
            push({ kind: "web", url, ok: false, step: stepNo });
            messages.push({ role: "user", content: `OBSERVATION (web): no valid URL provided.` });
          }
          continue;
        }

        const q = (act.query || question).trim();
        if (usedQueries.has(`${tool}:${q}`)) requeries++;
        usedQueries.add(`${tool}:${q}`);
        const hits = await retrieve(q, tool, topK);
        retrievals++;
        hits.forEach((h) => seenIds.add(h.i));
        push({ kind: "tool", tool, query: q, hits, step: stepNo });
        const obs = hits.length
          ? hits.map((h) => `[chunk ${h.i + 1}] (score ${h.score.toFixed(2)}) ${snippet(chunks[h.i]?.text || "")}`).join("\n")
          : "(no chunks matched)";
        messages.push({ role: "user", content: `OBSERVATION (tool=${tool}, query="${q}"):\n${obs}` });
      }
    } catch (e) {
      push({ kind: "error", text: (e as Error).message, step: stepNo + 1 });
    }
    setStats({ steps: stepNo, retrievals, requeries, ms: Math.round(performance.now() - t0) });
    setRunning(false);
  }

  const card = { border: "1px solid var(--border)", borderRadius: 12, background: "var(--panel)" } as const;
  const dotFor = (e: TraceEvent) => e.kind === "plan" ? "#a855f7" : e.kind === "reflect" ? "#f59e0b" : e.kind === "tool" ? (TOOL_META[e.tool].color) : e.kind === "web" ? "#3ecf7f" : e.kind === "grade" ? "#3ecf7f" : e.kind === "answer" ? "var(--good)" : "var(--crit)";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      {/* setup */}
      <div style={{ ...card, padding: compact ? 12 : 15 }}>
        <div style={{ fontSize: 10.5, fontFamily: "var(--mono)", textTransform: "uppercase", letterSpacing: ".06em", color: "var(--muted)", marginBottom: 12, display: "flex", alignItems: "center", gap: 7 }}>
          <span style={{ width: 7, height: 7, borderRadius: "50%", background: "#a855f7" }} />agent setup{!modelPicker && modelLabel ? <span style={{ marginLeft: "auto", color: "var(--faint)" }}>{modelLabel}</span> : null}
        </div>
        {modelPicker && (
          <div style={{ marginBottom: 14, paddingBottom: 14, borderBottom: "1px solid var(--border)" }}>
            <div className="note" style={{ marginBottom: 7 }}>Generation model — the LLM that runs the agent loop</div>
            {modelPicker}
          </div>
        )}
        <div style={{ display: "grid", gridTemplateColumns: compact ? "1fr" : "1.1fr 1fr", gap: 16 }}>
          <div>
            <div className="note" style={{ marginBottom: 7 }}>Tools the agent may call</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {tools.map((t) => {
                const on = enabled.has(t); const m = TOOL_META[t];
                return (
                  <button key={t} onClick={() => toggle(t)} style={{ display: "flex", alignItems: "center", gap: 9, fontSize: 12, padding: "7px 10px", borderRadius: 8, cursor: "pointer", textAlign: "left", border: `1px solid ${on ? m.color : "var(--border)"}`, background: on ? `color-mix(in srgb, ${m.color} 12%, transparent)` : "var(--surface)", color: on ? "var(--text)" : "var(--muted)" }}>
                    <span style={{ fontSize: 13, flex: "0 0 auto" }}>{m.icon}</span>
                    <span style={{ display: "flex", flexDirection: "column", minWidth: 0, flex: 1, gap: 1 }}>
                      <b style={{ fontWeight: 600 }}>{m.label}</b>
                      <span style={{ fontSize: 10, fontWeight: 400, color: "var(--faint)", lineHeight: 1.3 }}>{m.desc}</span>
                    </span>
                    <span style={{ flex: "0 0 auto", color: on ? m.color : "var(--faint)" }}>{on ? "✓" : "○"}</span>
                  </button>
                );
              })}
            </div>
            {note && <div style={{ marginTop: 8 }}>{note}</div>}
          </div>
          <div>
            <div className="row" style={{ justifyContent: "space-between", fontSize: 12, color: "var(--muted)", marginBottom: 6 }}><span>Max steps</span><b style={{ fontFamily: "var(--mono)", color: "var(--text)" }}>{maxSteps}</b></div>
            <input type="range" min={1} max={8} step={1} value={maxSteps} onChange={(e) => setMaxSteps(+e.target.value)} style={{ width: "100%" }} />
            <div className="row" style={{ justifyContent: "space-between", fontSize: 12, color: "var(--muted)", margin: "12px 0 6px" }}><span>Chunks per retrieval (k)</span><b style={{ fontFamily: "var(--mono)", color: "var(--text)" }}>{topK}</b></div>
            <input type="range" min={1} max={8} step={1} value={topK} onChange={(e) => setTopK(+e.target.value)} style={{ width: "100%" }} />
            <label style={{ display: "flex", alignItems: "center", gap: 7, fontSize: 12, color: "var(--muted)", marginTop: 12, cursor: "pointer" }}>
              <input type="checkbox" checked={selfCheck} onChange={(e) => setSelfCheck(e.target.checked)} style={{ width: "auto" }} />self-check retrieved chunks before answering
            </label>
          </div>
        </div>
      </div>

      {/* ask bar */}
      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
        <input type="text" value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="Ask a question…" onKeyDown={(e) => { if (e.key === "Enter") run(); }} style={{ flex: 1 }} />
        {running
          ? <button className="btn ghost" onClick={() => { cancelRef.current = true; }}>■ Stop</button>
          : <button className="btn" onClick={run} disabled={disabled || !question.trim() || activeTools.length === 0}>▶ Run agent</button>}
      </div>
      {activeTools.length === 0 && <div className="note" style={{ color: "var(--warn)" }}>Enable at least one tool above.</div>}

      {/* trace */}
      {(events.length > 0 || running) && (
        <div>
          <div className="note" style={{ marginBottom: 9, textTransform: "uppercase", letterSpacing: ".05em", fontSize: 10 }}>reasoning trace{stats ? ` · ${stats.steps} steps` : running ? " · running…" : ""}</div>
          <div style={{ position: "relative", paddingLeft: 22 }}>
            <div style={{ position: "absolute", left: 6, top: 6, bottom: 6, width: 2, background: "var(--border)" }} />
            {events.map((e, idx) => (
              <div key={idx} style={{ position: "relative", paddingBottom: idx === events.length - 1 ? 0 : 12, animation: "reveal-in .3s ease both" }}>
                <span style={{ position: "absolute", left: -22, top: 2, width: 14, height: 14, borderRadius: "50%", background: `color-mix(in srgb, ${dotFor(e)} 22%, transparent)`, border: `2px solid ${dotFor(e)}` }} />
                {e.kind === "tool" ? (
                  <div>
                    <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                      <span style={{ fontSize: 10.5, fontFamily: "var(--mono)", color: TOOL_META[e.tool].color, background: `color-mix(in srgb, ${TOOL_META[e.tool].color} 14%, transparent)`, borderRadius: 5, padding: "2px 7px" }}>tool · {TOOL_META[e.tool].label}</span>
                      <span style={{ fontSize: 11.5, fontFamily: "var(--mono)", color: "var(--muted)" }}>“{e.query}”</span>
                    </div>
                    <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 7 }}>
                      {e.hits.length ? e.hits.map((h) => (
                        <button key={h.i} onClick={() => setOpenChunk(openChunk === h.i ? null : h.i)} title="show chunk text" style={{ fontSize: 11, padding: "3px 8px", borderRadius: 6, cursor: "pointer", background: "var(--surface)", border: "1px solid var(--border)", color: "var(--text)" }}>
                          c{h.i + 1} <span style={{ color: TOOL_META[e.tool].color, fontFamily: "var(--mono)" }}>{h.score.toFixed(2)}</span>
                        </button>
                      )) : <span className="note">no chunks matched</span>}
                    </div>
                    {e.hits.some((h) => h.i === openChunk) && openChunk != null && (
                      <div style={{ marginTop: 7, fontSize: 11.5, color: "var(--muted)", fontFamily: "var(--mono)", lineHeight: 1.5, background: "var(--panel-2)", border: "1px solid var(--border)", borderRadius: 8, padding: "8px 10px" }}>
                        <b style={{ color: "var(--text)" }}>chunk {openChunk + 1}</b> · {chunks[openChunk]?.docName}<br />{snippet(chunks[openChunk]?.text || "", 400)}
                      </div>
                    )}
                  </div>
                ) : e.kind === "web" ? (
                  <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                    <span style={{ fontSize: 10.5, fontFamily: "var(--mono)", color: "#3ecf7f", background: "rgba(62,207,127,.14)", borderRadius: 5, padding: "2px 7px" }}>web fetch</span>
                    <span style={{ fontSize: 11.5, fontFamily: "var(--mono)", color: "var(--muted)" }}>{e.url}</span>
                    <span style={{ fontSize: 11, color: e.ok ? "var(--good)" : "var(--crit)" }}>{e.ok ? "✓ fetched" : "✗ failed"}</span>
                  </div>
                ) : e.kind === "answer" ? (
                  <div>
                    <span style={{ fontSize: 10.5, fontFamily: "var(--mono)", color: "var(--good)", background: "color-mix(in srgb, var(--good) 14%, transparent)", borderRadius: 5, padding: "2px 7px" }}>answer</span>
                  </div>
                ) : e.kind === "error" ? (
                  <div style={{ fontSize: 12, color: "var(--crit)" }}><b>error</b> · {e.text}</div>
                ) : (
                  <div>
                    <span style={{ fontSize: 10.5, fontFamily: "var(--mono)", color: dotFor(e), background: `color-mix(in srgb, ${dotFor(e)} 14%, transparent)`, borderRadius: 5, padding: "2px 7px" }}>{e.kind}</span>
                    <div style={{ fontSize: 12.5, marginTop: 6, color: "var(--text)", lineHeight: 1.5 }}>{e.text}</div>
                  </div>
                )}
              </div>
            ))}
            {running && <div style={{ position: "relative", paddingTop: 4 }}><span style={{ position: "absolute", left: -22, top: 4, width: 14, height: 14, borderRadius: "50%", background: "var(--panel-2)", border: "2px solid var(--border)" }} /><span className="busy-dot" /> <span className="note">thinking…</span></div>}
          </div>
        </div>
      )}

      {/* final answer */}
      {final && (
        <div style={{ border: "1px solid color-mix(in srgb, var(--good) 45%, var(--border))", borderRadius: 12, background: "color-mix(in srgb, var(--good) 8%, transparent)", padding: "14px 16px", position: "relative" }}>
          <CopyBtn text={final.answer} />
          <div style={{ fontSize: 10.5, fontFamily: "var(--mono)", textTransform: "uppercase", letterSpacing: ".06em", color: "var(--good)", marginBottom: 8 }}>◆ answer</div>
          <div style={{ fontSize: 14, lineHeight: 1.65, color: "var(--text)", whiteSpace: "pre-wrap", paddingRight: 40 }}>{final.answer}</div>
          {final.citations.length > 0 && (
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 12, paddingTop: 11, borderTop: "1px solid color-mix(in srgb, var(--good) 30%, var(--border))" }}>
              <span className="note">sources:</span>
              {final.citations.map((i) => <span key={i} style={{ fontSize: 11, fontFamily: "var(--mono)", padding: "2px 7px", borderRadius: 5, background: "var(--surface)", border: "1px solid var(--border)" }}>{chunks[i]?.docName || "doc"} · c{i + 1}</span>)}
              {stats && <span className="note" style={{ marginLeft: "auto" }}>{stats.retrievals} retrieval{stats.retrievals === 1 ? "" : "s"} · {stats.steps} steps{stats.requeries ? ` · ${stats.requeries} re-query` : ""} · {stats.ms}ms</span>}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
