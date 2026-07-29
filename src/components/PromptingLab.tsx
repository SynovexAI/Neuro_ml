"use client";

import { useEffect, useRef, useState } from "react";

type Msg = { role: string; content: string };

async function streamChat(body: object, onToken: (full: string) => void): Promise<{ error?: string; text: string; ms: number }> {
  const t0 = performance.now();
  const res = await fetch("/api/chat", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  if (!res.ok || !res.body) {
    const j = await res.json().catch(() => ({ error: "Request failed" }));
    return { error: j.error || "Request failed", text: "", ms: 0 };
  }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let text = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    text += dec.decode(value, { stream: true });
    onToken(text);
  }
  return { text, ms: Math.round(performance.now() - t0) };
}

export default function PromptingLab() {
  const [system, setSystem] = useState("You are a concise assistant. Answer in one short paragraph.");
  const [prompt, setPrompt] = useState("Explain what RAG is to a beginner, using a library analogy.");
  const [temp, setTemp] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(512);
  const [topP, setTopP] = useState(0.95);
  const [model, setModel] = useState("");
  const [models, setModels] = useState<string[]>([]);
  const [provider, setProvider] = useState<string | null>(null);
  const [outA, setOutA] = useState("Press Run to stream a response…");
  const [outB, setOutB] = useState("A second variant streams here for comparison…");
  const [metaA, setMetaA] = useState("idle");
  const [metaB, setMetaB] = useState("idle");
  const [tab, setTab] = useState<"out" | "trace">("out");
  const [trace, setTrace] = useState<{ who: string; what: string; ms: string; state: string }[]>([]);
  const [running, setRunning] = useState(false);
  const [showCode, setShowCode] = useState(false);
  const [saved, setSaved] = useState("");
  const traceTimer = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    fetch("/api/models").then((r) => r.json()).then((j) => {
      setProvider(j.provider);
      setModels(j.models || []);
      setModel(j.default || j.models?.[0] || "");
    }).catch(() => {});
  }, []);

  function playTrace() {
    const steps = [
      { who: "build", what: "assemble system + user messages", ms: "2ms" },
      { who: "POST /api/chat", what: `stream → ${provider || "provider"}`, ms: "—" },
      { who: "stream", what: "tokens arriving", ms: "…" },
      { who: "done", what: "response complete", ms: "—" },
    ].map((s) => ({ ...s, state: "" }));
    setTrace(steps);
    let i = 0;
    if (traceTimer.current) clearInterval(traceTimer.current);
    traceTimer.current = setInterval(() => {
      setTrace((prev) => prev.map((s, idx) => idx < i ? { ...s, state: "done" } : idx === i ? { ...s, state: "active" } : s));
      i++;
      if (i > steps.length && traceTimer.current) { clearInterval(traceTimer.current); setTrace((prev) => prev.map((s) => ({ ...s, state: "done" }))); }
    }, 500);
  }

  async function run(compare: boolean) {
    if (!model) { setOutA("No model available. Ask an admin to configure a provider in Admin → Providers."); return; }
    setRunning(true); playTrace();
    const messages: Msg[] = [{ role: "system", content: system }, { role: "user", content: prompt }];
    setMetaA("streaming…"); setOutA("");
    const a = streamChat({ messages, model, temperature: temp, maxTokens, topP }, (t) => setOutA(t));
    let b: Promise<{ error?: string; text: string; ms: number }> | null = null;
    if (compare) {
      setMetaB("streaming…"); setOutB("");
      b = streamChat({ messages, model, temperature: Math.min(1.2, temp + 0.4), maxTokens, topP }, (t) => setOutB(t));
    }
    const ra = await a;
    if (ra.error) { setOutA("⚠ " + ra.error); setMetaA("error"); } else { setMetaA(`~${Math.round(ra.text.length / 4)} tok · ${ra.ms}ms`); }
    if (b) { const rb = await b; if (rb.error) { setOutB("⚠ " + rb.error); setMetaB("error"); } else setMetaB(`~${Math.round(rb.text.length / 4)} tok · ${rb.ms}ms · temp ${Math.min(1.2, temp + 0.4).toFixed(2)}`); }
    setRunning(false);
  }

  async function save() {
    const r = await fetch("/api/projects", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ lab: "prompting", name: prompt.slice(0, 60), config: { system, prompt, temp, maxTokens, topP, model } }) });
    setSaved(r.ok ? "Saved to My Projects ✓" : "Save failed");
    setTimeout(() => setSaved(""), 2500);
  }

  const code = `from openai import OpenAI
client = OpenAI(base_url=PROVIDER_BASE_URL, api_key=KEY)
r = client.chat.completions.create(
    model="${model || "your-model"}", temperature=${temp}, top_p=${topP}, max_tokens=${maxTokens},
    messages=[
        {"role": "system", "content": ${JSON.stringify(system)}},
        {"role": "user", "content": ${JSON.stringify(prompt)}},
    ], stream=True)
for chunk in r:
    print(chunk.choices[0].delta.content or "", end="")`;

  return (
    <>
      <div className="lab-head">
        <div>
          <div className="eyebrow">Lab 01 · warm-up</div>
          <h2 className="page-h">Prompting Lab</h2>
          <p className="page-sub" style={{ margin: 0 }}>Write a prompt, tune the knobs, and stream a real response. Run two variants side-by-side to compare.</p>
        </div>
        <div className="acts">
          <button className="btn ghost sm" onClick={() => setShowCode(true)}>&lt;/&gt; Get code</button>
          <button className="btn ghost sm" onClick={save}>Save</button>
        </div>
      </div>

      {saved && <div className="ok">{saved}</div>}
      {provider === null && <div className="warnbar">No provider is configured yet — an admin needs to add one under Admin → Providers before generation works.</div>}

      <div className="split col-2" style={{ gridTemplateColumns: "1.7fr 1fr" }}>
        <div className="card">
          <div className="card-h"><span className="t">Prompt editor</span><span className="mono r">→ /api/chat</span></div>
          <div className="card-b">
            <label className="fld">System</label>
            <textarea rows={2} value={system} onChange={(e) => setSystem(e.target.value)} />
            <div style={{ height: 12 }} />
            <label className="fld">User</label>
            <textarea rows={3} value={prompt} onChange={(e) => setPrompt(e.target.value)} />
            <div style={{ height: 14 }} />
            <div className="row">
              <button className="btn" onClick={() => run(false)} disabled={running}>▶ Run</button>
              <button className="btn ghost" onClick={() => run(true)} disabled={running}>Run &amp; compare (A/B)</button>
            </div>
          </div>
        </div>
        <div className="card">
          <div className="card-h"><span className="t">Knobs</span></div>
          <div className="card-b">
            <div className="knob"><div className="kr"><span>Temperature</span><b>{temp.toFixed(2)}</b></div><input type="range" min={0} max={1.2} step={0.05} value={temp} onChange={(e) => setTemp(+e.target.value)} /></div>
            <div className="knob"><div className="kr"><span>Max tokens</span><b>{maxTokens}</b></div><input type="range" min={64} max={2048} step={64} value={maxTokens} onChange={(e) => setMaxTokens(+e.target.value)} /></div>
            <div className="knob"><div className="kr"><span>Top-p</span><b>{topP.toFixed(2)}</b></div><input type="range" min={0} max={1} step={0.05} value={topP} onChange={(e) => setTopP(+e.target.value)} /></div>
            <label className="fld">Model {provider ? `· ${provider}` : ""}</label>
            <select value={model} onChange={(e) => setModel(e.target.value)}>
              {models.length === 0 && <option value="">(no models — configure a provider)</option>}
              {models.map((m) => <option key={m} value={m}>{m}</option>)}
            </select>
          </div>
        </div>
      </div>

      <div style={{ height: 18 }} />
      <div className="split col-2e">
        <div className="card">
          <div className="card-h"><span className="t">Output A</span><span className="mono">{metaA}</span>
            <div className="tabs"><button className={tab === "out" ? "on" : ""} onClick={() => setTab("out")}>Output</button><button className={tab === "trace" ? "on" : ""} onClick={() => setTab("trace")}>Trace</button></div>
          </div>
          <div className="card-b">
            {tab === "out"
              ? <div className="out">{outA}{running && <span className="cur" />}</div>
              : <div className="tracebox">{trace.length === 0 ? "Run to see the execution trace." : trace.map((s, i) => <div key={i} className={`trow ${s.state}`}><span className="who">{s.who}</span><span>{s.what}</span><span className="ms">{s.ms}</span></div>)}</div>}
          </div>
        </div>
        <div className="card">
          <div className="card-h"><span className="t">Output B · compare</span><span className="mono">{metaB}</span></div>
          <div className="card-b"><div className="out">{outB}</div></div>
        </div>
      </div>

      <div className={`modal-wrap ${showCode ? "show" : ""}`} onClick={(e) => { if (e.target === e.currentTarget) setShowCode(false); }}>
        <div className="modal">
          <div className="mh"><b>Get the code · Prompting</b><button className="x" onClick={() => setShowCode(false)}>×</button></div>
          <div className="mb"><div className="code">{code}</div></div>
        </div>
      </div>
    </>
  );
}
