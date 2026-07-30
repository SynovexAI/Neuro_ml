"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { genDataset, initNet, trainEpochs, predict, DATA_BOUND, type Net } from "@/lib/dlUtils";

const DATASETS = [{ k: "spiral", l: "Spiral" }, { k: "circles", l: "Circles" }, { k: "xor", l: "XOR" }, { k: "moons", l: "Moons" }];
const BND = DATA_BOUND + 1;
const MAX_EPOCHS = 600;

export default function DlLab() {
  const [kind, setKind] = useState("spiral");
  const [noise, setNoise] = useState(0.15);
  const [seed, setSeed] = useState(3);
  const [hidden, setHidden] = useState<number[]>([8, 6]);
  const [act, setAct] = useState("tanh");
  const [lr, setLr] = useState(0.3);
  const [l2, setL2] = useState(0);

  const [running, setRunning] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const [loss, setLoss] = useState<number | null>(null);
  const [acc, setAcc] = useState<number | null>(null);
  const [lossHist, setLossHist] = useState<number[]>([]);
  const [tick, setTick] = useState(0);
  const [testPt, setTestPt] = useState<{ x: number; y: number; p: number } | null>(null);
  const [showCode, setShowCode] = useState(false);
  const [copied, setCopied] = useState(false);
  const [savedMsg, setSavedMsg] = useState("");

  // Load a saved build when opened from My Projects (?project=<id>).
  useEffect(() => {
    const id = new URLSearchParams(window.location.search).get("project");
    if (!id) return;
    fetch(`/api/projects?id=${id}`).then((r) => r.json()).then(({ project }) => {
      const c = project?.config; if (!c) return;
      if (c.kind) setKind(c.kind);
      if (c.noise != null) setNoise(c.noise);
      if (Array.isArray(c.hidden)) setHidden(c.hidden);
      if (c.act) setAct(c.act);
      if (c.lr != null) setLr(c.lr);
      if (c.l2 != null) setL2(c.l2);
    }).catch(() => {});
  }, []);

  const data = useMemo(() => genDataset(kind, 220, noise, seed), [kind, noise, seed]);
  const netRef = useRef<Net | null>(null);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);
  const epochRef = useRef(0);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const hiddenKey = hidden.join(",");

  function stop() { setRunning(false); if (timer.current) { clearInterval(timer.current); timer.current = null; } }
  function resetNet() {
    stop(); netRef.current = initNet(hidden, act); epochRef.current = 0;
    setEpoch(0); setLoss(null); setAcc(null); setLossHist([]); setTestPt(null); setTick((t) => t + 1);
  }
  // rebuild net whenever data or architecture changes
  useEffect(() => {
    resetNet();
    return () => { if (timer.current) clearInterval(timer.current); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, hiddenKey, act]);

  function train() {
    if (running || !netRef.current) return;
    setRunning(true);
    timer.current = setInterval(() => {
      if (!netRef.current) return;
      const m = trainEpochs(netRef.current, data, { lr, l2, epochs: 3 });
      epochRef.current += 3; setEpoch(epochRef.current); setLoss(m.loss); setAcc(m.acc);
      setLossHist((h) => [...h, m.loss]); setTick((t) => t + 1);
      if (epochRef.current >= MAX_EPOCHS) stop();
    }, 45);
  }

  // ── draw decision boundary + points ──
  useEffect(() => {
    const cv = canvasRef.current; const net = netRef.current; if (!cv || !net) return;
    const ctx = cv.getContext("2d")!; const W = cv.width, H = cv.height; const S = 8;
    const c0 = [91, 124, 255], c1 = [245, 158, 11];
    const toPx = (dx: number, dy: number): [number, number] => [((dx + BND) / (2 * BND)) * W, ((BND - dy) / (2 * BND)) * H];
    for (let px = 0; px < W; px += S) for (let py = 0; py < H; py += S) {
      const dx = ((px + S / 2) / W) * 2 * BND - BND, dy = BND - ((py + S / 2) / H) * 2 * BND;
      const p = predict(net, [dx, dy]);
      const r = Math.round(c0[0] + (c1[0] - c0[0]) * p), g = Math.round(c0[1] + (c1[1] - c0[1]) * p), b = Math.round(c0[2] + (c1[2] - c0[2]) * p);
      ctx.fillStyle = `rgba(${r},${g},${b},0.42)`; ctx.fillRect(px, py, S, S);
    }
    for (let i = 0; i < data.X.length; i++) { const [px, py] = toPx(data.X[i][0], data.X[i][1]); ctx.beginPath(); ctx.arc(px, py, 3.4, 0, 7); ctx.fillStyle = data.y[i] ? `rgb(${c1.join(",")})` : `rgb(${c0.join(",")})`; ctx.fill(); ctx.lineWidth = 1; ctx.strokeStyle = "rgba(255,255,255,.7)"; ctx.stroke(); }
    if (testPt) { const [px, py] = toPx(testPt.x, testPt.y); ctx.beginPath(); ctx.arc(px, py, 7, 0, 7); ctx.fillStyle = "#fff"; ctx.fill(); ctx.lineWidth = 3; ctx.strokeStyle = testPt.p > 0.5 ? `rgb(${c1.join(",")})` : `rgb(${c0.join(",")})`; ctx.stroke(); }
  }, [tick, data, testPt]);

  function onCanvasClick(e: React.MouseEvent<HTMLCanvasElement>) {
    const cv = canvasRef.current, net = netRef.current; if (!cv || !net) return;
    const r = cv.getBoundingClientRect(); const px = (e.clientX - r.left) * (cv.width / r.width), py = (e.clientY - r.top) * (cv.height / r.height);
    const dx = (px / cv.width) * 2 * BND - BND, dy = BND - (py / cv.height) * 2 * BND;
    setTestPt({ x: dx, y: dy, p: predict(net, [dx, dy]) });
  }

  function buildCode(): string {
    const acts: Record<string, string> = { relu: "nn.ReLU()", tanh: "nn.Tanh()", sigmoid: "nn.Sigmoid()" };
    const sizes = [2, ...hidden, 1];
    const layers: string[] = [];
    for (let l = 0; l < sizes.length - 1; l++) { layers.push(`    nn.Linear(${sizes[l]}, ${sizes[l + 1]}),`); if (l < sizes.length - 2) layers.push(`    ${acts[act]},`); }
    layers.push(`    nn.Sigmoid(),`);
    return `# AI Workbench · DL Lab — MLP for 2-D classification (PyTorch)
import torch, torch.nn as nn

model = nn.Sequential(
${layers.join("\n")}
)
opt = torch.optim.SGD(model.parameters(), lr=${lr}, weight_decay=${l2})
loss_fn = nn.BCELoss()

# X: (N,2) float tensor, y: (N,1) float tensor of 0/1  (dataset: ${kind})
for epoch in range(${MAX_EPOCHS}):
    opt.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward(); opt.step()

acc = ((model(X) > 0.5).float() == y).float().mean()
print("accuracy:", acc.item())`;
  }
  function copyCode() { navigator.clipboard.writeText(buildCode()).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500); }); }
  async function saveProject() {
    const config = { kind, noise, hidden, act, lr, l2 };
    try { const r = await fetch("/api/projects", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ lab: "dl", name: `${kind} net`, config }) }); setSavedMsg(r.ok ? "Saved ✓" : "Save failed"); }
    catch { setSavedMsg("Save failed"); }
    setTimeout(() => setSavedMsg(""), 2000);
  }

  const sizes = [2, ...hidden, 1];
  // network diagram layout
  const diagW = 260, diagH = 150, colGap = diagW / (sizes.length - 1 || 1);
  const neuronY = (count: number, i: number) => { const shown = Math.min(count, 8); const gap = diagH / (shown + 1); return gap * (i + 1); };

  return (
    <>
      <div className="lab-head">
        <div><div className="eyebrow">Lab 05 · runs in your browser</div><h2 className="page-h">DL Lab</h2><p className="page-sub" style={{ margin: 0 }}>Design a neural net, watch it learn a decision boundary live via real backprop, then click anywhere to test it.</p></div>
        <div className="acts"><button className="btn ghost sm" onClick={saveProject}>{savedMsg || "💾 Save"}</button><button className="btn ghost sm" onClick={() => setShowCode(true)}>&lt;/&gt; Get code (PyTorch)</button></div>
      </div>
      <div className="teach-note"><span className="ic">🎓</span><span><b>Teaching engine.</b> This is a real MLP with forward + backprop running live in your browser, kept small so you can watch it learn. For production, use the <b>Get code (PyTorch)</b> export.</span></div>

      <div className="split" style={{ gridTemplateColumns: "320px 1fr" }}>
        <div className="card">
          <div className="card-h"><span className="t">Data &amp; network</span></div>
          <div className="card-b">
            <label className="fld">Dataset</label>
            <div className="checklist">{DATASETS.map((d) => <span key={d.k} className={`chk ${kind === d.k ? "on" : ""}`} onClick={() => setKind(d.k)}>{d.l}</span>)}</div>
            <div className="row" style={{ gap: 12, marginTop: 12, flexWrap: "wrap" }}>
              <div className="knob" style={{ margin: 0, minWidth: 150 }}><div className="kr"><span>Noise</span><b>{noise.toFixed(2)}</b></div><input type="range" min={0} max={0.5} step={0.01} value={noise} onChange={(e) => setNoise(+e.target.value)} /></div>
              <button className="btn ghost sm" onClick={() => setSeed((s) => s + 1)}>↻ Regenerate</button>
            </div>

            <label className="fld" style={{ marginTop: 14 }}>Hidden layers ({hidden.length})</label>
            {hidden.map((h, i) => (
              <div key={i} className="knob" style={{ margin: "0 0 8px" }}>
                <div className="kr"><span>layer {i + 1} · {h} neurons</span><span className="rm" style={{ cursor: "pointer", color: "var(--faint)" }} onClick={() => setHidden((hs) => hs.filter((_, j) => j !== i))}>×</span></div>
                <input type="range" min={1} max={8} value={h} onChange={(e) => setHidden((hs) => hs.map((v, j) => (j === i ? +e.target.value : v)))} />
              </div>
            ))}
            {hidden.length < 4 && <button className="btn ghost sm" onClick={() => setHidden((hs) => [...hs, 4])}>+ Add layer</button>}

            <div className="split col-2e" style={{ marginTop: 14 }}>
              <div><label className="fld">Activation</label><select value={act} onChange={(e) => setAct(e.target.value)}><option value="tanh">tanh</option><option value="relu">relu</option><option value="sigmoid">sigmoid</option></select></div>
              <div><label className="fld">Learning rate · {lr}</label><input type="range" min={0.01} max={1} step={0.01} value={lr} onChange={(e) => setLr(+e.target.value)} /></div>
            </div>
            <div style={{ marginTop: 12 }}><label className="fld">L2 regularization · {l2}</label><input type="range" min={0} max={0.05} step={0.001} value={l2} onChange={(e) => setL2(+e.target.value)} /></div>

            <label className="fld" style={{ marginTop: 16 }}>Network</label>
            <svg width="100%" viewBox={`0 0 ${diagW + 30} ${diagH}`} className="net-diagram">
              {sizes.map((c, l) => l < sizes.length - 1 && Array.from({ length: Math.min(c, 8) }, (_, i) => Array.from({ length: Math.min(sizes[l + 1], 8) }, (_, j) => (
                <line key={`${l}-${i}-${j}`} x1={15 + l * colGap} y1={neuronY(c, i)} x2={15 + (l + 1) * colGap} y2={neuronY(sizes[l + 1], j)} stroke="var(--border-strong)" strokeWidth={0.5} opacity={0.6} />
              )))).flat(2)}
              {sizes.map((c, l) => Array.from({ length: Math.min(c, 8) }, (_, i) => (
                <circle key={`n${l}-${i}`} cx={15 + l * colGap} cy={neuronY(c, i)} r={5} fill={l === 0 ? "var(--sky)" : l === sizes.length - 1 ? "var(--good)" : "var(--accent)"} />
              )))}
            </svg>
            <div className="note">{sizes.join(" → ")} · {act}</div>
          </div>
        </div>

        <div className="card">
          <div className="card-h"><span className="t">Training</span>
            <div className="r">
              <button className="btn sm" onClick={running ? stop : train}>{running ? "⏸ Pause" : epoch >= MAX_EPOCHS ? "done" : "▶ Train"}</button>
              <button className="btn ghost sm" onClick={resetNet}>↻ Reset</button>
            </div>
          </div>
          <div className="card-b">
            <div className="dl-metrics">
              <div className="m"><span className="v">{epoch}</span><span className="k">epoch</span></div>
              <div className="m"><span className="v">{loss != null ? loss.toFixed(3) : "—"}</span><span className="k">loss</span></div>
              <div className="m"><span className="v">{acc != null ? `${Math.round(acc * 100)}%` : "—"}</span><span className="k">accuracy</span></div>
            </div>
            <div className="split col-2e" style={{ alignItems: "start", marginTop: 6 }}>
              <div>
                <div className="dl-canvas-wrap"><canvas ref={canvasRef} width={320} height={320} className="dl-canvas" onClick={onCanvasClick} /></div>
                <div className="note" style={{ marginTop: 6 }}>{testPt ? <>Tested point → <b style={{ color: testPt.p > 0.5 ? "var(--warn)" : "var(--accent)" }}>class {testPt.p > 0.5 ? 1 : 0}</b> (p={testPt.p.toFixed(2)})</> : "Click anywhere on the map to classify that point."}</div>
              </div>
              <div>
                <label className="fld">Training loss</label>
                <svg width="100%" viewBox="0 0 240 120" className="spark" preserveAspectRatio="none">
                  {lossHist.length > 1 && (() => { const mx = Math.max(...lossHist), mn = Math.min(...lossHist), sp = (mx - mn) || 1; const pts = lossHist.map((v, i) => `${(i / (lossHist.length - 1)) * 240},${116 - ((v - mn) / sp) * 108 - 4}`).join(" "); return <polyline points={pts} className="spark-line" />; })()}
                </svg>
                <div className="note" style={{ marginTop: 4 }}>{lossHist.length ? `start ${lossHist[0].toFixed(2)} → now ${lossHist[lossHist.length - 1].toFixed(3)}` : "train to see the loss fall"}</div>
                <div className="note" style={{ marginTop: 10, lineHeight: 1.5 }}>Blue = class 0, orange = class 1. The shaded map is the network&apos;s prediction everywhere; watch the boundary bend to fit the data as backprop runs.</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className={`modal-wrap ${showCode ? "show" : ""}`} onClick={(e) => { if (e.target === e.currentTarget) setShowCode(false); }}>
        <div className="modal"><div className="mh"><b>DL model · PyTorch</b><div className="r" style={{ marginLeft: "auto", display: "flex", gap: 8 }}><button className="btn ghost sm" onClick={copyCode}>{copied ? "Copied ✓" : "Copy"}</button></div><button className="x" onClick={() => setShowCode(false)}>×</button></div><div className="mb"><div className="code">{buildCode()}</div></div></div>
      </div>
    </>
  );
}
