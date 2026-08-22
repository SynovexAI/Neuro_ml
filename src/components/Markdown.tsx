"use client";

import { useState, type ReactNode } from "react";

// Lightweight, safe markdown renderer (React elements, no dangerouslySetInnerHTML).
// Handles headings, bold/italic, inline code, links, lists, tables, blockquotes,
// and fenced code — with html/svg/mermaid blocks rendered as sandboxed artifacts.

function inline(text: string, key: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  const re = /(\*\*(.+?)\*\*|__(.+?)__|\*(.+?)\*|`(.+?)`|\[([^\]]+)\]\s*\((https?:\/\/[^\s)]+)\)|(https?:\/\/[^\s<>)"]+))/g;
  let last = 0, m: RegExpExecArray | null, i = 0;
  while ((m = re.exec(text))) {
    if (m.index > last) nodes.push(text.slice(last, m.index));
    if (m[2] || m[3]) {
      nodes.push(<b key={`${key}-${i}`}>{m[2] || m[3]}</b>);
    } else if (m[4]) {
      nodes.push(<i key={`${key}-${i}`}>{m[4]}</i>);
    } else if (m[5]) {
      nodes.push(<code key={`${key}-${i}`} className="md-code">{m[5]}</code>);
    } else if (m[6] && m[7]) {
      nodes.push(
        <a
          key={`${key}-${i}`}
          href={m[7]}
          target="_blank"
          rel="noopener noreferrer"
          className="md-link touch-link"
          title={`Direct link to ${m[7]}`}
        >
          {m[6]}
          <span className="link-icon" style={{ marginLeft: 3, opacity: 0.75, fontSize: "0.85em" }}>↗</span>
        </a>
      );
    } else if (m[8]) {
      let url = m[8];
      let trailing = "";
      const matchTrail = url.match(/[.,;!]+$/);
      if (matchTrail) {
        trailing = matchTrail[0];
        url = url.slice(0, -trailing.length);
      }
      nodes.push(
        <a
          key={`${key}-${i}`}
          href={url}
          target="_blank"
          rel="noopener noreferrer"
          className="md-link touch-link"
          title={`Direct link to ${url}`}
        >
          {url}
          <span className="link-icon" style={{ marginLeft: 3, opacity: 0.75, fontSize: "0.85em" }}>↗</span>
        </a>
      );
      if (trailing) nodes.push(trailing);
    }
    last = m.index + m[0].length;
    i++;
  }
  if (last < text.length) nodes.push(text.slice(last));
  return nodes;
}

function Artifact({ lang, code }: { lang: string; code: string }) {
  const renderable = lang === "html" || lang === "svg";
  const [view, setView] = useState<"preview" | "code">(renderable ? "preview" : "code");
  const doc = lang === "svg" ? `<!doctype html><body style="margin:0;display:grid;place-items:center">${code}</body>` : code;
  return (
    <div className="md-artifact">
      <div className="md-artifact-h"><span className="badge">{lang || "code"}</span>{renderable && <div className="seg" style={{ width: 150, marginLeft: "auto" }}><button className={view === "preview" ? "on" : ""} onClick={() => setView("preview")}>Preview</button><button className={view === "code" ? "on" : ""} onClick={() => setView("code")}>Code</button></div>}</div>
      {renderable && view === "preview"
        ? <iframe title="artifact" sandbox="allow-scripts" srcDoc={doc} style={{ width: "100%", height: 320, border: "none", background: "#fff", borderRadius: "0 0 8px 8px" }} />
        : <pre className="md-pre"><code>{code}</code></pre>}
    </div>
  );
}

const PALETTE = ["var(--accent)", "#f59e0b", "#3b9e5f", "#8b5cf6", "#ef4444", "#0ea5e9", "#ec4899", "#14b8a6"];

// Native SVG chart from a ```chart JSON block: {type:'bar'|'line'|'pie', title?, data:[{label,value}]}
function ChartArtifact({ code }: { code: string }) {
  let spec: { type?: string; title?: string; data?: { label: string; value: number }[] } | null = null;
  let err = "";
  try { spec = JSON.parse(code); } catch (e) { err = (e as Error).message; }
  const data = (spec?.data || []).filter((d) => d && typeof d.value === "number");
  if (err || !data.length) {
    return <div className="md-artifact"><div className="md-artifact-h"><span className="badge">chart</span><span className="note" style={{ marginLeft: "auto" }}>{err ? "invalid JSON" : "no data"}</span></div><pre className="md-pre"><code>{code}</code></pre></div>;
  }
  const type = spec?.type === "line" ? "line" : spec?.type === "pie" ? "pie" : "bar";
  const W = 520, H = 250, padX = 44, padY = 26, iw = W - padX * 2, ih = H - padY * 2;
  const max = Math.max(...data.map((d) => d.value), 0) || 1;
  const body = () => {
    if (type === "pie") {
      const total = data.reduce((a, d) => a + Math.max(0, d.value), 0) || 1;
      const cx = 130, cy = H / 2, r = 90; let a0 = -Math.PI / 2;
      return <>
        {data.map((d, i) => {
          const frac = Math.max(0, d.value) / total; const a1 = a0 + frac * Math.PI * 2;
          const x0 = cx + r * Math.cos(a0), y0 = cy + r * Math.sin(a0), x1 = cx + r * Math.cos(a1), y1 = cy + r * Math.sin(a1);
          const large = frac > 0.5 ? 1 : 0; const path = `M${cx} ${cy} L${x0.toFixed(1)} ${y0.toFixed(1)} A${r} ${r} 0 ${large} 1 ${x1.toFixed(1)} ${y1.toFixed(1)} Z`;
          a0 = a1; return <path key={i} d={path} fill={PALETTE[i % PALETTE.length]} opacity={0.9} />;
        })}
        {data.map((d, i) => (
          <g key={i} transform={`translate(${W - 200}, ${padY + i * 22})`}>
            <rect width={12} height={12} rx={2} fill={PALETTE[i % PALETTE.length]} />
            <text x={18} y={10} fontSize={12} fill="var(--muted)">{d.label} · {d.value}</text>
          </g>
        ))}
      </>;
    }
    const x = (i: number) => padX + (data.length === 1 ? iw / 2 : (iw * i) / (data.length - 1));
    const y = (v: number) => padY + ih - (v / max) * ih;
    if (type === "line") {
      const pts = data.map((d, i) => `${x(i).toFixed(1)},${y(d.value).toFixed(1)}`).join(" ");
      return <>
        <polyline points={pts} fill="none" stroke="var(--accent)" strokeWidth={2.5} />
        {data.map((d, i) => <g key={i}><circle cx={x(i)} cy={y(d.value)} r={3.5} fill="var(--accent)" /><text x={x(i)} y={H - 8} fontSize={11} fill="var(--muted)" textAnchor="middle">{d.label}</text></g>)}
      </>;
    }
    const bw = (iw / data.length) * 0.62;
    return <>{data.map((d, i) => { const bx = padX + (iw * i) / data.length + (iw / data.length - bw) / 2; const bh = (d.value / max) * ih; return (
      <g key={i}>
        <rect x={bx} y={padY + ih - bh} width={bw} height={Math.max(0, bh)} rx={3} fill={PALETTE[i % PALETTE.length]} />
        <text x={bx + bw / 2} y={padY + ih - bh - 5} fontSize={11} fill="var(--muted)" textAnchor="middle">{d.value}</text>
        <text x={bx + bw / 2} y={H - 8} fontSize={11} fill="var(--muted)" textAnchor="middle">{d.label}</text>
      </g>
    ); })}</>;
  };
  return (
    <div className="md-artifact">
      <div className="md-artifact-h"><span className="badge">chart</span>{spec?.title && <b style={{ fontSize: 12.5 }}>{spec.title}</b>}</div>
      <div style={{ padding: 12, overflowX: "auto" }}>
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", maxWidth: W, height: "auto" }}>
          {type !== "pie" && <line x1={padX} y1={padY + ih} x2={W - padX} y2={padY + ih} stroke="var(--border)" />}
          {body()}
        </svg>
      </div>
    </div>
  );
}

// Mermaid diagram rendered in a sandboxed iframe (loads mermaid from CDN).
function MermaidArtifact({ code }: { code: string }) {
  const [view, setView] = useState<"preview" | "code">("preview");
  const esc = code.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const doc = `<!doctype html><html><head><meta charset="utf-8"><script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"><\/script></head><body style="margin:0;padding:10px;overflow:auto;font-family:system-ui"><pre class="mermaid">${esc}</pre><script>try{mermaid.initialize({startOnLoad:true,securityLevel:"strict"})}catch(e){document.body.innerHTML='<p style=\\'color:#b91c1c;font:13px system-ui\\'>Could not render diagram.</p>'}<\/script></body></html>`;
  return (
    <div className="md-artifact">
      <div className="md-artifact-h"><span className="badge">mermaid</span><div className="seg" style={{ width: 150, marginLeft: "auto" }}><button className={view === "preview" ? "on" : ""} onClick={() => setView("preview")}>Preview</button><button className={view === "code" ? "on" : ""} onClick={() => setView("code")}>Code</button></div></div>
      {view === "preview"
        ? <iframe title="mermaid" sandbox="allow-scripts" srcDoc={doc} style={{ width: "100%", height: 340, border: "none", background: "#fff", borderRadius: "0 0 8px 8px" }} />
        : <pre className="md-pre"><code>{code}</code></pre>}
    </div>
  );
}

export default function Markdown({ text }: { text: string }) {
  const src = (text || "")
    .replace(/\r\n?/g, "\n")
    .replace(/\[([^\]\n]+)\]\s*\n+\s*\((https?:\/\/[^\s)]+)\)/g, "[$1]($2)");
  const out: ReactNode[] = [];
  const lines = src.split("\n");
  let i = 0, k = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (line.startsWith("```")) {
      const lang = line.slice(3).trim().toLowerCase();
      const buf: string[] = []; i++;
      while (i < lines.length && !lines[i].startsWith("```")) { buf.push(lines[i]); i++; }
      i++; const code = buf.join("\n");
      if (lang === "chart") out.push(<ChartArtifact key={k++} code={code} />);
      else if (lang === "mermaid") out.push(<MermaidArtifact key={k++} code={code} />);
      else out.push(<Artifact key={k++} lang={lang} code={code} />);
      continue;
    }
    if (/^#{1,6}\s/.test(line)) { const lvl = line.match(/^#+/)![0].length; const t = line.replace(/^#+\s/, ""); out.push(<div key={k++} className={`md-h md-h${Math.min(lvl, 3)}`}>{inline(t, `h${k}`)}</div>); i++; continue; }
    if (/^>\s?/.test(line)) { const buf: string[] = []; while (i < lines.length && /^>\s?/.test(lines[i])) { buf.push(lines[i].replace(/^>\s?/, "")); i++; } out.push(<blockquote key={k++} className="md-quote">{inline(buf.join(" "), `q${k}`)}</blockquote>); continue; }
    if (/^\s*([-*]|\d+\.)\s/.test(line)) {
      const ordered = /^\s*\d+\./.test(line); const items: string[] = [];
      while (i < lines.length && /^\s*([-*]|\d+\.)\s/.test(lines[i])) { items.push(lines[i].replace(/^\s*([-*]|\d+\.)\s/, "")); i++; }
      out.push(ordered
        ? <ol key={k++} className="md-list">{items.map((it, j) => <li key={j}>{inline(it, `li${k}-${j}`)}</li>)}</ol>
        : <ul key={k++} className="md-list">{items.map((it, j) => <li key={j}>{inline(it, `li${k}-${j}`)}</li>)}</ul>);
      continue;
    }
    if (line.includes("|") && lines[i + 1] && /^\s*\|?[\s:|-]+\|?\s*$/.test(lines[i + 1])) {
      const cells = (s: string) => s.replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
      const head = cells(line); i += 2; const rows: string[][] = [];
      while (i < lines.length && lines[i].includes("|")) { rows.push(cells(lines[i])); i++; }
      out.push(<div key={k++} style={{ overflowX: "auto" }}><table className="tbl md-table"><thead><tr>{head.map((h, j) => <th key={j}>{inline(h, `th${k}-${j}`)}</th>)}</tr></thead><tbody>{rows.map((r, ri) => <tr key={ri}>{r.map((c, ci) => <td key={ci}>{inline(c, `td${k}-${ri}-${ci}`)}</td>)}</tr>)}</tbody></table></div>);
      continue;
    }
    if (line.trim() === "") { i++; continue; }
    const buf: string[] = [];
    while (i < lines.length && lines[i].trim() !== "" && !lines[i].startsWith("```") && !/^#{1,6}\s/.test(lines[i]) && !/^\s*([-*]|\d+\.)\s/.test(lines[i]) && !/^>\s?/.test(lines[i])) { buf.push(lines[i]); i++; }
    out.push(<p key={k++} className="md-p">{inline(buf.join(" "), `p${k}`)}</p>);
  }
  return <div className="md">{out}</div>;
}
