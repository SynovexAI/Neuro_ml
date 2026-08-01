"use client";

import { useState, type ReactNode } from "react";

// Lightweight, safe markdown renderer (React elements, no dangerouslySetInnerHTML).
// Handles headings, bold/italic, inline code, links, lists, tables, blockquotes,
// and fenced code — with html/svg/mermaid blocks rendered as sandboxed artifacts.

function inline(text: string, key: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  const re = /(\*\*(.+?)\*\*|__(.+?)__|\*(.+?)\*|`(.+?)`|\[(.+?)\]\((.+?)\))/g;
  let last = 0, m: RegExpExecArray | null, i = 0;
  while ((m = re.exec(text))) {
    if (m.index > last) nodes.push(text.slice(last, m.index));
    if (m[2] || m[3]) nodes.push(<b key={`${key}-${i}`}>{m[2] || m[3]}</b>);
    else if (m[4]) nodes.push(<i key={`${key}-${i}`}>{m[4]}</i>);
    else if (m[5]) nodes.push(<code key={`${key}-${i}`} className="md-code">{m[5]}</code>);
    else if (m[6]) nodes.push(<a key={`${key}-${i}`} href={m[7]} target="_blank" rel="noreferrer">{m[6]}</a>);
    last = m.index + m[0].length; i++;
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

export default function Markdown({ text }: { text: string }) {
  const src = (text || "").replace(/\r\n?/g, "\n");
  const out: ReactNode[] = [];
  const lines = src.split("\n");
  let i = 0, k = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (line.startsWith("```")) {
      const lang = line.slice(3).trim().toLowerCase();
      const buf: string[] = []; i++;
      while (i < lines.length && !lines[i].startsWith("```")) { buf.push(lines[i]); i++; }
      i++; out.push(<Artifact key={k++} lang={lang} code={buf.join("\n")} />);
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
