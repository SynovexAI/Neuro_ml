"use client";
import { useMemo } from "react";
import A2UI, { parseA2UI, type UINode } from "@/components/A2UI";

// Adaptive renderer for agent / tool output — used everywhere a tool or agent
// result is shown so every result renders consistently and readably:
//   • ```code fences```        → syntax-neutral code block
//   • pipe-delimited rows      → a real table
//   • "label: number" series   → a horizontal bar chart + values
//   • markdown text            → headings, paragraphs, bullet/numbered lists,
//                                 **bold**, *italic*, `code`, [links](url)

type Block =
  | { kind: "ui"; spec: UINode[] }                              // explicit ```ui structured components (A2UI)
  | { kind: "table"; header: string[]; rows: string[][] }
  | { kind: "kv"; items: { label: string; value: string }[] }   // "label: value" list → 2-col table
  | { kind: "chart"; items: { label: string; value: number }[] } // ONLY from an explicit ```chart block
  | { kind: "code"; code: string }
  | { kind: "text"; lines: string[] };

const isPipeRow = (l: string) => l.includes("|") && l.split("|").length >= 2;
const numRe = /^\s*(.+?)\s*[:=]\s*(-?[\d,]+(?:\.\d+)?)\s*$/;
const fenceRe = /^\s*```/;

function parse(text: string): Block[] {
  const lines = (text || "").replace(/\r/g, "").split("\n");
  const blocks: Block[] = [];
  let textBuf: string[] = [];
  const flush = () => { if (textBuf.length) { blocks.push({ kind: "text", lines: textBuf.slice() }); textBuf = []; } };
  let i = 0;
  while (i < lines.length) {
    // fenced block — ```chart (JSON of {label,value}) renders a chart; anything else is code
    if (fenceRe.test(lines[i])) {
      flush();
      const lang = lines[i].replace(/`+/g, "").trim().toLowerCase();
      i++;
      const body: string[] = [];
      while (i < lines.length && !fenceRe.test(lines[i])) { body.push(lines[i]); i++; }
      i++; // closing fence
      if (lang === "ui" || lang === "a2ui") {
        const spec = parseA2UI(body.join("\n"));
        if (spec) { blocks.push({ kind: "ui", spec }); continue; }
      }
      if (lang === "chart") {
        try {
          const arr = JSON.parse(body.join("\n"));
          const items = (Array.isArray(arr) ? arr : []).map((x) => ({ label: String(x.label ?? x.name ?? ""), value: Number(x.value ?? x.y ?? 0) })).filter((x) => x.label && !isNaN(x.value));
          if (items.length) { blocks.push({ kind: "chart", items }); continue; }
        } catch { /* fall through to code */ }
      }
      blocks.push({ kind: "code", code: body.join("\n") });
      continue;
    }
    // table: 2+ consecutive pipe rows
    if (isPipeRow(lines[i]) && i + 1 < lines.length && isPipeRow(lines[i + 1])) {
      flush();
      const tbl: string[][] = [];
      while (i < lines.length && isPipeRow(lines[i])) { tbl.push(lines[i].split("|").map((c) => c.trim())); i++; }
      // drop a markdown separator row like |---|---|
      const rows = tbl.filter((r) => !r.every((c) => /^:?-{2,}:?$/.test(c) || c === ""));
      const width = Math.max(...rows.map((r) => r.length));
      const norm = rows.map((r) => { const c = [...r]; while (c.length < width) c.push(""); return c; });
      if (norm.length) blocks.push({ kind: "table", header: norm[0], rows: norm.slice(1) });
      continue;
    }
    // "label: value" series (2+ lines) → a clean 2-column table (NOT a chart)
    if (numRe.test(lines[i]) && i + 1 < lines.length && numRe.test(lines[i + 1])) {
      flush();
      const items: { label: string; value: string }[] = [];
      while (i < lines.length && numRe.test(lines[i])) { const m = lines[i].match(numRe)!; items.push({ label: m[1], value: m[2] }); i++; }
      blocks.push({ kind: "kv", items });
      continue;
    }
    textBuf.push(lines[i]); i++;
  }
  flush();
  return blocks;
}

// Inline markdown: **bold**, *italic*, `code`, [text](url).
function inline(s: string) {
  const parts = s.split(/(\*\*[^*]+\*\*|__[^_]+__|\*[^*\s][^*]*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))/g).filter((x) => x !== "");
  return parts.map((p, j) => {
    if (/^(\*\*|__)[^]+(\*\*|__)$/.test(p)) return <b key={j}>{p.slice(2, -2)}</b>;
    if (/^\*[^]+\*$/.test(p)) return <i key={j}>{p.slice(1, -1)}</i>;
    if (/^`[^`]+`$/.test(p)) return <code key={j} style={{ fontFamily: "var(--mono)", background: "var(--panel-2)", padding: "1px 5px", borderRadius: 4, fontSize: "0.92em" }}>{p.slice(1, -1)}</code>;
    const link = p.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
    if (link) return <a key={j} href={link[2]} target="_blank" rel="noreferrer" style={{ color: "var(--accent)", textDecoration: "underline" }}>{link[1]}</a>;
    return <span key={j}>{p}</span>;
  });
}

// Render a run of text lines as markdown: headings, lists, paragraphs.
function renderText(lines: string[], keyBase: number) {
  const out: React.ReactNode[] = [];
  let i = 0, para: string[] = [], key = 0;
  const flushPara = () => {
    if (!para.length) return;
    out.push(<p key={`${keyBase}-p${key++}`} style={{ margin: "0 0 8px", color: "var(--muted)", lineHeight: 1.6 }}>{para.map((l, k) => <span key={k}>{k ? <br /> : null}{inline(l)}</span>)}</p>);
    para = [];
  };
  const listItem = (l: string) => l.match(/^\s*([-*•]|\d+[.)])\s+(.*)$/);
  while (i < lines.length) {
    const l = lines[i];
    if (l.trim() === "") { flushPara(); i++; continue; }
    const h = l.match(/^\s*(#{1,6})\s+(.*)$/) || (/^\s*\/\/\s*(.*)$/.test(l) ? ["", "##", l.replace(/^\s*\/\/\s*/, "")] as RegExpMatchArray : null);
    if (h) {
      flushPara();
      const lvl = Math.min(3, h[1].length); const size = lvl === 1 ? 16 : lvl === 2 ? 14 : 13;
      out.push(<div key={`${keyBase}-h${key++}`} style={{ fontWeight: 700, fontSize: size, color: "var(--text)", margin: "10px 0 4px" }}>{inline(h[2])}</div>);
      i++; continue;
    }
    if (listItem(l)) {
      flushPara();
      const ordered = /^\s*\d/.test(l);
      const items: string[] = [];
      while (i < lines.length && listItem(lines[i])) { items.push(listItem(lines[i])![2]); i++; }
      // If every item is a "label — value" (or "label: value") pair, it's really tabular → render a table.
      const splitKV = (s: string) => { const m = s.match(/^(.+?)(?:\s+[—–-]\s+|:\s+)(.+)$/); return m ? [m[1].trim(), m[2].trim()] as [string, string] : null; };
      const pairs = items.map(splitKV);
      if (items.length >= 2 && pairs.every(Boolean)) {
        out.push(
          <div key={`${keyBase}-lt${key++}`} style={{ overflowX: "auto", border: "1px solid var(--border)", borderRadius: 8, margin: "0 0 8px" }}>
            <table className="dtable" style={{ width: "100%", fontSize: 12.5, borderCollapse: "collapse" }}><tbody>
              {(pairs as [string, string][]).map((p, k) => (
                <tr key={k}>
                  <td style={{ padding: "6px 12px", borderBottom: "1px solid var(--border)" }}>{inline(p[0])}</td>
                  <td style={{ padding: "6px 12px", borderBottom: "1px solid var(--border)", textAlign: "right", fontFamily: "var(--mono)", whiteSpace: "nowrap" }}>{p[1]}</td>
                </tr>
              ))}
            </tbody></table>
          </div>
        );
        continue;
      }
      const Tag = ordered ? "ol" : "ul";
      out.push(<Tag key={`${keyBase}-l${key++}`} style={{ margin: "0 0 8px", paddingLeft: 20, color: "var(--muted)", lineHeight: 1.6, display: "flex", flexDirection: "column", gap: 2 }}>{items.map((it, k) => <li key={k}>{inline(it)}</li>)}</Tag>);
      continue;
    }
    para.push(l); i++;
  }
  flushPara();
  return out;
}

export default function AgentOutput({ text }: { text: string }) {
  const blocks = useMemo(() => parse(text), [text]);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10, fontSize: 13 }}>
      {blocks.map((b, bi) => {
        if (b.kind === "ui") return <A2UI key={bi} spec={b.spec} />;
        if (b.kind === "code") return (
          <pre key={bi} style={{ margin: 0, fontFamily: "var(--mono)", fontSize: 12, lineHeight: 1.55, background: "var(--panel-2)", border: "1px solid var(--border)", borderRadius: 8, padding: "10px 12px", overflowX: "auto", color: "var(--text)" }}>{b.code}</pre>
        );
        if (b.kind === "table") return (
          <div key={bi} style={{ overflowX: "auto", border: "1px solid var(--border)", borderRadius: 8 }}>
            <table className="dtable" style={{ width: "100%", fontSize: 12.5, borderCollapse: "collapse" }}><tbody>
              <tr>{b.header.map((h, j) => <th key={j} style={{ textAlign: "left", padding: "7px 12px", background: "var(--panel)", color: "var(--muted)", fontWeight: 600, whiteSpace: "nowrap", borderBottom: "1px solid var(--border)" }}>{h}</th>)}</tr>
              {b.rows.map((r, ri) => <tr key={ri}>{r.map((c, j) => <td key={j} style={{ padding: "6px 12px", borderBottom: "1px solid var(--border)", fontFamily: /^-?[\d.,]+$/.test(c) ? "var(--mono)" : "inherit", whiteSpace: "nowrap" }}>{c}</td>)}</tr>)}
            </tbody></table>
          </div>
        );
        if (b.kind === "kv") return (
          <div key={bi} style={{ overflowX: "auto", border: "1px solid var(--border)", borderRadius: 8 }}>
            <table className="dtable" style={{ width: "100%", fontSize: 12.5, borderCollapse: "collapse" }}><tbody>
              {b.items.map((it, j) => (
                <tr key={j}>
                  <td style={{ padding: "6px 12px", borderBottom: "1px solid var(--border)" }}>{it.label}</td>
                  <td style={{ padding: "6px 12px", borderBottom: "1px solid var(--border)", textAlign: "right", fontFamily: "var(--mono)", whiteSpace: "nowrap" }}>{it.value}</td>
                </tr>
              ))}
            </tbody></table>
          </div>
        );
        if (b.kind === "chart") {
          const max = Math.max(...b.items.map((x) => Math.abs(x.value)), 1e-9);
          return (
            <div key={bi} style={{ border: "1px solid var(--border)", borderRadius: 8, padding: "10px 12px", display: "flex", flexDirection: "column", gap: 6 }}>
              {b.items.map((it, j) => (
                <div key={j} style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <span style={{ width: 150, flex: "0 0 auto", fontSize: 12, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={it.label}>{it.label}</span>
                  <div style={{ flex: 1, background: "var(--panel-2)", borderRadius: 4, height: 16, position: "relative", overflow: "hidden" }}>
                    <div style={{ width: `${Math.max(2, (Math.abs(it.value) / max) * 100)}%`, height: "100%", background: "var(--accent)", borderRadius: 4 }} />
                  </div>
                  <span style={{ width: 66, flex: "0 0 auto", textAlign: "right", fontFamily: "var(--mono)", fontSize: 12, color: "var(--text)" }}>{it.value}</span>
                </div>
              ))}
            </div>
          );
        }
        return <div key={bi}>{renderText(b.lines, bi)}</div>;
      })}
    </div>
  );
}
