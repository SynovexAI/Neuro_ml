"use client";
import React from "react";

// A2UI — Agent-to-UI. A small, typed "generative UI" protocol: an agent or tool
// emits a JSON array of component nodes (usually inside a ```ui fenced block) and
// this renders each as a first-class component. Deterministic and rich — the
// reliable alternative to guessing structure from prose. Unknown/invalid nodes
// degrade to text, so it never hard-fails.

export type UINode =
  | { type: "heading"; text: string; level?: number }
  | { type: "text" | "markdown"; text: string }
  | { type: "callout"; tone?: "info" | "success" | "warn" | "error"; title?: string; text: string }
  | { type: "badge"; text: string; tone?: "info" | "success" | "warn" | "error" | "neutral" }
  | { type: "stats"; items: { label: string; value: string | number; delta?: string }[] }
  | { type: "kv"; items: { label: string; value: string | number }[] }
  | { type: "table"; columns?: string[]; rows: (string | number)[][] | Record<string, unknown>[] }
  | { type: "chart"; variant?: "bar" | "line"; title?: string; data: { label: string; value: number }[] }
  | { type: "list"; ordered?: boolean; items: string[] }
  | { type: "code"; code: string; lang?: string }
  | { type: "card"; title?: string; children: UINode[] }
  | { type: "divider" };

const TONE: Record<string, string> = { info: "var(--accent)", success: "var(--good)", warn: "var(--warn)", error: "var(--crit)", neutral: "var(--muted)" };
const s = (v: unknown) => (v == null ? "" : String(v));

function Chart({ data, variant }: { data: { label: string; value: number }[]; variant?: string }) {
  const max = Math.max(...data.map((d) => Math.abs(Number(d.value) || 0)), 1e-9);
  if (variant === "line") {
    const W = 320, H = 90, pad = 6;
    const pts = data.map((d, i) => { const x = pad + (i / Math.max(1, data.length - 1)) * (W - 2 * pad); const y = H - pad - (Math.abs(Number(d.value) || 0) / max) * (H - 2 * pad); return `${x},${y}`; }).join(" ");
    return <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} style={{ display: "block" }}><polyline points={pts} fill="none" stroke="var(--accent)" strokeWidth={2} /></svg>;
  }
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      {data.map((d, i) => (
        <div key={i} style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ width: 150, flex: "0 0 auto", fontSize: 12, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={d.label}>{d.label}</span>
          <div style={{ flex: 1, background: "var(--panel-2)", borderRadius: 4, height: 16, overflow: "hidden" }}><div style={{ width: `${Math.max(2, (Math.abs(Number(d.value) || 0) / max) * 100)}%`, height: "100%", background: "var(--accent)", borderRadius: 4 }} /></div>
          <span style={{ width: 66, flex: "0 0 auto", textAlign: "right", fontFamily: "var(--mono)", fontSize: 12 }}>{d.value}</span>
        </div>
      ))}
    </div>
  );
}

function Node({ node }: { node: UINode }) {
  switch (node.type) {
    case "heading": { const lvl = Math.min(3, node.level || 2); return <div style={{ fontWeight: 700, fontSize: lvl === 1 ? 17 : lvl === 2 ? 14.5 : 13, color: "var(--text)", margin: "2px 0" }}>{node.text}</div>; }
    case "text": case "markdown": return <div style={{ fontSize: 13, lineHeight: 1.6, color: "var(--muted)", whiteSpace: "pre-wrap" }}>{node.text}</div>;
    case "callout": { const c = TONE[node.tone || "info"]; return <div style={{ border: `1px solid color-mix(in srgb, ${c} 40%, transparent)`, background: `color-mix(in srgb, ${c} 8%, transparent)`, borderLeft: `3px solid ${c}`, borderRadius: 8, padding: "9px 12px" }}>{node.title && <div style={{ fontWeight: 600, color: c, fontSize: 12.5, marginBottom: 2 }}>{node.title}</div>}<div style={{ fontSize: 12.5, color: "var(--text)", lineHeight: 1.5 }}>{node.text}</div></div>; }
    case "badge": { const c = TONE[node.tone || "neutral"]; return <span style={{ display: "inline-block", fontSize: 11, fontWeight: 600, padding: "2px 9px", borderRadius: 20, color: c, background: `color-mix(in srgb, ${c} 14%, transparent)`, border: `1px solid color-mix(in srgb, ${c} 35%, transparent)` }}>{node.text}</span>; }
    case "stats": return (
      <div style={{ display: "grid", gridTemplateColumns: `repeat(auto-fit, minmax(120px, 1fr))`, gap: 10 }}>
        {node.items.map((it, i) => (
          <div key={i} style={{ background: "var(--panel)", border: "1px solid var(--border)", borderRadius: 10, padding: "10px 14px" }}>
            <div style={{ fontFamily: "var(--mono)", fontSize: 20, fontWeight: 600, lineHeight: 1.1 }}>{s(it.value)}</div>
            <div style={{ fontSize: 9.5, textTransform: "uppercase", letterSpacing: ".06em", color: "var(--faint)", marginTop: 3 }}>{it.label}</div>
            {it.delta && <div style={{ fontSize: 11, marginTop: 2, color: /^-/.test(s(it.delta)) ? "var(--crit)" : "var(--good)" }}>{it.delta}</div>}
          </div>
        ))}
      </div>
    );
    case "kv": return (
      <div style={{ overflowX: "auto", border: "1px solid var(--border)", borderRadius: 8 }}>
        <table className="dtable" style={{ width: "100%", fontSize: 12.5, borderCollapse: "collapse" }}><tbody>
          {node.items.map((it, i) => <tr key={i}><td style={{ padding: "6px 12px", borderBottom: "1px solid var(--border)", color: "var(--muted)" }}>{it.label}</td><td style={{ padding: "6px 12px", borderBottom: "1px solid var(--border)", textAlign: "right", fontFamily: "var(--mono)" }}>{s(it.value)}</td></tr>)}
        </tbody></table>
      </div>
    );
    case "table": {
      const objRows = node.rows.length && !Array.isArray(node.rows[0]);
      const rawCols = node.columns || (objRows ? Object.keys(node.rows[0] as Record<string, unknown>) : (node.rows[0] as unknown[] || []).map((_, i) => `col ${i + 1}`));
      const clean = (c: unknown) => s(c).replace(/^\s*\/\/\s*/, "").trim();               // strip stray "// " prefixes
      const isSep = (row: (string | number)[]) => row.every((c) => /^[-—–_\s]*$/.test(s(c))); // drop markdown separator rows
      const cols = rawCols.map(clean);
      const allRows: (string | number)[][] = objRows ? (node.rows as Record<string, unknown>[]).map((r) => rawCols.map((c) => r[c] as string | number)) : (node.rows as (string | number)[][]);
      const rows = allRows.filter((r) => Array.isArray(r) && !isSep(r));
      return (
        <div style={{ overflowX: "auto", border: "1px solid var(--border)", borderRadius: 8 }}>
          <table className="dtable" style={{ width: "100%", fontSize: 12.5, borderCollapse: "collapse" }}><tbody>
            <tr>{cols.map((c, j) => <th key={j} style={{ textAlign: "left", padding: "7px 12px", background: "var(--panel)", color: "var(--muted)", fontWeight: 600, whiteSpace: "nowrap", borderBottom: "1px solid var(--border)" }}>{c}</th>)}</tr>
            {rows.map((r, ri) => <tr key={ri}>{r.map((c, j) => { const cv = clean(c); return <td key={j} style={{ padding: "6px 12px", borderBottom: "1px solid var(--border)", fontFamily: /^-?[\d.,]+$/.test(cv) ? "var(--mono)" : "inherit", whiteSpace: "nowrap" }}>{cv}</td>; })}</tr>)}
          </tbody></table>
        </div>
      );
    }
    case "chart": return <div style={{ border: "1px solid var(--border)", borderRadius: 8, padding: "10px 12px" }}>{node.title && <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8 }}>{node.title}</div>}<Chart data={node.data || []} variant={node.variant} /></div>;
    case "list": { const Tag = node.ordered ? "ol" : "ul"; return <Tag style={{ margin: 0, paddingLeft: 20, color: "var(--muted)", fontSize: 13, lineHeight: 1.6, display: "flex", flexDirection: "column", gap: 2 }}>{node.items.map((it, i) => <li key={i}>{it}</li>)}</Tag>; }
    case "code": return <pre style={{ margin: 0, fontFamily: "var(--mono)", fontSize: 12, lineHeight: 1.55, background: "var(--panel-2)", border: "1px solid var(--border)", borderRadius: 8, padding: "10px 12px", overflowX: "auto" }}>{node.code}</pre>;
    case "card": return <div style={{ border: "1px solid var(--border)", borderRadius: 10, background: "var(--panel)", padding: 12 }}>{node.title && <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 8 }}>{node.title}</div>}<A2UI spec={node.children || []} /></div>;
    case "divider": return <div style={{ height: 1, background: "var(--border)", margin: "2px 0" }} />;
    default: return <div style={{ fontSize: 12.5, color: "var(--muted)" }}>{s((node as { text?: string }).text ?? JSON.stringify(node))}</div>;
  }
}

export default function A2UI({ spec }: { spec: UINode[] }) {
  if (!Array.isArray(spec)) return null;
  return <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>{spec.map((n, i) => <Node key={i} node={n} />)}</div>;
}

// Try to parse a JSON string into a UINode[] spec. Returns null if it isn't valid.
export function parseA2UI(json: string): UINode[] | null {
  try {
    const v = JSON.parse(json.trim());
    if (Array.isArray(v)) return v as UINode[];
    if (v && typeof v === "object") {
      const o = v as Record<string, unknown>;
      if (Array.isArray(o.ui)) return o.ui as UINode[];
      if (Array.isArray(o.components)) return o.components as UINode[];
      if (Array.isArray(o.blocks)) return o.blocks as UINode[];
      if (typeof o.type === "string") return [v as UINode]; // a single component → wrap
    }
    return null;
  } catch { return null; }
}
