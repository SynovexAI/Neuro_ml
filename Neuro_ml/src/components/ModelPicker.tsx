"use client";
import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

// Searchable model picker — filter a long model list by typing; also accepts a custom id.
// Shared by the Admin providers page and the RAG lab's neural-embedding setup.
// The list renders in a portal with fixed positioning so it never gets clipped by an
// ancestor's `overflow: hidden` (e.g. the RAG panel).
export default function ModelPicker({ models, value, onChange, placeholder, openKey }: { models: string[]; value: string; onChange: (v: string) => void; placeholder?: string; width?: number | string; openKey?: number }) {
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState(value);
  const [rect, setRect] = useState<{ left: number; top: number; width: number } | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => { setQ(value); }, [value]);

  const place = () => { const r = inputRef.current?.getBoundingClientRect(); if (r) setRect({ left: r.left, top: r.bottom, width: r.width }); };
  const show = () => { place(); setOpen(true); };

  // Parent bumps openKey (e.g. after "Fetch models") to pop the list open automatically.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { if (openKey) { setQ(""); show(); } }, [openKey]);

  // Keep the portal anchored while open if the page scrolls/resizes.
  useEffect(() => {
    if (!open) return;
    const on = () => place();
    window.addEventListener("scroll", on, true);
    window.addEventListener("resize", on);
    return () => { window.removeEventListener("scroll", on, true); window.removeEventListener("resize", on); };
  }, [open]);

  const ql = q.trim().toLowerCase();
  const filtered = ql ? models.filter((m) => m.toLowerCase().includes(ql)) : models;

  return (
    <div style={{ position: "relative", width: "100%" }}>
      <input
        ref={inputRef}
        type="text"
        value={q}
        placeholder={placeholder || "Search or type a model id…"}
        onFocus={show}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
        onChange={(e) => { setQ(e.target.value); onChange(e.target.value); show(); }}
        style={{ width: "100%" }}
      />
      {open && rect && models.length > 0 && typeof document !== "undefined" && createPortal(
        <div style={{ position: "fixed", left: rect.left, top: rect.top + 4, width: Math.max(rect.width, 280), maxWidth: "min(96vw, 420px)", zIndex: 9999, background: "var(--surface)", border: "1px solid var(--border-strong)", borderRadius: 10, boxShadow: "0 16px 40px rgba(0,0,0,.5)", maxHeight: 300, overflowY: "auto", padding: 5 }}>
          {filtered.length ? filtered.slice(0, 250).map((m) => (
            <div
              key={m}
              onMouseDown={(e) => { e.preventDefault(); onChange(m); setQ(m); setOpen(false); }}
              className="etl-load-row"
              style={{ display: "flex", alignItems: "center", gap: 8, padding: "7px 10px", borderRadius: 7, cursor: "pointer", fontSize: 12.5, fontFamily: "var(--mono)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", background: m === value ? "var(--accent-weak)" : undefined, border: `1px solid ${m === value ? "var(--accent)" : "transparent"}`, marginBottom: 2 }}
            >
              <span style={{ overflow: "hidden", textOverflow: "ellipsis" }}>{m}</span>
              {/embed/i.test(m) ? <span className="badge" style={{ fontSize: 8.5, flex: "0 0 auto" }}>embeddings</span> : null}
            </div>
          )) : <div className="note" style={{ padding: "7px 10px" }}>no match — will use “{q}”</div>}
        </div>,
        document.body,
      )}
    </div>
  );
}
