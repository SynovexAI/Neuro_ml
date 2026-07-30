"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

type Project = { id: string; lab: string; name: string; updatedAt?: string | null; createdAt?: string | null };

const LAB_META: Record<string, { label: string; href: string; icon: string }> = {
  prompting: { label: "Prompting", href: "/labs/prompting", icon: "✎" },
  rag: { label: "RAG", href: "/labs/rag", icon: "❖" },
  agent: { label: "Agent", href: "/labs/agent", icon: "◈" },
  ml: { label: "ML", href: "/labs/ml", icon: "⚙" },
  dl: { label: "DL", href: "/labs/dl", icon: "⟐" },
  etl: { label: "ETL", href: "/labs/etl", icon: "⇉" },
};

function when(p: Project): string {
  const s = p.updatedAt || p.createdAt;
  if (!s) return "";
  const d = new Date(s);
  if (isNaN(d.getTime())) return "";
  return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

export default function MyProjects() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<string>("all");
  const [busy, setBusy] = useState<string>("");

  async function load() {
    setLoading(true);
    try { const j = await fetch("/api/projects").then((r) => r.json()); setProjects(j.projects || []); }
    catch { setProjects([]); }
    setLoading(false);
  }
  useEffect(() => { load(); }, []);

  async function rename(p: Project) {
    const name = window.prompt("Rename project", p.name);
    if (!name || name.trim() === p.name) return;
    setBusy(p.id);
    await fetch("/api/projects", { method: "PATCH", headers: { "content-type": "application/json" }, body: JSON.stringify({ id: p.id, name: name.trim() }) }).catch(() => {});
    setProjects((ps) => ps.map((x) => (x.id === p.id ? { ...x, name: name.trim() } : x)));
    setBusy("");
  }
  async function remove(p: Project) {
    if (!window.confirm(`Delete "${p.name}"? This can't be undone.`)) return;
    setBusy(p.id);
    await fetch(`/api/projects?id=${encodeURIComponent(p.id)}`, { method: "DELETE" }).catch(() => {});
    setProjects((ps) => ps.filter((x) => x.id !== p.id));
    setBusy("");
  }

  const labs = Array.from(new Set(projects.map((p) => p.lab)));
  const shown = filter === "all" ? projects : projects.filter((p) => p.lab === filter);

  return (
    <>
      <div className="lab-head">
        <div><div className="eyebrow">Studio</div><h2 className="page-h">My Projects</h2><p className="page-sub" style={{ margin: 0 }}>Everything you&apos;ve saved from the labs. Open reloads a build; rename and delete manage your list.</p></div>
        <div className="acts"><button className="btn ghost sm" onClick={load}>↻ Refresh</button></div>
      </div>

      {!loading && projects.length > 0 && (
        <div className="chips" style={{ marginBottom: 14 }}>
          <button className={`chip ${filter === "all" ? "on" : ""}`} onClick={() => setFilter("all")}>All ({projects.length})</button>
          {labs.map((l) => <button key={l} className={`chip ${filter === l ? "on" : ""}`} onClick={() => setFilter(l)}>{LAB_META[l]?.label || l} ({projects.filter((p) => p.lab === l).length})</button>)}
        </div>
      )}

      {loading ? <div className="note">Loading…</div>
        : projects.length === 0 ? (
          <div className="card"><div className="card-b" style={{ textAlign: "center", padding: 40 }}>
            <div style={{ fontSize: 34, opacity: 0.4 }}>◇</div>
            <p className="page-sub" style={{ margin: "10px 0 4px" }}>No saved projects yet.</p>
            <div className="note">Open any lab, build something, and hit <b>💾 Save</b>. It&apos;ll show up here.</div>
          </div></div>
        ) : (
          <div className="proj-grid">
            {shown.map((p) => {
              const m = LAB_META[p.lab] || { label: p.lab, href: "/dashboard", icon: "◆" };
              return (
                <div key={p.id} className="card proj-card">
                  <div className="card-b">
                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                      <span className="proj-ic">{m.icon}</span>
                      <span className="badge">{m.label}</span>
                      <span className="spacer" style={{ flex: 1 }} />
                      <span className="note">{when(p)}</span>
                    </div>
                    <div className="proj-name" title={p.name}>{p.name}</div>
                    <div className="acts" style={{ marginTop: 12 }}>
                      <Link className="btn sm" href={`${m.href}?project=${p.id}`}>Open</Link>
                      <button className="btn ghost sm" disabled={busy === p.id} onClick={() => rename(p)}>Rename</button>
                      <button className="btn ghost sm danger" disabled={busy === p.id} onClick={() => remove(p)}>Delete</button>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
    </>
  );
}
