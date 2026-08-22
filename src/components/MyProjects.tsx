"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Trash2, CheckSquare, Square, RefreshCw, Check } from "lucide-react";
import { toast, confirmDialog } from "@/lib/toast";

type Project = { id: string; lab: string; name: string; published?: boolean; updatedAt?: string | null; createdAt?: string | null };

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
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  async function load() {
    setLoading(true);
    try {
      const j = await fetch("/api/projects").then((r) => r.json());
      setProjects(j.projects || []);
      setSelectedIds(new Set());
    } catch {
      setProjects([]);
    }
    setLoading(false);
  }
  useEffect(() => { load(); }, []);

  const labs = Array.from(new Set(projects.map((p) => p.lab)));
  const shown = filter === "all" ? projects : projects.filter((p) => p.lab === filter);
  const allShownSelected = shown.length > 0 && shown.every((p) => selectedIds.has(p.id));

  function toggleSelect(id: string) {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function toggleSelectAll() {
    const allShownIds = shown.map((p) => p.id);
    if (allShownSelected) {
      setSelectedIds((prev) => {
        const next = new Set(prev);
        allShownIds.forEach((id) => next.delete(id));
        return next;
      });
    } else {
      setSelectedIds((prev) => {
        const next = new Set(prev);
        allShownIds.forEach((id) => next.add(id));
        return next;
      });
    }
  }

  async function removeSelected() {
    const idsToDelete = Array.from(selectedIds);
    if (!idsToDelete.length) return;
    const confirmed = await confirmDialog(
      `Delete ${idsToDelete.length} selected project${idsToDelete.length > 1 ? "s" : ""}? This action cannot be undone.`,
      { confirmLabel: `Delete ${idsToDelete.length} Projects`, danger: true }
    );
    if (!confirmed) return;
    setBusy("bulk");
    try {
      const r = await fetch(`/api/projects?ids=${encodeURIComponent(idsToDelete.join(","))}`, { method: "DELETE" });
      if (r.ok) {
        setProjects((ps) => ps.filter((x) => !selectedIds.has(x.id)));
        setSelectedIds(new Set());
        toast(`Successfully deleted ${idsToDelete.length} project${idsToDelete.length > 1 ? "s" : ""}`, "success");
      } else {
        toast("Failed to delete selected projects", "error");
      }
    } catch {
      toast("Delete request failed", "error");
    } finally {
      setBusy("");
    }
  }

  async function rename(p: Project) {
    const name = window.prompt("Rename project", p.name);
    if (!name || name.trim() === p.name) return;
    setBusy(p.id);
    await fetch("/api/projects", { method: "PATCH", headers: { "content-type": "application/json" }, body: JSON.stringify({ id: p.id, name: name.trim() }) }).catch(() => {});
    setProjects((ps) => ps.map((x) => (x.id === p.id ? { ...x, name: name.trim() } : x)));
    setBusy("");
  }

  async function togglePublish(p: Project) {
    const next = !p.published;
    setBusy(p.id);
    const res = await fetch("/api/projects", { method: "PATCH", headers: { "content-type": "application/json" }, body: JSON.stringify({ id: p.id, published: next }) }).catch(() => null);
    if (res?.ok) { setProjects((ps) => ps.map((x) => (x.id === p.id ? { ...x, published: next } : x))); toast(next ? `Published “${p.name}” to the Workroom` : "Unpublished", "success"); }
    else toast("Could not update", "error");
    setBusy("");
  }

  async function remove(p: Project) {
    if (!(await confirmDialog(`Delete “${p.name}”? This can't be undone.`, { confirmLabel: "Delete", danger: true }))) return;
    setBusy(p.id);
    await fetch(`/api/projects?id=${encodeURIComponent(p.id)}`, { method: "DELETE" }).catch(() => {});
    setProjects((ps) => ps.filter((x) => x.id !== p.id));
    setSelectedIds((prev) => {
      const next = new Set(prev);
      next.delete(p.id);
      return next;
    });
    setBusy("");
    toast("Project deleted", "success");
  }

  return (
    <>
      <div className="lab-head">
        <div>
          <div className="eyebrow">Studio</div>
          <h2 className="page-h">My Projects</h2>
          <p className="page-sub" style={{ margin: 0 }}>
            Everything you&apos;ve saved from the labs. Select projects to manage or delete in bulk.
          </p>
        </div>
        <div className="acts" style={{ display: "flex", gap: 8, alignItems: "center" }}>
          {selectedIds.size > 0 && (
            <button
              className="btn sm danger"
              onClick={removeSelected}
              disabled={busy === "bulk"}
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 6,
                background: "#ef4444",
                borderColor: "#dc2626",
                color: "#ffffff",
                fontWeight: 600,
              }}
            >
              {busy === "bulk" ? <span className="busy-dot" /> : <Trash2 size={13} />}
              <span>Delete Selected ({selectedIds.size})</span>
            </button>
          )}
          <button className="btn ghost sm" onClick={load} style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
            <RefreshCw size={13} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {!loading && projects.length > 0 && (
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14, flexWrap: "wrap", gap: 10 }}>
          <div className="chips" style={{ margin: 0 }}>
            <button className={`chip ${filter === "all" ? "on" : ""}`} onClick={() => setFilter("all")}>
              All ({projects.length})
            </button>
            {labs.map((l) => (
              <button key={l} className={`chip ${filter === l ? "on" : ""}`} onClick={() => setFilter(l)}>
                {LAB_META[l]?.label || l} ({projects.filter((p) => p.lab === l).length})
              </button>
            ))}
          </div>

          {/* Bulk Select Control Bar */}
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginLeft: "auto" }}>
            <button
              className="btn ghost xs"
              onClick={toggleSelectAll}
              style={{ display: "inline-flex", alignItems: "center", gap: 6, fontWeight: 600 }}
            >
              {allShownSelected ? <CheckSquare size={13} color="#3b82f6" /> : <Square size={13} />}
              <span>{allShownSelected ? "Deselect All" : "Select All"}</span>
            </button>
            {selectedIds.size > 0 && (
              <span className="badge" style={{ background: "rgba(59,130,246,0.15)", color: "#38bdf8", fontWeight: 700 }}>
                {selectedIds.size} Selected
              </span>
            )}
          </div>
        </div>
      )}

      {loading ? (
        <div className="note">Loading…</div>
      ) : projects.length === 0 ? (
        <div className="card">
          <div className="card-b" style={{ textAlign: "center", padding: 40 }}>
            <div style={{ fontSize: 34, opacity: 0.4 }}>◇</div>
            <p className="page-sub" style={{ margin: "10px 0 4px" }}>No saved projects yet.</p>
            <div className="note">Open any lab, build something, and hit <b>💾 Save</b>. It&apos;ll show up here.</div>
          </div>
        </div>
      ) : (
        <div className="proj-grid">
          {shown.map((p) => {
            const m = LAB_META[p.lab] || { label: p.lab, href: "/dashboard", icon: "◆" };
            const isSelected = selectedIds.has(p.id);
            return (
              <div
                key={p.id}
                className={`card proj-card ${isSelected ? "selected" : ""}`}
                style={{
                  position: "relative",
                  transition: "all 0.15s ease",
                  border: isSelected ? "1.5px solid #3b82f6" : "1px solid var(--border)",
                  background: isSelected ? "linear-gradient(180deg, rgba(59,130,246,0.08) 0%, rgba(59,130,246,0.02) 100%)" : "var(--panel)",
                  boxShadow: isSelected ? "0 0 14px rgba(59,130,246,0.25)" : "none",
                }}
              >
                <div className="card-b">
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                    {/* Custom Selection Checkbox */}
                    <div
                      onClick={() => toggleSelect(p.id)}
                      style={{
                        width: 20,
                        height: 20,
                        borderRadius: 5,
                        border: isSelected ? "1.5px solid #3b82f6" : "1.5px solid var(--border)",
                        background: isSelected ? "#3b82f6" : "rgba(255,255,255,0.04)",
                        display: "grid",
                        placeItems: "center",
                        cursor: "pointer",
                        flex: "none",
                        color: "#ffffff",
                        transition: "all 0.15s ease",
                      }}
                      title={isSelected ? "Deselect project" : "Select project"}
                    >
                      {isSelected && <Check size={12} strokeWidth={3} />}
                    </div>

                    <span className="proj-ic">{m.icon}</span>
                    <span className="badge">{m.label}</span>
                    {p.published && <span className="badge" style={{ color: "#3b9e5f", borderColor: "#3b9e5f55" }}>● published</span>}
                    <span className="spacer" style={{ flex: 1 }} />
                    <span className="note" style={{ fontSize: 11 }}>{when(p)}</span>
                  </div>

                  <div
                    className="proj-name"
                    title={p.name}
                    style={{ fontWeight: 700, fontSize: 14, cursor: "pointer" }}
                    onClick={() => toggleSelect(p.id)}
                  >
                    {p.name}
                  </div>

                  <div className="acts" style={{ marginTop: 14, display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
                    <Link className="btn sm" href={`${m.href}?project=${p.id}`} style={{ padding: "4px 12px" }}>
                      Open
                    </Link>
                    {(p.lab === "agent" || p.lab === "agent-nat" || p.lab === "rag") && (
                      p.published ? (
                        <>
                          {p.lab !== "rag" && <Link className="btn ghost sm" href={`/workroom/${p.id}`}>Open in Workroom</Link>}
                          <button className="btn ghost sm" disabled={busy === p.id} onClick={() => togglePublish(p)}>
                            {p.lab === "rag" ? "Undeploy" : "Unpublish"}
                          </button>
                        </>
                      ) : (
                        <button className="btn ghost sm" disabled={busy === p.id} onClick={() => togglePublish(p)}>
                          {p.lab === "rag" ? "🚀 Deploy" : "🚀 Publish"}
                        </button>
                      )
                    )}
                    <button className="btn ghost sm" disabled={busy === p.id} onClick={() => rename(p)}>
                      Rename
                    </button>
                    <button className="btn ghost sm danger" disabled={busy === p.id} onClick={() => remove(p)}>
                      Delete
                    </button>
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
