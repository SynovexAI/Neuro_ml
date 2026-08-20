"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";

const CAP = 1024 * 1024 * 1024; // ~1 GB reference (Vercel Blob free tier)
const mb = (b: number) => (b < 1024 * 1024 ? `${(b / 1024).toFixed(0)} KB` : `${(b / 1024 / 1024).toFixed(0)} MB`);

// Admin-only banner: warns when object storage usage crosses 80% of the free-tier cap,
// so an admin can free space before users start hitting upload failures.
export default function StorageAlertBanner() {
  const path = usePathname() || "";
  const [pct, setPct] = useState<number | null>(null);
  const [used, setUsed] = useState(0);

  useEffect(() => {
    if (path.startsWith("/admin/storage")) { setPct(null); return; } // already managing it there
    let alive = true;
    fetch("/api/admin/storage").then((r) => (r.ok ? r.json() : null)).then((j) => {
      if (!alive || !j || !j.configured) { setPct(null); return; }
      setUsed(j.totalBytes || 0);
      setPct(((j.totalBytes || 0) / CAP) * 100);
    }).catch(() => {});
    return () => { alive = false; };
  }, [path]);

  if (pct == null || pct < 80) return null;
  const full = pct >= 100;
  return (
    <div className="warnbar" style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap", marginBottom: 14, borderColor: full ? "var(--crit)" : undefined }}>
      <span>{full ? "🔴" : "⚠"} <b>Storage {Math.min(pct, 100).toFixed(0)}% full</b> ({mb(used)} / ~1 GB){full ? " — uploads may start failing." : " — nearing the free-tier limit."} Delete old files to free space.</span>
      <Link href="/admin/storage" className="btn sm" style={{ marginLeft: "auto", whiteSpace: "nowrap" }}>Manage storage →</Link>
    </div>
  );
}
