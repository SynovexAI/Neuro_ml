"use client";

import { useEffect, useState } from "react";

export default function HealthBadge() {
  const [s, setS] = useState<{ ok: boolean; db: string; ms?: number } | null>(null);
  const [err, setErr] = useState(false);
  useEffect(() => { fetch("/api/health").then((r) => r.json()).then(setS).catch(() => setErr(true)); }, []);
  const up = s?.ok && !err;
  return (
    <span className={`badge ${up ? "good" : "warn"}`} title={s ? `db ${s.db}` : ""}>
      {err ? "health: unreachable" : s ? (up ? `healthy · db ${s.ms}ms` : "degraded") : "checking…"}
    </span>
  );
}
