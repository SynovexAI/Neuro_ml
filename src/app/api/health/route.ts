import { NextResponse } from "next/server";
import { sql } from "drizzle-orm";
import { db } from "@/lib/db";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Liveness + readiness probe for load balancers / uptime monitors.
// Checks the DB, and (best-effort) the NAT agent sidecar so the Control Room
// can surface when the agent runtime is down/sleeping.
async function natHealth(): Promise<"up" | "down" | "unconfigured"> {
  const url = process.env.NAT_SERVICE_URL;
  if (!url) return "unconfigured";
  try {
    const ctrl = new AbortController(); const t = setTimeout(() => ctrl.abort(), 4000);
    const r = await fetch(url.replace(/\/$/, "") + "/health", { signal: ctrl.signal }).finally(() => clearTimeout(t));
    return r.ok ? "up" : "down";
  } catch { return "down"; }
}

export async function GET() {
  const started = Date.now();
  let dbOk = true;
  try { await db.execute(sql`SELECT 1`); } catch { dbOk = false; }
  const nat = await natHealth();
  const ok = dbOk; // NAT being down degrades agents but the app is still live
  return NextResponse.json({ ok, db: dbOk ? "up" : "down", nat, ms: Date.now() - started }, { status: ok ? 200 : 503 });
}
