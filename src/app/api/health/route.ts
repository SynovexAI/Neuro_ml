import { NextResponse } from "next/server";
import { sql } from "drizzle-orm";
import { db } from "@/lib/db";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Liveness + DB-readiness probe for load balancers / uptime monitors.
export async function GET() {
  const started = Date.now();
  try {
    await db.execute(sql`SELECT 1`);
    return NextResponse.json({ ok: true, db: "up", ms: Date.now() - started });
  } catch {
    return NextResponse.json({ ok: false, db: "down", ms: Date.now() - started }, { status: 503 });
  }
}
