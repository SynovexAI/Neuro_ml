import { NextResponse } from "next/server";
import { desc, gte, sql } from "drizzle-orm";
import { db } from "@/lib/db";
import { pageViews, users, projects, knowledgeBases, agentRuns } from "@/lib/db/schema";
import { getSessionUser } from "@/lib/auth";
import { listFiles, storageConfigured } from "@/lib/storage";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// In-app traffic analytics (server-side, from our own page_views table).
export async function GET() {
  const u = await getSessionUser();
  if (!u || u.role !== "admin") return NextResponse.json({ error: "forbidden" }, { status: 403 });
  const since7 = new Date(Date.now() - 7 * 864e5);
  const since1 = new Date(Date.now() - 864e5);
  try {
    const [totalRow] = await db.select({ n: sql<number>`count(*)` }).from(pageViews).where(gte(pageViews.ts, since7));
    const [au24] = await db.select({ n: sql<number>`count(distinct ${pageViews.userId})` }).from(pageViews).where(gte(pageViews.ts, since1));
    const [au7] = await db.select({ n: sql<number>`count(distinct ${pageViews.userId})` }).from(pageViews).where(gte(pageViews.ts, since7));
    const perDay = await db.select({ day: sql<string>`DATE(${pageViews.ts})`, n: sql<number>`count(*)` })
      .from(pageViews).where(gte(pageViews.ts, since7)).groupBy(sql`DATE(${pageViews.ts})`).orderBy(sql`DATE(${pageViews.ts})`);
    const topPaths = await db.select({ path: pageViews.path, n: sql<number>`count(*)` })
      .from(pageViews).where(gte(pageViews.ts, since7)).groupBy(pageViews.path).orderBy(desc(sql`count(*)`)).limit(12);

    // Platform totals (server overview) — each guarded so a missing table doesn't break the page.
    const count1 = async (q: Promise<{ n: number }[]>) => { try { const [r] = await q; return Number(r?.n || 0); } catch { return 0; } };
    const totalUsers = await count1(db.select({ n: sql<number>`count(*)` }).from(users));
    const totalProjects = await count1(db.select({ n: sql<number>`count(*)` }).from(projects));
    const totalKbs = await count1(db.select({ n: sql<number>`count(*)` }).from(knowledgeBases));
    const runs7d = await count1(db.select({ n: sql<number>`count(*)` }).from(agentRuns).where(gte(agentRuns.ts, since7)));

    // Storage usage (Blob/R2). CAP ~1 GB reference for the % bar.
    let storage = { configured: storageConfigured(), bytes: 0, files: 0, pct: 0 };
    if (storage.configured) {
      try { const files = await listFiles(); const bytes = files.reduce((a, f) => a + (f.size || 0), 0); storage = { configured: true, bytes, files: files.length, pct: (bytes / (1024 * 1024 * 1024)) * 100 }; } catch { /* ignore */ }
    }

    return NextResponse.json({
      configured: true,
      views7d: Number(totalRow?.n || 0),
      activeUsers24h: Number(au24?.n || 0),
      activeUsers7d: Number(au7?.n || 0),
      perDay: perDay.map((r) => ({ day: String(r.day), n: Number(r.n) })),
      topPaths: topPaths.map((r) => ({ path: r.path, n: Number(r.n) })),
      totals: { users: totalUsers, projects: totalProjects, kbs: totalKbs, runs7d },
      storage,
    });
  } catch {
    return NextResponse.json({ configured: false, views7d: 0, activeUsers24h: 0, activeUsers7d: 0, perDay: [], topPaths: [] });
  }
}
