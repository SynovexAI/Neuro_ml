import { redirect } from "next/navigation";
import { desc, eq, gte, sql } from "drizzle-orm";
import { getSessionUser } from "@/lib/auth";
import { db } from "@/lib/db";
import { usage, users, auditLog } from "@/lib/db/schema";
import { DEFAULT_MONTHLY_TOKEN_LIMIT } from "@/lib/usage";
import Shell from "@/components/Shell";
import UsageDashboard from "@/components/UsageDashboard";

export const dynamic = "force-dynamic";

// Each aggregate is guarded so a not-yet-migrated table can't 500 the page.
async function safe<T>(fn: () => Promise<T>, fallback: T): Promise<T> {
  try { return await fn(); } catch { return fallback; }
}
const dayStr = (v: unknown) => v instanceof Date ? v.toISOString().slice(0, 10) : String(v).slice(0, 10);
const iso = (v: unknown) => v instanceof Date ? v.toISOString() : (v ? String(v) : null);

export default async function UsagePage() {
  const user = await getSessionUser();
  if (!user) redirect("/login");
  if (user.role !== "admin") redirect("/dashboard");

  const now = new Date();
  const monthStart = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), 1));
  const since14 = new Date(now.getTime() - 13 * 86_400_000);

  const totals = await safe(() => db.select({
    t: sql<number>`COALESCE(SUM(${usage.totalTokens}), 0)`,
    c: sql<number>`COUNT(*)`,
    est: sql<number>`COALESCE(SUM(CASE WHEN ${usage.estimated} THEN ${usage.totalTokens} ELSE 0 END), 0)`,
  }).from(usage).where(gte(usage.ts, monthStart)).then((r) => r[0]), { t: 0, c: 0, est: 0 });

  const byUserRaw = await safe(() => db.select({
    email: users.email, name: users.name, role: users.role, limit: users.monthlyTokenLimit,
    t: sql<number>`SUM(${usage.totalTokens})`, c: sql<number>`COUNT(*)`,
  }).from(usage).leftJoin(users, eq(users.id, usage.userId)).where(gte(usage.ts, monthStart))
    .groupBy(usage.userId, users.email, users.name, users.role, users.monthlyTokenLimit)
    .orderBy(desc(sql`SUM(${usage.totalTokens})`)).limit(12), [] as { email: string | null; name: string | null; role: string | null; limit: number | null; t: number; c: number }[]);

  const byLabRaw = await safe(() => db.select({ lab: usage.lab, t: sql<number>`SUM(${usage.totalTokens})` })
    .from(usage).where(gte(usage.ts, monthStart)).groupBy(usage.lab).orderBy(desc(sql`SUM(${usage.totalTokens})`)), [] as { lab: string | null; t: number }[]);

  const byModelRaw = await safe(() => db.select({ model: usage.model, t: sql<number>`SUM(${usage.totalTokens})`, c: sql<number>`COUNT(*)` })
    .from(usage).where(gte(usage.ts, monthStart)).groupBy(usage.model).orderBy(desc(sql`SUM(${usage.totalTokens})`)).limit(10), [] as { model: string | null; t: number; c: number }[]);

  const byDayRaw = await safe(() => db.select({ day: sql<string>`DATE(${usage.ts})`, t: sql<number>`SUM(${usage.totalTokens})` })
    .from(usage).where(gte(usage.ts, since14)).groupBy(sql`DATE(${usage.ts})`).orderBy(sql`DATE(${usage.ts})`), [] as { day: string; t: number }[]);

  const auditRaw = await safe(() => db.select().from(auditLog).orderBy(desc(auditLog.ts)).limit(40), [] as (typeof auditLog.$inferSelect)[]);

  const data = {
    total: { tokens: Number(totals?.t ?? 0), calls: Number(totals?.c ?? 0), estimated: Number(totals?.est ?? 0) },
    byUser: byUserRaw.map((r) => ({ email: r.email ?? "—", name: r.name, role: r.role ?? "student", limit: r.limit, tokens: Number(r.t), calls: Number(r.c) })),
    byLab: byLabRaw.map((r) => ({ lab: r.lab ?? "chat", tokens: Number(r.t) })),
    byModel: byModelRaw.map((r) => ({ model: r.model ?? "—", tokens: Number(r.t), calls: Number(r.c) })),
    byDay: byDayRaw.map((r) => ({ day: dayStr(r.day), tokens: Number(r.t) })),
    audit: auditRaw.map((r) => ({ id: r.id, ts: iso(r.ts), event: r.event, userId: r.userId, detail: r.detail as Record<string, unknown> | null })),
    defaultLimit: DEFAULT_MONTHLY_TOKEN_LIMIT,
  };

  return (
    <Shell user={user} title="Admin · Usage & Monitoring">
      <div className="eyebrow">Admin</div>
      <h2 className="page-h">Usage &amp; Monitoring</h2>
      <p className="page-sub">Token spend this month, per-user and per-model breakdowns, and a live event log — so you&apos;re not flying blind.</p>
      <UsageDashboard data={data} />
    </Shell>
  );
}
