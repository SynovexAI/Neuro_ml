import { redirect } from "next/navigation";
import { gte, sql } from "drizzle-orm";
import { getSessionUser } from "@/lib/auth";
import { db } from "@/lib/db";
import { users, usage } from "@/lib/db/schema";
import { DEFAULT_MONTHLY_TOKEN_LIMIT } from "@/lib/usage";
import Shell from "@/components/Shell";
import UsersManager from "@/components/UsersManager";

export default async function UsersPage() {
  const user = await getSessionUser();
  if (!user) redirect("/login");
  if (user.role !== "admin") redirect("/dashboard");

  const rows = await db.select({
    id: users.id, email: users.email, name: users.name, role: users.role, status: users.status,
    monthlyTokenLimit: users.monthlyTokenLimit,
  }).from(users);

  // Current-month token usage per user. Guarded so the page still renders if the
  // usage table hasn't been migrated yet.
  const now = new Date();
  const monthStart = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), 1));
  let byUser = new Map<string, number>();
  try {
    const agg = await db.select({ userId: usage.userId, t: sql<number>`COALESCE(SUM(${usage.totalTokens}), 0)` })
      .from(usage).where(gte(usage.ts, monthStart)).groupBy(usage.userId);
    byUser = new Map(agg.map((r) => [r.userId, Number(r.t)]));
  } catch { /* not migrated yet */ }
  const withUsage = rows.map((r) => ({ ...r, monthUsage: byUser.get(r.id) ?? 0 }));

  return (
    <Shell user={user} title="Admin · Users">
      <div className="eyebrow">Admin</div>
      <h2 className="page-h">Users</h2>
      <p className="page-sub">Approve pending student sign-ups, change roles, suspend accounts, and set each student&apos;s monthly LLM token budget. New sign-ups start as pending until you approve them.</p>
      <UsersManager initial={withUsage} meId={user.id} defaultLimit={DEFAULT_MONTHLY_TOKEN_LIMIT} />
    </Shell>
  );
}
