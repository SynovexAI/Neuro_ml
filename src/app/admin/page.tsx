import { redirect } from "next/navigation";
import Link from "next/link";
import { eq, gte, sql } from "drizzle-orm";
import { getSessionUser } from "@/lib/auth";
import { db } from "@/lib/db";
import { users, providers, usage, agentRuns, mcpServers, knowledgeBases } from "@/lib/db/schema";
import Shell from "@/components/Shell";
import HealthBadge from "@/components/HealthBadge";

export const dynamic = "force-dynamic";
async function one<T>(fn: () => Promise<T>, fb: T): Promise<T> { try { return await fn(); } catch { return fb; } }
const cnt = (rows: { c: number }[]) => Number(rows[0]?.c ?? 0);

export default async function AdminHome() {
  const user = await getSessionUser();
  if (!user) redirect("/login");
  if (user.role !== "admin") redirect("/dashboard");

  const now = new Date();
  const monthStart = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), 1));

  const [uAll, uPending, provAll, provOn, runs, monthTok, mcpOn, kbs] = await Promise.all([
    one(() => db.select({ c: sql<number>`COUNT(*)` }).from(users).then(cnt), 0),
    one(() => db.select({ c: sql<number>`COUNT(*)` }).from(users).where(eq(users.status, "pending")).then(cnt), 0),
    one(() => db.select({ c: sql<number>`COUNT(*)` }).from(providers).then(cnt), 0),
    one(() => db.select({ c: sql<number>`COUNT(*)` }).from(providers).where(eq(providers.enabled, true)).then(cnt), 0),
    one(() => db.select({ c: sql<number>`COUNT(*)` }).from(agentRuns).then(cnt), 0),
    one(() => db.select({ t: sql<number>`COALESCE(SUM(${usage.totalTokens}),0)` }).from(usage).where(gte(usage.ts, monthStart)).then((r) => Number(r[0]?.t ?? 0)), 0),
    one(() => db.select({ c: sql<number>`COUNT(*)` }).from(mcpServers).where(eq(mcpServers.enabled, true)).then(cnt), 0),
    one(() => db.select({ c: sql<number>`COUNT(*)` }).from(knowledgeBases).then(cnt), 0),
  ]);

  const fmt = (n: number) => n.toLocaleString();

  return (
    <Shell user={user} title="Control Room · Overview">
      <div className="lab-head">
        <div><div className="eyebrow">Control Room</div><h2 className="page-h">Overview</h2><p className="page-sub" style={{ margin: 0 }}>Platform health, usage, and access at a glance.</p></div>
        <div className="acts"><HealthBadge /></div>
      </div>

      <div className="cv-summary" style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 20 }}>
        <div className="metric"><span className="v">{fmt(uAll)}</span><span className="k">Users</span></div>
        <div className="metric"><span className="v" style={{ color: uPending ? "#f59e0b" : undefined }}>{fmt(uPending)}</span><span className="k">Pending approval</span></div>
        <div className="metric"><span className="v">{provOn}/{provAll}</span><span className="k">Providers enabled</span></div>
        <div className="metric"><span className="v">{fmt(monthTok)}</span><span className="k">Tokens this month</span></div>
        <div className="metric"><span className="v">{fmt(runs)}</span><span className="k">Agent runs</span></div>
        <div className="metric"><span className="v">{mcpOn}</span><span className="k">MCP servers</span></div>
        <div className="metric"><span className="v">{fmt(kbs)}</span><span className="k">Knowledge bases</span></div>
      </div>

      {uPending > 0 && <div className="warnbar" style={{ marginBottom: 18 }}>{uPending} account{uPending > 1 ? "s are" : " is"} awaiting approval — <Link href="/admin/users" style={{ color: "inherit", textDecoration: "underline" }}>review in Users</Link>.</div>}

      <div className="cards">
        <Link href="/admin/providers" className="lab-card"><h3>Providers &amp; models</h3><p>Configure LLM providers and save an encrypted API key used platform-wide.</p><div className="go">Open →</div></Link>
        <Link href="/admin/users" className="lab-card"><h3>Users</h3><p>Approve sign-ups, set roles, suspend accounts, and set per-student token budgets.</p><div className="go">Open →</div></Link>
        <Link href="/admin/usage" className="lab-card"><h3>Usage &amp; Monitoring</h3><p>Token spend, per-user &amp; per-model breakdowns, daily trend, health, audit log.</p><div className="go">Open →</div></Link>
        <Link href="/admin/agents" className="lab-card"><h3>Agent analytics</h3><p>Every agent run — success rate, iterations, tool usage, tokens, cost, latency.</p><div className="go">Open →</div></Link>
        <Link href="/admin/mcp" className="lab-card"><h3>MCP servers</h3><p>Connect Model Context Protocol servers; their tools become available to agents.</p><div className="go">Open →</div></Link>
      </div>
    </Shell>
  );
}
