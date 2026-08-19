import { redirect } from "next/navigation";
import Link from "next/link";
import { eq, gte, sql } from "drizzle-orm";
import { getSessionUser } from "@/lib/auth";
import { db } from "@/lib/db";
import { users, providers, usage, agentRuns, mcpServers, knowledgeBases } from "@/lib/db/schema";
import Shell from "@/components/Shell";
import HealthBadge from "@/components/HealthBadge";
import OverviewCharts from "@/components/OverviewCharts";

export const dynamic = "force-dynamic";
async function one<T>(fn: () => Promise<T>, fb: T): Promise<T> { try { return await fn(); } catch { return fb; } }
const cnt = (rows: { c: number }[]) => Number(rows[0]?.c ?? 0);

export default async function AdminHome() {
  const user = await getSessionUser();
  if (!user) redirect("/login");
  if (user.role !== "admin") redirect("/dashboard");

  const now = new Date();
  const monthStart = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), 1));
  const since14 = new Date(Date.now() - 13 * 86400000);

  const [uAll, uPending, provAll, provOn, runs, monthTok, mcpOn, kbs, dailyTok, dailyRun, uByStatus, tokByLab] = await Promise.all([
    one(() => db.select({ c: sql<number>`COUNT(*)` }).from(users).then(cnt), 0),
    one(() => db.select({ c: sql<number>`COUNT(*)` }).from(users).where(eq(users.status, "pending")).then(cnt), 0),
    one(() => db.select({ c: sql<number>`COUNT(*)` }).from(providers).then(cnt), 0),
    one(() => db.select({ c: sql<number>`COUNT(*)` }).from(providers).where(eq(providers.enabled, true)).then(cnt), 0),
    one(() => db.select({ c: sql<number>`COUNT(*)` }).from(agentRuns).then(cnt), 0),
    one(() => db.select({ t: sql<number>`COALESCE(SUM(${usage.totalTokens}),0)` }).from(usage).where(gte(usage.ts, monthStart)).then((r) => Number(r[0]?.t ?? 0)), 0),
    one(() => db.select({ c: sql<number>`COUNT(*)` }).from(mcpServers).where(eq(mcpServers.enabled, true)).then(cnt), 0),
    one(() => db.select({ c: sql<number>`COUNT(*)` }).from(knowledgeBases).then(cnt), 0),
    one(() => db.select({ day: sql<string>`DATE(${usage.ts})`, t: sql<number>`SUM(${usage.totalTokens})` }).from(usage).where(gte(usage.ts, since14)).groupBy(sql`DATE(${usage.ts})`), [] as { day: string; t: number }[]),
    one(() => db.select({ day: sql<string>`DATE(${agentRuns.ts})`, c: sql<number>`COUNT(*)` }).from(agentRuns).where(gte(agentRuns.ts, since14)).groupBy(sql`DATE(${agentRuns.ts})`), [] as { day: string; c: number }[]),
    one(() => db.select({ status: users.status, c: sql<number>`COUNT(*)` }).from(users).groupBy(users.status), [] as { status: string; c: number }[]),
    one(() => db.select({ lab: usage.lab, t: sql<number>`SUM(${usage.totalTokens})` }).from(usage).where(gte(usage.ts, monthStart)).groupBy(usage.lab).orderBy(sql`SUM(${usage.totalTokens}) DESC`), [] as { lab: string | null; t: number }[]),
  ]);

  const fmt = (n: number) => n.toLocaleString();
  const kfmt = (n: number) => n >= 1000 ? `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k` : String(n);
  const dayStr = (d: string | Date) => (typeof d === "string" ? d.slice(0, 10) : new Date(d).toISOString().slice(0, 10));
  const days: string[] = []; for (let i = 13; i >= 0; i--) days.push(new Date(Date.now() - i * 86400000).toISOString().slice(0, 10));
  const tokMap = Object.fromEntries(dailyTok.map((r) => [dayStr(r.day), Number(r.t)]));
  const runMap = Object.fromEntries(dailyRun.map((r) => [dayStr(r.day), Number(r.c)]));
  const tokensSeries = days.map((d) => tokMap[d] || 0);
  const runsSeries = days.map((d) => runMap[d] || 0);
  const usersByStatus = ["active", "pending", "suspended"].map((s) => ({ status: s, count: Number(uByStatus.find((r) => r.status === s)?.c ?? 0) })).filter((u) => u.count > 0);
  const tokensByLab = tokByLab.map((r) => ({ lab: r.lab ?? "chat", tokens: Number(r.t) }));

  const tiles: [string, string, string, string?][] = [
    ["👥", fmt(uAll), "Users", "var(--accent-strong)"],
    ["⏳", fmt(uPending), "Pending", uPending ? "var(--warn)" : undefined],
    ["◆", `${provOn}/${provAll}`, "Providers on", undefined],
    ["▤", kfmt(monthTok), "Tokens / mo", "var(--sky)"],
    ["◈", fmt(runs), "Agent runs", "var(--accent)"],
    ["🔌", String(mcpOn), "MCP servers", "var(--good)"],
    ["🗄", fmt(kbs), "Knowledge bases", undefined],
  ];
  const links: [string, string, string, string, string][] = [
    ["◆", "var(--accent-strong)", "/admin/providers", "Providers & models", "Configure LLM providers and save an encrypted key used platform-wide."],
    ["👥", "var(--accent-strong)", "/admin/users", "Users", "Approve sign-ups, set roles, suspend, and set per-student token budgets."],
    ["▤", "var(--sky)", "/admin/usage", "Usage & Monitoring", "Token spend, per-user & per-model breakdowns, daily trend, audit log."],
    ["◈", "var(--purple)", "/admin/agents", "Agent analytics", "Every agent run — success rate, tools, tokens, cost, latency."],
    ["🔌", "var(--good)", "/admin/mcp", "MCP servers", "Connect MCP servers; their tools become available to your agents."],
  ];

  return (
    <Shell user={user} title="Control Room · Overview">
      <div className="lab-head">
        <div><div className="eyebrow">Control Room</div><h2 className="page-h">Overview</h2><p className="page-sub" style={{ margin: 0 }}>Platform health, usage, and access at a glance.</p></div>
        <div className="acts"><HealthBadge /></div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(7, minmax(0,1fr))", gap: 11, marginBottom: 16 }}>
        {tiles.map(([ic, v, k, col]) => <div key={k} style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 12, padding: "13px 14px" }}>
          <div style={{ width: 26, height: 26, borderRadius: 7, display: "grid", placeItems: "center", fontSize: 13, marginBottom: 9, background: "var(--accent-weak)", color: "var(--accent-strong)" }}>{ic}</div>
          <div style={{ fontFamily: "var(--mono)", fontSize: 20, fontWeight: 600, letterSpacing: "-.02em", color: col }}>{v}</div>
          <div style={{ fontSize: 8.5, textTransform: "uppercase", letterSpacing: ".04em", color: "var(--faint)", marginTop: 3 }}>{k}</div>
        </div>)}
      </div>

      {uPending > 0 && <div className="warnbar" style={{ marginBottom: 16 }}>{uPending} account{uPending > 1 ? "s are" : " is"} awaiting approval — <Link href="/admin/users" style={{ color: "inherit", textDecoration: "underline" }}>review in Users</Link>.</div>}

      <OverviewCharts days={days} tokens={tokensSeries} runs={runsSeries} usersByStatus={usersByStatus} tokensByLab={tokensByLab} />

      <div className="sec-title">Manage</div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(232px, 1fr))", gap: 14 }}>
        {links.map(([ic, col, href, title, desc]) => <Link key={href} href={href} className="lab-card" style={{ display: "flex", flexDirection: "column" }}>
          <div style={{ width: 32, height: 32, borderRadius: 9, display: "grid", placeItems: "center", fontSize: 15, marginBottom: 10, background: "var(--accent-weak)", color: col }}>{ic}</div>
          <h3 style={{ margin: "0 0 5px" }}>{title}</h3>
          <p style={{ margin: 0, flex: 1 }}>{desc}</p>
          <div className="go" style={{ marginTop: 10 }}>Open →</div>
        </Link>)}
      </div>
    </Shell>
  );
}
