import { redirect } from "next/navigation";
import Link from "next/link";
import { desc, eq, and, gte, sql } from "drizzle-orm";
import { getSessionUser } from "@/lib/auth";
import { db } from "@/lib/db";
import { projects, usage, agentRuns, knowledgeBases, mcpServers } from "@/lib/db/schema";
import Shell from "@/components/Shell";
import ActivityChart from "@/components/ActivityChart";

const LAB_HREF: Record<string, string> = {
  prompting: "/labs/prompting", rag: "/labs/rag", agent: "/labs/agent",
  ml: "/labs/ml", dl: "/labs/dl", etl: "/labs/etl",
};

const QUICK = [
  { href: "/labs/agent", icon: "◈", title: "Build an agent", desc: "ReAct or workflow, with tools" },
  { href: "/workroom", icon: "◐", title: "Open Workroom", desc: "Chat with & deploy your agents" },
  { href: "/labs/rag", icon: "❖", title: "Build a RAG bot", desc: "Chunk, embed, retrieve" },
  { href: "/kb", icon: "▤", title: "New knowledge base", desc: "Files or URLs → vectors" },
];

const LABS = [
    { href: "/labs/etl", icon: "⚙", title: "ETL Lab", desc: "Extract, transform and load data with a visual workflow.", status: "Active" },
    { href: "/labs/agent", icon: "🤖", title: "Agent Lab", desc: "Compose AI agents using tools, data and memory.", status: "Beta" },
    { href: "/labs/streaming", icon: "⚡", title: "Streaming Lab", desc: "Stream live data into your pipeline in real time.", status: "New" },
    { href: "/labs/reverse-etl", icon: "🔄", title: "Reverse ETL Lab", desc: "Sync transformed data back into upstream systems.", status: "New" },
    { href: "/labs/rag", icon: "📚", title: "RAG Lab", desc: "Build retrieval-augmented bots grounded in your docs.", status: "Active" },
    { href: "/labs/ml", icon: "🧠", title: "ML Lab", desc: "Train models, evaluate features, and inspect metrics.", status: "Beta" },
];

export default async function Dashboard() {
  const user = await getSessionUser();
  if (!user) redirect("/login");

  const one = async <T,>(fn: () => Promise<T>, fb: T): Promise<T> => { try { return await fn(); } catch { return fb; } };
  const cnt = (rows: { c: number }[]) => Number(rows[0]?.c ?? 0);
  const monthStart = new Date(Date.UTC(new Date().getUTCFullYear(), new Date().getUTCMonth(), 1));
  const since14 = new Date(Date.now() - 13 * 86400000);

  const [recent, nProjects, nRuns, monthTok, nKbs, nMcp, dailyTok, dailyRun] = await Promise.all([
    one(() => db.select({ id: projects.id, lab: projects.lab, name: projects.name }).from(projects).where(eq(projects.userId, user.id)).orderBy(desc(projects.updatedAt)).limit(6), [] as { id: string; lab: string; name: string }[]),
    one(() => db.select({ c: sql<number>`COUNT(*)` }).from(projects).where(eq(projects.userId, user.id)).then(cnt), 0),
    one(() => db.select({ c: sql<number>`COUNT(*)` }).from(agentRuns).where(eq(agentRuns.userId, user.id)).then(cnt), 0),
    one(() => db.select({ t: sql<number>`COALESCE(SUM(${usage.totalTokens}),0)` }).from(usage).where(and(eq(usage.userId, user.id), gte(usage.ts, monthStart))).then((r) => Number(r[0]?.t ?? 0)), 0),
    one(() => db.select({ c: sql<number>`COUNT(*)` }).from(knowledgeBases).where(eq(knowledgeBases.userId, user.id)).then(cnt), 0),
    one(() => db.select({ c: sql<number>`COUNT(*)` }).from(mcpServers).where(and(eq(mcpServers.userId, user.id), eq(mcpServers.enabled, true))).then(cnt), 0),
    one(() => db.select({ day: sql<string>`DATE(${usage.ts})`, t: sql<number>`SUM(${usage.totalTokens})` }).from(usage).where(and(eq(usage.userId, user.id), gte(usage.ts, since14))).groupBy(sql`DATE(${usage.ts})`), [] as { day: string; t: number }[]),
    one(() => db.select({ day: sql<string>`DATE(${agentRuns.ts})`, c: sql<number>`COUNT(*)` }).from(agentRuns).where(and(eq(agentRuns.userId, user.id), gte(agentRuns.ts, since14))).groupBy(sql`DATE(${agentRuns.ts})`), [] as { day: string; c: number }[]),
  ]);

  const kfmt = (n: number) => n >= 1000 ? `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k` : String(n);
  const dayStr = (d: string) => d.slice(0, 10);
  const days: string[] = []; for (let i = 13; i >= 0; i--) days.push(new Date(Date.now() - i * 86400000).toISOString().slice(0, 10));
  const tokMap = Object.fromEntries(dailyTok.map((r) => [dayStr(r.day), Number(r.t)]));
  const runMap = Object.fromEntries(dailyRun.map((r) => [dayStr(r.day), Number(r.c)]));
  const tokensSeries = days.map((d) => tokMap[d] || 0);
  const runsSeries = days.map((d) => runMap[d] || 0);

  const kpis: [string, string, string, string?][] = [
    ["🗂", String(nProjects), "PROJECTS", "var(--text)"],
    ["◈", String(nRuns), "AGENT RUNS", "var(--text)"],
    ["▤", kfmt(monthTok), "TOKENS / MO", "var(--text)"],
    ["🗄", String(nKbs), "KNOWLEDGE BASES", "var(--text)"],
    ["🔌", String(nMcp), "MCP TOOLS", "var(--text)"],
  ];

  return (
    <Shell user={user} title="Studio">
      <div className="hero-banner">
        <div className="eyebrow" style={{ marginBottom: 6, color: "var(--accent)", letterSpacing: "0.1em" }}>STUDIO</div>
        <h2>Welcome back{user.name ? `, ${user.name.split(" ")[0]}` : ""}</h2>
        <p>Start something new, pick up where you left off, or open a Lab — you build real models, agents and RAG systems and watch exactly how they work.</p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(5, minmax(0,1fr))", gap: 14, margin: "0 0 20px" }}>
        {kpis.map(([ic, v, k, col]) => (
          <div key={k} className="kpi-dash-card card-clickable">
            <div className="kpi-dash-icon">{ic}</div>
            <div className="kpi-dash-value" style={{ color: col ?? "var(--text)" }}>{v}</div>
            <div className="kpi-dash-label">{k}</div>
          </div>
        ))}
      </div>

      <div className="activity-glass" style={{ marginBottom: 24 }}><ActivityChart title="Your activity · last 14 days" days={days} tokens={tokensSeries} runs={runsSeries} /></div>

      <div className="sec-title" style={{ marginTop: 32 }}>QUICK START</div>
      <div className="cards" style={{ marginBottom: 24 }}>
        {QUICK.map((q) => (
          <Link key={q.href} href={q.href} className="lab-card quick-start-card">
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <span className="zi">{q.icon}</span>
              <h3 style={{ margin: 0, fontSize: 15 }}>{q.title}</h3>
            </div>
            <p style={{ margin: "4px 0 16px", color: "var(--muted)", fontSize: 13 }}>{q.desc}</p>
            <div className="go" style={{ marginTop: "auto" }}>Start →</div>
          </Link>
        ))}
      </div>

      {recent.length > 0 && (<>
        <div className="sec-title">Continue building</div>
        <div className="cards" style={{ marginBottom: 24 }}>
          {recent.map((p) => (
            <Link key={p.id} href={`${LAB_HREF[p.lab] || "/projects"}?project=${p.id}`} className="lab-card">
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}><span className="badge">{p.lab}</span></div>
              <h3 style={{ fontSize: 14, margin: "4px 0 0" }}>{p.name}</h3><div className="go">Open →</div>
            </Link>
          ))}
        </div>
      </>)}

      <div className="sec-title">LABS</div>
      <div className="lab-grid">
        {LABS.map((l) => (
          <Link key={l.href} href={l.href} className="lab-card fade-in">
            <div>
              <h3 className="lab-card-title" style={{ fontSize: 15, marginBottom: 4 }}>{l.title}</h3>
              <p className="lab-card-desc" style={{ fontSize: 13, color: "var(--muted)", marginBottom: 16 }}>{l.desc}</p>
            </div>
            <div className="lab-card-meta" style={{ justifyContent: "flex-start" }}>
              <span className="go">Open lab →</span>
            </div>
          </Link>
        ))}
      </div>
    </Shell>
  );
}
