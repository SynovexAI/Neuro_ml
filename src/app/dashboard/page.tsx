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
  { href: "/labs/prompting", title: "Prompting", desc: "Write, tune and compare prompts." },
  { href: "/labs/rag", title: "RAG", desc: "Doc Q&A — see the pipeline run." },
  { href: "/labs/agent", title: "Agent", desc: "Build agents visually or by form." },
  { href: "/labs/ml", title: "ML", desc: "EDA, preprocess, train in-browser." },
  { href: "/labs/dl", title: "DL", desc: "Build a net and watch it train." },
  { href: "/labs/etl", title: "ETL", desc: "A Spark-style data pipeline." },
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
    ["🗂", String(nProjects), "Projects", "var(--accent-strong)"],
    ["◈", String(nRuns), "Agent runs", "var(--accent)"],
    ["▤", kfmt(monthTok), "Tokens / mo", "var(--sky)"],
    ["🗄", String(nKbs), "Knowledge bases", undefined],
    ["🔌", String(nMcp), "MCP tools", "var(--good)"],
  ];

  return (
    <Shell user={user} title="Studio">
      <div className="hero">
        <div className="eyebrow" style={{ marginBottom: 6 }}>Studio</div>
        <h2>Welcome back{user.name ? `, ${user.name.split(" ")[0]}` : ""}</h2>
        <p>Start something new, pick up where you left off, or open a Lab — you build real models, agents and RAG systems and watch exactly how they work.</p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(5, minmax(0,1fr))", gap: 12, margin: "0 0 16px" }}>
        {kpis.map(([ic, v, k, col]) => <div key={k} style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 12, padding: "13px 15px" }}>
          <div style={{ width: 26, height: 26, borderRadius: 7, display: "grid", placeItems: "center", fontSize: 13, marginBottom: 9, background: "var(--accent-weak)", color: "var(--accent-strong)" }}>{ic}</div>
          <div style={{ fontFamily: "var(--mono)", fontSize: 21, fontWeight: 600, letterSpacing: "-.02em", color: col }}>{v}</div>
          <div style={{ fontSize: 9, textTransform: "uppercase", letterSpacing: ".05em", color: "var(--faint)", marginTop: 3 }}>{k}</div>
        </div>)}
      </div>

      <div style={{ marginBottom: 22 }}><ActivityChart title="Your activity · last 14 days" days={days} tokens={tokensSeries} runs={runsSeries} /></div>

      <div className="sec-title">Quick start</div>
      <div className="cards" style={{ marginBottom: 24 }}>
        {QUICK.map((q) => (
          <Link key={q.href} href={q.href} className="lab-card">
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}><span className="zi">{q.icon}</span><h3 style={{ margin: 0 }}>{q.title}</h3></div>
            <p>{q.desc}</p><div className="go">Start →</div>
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

      <div className="sec-title">Labs</div>
      <div className="cards">
        {LABS.map((l) => (
          <Link key={l.href} href={l.href} className="lab-card"><h3>{l.title} Lab</h3><p>{l.desc}</p><div className="go">Open lab →</div></Link>
        ))}
      </div>
    </Shell>
  );
}
