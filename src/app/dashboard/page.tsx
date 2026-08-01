import { redirect } from "next/navigation";
import Link from "next/link";
import { desc, eq } from "drizzle-orm";
import { getSessionUser } from "@/lib/auth";
import { db } from "@/lib/db";
import { projects } from "@/lib/db/schema";
import Shell from "@/components/Shell";

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

  let recent: { id: string; lab: string; name: string }[] = [];
  try { recent = await db.select({ id: projects.id, lab: projects.lab, name: projects.name }).from(projects).where(eq(projects.userId, user.id)).orderBy(desc(projects.updatedAt)).limit(6); }
  catch { recent = []; }

  return (
    <Shell user={user} title="Studio">
      <div className="hero">
        <h2>Welcome{user.name ? `, ${user.name}` : ""} 👋</h2>
        <p>Your Studio — start something new, pick up where you left off, or open a Lab. You build real models, agents, and RAG systems and watch exactly how they work.</p>
      </div>

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
