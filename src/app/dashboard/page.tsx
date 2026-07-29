import { redirect } from "next/navigation";
import Link from "next/link";
import { getSessionUser } from "@/lib/auth";
import Shell from "@/components/Shell";

const LABS = [
  { href: "/labs/prompting", title: "Prompting Lab", desc: "Write, tune and compare prompts with live token & latency readouts." },
  { href: "/labs/rag", title: "RAG Lab", desc: "Build doc Q&A — chunk, embed, retrieve, and see the pipeline run." },
  { href: "/labs/agent", title: "Agent Lab", desc: "Build agents visually, by form, or from a prompt. Then run them." },
  { href: "/labs/ml", title: "ML Lab", desc: "Connect data, EDA, preprocess, tune a model — trains in your browser." },
  { href: "/labs/dl", title: "DL Lab", desc: "Build a neural net, watch it train, then test with a live demo." },
  { href: "/labs/etl", title: "ETL Lab", desc: "Build a Kafka + Spark-style data pipeline and watch records flow." },
];

export default async function Dashboard() {
  const user = await getSessionUser();
  if (!user) redirect("/login");
  return (
    <Shell user={user} title="Dashboard">
      <div className="hero">
        <h2>Welcome{user.name ? `, ${user.name}` : ""} — pick a Lab and start building.</h2>
        <p>Six practical Labs plus a Compose studio. No lessons — you build a real model, agent, or RAG system, and see exactly how it works.</p>
      </div>
      <div className="sec-title">Labs</div>
      <div className="cards">
        {LABS.map((l) => (
          <Link key={l.href} href={l.href} className="lab-card">
            <h3>{l.title}</h3><p>{l.desc}</p><div className="go">Open lab →</div>
          </Link>
        ))}
      </div>
      <div className="sec-title">Studio</div>
      <div className="cards">
        <Link href="/compose" className="lab-card"><h3>Compose</h3><p>Chain your builds — an agent that uses your RAG bot as a tool.</p><div className="go">Open →</div></Link>
        <Link href="/templates" className="lab-card"><h3>Templates</h3><p>One-click starter projects for every Lab.</p><div className="go">Browse →</div></Link>
      </div>
    </Shell>
  );
}
