import { redirect } from "next/navigation";
import Link from "next/link";
import { getSessionUser } from "@/lib/auth";
import Shell from "@/components/Shell";

export default async function AdminHome() {
  const user = await getSessionUser();
  if (!user) redirect("/login");
  if (user.role !== "admin") redirect("/dashboard");
  return (
    <Shell user={user} title="Admin">
      <div className="hero">
        <h2>Admin panel</h2>
        <p>Configure LLM providers &amp; API keys, manage student accounts, and watch token usage.</p>
      </div>
      <div className="cards">
        <Link href="/admin/providers" className="lab-card"><h3>Providers &amp; models</h3><p>Choose a provider, load its models, and save an encrypted API key used platform-wide.</p><div className="go">Open →</div></Link>
        <Link href="/admin/users" className="lab-card"><h3>Users</h3><p>Approve pending sign-ups, set roles, suspend accounts, and set per-student token budgets.</p><div className="go">Open →</div></Link>
        <Link href="/admin/usage" className="lab-card"><h3>Usage &amp; Monitoring</h3><p>Token spend this month, per-user &amp; per-model breakdowns, daily trend, health, and an audit event log.</p><div className="go">Open →</div></Link>
        <Link href="/admin/mcp" className="lab-card"><h3>MCP servers</h3><p>Connect Model Context Protocol servers (HTTP, SSE, or stdio). Their tools become available to agents.</p><div className="go">Open →</div></Link>
        <Link href="/admin/agents" className="lab-card"><h3>Agent analytics</h3><p>Every agent run — success rate, iterations, tool usage, tokens, estimated cost, and latency.</p><div className="go">Open →</div></Link>
      </div>
    </Shell>
  );
}
