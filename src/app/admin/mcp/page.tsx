import { redirect } from "next/navigation";
import { getSessionUser } from "@/lib/auth";
import Shell from "@/components/Shell";
import McpManager from "@/components/McpManager";

export default async function McpPage() {
  const user = await getSessionUser();
  if (!user) redirect("/login");
  if (user.role !== "admin") redirect("/dashboard");
  return (
    <Shell user={user} title="Admin · MCP servers">
      <div className="eyebrow">Admin</div>
      <h2 className="page-h">MCP servers</h2>
      <p className="page-sub">Connect Model Context Protocol servers once here; the tools from enabled servers become available to agents. Secrets are encrypted and never shown again.</p>
      <McpManager />
    </Shell>
  );
}
