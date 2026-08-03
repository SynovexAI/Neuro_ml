import { redirect } from "next/navigation";
import { getSessionUser } from "@/lib/auth";
import Shell from "@/components/Shell";
import McpManager from "@/components/McpManager";

// Per-user Studio feature (each user connects their own MCP servers). Lives under
// /studio — not /admin — because it is intentionally not admin-gated.
export default async function McpPage() {
  const user = await getSessionUser();
  if (!user) redirect("/login");
  return (
    <Shell user={user} title="MCP servers">
      <div className="eyebrow">Studio</div>
      <h2 className="page-h">MCP servers</h2>
      <p className="page-sub">Give your agents tools. Free web, research &amp; memory servers connect in one click; connect your own database to chat with your data. Keys are encrypted and never shown again.</p>
      <McpManager />
    </Shell>
  );
}
