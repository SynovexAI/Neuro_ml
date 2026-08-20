import { redirect } from "next/navigation";
import { getSessionUser } from "@/lib/auth";
import Shell from "@/components/Shell";
import AuditLogViewer from "@/components/AuditLogViewer";

export const dynamic = "force-dynamic";

export default async function AdminAuditPage() {
  const user = await getSessionUser();
  if (!user) redirect("/login");
  if (user.role !== "admin") redirect("/dashboard");
  return (
    <Shell user={user} title="Admin · Audit log">
      <div className="eyebrow">Control Room</div>
      <h2 className="page-h">Audit log</h2>
      <p className="page-sub">Every security-relevant event — logins, sign-ups, quota hits, provider &amp; key changes, account deletions, MCP connections. Filter by type and review who did what, when.</p>
      <AuditLogViewer />
    </Shell>
  );
}
