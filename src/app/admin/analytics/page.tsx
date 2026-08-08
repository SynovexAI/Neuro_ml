import { redirect } from "next/navigation";
import Link from "next/link";
import { getSessionUser } from "@/lib/auth";
import Shell from "@/components/Shell";
import AnalyticsDashboard from "@/components/AnalyticsDashboard";

export const dynamic = "force-dynamic";

export default async function AdminAnalyticsPage() {
  const user = await getSessionUser();
  if (!user) redirect("/login");
  if (user.role !== "admin") redirect("/dashboard");
  return (
    <Shell user={user} title="Admin · Analytics">
      <div className="eyebrow">Control Room</div>
      <h2 className="page-h">Analytics</h2>
      <p className="page-sub">Server-side, in-app traffic — page views, active users, top pages — computed from your own database (works on any host). LLM usage &amp; monitoring is on the Usage page.</p>

      <div className="card" style={{ marginBottom: 16 }}>
        <div className="card-h"><span className="t">📈 Traffic — in-app</span></div>
        <div className="card-b"><AnalyticsDashboard /></div>
      </div>

      <div className="card">
        <div className="card-h"><span className="t">📊 Deeper views</span></div>
        <div className="card-b">
          <p className="note" style={{ lineHeight: 1.6 }}>More detail lives on dedicated pages — all from your own database:</p>
          <div className="row" style={{ gap: 8, flexWrap: "wrap" }}>
            <Link className="btn sm" href="/admin/usage">Usage &amp; Monitoring →</Link>
            <Link className="btn ghost sm" href="/admin/agents">Agent analytics →</Link>
            <Link className="btn ghost sm" href="/admin/audit">Audit log →</Link>
            <Link className="btn ghost sm" href="/admin/storage">Storage →</Link>
          </div>
        </div>
      </div>
    </Shell>
  );
}
