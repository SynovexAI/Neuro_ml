import { redirect } from "next/navigation";
import Link from "next/link";
import { getSessionUser } from "@/lib/auth";
import Shell from "@/components/Shell";

export const dynamic = "force-dynamic";

export default async function AdminAnalyticsPage() {
  const user = await getSessionUser();
  if (!user) redirect("/login");
  if (user.role !== "admin") redirect("/dashboard");
  return (
    <Shell user={user} title="Admin · Analytics">
      <div className="eyebrow">Control Room</div>
      <h2 className="page-h">Analytics</h2>
      <p className="page-sub">Two views: <b>traffic</b> (page views, visitors, top pages) is collected by Vercel and viewed on Vercel&apos;s dashboard; <b>app &amp; LLM usage</b> (tokens, quota, per-user) is in-app.</p>

      <div className="split col-2" style={{ marginTop: 8 }}>
        <div className="card">
          <div className="card-h"><span className="t">🌐 Traffic — Vercel Web Analytics</span></div>
          <div className="card-b">
            <p className="note" style={{ lineHeight: 1.6 }}>Vercel collects page views and visitors automatically (tracking is already added to the app). The charts live on Vercel — they can&apos;t be embedded here.</p>
            <ol className="note" style={{ lineHeight: 1.8, paddingLeft: 18 }}>
              <li>Open <b>vercel.com/dashboard</b> → your project.</li>
              <li>Click the <b>Analytics</b> tab (enable Web Analytics once, free on Hobby).</li>
              <li>See visitors, page views, top pages, referrers, devices.</li>
            </ol>
            <a className="btn sm" href="https://vercel.com/dashboard" target="_blank" rel="noreferrer">Open Vercel dashboard ↗</a>
          </div>
        </div>
        <div className="card">
          <div className="card-h"><span className="t">📊 App &amp; LLM usage — in-app</span></div>
          <div className="card-b">
            <p className="note" style={{ lineHeight: 1.6 }}>Your own data: LLM tokens consumed, per-user quota, model usage, and monitoring — all queried from your database.</p>
            <div className="row" style={{ gap: 8, flexWrap: "wrap" }}>
              <Link className="btn sm" href="/admin/usage">Usage &amp; Monitoring →</Link>
              <Link className="btn ghost sm" href="/admin/agents">Agent analytics →</Link>
              <Link className="btn ghost sm" href="/admin/storage">Storage →</Link>
            </div>
          </div>
        </div>
      </div>
    </Shell>
  );
}
