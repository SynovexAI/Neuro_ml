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
        <p>Configure LLM providers &amp; API keys, and manage student accounts.</p>
      </div>
      <div className="cards">
        <Link href="/admin/providers" className="lab-card"><h3>Providers &amp; models</h3><p>Choose a provider, load its models, and save an encrypted API key used platform-wide.</p><div className="go">Open →</div></Link>
        <Link href="/admin/users" className="lab-card"><h3>Users</h3><p>Approve pending sign-ups, set roles, and suspend accounts.</p><div className="go">Open →</div></Link>
      </div>
    </Shell>
  );
}
