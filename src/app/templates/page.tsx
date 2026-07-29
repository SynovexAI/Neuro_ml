import { redirect } from "next/navigation";
import { getSessionUser } from "@/lib/auth";
import Shell from "@/components/Shell";

export default async function TemplatesPage() {
  const user = await getSessionUser();
  if (!user) redirect("/login");
  return (
    <Shell user={user} title="Templates">
      <div className="hero"><h2>Templates</h2><p>One-click starter projects for every Lab. Arrives after the labs are wired.</p></div>
    </Shell>
  );
}
