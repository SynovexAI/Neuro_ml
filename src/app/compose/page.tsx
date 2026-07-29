import { redirect } from "next/navigation";
import { getSessionUser } from "@/lib/auth";
import Shell from "@/components/Shell";

export default async function ComposePage() {
  const user = await getSessionUser();
  if (!user) redirect("/login");
  return (
    <Shell user={user} title="Compose">
      <div className="hero"><h2>Compose</h2><p>Chain your saved builds into one system — e.g. an agent that uses your RAG bot as a tool. Arrives after the labs are wired.</p></div>
    </Shell>
  );
}
