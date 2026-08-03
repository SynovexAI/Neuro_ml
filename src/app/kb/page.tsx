import { redirect } from "next/navigation";
import { getSessionUser } from "@/lib/auth";
import Shell from "@/components/Shell";
import KnowledgeBases from "@/components/KnowledgeBases";

export default async function KbPage() {
  const user = await getSessionUser();
  if (!user) redirect("/login");
  return (
    <Shell user={user} title="Knowledge bases">
      <div className="eyebrow">Studio</div>
      <h2 className="page-h">Knowledge bases</h2>
      <p className="page-sub">Upload files or add URLs, sync to build a vector store, then attach a knowledge base to an agent to ground its answers (RAG).</p>
      <KnowledgeBases />
    </Shell>
  );
}
