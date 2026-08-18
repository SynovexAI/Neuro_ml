import { redirect } from "next/navigation";
import { getSessionUser } from "@/lib/auth";
import Shell from "@/components/Shell";

const NAMES: Record<string, string> = {
  prompting: "Prompting Lab", rag: "RAG Lab", agent: "Agent Lab",
  ml: "ML Lab", dl: "DL Lab", etl: "ETL Lab",
};

export default async function LabPage({ params }: { params: Promise<{ lab: string }> }) {
  const user = await getSessionUser();
  if (!user) redirect("/login");
  const { lab } = await params;
  const name = NAMES[lab] ?? "Lab";
  return (
    <Shell user={user} title={name}>
      <div className="hero fade-in">
        <h2>{name}</h2>
        <p>This lab&apos;s full workflow is being wired to the live providers and in-browser runtimes. It arrives in the next build step.</p>
      </div>
    </Shell>
  );
}
