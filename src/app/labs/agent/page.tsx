import { redirect } from "next/navigation";
import { getSessionUser } from "@/lib/auth";
import Shell from "@/components/Shell";
import AgentLab from "@/components/AgentLab";

export default async function AgentPage() {
  const user = await getSessionUser();
  if (!user) redirect("/login");
  return (
    <Shell user={user} title="Agent Lab">
      <AgentLab />
    </Shell>
  );
}
