import { redirect } from "next/navigation";
import { and, eq } from "drizzle-orm";
import { getSessionUser } from "@/lib/auth";
import { db } from "@/lib/db";
import { projects } from "@/lib/db/schema";
import Shell from "@/components/Shell";
import WorkroomChat, { type AgentCfg } from "@/components/WorkroomChat";

export default async function WorkroomChatPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const user = await getSessionUser();
  if (!user) redirect("/login");
  const [row] = await db.select().from(projects).where(and(eq(projects.id, id), eq(projects.userId, user.id)));
  if (!row || !row.published) redirect("/workroom");
  const cfg = (row.config || {}) as AgentCfg;
  return (
    <Shell user={user} title={row.name}>
      <WorkroomChat agentName={row.name} cfg={cfg} />
    </Shell>
  );
}
