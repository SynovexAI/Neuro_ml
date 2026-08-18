import { redirect } from "next/navigation";
import { getSessionUser } from "@/lib/auth";
import Shell from "@/components/Shell";
import WorkroomList from "@/components/WorkroomList";

export default async function WorkroomPage() {
  const user = await getSessionUser();
  if (!user) redirect("/login");
  return (
    <Shell user={user} title="Workroom">
      <div className="eyebrow">Workroom</div>
      <h2 className="page-h">Your published agents</h2>
      <p className="page-sub">Agents you publish from the Agent Lab show up here — open one to chat with it like an assistant, or deploy it to a channel such as a Telegram bot.</p>
      <WorkroomList />
    </Shell>
  );
}
