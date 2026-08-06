import { redirect } from "next/navigation";
import { getSessionUser } from "@/lib/auth";
import Shell from "@/components/Shell";
import ChannelsManager from "@/components/ChannelsManager";

export default async function ChannelsPage() {
  const user = await getSessionUser();
  if (!user) redirect("/login");
  return (
    <Shell user={user} title="Channels & deploy">
      <div className="eyebrow">Workroom</div>
      <h2 className="page-h">Channels &amp; deploy</h2>
      <p className="page-sub">Put a published agent to work outside the platform — as a Telegram bot, an embeddable web chat, or a REST API.</p>
      <ChannelsManager />
    </Shell>
  );
}
