import { eq } from "drizzle-orm";
import { db } from "@/lib/db";
import { channels, projects } from "@/lib/db/schema";
import EmbedChat from "@/components/EmbedChat";

export const dynamic = "force-dynamic";

// Public, embeddable chat widget (no auth). Only serves `widget` channels.
export default async function EmbedPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const [ch] = await db.select().from(channels).where(eq(channels.id, id));
  if (!ch || ch.type !== "widget" || !ch.enabled) {
    return <div style={{ font: "14px system-ui", padding: 24, color: "#666" }}>This chat is unavailable.</div>;
  }
  const [proj] = await db.select().from(projects).where(eq(projects.id, ch.projectId));
  const name = proj?.published ? proj.name : "Assistant";
  return <EmbedChat channelId={id} agentName={name} />;
}
