import { NextResponse } from "next/server";
import { eq } from "drizzle-orm";
import { db } from "@/lib/db";
import { channels, projects } from "@/lib/db/schema";
import { decrypt } from "@/lib/crypto";
import { sendMessage, webhookSecret } from "@/lib/telegram";
import { runAgent } from "@/lib/agentRunner";
import { captureError } from "@/lib/monitor";
import { cfgToRunInput } from "@/lib/channels";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Telegram webhook. Telegram POSTs updates here and echoes our secret token in a
// header — we validate that before doing any work.
export async function POST(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const [ch] = await db.select().from(channels).where(eq(channels.id, id));
  if (!ch || ch.type !== "telegram" || !ch.enabled || !ch.secretEnc) return NextResponse.json({ ok: true });
  if (req.headers.get("x-telegram-bot-api-secret-token") !== webhookSecret(id)) return NextResponse.json({ ok: true });

  const token = decrypt(ch.secretEnc);
  const update = await req.json().catch(() => null) as { message?: { chat?: { id: number }; text?: string } } | null;
  const chatId = update?.message?.chat?.id;
  const text = (update?.message?.text || "").trim();
  if (!chatId || !text) return NextResponse.json({ ok: true });

  if (text === "/start") { await sendMessage(token, chatId, "👋 Hi! I'm your AI Workbench agent. Ask me anything."); return NextResponse.json({ ok: true }); }

  try {
    const [proj] = await db.select().from(projects).where(eq(projects.id, ch.projectId));
    if (!proj || !proj.published) { await sendMessage(token, chatId, "⚠ This agent is no longer available."); return NextResponse.json({ ok: true }); }
    const r = await runAgent({ ...cfgToRunInput(proj.config), userId: ch.userId, task: text, agentName: proj.name });
    await sendMessage(token, chatId, r.ok ? (r.answer || "(no answer)") : `⚠ ${r.error}`);
  } catch (e) {
    captureError(e, { where: "telegram.webhook", channel: id });
    try { await sendMessage(token, chatId, "⚠ Something went wrong handling your message."); } catch { /* ignore */ }
  }
  return NextResponse.json({ ok: true });
}
