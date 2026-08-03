import { createHmac } from "crypto";

const API = "https://api.telegram.org";

// A per-channel webhook secret Telegram echoes back in the
// X-Telegram-Bot-Api-Secret-Token header. Derived (not stored) so we can
// recompute it to validate incoming updates. Chars are hex → Telegram-safe.
export function webhookSecret(channelId: string): string {
  const key = process.env.ENCRYPTION_KEY || "workbench";
  return createHmac("sha256", key).update(`tg:${channelId}`).digest("hex");
}

async function tg(token: string, method: string, body: unknown): Promise<{ ok: boolean; description?: string }> {
  const res = await fetch(`${API}/bot${token}/${method}`, {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body),
  });
  return res.json().catch(() => ({ ok: false, description: `HTTP ${res.status}` }));
}

export function setWebhook(token: string, url: string, secret: string) {
  return tg(token, "setWebhook", { url, secret_token: secret, allowed_updates: ["message"], drop_pending_updates: true });
}

export function deleteWebhook(token: string) {
  return tg(token, "deleteWebhook", { drop_pending_updates: true });
}

export function sendMessage(token: string, chatId: number | string, text: string) {
  // Telegram caps messages at 4096 chars.
  return tg(token, "sendMessage", { chat_id: chatId, text: text.slice(0, 4096) });
}

export async function getMe(token: string): Promise<{ ok: boolean; username?: string; description?: string }> {
  const res = await fetch(`${API}/bot${token}/getMe`).then((r) => r.json()).catch(() => null);
  if (!res?.ok) return { ok: false, description: res?.description || "invalid token" };
  return { ok: true, username: res.result?.username };
}
