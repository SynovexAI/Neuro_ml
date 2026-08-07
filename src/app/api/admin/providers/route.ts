import { NextResponse } from "next/server";
import { db } from "@/lib/db";
import { providers } from "@/lib/db/schema";
import { getSessionUser, uid } from "@/lib/auth";
import { encrypt, decrypt, maskKey } from "@/lib/crypto";
import { audit } from "@/lib/monitor";

async function admin() {
  const u = await getSessionUser();
  return u && u.role === "admin" ? u : null;
}

export async function GET() {
  if (!(await admin())) return NextResponse.json({ error: "forbidden" }, { status: 403 });
  const rows = await db.select().from(providers);
  const safe = rows.map((p) => ({
    id: p.id, provider: p.provider, label: p.label, baseUrl: p.baseUrl,
    defaultModel: p.defaultModel, enabled: p.enabled,
    maskedKey: p.apiKeyEnc ? maskKey(decrypt(p.apiKeyEnc)) : "",
  }));
  return NextResponse.json({ providers: safe });
}

export async function POST(req: Request) {
  const me = await admin();
  if (!me) return NextResponse.json({ error: "forbidden" }, { status: 403 });
  const b = await req.json().catch(() => ({}));
  if (!b.provider || !b.baseUrl) return NextResponse.json({ error: "provider and baseUrl required" }, { status: 400 });
  const id = uid();
  await db.insert(providers).values({
    id,
    provider: String(b.provider),
    label: b.label ? String(b.label) : null,
    baseUrl: String(b.baseUrl),
    apiKeyEnc: b.apiKey ? encrypt(String(b.apiKey)) : null,
    defaultModel: b.defaultModel ? String(b.defaultModel) : null,
    enabled: b.enabled !== false,
  });
  await audit("provider_added", me.id, { provider: String(b.provider), scope: "global" }).catch(() => {});
  return NextResponse.json({ ok: true, id });
}
