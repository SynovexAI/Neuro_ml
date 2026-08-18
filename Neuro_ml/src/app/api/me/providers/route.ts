import { NextResponse } from "next/server";
import { eq, desc } from "drizzle-orm";
import { db } from "@/lib/db";
import { userProviders } from "@/lib/db/schema";
import { getSessionUser, uid } from "@/lib/auth";
import { encrypt, decrypt, maskKey } from "@/lib/crypto";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// A signed-in user's OWN LLM providers (their key). Separate from admin/global providers.
export async function GET() {
  const u = await getSessionUser();
  if (!u) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const rows = await db.select().from(userProviders).where(eq(userProviders.userId, u.id)).orderBy(desc(userProviders.createdAt));
  const safe = rows.map((p) => ({
    id: p.id, provider: p.provider, label: p.label, baseUrl: p.baseUrl,
    defaultModel: p.defaultModel, enabled: p.enabled,
    maskedKey: p.apiKeyEnc ? maskKey(decrypt(p.apiKeyEnc)) : "",
  }));
  return NextResponse.json({ providers: safe });
}

export async function POST(req: Request) {
  const u = await getSessionUser();
  if (!u) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const b = await req.json().catch(() => ({}));
  if (!b.provider || !b.baseUrl) return NextResponse.json({ error: "provider and baseUrl required" }, { status: 400 });
  const id = uid();
  try {
    await db.insert(userProviders).values({
      id,
      userId: u.id,
      provider: String(b.provider),
      label: b.label ? String(b.label) : null,
      baseUrl: String(b.baseUrl),
      apiKeyEnc: b.apiKey ? encrypt(String(b.apiKey)) : null,
      defaultModel: b.defaultModel ? String(b.defaultModel) : null,
      enabled: b.enabled !== false,
    });
  } catch (e) {
    return NextResponse.json({ error: `Could not save (has the user_providers table been created?): ${(e as Error).message}` }, { status: 500 });
  }
  return NextResponse.json({ ok: true, id });
}
