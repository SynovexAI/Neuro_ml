import { NextResponse } from "next/server";
import { and, eq } from "drizzle-orm";
import { getSessionUser } from "@/lib/auth";
import { fetchModels } from "@/lib/providers";
import { db } from "@/lib/db";
import { userProviders } from "@/lib/db/schema";
import { decrypt } from "@/lib/crypto";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 60;

// Load a provider's live model list for a signed-in user. Accepts a freshly typed
// {baseUrl, apiKey}, or an {id} of one of the user's own providers (uses its stored key).
export async function POST(req: Request) {
  const u = await getSessionUser();
  if (!u) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const body = await req.json().catch(() => ({}));
  let baseUrl: string = String(body.baseUrl || "");
  let apiKey: string = String(body.apiKey || "");
  if (body.id) {
    const rows = await db.select().from(userProviders).where(and(eq(userProviders.id, String(body.id)), eq(userProviders.userId, u.id))).limit(1);
    const p = rows[0];
    if (!p) return NextResponse.json({ error: "provider not found" }, { status: 404 });
    if (!baseUrl) baseUrl = p.baseUrl;
    if (!apiKey && p.apiKeyEnc) apiKey = decrypt(p.apiKeyEnc);
  }
  if (!baseUrl) return NextResponse.json({ error: "baseUrl required" }, { status: 400 });
  try {
    const models = await fetchModels(baseUrl, apiKey);
    return NextResponse.json({ models });
  } catch (e) {
    return NextResponse.json({ error: (e as Error).message || "failed to load models" }, { status: 502 });
  }
}
