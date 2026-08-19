import { NextResponse } from "next/server";
import { eq } from "drizzle-orm";
import { getSessionUser } from "@/lib/auth";
import { fetchModels } from "@/lib/providers";
import { db } from "@/lib/db";
import { providers } from "@/lib/db/schema";
import { decrypt } from "@/lib/crypto";

export async function POST(req: Request) {
  const u = await getSessionUser();
  if (!u || u.role !== "admin") return NextResponse.json({ error: "forbidden" }, { status: 403 });
  const body = await req.json().catch(() => ({}));
  let baseUrl: string = String(body.baseUrl || "");
  let apiKey: string = String(body.apiKey || "");
  // If an existing provider id is given (and no fresh key typed), use its stored base URL + key.
  if (body.id) {
    const rows = await db.select().from(providers).where(eq(providers.id, String(body.id))).limit(1);
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
