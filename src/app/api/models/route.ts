import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { getEnabledProviders, fetchModels } from "@/lib/providers";

export const dynamic = "force-dynamic";

// GET  /api/models               → returns all enabled providers
// GET  /api/models?providerId=x  → also returns models for that specific provider
export async function GET(req: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });

  let allProviders: Awaited<ReturnType<typeof getEnabledProviders>>;
  try {
    allProviders = await getEnabledProviders();
  } catch {
    return NextResponse.json({ providers: [], models: [], provider: null, default: "" });
  }

  if (allProviders.length === 0) {
    return NextResponse.json({ providers: [], models: [], provider: null, default: "" });
  }

  // If the caller requests models for a specific provider, fetch them live.
  const { searchParams } = new URL(req.url);
  const requestedId = searchParams.get("providerId");

  // Find the target provider (requested one, or the first enabled).
  const { db } = await import("@/lib/db");
  const { providers: provTable } = await import("@/lib/db/schema");
  const { eq } = await import("drizzle-orm");
  const { decrypt } = await import("@/lib/crypto");

  // Requested id may be stale (provider deleted & re-added) — fall back to the first enabled one.
  let targetId = requestedId ?? allProviders[0].id;
  let rows = await db.select().from(provTable).where(eq(provTable.id, targetId)).limit(1);
  if (!rows[0] && requestedId) { targetId = allProviders[0].id; rows = await db.select().from(provTable).where(eq(provTable.id, targetId)).limit(1); }
  const p = rows[0];
  if (!p) return NextResponse.json({ providers: allProviders, models: [], provider: null, default: "" });

  const apiKey = p.apiKeyEnc ? decrypt(p.apiKeyEnc) : "";
  let models: string[];
  try {
    models = await fetchModels(p.baseUrl, apiKey);
  } catch {
    models = p.defaultModel ? [p.defaultModel] : [];
  }

  return NextResponse.json({
    providers: allProviders,
    provider: p.provider,
    providerId: p.id,
    models,
    default: p.defaultModel || models[0] || "",
  });
}
