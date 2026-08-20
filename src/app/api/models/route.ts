import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { getEnabledProviders, getProviderById, fetchModels } from "@/lib/providers";

export const dynamic = "force-dynamic";
export const maxDuration = 60;

// GET  /api/models               → all providers available to this user (own first, then global)
// GET  /api/models?providerId=x  → also returns models for that specific provider
export async function GET(req: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });

  // Students see/use only their OWN keys; admins also see the shared/global providers.
  const includeGlobal = user.role === "admin";
  let allProviders: Awaited<ReturnType<typeof getEnabledProviders>>;
  try {
    allProviders = await getEnabledProviders(user.id, includeGlobal);
  } catch {
    return NextResponse.json({ providers: [], models: [], provider: null, default: "" });
  }
  if (allProviders.length === 0) {
    return NextResponse.json({ providers: [], models: [], provider: null, default: "" });
  }

  const { searchParams } = new URL(req.url);
  const requestedId = searchParams.get("providerId");

  // Resolve the target provider via getProviderById (checks the user's own keys AND global);
  // fall back to the first available one if the requested id is stale.
  let p = requestedId ? await getProviderById(requestedId, user.id, includeGlobal) : null;
  if (!p) p = await getProviderById(allProviders[0].id, user.id, includeGlobal);
  if (!p) return NextResponse.json({ providers: allProviders, models: [], provider: null, default: "" });

  let models: string[];
  try {
    models = await fetchModels(p.baseUrl, p.apiKey);
  } catch {
    models = p.model ? [p.model] : [];
  }

  return NextResponse.json({
    providers: allProviders,
    provider: p.provider,
    providerId: p.id,
    models,
    default: p.model || models[0] || "",
  });
}
