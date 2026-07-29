import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { getActiveProvider, fetchModels } from "@/lib/providers";

export const dynamic = "force-dynamic";

// Models available to any signed-in user, from the active provider (key stays server-side).
export async function GET() {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const prov = await getActiveProvider();
  if (!prov) return NextResponse.json({ provider: null, models: [], default: "" });
  try {
    const models = await fetchModels(prov.baseUrl, prov.apiKey);
    return NextResponse.json({ provider: prov.provider, models, default: prov.model });
  } catch {
    return NextResponse.json({ provider: prov.provider, models: prov.model ? [prov.model] : [], default: prov.model });
  }
}
