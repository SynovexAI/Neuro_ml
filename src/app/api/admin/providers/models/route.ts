import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { fetchModels } from "@/lib/providers";

export async function POST(req: Request) {
  const u = await getSessionUser();
  if (!u || u.role !== "admin") return NextResponse.json({ error: "forbidden" }, { status: 403 });
  const { baseUrl, apiKey } = await req.json().catch(() => ({}));
  if (!baseUrl) return NextResponse.json({ error: "baseUrl required" }, { status: 400 });
  try {
    const models = await fetchModels(String(baseUrl), String(apiKey || ""));
    return NextResponse.json({ models });
  } catch (e) {
    return NextResponse.json({ error: (e as Error).message || "failed to load models" }, { status: 502 });
  }
}
