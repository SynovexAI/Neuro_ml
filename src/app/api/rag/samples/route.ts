import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { ragSampleList, ragSampleById } from "@/lib/ragSamples";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// GET            → list built-in practice datasets (metadata only, no text).
// GET ?id=<id>   → the full text of one dataset, ready to load into the pipeline.
export async function GET(req: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const id = new URL(req.url).searchParams.get("id");
  if (id) {
    const s = ragSampleById(id);
    if (!s) return NextResponse.json({ error: "not found" }, { status: 404 });
    return NextResponse.json({ id: s.id, name: s.name, text: s.text, questions: s.questions });
  }
  return NextResponse.json({ samples: ragSampleList() });
}
