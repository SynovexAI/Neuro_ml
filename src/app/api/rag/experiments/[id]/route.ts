import { NextResponse } from "next/server";
import { and, eq } from "drizzle-orm";
import { db } from "@/lib/db";
import { ragExperiments } from "@/lib/db/schema";
import { getSessionUser } from "@/lib/auth";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// DELETE → remove one of the current user's experiments.
export async function DELETE(_req: Request, { params }: { params: Promise<{ id: string }> }) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const { id } = await params;
  await db.delete(ragExperiments).where(and(eq(ragExperiments.id, id), eq(ragExperiments.userId, user.id)));
  return NextResponse.json({ ok: true });
}
