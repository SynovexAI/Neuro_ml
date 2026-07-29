import { NextResponse } from "next/server";
import { eq } from "drizzle-orm";
import { db } from "@/lib/db";
import { users } from "@/lib/db/schema";
import { getSessionUser } from "@/lib/auth";

export async function PATCH(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const me = await getSessionUser();
  if (!me || me.role !== "admin") return NextResponse.json({ error: "forbidden" }, { status: 403 });
  const { id } = await params;
  const b = await req.json().catch(() => ({}));
  const patch: Record<string, unknown> = {};
  if (b.status && ["pending", "active", "suspended"].includes(b.status)) patch.status = b.status;
  if (b.role && ["admin", "student"].includes(b.role)) patch.role = b.role;
  // Don't let an admin lock themselves out.
  if (id === me.id && (patch.role === "student" || patch.status === "suspended"))
    return NextResponse.json({ error: "You can't demote or suspend your own account." }, { status: 400 });
  if (Object.keys(patch).length) await db.update(users).set(patch).where(eq(users.id, id));
  return NextResponse.json({ ok: true });
}
