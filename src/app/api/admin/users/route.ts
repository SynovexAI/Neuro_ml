import { NextResponse } from "next/server";
import { db } from "@/lib/db";
import { users } from "@/lib/db/schema";
import { getSessionUser } from "@/lib/auth";

export async function GET() {
  const u = await getSessionUser();
  if (!u || u.role !== "admin") return NextResponse.json({ error: "forbidden" }, { status: 403 });
  const rows = await db.select({
    id: users.id, email: users.email, name: users.name,
    role: users.role, status: users.status, createdAt: users.createdAt,
  }).from(users);
  return NextResponse.json({ users: rows });
}
