import { NextResponse } from "next/server";
import { and, desc, eq } from "drizzle-orm";
import { db } from "@/lib/db";
import { mcpServers } from "@/lib/db/schema";
import { getSessionUser } from "@/lib/auth";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Enabled MCP servers, minimal + secret-free, for the Agent Lab tool picker.
export async function GET() {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const rows = await db.select({ id: mcpServers.id, name: mcpServers.name, transport: mcpServers.transport, tools: mcpServers.tools })
    .from(mcpServers).where(and(eq(mcpServers.enabled, true), eq(mcpServers.userId, user.id))).orderBy(desc(mcpServers.createdAt));
  return NextResponse.json({ servers: rows });
}
