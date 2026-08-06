import { redirect } from "next/navigation";

// MCP is a per-user Studio feature — it moved to /studio/mcp. Keep this redirect
// so old links / bookmarks don't break.
export default function AdminMcpRedirect() {
  redirect("/studio/mcp");
}
