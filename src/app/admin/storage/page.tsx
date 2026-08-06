import { redirect } from "next/navigation";
import { getSessionUser } from "@/lib/auth";
import Shell from "@/components/Shell";
import StorageManager from "@/components/StorageManager";

export const dynamic = "force-dynamic";

export default async function AdminStoragePage() {
  const user = await getSessionUser();
  if (!user) redirect("/login");
  if (user.role !== "admin") redirect("/dashboard");
  return (
    <Shell user={user} title="Admin · Storage">
      <div className="eyebrow">Control Room</div>
      <h2 className="page-h">File storage</h2>
      <p className="page-sub">Files users upload (RAG docs, knowledge bases) are stored in object storage. When it fills up, delete files here to free space so users can upload again.</p>
      <StorageManager />
    </Shell>
  );
}
