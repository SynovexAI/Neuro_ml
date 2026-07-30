import { redirect } from "next/navigation";
import { getSessionUser } from "@/lib/auth";
import Shell from "@/components/Shell";
import RagLab from "@/components/RagLab";

export default async function RagPage() {
  const user = await getSessionUser();
  if (!user) redirect("/login");
  return (
    <Shell user={user} title="RAG Lab">
      <RagLab />
    </Shell>
  );
}
