import { redirect } from "next/navigation";
import { getSessionUser } from "@/lib/auth";
import Shell from "@/components/Shell";
import PromptingLab from "@/components/PromptingLab";

export default async function PromptingPage() {
  const user = await getSessionUser();
  if (!user) redirect("/login");
  return (
    <Shell user={user} title="Prompting Lab">
      <PromptingLab />
    </Shell>
  );
}
