import { redirect } from "next/navigation";
import { getSessionUser } from "@/lib/auth";
import Shell from "@/components/Shell";
import MlLab from "@/components/MlLab";

export default async function MlPage() {
  const user = await getSessionUser();
  if (!user) redirect("/login");
  return (
    <Shell user={user} title="ML Lab">
      <MlLab />
    </Shell>
  );
}
