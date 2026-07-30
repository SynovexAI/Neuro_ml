import { redirect } from "next/navigation";
import { getSessionUser } from "@/lib/auth";
import Shell from "@/components/Shell";
import DlLab from "@/components/DlLab";

export default async function DlPage() {
  const user = await getSessionUser();
  if (!user) redirect("/login");
  return (
    <Shell user={user} title="DL Lab">
      <DlLab />
    </Shell>
  );
}
