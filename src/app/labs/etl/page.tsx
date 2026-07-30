import { redirect } from "next/navigation";
import { getSessionUser } from "@/lib/auth";
import Shell from "@/components/Shell";
import EtlLab from "@/components/EtlLab";

export default async function EtlPage() {
  const user = await getSessionUser();
  if (!user) redirect("/login");
  return (
    <Shell user={user} title="ETL Lab">
      <EtlLab />
    </Shell>
  );
}
