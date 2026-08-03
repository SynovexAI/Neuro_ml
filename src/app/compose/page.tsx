import { redirect } from "next/navigation";
import { getSessionUser } from "@/lib/auth";
import Shell from "@/components/Shell";
import Compose from "@/components/Compose";

export default async function ComposePage() {
  const user = await getSessionUser();
  if (!user) redirect("/login");
  return (
    <Shell user={user} title="Compose">
      <Compose />
    </Shell>
  );
}
